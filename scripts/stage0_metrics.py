"""Stage 0, run 2: metrics that random dictionaries cannot fake.

  1. reconstruction FVE of k-sparse codes with a fitted linear decoder (held-out images)
  2. sparse probing (k = 1, 5, 10, 50 latents chosen on the source dataset) with cross-dataset transfer
  3. null-adjusted monosemanticity on CheXpert finding sets (Wesp et al. 2026 style)

--protocol thesis (default): run 2 unchanged. Outputs in results/stage0/: recon.csv, probe_transfer.csv,
selectivity_<dictionary>.csv, metrics_summary.json.

--protocol v1 (protocol sections 3, 4, 10): grouped split file; ridge decoder, PCA and probes fitted on
train images; FVE with each dictionary's own decoder next to the ridge refit (P1.1); probes with
regularisation tuned on val and the same pooling for codes and the dense reference (P1.2); selectivity
with uncertain CheXpert labels excluded per label and one "normal" column; cluster-bootstrap intervals and
complete separation of trained seeds against the nulls. Outputs in results/stage0_v1/metrics/:
results.csv (tidy), recon.csv, probe_transfer.csv, selectivity_<dictionary>.csv, summary.json,
config.json, run.log.

Run: ~/.venvs/nla-mech-interp/bin/python scripts/stage0_metrics.py [--protocol thesis|v1] [--out DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make the package importable without relying on the editable-install .pth file, which macOS
# repeatedly marks hidden in this iCloud-synced folder (hidden .pth files are skipped by Python).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR, ArchivedActivations
from medinterp.data.manifests import CHEXPERT_LABELS, DEFAULT_LABEL_DIR, build_manifests
from medinterp.dictionaries.baselines import random_sae, signed_basis, signed_pca
from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.probing import fit_eval, sparse_probe_transfer
from medinterp.eval.recon import FVEAccumulator, RidgeAccumulator, predict, split_images
from medinterp.eval.selectivity import monosemanticity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage0"
SEED = 0
TEST_FRAC = 0.2
RECON_TRAIN_IMAGES, RECON_TEST_IMAGES = 600, 200   # per dataset
PROBE_KS = (1, 5, 10, 50)
SELECTIVITY_QUANTILE = 0.95


LOG_FILES: list = []   # v1 also writes the log into its run directory


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for f in LOG_FILES:
        f.write(line + "\n")
        f.flush()


def setup(acts: ArchivedActivations):
    """Labels, splits and reconstruction image samples, shared by every dictionary."""
    man = build_manifests(acts)
    rng = np.random.default_rng(SEED)
    info = {}
    for d, m in man.items():
        patients = m.path.str.split("/").str[2].to_numpy() if d == "chexpert" else None
        test = split_images(len(m), patients, TEST_FRAC, SEED)
        g = m.group.to_numpy()
        labelled = (g == 0) | (g == 5)
        tr_idx, te_idx = np.flatnonzero(~test), np.flatnonzero(test)
        info[d] = {
            "test": test, "labelled": labelled, "positive": g == 0,
            "recon_train": set(rng.choice(tr_idx, min(RECON_TRAIN_IMAGES, tr_idx.size), replace=False)),
            "recon_test": set(rng.choice(te_idx, min(RECON_TEST_IMAGES, te_idx.size), replace=False)),
        }
    findings = man["chexpert"][list(CHEXPERT_LABELS)].to_numpy() == 1
    findings = np.column_stack([findings, ~findings.any(1)])        # pseudo-finding "none"
    return info, findings


def score_dictionary(acts, enc, info, d_model):
    """One encode pass for image codes + ridge statistics, one short pass for held-out FVE."""
    n_lat = enc.d_sae
    image_codes, pooled = {}, {}
    ridge = RidgeAccumulator(n_lat, d_model)
    for d, block in acts.blocks.items():
        image_codes[d] = np.zeros((block.n_images, n_lat), dtype=np.float16)
        pooled[d] = np.zeros((block.n_images, d_model), dtype=np.float32)
        for img0, x in acts.iter_token_chunks(d, 256):
            n_img = x.shape[0] // acts.n_patches
            z = enc.encode(x)
            image_codes[d][img0:img0 + n_img] = z.view(n_img, acts.n_patches, -1).amax(1).numpy()
            pooled[d][img0:img0 + n_img] = x.view(n_img, acts.n_patches, -1).mean(1).numpy()
            sel = [i for i in range(n_img) if img0 + i in info[d]["recon_train"]]
            if sel:
                rows = torch.cat([torch.arange(i * acts.n_patches, (i + 1) * acts.n_patches) for i in sel])
                ridge.add(z[rows], x[rows])
    W = ridge.solve()
    fve, fve_native = FVEAccumulator(d_model), FVEAccumulator(d_model)
    for d in acts.blocks:
        for img0, x in acts.iter_token_chunks(d, 256):
            n_img = x.shape[0] // acts.n_patches
            sel = [i for i in range(n_img) if img0 + i in info[d]["recon_test"]]
            if not sel:
                continue
            rows = torch.cat([torch.arange(i * acts.n_patches, (i + 1) * acts.n_patches) for i in sel])
            z = enc.encode(x[rows])
            fve.add(x[rows], predict(W, z))
            if isinstance(enc, TopKSAE):
                fve_native.add(x[rows], enc.decode(z))
    native = fve_native.fve() if fve_native.n else None
    return image_codes, pooled, fve.fve(), native


def main_thesis(OUT: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acts = ArchivedActivations()
    info, findings = setup(acts)
    trained = TopKSAE.from_checkpoint(DEFAULT_RUN_DIR / "checkpoints/step_0200000")
    k, d_sae, d_model = trained.k, trained.d_sae, acts.d_model

    def pca():
        rng = np.random.default_rng(SEED)
        parts = [x[torch.from_numpy(rng.random(x.shape[0]) < 50_000 / b.n_tokens)]
                 for n, b in acts.blocks.items() for _, x in acts.iter_token_chunks(n, 1024)]
        return signed_pca(torch.cat(parts), d_model, k)

    dictionaries = {
        "trained_step200k": lambda: trained,
        "trained_step020k": lambda: TopKSAE.from_checkpoint(DEFAULT_RUN_DIR / "checkpoints/step_0020000"),
        **{f"random_seed{s}": (lambda s=s: random_sae(d_model, d_sae, k, seed=s)) for s in (0, 1, 2)},
        "signed_pca": pca,
        "signed_basis": lambda: signed_basis(d_model, k),
    }

    recon_rows, probe_rows, sel_tables = [], [], {}
    dense_done = False
    for name, build in dictionaries.items():
        log(f"{name}: encode + reconstruction")
        codes, pooled, fve, native = score_dictionary(acts, build(), info, d_model)
        recon_rows.append({"dictionary": name, "fve_linear_decoder": fve, "fve_native_decoder": native})
        log(f"{name}: FVE linear {fve:.4f}  native {native}")

        lab = {d: codes[d][info[d]["labelled"]] for d in codes}
        y = {d: info[d]["positive"][info[d]["labelled"]] for d in codes}
        te = {d: info[d]["test"][info[d]["labelled"]] for d in codes}
        for r in sparse_probe_transfer(lab, y, te, PROBE_KS):
            probe_rows.append({"dictionary": name, **r})

        if not dense_done:  # dense reference: probe on all 1024 pooled dimensions
            P = {d: pooled[d][info[d]["labelled"]] for d in pooled}
            for src in P:
                evals = {src: (P[src][te[src]], y[src][te[src]])}
                evals.update({t: (P[t], y[t]) for t in P if t != src})
                for tgt, auc in fit_eval(P[src][~te[src]], y[src][~te[src]], evals).items():
                    probe_rows.append({"dictionary": "dense_pooled", "source": src, "k": d_model,
                                       "target": tgt, "auroc": auc})
            dense_done = True

        log(f"{name}: selectivity")
        s = monosemanticity(codes["chexpert"], findings, top_n=10, seed=SEED)
        sel_tables[name] = pd.DataFrame(s, columns=["coherence", "specificity", "mono"]).assign(latent=np.arange(len(s)))
        sel_tables[name].to_csv(OUT / f"selectivity_{name}.csv", index=False)
        del codes, pooled

    recon = pd.DataFrame(recon_rows)
    recon.to_csv(OUT / "recon.csv", index=False)
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(OUT / "probe_transfer.csv", index=False)

    # Selectivity threshold from random SAEs; count latents above it per dictionary.
    rand_mono = pd.concat([sel_tables[f"random_seed{s}"].mono for s in (0, 1, 2)]).dropna()
    thr = float(np.quantile(rand_mono, SELECTIVITY_QUANTILE))
    summary = {"settings": {"seed": SEED, "test_frac": TEST_FRAC, "probe_ks": PROBE_KS,
                            "recon_images": [RECON_TRAIN_IMAGES, RECON_TEST_IMAGES],
                            "mono_threshold_q95_random": thr},
               "recon": recon.to_dict("records"), "selectivity": {}}
    for name, t in sel_tables.items():
        m = t.mono.dropna().sort_values(ascending=False)
        summary["selectivity"][name] = {"n_scored": int(m.size), "above_random_q95": int((m > thr).sum()),
                                        "top10_mean": float(m.head(10).mean()), "median": float(m.median())}
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    log("done")


# ---------------------------------------------------------------- protocol v1

V1_PROBE_MS = (1, 5, 10, 50)
V1_PRIMARY_MS = (1, 5)            # P1.2
V1_N_BOOT, V1_CONFIDENCE = 2000, 0.95
V1_RANDOM_SEEDS = [0, 1, 2, 3, 4]
V1_FINDINGS = [c for c in CHEXPERT_LABELS]       # "No Finding" is the single normal column


def v1_score_dictionary(ctx, enc, recon_train: dict, eval_masks: dict, pooling: str, d_model: int):
    """One pass: pooled image codes, ridge statistics on the train sample, and per-image errors of the
    own decoder on the evaluation images. A second pass scores the ridge refit on the same images."""
    from medinterp.eval.recon import PerImageRecon
    n_lat, P = enc.d_sae, ctx.acts.n_patches
    codes, ridge, own = {}, RidgeAccumulator(n_lat, d_model), PerImageRecon()
    has_decoder = hasattr(enc, "decode")
    for d, block in ctx.acts.blocks.items():
        codes[d] = np.zeros((block.n_images, n_lat), dtype=np.float16)
        for img0, x in ctx.acts.iter_token_chunks(d, 256):
            n_img = x.shape[0] // P
            z = enc.encode(x)
            zi = z.view(n_img, P, -1)
            codes[d][img0:img0 + n_img] = (zi.amax(1) if pooling == "max" else zi.mean(1)).numpy()
            local = np.arange(img0, img0 + n_img)
            sel = np.flatnonzero(recon_train[d][local])
            if sel.size:
                rows = torch.from_numpy((sel[:, None] * P + np.arange(P)).ravel())
                ridge.add(z[rows], x[rows])
            ev = np.flatnonzero(eval_masks[d][local])
            if ev.size and has_decoder:
                rows = torch.from_numpy((ev[:, None] * P + np.arange(P)).ravel())
                own.add(x[rows], enc.decode(z[rows]), P)
    W = ridge.solve()
    refit = PerImageRecon()
    for d in ctx.acts.blocks:
        for img0, x in ctx.acts.iter_token_chunks(d, 256):
            n_img = x.shape[0] // P
            ev = np.flatnonzero(eval_masks[d][np.arange(img0, img0 + n_img)])
            if ev.size:
                rows = torch.from_numpy((ev[:, None] * P + np.arange(P)).ravel())
                refit.add(x[rows], predict(W, enc.encode(x[rows])), P)
    return codes, (own if has_decoder else None), refit


def v1_pooled_activations(ctx, pooling: str) -> dict[str, np.ndarray]:
    out = {}
    for d, block in ctx.acts.blocks.items():
        out[d] = np.zeros((block.n_images, ctx.acts.d_model), dtype=np.float32)
        for img0, x in ctx.acts.iter_token_chunks(d, 256):
            n_img = x.shape[0] // ctx.acts.n_patches
            xi = x.view(n_img, ctx.acts.n_patches, -1)
            out[d][img0:img0 + n_img] = (xi.amax(1) if pooling == "max" else xi.mean(1)).numpy()
    return out


def main_v1(args) -> None:
    from medinterp.eval.probing import eval_images, mean_cross_auroc_statistic, sparse_probe_transfer_tuned
    from medinterp.eval.stats import cluster_bootstrap, complete_separation
    from medinterp.runinfo import write_config
    from medinterp.stage0_v1 import TIDY_COLUMNS, load_context, null_specs, parse_sae_args, tidy_row, trained_spec

    out = args.out
    if (out / "summary.json").exists() and not args.force:
        log(f"{out / 'summary.json'} exists; nothing to do (use --force to recompute)")
        return
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILES.append(open(out / "run.log", "a"))
    ctx = load_context(args.run_dir, args.label_dir, args.split_file)
    names, d_model, ev_split = list(ctx.acts.blocks), ctx.acts.d_model, args.eval_split
    saes = parse_sae_args(args.sae) or {f"seed{s}": ROOT / f"results/stage0/sae_seeds/seed{s}" for s in (0, 1, 2)}
    specs = [trained_spec(n, p, args.sae_fit_split) for n, p in saes.items()]
    k, d_sae = specs[0].meta["k"], specs[0].meta["d_sae"]
    specs += null_specs(ctx, d_sae, k, V1_RANDOM_SEEDS, 50_000, seed=SEED)

    # Ridge decoder fitted on a seeded sample of train images; FVE on every image of the eval split.
    rng = np.random.default_rng(SEED)
    recon_train = {}
    for d in names:
        idx = np.flatnonzero(ctx.masks[d]["train"])
        pick = rng.choice(idx, min(RECON_TRAIN_IMAGES, idx.size), replace=False)
        recon_train[d] = np.isin(np.arange(len(ctx.masks[d]["train"])), pick)
    eval_all = {d: ctx.masks[d][ev_split] for d in names}
    recon_groups = np.concatenate([ctx.groups[d][eval_all[d]] for d in names])
    labels = {d: ctx.opacity[d] for d in names}
    probe_masks = eval_images(labels, ctx.masks, ev_split)
    probe_groups = np.concatenate([ctx.groups[d][probe_masks[d]] for d in names])

    chex = ctx.manifests["chexpert"]
    sel_imgs = ctx.masks["chexpert"][ev_split]
    raw = chex[V1_FINDINGS].to_numpy()[sel_imgs]
    findings, known = raw == 1, raw != -1          # NaN = not mentioned = a certain negative

    settings = {"protocol": "v1", "eval_split": ev_split, "pooling": args.pooling, "probe_ms": V1_PROBE_MS,
                "n_bootstrap": V1_N_BOOT, "confidence": V1_CONFIDENCE, "recon_train_images": RECON_TRAIN_IMAGES,
                "seed": SEED, "centring": "archive (per-dataset, all images; as the SAEs were trained)",
                "dictionaries": {s.name: s.meta for s in specs}}
    write_config(out, settings, ROOT, args.split_file)
    boot = lambda stat, groups: cluster_bootstrap(stat, groups, V1_N_BOOT, V1_CONFIDENCE, seed=SEED)

    tidy, recon_rows, probe_rows, summary = [], [], [], {"settings": settings, "dictionaries": {}}
    # Bootstrap samples share the resamples across dictionaries (same groups, same seed): paired differences.
    samples, points = {}, {}

    def probe_block(spec_like, feats, m_list, rank_all=False):
        rows = sparse_probe_transfer_tuned(feats, labels, ctx.masks, m_list, ev_split, keep_scores=True,
                                           rank_all=rank_all)
        res = {}
        for m in (m_list if rank_all else V1_PRIMARY_MS):
            b = boot(mean_cross_auroc_statistic(rows, m, labels, probe_masks), probe_groups)
            res[m] = b
            tidy.append(tidy_row(spec_like, "probe_mean_cross_auroc", f"m={m}", ev_split, b.estimate, b.ci_low, b.ci_high))
        for r in rows:
            probe_rows.append({"dictionary": spec_like.name, **{c: r[c] for c in ("source", "m", "target", "auroc", "C")}})
        return res

    for spec in specs:
        log(f"{spec.name}: encode, reconstruction")
        codes, own, refit = v1_score_dictionary(ctx, spec.build(), recon_train, eval_all, args.pooling, d_model)
        row = {"fit_split": spec.meta.get("fit_split")}
        for label, acc in (("fve_own_decoder", own), ("fve_ridge_refit", refit)):
            if acc is None:
                continue
            b = boot(acc.fve, recon_groups)
            samples[(spec.name, label)], points[(spec.name, label)] = b.samples, b.estimate
            row[label] = [b.estimate, b.ci_low, b.ci_high]
            tidy.append(tidy_row(spec, label, "all", ev_split, b.estimate, b.ci_low, b.ci_high))
        recon_rows.append({"dictionary": spec.name, **{k_: v[0] for k_, v in row.items() if isinstance(v, list)}})
        log(f"{spec.name}: probes")
        for m, b in probe_block(spec, codes, V1_PROBE_MS).items():
            samples[(spec.name, f"probe_m{m}")], points[(spec.name, f"probe_m{m}")] = b.samples, b.estimate
            row[f"probe_mean_cross_auroc_m{m}"] = [b.estimate, b.ci_low, b.ci_high]
        log(f"{spec.name}: selectivity")
        sel = monosemanticity(codes["chexpert"][sel_imgs], findings, top_n=10, seed=SEED, known=known)
        st = pd.DataFrame(sel, columns=["coherence", "specificity", "mono"]).assign(latent=np.arange(len(sel)))
        st.to_csv(out / f"selectivity_{spec.name}.csv", index=False)
        mono = st.mono.dropna().sort_values(ascending=False)
        row["selectivity"] = {"n_scored": int(mono.size), "top10_mean": float(mono.head(10).mean()),
                              "median": float(mono.median())}
        tidy.append(tidy_row(spec, "selectivity_top10_mean", "chexpert", ev_split, row["selectivity"]["top10_mean"]))
        summary["dictionaries"][spec.name] = row
        log(f"{spec.name}: {row}")
        del codes

    log("dense reference")
    from medinterp.stage0_v1 import DictionarySpec
    dense = DictionarySpec("dense_pooled", "dense", lambda: None, {"d_sae": d_model, "k": None, "steps": 0})
    db = probe_block(dense, v1_pooled_activations(ctx, args.pooling), (d_model,), rank_all=True)
    summary["dense_reference"] = {f"probe_mean_cross_auroc_m{d_model}": [db[d_model].estimate, db[d_model].ci_low,
                                                                        db[d_model].ci_high]}

    # Trained seeds against every null: complete separation and the paired difference to the best null.
    trained = [s.name for s in specs if s.kind == "trained"]
    nulls = [s.name for s in specs if s.kind != "trained"]
    comparisons = {}
    for metric in ["fve_own_decoder", "fve_ridge_refit"] + [f"probe_m{m}" for m in V1_PRIMARY_MS]:
        have_null = [n for n in nulls if (n, metric) in samples]
        if not have_null:
            continue
        point = {n: points[(n, metric)] for n in trained + have_null}
        sep = complete_separation([point[n] for n in trained], [point[n] for n in have_null])
        best_null = max(have_null, key=lambda n: point[n])
        diffs = {}
        for t in trained:
            d = samples[(t, metric)] - samples[(best_null, metric)]
            lo, hi = np.quantile(d, [(1 - V1_CONFIDENCE) / 2, 1 - (1 - V1_CONFIDENCE) / 2])
            diffs[t] = [point[t] - point[best_null], float(lo), float(hi)]
        comparisons[metric] = {"separated": sep.separated, "gap": sep.gap, "p_exact": sep.p_value,
                               "n_trained": len(trained), "n_null": len(have_null), "best_null": best_null,
                               "difference_to_best_null": diffs}
        log(f"{metric}: {comparisons[metric]}")
    summary["trained_vs_null"] = comparisons

    pd.DataFrame(recon_rows).to_csv(out / "recon.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(out / "probe_transfer.csv", index=False)
    pd.DataFrame(tidy, columns=TIDY_COLUMNS).to_csv(out / "results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {out}")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=("thesis", "v1"), default="thesis")
    ap.add_argument("--out", type=Path, help="output directory (default: results/stage0 or results/stage0_v1/metrics)")
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--eval-split", choices=("val", "test"), default="val",
                    help="v1: split for reported numbers; test only in a final evaluation")
    ap.add_argument("--pooling", choices=("max", "mean"), default="max",
                    help="v1: pooling of codes and of the dense reference (the same on both sides)")
    ap.add_argument("--sae", action="append", default=[], metavar="NAME=PATH",
                    help="v1: trained SAE checkpoint directory (repeatable; default: Stage 0.6 seeds 0-2)")
    ap.add_argument("--sae-fit-split", default="all",
                    help="v1: images the given SAEs were fitted on, if their cfg.json does not say")
    ap.add_argument("--force", action="store_true", help="v1: recompute even if summary.json exists")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="v1: archived activations (thesis harvest)")
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="v1: label directory")
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT if args.protocol == "thesis" else ROOT / f"results/stage0_v1/metrics_{args.pooling}"
    return args


if __name__ == "__main__":
    _args = parse_args()
    if _args.protocol == "thesis":
        main_thesis(_args.out)
    else:
        main_v1(_args)
