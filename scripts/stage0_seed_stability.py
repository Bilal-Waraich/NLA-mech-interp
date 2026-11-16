"""Stage 0.6: seed stability of the SAE dictionary, and how it relates to the run-1/run-2 metrics.

Runs compared (decoder directions only, no activations needed):
  thesis_step200k, thesis_step020k   archived thesis SAE (one unseeded run), final and 20k checkpoints
  seed0, seed1, seed2                SAEs retrained here with explicit seeds, 20k steps
  random0, random1                   untrained SAEs: the chance level for every match score

Stable latent (anchored on thesis_step200k, which carries the run-1/run-2 metrics): its one-to-one
partner in each of seed0, seed1, seed2 has cosine >= threshold. Thresholds: the q99.9 of the random
null, and the conventional 0.7 / 0.9.

--protocol thesis (default), outputs (results/stage0/stability/):
  pairs.csv        per pair of runs: mean / median one-to-one cosine, fraction above each threshold
  latents.csv      per thesis latent: partner cosines per seed, stability flags, run-1/run-2 metrics
  summary.json     counts, null threshold, metric comparison stable vs unstable

--protocol v1 (protocol sections 4-5): anchored on the first given seed; the null is a dictionary of random
directions drawn from the covariance of train tokens (data-aware); S of P1.3 over all seed pairs with its
min-max. Outputs (results/stage0_v1/stability/): pairs.csv, latents.csv, summary.json, results.csv,
config.json, run.log.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/stage0_seed_stability.py [--protocol thesis|v1] [--out DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.dictionaries.baselines import random_sae
from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.stability import cosine_matrix, one_to_one

ROOT = Path(__file__).resolve().parents[1]
STAGE0 = ROOT / "results/stage0"
OUT = STAGE0 / "stability"
SEEDS = [0, 1, 2]
NULL_QUANTILE = 0.999
FIXED_THRESHOLDS = (0.7, 0.9)


LOG_FILES: list = []   # v1 also writes the log into its run directory


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for f in LOG_FILES:
        f.write(line + "\n")
        f.flush()


def load_decoders() -> dict[str, "np.ndarray"]:
    ckpt = DEFAULT_RUN_DIR / "checkpoints"
    dec = {"thesis_step200k": TopKSAE.from_checkpoint(ckpt / "step_0200000").W_dec,
           "thesis_step020k": TopKSAE.from_checkpoint(ckpt / "step_0020000").W_dec}
    for s in SEEDS:
        dec[f"seed{s}"] = TopKSAE.from_checkpoint(STAGE0 / f"sae_seeds/seed{s}").W_dec
    d_sae, d_model = dec["thesis_step200k"].shape
    for s in (0, 1):  # untrained, seeded; distinct from the random SAEs' role in run 1/2 only by name
        dec[f"random{s}"] = random_sae(d_model, d_sae, 48, seed=100 + s).W_dec
    return dec


def main_thesis(OUT: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dec = load_decoders()
    log(f"loaded {list(dec)}")

    # 1. Chance level: one-to-one cosine between a trained run and an untrained dictionary.
    null = np.concatenate([one_to_one(cosine_matrix(dec["thesis_step200k"], dec[r]))[0]
                           for r in ("random0", "random1")])
    null_thr = float(np.quantile(null, NULL_QUANTILE))
    thresholds = {"null": null_thr, **{str(t): t for t in FIXED_THRESHOLDS}}
    log(f"null one-to-one cosine: mean {null.mean():.3f}, q{NULL_QUANTILE} {null_thr:.3f}")

    # 2. Every pair of runs (including trained-vs-random, as a visible reference row).
    partners: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for a, b in itertools.combinations(dec, 2):
        if a.startswith("random") and b.startswith("random"):
            continue
        cos, _ = one_to_one(cosine_matrix(dec[a], dec[b]))
        partners[(a, b)] = cos
        rows.append({"run_a": a, "run_b": b, "mean": cos.mean(), "median": float(np.median(cos)),
                     **{f"frac_ge_{n}": float((cos >= t).mean()) for n, t in thresholds.items()}})
        log(f"{a} vs {b}: mean {cos.mean():.3f}, >=0.7 {(cos >= 0.7).mean():.3f}")
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "pairs.csv", index=False)

    # 3. Per thesis latent: partner cosine in each retrained seed, then stability flags.
    lat = pd.DataFrame({"latent": np.arange(dec["thesis_step200k"].shape[0])})
    for s in SEEDS:
        lat[f"cos_seed{s}"] = partners[("thesis_step200k", f"seed{s}")]
    lat["cos_thesis_step020k"] = partners[("thesis_step200k", "thesis_step020k")]
    lat["min_cos_seeds"] = lat[[f"cos_seed{s}" for s in SEEDS]].min(axis=1)
    for n, t in thresholds.items():
        lat[f"stable_{n}"] = lat.min_cos_seeds >= t

    # 4. Join with the run-1 (counting rule) and run-2 (selectivity) results for the same latents.
    run1 = pd.read_csv(STAGE0 / "latents_trained_step200k.csv")[
        ["latent", "active_all", "cross_dataset", "label_consistent", "dataset_auroc"]]
    run2 = pd.read_csv(STAGE0 / "selectivity_trained_step200k.csv")[["latent", "mono"]]
    mono_thr = json.loads((STAGE0 / "metrics_summary.json").read_text())["settings"]["mono_threshold_q95_random"]
    thesis65 = set(json.loads((DEFAULT_RUN_DIR / "universality/eval/confirmed_universal_latents.json")
                              .read_text())["latents"])
    lat = lat.merge(run1, on="latent").merge(run2, on="latent")
    lat["selective"] = lat.mono > mono_thr
    lat["thesis65"] = lat.latent.isin(thesis65)
    lat.to_csv(OUT / "latents.csv", index=False)

    # 5. Are stable latents different on the other metrics? Report rates, not just p-values.
    summary = {"settings": {"seeds": SEEDS, "null_quantile": NULL_QUANTILE, "thresholds": thresholds,
                            "mono_threshold": mono_thr},
               "null_mean": float(null.mean()), "n_latents": len(lat), "stable": {}}
    for n in thresholds:
        st = lat[f"stable_{n}"]
        entry = {"n_stable": int(st.sum())}
        for group in ("active_all", "cross_dataset", "label_consistent", "selective", "thesis65"):
            g = lat[group].astype(bool)
            entry[f"stable_rate_in_{group}"] = float(st[g].mean()) if g.any() else None
            entry[f"n_{group}"] = int(g.sum())
        m_st, m_un = lat.mono[st].dropna(), lat.mono[~st].dropna()
        if len(m_st) and len(m_un):
            entry["mono_median_stable"], entry["mono_median_unstable"] = float(m_st.median()), float(m_un.median())
            entry["mono_mannwhitney_p"] = float(mannwhitneyu(m_st, m_un).pvalue)
        summary["stable"][n] = entry
        log(f"stable@{n}: {entry}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {OUT}")


# ---------------------------------------------------------------- protocol v1

V1_THRESHOLDS = (0.7, 0.9)        # 0.7 is the P1.3 threshold; 0.9 is reported alongside
V1_NULL_SEEDS = (0, 1)


def main_v1(args) -> None:
    from medinterp.data.manifests import DEFAULT_LABEL_DIR
    from medinterp.dictionaries.baselines import covariance_directions
    from medinterp.eval.stability import TokenMoments
    from medinterp.runinfo import write_config
    from medinterp.stage0_v1 import TIDY_COLUMNS, load_context, parse_sae_args, tidy_row, trained_spec

    out = args.out
    if (out / "summary.json").exists() and not args.force:
        log(f"{out / 'summary.json'} exists; nothing to do (use --force to recompute)")
        return
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILES.append(open(out / "run.log", "a"))
    saes = parse_sae_args(args.sae) or {f"seed{s}": STAGE0 / f"sae_seeds/seed{s}" for s in SEEDS}
    specs = {n: trained_spec(n, p, args.sae_fit_split) for n, p in saes.items()}
    dec = {n: TopKSAE.from_checkpoint(p).W_dec for n, p in saes.items()}
    anchor = next(iter(dec))
    d_sae, d_model = dec[anchor].shape

    # Data-aware null: covariance of train tokens only.
    ctx = load_context(DEFAULT_RUN_DIR, DEFAULT_LABEL_DIR, args.split_file)
    moments = TokenMoments(d_model)
    for name in ctx.acts.blocks:
        train = ctx.masks[name]["train"]
        for img0, x in ctx.acts.iter_token_chunks(name, 1024):
            n_img = x.shape[0] // ctx.acts.n_patches
            keep = np.repeat(train[img0:img0 + n_img], ctx.acts.n_patches)
            moments.add(x[torch.from_numpy(keep)])
    cov = moments.covariance()
    settings = {"protocol": "v1", "anchor": anchor, "thresholds": V1_THRESHOLDS, "null": "covariance of train tokens",
                "null_seeds": V1_NULL_SEEDS, "null_quantile": NULL_QUANTILE, "n_train_tokens": moments.n,
                "dictionaries": {n: s.meta for n, s in specs.items()}}
    write_config(out, settings, ROOT, args.split_file)

    null = np.concatenate([one_to_one(cosine_matrix(dec[anchor], covariance_directions(cov, d_sae, seed=s)))[0]
                           for s in V1_NULL_SEEDS])
    null_thr = float(np.quantile(null, NULL_QUANTILE))
    thresholds = {"null": null_thr, **{str(t): t for t in V1_THRESHOLDS}}
    log(f"covariance null vs {anchor}: mean {null.mean():.3f}, q{NULL_QUANTILE} {null_thr:.3f}, "
        f">=0.7 {(null >= 0.7).mean():.4f}")

    rows, partners = [], {}
    for a, b in itertools.combinations(dec, 2):
        cos, _ = one_to_one(cosine_matrix(dec[a], dec[b]))
        partners[(a, b)] = cos
        rows.append({"run_a": a, "run_b": b, "mean": float(np.nanmean(cos)), "median": float(np.nanmedian(cos)),
                     **{f"frac_ge_{n}": float(np.nanmean(cos >= t)) for n, t in thresholds.items()}})
        log(f"{a} vs {b}: mean {np.nanmean(cos):.3f}, >=0.7 {np.nanmean(cos >= 0.7):.3f}")
    rows.append({"run_a": anchor, "run_b": "covariance_null", "mean": float(null.mean()),
                 "median": float(np.median(null)),
                 **{f"frac_ge_{n}": float((null >= t).mean()) for n, t in thresholds.items()}})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out / "pairs.csv", index=False)

    seed_pairs = pairs[pairs.run_b != "covariance_null"]
    S = seed_pairs["frac_ge_0.7"]
    lat = pd.DataFrame({"latent": np.arange(d_sae)})
    others = [n for n in dec if n != anchor]
    for n in others:
        lat[f"cos_{n}"] = partners[(anchor, n)]
    lat["min_cos"] = lat[[f"cos_{n}" for n in others]].min(axis=1)
    for n, t in thresholds.items():
        lat[f"stable_{n}"] = lat.min_cos >= t
    lat.to_csv(out / "latents.csv", index=False)

    summary = {"settings": settings, "null_mean": float(null.mean()), "null_threshold": null_thr,
               "null_frac_ge_0.7": float((null >= 0.7).mean()),
               "S": {"mean": float(S.mean()), "min": float(S.min()), "max": float(S.max()), "n_pairs": int(len(S))},
               "anchor_stable_in_all_others": {n: int(lat[f"stable_{n}"].sum()) for n in thresholds}}
    tidy = [tidy_row(specs[anchor], "seed_overlap_S", "all_pairs", "all", summary["S"]["mean"],
                     summary["S"]["min"], summary["S"]["max"]),
            tidy_row(specs[anchor], "covariance_null_q999", "anchor", "all", null_thr)]
    pd.DataFrame(tidy, columns=TIDY_COLUMNS).to_csv(out / "results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"S = {summary['S']}; wrote {out}")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=("thesis", "v1"), default="thesis")
    ap.add_argument("--out", type=Path, help="output directory (default: results/stage0/stability or "
                                             "results/stage0_v1/stability)")
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--sae", action="append", default=[], metavar="NAME=PATH",
                    help="v1: trained SAE checkpoint directory (repeatable; the first is the anchor)")
    ap.add_argument("--sae-fit-split", default="all")
    ap.add_argument("--force", action="store_true", help="v1: recompute even if summary.json exists")
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT if args.protocol == "thesis" else ROOT / "results/stage0_v1/stability"
    return args


if __name__ == "__main__":
    _args = parse_args()
    if _args.protocol == "thesis":
        main_thesis(_args.out)
    else:
        main_v1(_args)
