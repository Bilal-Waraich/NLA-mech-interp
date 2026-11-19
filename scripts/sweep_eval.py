"""Evaluate the sweep (protocol sections 4-5): P1.1 and P1.3 per setting and checkpoint, secondary metrics.

For every setting whose runs are all finished, and every checkpoint step they share:
  - per run: own-decoder FVE on the eval tokens with a cluster-bootstrap interval; dead latents; latents
    active on every image
  - all seed pairs: one-to-one decoder cosine; S (share at the P1.3 threshold) and its min-max over pairs
  - data-aware null, anchored on the setting's first run: covariance-shaped random directions; its q99.9
    and the share of anchor latents above it in every other seed
  - functional matching on token-level sparse codes, anchor against every other run, with its own null
  - phase 2b: pairs split into "init only", "batch order only" and "both" differing
Per (d_sae, k): the nulls of P1.1 at matched sparsity (random SAEs, signed PCA and signed basis fitted on
train tokens, and the frozen-random-decoder run of phase 1n where it exists).

``--phase0`` adds the Stage 0.6 seeds 0-2 (fitted on all images) as setting ``phase0_alldata_d4096_k48``.
``--probes`` additionally runs ``stage0_metrics.py --protocol v1`` on the final checkpoints of the listed
settings (P1.2 and selectivity).

Results are cached per (setting, step) under ``<out>/cache``; a cached item is not recomputed unless
``--force``. Outputs: ``results.csv`` (tidy), ``pairs.csv``, ``fve.csv``, ``p11.csv``.
The eval tokens are those of the val split; the test split is never read here.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/sweep_eval.py --phase0
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.data.manifests import DEFAULT_LABEL_DIR
from medinterp.dictionaries.baselines import random_sae, signed_basis, signed_pca
from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.stability import TokenMoments
from medinterp.eval.stats import complete_separation
from medinterp.eval.sweep_metrics import (covariance_null, decoder_overlap, functional_null, functional_overlap,
                                          load_eval_tokens, score_dictionary)
from medinterp.runinfo import write_config
from medinterp.stage0_v1 import TIDY_COLUMNS, load_context, train_token_sample
from medinterp.sweep import checkpoint_steps, overlap_share, seedsplit_base_rows, summarise_pairs

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
EVAL_SPLIT = "val"
# Protocol values (sections 3-5): P1.3 threshold, null quantile, bootstrap settings, random-SAE seeds.
S_THRESHOLD = 0.7
NULL_QUANTILE = 0.999
N_BOOT, CONFIDENCE = 2000, 0.95
NULL_SEEDS = (0, 1)
RANDOM_SAE_SEEDS = (0, 1, 2, 3, 4)
PCA_TOKENS_PER_DATASET = 50_000
SEED = 0
PHASE0 = "phase0_alldata_d4096_k48"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


LOG_FILE = None


def collect_settings(table: pd.DataFrame, runs_root: Path, phase0_dir: Path | None) -> dict[str, dict]:
    """``{setting: {"meta": ..., "runs": {run_id: {"init_seed", "data_seed", "steps": {step: path}}}}}``
    for settings whose runs are all finished. Frozen-decoder settings are returned with kind "null"."""
    out = {}
    for setting, part in table.groupby("setting", sort=False):
        first = part.iloc[0]
        runs = {}
        rows = list(part.itertuples())
        if first.phase == "2b":
            # The 2 x 2 design reuses the phase 1 runs (0, 0) and (1, 1) of the same setting.
            rows = list(seedsplit_base_rows(table, int(first.d_sae), int(first.k)).itertuples()) + rows
        if not all((runs_root / r.run_id / "DONE.json").exists() for r in rows):
            continue
        for r in rows:
            runs[r.run_id] = {"init_seed": int(r.init_seed), "data_seed": int(r.data_seed),
                              "steps": {s: runs_root / r.run_id / f"step_{s}" for s in checkpoint_steps(r._asdict())}}
        out[setting] = {"meta": {"phase": first.phase, "d_sae": int(first.d_sae), "k": int(first.k),
                                 "kind": "null" if first.decoder != "trained" else "trained",
                                 "decoder": first.decoder},
                        "runs": runs}
    if phase0_dir is not None:
        seeds = phase0_dir
        out = {PHASE0: {"meta": {"phase": "0", "d_sae": 4096, "k": 48, "kind": "trained", "decoder": "trained"},
                        "runs": {f"alldata_seed{s}": {"init_seed": s, "data_seed": s,
                                                      "steps": {20_000: seeds / f"seed{s}"}}
                                 for s in (0, 1, 2)}},
               **out}
    return out


def train_covariance(ctx, cache: Path) -> torch.Tensor:
    if cache.exists():
        return torch.load(cache)
    moments = TokenMoments(ctx.acts.d_model)
    for name in ctx.acts.blocks:
        train = ctx.masks[name]["train"]
        for img0, x in ctx.acts.iter_token_chunks(name, 1024):
            n_img = x.shape[0] // ctx.acts.n_patches
            keep = np.repeat(train[img0:img0 + n_img], ctx.acts.n_patches)
            moments.add(x[torch.from_numpy(keep)])
    cov = moments.covariance()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cov, cache)
    return cov


def tidy(dictionary: str, d_sae, k, steps, init_seed, data_seed, metric: str, subset: str, value,
         ci_low=None, ci_high=None, split: str = EVAL_SPLIT) -> dict:
    return {"harvest": "thesis", "encoder": "umedpt", "layer": "final", "dictionary": dictionary, "d_sae": d_sae,
            "k": k, "steps": steps, "init_seed": init_seed, "data_seed": data_seed, "metric": metric,
            "subset": subset, "split": split, "value": value, "ci_low": ci_low, "ci_high": ci_high}


def score_row(name, kind, setting, d_sae, k, step, init_seed, data_seed, sc) -> tuple[dict, list[dict]]:
    fve = {"setting": setting, "dictionary": name, "dictionary_kind": kind, "d_sae": d_sae, "k": k, "step": step,
           "init_seed": init_seed, "data_seed": data_seed, "value": sc.fve, "ci_low": sc.fve_ci[0],
           "ci_high": sc.fve_ci[1], "n_dead": sc.n_dead, "n_every_image": sc.n_every_image}
    rows = [tidy(name, d_sae, k, step, init_seed, data_seed, "fve_own_decoder", "all", sc.fve, *sc.fve_ci),
            tidy(name, d_sae, k, step, init_seed, data_seed, "n_dead_latents", "all", sc.n_dead),
            tidy(name, d_sae, k, step, init_seed, data_seed, "n_latents_active_every_image", "all", sc.n_every_image)]
    return fve, rows


def evaluate_setting(setting: str, info: dict, step: int, tokens, cov, device: str, functional_max: int) -> dict:
    """All measurements of one setting at one checkpoint step (see the module docstring)."""
    meta, runs = info["meta"], info["runs"]
    d_sae, k = meta["d_sae"], meta["k"]
    order = sorted(runs, key=lambda r: (runs[r]["init_seed"], runs[r]["data_seed"]))
    saes = {r: TopKSAE.from_checkpoint(runs[r]["steps"][step]) for r in order}
    result = {"fve": [], "tidy": [], "pairs": []}
    for r in order:
        sc = score_dictionary(saes[r], tokens, N_BOOT, CONFIDENCE, SEED, device)
        f, t = score_row(r, meta["kind"] if meta["kind"] == "trained" else meta["decoder"], setting, d_sae, k, step,
                         runs[r]["init_seed"], runs[r]["data_seed"], sc)
        result["fve"].append(f)
        result["tidy"] += t
        log(f"{setting} step {step} {r}: FVE {sc.fve:.4f} [{sc.fve_ci[0]:.4f}, {sc.fve_ci[1]:.4f}], "
            f"dead {sc.n_dead}, every image {sc.n_every_image}")
    if meta["kind"] != "trained" or len(order) < 2:
        return result

    anchor = order[0]
    null = covariance_null(saes[anchor].W_dec, cov, NULL_SEEDS)
    null_q = float(np.nanquantile(null, NULL_QUANTILE))
    s_vals, s_null_vals = [], []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            cos = decoder_overlap(saes[a].W_dec, saes[b].W_dec)
            s, s_null = overlap_share(cos, S_THRESHOLD), overlap_share(cos, null_q)
            s_vals.append(s)
            s_null_vals.append(s_null)
            ra, rb = runs[a], runs[b]
            differs = ("init only" if ra["data_seed"] == rb["data_seed"] else
                       "batch order only" if ra["init_seed"] == rb["init_seed"] else "both")
            result["pairs"].append({"setting": setting, "step": step, "match": "decoder", "run_a": a, "run_b": b,
                                    "seeds_differ": differs, "mean": float(np.nanmean(cos)),
                                    f"share_ge_{S_THRESHOLD}": s, "share_ge_null_q999": s_null,
                                    "null_q999": null_q})
    S = summarise_pairs(s_vals)
    S_null = summarise_pairs(s_null_vals)
    null_share = overlap_share(null, S_THRESHOLD)
    result["S"] = S
    result["null"] = {"q999": null_q, f"share_ge_{S_THRESHOLD}": null_share, "mean": float(np.nanmean(null))}
    result["tidy"] += [
        tidy(setting, d_sae, k, step, None, None, "seed_overlap_S", "all_pairs", S["mean"], S["min"], S["max"], "none"),
        tidy(setting, d_sae, k, step, None, None, "share_above_covariance_null_q999", "all_pairs",
             S_null["mean"], S_null["min"], S_null["max"], "none"),
        tidy(setting, d_sae, k, step, None, None, "covariance_null_q999", "anchor", null_q, split="none"),
        tidy(setting, d_sae, k, step, None, None, f"covariance_null_share_ge_{S_THRESHOLD}", "anchor", null_share,
             split="none"),
    ]
    log(f"{setting} step {step}: S = {S['mean']:.4f} ({S['min']:.4f}-{S['max']:.4f}, {S['n_pairs']} pairs); "
        f"null q99.9 {null_q:.3f}, null share >= {S_THRESHOLD} {null_share:.4f}")

    if meta["phase"] == "2b":
        p = pd.DataFrame(result["pairs"])
        for kind, part in p.groupby("seeds_differ"):
            v = summarise_pairs(part[f"share_ge_{S_THRESHOLD}"])
            result["tidy"].append(tidy(setting, d_sae, k, step, None, None, "seed_overlap_S", f"differ: {kind}",
                                       v["mean"], v["min"], v["max"], "none"))

    if d_sae <= functional_max:
        fnull = functional_null(saes[anchor], cov, k, NULL_SEEDS[:1], tokens, device)
        fq = float(np.nanquantile(fnull, NULL_QUANTILE))
        f_vals, f_null_vals = [], []
        for b in order[1:]:
            sc, dead_a, dead_b = functional_overlap(saes[anchor], saes[b], tokens, device)
            f_vals.append(overlap_share(sc, S_THRESHOLD))
            f_null_vals.append(overlap_share(sc, fq))
            result["pairs"].append({"setting": setting, "step": step, "match": "functional", "run_a": anchor,
                                    "run_b": b, "mean": float(np.nanmean(sc)), f"share_ge_{S_THRESHOLD}": f_vals[-1],
                                    "share_ge_null_q999": f_null_vals[-1], "null_q999": fq,
                                    "dead_a": dead_a, "dead_b": dead_b})
        F, F_null = summarise_pairs(f_vals), summarise_pairs(f_null_vals)
        f_null_share = overlap_share(fnull, S_THRESHOLD)
        result["functional"] = {"share": F, "share_above_null": F_null, "null_q999": fq,
                                f"null_share_ge_{S_THRESHOLD}": f_null_share}
        result["tidy"] += [
            tidy(setting, d_sae, k, step, None, None, f"functional_share_ge_{S_THRESHOLD}", "anchor_pairs",
                 F["mean"], F["min"], F["max"]),
            tidy(setting, d_sae, k, step, None, None, "functional_share_above_null_q999", "anchor_pairs",
                 F_null["mean"], F_null["min"], F_null["max"]),
            tidy(setting, d_sae, k, step, None, None, "functional_null_q999", "anchor", fq),
            tidy(setting, d_sae, k, step, None, None, f"functional_null_share_ge_{S_THRESHOLD}", "anchor",
                 f_null_share),
        ]
        log(f"{setting} step {step}: functional share >= {S_THRESHOLD} {F['mean']:.4f} "
            f"({F['min']:.4f}-{F['max']:.4f}); above own null q99.9 ({fq:.3f}) {F_null['mean']:.4f}; "
            f"null share >= {S_THRESHOLD} {f_null_share:.4f}")
    return result


def evaluate_nulls(ctx, d_sae: int, k: int, tokens, device: str, pca_sample: torch.Tensor) -> dict:
    """Own-decoder FVE of the matched-sparsity nulls of P1.1 (the frozen-decoder runs are scored as
    settings of their own)."""
    d = ctx.acts.d_model
    nulls = {f"random_seed{s}": random_sae(d, d_sae, k, seed=s) for s in RANDOM_SAE_SEEDS}
    nulls["signed_pca"] = signed_pca(pca_sample, d, k)
    nulls["signed_basis"] = signed_basis(d, k)
    result = {"fve": [], "tidy": []}
    for name, enc in nulls.items():
        sc = score_dictionary(enc, tokens, N_BOOT, CONFIDENCE, SEED, device)
        kind = "random_sae" if name.startswith("random") else name
        f, t = score_row(name, kind, f"nulls_d{d_sae}_k{k}", enc.d_sae, k, 0, None, None, sc)
        result["fve"].append(f)
        result["tidy"] += t
        log(f"null {name} d{d_sae} k{k}: FVE {sc.fve:.4f}")
    return result


def run_probes(settings: dict, names: list[str], args) -> None:
    for name in names:
        info = settings[name]
        out = args.out / "probes" / name
        cmd = [sys.executable, str(REPO / "scripts/stage0_metrics.py"), "--protocol", "v1", "--out", str(out),
               "--split-file", str(args.split_file), "--sae-fit-split", "train"]
        for r, rinfo in info["runs"].items():
            last = max(rinfo["steps"])
            cmd += ["--sae", f"{r}={rinfo['steps'][last]}"]
        if name == PHASE0:
            cmd[cmd.index("train")] = "all"
        log(f"probes {name}: {' '.join(cmd[2:])}")
        subprocess.run(cmd, check=True)


def p11_table(fve: pd.DataFrame) -> pd.DataFrame:
    """P1.1 per (d_sae, k) at the final 20k checkpoint of phase-1 settings: trained seeds vs every null."""
    rows = []
    trained = fve[(fve.dictionary_kind == "trained") & fve.setting.str.match(r"^d\d+_k\d+_s20000$")
                  & (fve.step == 20_000)]
    for (d_sae, k), part in trained.groupby(["d_sae", "k"]):
        nulls = fve[(fve.dictionary_kind != "trained") & (fve.k == k)
                    & ((fve.setting == f"nulls_d{d_sae}_k{k}")
                       | (fve.setting == f"d{d_sae}_k{k}_s20000_frozeniso") & (fve.step == 20_000))]
        if nulls.empty:
            continue
        sep = complete_separation(part.value.to_numpy(), nulls.value.to_numpy())
        best = nulls.loc[nulls.value.idxmax()]
        rows.append({"d_sae": d_sae, "k": k, "n_trained": len(part), "n_null": len(nulls),
                     "worst_trained": part.value.min(), "best_null": best.dictionary,
                     "best_null_value": best.value, "gap": sep.gap, "separated": sep.separated,
                     "p_exact": sep.p_value})
    return pd.DataFrame(rows)


def main() -> None:
    global LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/sweep_v1.csv")
    ap.add_argument("--runs", type=Path, default=ROOT / "results/sweep_v1/runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/sweep_v1/eval")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--device", help="default: cuda, else mps, else cpu")
    ap.add_argument("--phase0", action="store_true", help="include the Stage 0.6 all-data seeds 0-2")
    ap.add_argument("--phase0-dir", type=Path, default=REPO / "results/stage0/sae_seeds",
                    help="directory holding seed0, seed1, seed2 of Stage 0.6")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only merge the caches under --out (e.g. fetched from several Kaggle runs)")
    ap.add_argument("--settings", nargs="*", help="only these settings (default: every finished one)")
    ap.add_argument("--functional-max-dsae", type=int, default=4096,
                    help="skip functional matching above this size (its correlation matrix is d_sae^2 float64)")
    ap.add_argument("--probes", nargs="*", default=None, metavar="SETTING",
                    help="also run stage0_metrics.py v1 (P1.2, selectivity) on these settings' final checkpoints")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(args.out / "run.log", "a")
    table = pd.read_csv(args.config)
    settings = collect_settings(table, args.runs, args.phase0_dir if args.phase0 else None)
    if args.settings:
        settings = {s: v for s, v in settings.items() if s in args.settings}
    log(f"device {device}; finished settings: {list(settings)}")
    if not args.aggregate_only:
        write_config(args.out, {"eval_split": EVAL_SPLIT, "s_threshold": S_THRESHOLD, "null_quantile": NULL_QUANTILE,
                            "n_bootstrap": N_BOOT, "confidence": CONFIDENCE, "null_seeds": NULL_SEEDS,
                            "random_sae_seeds": RANDOM_SAE_SEEDS, "seed": SEED, "config": str(args.config),
                            "settings": {s: {"meta": v["meta"], "runs": list(v["runs"])} for s, v in settings.items()}},
                 REPO, args.split_file)

    ctx = None if args.aggregate_only else load_context(args.run_dir, args.label_dir, args.split_file)
    cache = args.out / "cache"
    tokens = None
    cov = None

    def need_tokens():
        nonlocal tokens, cov
        if tokens is None:
            log(f"loading {EVAL_SPLIT} tokens")
            tokens = load_eval_tokens(ctx.acts, ctx.masks, ctx.groups, EVAL_SPLIT)
            log(f"{tokens.n_images} images, {tokens.x.shape[0]} tokens")
            cov = train_covariance(ctx, cache / "cov_train.pt")
        return tokens, cov

    if args.aggregate_only:
        settings = {}
    for name, info in settings.items():
        steps = sorted(set.intersection(*(set(r["steps"]) for r in info["runs"].values())))
        for step in steps:
            f = cache / name / f"step_{step}.json"
            if f.exists() and not args.force:
                continue
            tok, cv = need_tokens()
            res = evaluate_setting(name, info, step, tok, cv, device, args.functional_max_dsae)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(res, indent=1, default=float))

    pca_sample = None
    for d_sae, k in sorted({(v["meta"]["d_sae"], v["meta"]["k"]) for v in settings.values()
                            if v["meta"]["kind"] == "trained"}):
        f = cache / "nulls" / f"d{d_sae}_k{k}.json"
        if f.exists() and not args.force:
            continue
        tok, _ = need_tokens()
        if pca_sample is None:
            pca_sample = train_token_sample(ctx, PCA_TOKENS_PER_DATASET, SEED)
        res = evaluate_nulls(ctx, d_sae, k, tok, device, pca_sample)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, indent=1, default=float))

    fve, tidy_rows, pairs = [], [], []
    for f in sorted(cache.glob("*/*.json")):
        res = json.loads(f.read_text())
        fve += res["fve"]
        tidy_rows += res["tidy"]
        pairs += res.get("pairs", [])
    fve = pd.DataFrame(fve)
    fve.to_csv(args.out / "fve.csv", index=False)
    pd.DataFrame(tidy_rows, columns=TIDY_COLUMNS).to_csv(args.out / "results.csv", index=False)
    pd.DataFrame(pairs).to_csv(args.out / "pairs.csv", index=False)
    if not fve.empty:
        p11 = p11_table(fve)
        p11.to_csv(args.out / "p11.csv", index=False)
        if not p11.empty:
            log("P1.1 (val, exploratory):\n" + p11.to_string(index=False))
    if args.probes is not None:
        run_probes(settings, args.probes or list(settings), args)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
