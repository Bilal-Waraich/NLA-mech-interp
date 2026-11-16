"""Stage 0: score the archived UMedPT SAE and baseline dictionaries under one rule.

--protocol thesis (default): the rule of run 1, unchanged. Outputs (results/stage0/):
  latents_<dictionary>.csv   one row per latent: firing rates, dataset AUROCs, label AUROCs, decisions
  summary.json               counts per dictionary, thresholds, settings
  codes_trained_step200k.npz cached image-level codes of the trained SAE (reused by later stages)

--protocol v1: the protocol's rule (sections 2-4): grouped splits, opacity labels, a fixed dataset-identity
band instead of a threshold calibrated on random SAEs, per-group Mann-Whitney tests with
Benjamini-Hochberg instead of Hanley-McNeil, latents chosen on val, PCA fitted on train tokens, and the
effect sizes of passing latents reported on --eval-split. Outputs (results/stage0_v1/baseline/):
  latents_<dictionary>.csv, summary.json, results.csv (tidy), config.json, run.log

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/stage0_baseline.py [--protocol thesis|v1] [--out DIR]
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
from medinterp.data.manifests import build_manifests
from medinterp.dictionaries.baselines import random_sae, signed_basis, signed_pca
from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.image_codes import image_max_codes
from medinterp.eval.metrics import apply_rule, latent_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage0"
TAU = 0.05                      # min fraction of images a latent must fire on, in every dataset
DSAUC_QUANTILE = 0.95           # dataset-identity threshold = this quantile of random-SAE latents
RANDOM_SEEDS = [0, 1, 2, 3, 4]
PCA_TOKENS_PER_DATASET = 50_000
LABEL_SETS = {
    "opacity3": ["chexpert", "rsna", "siim_covid"],          # airspace opacity vs no finding
    "all4": ["chexpert", "rsna", "siim_covid", "tb"],        # TB positive counted as group 0
}


LOG_FILES: list = []   # v1 also writes the log into its run directory


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for f in LOG_FILES:
        f.write(line + "\n")
        f.flush()


def pca_sample(acts: ArchivedActivations, per_dataset: int, seed: int = 0) -> torch.Tensor:
    """Seeded, dataset-balanced token sample for fitting PCA."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, block in acts.blocks.items():
        frac = min(1.0, per_dataset / block.n_tokens)
        for _, x in acts.iter_token_chunks(name, images_per_chunk=1024):
            keep = rng.random(x.shape[0]) < frac
            parts.append(x[torch.from_numpy(keep)])
    return torch.cat(parts)


def main_thesis(OUT: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acts = ArchivedActivations()
    names = list(acts.blocks)
    groups = {d: m.group.to_numpy() for d, m in build_manifests(acts).items()}
    trained = TopKSAE.from_checkpoint(DEFAULT_RUN_DIR / "checkpoints/step_0200000")
    k, d_sae = trained.k, trained.d_sae

    dictionaries = {
        "trained_step200k": lambda: trained,
        "trained_step050k": lambda: TopKSAE.from_checkpoint(DEFAULT_RUN_DIR / "checkpoints/step_0050000"),
        "trained_step020k": lambda: TopKSAE.from_checkpoint(DEFAULT_RUN_DIR / "checkpoints/step_0020000"),
        **{f"random_seed{s}": (lambda s=s: random_sae(acts.d_model, d_sae, k, seed=s)) for s in RANDOM_SEEDS},
        "signed_pca": lambda: signed_pca(pca_sample(acts, PCA_TOKENS_PER_DATASET), acts.d_model, k),
        "signed_basis": lambda: signed_basis(acts.d_model, k),
    }

    tables: dict[str, pd.DataFrame] = {}
    for dname, build in dictionaries.items():
        log(f"{dname}: encoding")
        codes = image_max_codes(acts, build())
        if dname == "trained_step200k":
            np.savez_compressed(OUT / "codes_trained_step200k.npz", **codes)
        log(f"{dname}: metrics")
        tables[dname] = latent_table(codes, groups)
        del codes

    # Calibrate the dataset-identity threshold on random SAEs only (latents active everywhere).
    rand = pd.concat([tables[f"random_seed{s}"] for s in RANDOM_SEEDS])
    rand_active = rand[np.all([rand[f"fire_{d}"] >= TAU for d in names], axis=0)]
    dsauc_thr = float(np.quantile(rand_active.dataset_auroc, DSAUC_QUANTILE))
    log(f"dataset-AUROC threshold (q{DSAUC_QUANTILE} of {len(rand_active)} active random latents) = {dsauc_thr:.4f}")

    thesis65 = set(json.loads((DEFAULT_RUN_DIR / "universality/eval/confirmed_universal_latents.json")
                              .read_text())["latents"])
    summary = {"settings": {"tau": TAU, "dsauc_quantile": DSAUC_QUANTILE, "dsauc_threshold": dsauc_thr,
                            "label_sets": LABEL_SETS, "k": k, "random_seeds": RANDOM_SEEDS},
               "dictionaries": {}}
    for dname, table in tables.items():
        row = {"n_latents": len(table)}
        for lname, lsets in LABEL_SETS.items():
            t = apply_rule(table, names, TAU, dsauc_thr, lsets)
            row.update({"active_all": int(t.active_all.sum()), "cross_dataset": int(t.cross_dataset.sum()),
                        f"label_consistent_{lname}": int(t.label_consistent.sum()),
                        f"label_consistent_{lname}_pos": int((t.label_consistent & (t.label_direction == "+")).sum())})
            if lname == "opacity3":
                t.to_csv(OUT / f"latents_{dname}.csv", index=False)
                if dname == "trained_step200k":
                    sub = t[t.latent.isin(thesis65)]
                    row["thesis65"] = {"active_all": int(sub.active_all.sum()),
                                       "cross_dataset": int(sub.cross_dataset.sum()),
                                       "label_consistent_opacity3": int(sub.label_consistent.sum())}
        summary["dictionaries"][dname] = row
        log(f"{dname}: {row}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {OUT}")


# ---------------------------------------------------------------- protocol v1

# Values fixed by docs/eval-protocol.md section 3 (dataset band, FDR); tau and the null set as in run 1.
V1_DSAUC_BAND = (0.35, 0.65)
V1_FDR = 0.05
V1_LABEL_DATASETS = ["chexpert", "rsna", "siim_covid"]


def main_v1(args) -> None:
    from medinterp.data.manifests import DEFAULT_LABEL_DIR
    from medinterp.runinfo import write_config
    from medinterp.stage0_v1 import (TIDY_COLUMNS, load_context, null_specs, parse_sae_args, subset,
                                     tidy_row, trained_spec)
    from medinterp.eval.metrics import apply_rule_v1, latent_table_v1

    out = args.out
    if (out / "summary.json").exists() and not args.force:
        log(f"{out / 'summary.json'} exists; nothing to do (use --force to recompute)")
        return
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILES.append(open(out / "run.log", "a"))
    ctx = load_context(DEFAULT_RUN_DIR, DEFAULT_LABEL_DIR, args.split_file)
    names = list(ctx.acts.blocks)
    saes = parse_sae_args(args.sae) or {f"seed{s}": ROOT / f"results/stage0/sae_seeds/seed{s}" for s in (0, 1, 2)}
    specs = [trained_spec(n, p, args.sae_fit_split) for n, p in saes.items()]
    k, d_sae = specs[0].meta["k"], specs[0].meta["d_sae"]
    specs += null_specs(ctx, d_sae, k, RANDOM_SEEDS, PCA_TOKENS_PER_DATASET, seed=0)
    settings = {"protocol": "v1", "tau": TAU, "dsauc_band": V1_DSAUC_BAND, "fdr": V1_FDR,
                "label_datasets": V1_LABEL_DATASETS, "select_split": "val", "eval_split": args.eval_split,
                "pooling": "max", "centring": "archive (per-dataset, all images; as the SAEs were trained)",
                "dictionaries": {s.name: s.meta for s in specs}}
    write_config(out, settings, ROOT, args.split_file)

    summary, tidy = {"settings": settings, "dictionaries": {}}, []
    for spec in specs:
        log(f"{spec.name}: encoding")
        codes = image_max_codes(ctx.acts, spec.build())
        sel = latent_table_v1(subset(codes, ctx.masks, "val"), subset(ctx.opacity, ctx.masks, "val"),
                              subset(ctx.groups, ctx.masks, "val"))
        t = apply_rule_v1(sel, names, V1_LABEL_DATASETS, TAU, V1_DSAUC_BAND, V1_FDR)
        # Effect sizes on the evaluation split, for the latents selected on val.
        ev = latent_table_v1(subset(codes, ctx.masks, args.eval_split),
                             subset(ctx.opacity, ctx.masks, args.eval_split),
                             subset(ctx.groups, ctx.masks, args.eval_split))
        for d in V1_LABEL_DATASETS:
            t[f"lauc_{d}_{args.eval_split}"] = ev[f"lauc_{d}"]
        t.to_csv(out / f"latents_{spec.name}.csv", index=False)
        passing = t[t.label_consistent]
        eff = np.min([np.abs(passing[f"lauc_{d}_{args.eval_split}"] - 0.5) for d in V1_LABEL_DATASETS], axis=0) \
            if len(passing) else np.array([])
        row = {"n_latents": len(t), "active_all": int(t.active_all.sum()), "cross_dataset": int(t.cross_dataset.sum()),
               "label_consistent": int(t.label_consistent.sum()),
               "label_consistent_pos": int((t.label_consistent & (t.label_direction == "+")).sum()),
               f"consistent_effect_{args.eval_split}_median": float(np.median(eff)) if eff.size else None,
               f"consistent_effect_{args.eval_split}_max": float(eff.max()) if eff.size else None}
        summary["dictionaries"][spec.name] = row
        for metric in ("active_all", "cross_dataset", "label_consistent"):
            tidy.append(tidy_row(spec, f"count_{metric}", "all", "val", row[metric]))
        log(f"{spec.name}: {row}")
        del codes
    pd.DataFrame(tidy, columns=TIDY_COLUMNS).to_csv(out / "results.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {out}")


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=("thesis", "v1"), default="thesis")
    ap.add_argument("--out", type=Path, help="output directory (default: results/stage0 or results/stage0_v1/baseline)")
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--eval-split", choices=("val", "test"), default="val",
                    help="v1: split for reported effect sizes; test only in a final evaluation")
    ap.add_argument("--sae", action="append", default=[], metavar="NAME=PATH",
                    help="v1: trained SAE checkpoint directory (repeatable; default: Stage 0.6 seeds 0-2)")
    ap.add_argument("--sae-fit-split", default="all",
                    help="v1: images the given SAEs were fitted on, if their cfg.json does not say")
    ap.add_argument("--force", action="store_true", help="v1: recompute even if summary.json exists")
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT if args.protocol == "thesis" else ROOT / "results/stage0_v1/baseline"
    return args


if __name__ == "__main__":
    _args = parse_args()
    if _args.protocol == "thesis":
        main_thesis(_args.out)
    else:
        main_v1(_args)
