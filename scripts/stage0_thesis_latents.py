"""Thesis-latent check (protocol section 4): do the latents of the thesis SAE recur functionally in other seeds?

The thesis SAE (step 200,000) is the anchor; Stage 0.6 seeds 0-2 (same recipe, fitted on all images, like
the thesis SAE) are the other seeds. For every thesis latent and seed: the one-to-one code correlation and the
best code correlation with any latent of the seed (many-to-one), on val tokens. A functional null (the anchor
against a top-k dictionary of covariance-shaped random directions) gives the chance level. The group of 65
latents the thesis reported is compared with all live latents.

The result concerns the thesis and stays private: it goes into the local notes, never into public text.

Outputs in ``--out``: ``latents.csv`` (one row per thesis latent), ``summary.json``, ``config.json``,
``run.log``. A run whose ``summary.json`` exists is not repeated unless ``--force``.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/stage0_thesis_latents.py
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.data.manifests import DEFAULT_LABEL_DIR
from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.stability import TokenMoments
from medinterp.eval.sweep_metrics import functional_null, functional_partner_scores, load_eval_tokens
from medinterp.runinfo import write_config
from medinterp.stage0_v1 import load_context, summarise_latent_group

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
# Protocol values (sections 4-5): matching threshold, null quantile, null seed, evaluation split.
THRESHOLD = 0.7
NULL_QUANTILE = 0.999
NULL_SEEDS = (0,)
EVAL_SPLIT = "val"
SEEDS = ["seed0", "seed1", "seed2"]


def log(msg: str, handle) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="thesis archive (activations, SAE)")
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--seeds-dir", type=Path, default=ROOT / "results/stage0/sae_seeds")
    ap.add_argument("--cov", type=Path, default=ROOT / "results/sweep_v1/eval/cache/cov_train.pt",
                    help="covariance of the train tokens; computed if the file is missing")
    ap.add_argument("--out", type=Path, default=ROOT / "results/stage0/thesis_latents")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if (args.out / "summary.json").exists() and not args.force:
        print(f"{args.out / 'summary.json'} exists; nothing to do")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    handle = open(args.out / "run.log", "a")
    write_config(args.out, {"anchor": "thesis step_0200000", "seeds": SEEDS, "threshold": THRESHOLD,
                            "null_quantile": NULL_QUANTILE, "null_seeds": NULL_SEEDS, "eval_split": EVAL_SPLIT},
                 REPO, args.split_file)

    ctx = load_context(args.run_dir, args.label_dir, args.split_file)
    tokens = load_eval_tokens(ctx.acts, ctx.masks, ctx.groups, EVAL_SPLIT)
    log(f"{tokens.n_images} {EVAL_SPLIT} images, {tokens.x.shape[0]} tokens", handle)
    if args.cov.exists():
        cov = torch.load(args.cov)
    else:
        moments = TokenMoments(ctx.acts.d_model)
        for name in ctx.acts.blocks:
            train = ctx.masks[name]["train"]
            for img0, x in ctx.acts.iter_token_chunks(name, 1024):
                n_img = x.shape[0] // ctx.acts.n_patches
                moments.add(x[torch.from_numpy(np.repeat(train[img0:img0 + n_img], ctx.acts.n_patches))])
        cov = moments.covariance()

    thesis = TopKSAE.from_checkpoint(args.run_dir / "checkpoints/step_0200000")
    thesis65 = set(json.loads((args.run_dir / "universality/eval/confirmed_universal_latents.json").read_text())["latents"])
    table = pd.DataFrame({"latent": np.arange(thesis.d_sae)})
    table["thesis65"] = table.latent.isin(thesis65)
    for s in SEEDS:
        started = time.time()
        o2o, best = functional_partner_scores(thesis, TopKSAE.from_checkpoint(args.seeds_dir / s), tokens, args.device)
        table[f"one_to_one_{s}"], table[f"best_{s}"] = o2o, best
        log(f"{s}: done in {time.time() - started:.0f} s", handle)
    null = functional_null(thesis, cov, thesis.k, NULL_SEEDS, tokens, args.device)
    null_q = float(np.nanquantile(null, NULL_QUANTILE))
    log(f"functional null q{NULL_QUANTILE * 100:g} = {null_q:.4f}", handle)
    table.to_csv(args.out / "latents.csv", index=False)

    live = table[[f"best_{s}" for s in SEEDS]].notna().all(axis=1)
    summary = {"null_quantile_value": null_q, "n_live_in_thesis_sae": int(live.sum()),
               "thesis65": summarise_latent_group(table[table.thesis65], SEEDS, THRESHOLD, null_q),
               "all_live": summarise_latent_group(table[live], SEEDS, THRESHOLD, null_q),
               "thesis65_dead_on_val": int((table.thesis65 & ~live).sum())}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    for group in ("thesis65", "all_live"):
        g = summary[group]
        log(f"{group} (n = {g['n']}): one-to-one >= {THRESHOLD} {g['mean_one_to_one_ge_threshold']:.3f}, "
            f"above null {g['mean_one_to_one_ge_null']:.3f}, best >= {THRESHOLD} {g['mean_best_ge_threshold']:.3f}, "
            f"median best {g['mean_median_best']:.3f}, one-to-one in every seed "
            f"{g['one_to_one_ge_threshold_in_every_seed']:.3f}, best in every seed {g['best_ge_threshold_in_every_seed']:.3f}",
            handle)


if __name__ == "__main__":
    main()
