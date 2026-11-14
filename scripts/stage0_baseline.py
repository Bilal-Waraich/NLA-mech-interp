"""Stage 0: score the archived UMedPT SAE and baseline dictionaries under one rule.

Outputs (results/stage0/):
  latents_<dictionary>.csv   one row per latent: firing rates, dataset AUROCs, label AUROCs, decisions
  summary.json               counts per dictionary, thresholds, settings
  codes_trained_step200k.npz cached image-level codes of the trained SAE (reused by later stages)

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/stage0_baseline.py
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

OUT = Path(__file__).resolve().parents[1] / "results/stage0"
TAU = 0.05                      # min fraction of images a latent must fire on, in every dataset
DSAUC_QUANTILE = 0.95           # dataset-identity threshold = this quantile of random-SAE latents
RANDOM_SEEDS = [0, 1, 2, 3, 4]
PCA_TOKENS_PER_DATASET = 50_000
LABEL_SETS = {
    "opacity3": ["chexpert", "rsna", "siim_covid"],          # airspace opacity vs no finding
    "all4": ["chexpert", "rsna", "siim_covid", "tb"],        # TB positive counted as group 0
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def main() -> None:
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


if __name__ == "__main__":
    main()
