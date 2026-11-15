"""Stage 0.6: seed stability of the SAE dictionary, and how it relates to the run-1/run-2 metrics.

Runs compared (decoder directions only, no activations needed):
  thesis_step200k, thesis_step020k   archived thesis SAE (one unseeded run), final and 20k checkpoints
  seed0, seed1, seed2                SAEs retrained here with explicit seeds, 20k steps
  random0, random1                   untrained SAEs: the chance level for every match score

Stable latent (anchored on thesis_step200k, which carries the run-1/run-2 metrics): its one-to-one
partner in each of seed0, seed1, seed2 has cosine >= threshold. Thresholds: the q99.9 of the random
null, and the conventional 0.7 / 0.9.

Outputs (results/stage0/stability/):
  pairs.csv        per pair of runs: mean / median one-to-one cosine, fraction above each threshold
  latents.csv      per thesis latent: partner cosines per seed, stability flags, run-1/run-2 metrics
  summary.json     counts, null threshold, metric comparison stable vs unstable
Run:  ~/.venvs/nla-mech-interp/bin/python scripts/stage0_seed_stability.py
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def main() -> None:
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


if __name__ == "__main__":
    main()
