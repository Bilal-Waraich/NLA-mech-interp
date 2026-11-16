"""Per-latent metrics on image-level codes and the Stage-0 decision rule.

Definitions (image level, per latent j):
  firing_rate[d, j]   fraction of images of dataset d where latent j fires on any patch
  dataset_auroc[j]    how well j's image-level activation identifies the source dataset:
                      mean over datasets of one-vs-rest AUROC, folded to >= 0.5 (direction-free)
  label_auroc[d, j]   within dataset d, AUROC of j's activation for airspace_opacity (group 0)
                      vs no_finding (group 5), with a Hanley-McNeil standard error

Decision rule (identical for every dictionary; thresholds calibrated on random SAEs where noted):
  active_all        firing_rate >= tau on every dataset
  cross_dataset     active_all AND dataset_auroc <= q95 of random-SAE latents that are active_all
  label_consistent  cross_dataset AND label_auroc's 95% CI excludes 0.5, same sign, on every dataset
                    in ``label_datasets``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

POSITIVE_GROUP, NEGATIVE_GROUP = 0, 5  # airspace_opacity vs no_finding


def auroc_columns(scores: np.ndarray, positive: np.ndarray, chunk: int = 512) -> np.ndarray:
    """AUROC of every column of ``scores`` [N, L] for binary ``positive`` [N]; ties get average rank.

    Mann-Whitney formulation: (sum of positive ranks - n1(n1+1)/2) / (n1 * n0). O(L * N log N).
    """
    n1 = int(positive.sum())
    n0 = positive.size - n1
    out = np.empty(scores.shape[1])
    for c0 in range(0, scores.shape[1], chunk):
        ranks = rankdata(scores[:, c0:c0 + chunk].astype(np.float32), axis=0)
        out[c0:c0 + chunk] = (ranks[positive].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def hanley_mcneil_se(auc: np.ndarray, n1: int, n0: int) -> np.ndarray:
    """Standard error of an AUROC estimate (Hanley & McNeil 1982)."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    return np.sqrt(np.clip(var, 0, None))


def latent_table(codes: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute all per-latent metrics for one dictionary. One row per latent."""
    names = list(codes)
    n_lat = next(iter(codes.values())).shape[1]
    table = pd.DataFrame({"latent": np.arange(n_lat)})

    for d in names:
        table[f"fire_{d}"] = (codes[d] > 0).mean(0)

    # Dataset identity: one-vs-rest AUROC per dataset on the pooled images, folded, averaged.
    pooled = np.concatenate([codes[d] for d in names])
    owner = np.concatenate([np.full(len(codes[d]), i) for i, d in enumerate(names)])
    per_ds = []
    for i, d in enumerate(names):
        a = auroc_columns(pooled, owner == i)
        per_ds.append(np.maximum(a, 1 - a))
        table[f"dsauc_{d}"] = per_ds[-1]
    table["dataset_auroc"] = np.mean(per_ds, axis=0)
    del pooled

    # Label association within each dataset (airspace_opacity vs no_finding).
    for d in names:
        g = groups[d]
        keep = (g == POSITIVE_GROUP) | (g == NEGATIVE_GROUP)
        pos = g[keep] == POSITIVE_GROUP
        auc = auroc_columns(codes[d][keep], pos)
        table[f"lauc_{d}"] = auc
        table[f"lauc_se_{d}"] = hanley_mcneil_se(auc, int(pos.sum()), int((~pos).sum()))
    return table


def apply_rule(table: pd.DataFrame, datasets: list[str], tau: float, dsauc_threshold: float,
               label_datasets: list[str]) -> pd.DataFrame:
    """Add boolean decision columns; thresholds are passed in so all dictionaries share them."""
    t = table.copy()
    t["active_all"] = np.all([t[f"fire_{d}"] >= tau for d in datasets], axis=0)
    t["cross_dataset"] = t.active_all & (t.dataset_auroc <= dsauc_threshold)
    # Significant on each label dataset, and all in the same direction.
    # SE is exactly 0 at perfect separation (AUROC 0 or 1); floor it so such latents count as significant.
    z = np.stack([(t[f"lauc_{d}"] - 0.5) / t[f"lauc_se_{d}"].clip(lower=1e-9) for d in label_datasets])
    sig_pos = np.all(z > 1.96, axis=0)
    sig_neg = np.all(z < -1.96, axis=0)
    t["label_consistent"] = t.cross_dataset & (sig_pos | sig_neg)
    t["label_direction"] = np.where(sig_pos, "+", np.where(sig_neg, "-", ""))
    return t


# ---------------------------------------------------------------- protocol v1 (sections 3 and 4)

def latent_table_v1(codes: dict[str, np.ndarray], opacity: dict[str, np.ndarray],
                    groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-latent statistics for the v1 counting rule, on one split's image-level codes.

    Columns: ``fire_<d>`` firing rate; ``dsauc_<d>`` one-vs-rest dataset AUROC, unfolded (direction
    kept); ``lauc_<d>`` opacity AUROC and ``lp_<d>`` its p-value from the per-group Mann-Whitney test, for
    every dataset with both opacity classes. ``groups`` gives the cluster key of each image.
    """
    from .stats import grouped_mannwhitney

    names = list(codes)
    n_lat = next(iter(codes.values())).shape[1]
    table = pd.DataFrame({"latent": np.arange(n_lat)})
    for d in names:
        table[f"fire_{d}"] = (codes[d] > 0).mean(0)
    pooled = np.concatenate([codes[d] for d in names])
    owner = np.concatenate([np.full(len(codes[d]), i) for i, d in enumerate(names)])
    for i, d in enumerate(names):
        table[f"dsauc_{d}"] = auroc_columns(pooled, owner == i)
    del pooled
    for d in names:
        lab = opacity[d]
        keep = ~np.isnan(lab)
        if keep.sum() == 0 or len(np.unique(lab[keep])) < 2:
            continue
        p, auc = grouped_mannwhitney(codes[d][keep].astype(np.float32), lab[keep] == 1, groups[d][keep])
        table[f"lauc_{d}"], table[f"lp_{d}"] = auc, p
    return table


def apply_rule_v1(table: pd.DataFrame, datasets: list[str], label_datasets: list[str], tau: float,
                  dsauc_band: tuple[float, float], fdr: float) -> pd.DataFrame:
    """Add the v1 decision columns. All thresholds are passed in (they live in the protocol).

    active_all        firing rate >= tau on every dataset
    cross_dataset     active_all and every one-vs-rest dataset AUROC inside ``dsauc_band``
    label_consistent  cross_dataset and, on every label dataset, the opacity test is a discovery under
                      Benjamini-Hochberg over all (latent, label dataset) tests of this dictionary, with
                      the same direction everywhere
    """
    from .stats import benjamini_hochberg

    t = table.copy()
    lo, hi = dsauc_band
    t["active_all"] = np.all([t[f"fire_{d}"] >= tau for d in datasets], axis=0)
    t["cross_dataset"] = t.active_all & np.all(
        [(t[f"dsauc_{d}"] >= lo) & (t[f"dsauc_{d}"] <= hi) for d in datasets], axis=0)
    p = np.stack([t[f"lp_{d}"].to_numpy() for d in label_datasets])          # [n_label_ds, n_latents]
    _, reject = benjamini_hochberg(p.ravel(), fdr)
    reject = reject.reshape(p.shape)
    for i, d in enumerate(label_datasets):
        t[f"ldisc_{d}"] = reject[i]
    up = np.stack([t[f"lauc_{d}"] > 0.5 for d in label_datasets])
    pos = reject.all(0) & up.all(0)
    neg = reject.all(0) & (~up).all(0)
    t["label_consistent"] = t.cross_dataset & (pos | neg)
    t["label_direction"] = np.where(pos, "+", np.where(neg, "-", ""))
    return t
