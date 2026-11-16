"""Inference primitives of the evaluation protocol (section 3).

  cluster_bootstrap             percentile interval of a statistic, resampling whole groups (patients)
  cluster_bootstrap_difference  the same for a paired difference: both statistics see the same resample
  benjamini_hochberg            adjusted p-values and decisions controlling the false discovery rate
  holm                          adjusted p-values and decisions controlling the family-wise error rate
  complete_separation           do all values of one set beat all values of another, by a margin; with
                                the exact one-sided permutation p-value of the rank-sum statistic
  grouped_mannwhitney           per-column Mann-Whitney test after reducing each group to one value

The numbers the protocol fixes (resamples, levels) are passed in by the caller.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


@dataclass
class BootstrapResult:
    estimate: np.ndarray | float     # statistic on the original sample
    ci_low: np.ndarray | float
    ci_high: np.ndarray | float
    samples: np.ndarray              # [n_resamples, ...] statistic on each resample


def _group_index(groups: np.ndarray) -> list[np.ndarray]:
    """Row indices of each group, in the order of the sorted unique group ids."""
    codes, uniques = pd.factorize(pd.Series(groups), sort=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniques))]


def _resample_indices(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    picks = rng.integers(0, len(members), len(members))
    return np.concatenate([members[g] for g in picks])


def cluster_bootstrap(statistic: Callable[[np.ndarray], float | np.ndarray], groups: np.ndarray,
                      n_resamples: int, confidence: float, seed: int) -> BootstrapResult:
    """Percentile bootstrap interval of ``statistic`` with whole groups resampled with replacement.

    ``statistic(idx)`` receives an array of row indices (rows of one group stay together, repeated as
    often as the group is drawn) and returns a scalar or an array. Rows are identified by position in
    ``groups``. Complexity: O(n_resamples * cost of statistic).
    """
    members = _group_index(np.asarray(groups))
    rng = np.random.default_rng(seed)
    estimate = np.asarray(statistic(np.arange(len(groups))), dtype=float)
    samples = np.stack([np.asarray(statistic(_resample_indices(members, rng)), dtype=float)
                        for _ in range(n_resamples)])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha], axis=0)
    unwrap = (lambda a: float(a)) if estimate.ndim == 0 else (lambda a: a)
    return BootstrapResult(unwrap(estimate), unwrap(lo), unwrap(hi), samples)


def cluster_bootstrap_difference(stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
                                 groups: np.ndarray, n_resamples: int, confidence: float,
                                 seed: int) -> BootstrapResult:
    """Paired bootstrap of ``stat_a - stat_b``: both are evaluated on the same resampled rows."""
    return cluster_bootstrap(lambda idx: stat_a(idx) - stat_b(idx), groups, n_resamples, confidence, seed)


def benjamini_hochberg(p: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure. Returns (adjusted p-values, reject mask).

    NaN p-values (untestable hypotheses) are excluded from the family and returned as NaN / False.
    Complexity: O(m log m).
    """
    p = np.asarray(p, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = q.size
    if m:
        order = np.argsort(q)
        scaled = q[order] * m / np.arange(1, m + 1)
        # Step-up: the adjusted value of rank i is the minimum over ranks >= i.
        stepped = np.minimum.accumulate(scaled[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(stepped, 1.0)
        adj[ok] = out
    return adj, np.where(ok, adj <= alpha, False)


def holm(p: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down procedure. Returns (adjusted p-values, reject mask). Complexity: O(m log m)."""
    p = np.asarray(p, dtype=float)
    if np.isnan(p).any():
        raise ValueError("Holm needs a p-value for every hypothesis")
    m = p.size
    order = np.argsort(p)
    scaled = p[order] * (m - np.arange(m))
    stepped = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adj = np.empty(m)
    adj[order] = stepped
    return adj, adj <= alpha


@dataclass
class SeparationResult:
    separated: bool      # every value of ``better`` exceeds every value of ``worse`` by at least margin
    gap: float           # worst of ``better`` minus best of ``worse`` (oriented so that larger is better)
    p_value: float       # exact one-sided permutation p of the rank-sum of ``better``


def _exact_rank_sum_p(better: np.ndarray, worse: np.ndarray) -> float:
    """Share of all ways to label n_b of the pooled values as 'better' whose rank sum is at least the
    observed one. Ties get average ranks. Exact enumeration: O(C(n, n_b) * n_b)."""
    pooled = np.concatenate([better, worse])
    ranks = pd.Series(pooled).rank(method="average").to_numpy()
    observed = ranks[:len(better)].sum()
    n, k = len(pooled), len(better)
    if math.comb(n, k) > 2_000_000:
        raise ValueError("too many permutations for exact enumeration")
    count = total = 0
    for combo in itertools.combinations(range(n), k):
        total += 1
        count += ranks[list(combo)].sum() >= observed - 1e-12
    return count / total


def complete_separation(better: np.ndarray, worse: np.ndarray, margin: float = 0.0,
                        higher_is_better: bool = True) -> SeparationResult:
    """Test whether the values of ``better`` (e.g. trained seeds) all beat ``worse`` (null seeds).

    ``margin`` is the minimum gap the protocol requires (0 for plain separation). With
    ``higher_is_better=False`` smaller values win (e.g. an error).
    """
    b = np.asarray(better, dtype=float)
    w = np.asarray(worse, dtype=float)
    if not higher_is_better:
        b, w = -b, -w
    gap = float(b.min() - w.max())
    return SeparationResult(separated=gap > 0 and gap >= margin, gap=gap, p_value=_exact_rank_sum_p(b, w))


def grouped_mannwhitney(values: np.ndarray, positive: np.ndarray, groups: np.ndarray,
                        reduce: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Per-column two-sided Mann-Whitney test with one value per (group, class).

    The rows of ``values`` [N, L] (images) are first averaged (or max-reduced) within each group and
    class, so a patient with many images counts once per class. Ties are corrected (asymptotic method
    with continuity correction). Returns (p-values [L], AUROC of the reduced values [L]). Columns that
    are constant over the reduced rows get p = 1 and AUROC 0.5.
    Complexity: O(N * L) for the reduction plus O(L * G log G) for the tests.
    Raises ValueError for an unknown ``reduce`` or if a class is empty.
    """
    if reduce not in ("mean", "max"):
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    positive = np.asarray(positive, dtype=bool)
    reduced = pd.DataFrame(values).groupby([positive, np.asarray(groups)], sort=True).agg(reduce)
    cls = reduced.index.get_level_values(0).to_numpy(dtype=bool)
    if cls.all() or (~cls).all():
        raise ValueError("both classes need at least one group")
    x, y = reduced.to_numpy()[cls], reduced.to_numpy()[~cls]
    with np.errstate(invalid="ignore", divide="ignore"):
        res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=True, axis=0)
    auroc = res.statistic / (len(x) * len(y))
    constant = reduced.to_numpy().min(0) == reduced.to_numpy().max(0)
    p = np.where(constant, 1.0, res.pvalue)
    auroc = np.where(constant, 0.5, auroc)
    return p, auroc
