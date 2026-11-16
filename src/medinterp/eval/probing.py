"""Sparse probing with cross-dataset transfer (image level).

For a source dataset: rank latents by |AUROC - 0.5| on its training split, keep the top ``k``, fit a
logistic regression on those ``k`` features, then report AUROC on the source test split and on every
other dataset. Selecting on the source only means transfer is a genuine out-of-dataset test.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc_columns


def fit_eval(X_tr: np.ndarray, y_tr: np.ndarray, evals: dict[str, tuple[np.ndarray, np.ndarray]]
             ) -> dict[str, float]:
    """Fit a balanced, standardised logistic regression and return AUROC on each eval set."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X_tr, y_tr)
    return {name: float(roc_auc_score(y, clf.decision_function(X))) for name, (X, y) in evals.items()}


def sparse_probe_transfer(codes: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                          test_mask: dict[str, np.ndarray], ks: tuple[int, ...]) -> list[dict]:
    """Return rows {source, k, target, auroc}; target == source means the held-out source split.

    ``labels[d]`` is a boolean positive mask, NaN-free; images without a usable label are excluded by the
    caller.
    """
    rows = []
    for src in codes:
        tr = ~test_mask[src]
        Xs, ys = codes[src].astype(np.float32), labels[src]
        effect = np.abs(auroc_columns(Xs[tr], ys[tr]) - 0.5)
        order = np.argsort(-effect)
        for k in ks:
            cols = order[:k]
            evals = {src: (Xs[~tr][:, cols], ys[~tr])}
            evals.update({t: (codes[t][:, cols].astype(np.float32), labels[t]) for t in codes if t != src})
            for tgt, auc in fit_eval(Xs[tr][:, cols], ys[tr], evals).items():
                rows.append({"source": src, "k": k, "target": tgt, "auroc": auc})
    return rows


# ---------------------------------------------------------------- protocol v1: tuned probes

# The default C of LogisticRegression is part of the grid, so the tuned probe can never be worse on val.
DEFAULT_C = 1.0
C_GRID = tuple(float(c) for c in np.logspace(-4, 2, 7))
assert DEFAULT_C in C_GRID


def pool_tokens(values: np.ndarray, pooling: str) -> np.ndarray:
    """Reduce ``[n_images, n_patches, n_features]`` to ``[n_images, n_features]`` by max or mean.

    The same pooling is applied to dictionary codes and to the dense reference, so both sides see the
    image through the same reduction. Raises ValueError for an unknown pooling.
    """
    if pooling == "max":
        return values.max(axis=1)
    if pooling == "mean":
        return values.mean(axis=1)
    raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}")


def _probe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def fit_eval_tuned(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                   evals: dict[str, tuple[np.ndarray, np.ndarray]], Cs: tuple[float, ...] = C_GRID,
                   scores_out: dict | None = None) -> tuple[dict[str, float], float, dict[float, float]]:
    """Fit one probe per C on train, keep the C with the best val AUROC, report AUROC on ``evals``.

    Returns (AUROC per eval set, chosen C, val AUROC per C). Ties go to the larger C (weaker
    regularisation, the sklearn default side). If ``scores_out`` is given, the chosen probe's decision
    scores on every eval set are stored in it by name. Complexity: len(Cs) logistic-regression fits.
    """
    val_auc, models = {}, {}
    for C in Cs:
        clf = _probe(C).fit(X_tr, y_tr)
        val_auc[C] = float(roc_auc_score(y_val, clf.decision_function(X_val)))
        models[C] = clf
    best = max(Cs, key=lambda c: (val_auc[c], c))
    clf = models[best]
    out = {}
    for name, (X, y) in evals.items():
        scores = clf.decision_function(X)
        out[name] = float(roc_auc_score(y, scores))
        if scores_out is not None:
            scores_out[name] = scores
    return out, best, val_auc


def sparse_probe_transfer_tuned(codes: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                                splits: dict[str, dict[str, np.ndarray]], ms: tuple[int, ...],
                                eval_split: str = "val", Cs: tuple[float, ...] = C_GRID,
                                keep_scores: bool = False, rank_all: bool = False) -> list[dict]:
    """Probes on the m latents ranked best on the source's train split, regularisation tuned on its val
    split, scored on ``eval_split`` of the source and of every other dataset.

    ``codes[d]`` are image-level features; ``labels[d]`` is 1 / 0 / NaN (NaN images are left out);
    ``splits[d]`` maps "train" / "val" / "test" to boolean masks. Returns rows
    {source, m, target, auroc, C}; target == source is the source's own ``eval_split``. With
    ``keep_scores`` each row also carries ``scores`` (decision values on the target's evaluated images,
    in row order) for bootstrapping. ``rank_all`` skips the latent ranking (dense reference: m = all).
    """
    rows = []
    usable = {d: ~np.isnan(labels[d]) for d in codes}
    part = {d: {s: usable[d] & splits[d][s] for s in ("train", "val", eval_split)} for d in codes}
    for src in codes:
        tr, va = part[src]["train"], part[src]["val"]
        if tr.sum() == 0 or len(np.unique(labels[src][tr])) < 2:
            continue
        Xs = codes[src].astype(np.float32)
        ys = labels[src] == 1
        order = (np.arange(Xs.shape[1]) if rank_all
                 else np.argsort(-np.abs(auroc_columns(Xs[tr], ys[tr]) - 0.5), kind="stable"))
        for m in ms:
            cols = order[:m]
            evals = {}
            for tgt in codes:
                ev = part[tgt][eval_split]
                if ev.sum() and len(np.unique(labels[tgt][ev])) == 2:
                    evals[tgt] = (codes[tgt][ev][:, cols].astype(np.float32), labels[tgt][ev] == 1)
            scores: dict = {}
            aucs, C, _ = fit_eval_tuned(Xs[tr][:, cols], ys[tr], Xs[va][:, cols], ys[va], evals, Cs,
                                        scores_out=scores if keep_scores else None)
            for t, a in aucs.items():
                row = {"source": src, "m": m, "target": t, "auroc": a, "C": C}
                if keep_scores:
                    row["scores"] = scores[t]
                rows.append(row)
    return rows


def eval_images(labels: dict[str, np.ndarray], splits: dict[str, dict[str, np.ndarray]], eval_split: str
                ) -> dict[str, np.ndarray]:
    """Boolean mask per dataset of the images a transfer row is scored on (labelled, in ``eval_split``)."""
    return {d: ~np.isnan(labels[d]) & splits[d][eval_split] for d in labels}


def mean_cross_auroc_statistic(rows: list[dict], m: int, labels: dict[str, np.ndarray],
                               eval_masks: dict[str, np.ndarray]):
    """Build ``statistic(idx)`` for the cluster bootstrap of the mean cross-dataset AUROC at ``m``.

    The bootstrap rows are the concatenation (in dataset order of ``labels``) of every dataset's
    evaluated images; ``idx`` indexes that concatenation. For each source != target pair with ``m``
    latents, AUROC is recomputed on the target's resampled images; the statistic is their mean.
    Rows must come from ``sparse_probe_transfer_tuned(..., keep_scores=True)``.
    Raises ValueError if no cross-dataset row exists for ``m``.
    """
    names = list(labels)
    sizes = [int(eval_masks[d].sum()) for d in names]
    offsets = dict(zip(names, np.concatenate([[0], np.cumsum(sizes)])[:-1]))
    pairs = [(r["target"], r["scores"]) for r in rows if r["m"] == m and r["target"] != r["source"]]
    if not pairs:
        raise ValueError(f"no cross-dataset rows for m = {m}")
    y = {d: labels[d][eval_masks[d]] == 1 for d in names}
    bounds = {d: (offsets[d], offsets[d] + n) for d, n in zip(names, sizes)}

    def statistic(idx: np.ndarray) -> float:
        aucs = []
        for tgt, sc in pairs:
            lo, hi = bounds[tgt]
            local = idx[(idx >= lo) & (idx < hi)] - lo
            yy = y[tgt][local]
            aucs.append(roc_auc_score(yy, sc[local]) if 0 < yy.sum() < yy.size else np.nan)
        return float(np.nanmean(aucs))
    return statistic
