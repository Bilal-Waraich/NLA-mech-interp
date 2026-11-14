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
