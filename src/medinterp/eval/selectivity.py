"""Null-adjusted monosemanticity of a latent's top images (after Wesp et al., arXiv 2603.23794).

For latent j, take its top-``n`` images by activation and their finding sets (multi-label):
  coherence  C = mean pairwise Jaccard of finding sets, null-adjusted:
                 (C_raw - C_null) / (1 - C_null), C_null = expected C_raw for n random images
  specificity S = 1 - H(p) / log(L), p = distribution of findings pooled over the top-n images
  score M = max(C, 0) * S
Images with no positive finding get the pseudo-finding "none" so empty sets are comparable.
"""
from __future__ import annotations

import numpy as np


def _pairwise_jaccard(sets: np.ndarray, known: np.ndarray | None = None) -> float:
    """Mean pairwise Jaccard of rows of a boolean matrix [n, L].

    With ``known`` [n, L], a label enters the comparison of two images only if both have a certain
    value for it, so an uncertain label is neither a match nor a mismatch.
    """
    if known is None:
        inter = sets.astype(np.int32) @ sets.T.astype(np.int32)
        size = sets.sum(1)
        union = size[:, None] + size[None, :] - inter
    else:
        both = known[:, None, :] & known[None, :, :]
        inter = (sets[:, None, :] & sets[None, :, :] & both).sum(-1)
        union = ((sets[:, None, :] | sets[None, :, :]) & both).sum(-1)
    jac = inter / np.maximum(union, 1)
    iu = np.triu_indices(len(sets), 1)
    return float(jac[iu].mean())


def _specificity(sets: np.ndarray, known: np.ndarray | None = None) -> float:
    if known is not None:
        sets = sets & known
    counts = sets.sum(0).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(1 - (-(p * np.log(p)).sum()) / np.log(sets.shape[1]))


def monosemanticity(codes: np.ndarray, findings: np.ndarray, top_n: int = 10, n_null: int = 2000,
                    seed: int = 0, known: np.ndarray | None = None) -> np.ndarray:
    """Return array [n_latents, 3] of (C_adj, S, M). Latents firing on < top_n images get NaN.

    ``findings``: boolean [n_images, L] (already including a "none" / "normal" column).
    ``known``: optional boolean [n_images, L]; False marks an uncertain label, which is then left out of
    every comparison involving that image and label (protocol: uncertain excluded per label).
    Complexity: O(n_latents * (n_images + top_n^2 * L)).
    """
    rng = np.random.default_rng(seed)
    def rows(idx):
        return findings[idx], (None if known is None else known[idx])

    c_null = np.mean([_pairwise_jaccard(*rows(rng.choice(len(findings), top_n, replace=False)))
                      for _ in range(n_null)])
    out = np.full((codes.shape[1], 3), np.nan)
    top = np.argsort(-codes.astype(np.float32), axis=0)[:top_n]          # [top_n, n_latents]
    n_firing = (codes > 0).sum(0)
    for j in range(codes.shape[1]):
        if n_firing[j] < top_n:
            continue
        sets, kn = rows(top[:, j])
        c = (_pairwise_jaccard(sets, kn) - c_null) / (1 - c_null)
        s = _specificity(sets, kn)
        out[j] = (c, s, max(c, 0.0) * s)
    return out
