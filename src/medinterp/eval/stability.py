"""Seed stability: do two independently trained dictionaries learn the same directions?

Each latent is represented by its unit-norm decoder row. For a pair of dictionaries A and B:

  best_match   for every latent of A, the highest cosine to any latent of B (many-to-one allowed)
  one_to_one   for every latent of A, the cosine of its partner under the optimal one-to-one assignment
               (Hungarian algorithm on the cosine matrix); stricter, since two A latents cannot share
               one B latent

A latent is "stable" if it has a close partner in every other run. What counts as close is calibrated
against random dictionaries (where any match is chance), not assumed.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def unit_rows(W: torch.Tensor) -> torch.Tensor:
    """Normalise rows to unit length (decoder rows are already ~unit, this guards against drift)."""
    W = W.float()
    return W / W.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def cosine_matrix(A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
    """Cosine similarity of every row of A [n_a, d] with every row of B [n_b, d] -> [n_a, n_b]."""
    return (unit_rows(A) @ unit_rows(B).T).numpy()


def best_match(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of the cosine matrix: highest cosine and the index of the matching column."""
    return C.max(1), C.argmax(1)


def one_to_one(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one assignment maximising total cosine; returns per-row cosine and partner index.

    Complexity: O(n^3) worst case; about a minute for 4096 x 4096 on a laptop.
    """
    rows, cols = linear_sum_assignment(C, maximize=True)
    cos = np.empty(C.shape[0])
    partner = np.empty(C.shape[0], dtype=np.int64)
    cos[rows], partner[rows] = C[rows, cols], cols
    return cos, partner
