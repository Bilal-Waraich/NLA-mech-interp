"""Seed stability: do two independently trained dictionaries learn the same directions?

Each latent is represented by its unit-norm decoder row. For a pair of dictionaries A and B:

  best_match   for every latent of A, the highest cosine to any latent of B (many-to-one allowed)
  one_to_one   for every latent of A, the cosine of its partner under the optimal one-to-one assignment
               (Hungarian algorithm on the cosine matrix); stricter, since two A latents cannot share
               one B latent. Works for dictionaries of different size: rows left without a partner get
               NaN and partner -1.

Functional matching compares what latents do rather than where their decoder rows point: the Pearson
correlation of two latents' token-level codes over the same tokens (``CodeCorrelation``), matched with
the same one-to-one assignment (``functional_one_to_one``).

A latent is "stable" if it has a close partner in every other run. What counts as close is calibrated
against null dictionaries (where any match is chance), not assumed; ``TokenMoments`` provides the
covariance for the data-aware null (``dictionaries.baselines.covariance_directions``).
"""
from __future__ import annotations

from typing import Iterable

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
    """Optimal one-to-one assignment maximising the total score; returns per-row score and partner.

    ``C`` may be rectangular. When it has more rows than columns, the rows that receive no partner get
    score NaN and partner -1, so they cannot be mistaken for matches.
    Complexity: O(n^3) worst case; about a minute for 4096 x 4096 on a laptop.
    Raises ValueError if ``C`` contains NaN (drop dead latents first, see ``functional_one_to_one``).
    """
    if np.isnan(C).any():
        raise ValueError("score matrix contains NaN")
    rows, cols = linear_sum_assignment(C, maximize=True)
    score = np.full(C.shape[0], np.nan)
    partner = np.full(C.shape[0], -1, dtype=np.int64)
    score[rows], partner[rows] = C[rows, cols], cols
    return score, partner


class TokenMoments:
    """Running mean and covariance of tokens, accumulated over chunks in float64.

    Memory: O(d_model^2). Complexity: O(N_tokens * d_model^2) in total.
    """

    def __init__(self, d_model: int) -> None:
        self.sum = torch.zeros(d_model, dtype=torch.float64)
        self.outer = torch.zeros(d_model, d_model, dtype=torch.float64)
        self.n = 0

    def add(self, x: torch.Tensor) -> None:
        x = x.double()
        self.sum += x.sum(0)
        self.outer += x.T @ x
        self.n += x.shape[0]

    @property
    def mean(self) -> torch.Tensor:
        return self.sum / self.n

    def covariance(self) -> torch.Tensor:
        """Covariance around the accumulated mean (divisor n - 1)."""
        m = self.mean
        return (self.outer - self.n * torch.outer(m, m)) / (self.n - 1)


class CodeCorrelation:
    """Pearson correlation between the token-level codes of two dictionaries, accumulated over chunks.

    Call ``add(z_a, z_b)`` with the codes of the same tokens under A and B. Memory: O(L_a * L_b) for
    the cross-product (float64). A latent that never varies (dead) has an undefined correlation: its
    row or column is NaN and it is listed by ``dead_a`` / ``dead_b``.
    """

    def __init__(self, n_a: int, n_b: int) -> None:
        self.sa = torch.zeros(n_a, dtype=torch.float64)
        self.sb = torch.zeros(n_b, dtype=torch.float64)
        self.qa = torch.zeros(n_a, dtype=torch.float64)
        self.qb = torch.zeros(n_b, dtype=torch.float64)
        self.cross = torch.zeros(n_a, n_b, dtype=torch.float64)
        self.n = 0

    def add(self, z_a: torch.Tensor, z_b: torch.Tensor) -> None:
        if z_a.shape[0] != z_b.shape[0]:
            raise ValueError("z_a and z_b must hold the same tokens")
        za, zb = z_a.float(), z_b.float()
        # The chunk product runs in float32 for speed; sums across chunks are kept in float64.
        self.cross += (za.T @ zb).double()
        self.sa += za.double().sum(0)
        self.sb += zb.double().sum(0)
        self.qa += za.double().pow(2).sum(0)
        self.qb += zb.double().pow(2).sum(0)
        self.n += za.shape[0]

    def _var(self, s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (q - s * s / self.n).clamp(min=0)

    @property
    def dead_a(self) -> np.ndarray:
        return (self._var(self.sa, self.qa) <= 0).numpy()

    @property
    def dead_b(self) -> np.ndarray:
        return (self._var(self.sb, self.qb) <= 0).numpy()

    def correlation(self) -> np.ndarray:
        cov = self.cross - torch.outer(self.sa, self.sb) / self.n
        sd = torch.outer(self._var(self.sa, self.qa).sqrt(), self._var(self.sb, self.qb).sqrt())
        corr = cov / sd
        corr[sd == 0] = float("nan")
        return corr.numpy()


def functional_one_to_one(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one matching on a code-correlation matrix whose dead rows / columns are NaN.

    Dead latents are left out of the assignment; they get score NaN and partner -1, like rows without a
    partner. Returns per-row score and partner index (into the original columns).
    """
    live_r = ~np.isnan(corr).all(1)
    live_c = ~np.isnan(corr).all(0)
    sub = corr[np.ix_(live_r, live_c)]
    score = np.full(corr.shape[0], np.nan)
    partner = np.full(corr.shape[0], -1, dtype=np.int64)
    if sub.size:
        s, p = one_to_one(sub)
        cols = np.flatnonzero(live_c)
        rows = np.flatnonzero(live_r)
        score[rows] = s
        partner[rows] = np.where(p >= 0, cols[np.clip(p, 0, None)], -1)
    return score, partner


def code_correlation(enc_a, enc_b, token_chunks: Iterable[torch.Tensor]) -> CodeCorrelation:
    """Encode every chunk with both dictionaries and accumulate the correlation of their codes."""
    acc = None
    for x in token_chunks:
        za, zb = enc_a.encode(x), enc_b.encode(x)
        if acc is None:
            acc = CodeCorrelation(za.shape[1], zb.shape[1])
        acc.add(za, zb)
    if acc is None:
        raise ValueError("no tokens given")
    return acc
