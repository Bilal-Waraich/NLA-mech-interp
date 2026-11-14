"""Reconstruction at matched sparsity: how much of the activation a dictionary's codes retain.

Every dictionary is scored with the best *linear* decoder fitted on training tokens (ridge regression),
so dictionaries without a meaningful native decoder (random SAEs, signed basis, PCA) are compared on the
information content of their k-sparse codes rather than on decoder quality.
"""
from __future__ import annotations

import numpy as np
import torch


class RidgeAccumulator:
    """Accumulates sufficient statistics for ridge regression ``x ~ [z, 1] @ W`` over chunks.

    Memory: O(n_latents^2) for the Gram matrix (float64), independent of the number of tokens.
    """

    def __init__(self, n_latents: int, d_model: int) -> None:
        self.G = torch.zeros(n_latents + 1, n_latents + 1, dtype=torch.float64)
        self.B = torch.zeros(n_latents + 1, d_model, dtype=torch.float64)
        self.n = 0

    def add(self, z: torch.Tensor, x: torch.Tensor) -> None:
        z1 = torch.cat([z, torch.ones(z.shape[0], 1)], dim=1).double()
        self.G += z1.T @ z1
        self.B += z1.T @ x.double()
        self.n += z.shape[0]

    def solve(self, ridge: float = 1e-3) -> torch.Tensor:
        """Return W [n_latents + 1, d_model]; ridge is scaled by the mean diagonal for conditioning."""
        reg = ridge * self.G.diagonal().mean() * torch.eye(self.G.shape[0], dtype=torch.float64)
        reg[-1, -1] = 0  # do not shrink the intercept
        return torch.linalg.solve(self.G + reg, self.B).float()


class FVEAccumulator:
    """Fraction of variance explained on held-out tokens: 1 - SSE / SST (SST around the held-out mean)."""

    def __init__(self, d_model: int) -> None:
        self.sse = 0.0
        self.sum = torch.zeros(d_model, dtype=torch.float64)
        self.sumsq = 0.0
        self.n = 0

    def add(self, x: torch.Tensor, x_hat: torch.Tensor) -> None:
        self.sse += float(((x - x_hat).double() ** 2).sum())
        self.sum += x.double().sum(0)
        self.sumsq += float((x.double() ** 2).sum())
        self.n += x.shape[0]

    def fve(self) -> float:
        sst = self.sumsq - float((self.sum ** 2).sum()) / self.n
        return 1.0 - self.sse / sst


def predict(W: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.cat([z, torch.ones(z.shape[0], 1)], dim=1) @ W


def split_images(n_images: int, groups: np.ndarray | None, test_frac: float, seed: int) -> np.ndarray:
    """Boolean test mask over images; if ``groups`` is given, whole groups (patients) go to one side."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.random(n_images) < test_frac
    uniq = np.unique(groups)
    test_groups = set(uniq[rng.random(uniq.size) < test_frac])
    return np.array([g in test_groups for g in groups])
