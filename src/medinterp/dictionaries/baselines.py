"""Baseline dictionaries scored under exactly the same rules as the trained SAE.

All dictionaries share one interface: ``encode(x[B, d_model]) -> codes[B, n_latents]`` with at most ``k``
non-negative entries per row (per-token top-k), so density is matched by construction:

  random_sae(seed)  untrained BatchTopK SAE with the thesis initialisation (kaiming-uniform W_enc,
                    W_dec = row-normalised W_enc^T * 0.1, W_enc = W_dec^T, zero biases), explicitly seeded.
  signed_pca        top principal directions; each direction contributes two latents (+v and -v), so a
                    latent fires on one side only. (The thesis null used |projection|, which mixes both
                    sides and makes centroid tests meaningless.)
  signed_basis      the 1024 coordinate axes of the stage output, +/-: a crude "neuron" proxy. Note this
                    hook is a residual-stream stage output, not a privileged basis; true neurons (MLP
                    hidden units) need a separate harvest.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .sae import TopKSAE


@dataclass
class LinearTopK:
    """Encoder ``relu(x @ W) `` followed by per-token top-k; ``W`` is ``[d_model, n_latents]``."""

    W: torch.Tensor
    k: int
    name: str

    @property
    def d_sae(self) -> int:
        return self.W.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(x @ self.W)
        top = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)


def random_sae(d_model: int, d_sae: int, k: int, seed: int) -> TopKSAE:
    """Untrained SAE with the thesis initialisation, reproducible from ``seed``."""
    gen = torch.Generator().manual_seed(seed)
    W_enc = torch.empty(d_model, d_sae)
    # kaiming_uniform_ with default a=0 on a [d_model, d_sae] tensor: fan_in = d_sae (dim 1).
    bound = (6.0 / d_sae) ** 0.5
    W_enc.uniform_(-bound, bound, generator=gen)
    W_dec = W_enc.T.clone()
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8) * 0.1
    W_enc = W_dec.T.clone().contiguous()
    return TopKSAE(W_enc, torch.zeros(d_sae), W_dec, torch.zeros(d_model), k)


def signed_pca(sample: torch.Tensor, n_components: int, k: int) -> LinearTopK:
    """PCA on ``sample`` [N, d_model] (centred internally); latents are +/- each component.

    Complexity: O(N * d_model^2) for the SVD.
    """
    centred = sample - sample.mean(0)
    _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
    comps = Vt[:n_components]                                   # [n, d_model], unit rows
    return LinearTopK(torch.cat([comps, -comps]).T.contiguous(), k, "signed_pca")


def signed_basis(d_model: int, k: int) -> LinearTopK:
    """Coordinate axes +/-e_i as 2*d_model latents."""
    eye = torch.eye(d_model)
    return LinearTopK(torch.cat([eye, -eye]).T.contiguous(), k, "signed_basis")
