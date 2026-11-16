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
  frozen_decoder_sae  random decoder that is never trained, encoder trained on the data (Korznikov et al.);
                    decoder either isotropic or drawn from the activation covariance
  covariance_directions  random unit directions drawn from N(0, cov), the data-aware matching null
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

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Own decoder: the latents' directions are the columns of ``W`` (unit for PCA and basis)."""
        return z @ self.W.T


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


def isotropic_directions(n_latents: int, d_model: int, seed: int) -> torch.Tensor:
    """``[n_latents, d_model]`` unit rows drawn uniformly on the sphere."""
    gen = torch.Generator().manual_seed(seed)
    D = torch.randn(n_latents, d_model, generator=gen)
    return D / D.norm(dim=-1, keepdim=True)


def covariance_directions(cov: torch.Tensor, n_latents: int, seed: int) -> torch.Tensor:
    """``[n_latents, d_model]`` unit rows drawn from N(0, cov): random directions shaped like the data.

    This is the data-aware null of the protocol: such directions concentrate in the high-variance
    subspace, as trained decoder rows could if they merely followed variance.
    Complexity: O(d_model^3) for the eigendecomposition plus O(n_latents * d_model^2).
    Raises ValueError if ``cov`` is not square.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got {tuple(cov.shape)}")
    evals, evecs = torch.linalg.eigh(cov.double())
    root = evecs * evals.clamp(min=0).sqrt()             # cov = root @ root.T
    gen = torch.Generator().manual_seed(seed)
    g = torch.randn(n_latents, cov.shape[0], generator=gen, dtype=torch.float64)
    D = (g @ root.T).float()
    return D / D.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def frozen_decoder_sae(data, cfg, decoder: str = "isotropic", cov: torch.Tensor | None = None,
                       decoder_seed: int = 0, **train_kwargs) -> TopKSAE:
    """Frozen random decoder with a trained encoder (after Korznikov et al.).

    ``decoder`` is "isotropic" (uniform on the sphere) or "covariance" (drawn from ``cov``). The decoder
    seed is separate from the training seeds in ``cfg`` so the same decoder can be paired with several
    encoder runs. Other keyword arguments go to ``train_batchtopk`` (device, log, train_rows, ...).
    Raises ValueError for an unknown ``decoder`` or a covariance decoder without ``cov``.
    """
    from .train import train_batchtopk

    if decoder == "isotropic":
        W = isotropic_directions(cfg.d_sae, cfg.d_model, decoder_seed)
    elif decoder == "covariance":
        if cov is None:
            raise ValueError("decoder='covariance' needs cov")
        W = covariance_directions(cov, cfg.d_sae, decoder_seed)
    else:
        raise ValueError(f"unknown decoder {decoder!r}")
    return train_batchtopk(data, cfg, fixed_decoder=W, **train_kwargs)
