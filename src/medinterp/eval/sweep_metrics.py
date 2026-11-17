"""Per-dictionary and per-pair measurements of the seed-overlap sweep (protocol sections 4-5).

All measurements on tokens use one fixed token set held in memory (the val split of the development
harvest), so every dictionary of the sweep is scored on exactly the same tokens.

  EvalTokens          tokens of the chosen images, with their image index and group
  score_dictionary    own-decoder FVE with a cluster-bootstrap interval, dead latents, latents active on
                      every image
  decoder_overlap     one-to-one decoder cosine between two dictionaries
  covariance_null     the data-aware null of decoder matching: an anchor against random directions drawn
                      from the token covariance
  functional_overlap  one-to-one correlation of token-level sparse codes
  functional_null     the same against a top-k dictionary of covariance-shaped random directions (encoder
                      = its directions), its own null because code correlations have their own scale
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..dictionaries.baselines import LinearTopK, covariance_directions
from .recon import PerImageRecon
from .stability import CodeCorrelation, cosine_matrix, functional_one_to_one, one_to_one
from .stats import cluster_bootstrap


@dataclass
class EvalTokens:
    x: torch.Tensor            # [n_images * n_patches, d_model] float32, image-aligned
    groups: np.ndarray         # [n_images] group id per image (for the cluster bootstrap)
    n_patches: int

    @property
    def n_images(self) -> int:
        return self.x.shape[0] // self.n_patches

    def chunks(self, images_per_chunk: int = 512):
        step = images_per_chunk * self.n_patches
        for lo in range(0, self.x.shape[0], step):
            yield self.x[lo:lo + step]


def load_eval_tokens(acts, masks: dict[str, dict[str, np.ndarray]], groups: dict[str, np.ndarray],
                     split: str) -> EvalTokens:
    """All tokens of the images in ``split``, every dataset, in dataset and image order.

    Memory: n_images * n_patches * d_model * 4 bytes (about 0.55 GB for the val split of the thesis harvest).
    """
    parts, grp = [], []
    P = acts.n_patches
    for name in acts.blocks:
        keep = masks[name][split]
        for img0, x in acts.iter_token_chunks(name, 1024):
            n_img = x.shape[0] // P
            sel = np.flatnonzero(keep[img0:img0 + n_img])
            if sel.size:
                parts.append(x.view(n_img, P, -1)[torch.from_numpy(sel)].reshape(-1, x.shape[1]).clone())
        grp.append(groups[name][keep])
    return EvalTokens(torch.cat(parts), np.concatenate(grp), P)


@dataclass
class DictionaryScore:
    fve: float
    fve_ci: tuple[float, float]
    fve_samples: np.ndarray     # bootstrap samples, shared resamples across dictionaries (same seed)
    n_dead: int                 # latents that are zero on every token
    n_every_image: int          # latents that are non-zero on at least one token of every image
    firing_rate: np.ndarray     # share of tokens on which each latent is non-zero


def score_dictionary(enc, tokens: EvalTokens, n_resamples: int, confidence: float, seed: int,
                     device: str = "cpu", images_per_chunk: int = 256) -> DictionaryScore:
    """One pass over the tokens: own-decoder FVE (per-token top-k), its bootstrap interval over groups,
    and firing statistics. ``enc`` needs ``encode``, ``decode`` and ``d_sae`` (TopKSAE or LinearTopK).

    Complexity: O(N_tokens * d_model * d_sae) for the encode plus O(n_resamples * n_images * d_model).
    """
    recon = PerImageRecon()
    fires = torch.zeros(enc.d_sae, dtype=torch.float64)
    on_image = torch.zeros(enc.d_sae, dtype=torch.int64)
    P = tokens.n_patches
    moved = _to_device(enc, device)
    for x in tokens.chunks(images_per_chunk):
        xd = x.to(device)
        z = moved.encode(xd)
        recon.add(x, moved.decode(z).float().cpu(), P)
        nz = (z != 0).cpu()
        fires += nz.sum(0).double()
        on_image += nz.view(-1, P, nz.shape[1]).any(1).sum(0)
    boot = cluster_bootstrap(recon.fve, tokens.groups, n_resamples, confidence, seed)
    return DictionaryScore(boot.estimate, (boot.ci_low, boot.ci_high), boot.samples,
                           int((fires == 0).sum()), int((on_image == tokens.n_images).sum()),
                           (fires / tokens.x.shape[0]).numpy())


def _to_device(enc, device: str):
    if device == "cpu":
        return enc
    if isinstance(enc, LinearTopK):
        return LinearTopK(enc.W.to(device), enc.k, enc.name)
    return type(enc)(enc.W_enc.to(device), enc.b_enc.to(device), enc.W_dec.to(device), enc.b_dec.to(device),
                     enc.k)


def decoder_overlap(dec_a: torch.Tensor, dec_b: torch.Tensor) -> np.ndarray:
    """Per row of ``dec_a``: cosine to its one-to-one partner in ``dec_b`` (NaN without a partner)."""
    return one_to_one(cosine_matrix(dec_a, dec_b))[0]


def covariance_null(dec_anchor: torch.Tensor, cov: torch.Tensor, seeds: tuple[int, ...]) -> np.ndarray:
    """Pooled one-to-one cosines of the anchor against covariance-shaped random dictionaries of its size."""
    n = dec_anchor.shape[0]
    return np.concatenate([decoder_overlap(dec_anchor, covariance_directions(cov, n, seed=s)) for s in seeds])


def functional_overlap(enc_a, enc_b, tokens: EvalTokens, device: str = "cpu",
                       images_per_chunk: int = 256) -> tuple[np.ndarray, int, int]:
    """One-to-one code correlation of ``enc_a``'s latents with ``enc_b``'s on the eval tokens.

    Returns (score per latent of A, NaN for dead or unmatched latents; n dead in A; n dead in B).
    Memory: O(d_sae_a * d_sae_b) float64.
    """
    a, b = _to_device(enc_a, device), _to_device(enc_b, device)
    acc = None
    for x in tokens.chunks(images_per_chunk):
        xd = x.to(device)
        za, zb = a.encode(xd).cpu(), b.encode(xd).cpu()
        if acc is None:
            acc = CodeCorrelation(za.shape[1], zb.shape[1])
        acc.add(za, zb)
    scores, _ = functional_one_to_one(acc.correlation())
    return scores, int(acc.dead_a.sum()), int(acc.dead_b.sum())


def covariance_topk(cov: torch.Tensor, n_latents: int, k: int, seed: int) -> LinearTopK:
    """Top-k dictionary whose latents are covariance-shaped random directions (encoder = directions)."""
    return LinearTopK(covariance_directions(cov, n_latents, seed).T.contiguous(), k, f"covariance_topk_{seed}")


def functional_null(enc_anchor, cov: torch.Tensor, k: int, seeds: tuple[int, ...], tokens: EvalTokens,
                    device: str = "cpu") -> np.ndarray:
    """Pooled one-to-one code correlations of the anchor against covariance top-k dictionaries."""
    n = enc_anchor.d_sae
    return np.concatenate([functional_overlap(enc_anchor, covariance_topk(cov, n, k, s), tokens, device)[0]
                           for s in seeds])
