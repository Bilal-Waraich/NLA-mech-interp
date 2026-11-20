"""Subspace reproducibility across seeds (protocol section 5), after Gerasimov et al. 2026.

A latent's reappearance probability is the share of the other seeds that contain a decoder direction within
the matching threshold of it (best match, many-to-one, as in the paper). Latents that reappear in no other
seed form the unstable set, latents that reappear in all of them the stable set. The question is whether
the *space* spanned by one seed's unstable decoder directions explains another seed's unstable directions,
even though the directions themselves do not match.

  reappearance_probability   per anchor latent, the share of other seeds with a match
  stability_sets             indices of the unstable and stable latents
  effective_rank             exp of the entropy of the normalised singular values
  subspace_explained_variance  share of a set's squared norm inside another set's top-r subspace
"""
from __future__ import annotations

import numpy as np
import torch


def _unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def reappearance_probability(anchor: torch.Tensor, others: list[torch.Tensor], threshold: float,
                             rows_per_chunk: int = 2048) -> np.ndarray:
    """Per row of ``anchor`` (``[n_latents, d_model]`` decoder): the share of ``others`` holding a row whose
    cosine with it reaches ``threshold``. Raises ValueError without other seeds.

    Complexity: O(len(others) * n_latents^2 * d_model).
    """
    if not others:
        raise ValueError("need at least one other seed")
    a = _unit_rows(anchor.double())
    hits = np.zeros(anchor.shape[0])
    for other in others:
        b = _unit_rows(other.double())
        best = torch.cat([(chunk @ b.T).max(1).values for chunk in a.split(rows_per_chunk)])
        hits += (best >= threshold).numpy()
    return hits / len(others)


def stability_sets(probability: np.ndarray, epsilon: float, eligible: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Indices of unstable (probability <= epsilon) and stable (probability >= 1 - epsilon) latents among
    ``eligible`` (a boolean mask, e.g. the latents that fire on the eval tokens; default all)."""
    eligible = np.ones(len(probability), bool) if eligible is None else eligible
    return np.flatnonzero(eligible & (probability <= epsilon)), np.flatnonzero(eligible & (probability >= 1 - epsilon))


def effective_rank(rows: torch.Tensor) -> float:
    """exp(H(p)) with p the singular values of the unit-normalised ``rows`` divided by their sum.

    Raises ValueError for an empty set. Complexity: one SVD, O(min(m, d)^2 * max(m, d)).
    """
    if rows.shape[0] == 0:
        raise ValueError("empty set")
    s = torch.linalg.svdvals(_unit_rows(rows.double()))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def subspace_explained_variance(basis_rows: torch.Tensor, target_rows: torch.Tensor, ranks: tuple[int, ...]
                                ) -> dict[int, float]:
    """For each r in ``ranks`` (up to the rank of ``basis_rows``): the share of the squared Frobenius norm of
    the unit-normalised ``target_rows`` that lies in the span of the top-r right singular vectors of the
    unit-normalised ``basis_rows``. Both are ``[n, d_model]``. Ranks above the basis' size are left out.

    Raises ValueError for an empty basis or target.
    """
    if basis_rows.shape[0] == 0 or target_rows.shape[0] == 0:
        raise ValueError("empty basis or target")
    _, _, vh = torch.linalg.svd(_unit_rows(basis_rows.double()), full_matrices=False)
    target = _unit_rows(target_rows.double())
    total = float((target ** 2).sum())
    out = {}
    for r in ranks:
        if r > vh.shape[0]:
            continue
        out[r] = float(((target @ vh[:r].T) ** 2).sum()) / total
    return out


def principal_subspace_explained_variance(cov: torch.Tensor, target_rows: torch.Tensor, ranks: tuple[int, ...]
                                          ) -> dict[int, float]:
    """For each r in ``ranks``: the share of the squared norm of the unit-normalised ``target_rows`` inside the
    span of the top r eigenvectors of ``cov`` (the data's principal subspace; a basis null).

    Raises ValueError for an empty target or a non-square ``cov``.
    """
    if target_rows.shape[0] == 0:
        raise ValueError("empty target")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    _, vecs = torch.linalg.eigh(cov.double())
    top = vecs.flip(1).T                                  # rows ordered by decreasing variance
    target = _unit_rows(target_rows.double())
    total = float((target ** 2).sum())
    return {r: float(((target @ top[:r].T) ** 2).sum()) / total for r in ranks if r <= top.shape[0]}
