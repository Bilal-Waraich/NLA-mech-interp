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


# ------------------------------------------------- robustness checks (protocol section 5, 2026-11-24)


def principal_overlap(rows_a: torch.Tensor, rows_b: torch.Tensor, ranks: tuple[int, ...]) -> dict[int, float]:
    """For each r: (1/r) * ||V_a^r V_b^rT||_F^2, the mean squared cosine of the principal angles between the
    top-r right singular subspaces of the unit-normalised rows of ``rows_a`` and ``rows_b``.

    Symmetric in a and b and, unlike ``subspace_explained_variance``, independent of how many rows the target
    has. 1 for identical subspaces; about r / d_model for random ones. Ranks above either set's size are left
    out. Raises ValueError for an empty set.
    """
    if rows_a.shape[0] == 0 or rows_b.shape[0] == 0:
        raise ValueError("empty set")
    _, _, va = torch.linalg.svd(_unit_rows(rows_a.double()), full_matrices=False)
    _, _, vb = torch.linalg.svd(_unit_rows(rows_b.double()), full_matrices=False)
    return {r: float(((va[:r] @ vb[:r].T) ** 2).sum()) / r
            for r in ranks if r <= min(va.shape[0], vb.shape[0])}


def matched_subset(pool: np.ndarray, pool_rates: np.ndarray, target_rates: np.ndarray,
                   rng: np.random.Generator, n_bins: int = 10) -> np.ndarray:
    """A random subset of ``pool`` (latent indices) with as many members as ``target_rates`` and the same
    distribution over firing-rate bins, drawn without replacement.

    Bins are the ``n_bins`` quantile bins of ``pool_rates``. Each target latent asks for one pool latent from
    its bin; a bin with too few latents is filled from the nearest bins (lower first on ties), so the subset
    always has the target's size. Raises ValueError if the pool is smaller than the target.
    Complexity: O(len(pool) log len(pool)).
    """
    if len(target_rates) > len(pool):
        raise ValueError(f"pool of {len(pool)} is smaller than the target of {len(target_rates)}")
    edges = np.quantile(pool_rates, np.linspace(0, 1, n_bins + 1)[1:-1])
    pool_bin = np.searchsorted(edges, pool_rates, side="right")
    want = np.bincount(np.searchsorted(edges, target_rates, side="right"), minlength=n_bins)
    free = {b: list(rng.permutation(pool[pool_bin == b])) for b in range(n_bins)}
    chosen = []
    for b in range(n_bins):
        for _ in range(want[b]):
            # nearest non-empty bin, preferring the lower one on ties
            for dist in range(n_bins):
                hit = next((c for c in (b - dist, b + dist) if 0 <= c < n_bins and free[c]), None)
                if hit is not None:
                    chosen.append(free[hit].pop())
                    break
    return np.array(chosen, dtype=int)


def project_out(rows: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """``rows`` with the span of the orthonormal ``directions`` (``[n_dir, d_model]``) removed, renormalised.

    Rows that lie entirely in that span become zero rows (their norm is clamped, not divided by zero).
    """
    r = rows.double()
    d = directions.double()
    return _unit_rows(r - (r @ d.T) @ d)


def whitening_map(cov: torch.Tensor, variance_share: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(W, P)``: W = V diag(lam^-1/2) V^T over the leading eigenvectors of ``cov`` that hold
    ``variance_share`` of its variance, and P = V V^T, the projector onto that subspace (the covariance of an
    isotropic null there). Rows are whitened as ``rows @ W`` (W is symmetric).

    Raises ValueError for a non-square ``cov`` or a share outside (0, 1].
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    if not 0 < variance_share <= 1:
        raise ValueError("variance_share must be in (0, 1]")
    lam, vecs = torch.linalg.eigh(cov.double())
    lam, vecs = lam.flip(0).clamp(min=0), vecs.flip(1)
    q = int(np.searchsorted((lam.cumsum(0) / lam.sum()).numpy(), variance_share - 1e-12) + 1)
    v = vecs[:, :q]
    return (v / lam[:q].sqrt()) @ v.T, v @ v.T


def right_singular_basis(rows: torch.Tensor) -> torch.Tensor:
    """Right singular vectors (as rows, by decreasing singular value) of the unit-normalised ``rows``; computed
    once and reused by ``explained_variance_in`` and ``overlap_of`` when one basis meets many targets."""
    return torch.linalg.svd(_unit_rows(rows.double()), full_matrices=False)[2]


def explained_variance_in(basis: torch.Tensor, target_rows: torch.Tensor, ranks: tuple[int, ...]) -> dict[int, float]:
    """``subspace_explained_variance`` with a precomputed ``right_singular_basis``."""
    target = _unit_rows(target_rows.double())
    total = float((target ** 2).sum())
    return {r: float(((target @ basis[:r].T) ** 2).sum()) / total for r in ranks if r <= basis.shape[0]}


def overlap_of(basis_a: torch.Tensor, basis_b: torch.Tensor, ranks: tuple[int, ...]) -> dict[int, float]:
    """``principal_overlap`` with two precomputed ``right_singular_basis`` results (or any orthonormal rows)."""
    return {r: float(((basis_a[:r] @ basis_b[:r].T) ** 2).sum()) / r
            for r in ranks if r <= min(basis_a.shape[0], basis_b.shape[0])}
