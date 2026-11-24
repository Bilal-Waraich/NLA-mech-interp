"""Known-answer tests for subspace reproducibility (protocol section 5)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medinterp.eval.subspace import (effective_rank, reappearance_probability, stability_sets,
                                     subspace_explained_variance)


def test_reappearance_counts_seeds_with_a_close_direction():
    anchor = torch.eye(3)
    seed_a = torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, -1.0]])   # only latent 0 matches
    seed_b = torch.tensor([[2.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.0, 1.0, 0.0]])    # latents 0 and 1 match
    p = reappearance_probability(anchor, [seed_a, seed_b], threshold=0.7)
    assert p.tolist() == pytest.approx([1.0, 0.5, 0.0])
    unstable, stable = stability_sets(p, epsilon=0.05)
    assert unstable.tolist() == [2] and stable.tolist() == [0]
    unstable, stable = stability_sets(p, epsilon=0.05, eligible=np.array([False, True, True]))
    assert stable.tolist() == []
    with pytest.raises(ValueError):
        reappearance_probability(anchor, [], 0.7)


def test_effective_rank_known_answers():
    assert effective_rank(torch.eye(4)) == pytest.approx(4.0)
    assert effective_rank(torch.ones(5, 3)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        effective_rank(torch.zeros(0, 3))


def test_explained_variance_of_rotated_directions_in_a_shared_plane():
    """Directions that differ but share a plane are fully explained at rank 2 and not at rank 1."""
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotated = torch.tensor([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0]])
    ev = subspace_explained_variance(basis, rotated, ranks=(1, 2, 3))
    assert ev[2] == pytest.approx(1.0) and ev[1] < 1.0 and 3 not in ev
    outside = subspace_explained_variance(basis, torch.tensor([[0.0, 0.0, 1.0]]), ranks=(2,))
    assert outside[2] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        subspace_explained_variance(basis[:0], rotated, (1,))


def test_principal_subspace_explained_variance():
    from medinterp.eval.subspace import principal_subspace_explained_variance

    cov = torch.diag(torch.tensor([10.0, 5.0, 1.0]))
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ev = principal_subspace_explained_variance(cov, target, ranks=(1, 2, 3))
    assert ev[1] == pytest.approx(0.5) and ev[2] == pytest.approx(0.5) and ev[3] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        principal_subspace_explained_variance(cov, target[:0], (1,))


def test_principal_overlap_known_answers():
    from medinterp.eval.subspace import principal_overlap
    e = torch.eye(6)
    assert principal_overlap(e[:3], e[:3] * 2, (1, 2, 3))[3] == pytest.approx(1.0)
    assert principal_overlap(e[:3], e[3:], (3,))[3] == pytest.approx(0.0)
    half = principal_overlap(e[:2], torch.stack([e[0], e[2]]), (2,))[2]
    assert half == pytest.approx(0.5)
    with pytest.raises(ValueError):
        principal_overlap(e[:0], e[:2], (1,))


def test_matched_subset_matches_rate_bins_and_size():
    from medinterp.eval.subspace import matched_subset
    rng = np.random.default_rng(0)
    pool = np.arange(100)
    rates = np.linspace(0, 1, 100)
    target = rates[90:]                      # ten latents, all in the top decile
    sub = matched_subset(pool, rates, target, rng)
    assert len(sub) == 10 and len(set(sub)) == 10 and set(sub) <= set(range(90, 100))
    # 15 targets in the top decile: 10 from it, the rest from the nearest bin below
    sub = matched_subset(pool, rates, np.full(15, 0.99), rng)
    assert len(set(sub)) == 15 and set(sub) <= set(range(80, 100))
    with pytest.raises(ValueError):
        matched_subset(pool[:3], rates[:3], rates[:5], rng)


def test_project_out_and_whitening():
    from medinterp.eval.subspace import project_out, whitening_map
    rows = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    out = project_out(rows, torch.tensor([[1.0, 0.0, 0.0]]))
    assert out[0].tolist() == pytest.approx([0.0, 1.0, 0.0]) and out[1].tolist() == pytest.approx([0.0, 0.0, 1.0])
    cov = torch.diag(torch.tensor([4.0, 1.0, 1e-9]))
    w, p = whitening_map(cov, 0.99)
    assert torch.diag(w)[:2].tolist() == pytest.approx([0.5, 1.0]) and float(w[2, 2]) == pytest.approx(0.0)
    assert torch.diag(p).tolist() == pytest.approx([1.0, 1.0, 0.0])
    # whitened data has identity covariance on the kept subspace
    assert torch.allclose((w @ cov.double() @ w)[:2, :2], torch.eye(2, dtype=torch.float64))


def test_precomputed_basis_helpers_agree_with_the_direct_functions():
    from medinterp.eval.subspace import (explained_variance_in, overlap_of, principal_overlap,
                                         right_singular_basis)
    gen = torch.Generator().manual_seed(0)
    a, b = torch.randn(40, 16, generator=gen), torch.randn(30, 16, generator=gen)
    ranks = (2, 4, 8)
    assert explained_variance_in(right_singular_basis(a), b, ranks) == pytest.approx(subspace_explained_variance(a, b, ranks))
    assert overlap_of(right_singular_basis(a), right_singular_basis(b), ranks) == pytest.approx(principal_overlap(a, b, ranks))
