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
