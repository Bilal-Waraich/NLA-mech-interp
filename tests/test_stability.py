"""Known-answer tests for decoder matching."""
from __future__ import annotations

import numpy as np
import torch

from medinterp.eval.stability import best_match, cosine_matrix, one_to_one


def test_permuted_copy_matches_perfectly() -> None:
    g = torch.Generator().manual_seed(0)
    A = torch.randn(64, 32, generator=g)
    perm = torch.randperm(64, generator=g)
    B = A[perm] * 3.0                       # same directions, shuffled and rescaled
    C = cosine_matrix(A, B)
    cos, partner = one_to_one(C)
    assert np.allclose(cos, 1.0, atol=1e-5)
    assert np.array_equal(perm.numpy()[partner], np.arange(64))


def test_one_to_one_is_stricter_than_best_match() -> None:
    # Two A latents point the same way; only one of them can claim the single matching B latent.
    A = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    C = cosine_matrix(A, B)
    assert np.allclose(best_match(C)[0], [1.0, 1.0])
    assert np.allclose(sorted(one_to_one(C)[0]), [0.0, 1.0])


def test_random_dictionaries_match_poorly_in_high_dimension() -> None:
    g = torch.Generator().manual_seed(0)
    C = cosine_matrix(torch.randn(512, 1024, generator=g), torch.randn(512, 1024, generator=g))
    assert best_match(C)[0].max() < 0.25
