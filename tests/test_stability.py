"""Known-answer tests for decoder and functional matching, and the data-aware null on the archive."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medinterp.dictionaries.baselines import covariance_directions, frozen_decoder_sae, isotropic_directions
from medinterp.dictionaries.train import TrainConfig
from medinterp.eval.stability import (CodeCorrelation, TokenMoments, best_match, code_correlation,
                                      cosine_matrix, functional_one_to_one, one_to_one)


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


def test_one_to_one_rectangular_more_rows() -> None:
    """Rows without a partner are NaN with partner -1; matched rows are the optimal ones."""
    C = np.array([[0.9, 0.1], [0.2, 0.8], [0.95, 0.3], [0.0, 0.0]])
    score, partner = one_to_one(C)
    # Optimum: row 2 -> col 0 (0.95) and row 1 -> col 1 (0.8), total 1.75.
    assert partner.tolist() == [-1, 1, 0, -1]
    assert np.isnan(score[[0, 3]]).all()
    assert np.allclose(score[[1, 2]], [0.8, 0.95])


def test_one_to_one_rectangular_more_columns() -> None:
    C = np.array([[0.1, 0.9, 0.3], [0.7, 0.8, 0.2]])
    score, partner = one_to_one(C)
    assert partner.tolist() == [1, 0] and np.allclose(score, [0.9, 0.7])


def test_one_to_one_rejects_nan() -> None:
    with pytest.raises(ValueError):
        one_to_one(np.array([[np.nan, 1.0]]))


def test_token_moments_match_numpy() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 6)) @ rng.normal(size=(6, 6)) + 2.0
    m = TokenMoments(6)
    for i in range(0, 1000, 300):
        m.add(torch.from_numpy(x[i:i + 300]).float())
    np.testing.assert_allclose(m.covariance().numpy(), np.cov(x.T), rtol=1e-4, atol=1e-4)


def test_covariance_directions_follow_the_covariance() -> None:
    cov = torch.diag(torch.tensor([100.0, 1.0, 0.01]))
    D = covariance_directions(cov, 2000, seed=0)
    assert torch.allclose(D.norm(dim=1), torch.ones(2000), atol=1e-5)
    assert D[:, 0].abs().mean() > 0.9           # crowd into the high-variance axis
    assert torch.equal(D, covariance_directions(cov, 2000, seed=0))


def test_code_correlation_matches_numpy_and_flags_dead() -> None:
    rng = np.random.default_rng(1)
    za = np.maximum(rng.normal(size=(500, 5)), 0)
    zb = np.maximum(rng.normal(size=(500, 4)), 0)
    zb[:, 1] = za[:, 3] * 2 + 1                # perfectly correlated pair
    za[:, 0] = 0                               # dead latent
    acc = CodeCorrelation(5, 4)
    for i in range(0, 500, 128):
        acc.add(torch.from_numpy(za[i:i + 128]), torch.from_numpy(zb[i:i + 128]))
    corr = acc.correlation()
    ref = np.corrcoef(za[:, 1:].T, zb.T)[:4, 4:]
    np.testing.assert_allclose(corr[1:], ref, atol=1e-6)
    assert np.isnan(corr[0]).all() and acc.dead_a.tolist() == [True, False, False, False, False]
    score, partner = functional_one_to_one(corr)
    assert partner[0] == -1 and np.isnan(score[0])
    assert partner[3] == 1 and score[3] == pytest.approx(1.0)


def test_functional_matching_recovers_a_permuted_dictionary() -> None:
    from medinterp.dictionaries.baselines import random_sae
    a = random_sae(16, 32, 4, seed=0)
    perm = torch.randperm(32, generator=torch.Generator().manual_seed(1))
    from medinterp.dictionaries.sae import TopKSAE
    b = TopKSAE(a.W_enc[:, perm], a.b_enc[perm], a.W_dec[perm], a.b_dec, a.k)
    x = torch.randn(4000, 16, generator=torch.Generator().manual_seed(2))
    acc = code_correlation(a, b, x.split(1000))
    score, partner = functional_one_to_one(acc.correlation())
    live = ~np.isnan(score)
    assert np.allclose(score[live], 1.0, atol=1e-6)
    assert np.array_equal(perm.numpy()[partner[live]], np.flatnonzero(live))


def test_frozen_decoder_baseline_trains_only_the_encoder() -> None:
    rng = np.random.default_rng(0)
    data = {"a": rng.normal(size=(2000, 8)).astype(np.float16)}
    cfg = TrainConfig(d_model=8, d_sae=32, k=3, steps=30, batch_per_dataset=64, warmup=3, log_every=10**6)
    for kind, cov in (("isotropic", None), ("covariance", torch.eye(8) * 2)):
        sae = frozen_decoder_sae(data, cfg, decoder=kind, cov=cov, decoder_seed=4, log=lambda *_: None)
        W0 = (isotropic_directions(32, 8, 4) if kind == "isotropic"
              else covariance_directions(cov, 32, 4))
        assert torch.allclose(sae.W_dec, W0, atol=1e-6), kind           # decoder unchanged
        assert not torch.allclose(sae.W_enc, W0.T, atol=1e-4), kind     # encoder moved from W_dec^T
    with pytest.raises(ValueError):
        frozen_decoder_sae(data, cfg, decoder="covariance")


# ---------------------------------------------------------------- archive: data-aware null

from pathlib import Path  # noqa: E402

from medinterp.data.archive import DEFAULT_RUN_DIR  # noqa: E402

SEED0 = Path(__file__).resolve().parents[1] / "results/stage0/sae_seeds/seed0"


@pytest.mark.skipif(not (DEFAULT_RUN_DIR / "activations.safetensors").exists() or not SEED0.exists(),
                    reason="thesis archive or seed-0 SAE not present")
def test_covariance_null_reproduces_known_quantile() -> None:
    """Seed 0 vs 4,096 covariance-shaped random directions: q99.9 of partner cosine near 0.77."""
    from medinterp.data.archive import ArchivedActivations
    from medinterp.dictionaries.sae import TopKSAE

    acts = ArchivedActivations()
    moments = TokenMoments(acts.d_model)
    for name in acts.blocks:
        for _, x in acts.iter_token_chunks(name, 1024):
            moments.add(x)
    seed0 = TopKSAE.from_checkpoint(SEED0)
    null = covariance_directions(moments.covariance(), seed0.d_sae, seed=0)
    cos, _ = one_to_one(cosine_matrix(seed0.W_dec, null))
    q = float(np.quantile(cos, 0.999))
    print(f"covariance null: q99.9 = {q:.4f}, share >= 0.7 = {(cos >= 0.7).mean():.4f}")
    assert abs(q - 0.77) <= 0.02


def test_sweep_metrics_on_toy_tokens():
    """FVE of an exact dictionary is 1; a dictionary matches itself functionally and by decoder."""
    from medinterp.eval.sweep_metrics import EvalTokens, decoder_overlap, functional_overlap, score_dictionary
    from medinterp.dictionaries.baselines import LinearTopK

    gen = torch.Generator().manual_seed(0)
    basis = torch.linalg.qr(torch.randn(16, 16, generator=gen))[0]
    enc = LinearTopK(torch.cat([basis, -basis], 1), 16, "full")        # all 16 signed axes: exact
    x = torch.randn(40 * 4, 16, generator=gen)
    tokens = EvalTokens(x, np.repeat(np.arange(20), 2), 4)
    sc = score_dictionary(enc, tokens, n_resamples=50, confidence=0.95, seed=0)
    assert sc.fve == pytest.approx(1.0, abs=1e-5)
    assert sc.fve_ci[0] == pytest.approx(1.0, abs=1e-5)
    assert decoder_overlap(enc.W.T, enc.W.T) == pytest.approx(np.ones(32), abs=1e-5)
    scores, dead_a, dead_b = functional_overlap(enc, enc, tokens)
    assert dead_a == dead_b == 0 and np.allclose(scores, 1.0, atol=1e-5)
