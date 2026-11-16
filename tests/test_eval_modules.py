"""Known-answer tests for reconstruction, sparse probing and selectivity."""
from __future__ import annotations

import numpy as np
import torch

from medinterp.eval.probing import sparse_probe_transfer
from medinterp.eval.recon import FVEAccumulator, RidgeAccumulator, predict, split_images
from medinterp.eval.selectivity import monosemanticity


def test_ridge_recovers_linear_map_and_fve_is_one() -> None:
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2000, 8, generator=g)
    M = torch.randn(8, 5, generator=g)
    x = z @ M + 3.0
    acc = RidgeAccumulator(8, 5)
    for i in range(0, 2000, 500):          # chunked accumulation must equal a single solve
        acc.add(z[i:i + 500], x[i:i + 500])
    W = acc.solve(ridge=1e-9)
    fve = FVEAccumulator(5)
    fve.add(x, predict(W, z))
    assert fve.fve() > 0.9999


def test_fve_zero_for_mean_prediction() -> None:
    x = torch.randn(1000, 4)
    fve = FVEAccumulator(4)
    fve.add(x, x.mean(0, keepdim=True).expand_as(x))
    assert abs(fve.fve()) < 1e-6


def test_group_split_keeps_patients_together() -> None:
    groups = np.repeat(np.arange(100), 3)
    mask = split_images(len(groups), groups, 0.2, seed=0)
    assert all(len(set(mask[groups == g])) == 1 for g in range(100))


def test_monosemanticity_extremes() -> None:
    rng = np.random.default_rng(0)
    findings = rng.random((500, 6)) < 0.3
    findings[:, 0] = False
    findings[:20] = False
    findings[:20, 0] = True                 # 20 images share exactly one finding
    codes = np.zeros((500, 2), dtype=np.float16)
    codes[:20, 0] = 1.0                     # latent 0 fires only on them
    codes[:, 1] = rng.random(500)           # latent 1 fires at random
    scores = monosemanticity(codes, findings, top_n=10, n_null=200)
    assert scores[0, 0] > 0.99 and scores[0, 2] > 0.99
    assert scores[1, 2] < 0.3


def test_sparse_probe_transfer_finds_signal() -> None:
    rng = np.random.default_rng(0)
    codes, labels, masks = {}, {}, {}
    for d in ("a", "b"):
        y = rng.random(400) < 0.5
        z = rng.random((400, 20)).astype(np.float16)
        z[:, 7] += y * 2                    # one informative latent shared by both datasets
        codes[d], labels[d], masks[d] = z, y, rng.random(400) < 0.25
    rows = sparse_probe_transfer(codes, labels, masks, ks=(1,))
    assert all(r["auroc"] > 0.9 for r in rows)


# ---------------------------------------------------------------- protocol v1 additions

import pytest  # noqa: E402

from medinterp.dictionaries.baselines import signed_basis  # noqa: E402
from medinterp.dictionaries.sae import TopKSAE  # noqa: E402
from medinterp.eval.probing import (C_GRID, DEFAULT_C, fit_eval_tuned, pool_tokens,  # noqa: E402
                                    sparse_probe_transfer_tuned)
from medinterp.eval.recon import own_decoder_fve  # noqa: E402


def test_own_decoder_fve_exact_dictionary_is_one() -> None:
    x = torch.randn(600, 8)
    enc = signed_basis(8, k=8)              # every coordinate kept on its own side: exact reconstruction
    assert own_decoder_fve(enc, x.split(200)) == pytest.approx(1.0, abs=1e-6)


def test_own_decoder_fve_mean_is_zero() -> None:
    x = torch.randn(600, 8) + 5.0
    mean_only = TopKSAE(torch.zeros(8, 16), torch.zeros(16), torch.eye(16, 8), x.mean(0), k=2)
    assert own_decoder_fve(mean_only, [x]) == pytest.approx(0.0, abs=1e-6)


def test_pool_tokens() -> None:
    v = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.array_equal(pool_tokens(v, "max"), v[:, 2])
    assert np.array_equal(pool_tokens(v, "mean"), v[:, 1])
    with pytest.raises(ValueError):
        pool_tokens(v, "sum")


def _planted(seed: int, n: int = 600, n_feat: int = 30, signal: int = 7):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.4
    X = rng.normal(size=(n, n_feat)).astype(np.float32)
    X[:, signal] += 1.5 * y
    return X, y


def test_tuned_probe_never_worse_on_val_than_default() -> None:
    assert DEFAULT_C in C_GRID
    for seed in range(5):
        X, y = _planted(seed, n=300, n_feat=120)          # many noise features: regularisation matters
        tr, va = slice(0, 150), slice(150, 300)
        _, best, val_auc = fit_eval_tuned(X[tr], y[tr], X[va], y[va], {})
        assert val_auc[best] >= val_auc[DEFAULT_C]


def test_tuned_transfer_finds_planted_signal() -> None:
    codes, labels, splits = {}, {}, {}
    for i, d in enumerate(("a", "b")):
        X, y = _planted(i)
        lab = y.astype(float)
        lab[:40] = np.nan                                  # unlabelled images are skipped
        s = np.random.default_rng(10 + i).choice(["train", "val", "test"], len(y), p=[0.7, 0.1, 0.2])
        codes[d], labels[d] = X, lab
        splits[d] = {k: s == k for k in ("train", "val", "test")}
    rows = sparse_probe_transfer_tuned(codes, labels, splits, ms=(1, 5), eval_split="test")
    assert {(r["source"], r["target"]) for r in rows} == {("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")}
    assert all(r["auroc"] > 0.75 for r in rows if r["m"] == 1)


def test_selectivity_known_mask_ignores_uncertain_labels() -> None:
    rng = np.random.default_rng(0)
    findings = rng.random((300, 4)) < 0.2
    known = np.ones_like(findings)
    findings[:10] = [True, False, False, False]
    findings[:10:2, 1] = True                  # label 1 present on half of the top images ...
    known[:10, 1] = False                      # ... but only as an uncertain value
    codes = np.zeros((300, 1), dtype=np.float16)
    codes[:10, 0] = 1.0
    plain = monosemanticity(codes, findings, top_n=10, n_null=100)
    masked = monosemanticity(codes, findings, top_n=10, n_null=100, known=known)
    assert masked[0, 2] > plain[0, 2] and masked[0, 1] == pytest.approx(1.0)
    # Without a mask the result is unchanged from the old code path.
    np.testing.assert_array_equal(plain, monosemanticity(codes, findings, top_n=10, n_null=100, known=None))


def test_per_image_recon_equals_token_fve() -> None:
    from medinterp.eval.recon import PerImageRecon
    x = torch.randn(10 * 49, 6) * 3 + 1
    x_hat = x + torch.randn_like(x)
    per, tok = PerImageRecon(), FVEAccumulator(6)
    for chunk in range(0, 10 * 49, 5 * 49):
        per.add(x[chunk:chunk + 245], x_hat[chunk:chunk + 245], 49)
    tok.add(x, x_hat)
    assert per.fve() == pytest.approx(tok.fve(), abs=1e-9)
    # Resampling with repeats equals the FVE of the repeated tokens.
    idx = np.array([0, 0, 3, 7])
    rows = torch.cat([torch.arange(i * 49, (i + 1) * 49) for i in idx])
    rep = FVEAccumulator(6)
    rep.add(x[rows], x_hat[rows])
    assert per.fve(idx) == pytest.approx(rep.fve(), abs=1e-9)


def test_mean_cross_auroc_statistic_matches_rows() -> None:
    from medinterp.eval.probing import eval_images, mean_cross_auroc_statistic
    codes, labels, splits = {}, {}, {}
    for i, d in enumerate(("a", "b", "c")):
        X, y = _planted(20 + i)
        codes[d], labels[d] = X, y.astype(float)
        s = np.random.default_rng(i).choice(["train", "val", "test"], len(y), p=[0.6, 0.2, 0.2])
        splits[d] = {k: s == k for k in ("train", "val", "test")}
    rows = sparse_probe_transfer_tuned(codes, labels, splits, ms=(1,), eval_split="test", keep_scores=True)
    masks = eval_images(labels, splits, "test")
    stat = mean_cross_auroc_statistic(rows, 1, labels, masks)
    n = sum(int(m.sum()) for m in masks.values())
    expected = np.mean([r["auroc"] for r in rows if r["source"] != r["target"]])
    assert stat(np.arange(n)) == pytest.approx(expected)
