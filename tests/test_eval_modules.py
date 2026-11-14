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
