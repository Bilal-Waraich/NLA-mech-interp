"""Unit tests that need no archive: baseline dictionaries and metric primitives."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from medinterp.dictionaries.baselines import random_sae, signed_basis, signed_pca
from medinterp.eval.metrics import apply_rule, auroc_columns, latent_table


def test_auroc_matches_sklearn_with_ties() -> None:
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(500, 6)).astype(np.float32)   # many ties, like sparse codes
    y = rng.random(500) < 0.3
    ours = auroc_columns(scores, y)
    ref = [roc_auc_score(y, scores[:, j]) for j in range(6)]
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_random_sae_is_seeded_and_sparse() -> None:
    a, b, c = random_sae(32, 128, 4, seed=1), random_sae(32, 128, 4, seed=1), random_sae(32, 128, 4, seed=2)
    assert torch.equal(a.W_enc, b.W_enc) and not torch.equal(a.W_enc, c.W_enc)
    assert torch.allclose(a.W_dec.norm(dim=1), torch.full((128,), 0.1))
    z = a.encode(torch.randn(64, 32))
    assert ((z > 0).sum(1) <= 4).all()


def test_signed_dictionaries_fire_one_side_only() -> None:
    x = torch.randn(200, 16)
    for enc in (signed_basis(16, k=3), signed_pca(x, n_components=16, k=3)):
        z = enc.encode(x)
        n = z.shape[1] // 2
        # A token can never activate both +v and -v of the same direction.
        assert not ((z[:, :n] > 0) & (z[:, n:] > 0)).any()
        assert ((z > 0).sum(1) <= 3).all()


def test_rule_on_synthetic_table() -> None:
    rng = np.random.default_rng(1)
    names = ["a", "b"]
    n = 400
    # Latent 0: fires everywhere, tracks the label in both datasets. Latent 1: only in dataset a.
    groups = {d: np.where(rng.random(n) < 0.5, 0, 5) for d in names}
    codes = {}
    for d in names:
        z = np.zeros((n, 2), dtype=np.float16)
        z[:, 0] = (groups[d] == 0) * 2.0 + rng.random(n)
        z[:, 1] = rng.random(n) if d == "a" else 0
        codes[d] = z
    table = apply_rule(latent_table(codes, groups), names, tau=0.05, dsauc_threshold=0.6,
                       label_datasets=names)
    assert table.loc[0, "label_consistent"] and table.loc[0, "label_direction"] == "+"
    assert not table.loc[1, "active_all"]


def test_rule_v1_on_synthetic_table() -> None:
    from medinterp.eval.metrics import apply_rule_v1, latent_table_v1
    rng = np.random.default_rng(2)
    names, n = ["a", "b"], 600
    codes, opacity, groups = {}, {}, {}
    for d in names:
        lab = np.where(rng.random(n) < 0.5, 1.0, 0.0)
        lab[:50] = np.nan                                   # left out of the contrast
        z = np.zeros((n, 3), dtype=np.float16)
        z[:, 0] = (lab == 1) * 2.0 + rng.random(n)          # label latent, present everywhere
        z[:, 1] = rng.random(n) + (3.0 if d == "a" else 0)  # dataset latent
        z[:, 2] = rng.random(n)                             # noise
        codes[d], opacity[d], groups[d] = z, lab, np.repeat(np.arange(n // 2), 2).astype(str)
    t = apply_rule_v1(latent_table_v1(codes, opacity, groups), names, names, tau=0.05,
                      dsauc_band=(0.35, 0.65), fdr=0.05)
    assert t.loc[0, "label_consistent"] and t.loc[0, "label_direction"] == "+"
    assert t.loc[1, "active_all"] and not t.loc[1, "cross_dataset"]   # identifies its dataset
    assert t.loc[2, "cross_dataset"] and not t.loc[2, "label_consistent"]
