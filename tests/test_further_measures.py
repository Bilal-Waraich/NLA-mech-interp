"""Further secondary measures of protocol section 4 on small synthetic inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.further_measures import (RecoveryProbes, absorption, decode_pooled, image_features,
                                             joint_seed_overlap, pooled_activations, pw_mcc, secondary_targets,
                                             targeted_ablation)


def _sae(seed: int, d_model: int = 8, d_sae: int = 16, k: int = 3) -> TopKSAE:
    g = torch.Generator().manual_seed(seed)
    W_dec = torch.randn(d_sae, d_model, generator=g)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    return TopKSAE(W_dec.T.clone(), torch.zeros(d_sae), W_dec, torch.zeros(d_model), k)


def test_joint_overlap_and_pw_mcc_of_a_permuted_copy_are_one():
    a = _sae(0)
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(1))
    b = TopKSAE(a.W_enc[:, perm], a.b_enc[perm], a.W_dec[perm], a.b_dec, a.k)
    out = joint_seed_overlap(a, b, 0.7)
    assert out["share_joint"] == out["share_decoder"] == out["share_encoder"] == 1.0
    assert pw_mcc(a.W_dec, b.W_dec) == pytest.approx(1.0)


def test_joint_overlap_needs_the_same_partner():
    a = _sae(0)
    # Same decoder, encoder columns rotated by one: decoder matches, encoder partners all disagree.
    roll = torch.roll(torch.arange(16), 1)
    b = TopKSAE(a.W_enc[:, roll], a.b_enc, a.W_dec, a.b_dec, a.k)
    out = joint_seed_overlap(a, b, 0.7)
    assert out["share_decoder"] == 1.0 and out["share_encoder"] == 1.0 and out["share_joint"] == 0.0


class _Acts:
    """Minimal stand-in for ArchivedActivations: two datasets, 4 tokens per image."""

    def __init__(self, data: dict[str, np.ndarray]):
        self.data, self.n_patches, self.d_model = data, 4, next(iter(data.values())).shape[-1]
        self.blocks = {n: type("B", (), {"n_images": v.shape[0]})() for n, v in data.items()}

    def iter_token_chunks(self, name, images_per_chunk=512):
        v = self.data[name]
        for i in range(0, v.shape[0], images_per_chunk):
            yield i, torch.from_numpy(v[i:i + images_per_chunk].reshape(-1, self.d_model))


def test_image_features_and_pooled_reconstruction():
    rng = np.random.default_rng(0)
    acts = _Acts({"a": rng.normal(size=(6, 4, 8)).astype(np.float32), "b": rng.normal(size=(5, 4, 8)).astype(np.float32)})
    sae = _sae(0, k=16)                                   # keeps every positive code
    keep = {"a": np.array([1, 1, 0, 1, 0, 1], bool), "b": np.ones(5, bool)}
    f = image_features(acts, sae, keep, images_per_chunk=2)
    x = torch.from_numpy(acts.data["a"][3])
    z = sae.encode(x)
    assert np.allclose(f.max_codes["a"][3], z.amax(0).numpy(), atol=1e-3)
    assert np.allclose(f.mean_codes["a"][3], z.mean(0).numpy(), atol=1e-6)
    assert not f.max_codes["a"][2].any()                  # not kept
    rec = decode_pooled(sae, f.mean_codes)
    assert np.allclose(rec["a"][3], sae.decode(z).mean(0).numpy(), atol=1e-5)
    pooled = pooled_activations(acts, keep)
    assert np.allclose(pooled["b"][1], acts.data["b"][1].mean(0), atol=1e-6)


def test_secondary_targets():
    man = {"chexpert": pd.DataFrame({"sex": ["Male", "Female", "Unknown"], "Lung Opacity": [1, -1, np.nan],
                                     "Support Devices": [1, 0, np.nan]}),
           "rsna": pd.DataFrame({"x": [0, 1]})}
    opacity = {"chexpert": np.array([1, np.nan, 0]), "rsna": np.array([1, 0])}
    t = secondary_targets(man, opacity)
    assert np.array_equal(t["Lung Opacity"]["chexpert"], [1, np.nan, 0], equal_nan=True)
    assert np.isnan(t["Lung Opacity"]["rsna"]).all()
    assert np.array_equal(t["sex"]["chexpert"], [1, 0, np.nan], equal_nan=True)
    assert list(t["dataset_rsna"]["rsna"]) == [1, 1] and list(t["dataset_rsna"]["chexpert"]) == [0, 0, 0]
    assert "opacity_rsna" in t and np.isnan(t["opacity_rsna"]["chexpert"]).all()


def _split_masks(n: int, rng) -> dict[str, np.ndarray]:
    which = rng.choice(["train", "val"], size=n, p=[0.6, 0.4])
    return {"train": which == "train", "val": which == "val", "test": np.zeros(n, bool)}


def test_recovery_of_a_perfect_reconstruction_is_one():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(float)
    pooled, targets = {"a": X}, {"t": {"a": y}}
    splits, groups = {"a": _split_masks(n, rng)}, {"a": np.arange(n)}
    probes = RecoveryProbes(pooled, targets, splits, groups)
    row = probes.score(pooled, 50, 0.95, 0)[0]
    assert row["recovered_share"] == pytest.approx(1.0) and row["diff"] == pytest.approx(0.0)
    noise = probes.score({"a": rng.normal(size=(n, 6)).astype(np.float32)}, 50, 0.95, 0)[0]
    assert noise["recovered_share"] < 0.5


def test_absorption_counts_the_parent_latent_off():
    n = 200
    rng = np.random.default_rng(0)
    parent = (np.arange(n) % 2).astype(float)                 # half the images positive
    child = np.where(parent == 1, (np.arange(n) % 4 == 1).astype(float), 0.0)
    codes = np.zeros((n, 3), np.float16)
    codes[:, 0] = parent * 2                                  # latent 0 tracks the parent ...
    codes[child == 1, 0] = 0                                  # ... but is off on every child-positive image
    codes[:, 1] = rng.random(n)
    labels = {"Lung Opacity": parent, "Consolidation": child, "Pneumonia": np.zeros(n), "Edema": np.zeros(n),
              "Atelectasis": np.zeros(n)}
    sel = np.arange(n) < 120
    out = absorption(codes, labels, sel, ~sel)
    assert out["parent_latent"] == 0 and out["rate_Consolidation"] == 1.0 and out["rate_no_child"] == 0.0


def test_targeted_ablation_is_selective_for_the_latent_that_carries_the_confounder():
    rng = np.random.default_rng(0)
    n = 600
    codes = rng.random((n, 10)).astype(np.float32)
    conf = (codes[:, 0] > 0.5).astype(float)
    other = (codes[:, 5] > 0.5).astype(float)
    splits = {"a": _split_masks(n, rng)}
    rows = targeted_ablation({"a": codes}, {"sex": {"a": conf}, "opacity_a": {"a": other}}, splits,
                             confounders=("sex",), comparisons=("opacity_a",), ns=(1,))
    assert rows[0]["drop"] > 0.3 and rows[0]["selectivity"] > 0.3


def _sweep_eval():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("sweep_eval", Path(__file__).parents[1] / "scripts/sweep_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nearest_null_k_and_p11_by_architecture():
    se = _sweep_eval()
    nulls = [(1024, 8), (1024, 16), (1024, 32), (4096, 32)]
    assert se.nearest_null_k(11.0, 1024, nulls) == 8 and se.nearest_null_k(12.0, 1024, nulls) == 8
    assert se.nearest_null_k(13.0, 1024, nulls) == 16 and se.nearest_null_k(5.0, 2048, nulls) is None
    rows = []
    for s, l0, v in ((0, 15.0, 0.90), (1, 17.0, 0.91)):          # ReLU+L1 setting, measured L0 near 16
        rows.append({"setting": "d1024_k16_s80000_relul1", "dictionary": f"r{s}", "dictionary_kind": "trained",
                     "d_sae": 1024, "k": 16, "step": 80_000, "value": v, "mean_l0": l0})
    for s, v in ((0, 0.80), (1, 0.85)):                           # nulls at k = 16 and at k = 8
        rows.append({"setting": "nulls_d1024_k16", "dictionary": f"random_seed{s}", "dictionary_kind": "random_sae",
                     "d_sae": 1024, "k": 16, "step": 0, "value": v, "mean_l0": 16.0})
        rows.append({"setting": "nulls_d1024_k8", "dictionary": f"random_seed{s}", "dictionary_kind": "random_sae",
                     "d_sae": 1024, "k": 8, "step": 0, "value": v - 0.1, "mean_l0": 8.0})
    rows.append({"setting": "d1024_k8_s80000", "dictionary": "b0", "dictionary_kind": "trained", "d_sae": 1024,
                 "k": 8, "step": 80_000, "value": 0.84, "mean_l0": np.nan})    # old cache: no L0 recorded
    table = se.p11_architectures_table(pd.DataFrame(rows)).set_index("setting")
    relu = table.loc["d1024_k16_s80000_relul1"]
    assert relu.architecture == "relu_l1" and relu.null_k == 16 and bool(relu.separated)
    assert relu.gap == pytest.approx(0.90 - 0.85)
    ref = table.loc["d1024_k8_s80000"]
    assert ref.architecture == "batchtopk" and ref.null_k == 8 and ref.mean_l0 == 8.0


def test_recovery_mean_leaves_out_dataset_identity():
    se = _sweep_eval()
    rows = [{"target": "opacity_a", "recovered_share": 0.9}, {"target": "sex", "recovered_share": 0.7},
            {"target": "dataset_a", "recovered_share": -3.0}]
    assert se.recovery_mean(rows) == pytest.approx(0.8)
