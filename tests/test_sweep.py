"""Sweep run table, data subsets, resumable runs, headline rule and outcome table (protocol section 5)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest
import torch

from medinterp.dictionaries.sae import TopKSAE
from medinterp.sweep import (classify_outcome, headline_k, parse_rows, subset_train_images, sweep_table,
                             token_rows, train_run)


def test_table_phase_one_matches_protocol():
    t = sweep_table()
    assert sorted(t.phase.unique()) == ["1a", "1b", "1n"]
    assert t[t.phase == "1a"].run_units.sum() == 25
    assert t[t.phase == "1b"].run_units.sum() == pytest.approx(6.25 + 12)
    assert set(t[t.phase == "1a"].k) == {4, 8, 16, 32, 48}
    assert (t.groupby("setting").size()[lambda s: s.index.str.startswith("d4096_k")
                                          & ~s.index.str.endswith("frozeniso")] == 5).all()
    assert (t[t.phase != "1n"].decoder == "trained").all()


def test_table_with_headline_appends_phase_two_without_renumbering():
    base, full = sweep_table(), sweep_table(headline_k=16)
    assert full.iloc[:len(base)].equals(base)
    two = full[full.phase.str.startswith("2")]
    assert set(two.k) == {16}
    assert full.run_units.sum() == pytest.approx(74.25)
    seedsplit = two[two.phase == "2b"]
    assert sorted(zip(seedsplit.init_seed, seedsplit.data_seed)) == [(0, 1), (1, 0)]
    with pytest.raises(ValueError):
        sweep_table(headline_k=12)


def test_parse_rows():
    assert parse_rows("2:5", 10) == [2, 3, 4]
    assert parse_rows(":2", 10) == [0, 1]
    assert parse_rows("1,7", 10) == [1, 7]
    with pytest.raises(ValueError):
        parse_rows("8:12", 10)


def test_subset_is_grouped_nested_and_near_fraction():
    rng = np.random.default_rng(0)
    groups = {"a": rng.integers(0, 300, 1000).astype(str), "b": rng.integers(0, 50, 400).astype(str)}
    train = {d: rng.random(len(g)) < 0.7 for d, g in groups.items()}
    quarter = subset_train_images(train, groups, 0.25)
    half = subset_train_images(train, groups, 0.5)
    for d in groups:
        assert not (quarter[d] & ~train[d]).any()
        assert not (quarter[d] & ~half[d]).any()             # nested
        assert abs(half[d].sum() / train[d].sum() - 0.5) < 0.1
        inside = set(groups[d][quarter[d]])
        # whole groups: every train image of a chosen group is chosen
        assert all(quarter[d][i] for i in np.flatnonzero(train[d]) if groups[d][i] in inside)


def test_token_rows():
    assert token_rows(np.array([False, True, True]), 3).tolist() == [3, 4, 5, 6, 7, 8]


def _row(**kw):
    row = {"phase": "1a", "setting": "toy", "run_id": "toy_i0_b0", "decoder": "trained", "d_sae": 32, "k": 4,
           "steps": 40, "init_seed": 0, "data_seed": 0, "sampling": "equal", "l1_coeff": 2e-4,
           "data_fraction": 1.0, "checkpoints": "20;40", "run_units": 0.0}
    row.update(kw)
    return pd.Series(row)


def _data():
    rng = np.random.default_rng(0)
    return {"a": rng.normal(size=(500, 1024)).astype(np.float16), "b": rng.normal(size=(500, 1024)).astype(np.float16)}


def test_train_run_pause_resume_matches_uninterrupted(tmp_path):
    data, rows = _data(), {"a": np.arange(400), "b": np.arange(300)}
    quiet = lambda *_: None
    full = train_run(_row(), data, rows, tmp_path / "full", "cpu", time.time() + 3600, quiet, state_every=10)
    assert full.status == "done"
    # A deadline already in the past stops after the first step and leaves a state to resume from.
    paused = train_run(_row(), data, rows, tmp_path / "resumed", "cpu", 0.0, quiet, state_every=10)
    assert paused.status == "paused" and paused.step == 1
    assert (tmp_path / "resumed/toy_i0_b0/state.pt").exists()
    done = train_run(_row(), data, rows, tmp_path / "resumed", "cpu", time.time() + 3600, quiet, state_every=10)
    assert done.status == "done" and not (tmp_path / "resumed/toy_i0_b0/state.pt").exists()
    for step in (20, 40):
        a = TopKSAE.from_checkpoint(tmp_path / f"full/toy_i0_b0/step_{step}")
        b = TopKSAE.from_checkpoint(tmp_path / f"resumed/toy_i0_b0/step_{step}")
        assert torch.equal(a.W_dec, b.W_dec) and torch.equal(a.W_enc, b.W_enc)
    cfg = json.loads((tmp_path / "full/toy_i0_b0/step_40/cfg.json").read_text())
    assert cfg["fit_split"] == "train" and cfg["k"] == 4 and cfg["step"] == 40
    again = train_run(_row(), data, rows, tmp_path / "full", "cpu", time.time() + 3600, quiet)
    assert again.status == "skipped"


def test_frozen_decoder_run_keeps_its_decoder(tmp_path):
    from medinterp.dictionaries.baselines import isotropic_directions
    row = _row(decoder="frozen_isotropic", run_id="toy_frozen", checkpoints="40")
    train_run(row, _data(), {"a": np.arange(500), "b": np.arange(500)}, tmp_path, "cpu", time.time() + 3600,
              lambda *_: None)
    sae = TopKSAE.from_checkpoint(tmp_path / "toy_frozen/step_40")
    assert torch.allclose(sae.W_dec, isotropic_directions(32, 1024, 0), atol=1e-6)


def test_headline_rule_uses_worst_seed_against_best_null():
    fve = pd.DataFrame([
        *({"k": 4, "dictionary_kind": "trained", "dictionary": f"s{i}", "value": v} for i, v in enumerate([0.70, 0.72])),
        {"k": 4, "dictionary_kind": "signed_pca", "dictionary": "pca", "value": 0.60},
        *({"k": 48, "dictionary_kind": "trained", "dictionary": f"s{i}", "value": v} for i, v in enumerate([0.92, 0.95])),
        {"k": 48, "dictionary_kind": "signed_pca", "dictionary": "pca", "value": 0.88},
        {"k": 48, "dictionary_kind": "random_sae", "dictionary": "r", "value": 0.01},
    ])
    k, gaps = headline_k(fve)
    assert k == 4
    assert gaps.set_index("k").loc[48, "gap"] == pytest.approx(0.04)
    assert gaps.set_index("k").loc[4, "best_null"] == "pca"


def test_outcome_table_order():
    th = {"high": 0.30, "low": 0.15, "under_training": 0.10}
    small = ["d4096_k4", "d1024_k48"]
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.35}, "t", small, None, th) == "operating point"
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.10}, "t", small, 0.12, th) == "under-training"
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.10}, "t", small, 0.01, th) == "robust low"
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.10}, "t", small, None, th) == "in between"
    assert classify_outcome({"t": 0.20, "d4096_k4": 0.10}, "t", small, 0.01, th) == "in between"
