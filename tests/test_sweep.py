"""Sweep run table, data subsets, resumable runs, settings carried forward and outcome table (protocol section 5)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest
import torch

from medinterp.dictionaries.sae import TopKSAE
from medinterp.sweep import (classify_outcome, p11_supported, parse_rows, seedsplit_base_rows, setting_role,
                             subset_train_images, summarise_decoder_diagnostic, sweep_table, token_rows,
                             train_run, training_length_table)


def test_table_phase_one_matches_protocol():
    t = sweep_table()
    assert sorted(t.phase.unique()) == ["1a", "1b", "1n"]
    assert t[t.phase == "1a"].run_units.sum() == 25
    assert t[t.phase == "1b"].run_units.sum() == pytest.approx(6.25 + 12)
    assert set(t[t.phase == "1a"].k) == {4, 8, 16, 32, 48}
    assert (t.groupby("setting").size()[lambda s: s.index.str.startswith("d4096_k")
                                          & ~s.index.str.endswith("frozeniso")] == 5).all()
    assert (t[t.phase != "1n"].decoder == "trained").all()


def test_phase_two_is_appended_without_renumbering():
    base, full = sweep_table(), sweep_table(phase2=True)
    assert len(base) == 58 and len(full) == 115
    assert full.iloc[:len(base)].equals(base)
    assert full.run_units.sum() == pytest.approx(91.75)


def test_phase_two_matches_the_protocol_table():
    full = sweep_table(phase2=True)
    two = full[full.phase.str.startswith("2")]
    units = two.groupby("phase").run_units.sum()
    assert units.to_dict() == pytest.approx({"2a": 33.0, "2b": 1.5, "2c": 4.5, "2d": 4.5})
    all_tests = {(1024, 8), (1024, 16), (1024, 32)}
    assert set(zip(two[two.phase == "2a"].d_sae, two[two.phase == "2a"].k)) == all_tests | {(4096, 32), (4096, 48)}
    for phase in ("2b", "2c", "2d"):
        part = two[two.phase == phase]
        assert set(zip(part.d_sae, part.k)) == all_tests
    long_runs = two[two.phase == "2a"]
    assert (long_runs.steps == 80_000).all() and (long_runs.checkpoints == "20000;40000;80000").all()
    assert (long_runs.groupby("setting").size() == 3).all()
    seedsplit = two[(two.phase == "2b") & (two.k == 8)]
    assert sorted(zip(seedsplit.init_seed, seedsplit.data_seed)) == [(0, 1), (1, 0)]


def test_seedsplit_reuses_the_phase_one_runs_of_its_setting():
    full = sweep_table(phase2=True)
    base = seedsplit_base_rows(full, 1024, 16)
    assert base.run_id.tolist() == ["d1024_k16_s20000_i0_b0", "d1024_k16_s20000_i1_b1"]
    assert set(base.phase) == {"1b"}


def test_setting_roles():
    assert [setting_role(1024, k) for k in (8, 16, 32)] == ["core"] * 3
    assert setting_role(4096, 32) == "core"
    assert setting_role(4096, 48) == "dense reference"
    assert {setting_role(4096, k) for k in (4, 8, 16)} == {"appendix"}
    assert setting_role(1024, 48) == "appendix"
    assert setting_role(1024, 4) == setting_role(16_384, 16) == "not carried forward"


def test_p11_counts_every_setting_up_to_max_k_including_the_appendix():
    good = pd.DataFrame({"d_sae": [1024, 4096, 4096], "k": [8, 4, 48], "gap": [0.10, 0.14, 0.01],
                         "separated": [True, True, True]})
    assert p11_supported(good, margin=0.02, max_k=16)              # k = 48 lies outside the margin rule
    narrow_appendix = good.assign(gap=[0.10, 0.01, 0.03])
    assert not p11_supported(narrow_appendix, margin=0.02, max_k=16)
    overlapping = good.assign(separated=[True, False, True])
    assert not p11_supported(overlapping, margin=0.02, max_k=16)
    with pytest.raises(ValueError):
        p11_supported(good[good.k == 48], margin=0.02, max_k=16)


def test_training_length_compares_the_80k_runs_with_the_phase_one_runs():
    curve = pd.DataFrame([
        {"setting": "d1024_k8_s20000", "steps": 20_000, "S_mean": 0.35},
        {"setting": "d1024_k8_s80000", "steps": 20_000, "S_mean": 0.20},
        {"setting": "d1024_k8_s80000", "steps": 80_000, "S_mean": 0.47},
        {"setting": "d4096_k48_s20000", "steps": 20_000, "S_mean": 0.09},
    ])
    table = training_length_table(curve)
    assert table.setting.tolist() == ["d1024_k8_s20000"]         # k = 48 has no 80k runs yet
    assert table.loc[0, "delta"] == pytest.approx(0.12)
    assert table.loc[0, "S_80k_run_at_20k"] == pytest.approx(0.20)
    assert table.loc[0, "role"] == "core"


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


def test_decoder_diagnostic_summary_groups_seeds_and_adds_dead_latents():
    runs = pd.DataFrame({"run_id": ["a", "b", "c"], "d_sae": [1024, 1024, 4096], "k": [8, 8, 48],
                         "step": [20_000] * 3, "c_dec": [0.10, 0.20, 0.30]})
    fve = pd.DataFrame({"setting": ["d1024_k8_s20000", "d1024_k8_s20000", "d1024_k8_s80000"], "d_sae": [1024] * 3,
                        "k": [8] * 3, "step": [20_000] * 3, "dictionary_kind": ["trained"] * 3, "n_dead": [2, 4, 100]})
    table = summarise_decoder_diagnostic(runs, fve).set_index("setting")
    assert table.loc["d1024_k8_s20000", "c_dec_mean"] == pytest.approx(0.15)
    assert table.loc["d1024_k8_s20000", "n_runs"] == 2 and table.loc["d1024_k8_s20000", "role"] == "core"
    assert table.loc["d1024_k8_s20000", "mean_dead_on_val"] == pytest.approx(3.0)   # the 80k run is excluded
    assert table.loc["d4096_k48_s20000", "role"] == "dense reference"
    assert np.isnan(table.loc["d4096_k48_s20000", "mean_dead_on_val"])


def test_outcome_table_order():
    th = {"high": 0.30, "low": 0.15, "under_training": 0.10}
    small = ["d4096_k4", "d1024_k48"]
    flat, rising = {"a": 0.01, "b": 0.02}, {"a": 0.01, "b": 0.12}
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.35}, "t", small, rising, th) == "operating point"
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.10}, "t", small, rising, th) == "under-training"
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.10}, "t", small, flat, th) == "robust low"
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.10}, "t", small, None, th) == "in between"
    assert classify_outcome({"t": 0.09, "d4096_k4": 0.10}, "t", small, {}, th) == "in between"
    assert classify_outcome({"t": 0.20, "d4096_k4": 0.10}, "t", small, flat, th) == "in between"
