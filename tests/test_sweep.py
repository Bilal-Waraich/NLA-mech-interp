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


def test_later_phases_are_appended_without_renumbering():
    base, full = sweep_table(), sweep_table(full=True)
    assert len(base) == 58 and len(full) >= 192
    assert set(full.iloc[133:139].phase) == {"2e"} and set(full.iloc[133:139].steps) == {160_000}
    assert full.iloc[:len(base)].equals(base)
    assert full.iloc[:172].run_units.sum() == pytest.approx(165.25)
    tail = full.iloc[130:133]
    assert set(tail.phase) == {"2a"} and set(tail.setting) == {"d1024_k64_s80000"}
    # Phase 2 was decided before phase 1c, so it keeps rows 58-114 and phase 1c follows.
    assert set(full.iloc[58:115].phase) == {"2a", "2b", "2c", "2d"}
    assert set(full.iloc[115:130].phase) == {"1c"}


def test_k_extension_matches_the_protocol_table():
    ext = sweep_table(full=True).query("phase == '1c'")
    assert set(zip(ext.d_sae, ext.k)) == {(1024, 64), (1024, 96), (4096, 64)}
    assert (ext.groupby("setting").size() == 5).all() and (ext.steps == 20_000).all()
    assert (ext.init_seed == ext.data_seed).all() and ext.run_units.sum() == pytest.approx(7.5)
    assert ext.row.tolist() == list(range(115, 130))
    assert ext.checkpoints.eq("5000;10000;20000").all()


def test_phase_two_matches_the_protocol_table():
    full = sweep_table(full=True)
    two = full[full.phase.str.startswith("2")]
    units = two.groupby("phase").run_units.sum()
    assert units.to_dict() == pytest.approx({"2a": 36.0, "2b": 1.5, "2c": 4.5, "2d": 4.5, "2e": 12.0, "2f": 33.0, "2g": 18.0})
    all_tests = {(1024, 8), (1024, 16), (1024, 32)}
    assert set(zip(two[two.phase == "2a"].d_sae, two[two.phase == "2a"].k)) == all_tests | {(4096, 32), (4096, 48), (1024, 64)}
    for phase in ("2b", "2c", "2d"):
        part = two[two.phase == phase]
        assert set(zip(part.d_sae, part.k)) == all_tests
    long_runs = two[two.phase == "2a"]
    assert (long_runs.steps == 80_000).all() and (long_runs.checkpoints == "20000;40000;80000").all()
    assert (long_runs.groupby("setting").size() == 3).all()
    seedsplit = two[(two.phase == "2b") & (two.k == 8)]
    assert sorted(zip(seedsplit.init_seed, seedsplit.data_seed)) == [(0, 1), (1, 0)]


def test_seedsplit_reuses_the_phase_one_runs_of_its_setting():
    full = sweep_table(full=True)
    base = seedsplit_base_rows(full, 1024, 16)
    assert base.run_id.tolist() == ["d1024_k16_s20000_i0_b0", "d1024_k16_s20000_i1_b1"]
    assert set(base.phase) == {"1b"}


def test_setting_roles():
    assert [setting_role(1024, k) for k in (8, 16, 32)] == ["core"] * 3
    assert setting_role(4096, 32) == "core"
    assert setting_role(4096, 48) == "dense reference"
    assert {setting_role(4096, k) for k in (4, 8, 16)} == {"appendix"}
    assert setting_role(1024, 48) == "appendix"
    assert setting_role(1024, 64) == "dense reference" and setting_role(1024, 96) == "appendix"
    assert setting_role(4096, 64) == "appendix"
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


def test_per_latent_pair_table_and_summary():
    from medinterp.sweep import per_latent_pair_table, summarise_per_latent

    cos = np.array([0.9, 0.8, 0.2, np.nan])
    partner = np.array([1, 0, 2, -1])
    rate_a = np.array([0.5, 0.3, 0.1, 0.0])           # the last latent never fires
    rate_b = np.array([0.4, 0.6, 0.0])                # partner 2 never fires in run B
    table = per_latent_pair_table(cos, partner, rate_a, rate_b)
    assert table.image_rate_partner.tolist()[:3] == [0.6, 0.4, 0.0] and np.isnan(table.image_rate_partner[3])
    summary = summarise_per_latent([table], threshold=0.7)
    assert summary["S_all"]["mean"] == pytest.approx(0.5)          # 2 of 4 latents
    assert summary["S_live"]["mean"] == pytest.approx(2 / 3)       # 2 of the 3 live latents
    assert summary["recurring_live_in_both"]["mean"] == pytest.approx(1.0)
    assert summary["spearman_rate_cosine"]["mean"] == pytest.approx(1.0)
    assert summary["median_image_rate_recurring"] == pytest.approx(0.4)
    assert summary["median_image_rate_not_recurring"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        per_latent_pair_table(cos, partner[:3], rate_a, rate_b)
    with pytest.raises(ValueError):
        summarise_per_latent([], 0.7)


def _probe_summary(m1, m5):
    trained = {f"run{i}": {"probe_mean_cross_auroc_m1": [a, 0, 1], "probe_mean_cross_auroc_m5": [b, 0, 1]}
               for i, (a, b) in enumerate(zip(m1, m5))}
    nulls = {"random_seed0": {"probe_mean_cross_auroc_m1": [0.5, 0, 1], "probe_mean_cross_auroc_m5": [0.5, 0, 1]},
             "signed_pca": {"probe_mean_cross_auroc_m1": [0.6, 0, 1], "probe_mean_cross_auroc_m5": [0.6, 0, 1]}}
    return {"dictionaries": {**trained, **nulls},
            "trained_vs_null": {"probe_m1": {"best_null": "signed_pca", "separated": True, "gap": 0.1}}}


def test_label_transfer_table_and_extension_rule():
    from medinterp.sweep import extension_k_role, label_transfer_table

    core = {"d1024_k8_s20000": (0.80, 0.82), "d1024_k16_s20000": (0.79, 0.83),
            "d1024_k32_s20000": (0.78, 0.81), "d4096_k32_s20000": (0.77, 0.80)}
    summaries = {s: _probe_summary([a, a], [b, b]) for s, (a, b) in core.items()}
    summaries["d1024_k64_s20000"] = _probe_summary([0.81, 0.83], [0.85, 0.85])
    table = label_transfer_table(summaries)
    k64 = table.set_index("setting").loc["d1024_k64_s20000"]
    assert k64.n_seeds == 2 and k64.m1_mean == pytest.approx(0.82) and k64.m1_best_null == "signed_pca"
    assert extension_k_role(table) == "core"                     # above every core setting at m = 1 and 5
    weaker = summaries | {"d1024_k64_s20000": _probe_summary([0.81, 0.81], [0.82, 0.82])}
    assert extension_k_role(label_transfer_table(weaker)) == "dense reference"   # m = 5 not above k = 16
    with pytest.raises(ValueError):
        extension_k_role(table[table.k != 64])


def test_collect_settings_finds_seedsplit_runs_across_run_roots(tmp_path):
    """Phase 2b's 2 x 2 design takes two runs from phase 1 and two from phase 2, which live in different
    Kaggle outputs; the setting must be complete only when every run is found in one of the roots."""
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/sweep_eval.py"
    spec = importlib.util.spec_from_file_location("sweep_eval", script)
    sweep_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_eval)

    full = sweep_table(full=True)
    seedsplit = full[full.setting == "d1024_k8_s20000_seedsplit"]
    base = seedsplit_base_rows(full, 1024, 8)
    phase1_root, phase2_root = tmp_path / "phase1", tmp_path / "phase2"
    for root, rows in ((phase1_root, base), (phase2_root, seedsplit)):
        for run_id in rows.run_id:
            (root / run_id).mkdir(parents=True)
            (root / run_id / "DONE.json").write_text("{}")

    only_phase2 = sweep_eval.collect_settings(full, [phase2_root], None)
    assert "d1024_k8_s20000_seedsplit" not in only_phase2
    both = sweep_eval.collect_settings(full, [phase1_root, phase2_root], None)
    runs = both["d1024_k8_s20000_seedsplit"]["runs"]
    assert set(runs) == set(base.run_id) | set(seedsplit.run_id)
    for run_id in base.run_id:
        assert runs[run_id]["steps"][20_000] == phase1_root / run_id / "step_20000"


def test_phase_2f_and_2g_rows():
    full = sweep_table(full=True)
    f = full[full.phase == "2f"]
    assert len(f) == 15 and set(f.projection) == {"nopc1"} and set(f.steps) == {80_000}
    assert set(zip(f.d_sae, f.k)) == {(1024, 8), (1024, 16), (1024, 32), (4096, 32), (4096, 48)}
    g = full[full.phase == "2g"]
    assert len(g) == 18 and set(g.data_fraction) == {0.25, 0.5} and set(g.projection) == {"none"}
    assert set(full[~full.phase.isin(("2f",))].projection) == {"none"}
    assert list(full.iloc[139:172].phase.unique()) == ["2f", "2g"]


def test_remove_component_matches_dense_projection():
    from medinterp.sweep import remove_component, top_component
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 6)) * np.array([5.0, 1, 1, 1, 1, 1])
    cov = torch.from_numpy(np.cov(x, rowvar=False))
    u = top_component(cov)
    assert abs(abs(u[0]) - 1) < 0.05            # the planted high-variance axis
    out = remove_component(x.astype(np.float32).copy(), u, rows_per_chunk=64)
    assert np.allclose(out @ u, 0, atol=1e-4)
    t = remove_component(torch.from_numpy(x.astype(np.float32)), u, rows_per_chunk=64)
    assert np.allclose(t.numpy(), out, atol=1e-5)


def test_phase_3_rows(monkeypatch):
    import medinterp.sweep as sweep
    from medinterp.sweep import CORE_SETTINGS
    full = sweep_table(full=True)
    assert set(full.iloc[:172].architecture) == {"batchtopk"}
    assert set(full.iloc[172:192].phase) == {"3m"}
    grid = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)
    monkeypatch.setattr(sweep, "RELU_L1_GRID", grid)
    with_grid = sweep_table(full=True)
    assert with_grid.iloc[:192].equals(full.iloc[:192])
    c = with_grid[with_grid.phase == "3c"]
    assert len(c) == 16 and set(c.architecture) == {"relu_l1"} and list(c.index) == list(range(192, 208))
    assert set(c.steps) == {20_000} and (c.k == c.d_sae).all() and set(c.init_seed) == {0}
    assert sorted(set(c.l1_coeff)) == sorted(grid)
    m = full[full.phase == "3m"]
    assert set(m.architecture) == {"matryoshka"} and set(m.steps) == {80_000} and set(m.l1_coeff) == {2e-4}
    assert set(zip(m.d_sae, m.k)) == set(CORE_SETTINGS) and (m.groupby("setting").size() == 5).all()


def test_relu_coefficients_and_main_rows():
    from medinterp.sweep import CORE_SETTINGS, pick_l1_coefficients, relu_main_rows
    cal = pd.DataFrame({"d_sae": [1024] * 4 + [4096] * 2, "l1_coeff": [0.1, 0.05, 0.01, 0.005, 0.05, 0.01],
                        "mean_l0": [6.0, 10.0, 15.0, 40.0, 20.0, 44.0]})
    picked = pick_l1_coefficients(cal)
    # k = 8: 6 and 10 are equally close, the sparser (larger) coefficient wins.
    assert picked == {(1024, 8): 0.1, (1024, 16): 0.01, (1024, 32): 0.005, (4096, 32): 0.05}
    rows = pd.DataFrame(relu_main_rows(picked))
    assert len(rows) == 20 and set(rows.architecture) == {"relu_l1"} and set(rows.steps) == {80_000}
    assert set(zip(rows.d_sae, rows.k)) == set(CORE_SETTINGS) and rows.run_id.is_unique
    with pytest.raises(ValueError):
        pick_l1_coefficients(cal[cal.d_sae == 1024])


def test_rows_without_architecture_column_are_batchtopk(tmp_path):
    from medinterp.sweep import row_architecture, train_config
    row = _row()
    assert "architecture" not in row and row_architecture(row) == "batchtopk"
    assert train_config(row).architecture == "batchtopk"
    assert train_config(_row(architecture="matryoshka")).architecture == "matryoshka"


@pytest.mark.parametrize("architecture", ["matryoshka", "relu_l1"])
def test_phase_3_runs_pause_resume_and_save(tmp_path, architecture):
    data, rows = _data(), {"a": np.arange(400), "b": np.arange(300)}
    quiet = lambda *_: None
    row = _row(architecture=architecture, d_sae=64)
    train_run(row, data, rows, tmp_path / "full", "cpu", time.time() + 3600, quiet, state_every=10)
    train_run(row, data, rows, tmp_path / "resumed", "cpu", 0.0, quiet, state_every=10)
    train_run(row, data, rows, tmp_path / "resumed", "cpu", time.time() + 3600, quiet, state_every=10)
    a = TopKSAE.from_checkpoint(tmp_path / "full/toy_i0_b0/step_40")
    b = TopKSAE.from_checkpoint(tmp_path / "resumed/toy_i0_b0/step_40")
    assert torch.equal(a.W_dec, b.W_dec) and torch.equal(a.W_enc, b.W_enc)
    cfg = json.loads((tmp_path / "full/toy_i0_b0/step_40/cfg.json").read_text())
    assert cfg["architecture"] == architecture
    assert a.k == (64 if architecture == "relu_l1" else 4)
