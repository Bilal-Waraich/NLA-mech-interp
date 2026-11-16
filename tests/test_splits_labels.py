"""Splits keep groups together and hit their proportions; labels reproduce the recorded counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.data.manifests import DEFAULT_LABEL_DIR
from medinterp.data.splits import SPLITS, assign_splits


def _synthetic(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d, n_groups in (("a", 3000), ("b", 800)):
        for g in range(n_groups):
            size = int(rng.choice([1, 1, 1, 2, 3, 8, 20]))
            lab = rng.choice([1.0, 0.0, np.nan])
            rows += [{"dataset": d, "group_id": f"{d}/{g}", "opacity": lab} for _ in range(size)]
    return pd.DataFrame(rows)


def _check(table: pd.DataFrame, split: pd.Series, tol: float = 0.02) -> None:
    per_group = pd.DataFrame({"g": table.group_id, "s": split}).groupby("g").s.nunique()
    assert (per_group == 1).all(), "a group is in two splits"
    for d, part in split.groupby(table.dataset):
        share = part.value_counts(normalize=True).reindex(SPLITS).fillna(0).to_numpy()
        assert np.abs(share - [0.7, 0.1, 0.2]).max() <= tol, (d, share)


def test_split_groups_disjoint_and_proportions() -> None:
    t = _synthetic()
    _check(t, assign_splits(t, seed=0))


def test_split_deterministic_and_seed_dependent() -> None:
    t = _synthetic()
    a, b, c = assign_splits(t, seed=0), assign_splits(t, seed=0), assign_splits(t, seed=1)
    assert a.equals(b) and not a.equals(c)
    # Row order does not change any image's split.
    shuffled = t.sample(frac=1, random_state=3)
    assert assign_splits(shuffled, seed=0).sort_index().equals(a)


def test_split_stratifies_by_label() -> None:
    t = _synthetic()
    s = assign_splits(t, seed=0)
    for lab in (1.0, 0.0):
        part = s[(t.opacity == lab) & (t.dataset == "a")]
        assert abs((part == "train").mean() - 0.7) < 0.03


def test_bad_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        assign_splits(_synthetic(), fractions=(0.5, 0.1, 0.2))


# ------------------------------------------------------------------ on the thesis harvest

needs_data = pytest.mark.skipif(
    not (DEFAULT_RUN_DIR / "activations.safetensors").exists()
    or not (DEFAULT_LABEL_DIR / "rsna/rsna_to_nih.csv").exists(),
    reason="archive or label CSVs not present")


@pytest.fixture(scope="module")
def manifests():
    from medinterp.data.archive import ArchivedActivations
    from medinterp.data.manifests import build_manifests
    return build_manifests(ArchivedActivations())


@needs_data
def test_label_counts_match_the_record(manifests) -> None:
    from medinterp.data.labels import check_counts, label_counts
    check_counts(manifests)
    counts = label_counts(manifests)
    print("\n" + counts.to_string())
    opac = counts[counts.label == "opacity"].set_index(["dataset", "value"]).n
    # Totals per dataset equal the number of images; TB is entirely outside the opacity contrast.
    assert opac.groupby(level=0).sum().to_dict() == {"chexpert": 10000, "rsna": 10000,
                                                     "siim_covid": 6334, "tb": 662}
    assert opac[("tb", "left_out")] == 662
    assert opac[("rsna", "positive")] == 2260 and opac[("rsna", "negative")] == 3305
    assert opac[("siim_covid", "positive")] == 4294 and opac[("siim_covid", "negative")] == 1736


@needs_data
def test_chexpert_uncertain_policy(manifests) -> None:
    from medinterp.data.labels import chexpert_opacity
    strict = chexpert_opacity(manifests["chexpert"])
    loose = chexpert_opacity(manifests["chexpert"], uncertain="positive_wins")
    assert np.nansum(loose == 1) == 3820 and np.nansum(strict == 1) == 3820 - 1323
    assert np.nansum(strict == 0) == np.nansum(loose == 0) == 1872
    with pytest.raises(ValueError):
        chexpert_opacity(manifests["chexpert"], uncertain="ignore")


@needs_data
def test_real_split(manifests) -> None:
    from medinterp.data.labels import opacity_labels
    from medinterp.data.splits import build_split_table, group_ids, stratification_labels
    nih = pd.read_csv(DEFAULT_LABEL_DIR / "rsna/rsna_to_nih.csv", dtype=str)
    groups = group_ids(manifests, nih)
    # Group counts recorded in protocol section 2 and build spec section 4.
    for d, n, multi in (("chexpert", 8443, 1074), ("rsna", 5908, 1753), ("siim_covid", 6054, 232)):
        vc = groups[d].value_counts()
        assert (len(vc), int((vc > 1).sum())) == (n, multi), d
    strata = stratification_labels(manifests, opacity_labels(manifests))
    table = build_split_table(manifests, strata, nih, seed=0)
    assert len(table) == 26_996 and table.image_id.is_unique
    _check(table, table.split)
    again = build_split_table(manifests, strata, nih, seed=0)
    assert again.equals(table)


@needs_data
def test_make_splits_refuses_to_overwrite(tmp_path) -> None:
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/make_splits.py"
    out = tmp_path / "v1.csv"
    first = subprocess.run([sys.executable, str(script), "--out", str(out)], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    digest = out.read_bytes()
    for extra in ([], ["--final"]):
        again = subprocess.run([sys.executable, str(script), "--out", str(out), *extra],
                               capture_output=True, text=True)
        assert again.returncode != 0
    assert out.read_bytes() == digest
