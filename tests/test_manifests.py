"""Manifests must align with the archived activation order and stored harvest labels."""
from __future__ import annotations

import numpy as np
import pytest

from medinterp.data.archive import DEFAULT_RUN_DIR, ArchivedActivations
from medinterp.data.manifests import DEFAULT_LABEL_DIR, build_manifests

pytestmark = pytest.mark.skipif(
    not (DEFAULT_RUN_DIR / "activations.safetensors").exists()
    or not (DEFAULT_LABEL_DIR / "chexpert/train.csv").exists(),
    reason="archive or label CSVs not present",
)


@pytest.fixture(scope="module")
def setup():
    acts = ArchivedActivations()
    return acts, build_manifests(acts)


def test_every_dataset_aligned(setup) -> None:
    acts, man = setup
    for name, df in man.items():
        assert len(df) == acts.blocks[name].n_images
        assert df.stem.tolist() == acts.image_stems[name], name


def test_chexpert_reconstruction(setup) -> None:
    """Replayed selection reproduces the stored stems (above) AND every stored dominant label."""
    acts, man = setup
    df = man["chexpert"]
    assert df.path.is_unique
    assert (df.group.to_numpy() == acts.image_labels["chexpert"]).all()


def test_labels_match_harvest_where_harvest_was_correct(setup) -> None:
    acts, man = setup
    for name in ("siim_covid", "tb"):
        assert (man[name].group.to_numpy() == acts.image_labels[name]).all(), name


def test_rsna_labels_are_real(setup) -> None:
    """Harvest stored all-zero RSNA labels; the CSV gives the true prevalence (2,260 / 10,000)."""
    _, man = setup
    df = man["rsna"]
    assert df.target.notna().all()
    assert int(df.target.sum()) == 2260
    # Every positive has at least one box, every negative none.
    assert ((df.target == 1) == (df.boxes_xyxy_px.str.len() > 0)).all()


def test_siim_boxes_parse(setup) -> None:
    _, man = setup
    df = man["siim_covid"]
    assert df.StudyInstanceUID.notna().all()
    assert (df.boxes_xyxy_px.str.len() > 0).sum() > 0
