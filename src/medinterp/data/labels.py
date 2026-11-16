"""Image labels of the evaluation protocol (section 2) and the thesis grouping kept as a variant.

``opacity`` is the primary contrast, "airspace opacity present vs normal", with the same meaning in every
dataset: 1 = positive, 0 = negative, NaN = left out of the contrast. The per-dataset rules follow the
protocol's label table; they are implemented here and not restated.

``thesis_group`` is the 0-6 grouping of ``manifests.py`` (harvest-time canonical groups), so old and new
numbers can be shown side by side.

``check_counts`` asserts the counts that the build spec (section 4) records for the thesis harvest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHEXPERT_OPACITY_LABELS = ("Lung Opacity", "Consolidation", "Pneumonia")
UNCERTAIN_POLICIES = ("exclude", "positive_wins")

# Thesis-harvest counts recorded in docs/build-spec.md section 4 (checked 2026-11-16).
EXPECTED_COUNTS = {
    "chexpert_thesis_group": {0: 4116, 5: 2839, 2: 859, 6: 794, 1: 750, 3: 457, 4: 185},
    "rsna_detailed_class": {"Lung Opacity": 2260, "No Lung Opacity / Not Normal": 4435, "Normal": 3305},
    "siim_images": 6334,
    "siim_studies": 6054,
    "tb": {1: 336, 0: 326},
}


def chexpert_opacity(man: pd.DataFrame, uncertain: str = "exclude") -> np.ndarray:
    """Opacity label of CheXpert images.

    ``uncertain="exclude"`` (default) leaves out every image with an uncertain value on any of the
    opacity labels, even if another of them is positive: the literal reading of the protocol table.
    ``"positive_wins"`` keeps an image as positive when one opacity label is certain-positive and another
    is uncertain (a variant, not the protocol rule).
    Raises ValueError for an unknown policy.
    """
    if uncertain not in UNCERTAIN_POLICIES:
        raise ValueError(f"uncertain must be one of {UNCERTAIN_POLICIES}, got {uncertain!r}")
    vals = man[list(CHEXPERT_OPACITY_LABELS)]
    positive = (vals == 1).any(axis=1).to_numpy()
    unsure = (vals == -1).any(axis=1).to_numpy()
    normal = (man["No Finding"] == 1).to_numpy()
    out = np.full(len(man), np.nan)
    out[normal & ~positive & ~unsure] = 0
    if uncertain == "exclude":
        out[positive & ~unsure] = 1
    else:
        out[positive] = 1
    return out


def rsna_opacity(man: pd.DataFrame) -> np.ndarray:
    """1 = Lung Opacity (target 1), 0 = "Normal", NaN = "No Lung Opacity / Not Normal"."""
    out = np.full(len(man), np.nan)
    out[(man["target"] == 1).to_numpy()] = 1
    out[(man["detailed_class"] == "Normal").to_numpy()] = 0
    return out


def siim_opacity(man: pd.DataFrame) -> np.ndarray:
    """1 = the image has an opacity box; 0 = study negative for pneumonia and no box; else NaN."""
    has_box = (man["boxes_xyxy_px"].str.len() > 0).to_numpy()
    negative_study = (man["Negative for Pneumonia"] == 1).to_numpy()
    out = np.full(len(man), np.nan)
    out[has_box] = 1
    out[negative_study & ~has_box] = 0
    return out


def opacity_labels(manifests: dict[str, pd.DataFrame], uncertain: str = "exclude") -> dict[str, np.ndarray]:
    """``{dataset: float[n_images]}`` with 1 / 0 / NaN; Shenzhen TB is not part of this contrast (all NaN)."""
    return {
        "chexpert": chexpert_opacity(manifests["chexpert"], uncertain),
        "rsna": rsna_opacity(manifests["rsna"]),
        "siim_covid": siim_opacity(manifests["siim_covid"]),
        "tb": np.full(len(manifests["tb"]), np.nan),
    }


def thesis_groups(manifests: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """The 0-6 thesis grouping per image (as used by the Stage 0 scripts under ``--protocol thesis``)."""
    return {d: m["group"].to_numpy() for d, m in manifests.items()}


def label_counts(manifests: dict[str, pd.DataFrame], uncertain: str = "exclude") -> pd.DataFrame:
    """Tidy table ``dataset, label, value, n`` of the opacity label and the thesis grouping."""
    rows = []
    opac = opacity_labels(manifests, uncertain)
    for d, m in manifests.items():
        o = pd.Series(opac[d]).map({1.0: "positive", 0.0: "negative"}).fillna("left_out")
        for value, n in o.value_counts().items():
            rows.append({"dataset": d, "label": "opacity", "value": value, "n": int(n)})
        for value, n in m["group"].value_counts().items():
            rows.append({"dataset": d, "label": "thesis_group", "value": str(value), "n": int(n)})
    return pd.DataFrame(rows).sort_values(["dataset", "label", "value"]).reset_index(drop=True)


def check_counts(manifests: dict[str, pd.DataFrame]) -> None:
    """Assert the recorded thesis-harvest counts; raises AssertionError naming the first mismatch."""
    c, r, s, t = (manifests[d] for d in ("chexpert", "rsna", "siim_covid", "tb"))
    got = c["group"].value_counts().to_dict()
    assert got == EXPECTED_COUNTS["chexpert_thesis_group"], f"CheXpert thesis groups {got}"
    got = r["detailed_class"].value_counts().to_dict()
    assert got == EXPECTED_COUNTS["rsna_detailed_class"], f"RSNA classes {got}"
    assert int((r["target"] == 1).sum()) == EXPECTED_COUNTS["rsna_detailed_class"]["Lung Opacity"]
    assert len(s) == EXPECTED_COUNTS["siim_images"], f"SIIM images {len(s)}"
    assert s["StudyInstanceUID"].nunique() == EXPECTED_COUNTS["siim_studies"], "SIIM studies"
    got = t["tb"].value_counts().to_dict()
    assert got == EXPECTED_COUNTS["tb"], f"TB {got}"
