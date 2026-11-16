"""Grouped, stratified train / val / test split of the harvested images (protocol section 2).

Every image gets an ``image_id`` (``<dataset>/<native id>``) and a ``group_id`` (patient or study key,
prefixed with the dataset name). No group is ever split across train, val and test.

Group keys (build spec section 4):
  chexpert    patient id from the path (``.../train/patientXXXXX/studyN/viewN_frontal.jpg``)
  rsna        NIH patient id from RSNA's official mapping (``data/labels/rsna/rsna_to_nih.csv``)
  siim_covid  DICOM PatientID once WP2a has read it from the files; until then the study id
  tb          the image itself (an assumption: the release has no patient id)

Assignment: within each (dataset, stratum) cell the groups are shuffled with a seeded generator and
each group goes to the split furthest below its target share of images in that cell. The stratum of a
group is its primary-label status (any positive image / any negative image / neither). Greedy filling keeps
the realised shares close to the targets even when groups differ in size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def image_ids(manifests: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """``image_id`` per dataset, in harvest row order."""
    return {
        "chexpert": "chexpert/" + manifests["chexpert"]["path"],
        "rsna": "rsna/" + manifests["rsna"]["stem"],
        "siim_covid": "siim_covid/" + manifests["siim_covid"]["stem"],
        "tb": "tb/" + manifests["tb"]["stem"],
    }


def group_ids(manifests: dict[str, pd.DataFrame], rsna_to_nih: pd.DataFrame,
              siim_patient: pd.Series | None = None) -> dict[str, pd.Series]:
    """``group_id`` per dataset, in harvest row order.

    Parameters:
      rsna_to_nih   RSNA mapping with columns ``patientId, nih_patient_id`` (read as strings)
      siim_patient  optional Series mapping SIIM image id (stem) to DICOM PatientID; without it the
                    study id is the group key. An image whose PatientID is empty falls back to its study.
    Raises ValueError if an RSNA image has no NIH patient id or a CheXpert path has no patient part.
    """
    chex = manifests["chexpert"]["path"].str.split("/").str[2]
    if not chex.str.startswith("patient").all():
        raise ValueError("CheXpert path without a patientXXXXX component")
    rsna = manifests["rsna"]["stem"].map(rsna_to_nih.set_index("patientId")["nih_patient_id"])
    if rsna.isna().any():
        raise ValueError(f"{int(rsna.isna().sum())} RSNA images without an NIH patient id")
    siim = manifests["siim_covid"]
    study = "study:" + siim["StudyInstanceUID"]
    if siim_patient is not None:
        pid = siim["stem"].map(siim_patient)
        empty = pid.isna() | (pid.astype(str).str.strip() == "")
        siim_key = ("patient:" + pid.astype(str)).where(~empty, study)
    else:
        siim_key = study
    return {
        "chexpert": "chexpert/" + chex,
        "rsna": "rsna/" + rsna.astype(str),
        "siim_covid": "siim_covid/" + siim_key,
        "tb": "tb/" + manifests["tb"]["stem"],
    }


def _group_stratum(opacity: pd.Series, group: pd.Series) -> pd.Series:
    """Per group: "pos" if any image is positive, else "neg" if any is negative, else "none"."""
    o = pd.DataFrame({"g": group.to_numpy(), "o": opacity.to_numpy()})
    has_pos = o.assign(p=o.o == 1).groupby("g").p.any()
    has_neg = o.assign(n=o.o == 0).groupby("g").n.any()
    return pd.Series(np.where(has_pos, "pos", np.where(has_neg, "neg", "none")), index=has_pos.index)


def assign_splits(table: pd.DataFrame, fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
                  seed: int = 0) -> pd.Series:
    """Return the split of every row of ``table`` (columns ``dataset, group_id, opacity``).

    Deterministic for a given seed and row content (row order does not matter). Complexity:
    O(n_groups log n_groups) per cell.
    Raises ValueError if the fractions do not sum to 1.
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    target = np.asarray(fractions)
    group_split: dict[str, str] = {}
    for d_index, dataset in enumerate(sorted(table["dataset"].unique())):
        part = table[table["dataset"] == dataset]
        sizes = part.groupby("group_id").size()
        strata = _group_stratum(part["opacity"], part["group_id"])
        for s_index, stratum in enumerate(("neg", "none", "pos")):
            groups = np.sort(strata.index[strata == stratum].to_numpy())
            if groups.size == 0:
                continue
            # A fresh generator per cell: adding data to one dataset cannot reshuffle another.
            rng = np.random.default_rng([seed, d_index, s_index])
            filled = np.zeros(3)
            for g in groups[rng.permutation(groups.size)]:
                n = sizes[g]
                deficit = target * (filled.sum() + n) - filled
                choice = int(np.argmax(deficit))
                filled[choice] += n
                group_split[g] = SPLITS[choice]
    return table["group_id"].map(group_split)


def stratification_labels(manifests: dict[str, pd.DataFrame], opacity: dict[str, np.ndarray]
                          ) -> dict[str, np.ndarray]:
    """The primary label per dataset: opacity, except Shenzhen, whose own contrast is TB vs normal."""
    return {**opacity, "tb": manifests["tb"]["tb"].to_numpy(dtype=float)}


def build_split_table(manifests: dict[str, pd.DataFrame], strata_labels: dict[str, np.ndarray],
                      rsna_to_nih: pd.DataFrame, siim_patient: pd.Series | None = None,
                      seed: int = 0) -> pd.DataFrame:
    """Split file content: ``image_id, dataset, group_id, split`` in dataset and row order.

    ``strata_labels`` holds 1 / 0 / NaN per image (see ``stratification_labels``).
    """
    ids, groups = image_ids(manifests), group_ids(manifests, rsna_to_nih, siim_patient)
    table = pd.concat([pd.DataFrame({"image_id": ids[d].to_numpy(), "dataset": d,
                                     "group_id": groups[d].to_numpy(), "opacity": strata_labels[d]})
                       for d in manifests], ignore_index=True)
    if not table["image_id"].is_unique:
        raise ValueError("image ids are not unique")
    table["split"] = assign_splits(table, seed=seed)
    return table[["image_id", "dataset", "group_id", "split"]]


def load_split(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def split_masks(split_table: pd.DataFrame, manifests: dict[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    """``{dataset: {"train"|"val"|"test": bool[n_images]}}`` aligned to manifest row order.

    Raises ValueError if an image of the manifests is missing from the split table.
    """
    lookup = split_table.set_index("image_id")["split"]
    out = {}
    for d, ids in image_ids(manifests).items():
        s = ids.map(lookup)
        if s.isna().any():
            raise ValueError(f"{d}: {int(s.isna().sum())} images missing from the split table")
        out[d] = {name: (s == name).to_numpy() for name in SPLITS}
    return out


def split_groups(split_table: pd.DataFrame, manifests: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """``{dataset: group_id[n_images]}`` aligned to manifest row order (for cluster bootstraps)."""
    lookup = split_table.set_index("image_id")["group_id"]
    return {d: ids.map(lookup).to_numpy() for d, ids in image_ids(manifests).items()}
