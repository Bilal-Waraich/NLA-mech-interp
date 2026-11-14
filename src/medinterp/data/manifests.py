"""Per-image manifests for the archived harvest: identity, labels and boxes, in activation row order.

Every manifest row ``i`` describes image ``i`` of that dataset's block in ``activations.safetensors``.
Label CSVs are the public challenge files (see README "Data"); the expected location is
``data/labels/<dataset>/`` (gitignored).

Image identity per dataset:
  chexpert   : full relative ``Path`` from ``train.csv``, recovered by replaying the harvest selection
               (Frontal+PA rows in file order, ``random.Random(42).sample(rows, 10000)``); verified against
               the stored stem and label sequences.
  rsna       : patientId (stored stem).
  siim_covid : image id (stored stem); study id from ``train_image_level.csv``.
  tb         : file stem ``CHNCXR_XXXX_{0,1}`` (suffix 1 = TB).
"""
from __future__ import annotations

import ast
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .archive import ArchivedActivations

DEFAULT_LABEL_DIR = Path(__file__).resolve().parents[3] / "data/labels"

CHEXPERT_LABELS = (
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
)
# Harvest-time mapping (thesis datasets/chexpert_dataset.py), kept so the dominant label reproduces.
_CHEXPERT_TO_GROUP = {
    "No Finding": 5, "Enlarged Cardiomediastinum": 2, "Cardiomegaly": 2, "Lung Opacity": 0,
    "Lung Lesion": 0, "Edema": 4, "Consolidation": 0, "Pneumonia": 0, "Atelectasis": 3,
    "Pneumothorax": 6, "Pleural Effusion": 1, "Pleural Other": 1, "Fracture": 6, "Support Devices": 6,
}
SIIM_STUDY_LABELS = (
    "Negative for Pneumonia", "Typical Appearance", "Indeterminate Appearance", "Atypical Appearance",
)


def _chexpert_value(raw: str) -> float:
    """CheXpert stores '1.0', '0.0', '-1.0' or blank; return 1, 0, -1 (uncertain) or NaN (blank)."""
    raw = raw.strip()
    return float(raw) if raw else np.nan


def chexpert_manifest(label_dir: Path = DEFAULT_LABEL_DIR, seed: int = 42, n: int = 10_000) -> pd.DataFrame:
    """Replay the harvest selection and return one row per harvested CheXpert image.

    Columns: path, stem, group (harvest dominant label), sex, age, and the 14 raw label columns
    (1 / 0 / -1 uncertain / NaN not mentioned; the raw coding is kept so later analyses choose
    their own uncertainty policy).
    """
    rows = []
    with open(label_dir / "chexpert/train.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Frontal/Lateral"].strip() != "Frontal" or r["AP/PA"].strip() != "PA":
                continue
            # Dominant group exactly as at harvest: first positive in column order, airspace overrides.
            score, group = -1, 5
            for col in CHEXPERT_LABELS:
                if r[col].strip() in ("1.0", "1"):
                    g = _CHEXPERT_TO_GROUP[col]
                    if score < 1 or g == 0:
                        group, score = g, 1
            rows.append({
                "path": r["Path"], "stem": Path(r["Path"]).stem, "group": group,
                "sex": r["Sex"], "age": float(r["Age"]),
                **{c: _chexpert_value(r[c]) for c in CHEXPERT_LABELS},
            })
    # The harvester sampled from the row list in file order; same RNG + order => same selection.
    selected = random.Random(seed).sample(rows, n)
    return pd.DataFrame(selected).reset_index(names="image_index")


def rsna_manifest(stems: list[str], label_dir: Path = DEFAULT_LABEL_DIR) -> pd.DataFrame:
    """RSNA rows in harvest order with Target, detailed class and boxes in original 1024x1024 pixels."""
    labels = pd.read_csv(label_dir / "rsna/stage_2_train_labels.csv")
    detail = pd.read_csv(label_dir / "rsna/stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    # One patient can have several rows (one per box); collapse to target + list of boxes.
    boxes = (
        labels.dropna(subset=["x"])
        .assign(box=lambda d: list(zip(d.x, d.y, d.x + d.width, d.y + d.height)))
        .groupby("patientId")["box"].apply(list)
    )
    target = labels.groupby("patientId")["Target"].max()
    out = pd.DataFrame({"stem": stems})
    out["target"] = out.stem.map(target)
    out["detailed_class"] = out.stem.map(detail.set_index("patientId")["class"])
    out["boxes_xyxy_px"] = out.stem.map(boxes).apply(lambda b: b if isinstance(b, list) else [])
    out["group"] = np.where(out.target == 1, 0, 5)  # airspace_opacity vs no_finding
    return out.reset_index(names="image_index")


def siim_manifest(stems: list[str], label_dir: Path = DEFAULT_LABEL_DIR) -> pd.DataFrame:
    """SIIM rows in harvest order with study id, study-level labels and boxes in original pixels."""
    img = pd.read_csv(label_dir / "siim-covid/train_image_level.csv")
    img["stem"] = img["id"].str.replace("_image", "", regex=False)
    study = pd.read_csv(label_dir / "siim-covid/train_study_level.csv")
    study["StudyInstanceUID"] = study["id"].str.replace("_study", "", regex=False)

    def parse_boxes(raw) -> list[tuple[float, float, float, float]]:
        # 'boxes' holds a Python-literal list of {'x','y','width','height'} dicts, or NaN.
        if not isinstance(raw, str):
            return []
        return [(b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]) for b in ast.literal_eval(raw)]

    img["boxes_xyxy_px"] = img["boxes"].apply(parse_boxes)
    merged = img.merge(study.drop(columns="id"), on="StudyInstanceUID", how="left")
    out = pd.DataFrame({"stem": stems}).merge(
        merged[["stem", "StudyInstanceUID", "boxes_xyxy_px", *SIIM_STUDY_LABELS]], on="stem", how="left")
    out["group"] = np.where(out["Negative for Pneumonia"] == 1, 5, 0)
    return out.reset_index(names="image_index")


def tb_manifest(stems: list[str]) -> pd.DataFrame:
    """TB rows in harvest order; label from the filename suffix (_1 = TB)."""
    out = pd.DataFrame({"stem": stems})
    out["tb"] = out.stem.str.endswith("_1").astype(int)
    out["group"] = np.where(out.tb == 1, 0, 5)
    return out.reset_index(names="image_index")


def build_manifests(acts: ArchivedActivations, label_dir: Path = DEFAULT_LABEL_DIR) -> dict[str, pd.DataFrame]:
    """Return manifests for all four datasets, aligned to activation row order."""
    return {
        "chexpert": chexpert_manifest(label_dir),
        "rsna": rsna_manifest(acts.image_stems["rsna"], label_dir),
        "siim_covid": siim_manifest(acts.image_stems["siim_covid"], label_dir),
        "tb": tb_manifest(acts.image_stems["tb"]),
    }
