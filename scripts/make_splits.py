"""Write the split file ``data/splits/v1.csv`` for the thesis-harvest images (protocol section 2).

The file is written once. Until the SIIM PatientID is known (WP2a header pass) the SIIM group key is the
study id and the file is marked provisional in ``v1.json``. It is regenerated exactly once, with
``--final --siim-patients <csv>``; only then is its SHA-256 entered into the protocol. Any other attempt
to overwrite is refused.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/make_splits.py
      ~/.venvs/nla-mech-interp/bin/python scripts/make_splits.py --final --siim-patients patients.csv
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from medinterp.data.archive import DEFAULT_RUN_DIR, ArchivedActivations
from medinterp.data.labels import check_counts, label_counts, opacity_labels
from medinterp.data.manifests import DEFAULT_LABEL_DIR, build_manifests
from medinterp.data.splits import SPLITS, build_split_table, stratification_labels

ROOT = Path(os.environ.get("MEDINTERP_DATA", Path(__file__).resolve().parents[1]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", action="store_true",
                    help="replace the provisional file once, with SIIM groups from --siim-patients")
    ap.add_argument("--siim-patients", type=Path,
                    help="CSV with columns image_id (SIIM image id) and patient_id (DICOM PatientID)")
    args = ap.parse_args()

    meta_path = args.out.with_suffix(".json")
    if args.final and args.siim_patients is None:
        sys.exit("--final needs --siim-patients")
    if args.out.exists():
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not (args.final and meta.get("provisional") is True):
            sys.exit(f"refusing to overwrite {args.out}: the split file is written once "
                     "(the only exception is the single --final regeneration of a provisional file)")

    manifests = build_manifests(ArchivedActivations(args.run_dir), args.label_dir)
    check_counts(manifests)
    print(label_counts(manifests).to_string(index=False))
    nih = pd.read_csv(args.label_dir / "rsna/rsna_to_nih.csv", dtype=str)
    siim = None
    if args.siim_patients is not None:
        p = pd.read_csv(args.siim_patients, dtype=str)
        siim = p.set_index("image_id")["patient_id"]

    strata = stratification_labels(manifests, opacity_labels(manifests))
    table = build_split_table(manifests, strata, nih, siim, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    counts = {d: {s: int(n) for s, n in part.split.value_counts().reindex(SPLITS).items()}
              for d, part in table.groupby("dataset")}
    groups = {d: {"n_groups": int(part.group_id.nunique()),
                  "n_groups_multi": int((part.group_id.value_counts() > 1).sum())}
              for d, part in table.groupby("dataset")}
    meta = {
        "file": args.out.name, "sha256": sha256(args.out), "provisional": not args.final,
        "seed": args.seed, "fractions": [0.7, 0.1, 0.2],
        "siim_group_key": "dicom_patient_id" if siim is not None else "study_id",
        "stratified_by": "dataset and primary label (opacity; TB label for Shenzhen)",
        "counts": counts, "groups": groups, "git_revision": git_revision(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
