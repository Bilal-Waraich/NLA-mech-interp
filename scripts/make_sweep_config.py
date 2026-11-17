"""Write the sweep's run table ``configs/sweep_v1.csv`` from the protocol's phase table (section 5).

Without ``--headline`` the table holds phases 1a, 1b and 1n. Once ``scripts/sweep_report.py`` has fixed
the headline setting (``results/sweep_v1/headline.json``), run again with ``--headline`` to append phase 2.
Existing rows keep their row numbers: phase 2 rows are only appended, and a table whose phase-1 rows
differ from the regenerated ones is refused.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/make_sweep_config.py
      ~/.venvs/nla-mech-interp/bin/python scripts/make_sweep_config.py --headline results/sweep_v1/headline.json
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pathlib import Path

import pandas as pd

from medinterp.sweep import sweep_table

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "configs/sweep_v1.csv")
    ap.add_argument("--headline", type=Path, help="headline.json written by sweep_report.py")
    args = ap.parse_args()

    k = json.loads(args.headline.read_text())["headline_k"] if args.headline else None
    table = sweep_table(k)
    if args.out.exists():
        old = pd.read_csv(args.out)
        shared = table.iloc[:len(old)].reset_index(drop=True)
        if len(old) > len(table) or not old[["row", "run_id"]].equals(shared[["row", "run_id"]]):
            sys.exit(f"{args.out} has rows that the regenerated table does not keep; refusing to overwrite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    units = table.groupby("phase").run_units.sum()
    print(f"wrote {args.out}: {len(table)} runs, {table.run_units.sum():.2f} run units")
    print(units.to_string())


if __name__ == "__main__":
    main()
