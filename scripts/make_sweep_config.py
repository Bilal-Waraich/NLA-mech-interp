"""Write the sweep's run table ``configs/sweep_v1.csv`` from the protocol's phase table (section 5).

Without ``--full`` the table holds phases 1a, 1b and 1n. With it, phase 2 and then phase 1c (the k extension)
are appended. Existing rows keep their row numbers, and a table whose rows differ from the regenerated ones
is refused.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/make_sweep_config.py --full
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path

import pandas as pd

from medinterp.sweep import sweep_table

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "configs/sweep_v1.csv")
    ap.add_argument("--full", action="store_true", help="append phase 2 and phase 1c")
    args = ap.parse_args()

    table = sweep_table(full=args.full)
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
