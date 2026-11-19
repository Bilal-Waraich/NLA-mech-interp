"""Decoder diagnostic of protocol section 5: decoder pairwise cosine of the phase 1 SAEs, per setting.

Reads only decoder weights, so it needs no activations. Runs that have not finished under ``--runs`` are
skipped and counted, which lets each Kaggle output folder be processed on its own.

Outputs in ``--out``: ``decoder_diagnostic_runs.csv`` (one row per run), ``decoder_diagnostic.csv`` (one row per
setting), ``config.json`` and ``run.log``. With ``--merge`` the per-run tables of several outputs are combined
into one per-setting table, and no decoder is read.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/sweep_decoder_diagnostic.py --runs results/sweep_v1/runs
      ~/.venvs/nla-mech-interp/bin/python scripts/sweep_decoder_diagnostic.py --merge A/decoder_diagnostic_runs.csv \
          B/decoder_diagnostic_runs.csv --fve results/sweep_v1/eval/fve.csv
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.sweep_metrics import decoder_pairwise_cosine
from medinterp.runinfo import write_config
from medinterp.sweep import BASE_STEPS, run_dir_checkpoint, summarise_decoder_diagnostic

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
PHASES = ("1a", "1b")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/sweep_v1.csv")
    ap.add_argument("--runs", type=Path, default=ROOT / "results/sweep_v1/runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/sweep_v1/decoder_diagnostic")
    ap.add_argument("--step", type=int, default=BASE_STEPS)
    ap.add_argument("--fve", type=Path, help="fve.csv of the evaluation, to report dead latents next to the metric")
    ap.add_argument("--merge", nargs="+", type=Path, metavar="RUNS_CSV", help="combine these per-run tables")
    ap.add_argument("--force", action="store_true", help="recompute even if the final output exists")
    args = ap.parse_args()

    final = args.out / "decoder_diagnostic.csv"
    if final.exists() and not args.force:
        print(f"{final} exists; nothing to do")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    fve = pd.read_csv(args.fve) if args.fve else None
    write_config(args.out, {"step": args.step, "phases": PHASES, "config": str(args.config), "runs": str(args.runs),
                            "merged": [str(f) for f in args.merge or []], "fve": str(args.fve)}, REPO)

    if args.merge:
        runs = pd.concat([pd.read_csv(f) for f in args.merge], ignore_index=True)
        if runs.run_id.duplicated().any():
            sys.exit("the merged tables contain the same run twice")
    else:
        table = pd.read_csv(args.config)
        table = table[table.phase.isin(PHASES)]
        rows, skipped = [], 0
        with open(args.out / "run.log", "a") as log:
            for row in table.itertuples():
                if not (args.runs / row.run_id / "DONE.json").exists():
                    skipped += 1
                    continue
                started = time.time()
                decoder = TopKSAE.from_checkpoint(run_dir_checkpoint(args.runs, row.run_id, args.step)).W_dec
                c_dec = decoder_pairwise_cosine(decoder)
                rows.append({"run_id": row.run_id, "d_sae": row.d_sae, "k": row.k, "step": args.step, "c_dec": c_dec})
                line = f"{row.run_id}: c_dec {c_dec:.5f} ({time.time() - started:.1f} s)"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            log.write(f"{len(rows)} runs read, {skipped} not finished here\n")
        if not rows:
            sys.exit(f"no finished phase 1 run found under {args.runs}")
        runs = pd.DataFrame(rows)
        runs.to_csv(args.out / "decoder_diagnostic_runs.csv", index=False)

    summary = summarise_decoder_diagnostic(runs, fve)
    summary.to_csv(final, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
