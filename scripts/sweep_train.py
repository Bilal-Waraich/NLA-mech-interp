"""Train rows of the sweep's run table on train-split tokens (protocol section 5), resumable.

Each row becomes ``<out>/<run_id>/`` with ``step_<n>/`` checkpoints; see ``medinterp.sweep``. Finished
runs are skipped. When the time budget is used up, the running row saves its state, ``<out>/STATE.json``
gets ``{"finished": false, ...}`` and the script exits with code 0; the next call continues.

``--devices cuda:0,cuda:1`` splits the unfinished rows between one child process per device (balanced by
run units; a Kaggle GPU session has two T4s) and merges their states.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/sweep_train.py --rows 0:5
      ~/.venvs/nla-mech-interp/bin/python scripts/sweep_train.py --rows 20,21 --time-budget-hours 3
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.data.manifests import DEFAULT_LABEL_DIR
from medinterp.runinfo import git_revision, package_versions, sha256_file
from medinterp.stage0_v1 import load_context
from medinterp.sweep import load_tokens_fp16, parse_rows, subset_train_images, token_rows, train_run

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
# Train-image subsets of phase 2c are drawn with this seed, the same for every run seed, so runs of one
# setting differ only in their own seeds.
SUBSET_SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/sweep_v1.csv")
    ap.add_argument("--rows", required=True, help="'a:b' (half-open) or '3,5,9'")
    ap.add_argument("--out", type=Path, default=ROOT / "results/sweep_v1/runs")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="archived activations")
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--device", help="default: cuda, else mps, else cpu")
    ap.add_argument("--time-budget-hours", type=float, default=10.0)
    ap.add_argument("--state-every", type=int, default=2_000, help="steps between saved training states")
    ap.add_argument("--devices", help="comma-separated devices; one child process each")
    ap.add_argument("--state-file", default="STATE.json")
    args = ap.parse_args()
    if args.devices:
        return run_parallel(args)

    t_start = time.time()
    deadline = t_start + args.time_budget_hours * 3600
    table = pd.read_csv(args.config)
    rows = parse_rows(args.rows, len(table))
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out / "sweep_train.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    todo = [r for r in rows if not (args.out / table.loc[r, "run_id"] / "DONE.json").exists()]
    log(f"rows {args.rows}: {len(rows)} listed, {len(todo)} not finished; device {device}; "
        f"budget {args.time_budget_hours} h")
    state = {"finished": True, "rows": args.rows, "done": [], "paused": None}
    if todo:
        ctx = load_context(args.run_dir, args.label_dir, args.split_file)
        log("loading tokens (float16)")
        data = load_tokens_fp16(ctx.acts)
        train_masks = {d: ctx.masks[d]["train"] for d in data}
        run_info = {"split_sha256": sha256_file(args.split_file), "git_revision": git_revision(REPO),
                    "packages": package_versions(), "config_file": str(args.config),
                    "subset_seed": SUBSET_SEED}
        for r in todo:
            row = table.loc[r]
            images = subset_train_images(train_masks, ctx.groups, float(row.data_fraction), SUBSET_SEED)
            train_rows = {d: token_rows(images[d], ctx.acts.n_patches) for d in data}
            run_dir = args.out / row.run_id

            def run_logger(msg: str, rid=row.run_id, run_dir=run_dir) -> None:
                log(f"{rid} {msg}")
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(run_dir / "train.log", "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

            log(f"row {r} {row.run_id}: train tokens {sum(len(v) for v in train_rows.values())}")
            res = train_run(row, data, train_rows, args.out, device, deadline, run_logger, args.state_every,
                            run_info={**run_info, "n_train_images": {d: int(m.sum()) for d, m in images.items()}})
            log(f"row {r} {row.run_id}: {res.status} at step {res.step} ({res.seconds / 60:.1f} min)")
            if res.status == "paused":
                state.update(finished=False, paused=row.run_id)
                break
            state["done"].append(row.run_id)
    state["elapsed_hours"] = (time.time() - t_start) / 3600
    (args.out / args.state_file).write_text(json.dumps(state, indent=2))
    log(f"state: {state}")


def run_parallel(args) -> None:
    """Split the unfinished rows over the devices (largest first, to the least loaded device) and run one
    child per device; the children share the output directory but write separate state files."""
    table = pd.read_csv(args.config)
    rows = [r for r in parse_rows(args.rows, len(table))
            if not (args.out / table.loc[r, "run_id"] / "DONE.json").exists()]
    devices = args.devices.split(",")
    load, assigned = [0.0] * len(devices), [[] for _ in devices]
    for r in sorted(rows, key=lambda r: -table.loc[r, "run_units"]):
        i = load.index(min(load))
        load[i] += float(table.loc[r, "run_units"])
        assigned[i].append(r)
    args.out.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, (device, mine) in enumerate(zip(devices, assigned)):
        if not mine:
            continue
        cmd = [sys.executable, __file__, "--config", str(args.config), "--rows", ",".join(map(str, sorted(mine))),
               "--out", str(args.out), "--run-dir", str(args.run_dir), "--label-dir", str(args.label_dir),
               "--split-file", str(args.split_file), "--device", device,
               "--time-budget-hours", str(args.time_budget_hours), "--state-every", str(args.state_every),
               "--state-file", f"STATE_{i}.json"]
        print(f"{device}: rows {sorted(mine)} ({load[i]:.2f} run units)", flush=True)
        procs.append((i, subprocess.Popen(cmd)))
    codes = [p.wait() for _, p in procs]
    parts = [json.loads((args.out / f"STATE_{i}.json").read_text()) for i, _ in procs
             if (args.out / f"STATE_{i}.json").exists()]
    state = {"finished": all(c == 0 for c in codes) and all(s["finished"] for s in parts) and len(parts) == len(procs),
             "rows": args.rows, "children": parts, "exit_codes": codes}
    (args.out / args.state_file).write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2), flush=True)
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
