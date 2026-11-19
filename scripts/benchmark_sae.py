"""Time SAE training steps per dictionary size on this machine (sweep step 0, build spec WP1).

Synthetic float16 tokens with the harvest's shape are used: the cost of a step does not depend on the
values, and the benchmark can then run anywhere (Kaggle included) without UMedPT-derived data. Each size
runs a few untimed warm-up steps, then ``--steps`` timed steps with the real recipe (BatchTopKTrainer,
batch 512 per dataset, 4 datasets). The projection multiplies the measured time per step by the steps of
every row of the run table.

Output: ``<out>/sae_<label>.json`` with steps per second per size, hours per run unit, and the projected
hours of the sweep table on this device.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/benchmark_sae.py --device mps --label laptop_mps
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from medinterp.dictionaries.train import BatchTopKTrainer, TrainConfig
from medinterp.sweep import BASE_D_SAE, BASE_STEPS, sweep_table

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
# (d_sae, k) pairs as they occur in the run table; k changes the top-k cost only slightly.
SIZES = ((1024, 48), (4096, 48), (16_384, 16))
N_DATASETS, TOKENS_PER_DATASET, D_MODEL = 4, 50_000, 1024


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_size(d_sae: int, k: int, device: str, steps: int, warmup: int, threads: int | None) -> float:
    """Seconds per training step for one dictionary size."""
    rng = np.random.default_rng(0)
    data = {f"d{i}": rng.standard_normal((TOKENS_PER_DATASET, D_MODEL), dtype=np.float32).astype(np.float16)
            for i in range(N_DATASETS)}
    cfg = TrainConfig(d_sae=d_sae, k=k, steps=warmup + steps, log_every=10 ** 9)
    trainer = BatchTopKTrainer(data, cfg, device, log=lambda *_: None)
    trainer.run(until_step=warmup)
    sync(device)
    t0 = time.perf_counter()
    trainer.run()
    sync(device)
    return (time.perf_counter() - t0) / steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--label", required=True, help="name of this machine/device, e.g. laptop_mps, kaggle_t4")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/benchmark")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    per_step = {}
    for d_sae, k in SIZES:
        s = time_size(d_sae, k, args.device, args.steps, args.warmup, args.threads)
        per_step[d_sae] = s
        print(f"d_sae {d_sae:>6} k {k:>2}: {1 / s:7.2f} steps/s  ({s * BASE_STEPS / 3600:.2f} h per 20k steps)",
              flush=True)
    table = sweep_table(phase2=True)
    hours = float(sum(r.steps * per_step[r.d_sae] for r in table.itertuples()) / 3600)
    result = {"label": args.label, "device": args.device, "platform": platform.platform(),
              "processor": platform.processor(), "cpu_count": os.cpu_count(),
              "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
              "steps_timed": args.steps, "warmup_steps": args.warmup,
              "steps_per_second": {str(d): 1 / s for d, s in per_step.items()},
              "hours_per_run_unit": per_step[BASE_D_SAE] * BASE_STEPS / 3600,
              "sweep_runs": len(table), "sweep_run_units": float(table.run_units.sum()),
              "sweep_hours_one_worker": hours}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"sae_{args.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
