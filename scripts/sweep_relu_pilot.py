"""Pilot for the ReLU+L1 calibration of WP1 phase 3: short runs over a wide range of L1 coefficients, to set
the range of the 8-value calibration grid so that it brackets L0 = 8 to 32 (protocol section 5, "Phase 3").

Each coefficient trains one ReLU+L1 SAE on train-split tokens (seed 0, ``--steps`` steps, its own shortened
schedule) and reports mean L0, FVE and dead latents on a fixed sample of val tokens. The pilot only places
the grid; no number from it is reported as a result. Writes ``<out>/relu_pilot.csv``.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/sweep_relu_pilot.py --device cuda --out results/sweep_v1/relu_pilot
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

import numpy as np
import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.data.manifests import DEFAULT_LABEL_DIR
from medinterp.dictionaries.train import TrainConfig, train_batchtopk
from medinterp.stage0_v1 import load_context
from medinterp.sweep import load_tokens_fp16, token_rows

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
PILOT_COEFFICIENTS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
VAL_TOKENS_PER_DATASET = 5000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/sweep_v1/relu_pilot")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="archived activations")
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--d-sae", type=int, nargs="+", default=[1024, 4096])
    args = ap.parse_args()

    ctx = load_context(args.run_dir, args.label_dir, args.split_file)
    data = load_tokens_fp16(ctx.acts)
    train_rows = {d: token_rows(ctx.masks[d]["train"], ctx.acts.n_patches) for d in data}
    rng = np.random.default_rng(0)
    parts = []
    for d in data:
        val_rows = token_rows(ctx.masks[d]["val"], ctx.acts.n_patches)
        parts.append(data[d][rng.choice(val_rows, min(VAL_TOKENS_PER_DATASET, len(val_rows)), replace=False)])
    xv = torch.from_numpy(np.concatenate(parts).astype(np.float32)).to(args.device)
    total = (xv - xv.mean(0)).pow(2).sum()

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for d_sae in args.d_sae:
        for c in PILOT_COEFFICIENTS:
            t0 = time.time()
            cfg = TrainConfig(d_sae=d_sae, k=d_sae, steps=args.steps, l1_coeff=c, architecture="relu_l1",
                              log_every=10 ** 9)
            sae = train_batchtopk(data, cfg, device=args.device, train_rows=train_rows, log=lambda *_: None)
            sae.W_enc, sae.b_enc, sae.W_dec, sae.b_dec = (t.to(args.device)
                                                          for t in (sae.W_enc, sae.b_enc, sae.W_dec, sae.b_dec))
            z = sae.encode(xv)
            rec = {"d_sae": d_sae, "l1_coeff": c, "steps": args.steps,
                   "mean_l0": float((z > 0).sum(-1).float().mean()),
                   "fve": float(1 - (xv - sae.decode(z)).pow(2).sum() / total),
                   "n_dead_on_sample": int(((z > 0).sum(0) == 0).sum()), "seconds": time.time() - t0}
            records.append(rec)
            print(rec, flush=True)
            pd.DataFrame(records).to_csv(args.out / "relu_pilot.csv", index=False)


if __name__ == "__main__":
    main()
