"""Read the ReLU+L1 calibration of WP1 phase 3 (protocol section 5, "Phase 3"): mean L0 on val per run, and the
coefficient picked for each core setting.

For every phase "3c" row of the run table whose run has finished, the final checkpoint encodes all val tokens
and the mean number of non-zero codes per token is recorded, with FVE and dead latents for context. The rule
(``medinterp.sweep.pick_l1_coefficients``) then picks one coefficient per core setting. Writes
``<out>/relu_calibration.csv`` and ``<out>/relu_coefficients.json``.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/sweep_relu_calibration.py --runs results/sweep_v1/runs_3c
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.data.manifests import DEFAULT_LABEL_DIR
from medinterp.dictionaries.sae import TopKSAE
from medinterp.stage0_v1 import load_context
from medinterp.sweep import pick_l1_coefficients

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))


def val_statistics(sae: TopKSAE, ctx, device: str) -> dict:
    """Mean L0 per val token, FVE on val tokens and latents that never fire on val, in one pass.

    Complexity: O(N_val * d_model * d_sae).
    """
    sae_dev = TopKSAE(*(t.to(device) for t in (sae.W_enc, sae.b_enc, sae.W_dec, sae.b_dec)), sae.k)
    n, nnz, sse, s1, s2 = 0, 0.0, 0.0, None, None
    fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    for name in ctx.acts.blocks:
        val = ctx.masks[name]["val"]
        for img0, x in ctx.acts.iter_token_chunks(name, 1024):
            n_img = x.shape[0] // ctx.acts.n_patches
            keep = np.repeat(val[img0:img0 + n_img], ctx.acts.n_patches)
            if not keep.any():
                continue
            xv = x[torch.from_numpy(keep)].float().to(device)
            z = sae_dev.encode(xv)
            active = z > 0
            nnz += float(active.sum())
            fired |= active.any(0)
            sse += float((xv - sae_dev.decode(z)).pow(2).sum())
            s1 = xv.sum(0) if s1 is None else s1 + xv.sum(0)
            s2 = xv.pow(2).sum(0) if s2 is None else s2 + xv.pow(2).sum(0)
            n += len(xv)
    total_var = float((s2 - s1.pow(2) / n).sum())
    return {"n_val_tokens": n, "mean_l0": nnz / n, "fve": 1 - sse / total_var, "n_dead_on_val": int((~fired).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/sweep_v1.csv")
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="run roots holding the 3c runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/sweep_v1/relu_calibration")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="archived activations")
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    table = pd.read_csv(args.config)
    rows = table[table.phase == "3c"]
    ctx = load_context(args.run_dir, args.label_dir, args.split_file)
    records = []
    for row in rows.itertuples():
        run = next((r / row.run_id for r in args.runs if (r / row.run_id / "DONE.json").exists()), None)
        if run is None:
            print(f"{row.run_id}: not finished, skipped", flush=True)
            continue
        sae = TopKSAE.from_checkpoint(run / f"step_{int(row.steps)}")
        stats = val_statistics(sae, ctx, args.device)
        records.append({"run_id": row.run_id, "d_sae": int(row.d_sae), "l1_coeff": float(row.l1_coeff), **stats})
        print(f"{row.run_id}: L0 {stats['mean_l0']:.2f}, FVE {stats['fve']:.4f}, dead {stats['n_dead_on_val']}",
              flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    cal = pd.DataFrame(records)
    cal.to_csv(args.out / "relu_calibration.csv", index=False)
    picked = pick_l1_coefficients(cal)
    (args.out / "relu_coefficients.json").write_text(
        json.dumps({f"d{d}_k{k}": c for (d, k), c in picked.items()}, indent=2))
    print(json.dumps({f"d{d}_k{k}": c for (d, k), c in picked.items()}, indent=2))


if __name__ == "__main__":
    main()
