"""Stage 0.6: retrain the SAE with explicit seeds on the archived activations (Apple GPU if available).

--protocol thesis (default): all images, one seed for initialisation and batch order.
  Output: results/stage0/sae_seeds/seed<N>/{sae_weights.safetensors, cfg.json, train.log}
--protocol v1: train-split tokens only (split file), separate --init-seed / --data-seed, equal or
  proportional sampling, decoders saved at --checkpoints. Output:
  results/stage0_v1/sae_seeds/<encoder>_<layer>_d<d_sae>_k<k>_s<steps>_i<init>_b<data>[_<variant>]/
  with step_<n>/ checkpoints, config.json and train.log. A run whose final weights exist is skipped.

Run:    ~/.venvs/nla-mech-interp/bin/python scripts/stage0_train_seeds.py [seed ...]   (default 0 1 2)
        ~/.venvs/nla-mech-interp/bin/python scripts/stage0_train_seeds.py --protocol v1 --init-seed 0 --data-seed 0
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from medinterp.data.archive import DEFAULT_RUN_DIR as DEFAULT_RUN_DIR_V1
from medinterp.data.archive import ArchivedActivations
from medinterp.dictionaries.train import TrainConfig, train_batchtopk

OUT = Path(__file__).resolve().parents[1] / "results/stage0/sae_seeds"


def load_fp16(acts: ArchivedActivations) -> dict[str, np.ndarray]:
    """All tokens per dataset as float16 (~2.7 GB total) so training fits in 8 GB RAM."""
    data = {}
    for name, block in acts.blocks.items():
        arr = np.empty((block.n_tokens, acts.d_model), dtype=np.float16)
        pos = 0
        for _, x in acts.iter_token_chunks(name, 1024):
            arr[pos:pos + len(x)] = x.numpy().astype(np.float16)
            pos += len(x)
        data[name] = arr
    return data


def main(seeds: list[int]) -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load_fp16(ArchivedActivations())
    for seed in seeds:
        out = OUT / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        log_file = open(out / "train.log", "w")

        def log(msg: str) -> None:
            line = f"[{time.strftime('%H:%M:%S')}] seed{seed} {msg}"
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

        cfg = TrainConfig(seed=seed)
        log(f"device={device} cfg={asdict(cfg)}")
        sae = train_batchtopk(data, cfg, device=device, log=log)
        save_file({"W_enc": sae.W_enc.contiguous(), "b_enc": sae.b_enc, "W_dec": sae.W_dec.contiguous(),
                   "b_dec": sae.b_dec}, str(out / "sae_weights.safetensors"))
        (out / "cfg.json").write_text(json.dumps({**asdict(cfg), "architecture": "batch_topk",
                                                  "expansion_factor": cfg.d_sae // cfg.d_model}, indent=2))
        log("saved")
        log_file.close()


def save_sae(sae, out: Path, cfg_dict: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_file({"W_enc": sae.W_enc.contiguous(), "b_enc": sae.b_enc, "W_dec": sae.W_dec.contiguous(),
               "b_dec": sae.b_dec}, str(out / "sae_weights.safetensors"))
    (out / "cfg.json").write_text(json.dumps(cfg_dict, indent=2))


def main_v1(args) -> None:
    from medinterp.data.manifests import DEFAULT_LABEL_DIR
    from medinterp.runinfo import write_config
    from medinterp.stage0_v1 import load_context

    cfg = TrainConfig(steps=args.steps, init_seed=args.init_seed, data_seed=args.data_seed,
                      sampling=args.sampling, checkpoint_steps=tuple(args.checkpoints), d_sae=args.d_sae, k=args.k,
                      seed=args.init_seed)
    variant = "" if args.sampling == "equal" else f"_{args.sampling}"
    name = (f"umedpt_final_d{cfg.d_sae}_k{cfg.k}_s{cfg.steps}_i{cfg.init_seed}_b{cfg.data_seed}{variant}")
    out = args.out / name
    if (out / "sae_weights.safetensors").exists():
        print(f"{out} exists; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ctx = load_context(DEFAULT_RUN_DIR_V1, DEFAULT_LABEL_DIR, args.split_file)
    data = load_fp16(ctx.acts)
    P = ctx.acts.n_patches
    # Token rows of train images: image i owns rows [P*i, P*i + P) of its dataset block.
    train_rows = {d: (np.flatnonzero(ctx.masks[d]["train"])[:, None] * P + np.arange(P)).ravel() for d in data}
    cfg_dict = {**asdict(cfg), "architecture": "batch_topk", "expansion_factor": cfg.d_sae // cfg.d_model,
                "fit_split": "train", "device": device}
    write_config(out, cfg_dict, Path(__file__).resolve().parents[1], args.split_file)
    log_file = open(out / "train.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {name} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"device={device} cfg={cfg_dict} train_tokens={ {d: len(r) for d, r in train_rows.items()} }")
    sae = train_batchtopk(data, cfg, device=device, log=log, train_rows=train_rows,
                          on_checkpoint=lambda step, s: save_sae(s, out / f"step_{step}", {**cfg_dict, "step": step}))
    save_sae(sae, out, cfg_dict)
    log("saved")
    log_file.close()


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seeds", nargs="*", type=int, help="thesis: seeds to train (default 0 1 2)")
    ap.add_argument("--protocol", choices=("thesis", "v1"), default="thesis")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results/stage0_v1/sae_seeds")
    ap.add_argument("--split-file", type=Path, default=Path(__file__).resolve().parents[1] / "data/splits/v1.csv")
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--sampling", choices=("equal", "proportional"), default="equal")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--d-sae", type=int, default=4096)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--checkpoints", type=int, nargs="*", default=[5_000, 10_000, 20_000])
    ap.add_argument("--device", help="default: mps if available, else cpu")
    return ap.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if _args.protocol == "thesis":
        main(_args.seeds or [0, 1, 2])
    else:
        main_v1(_args)
