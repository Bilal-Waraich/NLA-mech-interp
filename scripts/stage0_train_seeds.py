"""Stage 0.6: retrain the SAE with explicit seeds on the archived activations (Apple GPU if available).

Output: results/stage0/sae_seeds/seed<N>/{sae_weights.safetensors, cfg.json, train.log}
Run:    ~/.venvs/nla-mech-interp/bin/python scripts/stage0_train_seeds.py [seed ...]   (default 0 1 2)
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


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
