"""Tiny synthetic training runs: learning happens, and seeds make runs reproducible."""
from __future__ import annotations

import numpy as np
import torch

from medinterp.dictionaries.train import TrainConfig, train_batchtopk


def _data(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(32, 16))
    def sample(n):
        z = (rng.random((n, 32)) < 0.1) * rng.random((n, 32))   # sparse ground-truth codes
        return (z @ atoms).astype(np.float16)
    return {"a": sample(4000), "b": sample(4000)}


def test_training_reduces_reconstruction_error() -> None:
    # lr above the real-run 2e-4 so a 400-step toy problem converges (0.81 FVE measured; 2e-4 gives 0.41).
    cfg = TrainConfig(d_model=16, d_sae=64, k=4, steps=400, batch_per_dataset=64, warmup=20, lr=2e-3,
                      log_every=10_000)
    data = _data()
    sae = train_batchtopk(data, cfg, log=lambda *_: None)
    x = torch.from_numpy(data["a"][:1000].astype(np.float32))
    fve = 1 - (x - sae.decode(sae.encode(x))).var() / x.var()
    assert fve > 0.5


def test_same_seed_same_weights() -> None:
    cfg = TrainConfig(d_model=16, d_sae=64, k=4, steps=50, batch_per_dataset=32, warmup=5, log_every=10_000)
    a = train_batchtopk(_data(), cfg, log=lambda *_: None)
    b = train_batchtopk(_data(), cfg, log=lambda *_: None)
    assert torch.allclose(a.W_enc, b.W_enc)
