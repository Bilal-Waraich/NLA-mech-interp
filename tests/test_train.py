"""Tiny synthetic training runs: learning happens, and seeds make runs reproducible."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _legacy_train import TrainConfig as LegacyConfig
from _legacy_train import train_batchtopk as legacy_train
from medinterp.dictionaries.train import TrainConfig, train_batchtopk


def _data(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(32, 16))
    def sample(n):
        z = (rng.random((n, 32)) < 0.1) * rng.random((n, 32))   # sparse ground-truth codes
        return (z @ atoms).astype(np.float16)
    return {"a": sample(4000), "b": sample(4000)}


SMALL = dict(d_model=16, d_sae=64, k=4, steps=50, batch_per_dataset=32, warmup=5, log_every=10_000)
quiet = dict(log=lambda *_: None)


def test_training_reduces_reconstruction_error() -> None:
    # lr above the real-run 2e-4 so a 400-step toy problem converges (0.81 FVE measured; 2e-4 gives 0.41).
    cfg = TrainConfig(d_model=16, d_sae=64, k=4, steps=400, batch_per_dataset=64, warmup=20, lr=2e-3,
                      log_every=10_000)
    data = _data()
    sae = train_batchtopk(data, cfg, **quiet)
    x = torch.from_numpy(data["a"][:1000].astype(np.float32))
    fve = 1 - (x - sae.decode(sae.encode(x))).var() / x.var()
    assert fve > 0.5


def test_same_seed_same_weights() -> None:
    cfg = TrainConfig(**SMALL)
    a = train_batchtopk(_data(), cfg, **quiet)
    b = train_batchtopk(_data(), cfg, **quiet)
    assert torch.allclose(a.W_enc, b.W_enc)


@pytest.mark.parametrize("s", [0, 3])
def test_defaults_reproduce_the_previous_trainer(s: int) -> None:
    """init_seed == data_seed == s with default settings gives exactly the weights of the old code."""
    old = legacy_train(_data(), LegacyConfig(**SMALL, seed=s), **quiet)
    for cfg in (TrainConfig(**SMALL, seed=s), TrainConfig(**SMALL, init_seed=s, data_seed=s)):
        new = train_batchtopk(_data(), cfg, **quiet)
        for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
            assert torch.equal(getattr(old, name), getattr(new, name)), name


def test_full_train_rows_equal_no_restriction() -> None:
    data = _data()
    rows = {n: np.arange(len(v)) for n, v in data.items()}
    a = train_batchtopk(data, TrainConfig(**SMALL), **quiet)
    b = train_batchtopk(data, TrainConfig(**SMALL), train_rows=rows, **quiet)
    assert torch.equal(a.W_enc, b.W_enc)


def test_seeds_act_separately() -> None:
    data = _data()
    base = train_batchtopk(data, TrainConfig(**SMALL, init_seed=0, data_seed=0), **quiet)
    other_order = train_batchtopk(data, TrainConfig(**SMALL, init_seed=0, data_seed=1), **quiet)
    other_init = train_batchtopk(data, TrainConfig(**SMALL, init_seed=1, data_seed=0), **quiet)
    assert not torch.equal(base.W_enc, other_order.W_enc)
    assert not torch.equal(base.W_enc, other_init.W_enc)
    # With zero steps only the initialisation remains: it depends on init_seed alone.
    zero = dict(SMALL, steps=0)
    a = train_batchtopk(data, TrainConfig(**zero, init_seed=5, data_seed=0), **quiet)
    b = train_batchtopk(data, TrainConfig(**zero, init_seed=5, data_seed=9), **quiet)
    assert torch.equal(a.W_dec, b.W_dec)


def test_train_rows_restrict_the_tokens_used() -> None:
    """Rows outside train_rows are replaced by NaN: a run that touched them would produce NaN weights."""
    data = {n: v.copy() for n, v in _data().items()}
    rows = {n: np.arange(0, len(v), 2) for n, v in data.items()}
    for n, v in data.items():
        v[1::2] = np.nan
    for sampling in ("equal", "proportional"):
        sae = train_batchtopk(data, TrainConfig(**SMALL, sampling=sampling), train_rows=rows, **quiet)
        assert torch.isfinite(sae.W_enc).all()


def test_proportional_sampling_follows_dataset_sizes() -> None:
    from medinterp.dictionaries.train import _BatchSampler
    data = {"big": np.zeros((9000, 2), np.float16), "small": np.ones((1000, 2), np.float16)}
    cfg = TrainConfig(d_model=2, batch_per_dataset=5000, sampling="proportional")
    batch = _BatchSampler(data, cfg, None)()
    assert len(batch) == 10_000
    assert abs(batch[:, 0].mean() - 0.1) < 0.01            # share of "small" tokens


def test_checkpoints_are_taken_at_the_given_steps() -> None:
    seen = {}
    final = train_batchtopk(_data(), TrainConfig(**SMALL, checkpoint_steps=(10, 50)),
                            on_checkpoint=lambda s, sae: seen.__setitem__(s, sae), **quiet)
    assert sorted(seen) == [10, 50]
    assert torch.equal(seen[50].W_enc, final.W_enc)
    assert not torch.equal(seen[10].W_enc, final.W_enc)


def test_unknown_sampling_is_rejected() -> None:
    with pytest.raises(ValueError):
        train_batchtopk(_data(), TrainConfig(**SMALL, sampling="stratified"), **quiet)
