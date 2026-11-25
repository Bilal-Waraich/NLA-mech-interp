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


def test_resume_from_state_is_identical():
    """Stopping, saving the state through torch.save and resuming gives the uninterrupted weights."""
    import io
    from medinterp.dictionaries.train import BatchTopKTrainer

    cfg = TrainConfig(**SMALL, checkpoint_steps=(30,))
    full = train_batchtopk(_data(), cfg, **quiet)
    first = BatchTopKTrainer(_data(), cfg, **quiet)
    first.run(until_step=23)
    buf = io.BytesIO()
    torch.save(first.state_dict(), buf)
    buf.seek(0)
    seen = []
    second = BatchTopKTrainer(_data(), cfg, on_checkpoint=lambda s, _: seen.append(s), **quiet)
    second.load_state_dict(torch.load(buf, weights_only=False))
    second.run()
    resumed = second.snapshot()
    assert second.finished and seen == [30]
    for name in ("W_enc", "b_enc", "W_dec", "b_dec"):
        assert torch.equal(getattr(full, name), getattr(resumed, name)), name


def test_should_stop_ends_after_a_whole_step():
    from medinterp.dictionaries.train import BatchTopKTrainer

    trainer = BatchTopKTrainer(_data(), TrainConfig(**SMALL), **quiet)
    calls = []
    trainer.run(should_stop=lambda: calls.append(1) or len(calls) >= 7)
    assert trainer.step == 7 and not trainer.finished


# ---------------------------------------------------------------- phase 3 architectures


def _equal_weights(a, b) -> bool:
    return all(torch.equal(getattr(a, n), getattr(b, n)) for n in ("W_enc", "b_enc", "W_dec", "b_dec"))


def test_matryoshka_with_one_prefix_is_batchtopk() -> None:
    a = train_batchtopk(_data(), TrainConfig(**SMALL), **quiet)
    b = train_batchtopk(_data(), TrainConfig(**SMALL, architecture="matryoshka", matryoshka_fractions=(1.0,)),
                        **quiet)
    assert _equal_weights(a, b)


def test_relu_l1_without_penalty_is_a_relu_autoencoder() -> None:
    """With L1 = 0, ReLU+L1 equals BatchTopK that keeps every latent (k = d_sae) without the auxiliary loss:
    both then train a plain ReLU autoencoder."""
    relu = train_batchtopk(_data(), TrainConfig(**SMALL, l1_coeff=0.0, architecture="relu_l1"), **quiet)
    plain = dict(SMALL, k=SMALL["d_sae"])
    ref = train_batchtopk(_data(), TrainConfig(**plain, l1_coeff=0.0, aux_penalty=0.0), **quiet)
    assert relu.k == SMALL["d_sae"] and _equal_weights(relu, ref)


def test_relu_l1_penalty_lowers_l0() -> None:
    cfg = dict(d_model=16, d_sae=64, k=4, steps=300, batch_per_dataset=64, warmup=20, lr=2e-3, log_every=10_000)
    x = torch.from_numpy(_data()["a"][:1000].astype(np.float32))
    l0 = []
    for coeff in (0.0, 0.05):
        sae = train_batchtopk(_data(), TrainConfig(**cfg, l1_coeff=coeff, architecture="relu_l1"), **quiet)
        l0.append(float((sae.encode(x) > 0).sum(-1).float().mean()))
    assert l0[1] < l0[0]


def test_matryoshka_first_step_loss_is_the_sum_over_prefixes() -> None:
    from medinterp.dictionaries.train import BatchTopKTrainer, _BatchSampler

    cfg = TrainConfig(**SMALL, architecture="matryoshka", aux_penalty=0.0)
    logged = []
    trainer = BatchTopKTrainer(_data(), cfg, log=logged.append)
    p = {n: v.detach().clone() for n, v in trainer.params.items()}
    x = torch.from_numpy(_BatchSampler(_data(), cfg, None)().astype(np.float32))
    pre = torch.relu((x - p["b_dec"]) @ p["W_enc"] + p["b_enc"])
    top = torch.topk(pre.flatten(), cfg.k * x.shape[0])
    z = torch.zeros_like(pre.flatten()).scatter(-1, top.indices, top.values).view_as(pre)
    expected = sum((z[:, :m] @ p["W_dec"][:m] + p["b_dec"] - x).pow(2).mean() for m in (4, 8, 16, 32, 64))
    expected += cfg.l1_coeff * z.abs().sum(-1).mean()
    trainer.run(until_step=1)
    assert f"loss {expected.item():.3f}" in logged[0]


def test_phase_3_resume_is_identical() -> None:
    import io
    from medinterp.dictionaries.train import BatchTopKTrainer

    for arch in ("matryoshka", "relu_l1"):
        cfg = TrainConfig(**SMALL, architecture=arch)
        full = train_batchtopk(_data(), cfg, **quiet)
        first = BatchTopKTrainer(_data(), cfg, **quiet)
        first.run(until_step=23)
        buf = io.BytesIO()
        torch.save(first.state_dict(), buf)
        buf.seek(0)
        second = BatchTopKTrainer(_data(), cfg, **quiet)
        second.load_state_dict(torch.load(buf, weights_only=False))
        second.run()
        assert _equal_weights(full, second.snapshot()), arch


def test_bad_architecture_and_prefixes_are_rejected() -> None:
    from medinterp.dictionaries.train import matryoshka_prefixes

    with pytest.raises(ValueError):
        train_batchtopk(_data(), TrainConfig(**SMALL, architecture="gated"), **quiet)
    assert matryoshka_prefixes(1024, (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)) == (64, 128, 256, 512, 1024)
    for bad in ((0.5,), (0.5, 0.25, 1.0), (1 / 3, 1.0)):
        with pytest.raises(ValueError):
            matryoshka_prefixes(64, bad)


def test_full_k_encoding_is_the_relu_code() -> None:
    from medinterp.dictionaries.sae import TopKSAE

    g = torch.Generator().manual_seed(0)
    sae = TopKSAE(torch.randn(16, 32, generator=g), torch.randn(32, generator=g), torch.randn(32, 16, generator=g),
                  torch.zeros(16), k=32)
    x = torch.randn(10, 16, generator=g)
    assert torch.equal(sae.encode(x), torch.relu(x @ sae.W_enc + sae.b_enc))


def test_l1_warmup_scales_the_first_steps() -> None:
    """With warm-up over all 50 steps, step 1 uses 1/50 of the coefficient: the same first step as a run with
    that smaller coefficient and no warm-up."""
    from medinterp.dictionaries.train import BatchTopKTrainer

    warm = BatchTopKTrainer(_data(), TrainConfig(**SMALL, architecture="relu_l1", l1_coeff=0.5,
                                                 l1_warmup_fraction=1.0), **quiet)
    flat = BatchTopKTrainer(_data(), TrainConfig(**SMALL, architecture="relu_l1", l1_coeff=0.01), **quiet)
    warm.run(until_step=1)
    flat.run(until_step=1)
    assert _equal_weights(warm.snapshot(), flat.snapshot())
