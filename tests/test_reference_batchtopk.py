"""One training step of our BatchTopK trainer against an independent re-implementation.

The reference follows the BatchTopK trainer of ``saprmarks/dictionary_learning``
(``dictionary_learning/trainers/batch_top_k.py``, read 2026-11-17), which implements Bussmann et al. 2024.
It is written here from scratch, not imported, so the repository gains no dependency.

What must agree exactly, because our recipe claims to do the same thing:
  - the encoder pre-activation ReLU(W_enc^T (x - b_dec) + b_enc);
  - the BatchTopK mask (top k * B entries over the flattened batch);
  - the reconstruction f W_dec + b_dec;
  - the auxiliary loss's choice of dead latents (top k_aux of the dead latents' pre-activations), its
    target (the detached residual) and its decoder without the bias;
  - the projection that removes each decoder row's gradient component along the row.

Deliberate differences from the reference that this test does NOT check. They are recorded in
docs/framing.md section 10 and docs/thesis-audit/thesis-archive-audit.md; all of them come from the thesis recipe:
  - reconstruction loss averaged over dimensions (reference: summed over dimensions);
  - an L1 term on the codes (reference: none);
  - auxiliary loss scaled by min(n_dead / k_aux, 1) and not normalised by the residual variance;
  - a latent is dead after 25 batches without firing (reference: 10M tokens);
  - decoder rows renormalised before the optimiser step (reference: after), and gradient clipping
    before the projection (reference: after);
  - encoder initialised at 0.1 times the unit decoder; b_dec starts at zero (reference: geometric median);
  - per-token top-k at inference (reference: a global threshold estimated during training).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from medinterp.dictionaries.train import BatchTopKTrainer, TrainConfig

D_MODEL, D_SAE, K = 16, 64, 4
BATCH_PER_DATASET, N_DATASETS = 16, 2


def _reference_step_gradients(params: dict[str, torch.Tensor], x: torch.Tensor, inactive: torch.Tensor,
                              cfg: TrainConfig) -> dict[str, torch.Tensor]:
    """Gradients of one step, computed the reference's way, with our loss scaling plugged in."""
    p = {n: v.detach().clone().requires_grad_(True) for n, v in params.items()}
    batch = x.shape[0]

    # Reference encode(use_threshold=False): BatchTopK over the flattened batch.
    post_relu = torch.relu((x - p["b_dec"]) @ p["W_enc"] + p["b_enc"])
    kept = post_relu.flatten().topk(cfg.k * batch, sorted=False)
    codes = torch.zeros_like(post_relu.flatten()).scatter_(-1, kept.indices, kept.values).view_as(post_relu)
    x_hat = codes @ p["W_dec"] + p["b_dec"]

    # Dead-latent bookkeeping in batches (ours) instead of tokens (reference); the update rule is the same.
    fired = codes.sum(0) > 0
    inactive_after = torch.where(fired, torch.zeros_like(inactive), inactive + 1)
    dead = inactive_after >= cfg.n_batches_to_dead
    n_dead = int(dead.sum())

    residual = (x - x_hat).detach()
    aux = torch.zeros(())
    if n_dead > 0:
        # Reference get_auxiliary_loss: -inf outside the dead set, top min(k_aux, n_dead), decoder
        # without bias, squared error summed over dimensions and averaged over the batch.
        k_aux_cap = D_MODEL // 2
        k_aux = min(k_aux_cap, n_dead)
        masked = torch.where(dead[None], post_relu, torch.tensor(-torch.inf))
        top = masked.topk(k_aux, sorted=False)
        aux_codes = torch.zeros_like(post_relu).scatter_(-1, top.indices, top.values)
        pre_norm = (residual - aux_codes @ p["W_dec"]).pow(2).sum(-1).mean()
        # Our scaling of the same quantity: mean over dimensions, times penalty * min(n_dead / k_aux, 1).
        aux = cfg.aux_penalty * min(n_dead / k_aux_cap, 1.0) * pre_norm / D_MODEL

    l2 = (x_hat - x).pow(2).mean()
    l1 = cfg.l1_coeff * codes.abs().sum(-1).mean()
    (l2 + l1 + aux).backward()

    grads = {n: v.grad.detach().clone() for n, v in p.items()}
    # Reference remove_gradient_parallel_to_decoder_directions, on unit-normalised rows.
    unit_rows = p["W_dec"].detach() / p["W_dec"].detach().norm(dim=-1, keepdim=True)
    g = grads["W_dec"]
    grads["W_dec"] = g - (g * unit_rows).sum(-1, keepdim=True) * unit_rows
    return grads


def _trainer_with_fixed_batch(cfg: TrainConfig, dead_latents: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    data = {f"ds{i}": rng.standard_normal((200, D_MODEL)).astype(np.float16) for i in range(N_DATASETS)}
    trainer = BatchTopKTrainer(data, cfg, device="cpu", log=lambda *_: None)
    batch = rng.standard_normal((BATCH_PER_DATASET * N_DATASETS, D_MODEL)).astype(np.float32)
    trainer.sampler = lambda: batch
    trainer.inactive[torch.as_tensor(dead_latents, dtype=torch.long)] = 100.0
    return trainer, torch.from_numpy(batch)


@pytest.mark.parametrize("n_dead", [0, 5, 40])        # none; fewer than k_aux = 8; more than k_aux
@pytest.mark.parametrize("l1_coeff", [0.0, 2e-4])
def test_one_step_gradients_match_reference(n_dead: int, l1_coeff: float) -> None:
    # Clipping is switched off: its order relative to the projection is a known difference (docstring).
    cfg = TrainConfig(d_model=D_MODEL, d_sae=D_SAE, k=K, steps=10, warmup=0, batch_per_dataset=BATCH_PER_DATASET,
                      l1_coeff=l1_coeff, max_grad_norm=1e9, init_seed=3, data_seed=3)
    dead_latents = np.arange(n_dead)
    trainer, x = _trainer_with_fixed_batch(cfg, dead_latents, seed=n_dead)
    initial = {n: v.detach().clone() for n, v in trainer.params.items()}
    inactive_before = trainer.inactive.clone()

    expected = _reference_step_gradients(initial, x, inactive_before, cfg)
    trainer.run(until_step=1)

    for name, grad in expected.items():
        ours = trainer.params[name].grad
        torch.testing.assert_close(ours, grad, rtol=1e-5, atol=1e-7, msg=f"gradient of {name}")
    if n_dead > 0:
        # The test only means something if the auxiliary term was active in this step.
        assert expected["W_dec"][:n_dead].abs().sum() > 0


def test_decoder_rows_leave_the_unit_sphere_by_at_most_one_step() -> None:
    """Known difference (renormalise before the step): after a step the rows are off unit norm by at most
    one Adam step, lr * sqrt(d_model); the saved dictionary renormalises them."""
    cfg = TrainConfig(d_model=D_MODEL, d_sae=D_SAE, k=K, steps=10, warmup=0, batch_per_dataset=BATCH_PER_DATASET,
                      init_seed=1, data_seed=1)
    trainer, _ = _trainer_with_fixed_batch(cfg, np.arange(0), seed=1)
    trainer.run(until_step=1)
    norms = trainer.params["W_dec"].detach().norm(dim=-1)
    lr_step1 = trainer.opt.param_groups[0]["lr"]
    assert float((norms - 1).abs().max()) <= lr_step1 * D_MODEL ** 0.5 * 1.01
    snapshot_norms = trainer.snapshot().W_dec.norm(dim=-1)
    torch.testing.assert_close(snapshot_norms, torch.ones(D_SAE))
