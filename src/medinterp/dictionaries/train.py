"""Seeded BatchTopK SAE training, following the thesis recipe (train_sae.py) with explicit seeding.

Recipe (from the archived run's cfg.json / train.log): BatchTopK over the batch during training, per-token
top-k at inference; unit-norm decoder rows with tangent-projected gradients; auxiliary dead-latent loss
(k_aux = d_model // 2, penalty 1/32, dead after 25 inactive batches); L1 2e-4 on codes; Adam lr 2e-4,
betas (0.9, 0.999), grad clip 1.0, linear warmup 1000 steps then cosine decay to 0.1x; batches stratified
to an equal number of tokens per dataset. Differences from the thesis run: fp32 instead of AMP fp16,
explicit seeds, and b_dec initialised to zero (the data are per-dataset centred).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .sae import TopKSAE


@dataclass
class TrainConfig:
    d_model: int = 1024
    d_sae: int = 4096
    k: int = 48
    steps: int = 20_000
    batch_per_dataset: int = 512
    lr: float = 2e-4
    warmup: int = 1000
    l1_coeff: float = 2e-4
    aux_penalty: float = 1 / 32
    n_batches_to_dead: int = 25
    max_grad_norm: float = 1.0
    seed: int = 0
    log_every: int = 1000


def _lr_factor(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return step / max(1, cfg.warmup)
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def train_batchtopk(data: dict[str, np.ndarray], cfg: TrainConfig, device: str = "cpu",
                    log=print) -> TopKSAE:
    """Train on ``data`` = {dataset: float16[n_tokens, d_model]}; returns a CPU ``TopKSAE``.

    Every source of randomness (init, batch sampling) derives from ``cfg.seed``.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = torch.device(device)

    # Thesis initialisation: kaiming-uniform W_enc, W_dec = row-normalised W_enc^T * 0.1, W_enc = W_dec^T.
    W_enc = torch.empty(cfg.d_model, cfg.d_sae)
    torch.nn.init.kaiming_uniform_(W_enc)
    W_dec = W_enc.T.clone()
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True) * 0.1
    params = {
        "W_enc": torch.nn.Parameter(W_dec.T.clone().contiguous().to(dev)),
        "W_dec": torch.nn.Parameter(W_dec.to(dev)),
        "b_enc": torch.nn.Parameter(torch.zeros(cfg.d_sae, device=dev)),
        "b_dec": torch.nn.Parameter(torch.zeros(cfg.d_model, device=dev)),
    }
    opt = torch.optim.Adam(params.values(), lr=cfg.lr, betas=(0.9, 0.999))
    inactive = torch.zeros(cfg.d_sae, device=dev)
    names = list(data)

    for step in range(1, cfg.steps + 1):
        # Stratified batch: the same number of random tokens from every dataset.
        x = torch.from_numpy(np.concatenate([
            data[n][rng.integers(0, len(data[n]), cfg.batch_per_dataset)] for n in names
        ]).astype(np.float32)).to(dev)
        B = x.shape[0]

        pre = torch.relu((x - params["b_dec"]) @ params["W_enc"] + params["b_enc"])
        top = torch.topk(pre.flatten(), cfg.k * B)                      # BatchTopK
        z = torch.zeros_like(pre.flatten()).scatter(-1, top.indices, top.values).view_as(pre)
        x_hat = z @ params["W_dec"] + params["b_dec"]

        with torch.no_grad():
            fired = z.sum(0) > 0
            inactive += (~fired).float()
            inactive[fired] = 0
        l2 = (x_hat - x).pow(2).mean()
        l1 = cfg.l1_coeff * z.abs().sum(-1).mean()
        aux = torch.zeros((), device=dev)
        dead = inactive >= cfg.n_batches_to_dead
        if dead.any():
            k_aux = min(cfg.d_model // 2, int(dead.sum()))
            scale = min(int(dead.sum()) / (cfg.d_model // 2), 1.0)
            pre_dead = pre[:, dead]
            t = torch.topk(pre_dead, k_aux, dim=-1)
            z_aux = torch.zeros_like(pre_dead).scatter(-1, t.indices, t.values)
            aux = cfg.aux_penalty * scale * (z_aux @ params["W_dec"][dead] - (x - x_hat).detach()).pow(2).mean()
        loss = l2 + l1 + aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), cfg.max_grad_norm)
        with torch.no_grad():  # keep decoder rows on the unit sphere
            Wd = params["W_dec"]
            Wn = Wd / Wd.norm(dim=-1, keepdim=True)
            Wd.grad -= (Wd.grad * Wn).sum(-1, keepdim=True) * Wn
            Wd.copy_(Wn)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * _lr_factor(step, cfg)
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            ev = 1 - (x - x_hat).var().item() / x.var().item()
            log(f"step {step:>6}  loss {loss.item():.3f}  l2 {l2.item():.3f}  aux {aux.item():.4f}  "
                f"EV {ev:.4f}  dead {int(dead.sum())}")

    with torch.no_grad():
        Wd = params["W_dec"] / params["W_dec"].norm(dim=-1, keepdim=True)
    return TopKSAE(params["W_enc"].detach().cpu(), params["b_enc"].detach().cpu(), Wd.cpu(),
                   params["b_dec"].detach().cpu(), cfg.k)
