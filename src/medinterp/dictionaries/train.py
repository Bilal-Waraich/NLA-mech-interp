"""Seeded BatchTopK SAE training, following the thesis recipe (train_sae.py) with explicit seeding.

Recipe (from the archived run's cfg.json / train.log): BatchTopK over the batch during training, per-token
top-k at inference; unit-norm decoder rows with tangent-projected gradients; auxiliary dead-latent loss
(k_aux = d_model // 2, penalty 1/32, dead after 25 inactive batches); L1 2e-4 on codes; Adam lr 2e-4,
betas (0.9, 0.999), grad clip 1.0, linear warmup 1000 steps then cosine decay to 0.1x; batches stratified
to an equal number of tokens per dataset. Differences from the thesis run: fp32 instead of AMP fp16,
explicit seeds, and b_dec initialised to zero (the data are per-dataset centred).

Options added for the protocol (all default to the behaviour above, which tests/_legacy_train.py freezes):
  init_seed / data_seed   separate seeds for the initial weights and for the batch order
  sampling                "equal" (same tokens per dataset per batch) or "proportional" (uniform over all
                          allowed tokens, so each dataset contributes in proportion to its size)
  train_rows              restricts sampling to the given token rows per dataset (e.g. the train split)
  checkpoint_steps        steps at which a copy of the dictionary is handed to ``on_checkpoint``
  fixed_decoder           a decoder that is never updated (frozen-decoder baseline); the encoder is trained
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

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
    # None means "use seed", which keeps old configs and old results valid.
    init_seed: int | None = None
    data_seed: int | None = None
    sampling: str = "equal"
    checkpoint_steps: tuple[int, ...] = ()

    @property
    def resolved_init_seed(self) -> int:
        return self.seed if self.init_seed is None else self.init_seed

    @property
    def resolved_data_seed(self) -> int:
        return self.seed if self.data_seed is None else self.data_seed


def _lr_factor(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return step / max(1, cfg.warmup)
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


class _BatchSampler:
    """Draws token batches; the RNG call sequence for ``equal`` matches the original trainer exactly."""

    def __init__(self, data: dict[str, np.ndarray], cfg: TrainConfig,
                 train_rows: dict[str, np.ndarray] | None) -> None:
        if cfg.sampling not in ("equal", "proportional"):
            raise ValueError(f"sampling must be 'equal' or 'proportional', got {cfg.sampling!r}")
        self.data = data
        self.names = list(data)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.resolved_data_seed)
        # None keeps the original indexing path (rng over len(data)), so default runs are bit-identical.
        self.rows = None if train_rows is None else {n: np.asarray(train_rows[n], dtype=np.int64)
                                                      for n in self.names}
        if self.rows is not None:
            for n, r in self.rows.items():
                if r.size == 0:
                    raise ValueError(f"train_rows[{n!r}] is empty")
        sizes = np.array([self._n_allowed(n) for n in self.names], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])

    def _n_allowed(self, name: str) -> int:
        return len(self.data[name]) if self.rows is None else len(self.rows[name])

    def _take(self, name: str, local: np.ndarray) -> np.ndarray:
        idx = local if self.rows is None else self.rows[name][local]
        return self.data[name][idx]

    def __call__(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.sampling == "equal":
            return np.concatenate([
                self._take(n, self.rng.integers(0, self._n_allowed(n), cfg.batch_per_dataset))
                for n in self.names])
        # Proportional: uniform over the pooled allowed tokens, same total batch size as "equal".
        flat = self.rng.integers(0, self.offsets[-1], cfg.batch_per_dataset * len(self.names))
        which = np.searchsorted(self.offsets, flat, side="right") - 1
        parts = [self._take(n, flat[which == i] - self.offsets[i]) for i, n in enumerate(self.names)]
        return np.concatenate(parts)


def _snapshot(params: dict[str, torch.Tensor], k: int) -> TopKSAE:
    with torch.no_grad():
        Wd = params["W_dec"] / params["W_dec"].norm(dim=-1, keepdim=True)
    return TopKSAE(params["W_enc"].detach().cpu().clone(), params["b_enc"].detach().cpu().clone(),
                   Wd.detach().cpu().clone(), params["b_dec"].detach().cpu().clone(), k)


def train_batchtopk(data: dict[str, np.ndarray], cfg: TrainConfig, device: str = "cpu",
                    log=print, train_rows: dict[str, np.ndarray] | None = None,
                    on_checkpoint: Callable[[int, TopKSAE], None] | None = None,
                    fixed_decoder: torch.Tensor | None = None) -> TopKSAE:
    """Train on ``data`` = {dataset: float16[n_tokens, d_model]}; returns a CPU ``TopKSAE``.

    Parameters:
      train_rows       optional {dataset: token row indices} that batches may draw from
      on_checkpoint    called as ``on_checkpoint(step, sae)`` at every step in ``cfg.checkpoint_steps``
      fixed_decoder    optional ``[d_sae, d_model]`` decoder that stays frozen (rows are unit-normalised
                       once); the encoder is initialised as its transpose, as in the thesis recipe
    Raises ValueError for an unknown sampling mode, an empty ``train_rows`` entry or a decoder of the
    wrong shape.
    """
    torch.manual_seed(cfg.resolved_init_seed)
    sampler = _BatchSampler(data, cfg, train_rows)
    dev = torch.device(device)

    if fixed_decoder is None:
        # Thesis initialisation: kaiming-uniform W_enc, W_dec = row-normalised W_enc^T * 0.1, W_enc = W_dec^T.
        W_enc = torch.empty(cfg.d_model, cfg.d_sae)
        torch.nn.init.kaiming_uniform_(W_enc)
        W_dec = W_enc.T.clone()
        W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True) * 0.1
    else:
        if tuple(fixed_decoder.shape) != (cfg.d_sae, cfg.d_model):
            raise ValueError(f"fixed_decoder has shape {tuple(fixed_decoder.shape)}, "
                             f"expected {(cfg.d_sae, cfg.d_model)}")
        W_dec = fixed_decoder.float() / fixed_decoder.float().norm(dim=-1, keepdim=True)
    params = {
        "W_enc": torch.nn.Parameter(W_dec.T.clone().contiguous().to(dev)),
        "W_dec": torch.nn.Parameter(W_dec.clone().to(dev), requires_grad=fixed_decoder is None),
        "b_enc": torch.nn.Parameter(torch.zeros(cfg.d_sae, device=dev)),
        "b_dec": torch.nn.Parameter(torch.zeros(cfg.d_model, device=dev)),
    }
    trainable = [p for p in params.values() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.lr, betas=(0.9, 0.999))
    inactive = torch.zeros(cfg.d_sae, device=dev)
    checkpoints = set(cfg.checkpoint_steps)

    for step in range(1, cfg.steps + 1):
        x = torch.from_numpy(sampler().astype(np.float32)).to(dev)
        B = x.shape[0]

        pre = torch.relu((x - params["b_dec"]) @ params["W_enc"] + params["b_enc"])
        top = torch.topk(pre.flatten(), cfg.k * B)                      # BatchTopK
        z = torch.zeros_like(pre.flatten()).scatter(-1, top.indices, top.values).view_as(pre)
        x_hat = z @ params["W_dec"] + params["b_dec"]

        # All tensors below keep fixed shapes. Data-dependent shapes (boolean indexing such as
        # pre[:, dead]) make the MPS backend compile and cache a new graph for every distinct number
        # of dead latents, which leaked ~50 MB/min of host memory in the first run.
        with torch.no_grad():
            fired = z.sum(0) > 0
            inactive = torch.where(fired, torch.zeros_like(inactive), inactive + 1)
        l2 = (x_hat - x).pow(2).mean()
        l1 = cfg.l1_coeff * z.abs().sum(-1).mean()
        aux = torch.zeros((), device=dev)
        dead = inactive >= cfg.n_batches_to_dead
        n_dead = int(dead.sum())
        if n_dead > 0:
            k_aux = cfg.d_model // 2
            scale = min(n_dead / k_aux, 1.0)
            # Mask instead of slice: non-dead latents become 0, so when fewer than k_aux latents are
            # dead the extra selected entries are zeros and contribute nothing (same result as slicing).
            pre_dead = pre * dead.to(pre.dtype)
            t = torch.topk(pre_dead, k_aux, dim=-1)
            z_aux = torch.zeros_like(pre_dead).scatter(-1, t.indices, t.values)
            aux = cfg.aux_penalty * scale * (z_aux @ params["W_dec"] - (x - x_hat).detach()).pow(2).mean()
        loss = l2 + l1 + aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        if fixed_decoder is None:
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
        if on_checkpoint is not None and step in checkpoints:
            on_checkpoint(step, _snapshot(params, cfg.k))

    return _snapshot(params, cfg.k)
