"""Inference-only BatchTopK SAE matching the thesis checkpoints.

Encode logic follows ``BatchTopKSAE.encode`` from the thesis code
(github.com/Bilal-Waraich/mechinterp-medical-vision, co-developed with Nikolay Tsonev; architecture after
github.com/bartbussmann/BatchTopK): ``relu((x - b_dec) @ W_enc + b_enc)`` followed by a per-token top-k.
Training-time BatchTopK selection is not needed for evaluation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass
class TopKSAE:
    """Frozen SAE weights with per-token top-k encoding."""

    W_enc: torch.Tensor  # [d_model, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    W_dec: torch.Tensor  # [d_sae, d_model], rows unit-norm
    b_dec: torch.Tensor  # [d_model]
    k: int

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    @classmethod
    def from_checkpoint(cls, ckpt_dir: Path | str) -> "TopKSAE":
        """Load ``sae_weights.safetensors`` + ``cfg.json`` written by the thesis trainer."""
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "cfg.json").read_text())
        w = load_file(str(ckpt_dir / "sae_weights.safetensors"))
        return cls(w["W_enc"], w["b_enc"], w["W_dec"], w["b_dec"], int(cfg["k"]))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse codes ``[B, d_sae]`` with at most ``k`` non-zeros per row.

        Complexity: O(B * d_model * d_sae) for the matmul plus O(B * d_sae) for top-k.
        """
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        if self.k >= self.d_sae:  # ReLU+L1 dictionaries (k = d_sae): top-k over all latents is the identity
            return pre
        top = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec
