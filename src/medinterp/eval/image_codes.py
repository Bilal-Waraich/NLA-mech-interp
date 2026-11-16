"""Image-level codes: max over an image's patch tokens, per latent.

Image level is the unit of analysis because labels, datasets and (later) other encoders are defined per
image, and because token-level probes on sparse codes cannot exceed AUROC ~ 0.5 + density / 2.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from ..data.archive import ArchivedActivations


class Encoder(Protocol):
    d_sae: int
    k: int

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad()
def image_max_codes(acts: ArchivedActivations, enc: Encoder, images_per_chunk: int = 256
                    ) -> dict[str, np.ndarray]:
    """Return ``{dataset: float16[n_images, n_latents]}`` of per-image max activation.

    Also checks the per-token density so a mis-configured encoder fails loudly.
    Complexity: one encode pass over all tokens, O(N_tokens * d_model * n_latents).
    """
    out: dict[str, np.ndarray] = {}
    for name, block in acts.blocks.items():
        result = np.zeros((block.n_images, enc.d_sae), dtype=np.float16)
        active_total = 0
        for img0, x in acts.iter_token_chunks(name, images_per_chunk):
            codes = enc.encode(x)
            active_total += int((codes > 0).sum())
            n_img = x.shape[0] // acts.n_patches
            per_image = codes.view(n_img, acts.n_patches, -1).amax(dim=1)
            result[img0:img0 + n_img] = per_image.numpy().astype(np.float16)
        # Each token keeps at most k latents; equality fails only if pre-activations are <= 0.
        density = active_total / (block.n_tokens * enc.d_sae)
        assert density <= enc.k / enc.d_sae + 1e-9, f"{name}: density {density} exceeds k/d_sae"
        out[name] = result
    return out


@torch.no_grad()
def image_codes(acts: ArchivedActivations, enc: Encoder, pooling: str = "max", images_per_chunk: int = 256
                ) -> dict[str, np.ndarray]:
    """Like ``image_max_codes`` with a choice of pooling over an image's tokens ("max" or "mean").

    Raises ValueError for an unknown pooling.
    """
    if pooling == "max":
        return image_max_codes(acts, enc, images_per_chunk)
    if pooling != "mean":
        raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}")
    out: dict[str, np.ndarray] = {}
    for name, block in acts.blocks.items():
        result = np.zeros((block.n_images, enc.d_sae), dtype=np.float16)
        for img0, x in acts.iter_token_chunks(name, images_per_chunk):
            n_img = x.shape[0] // acts.n_patches
            result[img0:img0 + n_img] = enc.encode(x).view(n_img, acts.n_patches, -1).mean(1).numpy()
        out[name] = result
    return out
