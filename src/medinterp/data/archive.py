"""Read-only access to the archived UMedPT Stage-4 activation harvest (thesis run ``xray_v2``).

Layout of ``activations.safetensors`` (verified, see docs/thesis-archive-audit.md §1):
  activations  [N_tokens, 1024] float32  RMS-normalised to token norm 1024, then per-dataset centred
  dataset_ids  [N_tokens] int32          contiguous blocks: chexpert, rsna, siim_covid, tb
  labels       [N_tokens] int32          image-level canonical group repeated over the image's tokens
  patch_ids    [N_tokens] int32          misnamed: image index within the dataset
Tokens of image i occupy rows [49*i, 49*i + 49) of its dataset block, in row-major 7x7 order.

The per-dataset ``_partials/*.partial.safetensors`` hold the same tokens before centring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from safetensors import safe_open

# Default location of the run inside this repo (gitignored).
DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parents[3]
    / "archive/BscThesis-BilalWaraich/Output/xray_v2"
)

# Canonical label groups used at harvest time (index -> name).
CANONICAL_GROUPS = (
    "airspace_opacity", "pleural", "cardiac_mediastinal", "atelectasis",
    "vascular_edema", "no_finding", "other",
)


@dataclass(frozen=True)
class DatasetBlock:
    """Contiguous token range of one dataset inside the merged activation file."""

    name: str
    index: int
    token_start: int
    token_end: int
    n_patches: int

    @property
    def n_tokens(self) -> int:
        return self.token_end - self.token_start

    @property
    def n_images(self) -> int:
        return self.n_tokens // self.n_patches


class ArchivedActivations:
    """Lazy, chunked reader over the archived merged activation file.

    Nothing is loaded eagerly except small per-token index tensors; activation rows are
    read in slices so the 5.4 GB file fits in limited RAM.
    """

    def __init__(self, run_dir: Path | str = DEFAULT_RUN_DIR, centred: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.centred = centred
        self._file = safe_open(str(self.run_dir / "activations.safetensors"), framework="pt")
        meta = self._file.metadata()
        self.d_model = int(meta["d_model"])
        self.spatial_h = int(meta["spatial_h"])
        self.spatial_w = int(meta["spatial_w"])
        self.n_patches = self.spatial_h * self.spatial_w
        self.dataset_names: list[str] = json.loads(meta["dataset_names"])
        self.dataset_means = np.asarray(json.loads(meta["dataset_means"]), dtype=np.float32)
        self.image_stems: dict[str, list[str]] = json.loads(meta["image_stems"])

        # Derive block boundaries from dataset_ids instead of trusting counts in metadata.
        ds_ids = self._file.get_tensor("dataset_ids").numpy()
        self.blocks: dict[str, DatasetBlock] = {}
        for idx, name in enumerate(self.dataset_names):
            rows = np.flatnonzero(ds_ids == idx)
            start, end = int(rows[0]), int(rows[-1]) + 1
            # Blocks must be contiguous and image-aligned for the image-level reshapes below.
            assert end - start == rows.size, f"{name}: rows are not contiguous"
            assert (end - start) % self.n_patches == 0, f"{name}: not a multiple of n_patches"
            self.blocks[name] = DatasetBlock(name, idx, start, end, self.n_patches)

        labels = self._file.get_tensor("labels").numpy()
        # One label per image (first token of each image); harvest broadcast it over all tokens.
        self.image_labels: dict[str, np.ndarray] = {
            name: labels[b.token_start:b.token_end:self.n_patches].copy()
            for name, b in self.blocks.items()
        }
        self._acts = self._file.get_slice("activations")

    def iter_token_chunks(
        self, dataset: str, images_per_chunk: int = 512
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield ``(first_image_index, tokens[n_img * n_patches, d_model])`` for one dataset.

        Chunks are image-aligned so callers can reshape to ``[n_img, n_patches, d_model]``.
        If ``centred=False`` the stored dataset mean is added back, which reproduces the
        uncentred partials exactly (verified: max abs diff 0.0).
        """
        block = self.blocks[dataset]
        mean = torch.from_numpy(self.dataset_means[block.index])
        for img0 in range(0, block.n_images, images_per_chunk):
            img1 = min(img0 + images_per_chunk, block.n_images)
            lo = block.token_start + img0 * self.n_patches
            hi = block.token_start + img1 * self.n_patches
            x = self._acts[lo:hi].float()
            if not self.centred:
                x = x + mean
            yield img0, x
