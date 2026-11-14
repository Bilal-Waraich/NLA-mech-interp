"""Reproduce numbers stored in the thesis archive from the archived checkpoint and activations.

These tests guard the data path every later stage depends on. They are skipped when the
(gitignored) archive is not present.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR, ArchivedActivations
from medinterp.dictionaries.sae import TopKSAE

pytestmark = pytest.mark.skipif(
    not (DEFAULT_RUN_DIR / "activations.safetensors").exists(), reason="thesis archive not present"
)

CKPT = DEFAULT_RUN_DIR / "checkpoints/step_0200000"


@pytest.fixture(scope="module")
def acts() -> ArchivedActivations:
    return ArchivedActivations()


def test_layout(acts: ArchivedActivations) -> None:
    # Image counts verified during the audit.
    assert {n: b.n_images for n, b in acts.blocks.items()} == {
        "chexpert": 10000, "rsna": 10000, "siim_covid": 6334, "tb": 662,
    }
    assert acts.d_model == 1024 and acts.n_patches == 49


def test_uncentred_matches_partial(acts: ArchivedActivations) -> None:
    """Adding back the stored mean must reproduce the uncentred TB partial exactly."""
    from safetensors import safe_open

    raw = ArchivedActivations(centred=False)
    _, x = next(raw.iter_token_chunks("tb", images_per_chunk=8))
    part = safe_open(str(DEFAULT_RUN_DIR / "_partials/activations.tb.partial.safetensors"), "pt")
    ref = part.get_slice("activations")[: x.shape[0]].float()
    assert torch.allclose(x, ref, atol=1e-3)
    # Harvest RMS-normalised each token to L2 norm d_model.
    assert torch.allclose(ref.norm(dim=1), torch.full((x.shape[0],), 1024.0), atol=1e-2)


def test_frequencies_match_archived_csv(acts: ArchivedActivations) -> None:
    """Per-dataset token firing frequencies of the confirmed latents match the archived CSV."""
    sae = TopKSAE.from_checkpoint(CKPT)
    rows = {int(r["latent_idx"]): r for r in csv.DictReader(
        open(DEFAULT_RUN_DIR / "universality/eval/universal_candidates.csv"))}
    latents = np.array(sorted(rows))
    for name, block in acts.blocks.items():
        count = torch.zeros(sae.d_sae)
        for _, x in acts.iter_token_chunks(name):
            count += (sae.encode(x) > 0).sum(0)
        freq = (count / block.n_tokens).numpy()
        # Per-token top-k forces mean density to exactly k / d_sae.
        assert freq.mean() == pytest.approx(sae.k / sae.d_sae, abs=1e-6)
        stored = np.array([float(rows[i][f"freq_{name}"]) for i in latents])
        assert np.abs(freq[latents] - stored).max() < 1e-5
