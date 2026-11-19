"""Shared setup of the Stage 0 entry points under ``--protocol v1`` (protocol sections 2-4, 10).

What v1 changes against the thesis protocol: images are split by group with the split file; labels are
the protocol's opacity contrast; everything fitted inside a script (PCA, ridge decoder, probes, the
covariance of the null) uses train images only; latents and hyperparameters are chosen on val; numbers
are reported on ``eval_split`` with cluster-bootstrap intervals over groups.

The trained SAEs are passed in. SAEs fitted on all images (the Stage 0.6 seeds) are accepted for code
checks and are labelled ``fit_split = "all"`` in every output, because the protocol needs train-only SAEs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .data.archive import ArchivedActivations
from .data.labels import opacity_labels
from .data.manifests import build_manifests
from .data.splits import load_split, split_groups, split_masks
from .dictionaries.baselines import random_sae, signed_basis, signed_pca
from .dictionaries.sae import TopKSAE

TIDY_COLUMNS = ["harvest", "encoder", "layer", "dictionary", "d_sae", "k", "steps", "init_seed", "data_seed",
                "metric", "subset", "split", "value", "ci_low", "ci_high"]


@dataclass
class Context:
    acts: ArchivedActivations
    manifests: dict[str, pd.DataFrame]
    masks: dict[str, dict[str, np.ndarray]]      # dataset -> split -> bool[n_images]
    groups: dict[str, np.ndarray]                # dataset -> group id per image
    opacity: dict[str, np.ndarray]               # dataset -> 1 / 0 / NaN per image


def load_context(run_dir: Path, label_dir: Path, split_file: Path) -> Context:
    """Activations (archive centring, as the Stage 0 SAEs were trained), manifests, splits and labels.

    Raises FileNotFoundError if the split file is missing (create it with scripts/make_splits.py).
    """
    if not Path(split_file).exists():
        raise FileNotFoundError(f"{split_file} not found; run scripts/make_splits.py first")
    acts = ArchivedActivations(run_dir)
    manifests = build_manifests(acts, label_dir)
    table = load_split(split_file)
    return Context(acts, manifests, split_masks(table, manifests), split_groups(table, manifests),
                   opacity_labels(manifests))


@dataclass
class DictionarySpec:
    name: str
    kind: str                                    # trained / random_sae / signed_pca / signed_basis
    build: Callable[[], object]
    meta: dict = field(default_factory=dict)     # d_sae, k, steps, init_seed, data_seed, fit_split


def parse_sae_args(items: list[str]) -> dict[str, Path]:
    """``["name=path", ...]`` -> {name: path}. Raises ValueError for an item without '='."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--sae expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def trained_spec(name: str, path: Path, fit_split: str) -> DictionarySpec:
    cfg = json.loads((path / "cfg.json").read_text())
    seed = cfg.get("seed")
    meta = {"d_sae": cfg.get("d_sae", cfg.get("dict_size")), "k": cfg["k"],
            "steps": cfg.get("steps", cfg.get("num_steps")),
            "init_seed": cfg.get("init_seed", seed), "data_seed": cfg.get("data_seed", seed),
            "fit_split": cfg.get("fit_split", fit_split), "path": str(path)}
    return DictionarySpec(name, "trained", lambda: TopKSAE.from_checkpoint(path), meta)


def train_token_sample(ctx: Context, per_dataset: int, seed: int) -> torch.Tensor:
    """Seeded, dataset-balanced sample of tokens from train images only (for PCA)."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, block in ctx.acts.blocks.items():
        train = ctx.masks[name]["train"]
        n_train_tokens = int(train.sum()) * ctx.acts.n_patches
        frac = min(1.0, per_dataset / max(n_train_tokens, 1))
        for img0, x in ctx.acts.iter_token_chunks(name, images_per_chunk=1024):
            n_img = x.shape[0] // ctx.acts.n_patches
            token_train = np.repeat(train[img0:img0 + n_img], ctx.acts.n_patches)
            keep = (rng.random(x.shape[0]) < frac) & token_train
            parts.append(x[torch.from_numpy(keep)])
    return torch.cat(parts)


def null_specs(ctx: Context, d_sae: int, k: int, random_seeds: list[int], pca_tokens: int,
               seed: int) -> list[DictionarySpec]:
    """Untrained random SAEs, signed PCA fitted on train tokens, signed basis; all at top-k = ``k``."""
    d = ctx.acts.d_model
    specs = [DictionarySpec(f"random_seed{s}", "random_sae", (lambda s=s: random_sae(d, d_sae, k, seed=s)),
                            {"d_sae": d_sae, "k": k, "steps": 0, "init_seed": s, "data_seed": None,
                             "fit_split": "none"}) for s in random_seeds]
    specs.append(DictionarySpec("signed_pca", "signed_pca",
                                lambda: signed_pca(train_token_sample(ctx, pca_tokens, seed), d, k),
                                {"d_sae": 2 * d, "k": k, "steps": 0, "init_seed": seed, "data_seed": seed,
                                 "fit_split": "train"}))
    specs.append(DictionarySpec("signed_basis", "signed_basis", lambda: signed_basis(d, k),
                                {"d_sae": 2 * d, "k": k, "steps": 0, "init_seed": None, "data_seed": None,
                                 "fit_split": "none"}))
    return specs


def summarise_latent_group(table: pd.DataFrame, seeds: list[str], threshold: float, null_quantile: float) -> dict:
    """Functional recurrence of one group of latents of an anchor SAE in other seeds (protocol section 4,
    thesis-latent check).

    ``table`` has one row per latent of the group and, per seed s, columns ``one_to_one_<s>`` and ``best_<s>``
    (code correlations; NaN for a latent that never fires). Returns, per seed and as the mean over seeds: the
    share whose one-to-one partner reaches ``threshold`` and ``null_quantile``, the share whose best partner
    reaches ``threshold``, and the median best correlation; plus the shares that recur in every seed.
    Raises ValueError for an empty group.
    """
    if table.empty:
        raise ValueError("empty group")
    out = {"n": int(len(table)), "per_seed": {}}
    for s in seeds:
        o2o, best = table[f"one_to_one_{s}"].fillna(-np.inf), table[f"best_{s}"].fillna(-np.inf)
        out["per_seed"][s] = {"one_to_one_ge_threshold": float((o2o >= threshold).mean()),
                              "one_to_one_ge_null": float((o2o >= null_quantile).mean()),
                              "best_ge_threshold": float((best >= threshold).mean()),
                              "median_best": float(table[f"best_{s}"].median())}
    for key in ("one_to_one_ge_threshold", "one_to_one_ge_null", "best_ge_threshold", "median_best"):
        out[f"mean_{key}"] = float(np.mean([v[key] for v in out["per_seed"].values()]))
    o2o_all = np.column_stack([table[f"one_to_one_{s}"].fillna(-np.inf) for s in seeds])
    best_all = np.column_stack([table[f"best_{s}"].fillna(-np.inf) for s in seeds])
    out["one_to_one_ge_threshold_in_every_seed"] = float((o2o_all >= threshold).all(1).mean())
    out["best_ge_threshold_in_every_seed"] = float((best_all >= threshold).all(1).mean())
    return out


def subset(codes: dict[str, np.ndarray], masks: dict[str, dict[str, np.ndarray]], split: str
           ) -> dict[str, np.ndarray]:
    return {d: codes[d][masks[d][split]] for d in codes}


def tidy_row(spec: DictionarySpec, metric: str, subset_name: str, split: str, value: float,
             ci_low: float | None = None, ci_high: float | None = None) -> dict:
    m = spec.meta
    return {"harvest": "thesis", "encoder": "umedpt", "layer": "final", "dictionary": spec.name,
            "d_sae": m.get("d_sae"), "k": m.get("k"), "steps": m.get("steps"), "init_seed": m.get("init_seed"),
            "data_seed": m.get("data_seed"), "metric": metric, "subset": subset_name, "split": split,
            "value": value, "ci_low": ci_low, "ci_high": ci_high}
