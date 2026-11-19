"""The seed-overlap sweep (protocol section 5): run table, resumable training, per-setting evaluation.

Run table (``configs/sweep_v1.csv``): one row per training run, generated from the protocol's phase table
by ``sweep_table``. A *setting* is a group of runs that differ only in their seeds; stability is measured
between the runs of one setting. Phase 2 rows (for the settings the protocol carries forward) and then the
phase 1c rows (the k extension) are appended after phase 1.

Two deliberate readings of the phase table (also in the WP1 notes):
  2b   the 2 x 2 design of initialisation seed x batch-order seed reuses the phase 1 runs (0, 0) and
       (1, 1) of the same setting, so only (0, 1) and (1, 0) are new runs.
  1n   the frozen-random-decoder null of P1.1 (Korznikov) is trained once per k at 4,096 latents, so
       P1.1 sees every null the protocol lists.

Runs are trained on train-split tokens only. A run directory holds ``step_<n>/`` checkpoints (TopKSAE
format), ``config.json``, ``train.log``, ``state.pt`` while unfinished and ``DONE.json`` when finished.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from .dictionaries.sae import TopKSAE
from .dictionaries.train import BatchTopKTrainer, TrainConfig

SWEEP_COLUMNS = ["row", "phase", "setting", "run_id", "decoder", "d_sae", "k", "steps", "init_seed",
                 "data_seed", "sampling", "l1_coeff", "data_fraction", "checkpoints", "run_units"]
PHASE1_KS = (4, 8, 16, 32, 48)
# Phase 1c: larger k at 1,024 latents, added after the decoder diagnostic had no minimum up to k = 48.
EXTENSION_D_SAE = 1024
EXTENSION_KS = (64, 96)
BASE_D_SAE = 4096
BASE_STEPS = 20_000
BASE_L1 = TrainConfig().l1_coeff
BASE_CHECKPOINTS = (5_000, 10_000, 20_000)
LONG_STEPS = 80_000
LONG_CHECKPOINTS = (20_000, 40_000, 80_000)

# Settings carried forward, as (d_sae, k); roles and rules are in protocol section 5.
CORE_SETTINGS = ((1024, 8), (1024, 16), (1024, 32), (4096, 32))
# The setting of the thesis SAE, which the protocol's outcome table names.
THESIS_SETTING = (4096, 48)
# (1024, 64) becomes a core setting instead if the label-transfer rule of protocol section 5 says so.
DENSE_REFERENCES = ((4096, 48), (1024, 64))
APPENDIX_SETTINGS = ((1024, 48), (1024, 96), (4096, 4), (4096, 8), (4096, 16))
PHASE2_ALL_TESTS = ((1024, 8), (1024, 16), (1024, 32))
PHASE2_TRAINING_LENGTH_ONLY = ((4096, 32), (4096, 48))


def setting_role(d_sae: int, k: int) -> str:
    """Role of a (d_sae, k) pair: "core", "dense reference", "appendix" or "not carried forward"."""
    if (d_sae, k) in CORE_SETTINGS:
        return "core"
    if (d_sae, k) in DENSE_REFERENCES:
        return "dense reference"
    if (d_sae, k) in APPENDIX_SETTINGS:
        return "appendix"
    return "not carried forward"


def run_units(d_sae: int, steps: int) -> float:
    """Cost relative to one run unit (4,096 latents for 20k steps); time per step grows ~linearly in d_sae."""
    return d_sae / BASE_D_SAE * steps / BASE_STEPS


def setting_name(d_sae: int, k: int, steps: int, variant: str = "") -> str:
    return f"d{d_sae}_k{k}_s{steps}" + (f"_{variant}" if variant else "")


def _row(phase: str, d_sae: int, k: int, init_seed: int, data_seed: int, *, steps: int = BASE_STEPS,
         sampling: str = "equal", l1_coeff: float = BASE_L1, data_fraction: float = 1.0,
         decoder: str = "trained", checkpoints: tuple[int, ...] = BASE_CHECKPOINTS, variant: str = "",
         setting: str | None = None) -> dict:
    setting = setting or setting_name(d_sae, k, steps, variant)
    run_id = f"{setting_name(d_sae, k, steps, variant)}_i{init_seed}_b{data_seed}"
    return {"phase": phase, "setting": setting, "run_id": run_id, "decoder": decoder, "d_sae": d_sae, "k": k,
            "steps": steps, "init_seed": init_seed, "data_seed": data_seed, "sampling": sampling,
            "l1_coeff": l1_coeff, "data_fraction": data_fraction,
            "checkpoints": ";".join(str(c) for c in checkpoints), "run_units": run_units(d_sae, steps)}


def _training_length_rows(d_sae: int, k: int) -> list[dict]:
    """Phase 2a at one setting: three seeds trained for the long schedule."""
    return [_row("2a", d_sae, k, s, s, steps=LONG_STEPS, checkpoints=LONG_CHECKPOINTS) for s in range(3)]


def _other_phase2_rows(d_sae: int, k: int) -> list[dict]:
    """Phases 2b, 2c and 2d at one setting."""
    # The 2 x 2 design shares its (0, 0) and (1, 1) cells with phase 1; see the module docstring.
    seedsplit = setting_name(d_sae, k, BASE_STEPS, "seedsplit")
    rows = [_row("2b", d_sae, k, i, b, setting=seedsplit) for i, b in ((0, 1), (1, 0))]
    for frac, tag in ((0.25, "frac25"), (0.5, "frac50")):
        rows += [_row("2c", d_sae, k, s, s, data_fraction=frac, variant=tag) for s in range(3)]
    rows += [_row("2d", d_sae, k, s, s, l1_coeff=0.0, variant="l1off") for s in range(3)]
    rows += [_row("2d", d_sae, k, s, s, sampling="proportional", variant="prop") for s in range(3)]
    return rows


def sweep_table(full: bool = False) -> pd.DataFrame:
    """The run table: phases 1a, 1b and 1n, followed, if ``full`` is true, by phase 2 and then phase 1c.

    Rows are only ever appended in the order they were decided, so row numbers of finished runs never
    change; that is why phase 1c comes after phase 2.
    Seeds: a phase-1 seed s sets both the initialisation and the batch order (as the Stage 0.6 seeds did).
    """
    rows = []
    for k in PHASE1_KS:
        rows += [_row("1a", BASE_D_SAE, k, s, s) for s in range(5)]
    for k in PHASE1_KS:
        rows += [_row("1b", 1024, k, s, s) for s in range(5)]
    rows += [_row("1b", 16_384, 16, s, s) for s in range(3)]
    for k in PHASE1_KS:
        rows.append(_row("1n", BASE_D_SAE, k, 0, 0, decoder="frozen_isotropic", variant="frozeniso"))
    if full:
        for d_sae, k in PHASE2_ALL_TESTS:
            rows += _training_length_rows(d_sae, k) + _other_phase2_rows(d_sae, k)
        for d_sae, k in PHASE2_TRAINING_LENGTH_ONLY:
            rows += _training_length_rows(d_sae, k)
        for k in EXTENSION_KS:
            rows += [_row("1c", EXTENSION_D_SAE, k, s, s) for s in range(5)]
    table = pd.DataFrame(rows)
    table.insert(0, "row", np.arange(len(table)))
    if not table.run_id.is_unique:
        raise AssertionError("duplicate run ids in the sweep table")
    return table[SWEEP_COLUMNS]


def parse_rows(spec: str, n_rows: int) -> list[int]:
    """``"a:b"`` (half-open, either end optional) or ``"3,5,9"`` -> row numbers. Raises ValueError."""
    if ":" in spec:
        a, b = spec.split(":", 1)
        lo, hi = int(a) if a else 0, int(b) if b else n_rows
        if not 0 <= lo <= hi <= n_rows:
            raise ValueError(f"row range {spec!r} outside 0:{n_rows}")
        return list(range(lo, hi))
    rows = [int(r) for r in spec.split(",") if r]
    if any(not 0 <= r < n_rows for r in rows):
        raise ValueError(f"rows {spec!r} outside 0:{n_rows}")
    return rows


def checkpoint_steps(row) -> tuple[int, ...]:
    return tuple(int(c) for c in str(row["checkpoints"]).split(";") if c)


def seedsplit_base_rows(table: pd.DataFrame, d_sae: int, k: int) -> pd.DataFrame:
    """The phase 1 runs (0, 0) and (1, 1) that complete the 2 x 2 design of phase 2b at one setting."""
    base = table[table.phase.isin(("1a", "1b", "1c")) & (table.d_sae == d_sae) & (table.k == k)]
    return base[[(i, b) in ((0, 0), (1, 1)) for i, b in zip(base.init_seed, base.data_seed)]]


# ---------------------------------------------------------------- training data


def token_rows(image_mask: np.ndarray, n_patches: int) -> np.ndarray:
    """Token row indices of the selected images: image i owns rows [P*i, P*i + P) of its dataset block."""
    return (np.flatnonzero(image_mask)[:, None] * n_patches + np.arange(n_patches)).ravel()


def subset_train_images(train_mask: dict[str, np.ndarray], groups: dict[str, np.ndarray], fraction: float,
                        seed: int = 0) -> dict[str, np.ndarray]:
    """A share ``fraction`` of the train images per dataset, taken as whole groups (patients).

    Groups are visited in one seeded order per dataset and added until the share of images is reached,
    so a smaller fraction is a subset of a larger one (25% inside 50%). Raises ValueError for a fraction
    outside (0, 1].
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return {d: m.copy() for d, m in train_mask.items()}
    out = {}
    for d_index, d in enumerate(sorted(train_mask)):
        mask = train_mask[d]
        g = pd.Series(groups[d][mask])
        sizes = g.value_counts().sort_index()
        order = np.random.default_rng([seed, d_index]).permutation(len(sizes))
        cum = np.cumsum(sizes.to_numpy()[order])
        n_keep = int(np.searchsorted(cum, fraction * mask.sum())) + 1
        keep_groups = set(sizes.index[order[:n_keep]])
        out[d] = mask & np.isin(groups[d], list(keep_groups))
    return out


# ---------------------------------------------------------------- one run


def save_dictionary(sae: TopKSAE, out: Path, cfg_dict: dict) -> None:
    """TopKSAE checkpoint layout (``sae_weights.safetensors`` + ``cfg.json``), written atomically."""
    tmp = out.with_name(out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    save_file({"W_enc": sae.W_enc.contiguous(), "b_enc": sae.b_enc, "W_dec": sae.W_dec.contiguous(),
               "b_dec": sae.b_dec}, str(tmp / "sae_weights.safetensors"))
    (tmp / "cfg.json").write_text(json.dumps(cfg_dict, indent=2))
    if out.exists():
        for f in out.iterdir():
            f.unlink()
        out.rmdir()
    tmp.rename(out)


def train_config(row) -> TrainConfig:
    return TrainConfig(d_sae=int(row["d_sae"]), k=int(row["k"]), steps=int(row["steps"]),
                       seed=int(row["init_seed"]), init_seed=int(row["init_seed"]),
                       data_seed=int(row["data_seed"]), sampling=str(row["sampling"]),
                       l1_coeff=float(row["l1_coeff"]), checkpoint_steps=checkpoint_steps(row))


@dataclass
class RunResult:
    run_id: str
    status: str          # "done", "skipped" (already done) or "paused" (time budget used up)
    step: int
    seconds: float


def train_run(row, data: dict[str, np.ndarray], train_rows: dict[str, np.ndarray], out_root: Path,
              device: str, deadline: float, log: Callable[[str], None], state_every: int = 2_000,
              run_info: dict | None = None) -> RunResult:
    """Train (or continue) one row of the sweep table into ``out_root / run_id``.

    The state is saved every ``state_every`` steps and when ``time.time()`` passes ``deadline``; a later
    call continues from it. A run with ``DONE.json`` is skipped. ``run_info`` is merged into
    ``config.json`` (split hash, git revision, ...).
    Raises ValueError for an unknown decoder kind.
    """
    out = Path(out_root) / row["run_id"]
    if (out / "DONE.json").exists():
        return RunResult(row["run_id"], "skipped", int(row["steps"]), 0.0)
    out.mkdir(parents=True, exist_ok=True)
    cfg = train_config(row)
    fixed = None
    if row["decoder"] == "frozen_isotropic":
        from .dictionaries.baselines import isotropic_directions
        # Decoder seed = initialisation seed: the decoder is part of the run's initial state.
        fixed = isotropic_directions(cfg.d_sae, cfg.d_model, int(row["init_seed"]))
    elif row["decoder"] != "trained":
        raise ValueError(f"unknown decoder kind {row['decoder']!r}")
    cfg_dict = {**asdict(cfg), "architecture": "batch_topk", "decoder": row["decoder"], "fit_split": "train",
                "data_fraction": float(row["data_fraction"]), "phase": row["phase"],
                "setting": row["setting"], "run_id": row["run_id"], "device": device, **(run_info or {})}
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))

    def on_checkpoint(step: int, sae: TopKSAE) -> None:
        save_dictionary(sae, out / f"step_{step}", {**cfg_dict, "step": step})

    trainer = BatchTopKTrainer(data, cfg, device, log, train_rows, on_checkpoint, fixed)
    state_path = out / "state.pt"
    if state_path.exists():
        trainer.load_state_dict(torch.load(state_path, weights_only=False))
        log(f"resumed at step {trainer.step}")
    t0 = time.time()
    over = lambda: time.time() > deadline
    while not trainer.finished:
        target = (trainer.step // state_every + 1) * state_every
        trainer.run(until_step=target, should_stop=over)
        if trainer.finished:
            break
        tmp = state_path.with_suffix(".tmp")
        torch.save(trainer.state_dict(), tmp)
        os.replace(tmp, state_path)
        if over():
            log(f"time budget used up at step {trainer.step}; state saved")
            return RunResult(row["run_id"], "paused", trainer.step, time.time() - t0)
    if cfg.steps not in cfg.checkpoint_steps:
        on_checkpoint(cfg.steps, trainer.snapshot())
    (out / "DONE.json").write_text(json.dumps({"finished": True, "steps": cfg.steps,
                                               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    state_path.unlink(missing_ok=True)
    return RunResult(row["run_id"], "done", trainer.step, time.time() - t0)


def run_dir_checkpoint(out_root: Path, run_id: str, step: int) -> Path:
    return Path(out_root) / run_id / f"step_{step}"


# ---------------------------------------------------------------- evaluation helpers


def seed_pairs(names: list[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(names, 2))


def overlap_share(scores: np.ndarray, threshold: float) -> float:
    """Share of latents whose one-to-one partner score reaches ``threshold``; unmatched (NaN) count as no."""
    return float(np.mean(np.nan_to_num(scores, nan=-np.inf) >= threshold))


def summarise_pairs(values: Iterable[float]) -> dict:
    v = np.asarray(list(values), dtype=float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "n_pairs": int(v.size)}


def p11_supported(p11: pd.DataFrame, margin: float, max_k: int) -> bool:
    """P1.1 (protocol section 4): complete separation and the margin at every evaluated setting with k <= ``max_k``.

    ``p11`` has columns ``k, gap, separated``, one row per evaluated setting; appendix settings count.
    Raises ValueError if no setting has k <= ``max_k``.
    """
    decisive = p11[p11.k <= max_k]
    if decisive.empty:
        raise ValueError(f"no evaluated setting has k <= {max_k}")
    return bool(decisive.separated.astype(bool).all() and (decisive.gap >= margin).all())


def training_length_table(curve: pd.DataFrame) -> pd.DataFrame:
    """Phase 2a per setting: S of the 80k runs against S of the 20k runs of phase 1.

    ``curve`` has columns ``setting, steps, S_mean``. The 80k runs' own 20k checkpoint sits halfway through
    a longer learning-rate schedule, so it is listed but does not enter ``delta``. Settings whose 80k or
    20k runs have not been evaluated are left out.
    """
    S = {(r.setting, int(r.steps)): float(r.S_mean) for r in curve.itertuples()}
    rows = []
    for d_sae, k in PHASE2_ALL_TESTS + PHASE2_TRAINING_LENGTH_ONLY:
        short, long = setting_name(d_sae, k, BASE_STEPS), setting_name(d_sae, k, LONG_STEPS)
        if (short, BASE_STEPS) not in S or (long, LONG_STEPS) not in S:
            continue
        rows.append({"setting": short, "d_sae": d_sae, "k": k, "role": setting_role(d_sae, k),
                     "S_20k_run": S[short, BASE_STEPS], "S_80k_run": S[long, LONG_STEPS],
                     "S_80k_run_at_20k": S.get((long, BASE_STEPS), math.nan),
                     "delta": S[long, LONG_STEPS] - S[short, BASE_STEPS]})
    return pd.DataFrame(rows, columns=["setting", "d_sae", "k", "role", "S_20k_run", "S_80k_run",
                                       "S_80k_run_at_20k", "delta"])


def summarise_decoder_diagnostic(runs: pd.DataFrame, fve: pd.DataFrame | None = None) -> pd.DataFrame:
    """Decoder pairwise cosine per setting (protocol section 5): mean and min-max over seeds, with the role.

    ``runs`` has columns ``d_sae, k, step, c_dec``, one row per run. ``fve`` (columns ``d_sae, k, step,
    dictionary_kind, n_dead``) adds the mean number of latents dead on val, because the decoders of dead
    latents stay near their initial directions and pull the metric down.
    """
    rows = []
    for (d_sae, k, step), part in runs.groupby(["d_sae", "k", "step"]):
        row = {"setting": setting_name(int(d_sae), int(k), BASE_STEPS), "d_sae": int(d_sae), "k": int(k),
               "role": setting_role(int(d_sae), int(k)), "step": int(step), "n_runs": len(part),
               "c_dec_mean": float(part.c_dec.mean()), "c_dec_min": float(part.c_dec.min()),
               "c_dec_max": float(part.c_dec.max())}
        if fve is not None:
            trained = fve[(fve.d_sae == d_sae) & (fve.k == k) & (fve.step == step)
                          & (fve.dictionary_kind == "trained")
                          & (fve.setting == setting_name(int(d_sae), int(k), BASE_STEPS))]
            row["mean_dead_on_val"] = float(trained.n_dead.mean()) if len(trained) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def per_latent_pair_table(cosine: np.ndarray, partner: np.ndarray, image_rate_a: np.ndarray,
                          image_rate_b: np.ndarray) -> pd.DataFrame:
    """One row per latent of run A: its one-to-one partner in run B, their decoder cosine and firing rates.

    ``cosine`` and ``partner`` come from ``one_to_one`` (NaN and -1 where a latent has no partner); the rates
    are image shares from ``firing_rates``. Raises ValueError if the lengths of A's arrays differ.
    """
    n = len(cosine)
    if not len(partner) == len(image_rate_a) == n:
        raise ValueError("cosine, partner and image_rate_a must have one entry per latent of run A")
    has_partner = partner >= 0
    rate_b = np.full(n, np.nan)
    rate_b[has_partner] = image_rate_b[partner[has_partner]]
    return pd.DataFrame({"latent": np.arange(n), "partner": partner, "cosine": cosine,
                         "image_rate_a": image_rate_a, "image_rate_partner": rate_b})


def summarise_per_latent(pairs: list[pd.DataFrame], threshold: float) -> dict:
    """Per-latent summary of one setting over its seed pairs (protocol section 5, per-latent check).

    For each pair: S over all latents (as P1.3), S over the latents of run A that fire on at least one image,
    the share of recurring latents that fire in both runs, and the Spearman correlation between a live
    latent's image firing rate and its partner cosine. Returns the mean and min-max over pairs of each, and
    the median firing rate of recurring and of non-recurring live latents, pooled over pairs.
    Raises ValueError for an empty list.
    """
    from scipy.stats import spearmanr

    if not pairs:
        raise ValueError("need at least one seed pair")
    per_pair, recurring_rates, other_rates = [], [], []
    for t in pairs:
        cos = np.nan_to_num(t.cosine.to_numpy(), nan=-np.inf)
        recurring = cos >= threshold
        live = t.image_rate_a.to_numpy() > 0
        live_both = live & (np.nan_to_num(t.image_rate_partner.to_numpy(), nan=0.0) > 0)
        rho = spearmanr(t.image_rate_a[live], t.cosine[live], nan_policy="omit").statistic if live.sum() > 2 else np.nan
        per_pair.append({"S_all": recurring.mean(), "S_live": recurring[live].mean() if live.any() else np.nan,
                         "recurring_live_in_both": live_both[recurring].mean() if recurring.any() else np.nan,
                         "spearman_rate_cosine": float(rho), "n_live_a": int(live.sum())})
        recurring_rates.append(t.image_rate_a[recurring & live])
        other_rates.append(t.image_rate_a[~recurring & live])
    table = pd.DataFrame(per_pair)
    out = {"n_pairs": len(pairs)}
    for col in table.columns:
        out[col] = {"mean": float(table[col].mean()), "min": float(table[col].min()), "max": float(table[col].max())}
    out["median_image_rate_recurring"] = float(pd.concat(recurring_rates).median()) if len(pd.concat(recurring_rates)) else math.nan
    out["median_image_rate_not_recurring"] = float(pd.concat(other_rates).median()) if len(pd.concat(other_rates)) else math.nan
    return out


def label_transfer_table(summaries: Mapping[str, dict], ms: tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    """P1.2's statistic per setting from the probe outputs: mean over trained seeds, and trained vs nulls.

    ``summaries`` maps a setting name (``d<d_sae>_k<k>_s<steps>``) to the ``summary.json`` content of
    ``stage0_metrics.py --protocol v1`` run on that setting's seeds. Trained seeds are the dictionaries
    without a null name. Raises KeyError if a summary lacks a probe value for one of ``ms``.
    """
    null_names = ("random_seed", "signed_pca", "signed_basis")
    rows = []
    for setting, summary in summaries.items():
        d_sae, k = int(setting.split("_")[0][1:]), int(setting.split("_k")[1].split("_")[0])
        trained = {n: v for n, v in summary["dictionaries"].items() if not n.startswith(null_names)}
        row = {"setting": setting, "d_sae": d_sae, "k": k, "role": setting_role(d_sae, k), "n_seeds": len(trained)}
        for m in ms:
            values = [v[f"probe_mean_cross_auroc_m{m}"][0] for v in trained.values()]
            comparison = summary.get("trained_vs_null", {}).get(f"probe_m{m}", {})
            row.update({f"m{m}_mean": float(np.mean(values)), f"m{m}_min": float(np.min(values)),
                        f"m{m}_max": float(np.max(values)), f"m{m}_best_null": comparison.get("best_null"),
                        f"m{m}_separated": comparison.get("separated"), f"m{m}_gap": comparison.get("gap")})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["d_sae", "k"]).reset_index(drop=True)


def extension_k_role(table: pd.DataFrame, candidate: tuple[int, int] = (EXTENSION_D_SAE, EXTENSION_KS[0]),
                     ms: tuple[int, ...] = (1, 5)) -> str:
    """The protocol's label-transfer rule for the k extension (section 5): "core" if the candidate's mean
    statistic is above every core setting's for every m in ``ms``, otherwise "dense reference".

    Raises ValueError if the candidate or a core setting is missing from ``table``.
    """
    index = {(int(r.d_sae), int(r.k)): r for r in table.itertuples()}
    missing = [s for s in (candidate, *CORE_SETTINGS) if s not in index]
    if missing:
        raise ValueError(f"label-transfer results missing for {missing}")
    above = all(getattr(index[candidate], f"m{m}_mean") > getattr(index[s], f"m{m}_mean")
                for m in ms for s in CORE_SETTINGS)
    return "core" if above else "dense reference"


def classify_outcome(S_by_setting: dict[str, float], thesis_setting: str, small_settings: list[str],
                     under_training_deltas: Mapping[str, float] | None, thresholds: dict[str, float]) -> str:
    """Return "operating point", "under-training", "robust low" or "in between" (protocol section 5).

    Parameters:
      S_by_setting          mean S per setting name (all settings that were run)
      thesis_setting        the k = 48 / 4,096 setting
      small_settings        the settings at k <= 16 or 1,024 latents
      under_training_deltas S(80k) - S(20k) per phase 2a setting; None or empty until phase 2a is complete
      thresholds            ``high``, ``low`` and ``under_training`` from the protocol table
    Rows are checked in the table's order; an earlier row that applies wins. Raises KeyError if the
    thesis setting has no S.
    """
    hi, lo, ut = thresholds["high"], thresholds["low"], thresholds["under_training"]
    thesis = S_by_setting[thesis_setting]
    if any(S_by_setting.get(s, -math.inf) >= hi for s in small_settings) and thesis < lo:
        return "operating point"
    if under_training_deltas and any(delta >= ut for delta in under_training_deltas.values()):
        return "under-training"
    if under_training_deltas and all(v < lo for v in S_by_setting.values()):
        return "robust low"
    return "in between"


def load_tokens_fp16(acts) -> dict[str, np.ndarray]:
    """All tokens per dataset as float16 (about 2.7 GB for the thesis harvest), as the Stage 0.6 trainer
    loaded them; the sampler indexes into these arrays with ``train_rows``."""
    data = {}
    for name, block in acts.blocks.items():
        arr = np.empty((block.n_tokens, acts.d_model), dtype=np.float16)
        pos = 0
        for _, x in acts.iter_token_chunks(name, 1024):
            arr[pos:pos + len(x)] = x.numpy().astype(np.float16)
            pos += len(x)
        data[name] = arr
    return data
