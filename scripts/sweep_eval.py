"""Evaluate the sweep (protocol sections 4-5): P1.1 and P1.3 per setting and checkpoint, secondary metrics.

For every setting whose runs are all finished, and every checkpoint step they share:
  - per run: own-decoder FVE on the eval tokens with a cluster-bootstrap interval; dead latents; latents
    active on every image
  - all seed pairs: one-to-one decoder cosine; S (share at the P1.3 threshold) and its min-max over pairs
  - data-aware null, anchored on the setting's first run: covariance-shaped random directions; its q99.9
    and the share of anchor latents above it in every other seed
  - functional matching on token-level sparse codes, anchor against every other run, with its own null
  - phase 2b: pairs split into "init only", "batch order only" and "both" differing
Per (d_sae, k): the nulls of P1.1 at matched sparsity (random SAEs, signed PCA and signed basis fitted on
train tokens, and the frozen-random-decoder run of phase 1n where it exists).

``--phase0`` adds the Stage 0.6 seeds 0-2 (fitted on all images) as setting ``phase0_alldata_d4096_k48``.
``--further`` adds the further measures of protocol section 4 (2026-11-25) at each setting's last checkpoint:
joint seed overlap and PW-MCC per seed pair; per dictionary (trained runs and the matched-sparsity nulls)
performance recovery, absorption on the CheXpert hierarchy and targeted ablation of confounders
(``<out>/further/``). On the development data, latents and probes are selected on train (C tuned on val)
and measured on val, because the test split is not read here. P1.1 for the phase 3 architectures at
their measured L0 goes to ``p11_architectures.csv``.
``--probes`` additionally runs ``stage0_metrics.py --protocol v1`` on the final checkpoints of the listed
settings (P1.2 and selectivity, on the val split). ``--per-latent`` adds the per-latent check of protocol
section 5 (``<out>/per_latent/``). ``--subspace`` adds subspace reproducibility (``<out>/subspace/``). ``--skip-sweep-metrics`` runs only these
additional checks.

Results are cached per (setting, step) under ``<out>/cache``; a cached item is not recomputed unless
``--force``. Outputs: ``results.csv`` (tidy), ``pairs.csv``, ``fve.csv``, ``p11.csv``.
The eval tokens are those of the val split; the test split is never read here.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/sweep_eval.py --phase0
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR
from medinterp.data.manifests import DEFAULT_LABEL_DIR
from medinterp.dictionaries.baselines import covariance_directions, random_sae, signed_basis, signed_pca
from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.stability import TokenMoments, cosine_matrix, one_to_one
from medinterp.eval.further_measures import (RecoveryProbes, absorption, decode_pooled, image_features,
                                             joint_seed_overlap, pooled_activations, secondary_targets,
                                             targeted_ablation)
from medinterp.eval.stats import complete_separation
from medinterp.eval.subspace import (effective_rank, explained_variance_in, matched_subset, overlap_of,
                                     project_out, right_singular_basis, whitening_map,
                                     principal_subspace_explained_variance,
                                     reappearance_probability, stability_sets, subspace_explained_variance)
from medinterp.eval.sweep_metrics import (covariance_null, decoder_overlap, firing_rates, functional_null,
                                          functional_overlap, load_eval_tokens, score_dictionary, EvalTokens)
from medinterp.runinfo import write_config
from medinterp.stage0_v1 import TIDY_COLUMNS, load_context, train_token_sample
from medinterp.sweep import (remove_component, top_component, train_token_covariance, checkpoint_steps, overlap_share, per_latent_pair_table, seedsplit_base_rows,
                             summarise_pairs, summarise_per_latent)

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
EVAL_SPLIT = "val"
# Protocol values (sections 3-5): P1.3 threshold, null quantile, bootstrap settings, random-SAE seeds.
S_THRESHOLD = 0.7
NULL_QUANTILE = 0.999
N_BOOT, CONFIDENCE = 2000, 0.95
NULL_SEEDS = (0, 1)
RANDOM_SAE_SEEDS = (0, 1, 2, 3, 4)
PCA_TOKENS_PER_DATASET = 50_000
SEED = 0
PHASE0 = "phase0_alldata_d4096_k48"
# Protocol section 5, subspace reproducibility: endpoint cutoff and the ranks of the compared subspaces.
SUBSPACE_EPSILON = 0.05
SUBSPACE_RANKS = (8, 16, 32, 64, 128)
SUBSET_NULL_DRAWS = 20          # latent-subset null of the subspace robustness checks (protocol section 5)
SUBSET_NULL_QUANTILE = 0.95
WHITENING_VARIANCE_SHARE = 0.99
ROBUST_PHASES = ("1a", "1b", "1c")
ROBUST_SEEDS = 5
# Protocol section 4, further measures (2026-11-25): ablated latents per confounder and the confounders.
ABLATION_NS = (5, 20)
CONFOUNDERS = ("Support Devices", "sex", "view")
PHASE3_ARCHITECTURE_PHASES = ("3m", "3r")
# The development tokens are centred per dataset with the mean over all of a dataset's images, so train and val
# means of a dataset mirror each other and a dataset-identity probe anti-predicts on val (AUROC < 0.5 on the
# originals). Performance recovery leaves those probes out on such tokens (protocol section 4, change log
# 2026-11-25).
RECOVERY_EXCLUDED_PREFIXES = ("dataset_",)


def recovery_mean(rows: list[dict]) -> float | None:
    """Mean recovered share over the probes that count for performance recovery (protocol section 4)."""
    kept = [r["recovered_share"] for r in rows if not r["target"].startswith(RECOVERY_EXCLUDED_PREFIXES)]
    return float(np.nanmean(kept)) if kept else None


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


LOG_FILE = None


def find_run(run_id: str, runs_roots: list[Path]) -> Path | None:
    """The first root that holds a finished ``run_id``, or None."""
    return next((root / run_id for root in runs_roots if (root / run_id / "DONE.json").exists()), None)


def collect_settings(table: pd.DataFrame, runs_roots: list[Path], phase0_dir: Path | None) -> dict[str, dict]:
    """``{setting: {"meta": ..., "runs": {run_id: {"init_seed", "data_seed", "steps": {step: path}}}}}``
    for settings whose runs are all finished. Frozen-decoder settings are returned with kind "null".

    Several run roots are accepted because a setting's runs can come from different training outputs:
    phase 2b reuses two phase 1 runs, which sit in another Kaggle run's output than the phase 2 runs.
    """
    out = {}
    for setting, part in table.groupby("setting", sort=False):
        first = part.iloc[0]
        runs = {}
        rows = list(part.itertuples())
        if first.phase == "2b":
            # The 2 x 2 design reuses the phase 1 runs (0, 0) and (1, 1) of the same setting.
            rows = list(seedsplit_base_rows(table, int(first.d_sae), int(first.k)).itertuples()) + rows
        found = {r.run_id: find_run(r.run_id, runs_roots) for r in rows}
        if any(d is None for d in found.values()):
            continue
        for r in rows:
            runs[r.run_id] = {"init_seed": int(r.init_seed), "data_seed": int(r.data_seed),
                              "steps": {s: found[r.run_id] / f"step_{s}" for s in checkpoint_steps(r._asdict())}}
        out[setting] = {"meta": {"phase": first.phase, "d_sae": int(first.d_sae), "k": int(first.k),
                                 "kind": "null" if first.decoder != "trained" else "trained",
                                 "decoder": first.decoder,
                                 "projection": str(getattr(first, "projection", "none") or "none")},
                        "runs": runs}
    if phase0_dir is not None:
        seeds = phase0_dir
        out = {PHASE0: {"meta": {"phase": "0", "d_sae": 4096, "k": 48, "kind": "trained", "decoder": "trained"},
                        "runs": {f"alldata_seed{s}": {"init_seed": s, "data_seed": s,
                                                      "steps": {20_000: seeds / f"seed{s}"}}
                                 for s in (0, 1, 2)}},
               **out}
    return out


def train_covariance(ctx, cache: Path) -> torch.Tensor:
    if cache.exists():
        return torch.load(cache)
    cov = train_token_covariance(ctx.acts, ctx.masks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cov, cache)
    return cov


def tidy(dictionary: str, d_sae, k, steps, init_seed, data_seed, metric: str, subset: str, value,
         ci_low=None, ci_high=None, split: str = EVAL_SPLIT) -> dict:
    return {"harvest": "thesis", "encoder": "umedpt", "layer": "final", "dictionary": dictionary, "d_sae": d_sae,
            "k": k, "steps": steps, "init_seed": init_seed, "data_seed": data_seed, "metric": metric,
            "subset": subset, "split": split, "value": value, "ci_low": ci_low, "ci_high": ci_high}


def score_row(name, kind, setting, d_sae, k, step, init_seed, data_seed, sc) -> tuple[dict, list[dict]]:
    fve = {"setting": setting, "dictionary": name, "dictionary_kind": kind, "d_sae": d_sae, "k": k, "step": step,
           "init_seed": init_seed, "data_seed": data_seed, "value": sc.fve, "ci_low": sc.fve_ci[0],
           "ci_high": sc.fve_ci[1], "n_dead": sc.n_dead, "n_every_image": sc.n_every_image,
           "mean_l0": float(sc.firing_rate.sum())}
    rows = [tidy(name, d_sae, k, step, init_seed, data_seed, "fve_own_decoder", "all", sc.fve, *sc.fve_ci),
            tidy(name, d_sae, k, step, init_seed, data_seed, "n_dead_latents", "all", sc.n_dead),
            tidy(name, d_sae, k, step, init_seed, data_seed, "n_latents_active_every_image", "all", sc.n_every_image),
            tidy(name, d_sae, k, step, init_seed, data_seed, "mean_l0", "all", float(sc.firing_rate.sum()))]
    return fve, rows


def evaluate_setting(setting: str, info: dict, step: int, tokens, cov, device: str, functional_max: int) -> dict:
    """All measurements of one setting at one checkpoint step (see the module docstring)."""
    meta, runs = info["meta"], info["runs"]
    d_sae, k = meta["d_sae"], meta["k"]
    order = sorted(runs, key=lambda r: (runs[r]["init_seed"], runs[r]["data_seed"]))
    saes = {r: TopKSAE.from_checkpoint(runs[r]["steps"][step]) for r in order}
    result = {"fve": [], "tidy": [], "pairs": []}
    for r in order:
        sc = score_dictionary(saes[r], tokens, N_BOOT, CONFIDENCE, SEED, device)
        f, t = score_row(r, meta["kind"] if meta["kind"] == "trained" else meta["decoder"], setting, d_sae, k, step,
                         runs[r]["init_seed"], runs[r]["data_seed"], sc)
        result["fve"].append(f)
        result["tidy"] += t
        log(f"{setting} step {step} {r}: FVE {sc.fve:.4f} [{sc.fve_ci[0]:.4f}, {sc.fve_ci[1]:.4f}], "
            f"dead {sc.n_dead}, every image {sc.n_every_image}")
    if meta["kind"] != "trained" or len(order) < 2:
        return result

    anchor = order[0]
    null = covariance_null(saes[anchor].W_dec, cov, NULL_SEEDS)
    null_q = float(np.nanquantile(null, NULL_QUANTILE))
    s_vals, s_null_vals = [], []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            cos = decoder_overlap(saes[a].W_dec, saes[b].W_dec)
            s, s_null = overlap_share(cos, S_THRESHOLD), overlap_share(cos, null_q)
            s_vals.append(s)
            s_null_vals.append(s_null)
            ra, rb = runs[a], runs[b]
            differs = ("init only" if ra["data_seed"] == rb["data_seed"] else
                       "batch order only" if ra["init_seed"] == rb["init_seed"] else "both")
            result["pairs"].append({"setting": setting, "step": step, "match": "decoder", "run_a": a, "run_b": b,
                                    "seeds_differ": differs, "mean": float(np.nanmean(cos)),
                                    f"share_ge_{S_THRESHOLD}": s, "share_ge_null_q999": s_null,
                                    "null_q999": null_q})
    S = summarise_pairs(s_vals)
    S_null = summarise_pairs(s_null_vals)
    null_share = overlap_share(null, S_THRESHOLD)
    result["S"] = S
    result["null"] = {"q999": null_q, f"share_ge_{S_THRESHOLD}": null_share, "mean": float(np.nanmean(null))}
    result["tidy"] += [
        tidy(setting, d_sae, k, step, None, None, "seed_overlap_S", "all_pairs", S["mean"], S["min"], S["max"], "none"),
        tidy(setting, d_sae, k, step, None, None, "share_above_covariance_null_q999", "all_pairs",
             S_null["mean"], S_null["min"], S_null["max"], "none"),
        tidy(setting, d_sae, k, step, None, None, "covariance_null_q999", "anchor", null_q, split="none"),
        tidy(setting, d_sae, k, step, None, None, f"covariance_null_share_ge_{S_THRESHOLD}", "anchor", null_share,
             split="none"),
    ]
    log(f"{setting} step {step}: S = {S['mean']:.4f} ({S['min']:.4f}-{S['max']:.4f}, {S['n_pairs']} pairs); "
        f"null q99.9 {null_q:.3f}, null share >= {S_THRESHOLD} {null_share:.4f}")

    if meta["phase"] == "2b":
        p = pd.DataFrame(result["pairs"])
        for kind, part in p.groupby("seeds_differ"):
            v = summarise_pairs(part[f"share_ge_{S_THRESHOLD}"])
            result["tidy"].append(tidy(setting, d_sae, k, step, None, None, "seed_overlap_S", f"differ: {kind}",
                                       v["mean"], v["min"], v["max"], "none"))

    if d_sae <= functional_max:
        fnull = functional_null(saes[anchor], cov, k, NULL_SEEDS[:1], tokens, device)
        fq = float(np.nanquantile(fnull, NULL_QUANTILE))
        f_vals, f_null_vals = [], []
        for b in order[1:]:
            sc, dead_a, dead_b = functional_overlap(saes[anchor], saes[b], tokens, device)
            f_vals.append(overlap_share(sc, S_THRESHOLD))
            f_null_vals.append(overlap_share(sc, fq))
            result["pairs"].append({"setting": setting, "step": step, "match": "functional", "run_a": anchor,
                                    "run_b": b, "mean": float(np.nanmean(sc)), f"share_ge_{S_THRESHOLD}": f_vals[-1],
                                    "share_ge_null_q999": f_null_vals[-1], "null_q999": fq,
                                    "dead_a": dead_a, "dead_b": dead_b})
        F, F_null = summarise_pairs(f_vals), summarise_pairs(f_null_vals)
        f_null_share = overlap_share(fnull, S_THRESHOLD)
        result["functional"] = {"share": F, "share_above_null": F_null, "null_q999": fq,
                                f"null_share_ge_{S_THRESHOLD}": f_null_share}
        result["tidy"] += [
            tidy(setting, d_sae, k, step, None, None, f"functional_share_ge_{S_THRESHOLD}", "anchor_pairs",
                 F["mean"], F["min"], F["max"]),
            tidy(setting, d_sae, k, step, None, None, "functional_share_above_null_q999", "anchor_pairs",
                 F_null["mean"], F_null["min"], F_null["max"]),
            tidy(setting, d_sae, k, step, None, None, "functional_null_q999", "anchor", fq),
            tidy(setting, d_sae, k, step, None, None, f"functional_null_share_ge_{S_THRESHOLD}", "anchor",
                 f_null_share),
        ]
        log(f"{setting} step {step}: functional share >= {S_THRESHOLD} {F['mean']:.4f} "
            f"({F['min']:.4f}-{F['max']:.4f}); above own null q99.9 ({fq:.3f}) {F_null['mean']:.4f}; "
            f"null share >= {S_THRESHOLD} {f_null_share:.4f}")
    return result


def evaluate_nulls(ctx, d_sae: int, k: int, tokens, device: str, pca_sample: torch.Tensor) -> dict:
    """Own-decoder FVE of the matched-sparsity nulls of P1.1 (the frozen-decoder runs are scored as
    settings of their own)."""
    d = ctx.acts.d_model
    nulls = {f"random_seed{s}": random_sae(d, d_sae, k, seed=s) for s in RANDOM_SAE_SEEDS}
    nulls["signed_pca"] = signed_pca(pca_sample, d, k)
    nulls["signed_basis"] = signed_basis(d, k)
    result = {"fve": [], "tidy": []}
    for name, enc in nulls.items():
        sc = score_dictionary(enc, tokens, N_BOOT, CONFIDENCE, SEED, device)
        kind = "random_sae" if name.startswith("random") else name
        f, t = score_row(name, kind, f"nulls_d{d_sae}_k{k}", enc.d_sae, k, 0, None, None, sc)
        result["fve"].append(f)
        result["tidy"] += t
        log(f"null {name} d{d_sae} k{k}: FVE {sc.fve:.4f}")
    return result


def run_probes(settings: dict, names: list[str], args) -> None:
    for name in names:
        info = settings[name]
        out = args.out / "probes" / name
        cmd = [sys.executable, str(REPO / "scripts/stage0_metrics.py"), "--protocol", "v1", "--out", str(out),
               "--split-file", str(args.split_file), "--sae-fit-split", "train", "--eval-split", EVAL_SPLIT,
               "--run-dir", str(args.run_dir), "--label-dir", str(args.label_dir)]
        for r, rinfo in info["runs"].items():
            last = max(rinfo["steps"])
            cmd += ["--sae", f"{r}={rinfo['steps'][last]}"]
        if name == PHASE0:
            cmd[cmd.index("train")] = "all"
        log(f"probes {name}: {' '.join(cmd[2:])}")
        subprocess.run(cmd, check=True)


def run_per_latent(settings: dict, tokens, device: str, out: Path, force: bool) -> None:
    """Per-latent check (protocol section 5) at each setting's last common checkpoint: for every seed pair, each
    latent's one-to-one partner, decoder cosine and image firing rates; a summary per setting."""
    out.mkdir(parents=True, exist_ok=True)
    for name, info in settings.items():
        if info["meta"]["kind"] != "trained" or len(info["runs"]) < 2 or name == PHASE0:
            continue
        target = out / f"{name}_summary.json"
        if target.exists() and not force:
            continue
        runs = info["runs"]
        step = max(set.intersection(*(set(r["steps"]) for r in runs.values())))
        order = sorted(runs, key=lambda r: (runs[r]["init_seed"], runs[r]["data_seed"]))
        saes = {r: TopKSAE.from_checkpoint(runs[r]["steps"][step]) for r in order}
        rates = {r: firing_rates(saes[r], tokens, device)[1] for r in order}
        tables = []
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                cos, partner = one_to_one(cosine_matrix(saes[a].W_dec, saes[b].W_dec))
                tables.append(per_latent_pair_table(cos, partner, rates[a], rates[b]).assign(run_a=a, run_b=b))
        pd.concat(tables).to_csv(out / f"{name}_latents.csv.gz", index=False)
        summary = {"setting": name, "d_sae": info["meta"]["d_sae"], "k": info["meta"]["k"], "step": step,
                   **summarise_per_latent(tables, S_THRESHOLD)}
        target.write_text(json.dumps(summary, indent=1, default=float))
        log(f"per-latent {name} step {step}: S {summary['S_all']['mean']:.4f}, among live {summary['S_live']['mean']:.4f}, "
            f"rate-cosine rho {summary['spearman_rate_cosine']['mean']:.3f}")


def run_subspace(settings: dict, tokens, cov, device: str, out: Path, force: bool) -> None:
    """Subspace reproducibility (protocol section 5) at each setting's last common checkpoint: every seed is
    an anchor; its unstable and stable latents (among those that fire on val) are compared with each other
    seed's, and the unstable sets also with covariance-shaped random directions of the same count."""
    out.mkdir(parents=True, exist_ok=True)
    for name, info in settings.items():
        if info["meta"]["kind"] != "trained" or len(info["runs"]) < 3 or name == PHASE0:
            continue
        target = out / f"{name}.json"
        if target.exists() and not force:
            continue
        runs = info["runs"]
        step = max(set.intersection(*(set(r["steps"]) for r in runs.values())))
        order = sorted(runs, key=lambda r: (runs[r]["init_seed"], runs[r]["data_seed"]))
        saes = {r: TopKSAE.from_checkpoint(runs[r]["steps"][step]) for r in order}
        live = {r: firing_rates(saes[r], tokens, device)[1] > 0 for r in order}
        decs = {r: saes[r].W_dec.float() for r in order}
        sets, per_seed = {}, []
        for r in order:
            p = reappearance_probability(decs[r], [decs[o] for o in order if o != r], S_THRESHOLD)
            unstable, stable = stability_sets(p, SUBSPACE_EPSILON, live[r])
            sets[r] = (unstable, stable)
            d = decs[r].shape[1]
            per_seed.append({"run": r, "n_live": int(live[r].sum()), "n_unstable": len(unstable), "n_stable": len(stable),
                             "er_ratio_unstable": effective_rank(decs[r][unstable]) / d if len(unstable) else None,
                             "er_ratio_stable": effective_rank(decs[r][stable]) / d if len(stable) else None})
        pairs = []
        for a in order:
            for s in order:
                if a == s:
                    continue
                row = {"anchor": a, "other": s}
                for kind, i in (("unstable", 0), ("stable", 1)):
                    if len(sets[a][i]) and len(sets[s][i]):
                        ev = subspace_explained_variance(decs[a][sets[a][i]], decs[s][sets[s][i]], SUBSPACE_RANKS)
                        row.update({f"{kind}_r{r}": v for r, v in ev.items()})
                if len(sets[a][0]) and len(sets[s][0]):
                    null = covariance_directions(cov, len(sets[s][0]), seed=SEED)
                    ev = subspace_explained_variance(decs[a][sets[a][0]], null.float(), SUBSPACE_RANKS)
                    row.update({f"null_r{r}": v for r, v in ev.items()})
                    # Basis nulls: the same target, explained by a random data-shaped subspace of the anchor
                    # set's size and by the data's principal subspace.
                    random_basis = covariance_directions(cov, len(sets[a][0]), seed=SEED + 1).float()
                    ev = subspace_explained_variance(random_basis, decs[s][sets[s][0]], SUBSPACE_RANKS)
                    row.update({f"basis_random_r{r}": v for r, v in ev.items()})
                    ev = principal_subspace_explained_variance(cov, decs[s][sets[s][0]], SUBSPACE_RANKS)
                    row.update({f"basis_pca_r{r}": v for r, v in ev.items()})
                pairs.append(row)
        pair_table = pd.DataFrame(pairs)
        pair_table.to_csv(out / f"{name}_pairs.csv", index=False)
        summary = {"setting": name, "d_sae": info["meta"]["d_sae"], "k": info["meta"]["k"], "step": step,
                   "epsilon": SUBSPACE_EPSILON, "per_seed": per_seed,
                   "mean_explained_variance": {c: float(pair_table[c].mean()) for c in pair_table.columns
                                               if c not in ("anchor", "other")}}
        target.write_text(json.dumps(summary, indent=1, default=float))
        ev = summary["mean_explained_variance"]
        log(f"subspace {name}: unstable {np.mean([p['n_unstable'] for p in per_seed]):.0f} latents, "
            + ", ".join(f"r{r}: {ev.get(f'unstable_r{r}', float('nan')):.3f} vs random basis "
                        f"{ev.get(f'basis_random_r{r}', float('nan')):.3f}, PCA {ev.get(f'basis_pca_r{r}', float('nan')):.3f}"
                        for r in SUBSPACE_RANKS))


def nearest_null_k(mean_l0: float, d_sae: int, null_ks: list[tuple[int, int]]) -> int | None:
    """The k of the null dictionaries (same size) closest to a measured mean L0; ties go to the smaller k."""
    ks = sorted(k for d, k in null_ks if d == d_sae)
    return min(ks, key=lambda k: (abs(k - mean_l0), k)) if ks else None


def p11_architectures_table(fve: pd.DataFrame) -> pd.DataFrame:
    """P1.1 for the phase 3 architectures (protocol section 5, "Phase 3"): each setting's seeds at their final
    checkpoint against every null at the k nearest to the setting's mean measured L0, next to the measured L0
    and FVE, so reconstruction can be read against measured sparsity at fixed width. BatchTopK at 80k
    (phase 2a) is listed the same way as the reference."""
    null_ks = sorted({(int(d), int(k)) for d, k in zip(fve[fve.setting.str.startswith("nulls_")].d_sae,
                                                       fve[fve.setting.str.startswith("nulls_")].k)})
    rows = []
    trained = fve[(fve.dictionary_kind == "trained") & fve.setting.str.match(r"^d\d+_k\d+_s80000(_matryoshka|_relul1)?$")]
    for setting, part in trained.groupby("setting"):
        last = part[part.step == part.step.max()]
        d_sae = int(last.d_sae.iloc[0])
        l0 = float(last.mean_l0.mean()) if "mean_l0" in last else float("nan")
        if np.isnan(l0):
            # Caches written before mean L0 was recorded hold BatchTopK runs, whose per-token top-k fixes L0 at
            # k (at most; fewer only where pre-activations are not positive).
            l0 = float(last.k.iloc[0])
        k_null = nearest_null_k(l0, d_sae, null_ks)
        if k_null is None:
            continue
        nulls = fve[(fve.setting == f"nulls_d{d_sae}_k{k_null}")
                    | ((fve.setting == f"d{d_sae}_k{k_null}_s20000_frozeniso") & (fve.step == 20_000))]
        sep = complete_separation(last.value.to_numpy(), nulls.value.to_numpy())
        best = nulls.loc[nulls.value.idxmax()]
        architecture = ("matryoshka" if setting.endswith("_matryoshka") else
                        "relu_l1" if setting.endswith("_relul1") else "batchtopk")
        rows.append({"setting": setting, "architecture": architecture, "d_sae": d_sae, "row_k": int(last.k.iloc[0]),
                     "step": int(last.step.iloc[0]), "n_trained": len(last), "mean_l0": l0,
                     "mean_l0_min": float(last.mean_l0.min()), "mean_l0_max": float(last.mean_l0.max()),
                     "fve_mean": float(last.value.mean()), "worst_trained": float(last.value.min()),
                     "null_k": k_null, "n_null": len(nulls), "best_null": best.dictionary,
                     "best_null_value": best.value, "gap": sep.gap, "separated": sep.separated,
                     "p_exact": sep.p_value})
    return pd.DataFrame(rows)


def further_dictionary(name: str, kind: str, enc, ctx, keep, probes: RecoveryProbes, targets, device: str) -> dict:
    """Performance recovery, absorption and targeted ablation of one dictionary (protocol section 4)."""
    feats = image_features(ctx.acts, enc, keep, device)
    recovery = probes.score(decode_pooled(enc, feats.mean_codes), N_BOOT, CONFIDENCE, SEED)
    codes = {d: c.astype(np.float32) for d, c in feats.max_codes.items()}
    out = {"dictionary": name, "kind": kind, "recovery": recovery,
           "mean_recovered_share": recovery_mean(recovery)}
    if "chexpert" in codes:
        labels = {lab: targets[lab]["chexpert"] for lab in targets if lab in (
            "Lung Opacity", "Consolidation", "Pneumonia", "Edema", "Atelectasis")}
        if len(labels) == 5:
            m = ctx.masks["chexpert"]
            out["absorption"] = absorption(codes["chexpert"], labels, m["train"], m[EVAL_SPLIT])
    comparisons = tuple(t for t in targets if t.startswith("opacity_"))
    out["ablation"] = targeted_ablation(codes, targets, ctx.masks, CONFOUNDERS, comparisons, ABLATION_NS,
                                        "train", "val", EVAL_SPLIT)
    return out


def run_further(settings: dict, ctx, device: str, out: Path, force: bool) -> None:
    """Further measures of protocol section 4 for every trained setting (and its matched nulls) at the last
    common checkpoint. Settings trained on projected tokens (phase 2f) and the ReLU+L1 calibration are skipped."""
    out.mkdir(parents=True, exist_ok=True)
    chosen = {n: v for n, v in settings.items() if v["meta"]["kind"] == "trained" and n != PHASE0
              and v["meta"].get("projection", "none") == "none" and v["meta"]["phase"] != "3c"}
    if not chosen:
        return
    keep = {d: ctx.masks[d]["train"] | ctx.masks[d][EVAL_SPLIT] for d in ctx.acts.blocks}
    targets = secondary_targets(ctx.manifests, ctx.opacity)
    log(f"further: {len(targets)} targets; fitting recovery probes on pooled originals")
    recovery_targets = {t: v for t, v in targets.items() if not t.startswith(RECOVERY_EXCLUDED_PREFIXES)}
    probes = RecoveryProbes(pooled_activations(ctx.acts, keep), recovery_targets, ctx.masks, ctx.groups,
                            "train", "val", EVAL_SPLIT)
    pd.DataFrame([{"target": t, "auroc_orig": a} for t, a in probes.orig_auc.items()]).to_csv(
        out / "recovery_probes.csv", index=False)
    pca_sample = None
    done_nulls = set()
    for name, info in chosen.items():
        target_file = out / f"{name}.json"
        if target_file.exists() and not force:
            continue
        runs = info["runs"]
        step = max(set.intersection(*(set(r["steps"]) for r in runs.values())))
        order = sorted(runs, key=lambda r: (runs[r]["init_seed"], runs[r]["data_seed"]))
        saes = {r: TopKSAE.from_checkpoint(runs[r]["steps"][step]) for r in order}
        result = {"setting": name, "d_sae": info["meta"]["d_sae"], "k": info["meta"]["k"], "step": step,
                  "pairs": [], "dictionaries": []}
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                result["pairs"].append({"run_a": a, "run_b": b, **joint_seed_overlap(saes[a], saes[b], S_THRESHOLD)})
        for r in order:
            result["dictionaries"].append(further_dictionary(r, "trained", saes[r], ctx, keep, probes, targets, device))
            d = result["dictionaries"][-1]
            log(f"further {name} {r}: recovered share {d['mean_recovered_share']:.3f}")
        target_file.write_text(json.dumps(result, indent=1, default=float))
        # The matched nulls of this (d_sae, k), once per pair: the same measures under the same rules.
        key = (info["meta"]["d_sae"], info["meta"]["k"])
        null_file = out / f"nulls_d{key[0]}_k{key[1]}.json"
        if key in done_nulls or (null_file.exists() and not force):
            continue
        if pca_sample is None:
            pca_sample = train_token_sample(ctx, PCA_TOKENS_PER_DATASET, SEED)
        d_model = ctx.acts.d_model
        nulls = {f"random_seed{s}": random_sae(d_model, key[0], key[1], seed=s) for s in RANDOM_SAE_SEEDS}
        nulls["signed_pca"] = signed_pca(pca_sample, d_model, key[1])
        nulls["signed_basis"] = signed_basis(d_model, key[1])
        null_result = {"d_sae": key[0], "k": key[1], "dictionaries": [
            further_dictionary(n, "random_sae" if n.startswith("random") else n, enc, ctx, keep, probes, targets, device)
            for n, enc in nulls.items()]}
        null_file.write_text(json.dumps(null_result, indent=1, default=float))
        done_nulls.add(key)
        log(f"further nulls d{key[0]} k{key[1]}: recovered share "
            + ", ".join(f"{d['dictionary']} {d['mean_recovered_share']:.3f}" for d in null_result["dictionaries"]))


def p11_table(fve: pd.DataFrame) -> pd.DataFrame:
    """P1.1 per (d_sae, k) at the final 20k checkpoint of phase-1 settings: trained seeds vs every null."""
    rows = []
    trained = fve[(fve.dictionary_kind == "trained") & fve.setting.str.match(r"^d\d+_k\d+_s20000$")
                  & (fve.step == 20_000)]
    for (d_sae, k), part in trained.groupby(["d_sae", "k"]):
        nulls = fve[(fve.dictionary_kind != "trained") & (fve.k == k)
                    & ((fve.setting == f"nulls_d{d_sae}_k{k}")
                       | (fve.setting == f"d{d_sae}_k{k}_s20000_frozeniso") & (fve.step == 20_000))]
        if nulls.empty:
            continue
        sep = complete_separation(part.value.to_numpy(), nulls.value.to_numpy())
        best = nulls.loc[nulls.value.idxmax()]
        rows.append({"d_sae": d_sae, "k": k, "n_trained": len(part), "n_null": len(nulls),
                     "worst_trained": part.value.min(), "best_null": best.dictionary,
                     "best_null_value": best.value, "gap": sep.gap, "separated": sep.separated,
                     "p_exact": sep.p_value})
    return pd.DataFrame(rows)


def _coordinate_variants(cov: torch.Tensor) -> dict[str, dict]:
    """The three coordinate systems of the subspace robustness checks (protocol section 5): per variant a map
    applied to decoder rows, the covariance of its random basis null, and the covariance whose top
    eigenvectors give its principal-component basis null."""
    lam, vecs = torch.linalg.eigh(cov.double())
    pc1 = vecs[:, -1:].T                                  # top eigenvector as a row
    off = torch.eye(cov.shape[0], dtype=torch.float64) - pc1.T @ pc1
    deflated = off @ cov.double() @ off
    white, projector = whitening_map(cov, WHITENING_VARIANCE_SHARE)
    unit = lambda x: x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return {
        "raw": {"map": lambda rows: unit(rows.double()), "null_cov": cov.double(), "pca_cov": cov.double()},
        "nopc1": {"map": lambda rows: project_out(rows, pc1), "null_cov": deflated, "pca_cov": deflated},
        # in whitened coordinates the eigenvectors of cov keep their directions, so they still give the PCA null
        "white": {"map": lambda rows: unit(rows.double() @ white), "null_cov": projector, "pca_cov": cov.double()},
    }


def run_subspace_robust(settings: dict, tokens, cov, device: str, out: Path, force: bool) -> None:
    """Robustness checks of subspace reproducibility (protocol section 5, stated 2026-11-24): the latent-subset
    null matched on firing rate, the coordinate variants without the first principal component and whitened,
    and the symmetric principal-angle overlap, at each phase 1 / 1c setting with five seeds."""
    out.mkdir(parents=True, exist_ok=True)
    variants = _coordinate_variants(cov)
    for name, info in settings.items():
        if (info["meta"]["kind"] != "trained" or info["meta"]["phase"] not in ROBUST_PHASES
                or len(info["runs"]) != ROBUST_SEEDS):
            continue
        target = out / f"{name}.json"
        if target.exists() and not force:
            continue
        runs = info["runs"]
        step = max(set.intersection(*(set(r["steps"]) for r in runs.values())))
        order = sorted(runs, key=lambda r: (runs[r]["init_seed"], runs[r]["data_seed"]))
        saes = {r: TopKSAE.from_checkpoint(runs[r]["steps"][step]) for r in order}
        rates = {r: firing_rates(saes[r], tokens, device)[1] for r in order}
        decs = {r: saes[r].W_dec.float() for r in order}
        unstable = {}
        for r in order:
            p = reappearance_probability(decs[r], [decs[o] for o in order if o != r], S_THRESHOLD)
            unstable[r] = stability_sets(p, SUBSPACE_EPSILON, rates[r] > 0)[0]
        rng = np.random.default_rng(SEED)
        rows = []
        for a in order:
            live_a = np.flatnonzero(rates[a] > 0)
            if not len(unstable[a]):
                continue
            # the same draws serve every target seed and every coordinate variant of this anchor
            draws = [matched_subset(live_a, rates[a][live_a], rates[a][unstable[a]], rng)
                     for _ in range(SUBSET_NULL_DRAWS)]
            for vname, v in variants.items():
                basis_obs = right_singular_basis(v["map"](decs[a][unstable[a]]))
                basis_draws = [right_singular_basis(v["map"](decs[a][d])) for d in draws]
                basis_rand = right_singular_basis(covariance_directions(v["null_cov"], len(unstable[a]), seed=SEED + 1))
                pca = torch.linalg.eigh(v["pca_cov"])[1].flip(1).T
                for s in order:
                    if s == a or not len(unstable[s]):
                        continue
                    tgt = v["map"](decs[s][unstable[s]])
                    basis_tgt = right_singular_basis(tgt)
                    row = {"anchor": a, "other": s, "variant": vname}
                    ev = explained_variance_in(basis_obs, tgt, SUBSPACE_RANKS)
                    ov = overlap_of(basis_obs, basis_tgt, SUBSPACE_RANKS)
                    ev_draws = [explained_variance_in(b, tgt, SUBSPACE_RANKS) for b in basis_draws]
                    ov_draws = [overlap_of(b, basis_tgt, SUBSPACE_RANKS) for b in basis_draws]
                    ev_rand = explained_variance_in(basis_rand, tgt, SUBSPACE_RANKS)
                    ev_pca = explained_variance_in(pca, tgt, SUBSPACE_RANKS)
                    ov_rand = overlap_of(basis_rand, basis_tgt, SUBSPACE_RANKS)
                    ov_pca = overlap_of(pca, basis_tgt, SUBSPACE_RANKS)
                    for r in ev:
                        dr = np.array([d[r] for d in ev_draws])
                        orr = np.array([d[r] for d in ov_draws if r in d])
                        row.update({f"ev_unstable_r{r}": ev[r], f"ev_subset_mean_r{r}": dr.mean(),
                                    f"ev_subset_q_r{r}": np.quantile(dr, SUBSET_NULL_QUANTILE),
                                    f"ev_share_draws_exceeded_r{r}": float((ev[r] > dr).mean()),
                                    f"ev_basis_random_r{r}": ev_rand.get(r), f"ev_basis_pca_r{r}": ev_pca.get(r)})
                        if r in ov:
                            row.update({f"ov_unstable_r{r}": ov[r], f"ov_basis_random_r{r}": ov_rand.get(r),
                                        f"ov_basis_pca_r{r}": ov_pca.get(r),
                                        f"ov_subset_q_r{r}": np.quantile(orr, SUBSET_NULL_QUANTILE) if len(orr) else None})
                    rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(out / f"{name}_pairs.csv", index=False)
        means = {vname: {c: float(g[c].mean()) for c in g.columns if c not in ("anchor", "other", "variant")}
                 for vname, g in table.groupby("variant")}
        summary = {"setting": name, "d_sae": info["meta"]["d_sae"], "k": info["meta"]["k"], "step": step,
                   "n_draws": SUBSET_NULL_DRAWS, "quantile": SUBSET_NULL_QUANTILE,
                   "whitening_variance_share": WHITENING_VARIANCE_SHARE,
                   "n_unstable": {r: len(unstable[r]) for r in order}, "mean_over_pairs": means}
        target.write_text(json.dumps(summary, indent=1, default=float))
        m = means.get("raw", {})
        log(f"subspace robust {name}: " + ", ".join(
            f"r{r}: unstable {m.get(f'ev_unstable_r{r}', float('nan')):.3f} vs subset q "
            f"{m.get(f'ev_subset_q_r{r}', float('nan')):.3f}" for r in SUBSPACE_RANKS))


def main() -> None:
    global LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/sweep_v1.csv")
    ap.add_argument("--runs", type=Path, nargs="+", default=[ROOT / "results/sweep_v1/runs"],
                    help="one or more run roots; a run is taken from the first root that holds it finished")
    ap.add_argument("--out", type=Path, default=ROOT / "results/sweep_v1/eval")
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--split-file", type=Path, default=ROOT / "data/splits/v1.csv")
    ap.add_argument("--device", help="default: cuda, else mps, else cpu")
    ap.add_argument("--phase0", action="store_true", help="include the Stage 0.6 all-data seeds 0-2")
    ap.add_argument("--phase0-dir", type=Path, default=REPO / "results/stage0/sae_seeds",
                    help="directory holding seed0, seed1, seed2 of Stage 0.6")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only merge the caches under --out (e.g. fetched from several Kaggle runs)")
    ap.add_argument("--settings", nargs="*", help="only these settings (default: every finished one)")
    ap.add_argument("--functional-max-dsae", type=int, default=4096,
                    help="skip functional matching above this size (its correlation matrix is d_sae^2 float64)")
    ap.add_argument("--probes", nargs="*", default=None, metavar="SETTING",
                    help="also run stage0_metrics.py v1 (P1.2, selectivity) on these settings' final checkpoints")
    ap.add_argument("--per-latent", action="store_true",
                    help="also run the per-latent check (protocol section 5) on the finished settings")
    ap.add_argument("--subspace", action="store_true",
                    help="also run the subspace-reproducibility check (protocol section 5) on the finished settings")
    ap.add_argument("--subspace-robust", action="store_true",
                    help="also run the subspace robustness checks of protocol section 5 (2026-11-24)")
    ap.add_argument("--further", action="store_true",
                    help="also run the further measures of protocol section 4 (2026-11-25) on the finished settings")
    ap.add_argument("--skip-sweep-metrics", action="store_true",
                    help="skip the per-setting sweep metrics and their aggregation (with --per-latent or --probes)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(args.out / "run.log", "a")
    table = pd.read_csv(args.config)
    settings = collect_settings(table, args.runs, args.phase0_dir if args.phase0 else None)
    if args.settings:
        settings = {s: v for s, v in settings.items() if s in args.settings}
    log(f"device {device}; finished settings: {list(settings)}")
    if not args.aggregate_only:
        write_config(args.out, {"eval_split": EVAL_SPLIT, "s_threshold": S_THRESHOLD, "null_quantile": NULL_QUANTILE,
                            "n_bootstrap": N_BOOT, "confidence": CONFIDENCE, "null_seeds": NULL_SEEDS,
                            "random_sae_seeds": RANDOM_SAE_SEEDS, "seed": SEED, "config": str(args.config),
                            "settings": {s: {"meta": v["meta"], "runs": list(v["runs"])} for s, v in settings.items()}},
                 REPO, args.split_file)

    ctx = None if args.aggregate_only else load_context(args.run_dir, args.label_dir, args.split_file)
    cache = args.out / "cache"
    tokens = None
    cov = None

    def need_tokens(with_covariance: bool = True):
        nonlocal tokens, cov
        if tokens is None:
            log(f"loading {EVAL_SPLIT} tokens")
            tokens = load_eval_tokens(ctx.acts, ctx.masks, ctx.groups, EVAL_SPLIT)
            log(f"{tokens.n_images} images, {tokens.x.shape[0]} tokens")
        if with_covariance and cov is None:
            cov = train_covariance(ctx, cache / "cov_train.pt")
        return tokens, cov

    if args.aggregate_only:
        settings = {}
    to_evaluate = {} if args.skip_sweep_metrics else settings
    projected = None    # (tokens, covariance) without the first principal component, for phase 2f settings

    def inputs_for(info):
        nonlocal projected
        tok, cv = need_tokens()
        if info["meta"].get("projection", "none") != "nopc1":
            return tok, cv
        if projected is None:
            u = top_component(cv)
            x = remove_component(tok.x.clone(), u)
            off = torch.eye(cv.shape[0], dtype=torch.float64) - torch.outer(torch.from_numpy(u), torch.from_numpy(u))
            projected = (EvalTokens(x=x, groups=tok.groups, n_patches=tok.n_patches), off @ cv.double() @ off)
        return projected

    for name, info in to_evaluate.items():
        steps = sorted(set.intersection(*(set(r["steps"]) for r in info["runs"].values())))
        for step in steps:
            f = cache / name / f"step_{step}.json"
            if f.exists() and not args.force:
                continue
            tok, cv = inputs_for(info)
            res = evaluate_setting(name, info, step, tok, cv, device, args.functional_max_dsae)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(res, indent=1, default=float))

    pca_sample = None
    for d_sae, k in sorted({(v["meta"]["d_sae"], v["meta"]["k"]) for v in to_evaluate.values()
                            if v["meta"]["kind"] == "trained" and v["meta"].get("projection", "none") == "none"
                            and v["meta"]["phase"] != "3c"}):
        f = cache / "nulls" / f"d{d_sae}_k{k}.json"
        if f.exists() and not args.force:
            continue
        tok, _ = need_tokens()
        if pca_sample is None:
            pca_sample = train_token_sample(ctx, PCA_TOKENS_PER_DATASET, SEED)
        res = evaluate_nulls(ctx, d_sae, k, tok, device, pca_sample)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, indent=1, default=float))

    if args.skip_sweep_metrics:
        if args.per_latent:
            run_per_latent(settings, need_tokens(with_covariance=False)[0], device, args.out / "per_latent", args.force)
        if args.subspace:
            tok, cv = need_tokens()
            run_subspace(settings, tok, cv, device, args.out / "subspace", args.force)
        if args.subspace_robust:
            tok, cv = need_tokens()
            run_subspace_robust(settings, tok, cv, device, args.out / "subspace_robust", args.force)
        if args.further:
            run_further(settings, ctx, device, args.out / "further", args.force)
        if args.probes is not None:
            run_probes(settings, args.probes or list(settings), args)
        log(f"wrote {args.out}")
        return

    fve, tidy_rows, pairs = [], [], []
    for f in sorted(cache.glob("*/*.json")):
        res = json.loads(f.read_text())
        fve += res["fve"]
        tidy_rows += res["tidy"]
        pairs += res.get("pairs", [])
    fve = pd.DataFrame(fve)
    fve.to_csv(args.out / "fve.csv", index=False)
    pd.DataFrame(tidy_rows, columns=TIDY_COLUMNS).to_csv(args.out / "results.csv", index=False)
    pd.DataFrame(pairs).to_csv(args.out / "pairs.csv", index=False)
    if not fve.empty:
        p11 = p11_table(fve)
        p11.to_csv(args.out / "p11.csv", index=False)
        if not p11.empty:
            log("P1.1 (val, exploratory):\n" + p11.to_string(index=False))
        if "mean_l0" in fve:
            arch = p11_architectures_table(fve)
            arch.to_csv(args.out / "p11_architectures.csv", index=False)
            if not arch.empty:
                log("P1.1 by architecture at measured L0 (val, exploratory):\n" + arch.to_string(index=False))
    if args.per_latent and settings:
        run_per_latent(settings, need_tokens(with_covariance=False)[0], device, args.out / "per_latent", args.force)
    if args.subspace and settings:
        tok, cv = need_tokens()
        run_subspace(settings, tok, cv, device, args.out / "subspace", args.force)
    if args.further and settings:
        run_further(settings, ctx, device, args.out / "further", args.force)
    if args.probes is not None:
        run_probes(settings, args.probes or list(settings), args)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
