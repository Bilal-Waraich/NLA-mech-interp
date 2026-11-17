"""Sweep report (protocol section 5): headline setting, stability curve, outcome row.

1. Headline setting. Chosen from the val FVE table alone (``fve.csv``; the stability numbers are not read
   for this step): at 4,096 latents, the k with the largest gap between the worst trained seed and the
   best null. It is written once to ``<sweep>/headline.json`` and never changed: if the file exists, it is
   used as it is, and a recomputation that disagrees is reported but does not replace it.
2. Stability curve: S (mean, min-max over seed pairs) for every setting and checkpoint
   (``stability_curve.csv``, plus ``stability_curve.png`` if matplotlib is installed).
3. Outcome: which row of the protocol's outcome table applies, with the inputs that decided it.
   Training length is judged by S of the 80k runs (phase 2a) against S of the 20k runs at the headline
   setting (phase 1a): the 80k runs' own 20k checkpoint sits halfway through a longer learning-rate
   schedule and is reported alongside, not used.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/sweep_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Package import without relying on the editable-install .pth (see stage0_metrics.py).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from medinterp.sweep import BASE_D_SAE, BASE_STEPS, PHASE1_KS, classify_outcome, headline_k, setting_name

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
# Protocol section 5, outcome table.
OUTCOME_THRESHOLDS = {"high": 0.30, "low": 0.15, "under_training": 0.10}
THESIS_K = 48


def fve_for_headline(fve: pd.DataFrame) -> pd.DataFrame | None:
    """Rows the headline rule needs, or None while any k still lacks its trained runs or nulls."""
    parts = []
    for k in PHASE1_KS:
        s = setting_name(BASE_D_SAE, k, BASE_STEPS)
        trained = fve[(fve.setting == s) & (fve.step == BASE_STEPS)]
        nulls = fve[(fve.setting == f"nulls_d{BASE_D_SAE}_k{k}")
                    | ((fve.setting == f"{s}_frozeniso") & (fve.step == BASE_STEPS))]
        if len(trained) < 5 or nulls.empty:
            return None
        parts += [trained, nulls]
    return pd.concat(parts)[["k", "dictionary_kind", "dictionary", "value"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, default=ROOT / "results/sweep_v1")
    args = ap.parse_args()
    ev = args.sweep / "eval"
    fve = pd.read_csv(ev / "fve.csv")
    report: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # 1. Headline setting (FVE only).
    head_file = args.sweep / "headline.json"
    rows = fve_for_headline(fve)
    if rows is not None:
        k, gaps = headline_k(rows)
        if head_file.exists():
            fixed = json.loads(head_file.read_text())
            if fixed["headline_k"] != k:
                print(f"note: recomputed headline k = {k} differs from the fixed k = {fixed['headline_k']}; "
                      f"the fixed value stays")
        else:
            head_file.write_text(json.dumps({"headline_k": k, "d_sae": BASE_D_SAE, "steps": BASE_STEPS,
                                             "rule": "largest val FVE gap, worst trained seed minus best null",
                                             "gaps": gaps.to_dict("records"),
                                             "fixed_at": report["generated"]}, indent=2))
            print(f"headline setting fixed: k = {k}")
        report["fve_gaps"] = gaps.to_dict("records")
    report["headline"] = json.loads(head_file.read_text()) if head_file.exists() else None

    # 2. Stability curve.
    res = pd.read_csv(ev / "results.csv")
    S = res[(res.metric == "seed_overlap_S") & (res.subset == "all_pairs")]
    curve = S[["dictionary", "d_sae", "k", "steps", "value", "ci_low", "ci_high"]].rename(
        columns={"dictionary": "setting", "value": "S_mean", "ci_low": "S_min", "ci_high": "S_max"})
    curve = curve.sort_values(["d_sae", "k", "setting", "steps"]).reset_index(drop=True)
    curve.to_csv(args.sweep / "stability_curve.csv", index=False)
    report["curve"] = curve.to_dict("records")
    _plot(curve, args.sweep / "stability_curve.png")

    # 3. Outcome row.
    final = curve[curve.steps == BASE_STEPS].set_index("setting").S_mean.to_dict()
    phase1 = {s: v for s, v in final.items() if s.count("_") == 2 and s.startswith("d")}
    thesis = setting_name(BASE_D_SAE, THESIS_K, BASE_STEPS)
    small = [s for s in phase1 if int(s.split("_k")[1].split("_")[0]) <= 16 or s.startswith("d1024_")]
    delta = None
    if report["headline"]:
        h = report["headline"]["headline_k"]
        long_s = curve[(curve.setting == setting_name(BASE_D_SAE, h, 80_000))]
        base_s = final.get(setting_name(BASE_D_SAE, h, BASE_STEPS))
        at80 = long_s[long_s.steps == 80_000].S_mean
        if len(at80) and base_s is not None:
            delta = float(at80.iloc[0] - base_s)
            report["training_length"] = {"S_80k": float(at80.iloc[0]), "S_20k_run": base_s,
                                         "S_80k_run_at_20k": long_s[long_s.steps == 20_000].S_mean.tolist()}
    if thesis in phase1:
        report["outcome"] = {"row": classify_outcome(phase1, thesis, small, delta, OUTCOME_THRESHOLDS),
                             "S_thesis_setting": phase1[thesis], "small_settings": {s: phase1[s] for s in small},
                             "under_training_delta": delta, "thresholds": OUTCOME_THRESHOLDS,
                             "complete": delta is not None and len(phase1) == 11}
    (args.sweep / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report.get(k) for k in ("headline", "outcome")}, indent=2, default=float))


def _plot(curve: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    final = curve[(curve.steps == BASE_STEPS) & curve.setting.str.match(r"^d\d+_k\d+_s20000$")]
    if final.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for d_sae, part in final.groupby("d_sae"):
        part = part.sort_values("k")
        ax.errorbar(part.k, part.S_mean, yerr=[part.S_mean - part.S_min, part.S_max - part.S_mean],
                    marker="o", capsize=3, label=f"{d_sae} latents")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (active latents per token)")
    ax.set_ylabel("S: share of latents with a partner at cosine >= 0.7")
    ax.set_title("Seed overlap of decoder directions, 20k steps, SAEs fitted on train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
