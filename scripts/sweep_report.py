"""Sweep report (protocol section 5): P1.1 per setting, stability curve, training length, outcome row.

1. P1.1 per evaluated setting with its role (``p11_by_setting.csv``), on val and therefore exploratory.
2. Stability curve: S (mean, min-max over seed pairs) for every setting and checkpoint
   (``stability_curve.csv``, plus ``stability_curve.png`` if matplotlib is installed).
3. Training length per phase 2a setting (``training_length.csv``): S of the 80k runs against S of the 20k
   runs of phase 1. The 80k runs' own 20k checkpoint sits halfway through a longer learning-rate schedule
   and is listed alongside, not used.
4. Outcome: which row of the protocol's outcome table applies, with the inputs that decided it. The training
   length result is reported next to the row whichever row applies.

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

from medinterp.sweep import (BASE_STEPS, PHASE2_ALL_TESTS, PHASE2_TRAINING_LENGTH_ONLY, THESIS_SETTING,
                             classify_outcome, p11_supported, setting_name, setting_role,
                             training_length_table)

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MEDINTERP_DATA", REPO))
# Protocol sections 4 and 5.
P11_MARGIN, P11_MAX_K = 0.02, 16
OUTCOME_THRESHOLDS = {"high": 0.30, "low": 0.15, "under_training": 0.10}
OUTCOME_SMALL_MAX_K, OUTCOME_SMALL_D_SAE = 16, 1024
# Settings at 20k steps that the outcome row reads: 11 of phases 1a and 1b, and 3 of phase 1c.
N_BASE_SETTINGS = 14


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, default=ROOT / "results/sweep_v1")
    args = ap.parse_args()
    ev = args.sweep / "eval"
    report: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # 1. P1.1 per setting.
    p11_file = ev / "p11.csv"
    if p11_file.exists():
        p11 = pd.read_csv(p11_file)
        p11.insert(2, "role", [setting_role(d, k) for d, k in zip(p11.d_sae, p11.k)])
        p11.to_csv(args.sweep / "p11_by_setting.csv", index=False)
        report["p11"] = {"split": "val", "supported": p11_supported(p11, P11_MARGIN, P11_MAX_K),
                         "settings": p11.to_dict("records")}

    # 2. Stability curve.
    res = pd.read_csv(ev / "results.csv")
    S = res[(res.metric == "seed_overlap_S") & (res.subset == "all_pairs")]
    curve = S[["dictionary", "d_sae", "k", "steps", "value", "ci_low", "ci_high"]].rename(
        columns={"dictionary": "setting", "value": "S_mean", "ci_low": "S_min", "ci_high": "S_max"})
    curve = curve.sort_values(["d_sae", "k", "setting", "steps"]).reset_index(drop=True)
    curve.insert(3, "role", [setting_role(d, k) for d, k in zip(curve.d_sae, curve.k)])
    curve.to_csv(args.sweep / "stability_curve.csv", index=False)
    report["curve"] = curve.to_dict("records")
    _plot(curve, args.sweep / "stability_curve.png")

    # 3. Training length per phase 2a setting.
    training = training_length_table(curve)
    training.to_csv(args.sweep / "training_length.csv", index=False)
    report["training_length"] = training.to_dict("records")
    training_complete = len(training) == len(PHASE2_ALL_TESTS + PHASE2_TRAINING_LENGTH_ONLY) + 1   # + k = 64 at 1,024

    # 4. Outcome row. Only the 20k settings of phase 1 (1a, 1b, 1c) enter it: the 80k runs also have a 20k
    # checkpoint, and the phase 2 variants carry a suffix.
    at_base = curve[(curve.steps == BASE_STEPS) & curve.setting.str.fullmatch(rf"d\d+_k\d+_s{BASE_STEPS}")]
    phase1 = at_base.set_index("setting").S_mean.to_dict()
    thesis = setting_name(*THESIS_SETTING, BASE_STEPS)
    small = at_base[(at_base.k <= OUTCOME_SMALL_MAX_K) | (at_base.d_sae == OUTCOME_SMALL_D_SAE)].setting.tolist()
    deltas = training.set_index("setting").delta.to_dict() if training_complete else None
    if thesis in phase1:
        report["outcome"] = {"row": classify_outcome(phase1, thesis, small, deltas, OUTCOME_THRESHOLDS),
                             "S_thesis_setting": phase1[thesis], "small_settings": {s: phase1[s] for s in small},
                             "under_training_deltas": deltas, "thresholds": OUTCOME_THRESHOLDS,
                             "complete": training_complete and len(phase1) == N_BASE_SETTINGS}
    (args.sweep / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report.get(k) for k in ("outcome", "training_length")}, indent=2, default=float))
    if "p11" in report:
        print(f"P1.1 on val (exploratory): supported = {report['p11']['supported']}")


def _plot(curve: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    final = curve[(curve.steps == BASE_STEPS) & curve.setting.str.fullmatch(rf"d\d+_k\d+_s{BASE_STEPS}")]
    if final.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for d_sae, part in final.groupby("d_sae"):
        part = part.sort_values("k")
        ax.errorbar(part.k, part.S_mean, yerr=[part.S_mean - part.S_min, part.S_max - part.S_mean],
                    marker="o", capsize=3, label=f"{d_sae} latents")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (active latents per token)")
    ax.set_ylabel("S: share of latents with a recurring partner")
    ax.set_title("Seed overlap of decoder directions, 20k steps, SAEs fitted on train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
