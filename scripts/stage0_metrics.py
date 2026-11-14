"""Stage 0, run 2: metrics that random dictionaries cannot fake.

  1. reconstruction FVE of k-sparse codes with a fitted linear decoder (held-out images)
  2. sparse probing (k = 1, 5, 10, 50 latents chosen on the source dataset) with cross-dataset transfer
  3. null-adjusted monosemanticity on CheXpert finding sets (Wesp et al. 2026 style)

Outputs in results/stage0/: recon.csv, probe_transfer.csv, selectivity_<dictionary>.csv,
metrics_summary.json. Run: ~/.venvs/nla-mech-interp/bin/python scripts/stage0_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Make the package importable without relying on the editable-install .pth file, which macOS
# repeatedly marks hidden in this iCloud-synced folder (hidden .pth files are skipped by Python).
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from medinterp.data.archive import DEFAULT_RUN_DIR, ArchivedActivations
from medinterp.data.manifests import CHEXPERT_LABELS, build_manifests
from medinterp.dictionaries.baselines import random_sae, signed_basis, signed_pca
from medinterp.dictionaries.sae import TopKSAE
from medinterp.eval.probing import fit_eval, sparse_probe_transfer
from medinterp.eval.recon import FVEAccumulator, RidgeAccumulator, predict, split_images
from medinterp.eval.selectivity import monosemanticity

OUT = Path(__file__).resolve().parents[1] / "results/stage0"
SEED = 0
TEST_FRAC = 0.2
RECON_TRAIN_IMAGES, RECON_TEST_IMAGES = 600, 200   # per dataset
PROBE_KS = (1, 5, 10, 50)
SELECTIVITY_QUANTILE = 0.95


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup(acts: ArchivedActivations):
    """Labels, splits and reconstruction image samples, shared by every dictionary."""
    man = build_manifests(acts)
    rng = np.random.default_rng(SEED)
    info = {}
    for d, m in man.items():
        patients = m.path.str.split("/").str[2].to_numpy() if d == "chexpert" else None
        test = split_images(len(m), patients, TEST_FRAC, SEED)
        g = m.group.to_numpy()
        labelled = (g == 0) | (g == 5)
        tr_idx, te_idx = np.flatnonzero(~test), np.flatnonzero(test)
        info[d] = {
            "test": test, "labelled": labelled, "positive": g == 0,
            "recon_train": set(rng.choice(tr_idx, min(RECON_TRAIN_IMAGES, tr_idx.size), replace=False)),
            "recon_test": set(rng.choice(te_idx, min(RECON_TEST_IMAGES, te_idx.size), replace=False)),
        }
    findings = man["chexpert"][list(CHEXPERT_LABELS)].to_numpy() == 1
    findings = np.column_stack([findings, ~findings.any(1)])        # pseudo-finding "none"
    return info, findings


def score_dictionary(acts, enc, info, d_model):
    """One encode pass for image codes + ridge statistics, one short pass for held-out FVE."""
    n_lat = enc.d_sae
    image_codes, pooled = {}, {}
    ridge = RidgeAccumulator(n_lat, d_model)
    for d, block in acts.blocks.items():
        image_codes[d] = np.zeros((block.n_images, n_lat), dtype=np.float16)
        pooled[d] = np.zeros((block.n_images, d_model), dtype=np.float32)
        for img0, x in acts.iter_token_chunks(d, 256):
            n_img = x.shape[0] // acts.n_patches
            z = enc.encode(x)
            image_codes[d][img0:img0 + n_img] = z.view(n_img, acts.n_patches, -1).amax(1).numpy()
            pooled[d][img0:img0 + n_img] = x.view(n_img, acts.n_patches, -1).mean(1).numpy()
            sel = [i for i in range(n_img) if img0 + i in info[d]["recon_train"]]
            if sel:
                rows = torch.cat([torch.arange(i * acts.n_patches, (i + 1) * acts.n_patches) for i in sel])
                ridge.add(z[rows], x[rows])
    W = ridge.solve()
    fve, fve_native = FVEAccumulator(d_model), FVEAccumulator(d_model)
    for d in acts.blocks:
        for img0, x in acts.iter_token_chunks(d, 256):
            n_img = x.shape[0] // acts.n_patches
            sel = [i for i in range(n_img) if img0 + i in info[d]["recon_test"]]
            if not sel:
                continue
            rows = torch.cat([torch.arange(i * acts.n_patches, (i + 1) * acts.n_patches) for i in sel])
            z = enc.encode(x[rows])
            fve.add(x[rows], predict(W, z))
            if isinstance(enc, TopKSAE):
                fve_native.add(x[rows], enc.decode(z))
    native = fve_native.fve() if fve_native.n else None
    return image_codes, pooled, fve.fve(), native


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acts = ArchivedActivations()
    info, findings = setup(acts)
    trained = TopKSAE.from_checkpoint(DEFAULT_RUN_DIR / "checkpoints/step_0200000")
    k, d_sae, d_model = trained.k, trained.d_sae, acts.d_model

    def pca():
        rng = np.random.default_rng(SEED)
        parts = [x[torch.from_numpy(rng.random(x.shape[0]) < 50_000 / b.n_tokens)]
                 for n, b in acts.blocks.items() for _, x in acts.iter_token_chunks(n, 1024)]
        return signed_pca(torch.cat(parts), d_model, k)

    dictionaries = {
        "trained_step200k": lambda: trained,
        "trained_step020k": lambda: TopKSAE.from_checkpoint(DEFAULT_RUN_DIR / "checkpoints/step_0020000"),
        **{f"random_seed{s}": (lambda s=s: random_sae(d_model, d_sae, k, seed=s)) for s in (0, 1, 2)},
        "signed_pca": pca,
        "signed_basis": lambda: signed_basis(d_model, k),
    }

    recon_rows, probe_rows, sel_tables = [], [], {}
    dense_done = False
    for name, build in dictionaries.items():
        log(f"{name}: encode + reconstruction")
        codes, pooled, fve, native = score_dictionary(acts, build(), info, d_model)
        recon_rows.append({"dictionary": name, "fve_linear_decoder": fve, "fve_native_decoder": native})
        log(f"{name}: FVE linear {fve:.4f}  native {native}")

        lab = {d: codes[d][info[d]["labelled"]] for d in codes}
        y = {d: info[d]["positive"][info[d]["labelled"]] for d in codes}
        te = {d: info[d]["test"][info[d]["labelled"]] for d in codes}
        for r in sparse_probe_transfer(lab, y, te, PROBE_KS):
            probe_rows.append({"dictionary": name, **r})

        if not dense_done:  # dense reference: probe on all 1024 pooled dimensions
            P = {d: pooled[d][info[d]["labelled"]] for d in pooled}
            for src in P:
                evals = {src: (P[src][te[src]], y[src][te[src]])}
                evals.update({t: (P[t], y[t]) for t in P if t != src})
                for tgt, auc in fit_eval(P[src][~te[src]], y[src][~te[src]], evals).items():
                    probe_rows.append({"dictionary": "dense_pooled", "source": src, "k": d_model,
                                       "target": tgt, "auroc": auc})
            dense_done = True

        log(f"{name}: selectivity")
        s = monosemanticity(codes["chexpert"], findings, top_n=10, seed=SEED)
        sel_tables[name] = pd.DataFrame(s, columns=["coherence", "specificity", "mono"]).assign(latent=np.arange(len(s)))
        sel_tables[name].to_csv(OUT / f"selectivity_{name}.csv", index=False)
        del codes, pooled

    recon = pd.DataFrame(recon_rows)
    recon.to_csv(OUT / "recon.csv", index=False)
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(OUT / "probe_transfer.csv", index=False)

    # Selectivity threshold from random SAEs; count latents above it per dictionary.
    rand_mono = pd.concat([sel_tables[f"random_seed{s}"].mono for s in (0, 1, 2)]).dropna()
    thr = float(np.quantile(rand_mono, SELECTIVITY_QUANTILE))
    summary = {"settings": {"seed": SEED, "test_frac": TEST_FRAC, "probe_ks": PROBE_KS,
                            "recon_images": [RECON_TRAIN_IMAGES, RECON_TEST_IMAGES],
                            "mono_threshold_q95_random": thr},
               "recon": recon.to_dict("records"), "selectivity": {}}
    for name, t in sel_tables.items():
        m = t.mono.dropna().sort_values(ascending=False)
        summary["selectivity"][name] = {"n_scored": int(m.size), "above_random_q95": int((m > thr).sum()),
                                        "top10_mean": float(m.head(10).mean()), "median": float(m.median())}
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    log("done")


if __name__ == "__main__":
    main()
