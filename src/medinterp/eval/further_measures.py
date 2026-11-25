"""Further secondary measures of protocol section 4 (added 2026-11-25) for WP1 phase 3 and the clean harvest.

  joint_seed_overlap    one-to-one matching on decoder and on encoder cosine; shared if both agree (Paulo &
                        Belrose 2025)
  pw_mcc                mean decoder cosine of the one-to-one matched pairs (Song et al. 2025)
  image_features        one pass over the tokens of chosen images: per-image max codes and mean-pooled codes
  secondary_targets     the probe targets available on a harvest (opacity per dataset, CheXpert labels, sex,
                        devices, view if recorded, dataset identity)
  RecoveryProbes        performance recovery: probes fitted once on pooled original activations, then
                        applied to pooled reconstructions
  absorption            absorption rate of the parent latent on the CheXpert opacity hierarchy
  targeted_ablation     zeroing the n latents an L1 probe weights most for a confounder (after SAEBench TPP)

Every function takes the split names it selects on and evaluates on, so the development data (WP1: select on
train, evaluate on val; the test split is not read) and the clean harvest (protocol: val and test) use the
same code. Thresholds and definitions live in the protocol, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc_columns
from .probing import C_GRID, fit_eval_tuned
from .stability import cosine_matrix, one_to_one
from .stats import cluster_bootstrap_difference

CHEXPERT_LABELS = ("No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                   "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                   "Pleural Other", "Fracture", "Support Devices")
ABSORPTION_PARENT = "Lung Opacity"
ABSORPTION_CHILDREN = ("Consolidation", "Pneumonia", "Edema", "Atelectasis")


# ---------------------------------------------------------------- seed matching


def joint_seed_overlap(sae_a, sae_b, threshold: float) -> dict:
    """Shares of run A's latents that are shared with run B under decoder matching, encoder matching and both.

    Encoder directions are the columns of ``W_enc``. A latent counts as jointly shared if both one-to-one
    matchings give it the same partner and both cosines reach ``threshold``.
    Complexity: two assignments, O(d_sae^3) each.
    """
    dec_cos, dec_partner = one_to_one(cosine_matrix(sae_a.W_dec.float(), sae_b.W_dec.float()))
    enc_cos, enc_partner = one_to_one(cosine_matrix(sae_a.W_enc.T.float(), sae_b.W_enc.T.float()))
    dec_ok = np.nan_to_num(dec_cos, nan=-np.inf) >= threshold
    enc_ok = np.nan_to_num(enc_cos, nan=-np.inf) >= threshold
    joint = dec_ok & enc_ok & (dec_partner == enc_partner) & (dec_partner >= 0)
    return {"share_decoder": float(dec_ok.mean()), "share_encoder": float(enc_ok.mean()),
            "share_joint": float(joint.mean()), "pw_mcc": float(np.nanmean(dec_cos))}


def pw_mcc(dec_a: torch.Tensor, dec_b: torch.Tensor) -> float:
    """Mean decoder cosine of the one-to-one matched pairs over all latents (threshold-free)."""
    cos, _ = one_to_one(cosine_matrix(dec_a.float(), dec_b.float()))
    return float(np.nanmean(cos))


# ---------------------------------------------------------------- image-level features


@dataclass
class ImageFeatures:
    """Per dataset, arrays over all images of the dataset; rows of images outside the passed masks stay 0."""
    max_codes: dict[str, np.ndarray] = field(default_factory=dict)     # float16 [n_images, d_sae]
    mean_codes: dict[str, np.ndarray] = field(default_factory=dict)    # float32 [n_images, d_sae]


@torch.no_grad()
def image_features(acts, enc, keep: dict[str, np.ndarray], device: str = "cpu", want_mean: bool = True,
                   images_per_chunk: int = 256) -> ImageFeatures:
    """Encode the tokens of the images in ``keep`` once; return per-image max and mean-pooled codes.

    The mean-pooled reconstruction of an image is ``mean_codes @ W_dec + b_dec`` because decoding is linear.
    Complexity: O(N_kept_tokens * d_model * d_sae).
    """
    out = ImageFeatures()
    P = acts.n_patches
    for name, block in acts.blocks.items():
        mx = np.zeros((block.n_images, enc.d_sae), dtype=np.float16)
        mn = np.zeros((block.n_images, enc.d_sae), dtype=np.float32) if want_mean else None
        for img0, x in acts.iter_token_chunks(name, images_per_chunk):
            n_img = x.shape[0] // P
            sel = np.flatnonzero(keep[name][img0:img0 + n_img])
            if not sel.size:
                continue
            xs = x.view(n_img, P, -1)[torch.from_numpy(sel)].reshape(-1, x.shape[1]).to(device)
            z = enc.encode(xs).view(len(sel), P, -1)
            mx[img0 + sel] = z.amax(1).float().cpu().numpy().astype(np.float16)
            if want_mean:
                mn[img0 + sel] = z.mean(1).float().cpu().numpy()
        out.max_codes[name] = mx
        if want_mean:
            out.mean_codes[name] = mn
    return out


@torch.no_grad()
def pooled_activations(acts, keep: dict[str, np.ndarray], images_per_chunk: int = 256) -> dict[str, np.ndarray]:
    """Mean-pooled original tokens per image (rows outside ``keep`` stay 0). Complexity: O(N_kept_tokens * d)."""
    P, out = acts.n_patches, {}
    for name, block in acts.blocks.items():
        res = np.zeros((block.n_images, acts.d_model), dtype=np.float32)
        for img0, x in acts.iter_token_chunks(name, images_per_chunk):
            n_img = x.shape[0] // P
            sel = np.flatnonzero(keep[name][img0:img0 + n_img])
            if sel.size:
                res[img0 + sel] = x.view(n_img, P, -1)[torch.from_numpy(sel)].mean(1).numpy()
        out[name] = res
    return out


def decode_pooled(enc, mean_codes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean-pooled reconstruction from mean-pooled codes; exact because every dictionary's decoder is affine.
    Works for trained SAEs and for the linear nulls (signed PCA, signed basis) through their own ``decode``."""
    with torch.no_grad():
        return {d: enc.decode(torch.from_numpy(c)).float().numpy() for d, c in mean_codes.items()}


# ---------------------------------------------------------------- targets


def secondary_targets(manifests: dict, opacity: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """``{target: {dataset: 1 / 0 / NaN per image}}`` for the targets the harvest records.

    - ``opacity_<d>``: the primary contrast within dataset d (other datasets NaN)
    - CheXpert's 14 labels: 1 positive, 0 negative or not mentioned (blank), NaN uncertain (-1); CheXpert only
    - ``sex``: 1 male, 0 female, NaN otherwise; ``view``: 1 PA, 0 AP, where a manifest has a ``view`` column
    - ``dataset_<d>``: one-vs-rest dataset identity over all datasets
    Datasets without a field get NaN, so a probe sees only the images that carry the label.
    """
    names = list(manifests)
    nan = {d: np.full(len(manifests[d]), np.nan) for d in names}
    targets: dict[str, dict[str, np.ndarray]] = {}
    for d in names:
        if d in opacity and np.isfinite(opacity[d]).any():
            targets[f"opacity_{d}"] = {**{o: nan[o] for o in names}, d: opacity[d].astype(float)}
    for d in names:
        man = manifests[d]
        for lab in CHEXPERT_LABELS:
            if lab in man.columns:
                v = man[lab].to_numpy(dtype=float)
                y = np.where(np.isnan(v), 0.0, np.where(v == -1, np.nan, v))
                targets.setdefault(lab, {o: nan[o] for o in names})[d] = y
        if "sex" in man.columns:
            s = man["sex"].astype(str).str.lower()
            targets.setdefault("sex", {o: nan[o] for o in names})[d] = np.where(
                s == "male", 1.0, np.where(s == "female", 0.0, np.nan))
        if "view" in man.columns:
            v = man["view"].astype(str).str.upper()
            targets.setdefault("view", {o: nan[o] for o in names})[d] = np.where(
                v == "PA", 1.0, np.where(v == "AP", 0.0, np.nan))
    for d in names:
        targets[f"dataset_{d}"] = {o: np.full(len(manifests[o]), 1.0 if o == d else 0.0) for o in names}
    return {t: v for t, v in targets.items() if _usable(v)}


def _usable(target: dict[str, np.ndarray]) -> bool:
    y = np.concatenate(list(target.values()))
    y = y[np.isfinite(y)]
    return y.size > 0 and 0 < y.sum() < y.size


def _stack(features: dict[str, np.ndarray], target: dict[str, np.ndarray], splits: dict[str, dict[str, np.ndarray]],
           split: str, groups: dict[str, np.ndarray] | None = None):
    """Concatenate the labelled images of ``split`` over datasets (dataset order of ``features``)."""
    xs, ys, gs = [], [], []
    for d in features:
        m = splits[d][split] & np.isfinite(target[d])
        xs.append(features[d][m])
        ys.append(target[d][m] == 1)
        if groups is not None:
            gs.append(np.char.add(f"{d}:", groups[d][m].astype(str)))
    X, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys)
    return (X, y, np.concatenate(gs)) if groups is not None else (X, y)


# ---------------------------------------------------------------- performance recovery


class RecoveryProbes:
    """Performance recovery (protocol section 4): one probe per target on the mean-pooled original activation,
    fitted on ``fit_split`` with C tuned on ``tune_split``, then frozen.

    ``score(recon)`` applies the frozen probes to pooled reconstructions of the ``eval_split`` images and
    reports per target both AUROCs, the recovered share (AUROC_rec - 0.5) / (AUROC_orig - 0.5) and the
    paired cluster bootstrap of AUROC_orig - AUROC_rec. Complexity: len(C_GRID) fits per target at
    construction; one decision function per target and dictionary afterwards, plus the bootstrap.
    """

    def __init__(self, pooled: dict[str, np.ndarray], targets: dict[str, dict[str, np.ndarray]],
                 splits: dict[str, dict[str, np.ndarray]], groups: dict[str, np.ndarray],
                 fit_split: str = "train", tune_split: str = "val", eval_split: str = "val"):
        self.targets, self.splits, self.groups, self.eval_split = targets, splits, groups, eval_split
        self.models, self.orig_auc = {}, {}
        for t, y_all in targets.items():
            X_tr, y_tr = _stack(pooled, y_all, splits, fit_split)
            X_va, y_va = _stack(pooled, y_all, splits, tune_split)
            X_ev, y_ev = _stack(pooled, y_all, splits, eval_split)
            if min(len(np.unique(y_tr)), len(np.unique(y_va)), len(np.unique(y_ev))) < 2:
                continue
            _, C, _ = fit_eval_tuned(X_tr, y_tr, X_va, y_va, {}, C_GRID)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            self.models[t] = clf.fit(X_tr, y_tr)
            self.orig_auc[t] = float(roc_auc_score(y_ev, clf.decision_function(X_ev)))
        self._orig_scores = {t: self.models[t].decision_function(_stack(pooled, self.targets[t], splits, eval_split)[0])
                             for t in self.models}

    def score(self, recon: dict[str, np.ndarray], n_resamples: int, confidence: float, seed: int) -> list[dict]:
        rows = []
        for t, clf in self.models.items():
            X, y, g = _stack(recon, self.targets[t], self.splits, self.eval_split, self.groups)
            s_rec, s_orig = clf.decision_function(X), self._orig_scores[t]
            auc_rec = float(roc_auc_score(y, s_rec))

            def auc(scores):
                return lambda idx: (roc_auc_score(y[idx], scores[idx]) if 0 < y[idx].sum() < idx.size else np.nan)
            boot = cluster_bootstrap_difference(auc(s_orig), auc(s_rec), g, n_resamples, confidence, seed)
            denom = self.orig_auc[t] - 0.5
            rows.append({"target": t, "auroc_orig": self.orig_auc[t], "auroc_rec": auc_rec,
                         "recovered_share": (auc_rec - 0.5) / denom if denom != 0 else np.nan,
                         "diff": self.orig_auc[t] - auc_rec, "diff_ci_low": boot.ci_low, "diff_ci_high": boot.ci_high,
                         "n_eval": int(y.size)})
        return rows


# ---------------------------------------------------------------- absorption


def absorption(max_codes: np.ndarray, labels: dict[str, np.ndarray], select: np.ndarray, evaluate: np.ndarray,
               parent: str = ABSORPTION_PARENT, children: tuple[str, ...] = ABSORPTION_CHILDREN) -> dict:
    """Absorption on the CheXpert hierarchy (protocol section 4).

    ``max_codes`` [n_images, d_sae] image codes of CheXpert; ``labels`` maps a label to 1 / 0 / NaN per image
    (as in ``secondary_targets``); ``select`` and ``evaluate`` are image masks. The parent latent is the one
    with the highest AUROC for the parent on ``select``. Per child: among ``evaluate`` images positive for the
    parent and the child, the share where the parent latent's image code is zero; the reference is the same
    share among parent-positive images with no child positive. Raises ValueError if the parent has one class.
    """
    yp = labels[parent]
    sel = select & np.isfinite(yp)
    if len(np.unique(yp[sel])) < 2:
        raise ValueError("parent label has one class on the selection images")
    auc = auroc_columns(max_codes[sel].astype(np.float32), yp[sel] == 1)
    j = int(np.argmax(auc))
    off = max_codes[:, j] == 0
    ppos = evaluate & (yp == 1)
    any_child = np.zeros_like(ppos)
    out = {"parent_latent": j, "parent_auroc_select": float(auc[j])}
    for c in children:
        both = ppos & (labels[c] == 1)
        any_child |= labels[c] == 1
        out[f"rate_{c}"] = float(off[both].mean()) if both.any() else np.nan
        out[f"n_{c}"] = int(both.sum())
    ref = ppos & ~any_child
    out["rate_no_child"] = float(off[ref].mean()) if ref.any() else np.nan
    out["n_no_child"] = int(ref.sum())
    return out


# ---------------------------------------------------------------- targeted ablation


def _l1_logistic(C: float) -> LogisticRegression:
    """L1-penalised logistic regression; scikit-learn 1.8 replaced ``penalty`` by ``l1_ratio``, and the Kaggle
    image may run either version."""
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, class_weight="balanced", max_iter=2000)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=2000)


def _l1_probe(X_tr, y_tr, X_va, y_va):
    best = None
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(), _l1_logistic(C))
        clf.fit(X_tr, y_tr)
        a = roc_auc_score(y_va, clf.decision_function(X_va))
        if best is None or (a, C) > (best[0], best[1]):
            best = (a, C, clf)
    return best[2]


def targeted_ablation(codes: dict[str, np.ndarray], targets: dict[str, dict[str, np.ndarray]],
                      splits: dict[str, dict[str, np.ndarray]], confounders: tuple[str, ...],
                      comparisons: tuple[str, ...], ns: tuple[int, ...], fit_split: str = "train",
                      tune_split: str = "val", eval_split: str = "val") -> list[dict]:
    """Targeted ablation of confounders (protocol section 4, after SAEBench's TPP).

    One L1 probe per target in ``confounders + comparisons`` on image codes (fitted on ``fit_split``, C tuned
    on ``tune_split``). For each confounder and n: the n latents with the largest absolute weight are set to
    zero in the codes of ``eval_split``; every probe is re-scored on the ablated codes. Reported: the
    confounder probe's AUROC drop, the mean drop of the other probes, and their difference (selectivity).
    Targets absent from ``targets`` are skipped. Complexity: len(C_GRID) L1 fits per target.
    """
    names = [t for t in (*confounders, *comparisons) if t in targets]
    probes, evals = {}, {}
    for t in names:
        X_tr, y_tr = _stack(codes, targets[t], splits, fit_split)
        X_va, y_va = _stack(codes, targets[t], splits, tune_split)
        X_ev, y_ev = _stack(codes, targets[t], splits, eval_split)
        if min(len(np.unique(y_tr)), len(np.unique(y_va)), len(np.unique(y_ev))) < 2:
            continue
        probes[t] = _l1_probe(X_tr, y_tr, X_va, y_va)
        evals[t] = (X_ev, y_ev, float(roc_auc_score(y_ev, probes[t].decision_function(X_ev))))
    rows = []
    for conf in confounders:
        if conf not in probes:
            continue
        weights = np.abs(probes[conf][-1].coef_.ravel())
        for n in ns:
            cols = np.argsort(-weights, kind="stable")[:n]
            drops = {}
            for t, (X, y, base) in evals.items():
                Xa = X.copy()
                Xa[:, cols] = 0
                drops[t] = base - float(roc_auc_score(y, probes[t].decision_function(Xa)))
            others = [v for t, v in drops.items() if t != conf]
            rows.append({"confounder": conf, "n": n, "n_nonzero_weights": int((weights > 0).sum()),
                         "auroc_before": evals[conf][2], "drop": drops[conf],
                         "mean_drop_others": float(np.mean(others)) if others else np.nan,
                         "selectivity": drops[conf] - float(np.mean(others)) if others else np.nan,
                         **{f"drop_{t}": v for t, v in drops.items() if t != conf}})
    return rows
