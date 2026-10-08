"""
readmission/source_stats.py — statistics of the source analysis, computed from the
checkpoints written by readmission/sources.py (no model is fitted here).

Kept apart from sources.py so that a change here never invalidates the fitted checkpoints
(the sources fingerprint hashes sources.py only, as the main run's fingerprint leaves out
metrics.py and fairness.py). See sources.py for what each statistic answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from sklearn.metrics import auc, roc_curve

from readmission import cohort, fairness, metrics
from readmission.decision import flag_top_q
from readmission.sources import proxy_classes

N_DEGRADATION_DRAWS = 5
# Worker threads for the noise draws of matched_degradation (numpy sorts and sums release the
# GIL). A runtime knob: every draw has its own seeded generator, so the number of workers
# changes no result. -1 = every CPU the run may use, capped like the models' n_jobs=-1 by
# readmission.runtime (6 on the 12-thread laptop by default, 2 on Colab's free tier).
N_JOBS = -1


def _usable_idx(df: pd.DataFrame):
    """(attribute, group) pairs, mask matrix and the audit's usable groups per attribute."""
    y = df[cohort.TARGET].to_numpy().astype(np.float32)
    pairs, M = fairness._flat(fairness.group_masks(df), fairness.MAIN_ATTRIBUTES)
    n_pos = (M * y).sum(axis=1)
    usable = {"tpr": n_pos >= fairness.MIN_POSITIVES,
              "fpr": M.sum(axis=1) - n_pos >= fairness.MIN_NEGATIVES}
    idx = {(a, m): fairness._usable(pairs, usable, a, m)
           for a in fairness.MAIN_ATTRIBUTES for m in ("tpr", "fpr")}
    return y, pairs, M, idx


def _eo(y, M, idx, flags: np.ndarray) -> dict[str, float]:
    """Audit-rule EO difference of every main attribute for repeat-stacked flags (R, n)."""
    r = fairness._rates(np.ones((1, len(y)), dtype=np.float32), M, y, flags.astype(np.float32))
    return {a: float(np.fmax(fairness._range(r["tpr"], idx[(a, "tpr")]),
                             fairness._range(r["fpr"], idx[(a, "fpr")]))[0])
            for a in fairness.MAIN_ATTRIBUTES}


def _auc(y, score) -> float:
    """ROC-AUC of 0/1 labels: the two calls sklearn's roc_auc_score makes (the ROC curve and
    the area under it), without the input checks it repeats on every call — the bisection
    below makes tens of thousands of calls on the same fold."""
    fpr, tpr, _ = roc_curve(y, score)
    return auc(fpr, tpr)


def _match_sigma(y, z, eps, target: float, auc_z: float) -> float:
    """Noise scale at which ROC-AUC(z + sigma·eps) falls to `target` (0 if already there);
    auc_z = ROC-AUC of z itself."""
    if auc_z <= target:
        return 0.0
    lo, hi = 0.0, 1.0
    while _auc(y, z + hi * eps) > target and hi < 1e4:
        hi *= 2
    for _ in range(30):
        mid = (lo + hi) / 2
        if _auc(y, z + mid * eps) > target:
            lo = mid
        else:
            hi = mid
    return hi


def _degrade_once(yi, folds, z_folds, auc_folds, targets, seed, k, d, shape) -> tuple:
    """One noise draw d of arm k over all folds: flags (R, n), the fold ROC-AUCs reached and
    the largest amount by which a fold stayed above its target (<= 0 unless the noise could
    not get there). Each fold has its own generator, so the draws are independent of the
    order they run in."""
    flags = np.zeros(shape, bool)
    aucs, miss = [], 0.0
    for (r, f, te, q), z, auc_z, target in zip(folds, z_folds, auc_folds, targets, strict=True):
        rng = np.random.default_rng([seed, k, d, r, f])
        yf = yi[te]
        eps = rng.standard_normal(len(te))
        s = z + _match_sigma(yf, z, eps, target, auc_z) * eps
        reached = _auc(yf, s)
        aucs.append(reached)
        miss = max(miss, reached - target)
        flags[r, te] = flag_top_q(s, q, rng=rng)
    return flags, aucs, miss


def matched_degradation(df: pd.DataFrame, oof: dict, arms, *, n_draws: int = N_DEGRADATION_DRAWS,
                        seed: int = 0, n_jobs: int = N_JOBS) -> dict:
    """EO the base ranking shows after losing each arm's ROC-AUC to random noise.

    Per test fold: the base scores' normal scores z (rank-based, same ROC-AUC) plus sigma·noise,
    sigma set so the fold's ROC-AUC equals the arm's on that fold; flag the top q % with the
    fold's q; EO with the audit's rules; mean, min and max over `n_draws` noise draws. On a fold
    where the arm ranks no worse than the base, sigma = 0 and the reference keeps the base
    ranking. `max_auc_miss` = the largest amount by which a fold stayed above its target.
    """
    y, _, M, idx = _usable_idx(df)
    yi = y.astype(int)
    R, n = oof["q_ref"].shape
    folds = oof["folds"]
    # per fold: the base ranking's normal scores and ROC-AUC (the same for every arm and draw)
    z_folds = [norm.ppf(rankdata(oof["scores"]["base"][r, te]) / (len(te) + 1))
               for r, f, te, q in folds]
    auc_folds = [_auc(yi[te], z) for (r, f, te, q), z in zip(folds, z_folds, strict=True)]
    targets = {arm: [_auc(yi[te], oof["scores"][arm][r, te]) for r, f, te, q in folds]
               for arm in arms}
    draws = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_degrade_once)(yi, folds, z_folds, auc_folds, targets[arm], seed, k, d, (R, n))
        for k, arm in enumerate(arms) for d in range(n_draws))
    out = {}
    for k, arm in enumerate(arms):
        per_draw, aucs, miss = [], [], 0.0
        for d in range(n_draws):
            flags, fold_aucs, fold_miss = draws[k * n_draws + d]
            aucs += fold_aucs
            miss = max(miss, fold_miss)
            per_draw.append(_eo(y, M, idx, flags))
        out[arm] = {}
        for a in fairness.MAIN_ATTRIBUTES:
            vals = [p[a] for p in per_draw]
            out[arm] |= {f"eo_{a}": float(np.mean(vals)), f"eo_{a}_min": float(np.min(vals)),
                         f"eo_{a}_max": float(np.max(vals))}
        out[arm]["mean_fold_roc_auc"] = float(np.mean(aucs))
        out[arm]["max_auc_miss"] = float(miss)
    return out


def group_rates(df: pd.DataFrame, oof: dict, keys, attr: str) -> dict:
    """Repeat-averaged TPR and FPR of every group of `attr` for each flag key (point values)."""
    y = df[cohort.TARGET].to_numpy().astype(np.float32)
    masks = fairness.group_masks(df)
    pairs, M = fairness._flat(masks, (attr,))
    ones = np.ones((1, len(y)), dtype=np.float32)
    out = {}
    for k in keys:
        r = fairness._rates(ones, M, y, oof["flags"][k].astype(np.float32))
        out[k] = {g: {"tpr": float(r["tpr"][0, i]), "fpr": float(r["fpr"][0, i])}
                  for i, (_, g) in enumerate(pairs)}
    return out


def _auc_draws(W, y, score_by_repeat) -> np.ndarray:
    """ROC-AUC per draw, averaged over repeats (rows of score_by_repeat)."""
    return np.mean([metrics.weighted_auc(W, y, s.astype(float)) for s in score_by_repeat], axis=0)


def proxy_auc(df: pd.DataFrame, oof: dict, attr: str, *, n_boot: int,
              rng: np.random.Generator) -> dict:
    """How well the other model inputs predict `attr`: one-vs-rest ROC-AUC per class and
    their mean, with patient-bootstrap CIs (race 'Unknown' rows carry weight 0)."""
    classes = proxy_classes(df, attr)
    target = df[attr].astype(str).to_numpy()
    keep = np.isin(target, classes).astype(np.float32)
    codes, n_pat = metrics.patient_codes(df[cohort.GROUP])

    def stats(W):
        Wk = W * keep
        per = {c: _auc_draws(Wk, (target == c).astype(float), oof["scores"][f"proxy_{attr}__{c}"])
               for c in classes}
        per["macro"] = np.mean([per[c] for c in classes], axis=0)
        return per

    point = stats(np.ones((1, len(df)), dtype=np.float32))
    draws = [stats(W) for W in metrics.bootstrap_weights(codes, n_pat, n_boot, rng)]
    return {k: {"estimate": float(point[k][0]), **metrics._interval(np.concatenate([d[k] for d in draws]))}
            for k in point}


def group_ranking(df: pd.DataFrame, scores: np.ndarray, flags: np.ndarray, attributes, *,
                  n_boot: int, rng: np.random.Generator) -> dict:
    """Per group of each attribute (scores and flags: (repeats, n) of one model and rule):

    roc_auc              ROC-AUC within the group (patient-bootstrap CI)
    tpr / fpr            the group's rates under the shared rule (the audited flags)
    tpr_at_overall_fpr   the group's TPR at its OWN cut-off that gives it the overall FPR of
                         the rule (its ROC curve read at that FPR; tied scores are split in
                         proportion, as random tie-breaking does in expectation): its ranking
                         quality at the operating point. tpr − tpr_at_overall_fpr is what the
                         shared cut-off adds or removes for the group (where its scores lie).
    mean risks           mean predicted risk of its readmitted and not-readmitted encounters
    """
    y = df[cohort.TARGET].to_numpy().astype(float)
    pos, neg = y == 1, y == 0
    masks = fairness.group_masks(df)
    codes, n_pat = metrics.patient_codes(df[cohort.GROUP])
    pairs = {(a, g): masks[a][g] for a in attributes for g in masks[a]}

    def stats(W):  # the ROC-AUC inside a group needs only the group's rows
        return {(a, g): _auc_draws(W[:, rows], y[rows], scores[:, rows])
                for (a, g), rows in pairs.items()}

    point = stats(np.ones((1, len(df)), dtype=np.float32))
    draws = [stats(W) for W in metrics.bootstrap_weights(codes, n_pat, n_boot, rng)]
    out = {}
    for (a, g), rows in pairs.items():
        own = []
        for s, f in zip(scores.astype(float), flags, strict=True):
            fg, tg, _ = roc_curve(y[rows], s[rows])
            own.append(float(np.interp(f[neg].mean(), fg, tg)))
        out.setdefault(a, {})[g] = {
            "roc_auc": {"estimate": float(point[(a, g)][0]),
                        **metrics._interval(np.concatenate([d[(a, g)] for d in draws]))},
            "tpr": float(np.mean([f[rows & pos].mean() for f in flags])),
            "fpr": float(np.mean([f[rows & neg].mean() for f in flags])),
            "tpr_at_overall_fpr": float(np.mean(own)),
            "mean_risk_readmitted": float(scores[:, rows & pos].mean()),
            "mean_risk_not_readmitted": float(scores[:, rows & neg].mean()),
            "n_pos": int(y[rows].sum())}
    return out
