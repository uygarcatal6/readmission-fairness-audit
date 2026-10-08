"""
readmission/tuning.py — choosing tree depth and k on inner validation folds (one-SE rule).

docs/APPROVED_PROPOSAL.md: "Depth and k are chosen with the one-standard-error rule on
validation data. Test folds are never used for any choice."

one-SE rule: among the candidates whose mean inner-CV score is within one standard error of
the best mean, take the SIMPLEST. Course anchor: 02-dt, Occam's razor ("given two models
with a similar generalization error, always choose the simplest one"); "similar" is made
precise as "within one standard error". Simpler means a shallower tree and a LARGER k
(k-NN averages over more neighbours, so its boundary is smoother).

The score is inner-fold ROC-AUC: the operating point flags the top q % by score, so what
matters is the ranking, and AUC measures ranking without a threshold.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier


def _rank(value) -> float:
    """None means 'no limit' (a fully grown tree): the most complex end."""
    return np.inf if value is None else float(value)


def one_se_choice(values: Sequence, scores: np.ndarray, *, prefer: str) -> dict:
    """Pick the simplest candidate within one SE of the best mean score.

    Args:
        values: candidate hyperparameter values, e.g. depths (None = unlimited) or k.
        scores: array (n_values, n_folds) of validation scores, higher is better.
        prefer: "smaller" if smaller values are simpler (depth), "larger" if larger are (k).
    """
    scores = np.asarray(scores, dtype=float)
    if scores.shape[0] != len(values) or scores.shape[1] < 2:
        raise ValueError("scores must be (n_values, n_folds) with at least 2 folds")
    mean = scores.mean(axis=1)
    se = scores.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    best = int(np.argmax(mean))
    threshold = mean[best] - se[best]
    eligible = [i for i in range(len(values)) if mean[i] >= threshold]
    order = sorted(eligible, key=lambda i: _rank(values[i]))
    pick = order[0] if prefer == "smaller" else order[-1]
    return {
        "value": values[pick],
        "index": pick,
        "best_value": values[best],
        # True when the best candidate sits at either end of the grid: then the grid, not the
        # rule, limits the choice and the grid should be widened.
        "at_grid_edge": best in (0, len(values) - 1),
        "threshold": float(threshold),
        "mean": mean.round(5).tolist(),
        "se": se.round(5).tolist(),
    }


def fold_aucs(y: np.ndarray, oof: np.ndarray, folds) -> np.ndarray:
    """(n_candidates, n_folds) ROC-AUC of out-of-fold candidate scores on each validation fold."""
    return np.array([[roc_auc_score(y[val], oof[val, j]) for _, val in folds]
                     for j in range(oof.shape[1])])


def tree_depth_scores(X_fit, y_fit, X_val, depths: Sequence, *, random_state: int = 0,
                      n_jobs: int = -1) -> np.ndarray:
    """Validation scores (n_val, n_depths) of one decision tree per candidate depth."""
    def one(depth):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        return tree.fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
    return np.column_stack(Parallel(n_jobs=n_jobs)(delayed(one)(d) for d in depths))


def knn_k_scores(X_fit, y_fit, X_val, ks: Sequence[int]) -> np.ndarray:
    """Validation scores (n_val, n_ks) of k-NN for every k from ONE neighbour search.

    The neighbours of each validation row are found once, sorted nearest first; the k-NN
    score for any k is the share of readmitted patients among the first k of them — the
    same number KNeighborsClassifier(k, weights="uniform").predict_proba returns.
    """
    k_max = max(ks)
    idx = NearestNeighbors(n_neighbors=k_max, n_jobs=-1).fit(X_fit).kneighbors(
        X_val, return_distance=False)
    positives_so_far = np.cumsum(np.asarray(y_fit)[idx], axis=1)
    return np.column_stack([positives_so_far[:, k - 1] / k for k in ks])


def tree_complexity_curve(X_fit, y_fit, X_val, y_val, *, param: str, values: Sequence,
                          random_state: int = 0) -> dict:
    """Train vs validation AUC along one tree complexity knob (max_depth or ccp_alpha).

    For the report's overfitting / pruning figure (02-dt), computed on one fold only.
    """
    train, val = [], []
    for v in values:
        tree = DecisionTreeClassifier(random_state=random_state, **{param: v}).fit(X_fit, y_fit)
        train.append(roc_auc_score(y_fit, tree.predict_proba(X_fit)[:, 1]))
        val.append(roc_auc_score(y_val, tree.predict_proba(X_val)[:, 1]))
    return {"param": param, "values": list(values), "train_auc": train, "val_auc": val}
