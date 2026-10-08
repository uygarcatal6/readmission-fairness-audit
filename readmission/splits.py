"""
readmission/splits.py — patient-level splits (docs/APPROVED_PROPOSAL.md §2, "Validation").

Outer loop: StratifiedGroupKFold, 5 folds x 5 repeats = 25 train/test pairs.
    group       = patient: all encounters of a patient land in the same fold, so a model
                  is never tested on a patient it saw during training.
    stratified  = every fold keeps roughly the cohort's 11.4 % readmission rate.
    repeats     = a different shuffle per repeat (seed + repeat, see below); each repeat
                  is a full partition, so every encounter is tested exactly once per repeat.

Inner splits are drawn from ONE training fold only; the test fold is never touched:
    inner_folds       K patient-level folds -> one-SE rule for depth / k, Youden threshold
    validation_split  one patient-level hold-out -> ThresholdOptimizer fitting
Inner functions return positions relative to the arrays they receive; map them back with
train[fit_pos] / train[val_pos].

Why StratifiedGroupKFold(shuffle=False) plus our own shuffle: in scikit-learn 1.6.1,
shuffle=True permutes the rows of the per-group class-count matrix
(sklearn/model_selection/_split.py:1042) but then records the permuted row number as the
group placed in the fold (:1058). Folds stay patient-disjoint, yet the balancing is done
on another patient's labels, so stratification is lost (fold rates 0.113-0.119 instead of
0.1144 on this cohort). We shuffle by giving the patients random integer codes instead:
the splitter visits equally-ranked groups in code order, so each seed gives a different,
still stratified, partition.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
N_REPEATS = 5


def _random_group_codes(groups, seed: int) -> np.ndarray:
    """Replace each group label with a random integer code (same label -> same code)."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    codes = np.random.default_rng(seed).permutation(inverse.max() + 1)
    return codes[inverse]


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    train: np.ndarray  # row positions in the cohort frame
    test: np.ndarray


def outer_splits(
    y,
    groups,
    *,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    seed: int = 0,
) -> Iterator[Split]:
    """Yield the repeated patient-level stratified folds, repeat by repeat."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    placeholder = np.zeros(len(y))  # StratifiedGroupKFold only reads y and groups
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    for r in range(n_repeats):
        codes = _random_group_codes(groups, seed + r)
        for f, (train, test) in enumerate(cv.split(placeholder, y, codes)):
            yield Split(repeat=r, fold=f, train=train, test=test)


def inner_folds(
    y,
    groups,
    *,
    n_splits: int = N_SPLITS,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patient-level stratified K folds inside a training fold: [(fit_pos, val_pos), ...]."""
    y = np.asarray(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    return list(cv.split(np.zeros(len(y)), y, _random_group_codes(groups, seed)))


def validation_split(
    y,
    groups,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One patient-level stratified hold-out inside a training fold: (fit_pos, val_pos).

    Implemented as the first fold of a round(1 / val_fraction)-fold inner split, so the
    validation part is patient-disjoint and keeps the readmission rate.
    """
    n_splits = round(1.0 / val_fraction)
    if n_splits < 2:
        raise ValueError(f"val_fraction must be <= 0.5, got {val_fraction}")
    return inner_folds(y, groups, n_splits=n_splits, seed=seed)[0]
