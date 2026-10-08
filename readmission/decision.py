"""
readmission/decision.py — turning risk scores into yes/no flags.

Main rule (docs/APPROVED_PROPOSAL.md §2, "Usability"): in each test fold, flag the
highest-risk q % of encounters, with q = the readmission rate of the TRAINING fold. It
models a hospital that can follow up a fixed number of patients, and it reads no test
labels. Sensitivity checks reuse flag_top_q with q = 5/10/20/30 %, plus a Youden threshold
chosen on inner-CV predictions and a class-weighted model at 0.5.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

SENSITIVITY_Q = (0.05, 0.10, 0.20, 0.30)


def flag_top_q(
    score: np.ndarray,
    q: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flag the q share of rows with the highest score.

    Args:
        score: predicted readmission risk for one test fold, shape (n,).
        q: share to flag, 0 <= q <= 1 (e.g. the training-fold readmission rate, ~0.114).
        rng: random generator for breaking ties at the cut-off (see below). Required:
            the caller (the evaluation loop) owns the random stream, so one seed fixes
            every flag of a run; there is deliberately no row-order fallback.

    Returns:
        Boolean array of shape (n,), True = flagged for follow-up.

    Contract (tests/test_readmission_decision.py):
        * exactly k = round(q * n) rows are flagged — the capacity is fixed;
        * every flagged row has a score >= every unflagged row;
        * q = 0 flags nothing, q = 1 flags everything.

    Why ties matter: logistic-regression and random-forest scores are practically all
    distinct, but a decision tree can only output one score per leaf (a depth-5 tree has at
    most 32) and k-NN only k + 1 values, so large blocks of encounters share the cut-off
    score and only part of the block fits into the capacity. Choosing that part by row
    position would follow the file order (the cohort is sorted by encounter_id), which is
    arbitrary and could line up with a subgroup; a seeded random order reflects that the
    model cannot tell the tied encounters apart.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be between 0 and 1, got {q}")
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = round(q * n)
    # lexsort sorts by the last key first; we want highest score first,
    # ties broken by a random secondary key so the choice is fair.
    tiebreak = rng.random(n)
    order = np.lexsort((tiebreak, -score))
    flags = np.zeros(n, dtype=bool)
    flags[order[:k]] = True
    return flags


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Threshold that maximises Youden's J = TPR - FPR on (inner-CV) predictions.

    Rows with score >= threshold are flagged. roc_curve's first threshold is +inf (flag
    nothing, J = 0); it is only returned if no finite threshold does better.
    """
    fpr, tpr, thresholds = roc_curve(y_true, score)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best])
