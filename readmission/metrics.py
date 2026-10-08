"""
readmission/metrics.py — metrics and patient-level bootstrap confidence intervals.

docs/APPROVED_PROPOSAL.md: "Confidence intervals come from a bootstrap over patients, not
encounters." One bootstrap replicate draws P patients with replacement (P = number of
patients); every encounter of a patient drawn c times gets weight c. All metrics below are
therefore WEIGHTED metrics; the point estimate is the same function with all weights 1.

Repeats: the 5 repeats give 5 out-of-fold predictions per encounter. A metric is computed per
repeat and averaged over repeats; inside a bootstrap replicate the SAME patient draw is used
for all repeats, so the interval covers patient sampling while the repeat average smooths the
split-to-split noise.

Ranking metrics (need scores):   roc_auc, pr_auc, brier, brier_skill
Decision metrics (need flags):   accuracy, acc_minus_nir, tpr, fpr, ppv, selection_rate
    acc_minus_nir = accuracy - NIR with NIR FIXED at the evaluated rows' no-information rate
    (0.8856 for the full cohort), exactly as approved: "the lower confidence bound of
    (accuracy - 0.886) is above zero". acc_minus_nir_paired recomputes the NIR inside every
    bootstrap draw (a slightly narrower interval); it is reported, never used for the gate.
    Draws where a metric is undefined (e.g. no flagged rows in a small subgroup) are counted
    in n_nan; a CI with more than 1 % undefined draws is marked ci_reliable = False.
brier_skill = 1 - Brier / Brier_ref, reference forecast = the training-fold readmission rate
    of the fold that scored the encounter ("predict the base rate for everyone").
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

CI_LEVEL = 0.95


# ── weighted metric kernels; W has shape (B, n): one row of weights per replicate ─────────


def _tie_blocks(score: np.ndarray, descending: bool):
    order = np.argsort(-score if descending else score, kind="mergesort")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return order, starts


def _cumulative_weight(W: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """(B, len(rows) + 1): total weight of the first k of `rows` for k = 0 .. len(rows).

    The weights are integer counts (times a patient was drawn) whose sums stay far below
    2**24, so the float32 running sums are exact: every value below is the same to the last
    bit whatever the order or dtype it is summed in.
    """
    out = np.zeros((len(W), len(rows) + 1), dtype=np.float32)
    np.cumsum(W[:, rows], axis=1, out=out[:, 1:])
    return out


def weighted_auc(W: np.ndarray, y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """ROC-AUC per row of W. Ties count 1/2 (same as sklearn's roc_auc_score).

    Mann-Whitney form: the weighted share of (positive, negative) pairs in which the positive
    scores higher, a tie counting one half. Sorting the negatives by score gives, for every
    positive, the weight of the negatives below it (and tied with it) from one running sum.
    """
    order = np.argsort(score, kind="mergesort")
    yo = y[order]
    pos, neg = order[yo == 1], order[yo == 0]  # both in increasing score order
    below = np.searchsorted(score[neg], score[pos], side="left")  # negatives scoring lower
    tied = np.searchsorted(score[neg], score[pos], side="right") - below
    cum = _cumulative_weight(W, neg)
    Wp = W[:, pos]
    num = (Wp * (cum[:, below] + 0.5 * (cum[:, below + tied] - cum[:, below]))).sum(
        axis=1, dtype=np.float64)
    return num / (Wp.sum(axis=1, dtype=np.float64) * cum[:, -1].astype(np.float64))


def weighted_average_precision(W: np.ndarray, y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """PR-AUC as average precision per row of W (same definition as sklearn)."""
    order, starts = _tie_blocks(score, descending=True)
    yo = y[order]
    pos, neg = order[yo == 1], order[yo == 0]  # both in decreasing score order
    ends = np.r_[starts[1:], len(order)]  # block b holds sorted positions starts[b]:ends[b]
    # cumulative true / false positives at the end of every block, read off the running sums
    # of the positives and of the negatives (exact integers, so the same values as before;
    # order="C" keeps the row-wise sum below in its usual summation order)
    tp = _cumulative_weight(W, pos)[:, np.cumsum(yo == 1)[ends - 1]].astype(np.float64, order="C")
    fp = _cumulative_weight(W, neg)[:, np.cumsum(yo == 0)[ends - 1]].astype(np.float64, order="C")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall_step = np.diff(np.c_[np.zeros(len(W)), tp], axis=1) / tp[:, -1:]
    return (recall_step * precision).sum(axis=1)


def weighted_brier(W: np.ndarray, y: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return (W * (forecast - y) ** 2).sum(axis=1) / W.sum(axis=1)


def weighted_decision(W: np.ndarray, y: np.ndarray, flag: np.ndarray,
                      nir: float | None = None) -> dict[str, np.ndarray]:
    """Confusion-matrix metrics per row of W for yes/no flags.

    nir: fixed no-information rate for acc_minus_nir; None = recompute it per row of W.
    Vectors are cast to W's dtype: the sums are of integer weights below 2**24, so float32 is
    exact and avoids a slow float32 x float64 product.
    """
    f = flag.astype(W.dtype)
    yw = np.asarray(y, dtype=W.dtype)
    # one product for the four sums (true positives, false positives, positives, total)
    tp, fp, pos, tot = (W @ np.stack([yw * f, (1 - yw) * f, yw, np.ones_like(yw)], axis=1)).T
    neg = tot - pos
    tn = neg - fp
    p = pos / tot
    accuracy = (tp + tn) / tot
    paired = accuracy - np.maximum(p, 1 - p)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "accuracy": accuracy,
            "acc_minus_nir": paired if nir is None else accuracy - nir,
            "acc_minus_nir_paired": paired,
            "tpr": tp / pos,
            "fpr": fp / neg,
            "ppv": tp / (tp + fp),
            "selection_rate": (tp + fp) / tot,
            "prevalence": p,
        }


# ── patient bootstrap ─────────────────────────────────────────────────────────────────────


def patient_codes(groups) -> tuple[np.ndarray, int]:
    """Patient label per encounter -> integer code 0..P-1 and P."""
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    return codes, int(codes.max()) + 1


def bootstrap_weights(codes: np.ndarray, n_patients: int, n_boot: int, rng: np.random.Generator,
                      batch: int = 100) -> Iterator[np.ndarray]:
    """Yield weight matrices (b, n_encounters): encounter weight = times its patient was drawn."""
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        counts = np.stack([np.bincount(rng.integers(0, n_patients, n_patients),
                                       minlength=n_patients) for _ in range(b)])
        yield counts[:, codes].astype(np.float32)
        done += b


MAX_NAN_SHARE = 0.01


def _interval(values: np.ndarray) -> dict:
    """Percentile CI over the defined draws, with a count of the undefined ones."""
    a = (1 - CI_LEVEL) / 2
    n_nan = int(np.isnan(values).sum())
    if n_nan == len(values):
        return {"ci": [float("nan"), float("nan")], "n_nan": n_nan, "ci_reliable": False}
    lo, hi = np.nanquantile(values, [a, 1 - a])
    return {"ci": [float(lo), float(hi)], "n_nan": n_nan,
            "ci_reliable": n_nan <= MAX_NAN_SHARE * len(values)}


def summarize(y: np.ndarray, groups, oof: dict, *, n_boot: int, rng: np.random.Generator,
              mask: np.ndarray | None = None) -> dict:
    """Point estimates (mean over repeats) and patient-bootstrap CIs for every model and rule.

    Args:
        y: 0/1 target per encounter.
        groups: patient id per encounter.
        oof: output of readmission.evaluate.load_oof.
        mask: optional boolean row filter (e.g. one subgroup); weights outside it are zero.
    """
    y = np.asarray(y, dtype=float)
    keep = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    codes, n_pat = patient_codes(groups)
    R = oof["q_ref"].shape[0]

    rank_items = [("model", m, s) for m, s in oof["scores"].items()]
    rank_items += [("class_weighted", m, s) for m, s in oof["cw_scores"].items()]

    yk = y[keep]
    p_all = float(yk.mean())
    nir = max(p_all, 1 - p_all)  # fixed NIR of the evaluated rows (0.8856 for the cohort)

    def rank_metrics(W):
        Wk = W if mask is None else W[:, keep]  # copy once per batch, not per model
        # Brier of the reference forecast: the same for every model, so once per repeat
        brier_ref = [weighted_brier(Wk, yk, oof["q_ref"][r, keep]) for r in range(R)]
        res = {}
        for kind, m, s in rank_items:
            per_rep = {"roc_auc": [], "pr_auc": [], "brier": [], "brier_skill": []}
            for r in range(R):
                sk = s[r, keep].astype(float)
                per_rep["roc_auc"].append(weighted_auc(Wk, yk, sk))
                per_rep["pr_auc"].append(weighted_average_precision(Wk, yk, sk))
                b = weighted_brier(Wk, yk, sk)
                per_rep["brier"].append(b)
                per_rep["brier_skill"].append(1 - b / brier_ref[r])
            res[f"{kind}:{m}"] = {k: np.mean(v, axis=0) for k, v in per_rep.items()}
        return res

    def decision_metrics(W):
        Wk = W if mask is None else W[:, keep]
        res = {}
        for key, fl in oof["flags"].items():
            per_rep = [weighted_decision(Wk, yk, fl[r, keep], nir=nir) for r in range(R)]
            res[key] = {k: np.nanmean([d[k] for d in per_rep], axis=0) for k in per_rep[0]}
        return res

    ones = np.ones((1, len(y)), dtype=np.float32)
    point = {"rank": rank_metrics(ones), "decision": decision_metrics(ones)}
    boot_rank, boot_dec = [], []
    for W in bootstrap_weights(codes, n_pat, n_boot, rng):
        boot_rank.append(rank_metrics(W))
        boot_dec.append(decision_metrics(W))

    def collect(point_part, boot_parts):
        out = {}
        for item, metrics in point_part.items():
            out[item] = {}
            for name, value in metrics.items():
                draws = np.concatenate([bp[item][name] for bp in boot_parts])
                out[item][name] = {"estimate": float(value[0]), **_interval(draws)}
        return out

    result = {"n_encounters": int(keep.sum()), "n_repeats": R, "n_boot": n_boot,
              "prevalence": p_all, "nir": nir,
              "ranking": collect(point["rank"], boot_rank),
              "decision": collect(point["decision"], boot_dec)}
    for metrics in result["decision"].values():
        metrics["usable"] = metrics["acc_minus_nir"]["ci"][0] > 0
    return result
