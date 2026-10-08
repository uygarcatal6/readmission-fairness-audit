"""
readmission/fairness.py — the fairness audit of the approved protocol.

docs/APPROVED_PROPOSAL.md: "By sex, age band (20–39, 40–59, 60–79, 80+) and race, I report
TPR, FPR, PPV, selection rate, the equalized-odds difference and sex × age intersections.
Groups with fewer than 50 positives (for TPR) or 50 negatives (for FPR) are reported as
insufficient, not dropped. [...] base rates and calibration by group."

How the numbers are made
    * Every repeat tests every encounter once, so group rates are computed on the pooled
      out-of-fold flags of a repeat and averaged over the 5 repeats (a single fold would leave
      small groups, e.g. Asian with 7–22 positives, far below 50).
    * 95 % CIs: the same patient bootstrap as the main results (readmission/metrics.py); one
      draw re-weights all groups and repeats at once.
    * Insufficient: TPR if the group has < 50 positives, FPR if < 50 negatives, PPV if < 50
      flagged encounters (PPV rule: our addition). Reported and marked, left out of the gaps.
    * Race "Unknown" is the dataset's missing-race code, not a racial group: it stays in the
      tables, is left out of the race gaps, and a gap including it is reported as secondary.
    * Gaps are computed from the repeat-averaged group rates shown in the tables:
        tpr_gap = max − min TPR over usable groups, fpr_gap likewise,
        eo_diff = max(tpr_gap, fpr_gap)  (fairlearn's equalized-odds difference).
      A max − min range is never below 0, so its percentile CI cannot show "no gap". Two
      additions make the gaps testable:
        pair_*  the signed difference between the two groups that are highest and lowest at
                the point estimate, bootstrapped with that pair fixed (its CI can cover 0);
                for sex this is simply Female − Male.
        null    a permutation reference: patient-level shuffles of the attribute (same flags,
                same group sizes) give the range expected with NO disparity; reported as its
                mean and 95th percentile with a permutation p-value.
    * Calibration by group: calibration-in-the-large = observed rate − mean predicted risk
      (with CI) and the expected calibration error over 10 equal-count bins, next to the ECE a
      perfectly calibrated model would show in a group of that size (small groups have a
      large ECE from binning noise alone).

Choices written down before the results
    * primary model = the approved model with the highest ROC-AUC point estimate.
    * leading attribute (amended 23 Sep 2026, after an independent code review and before the
      full results): among sex, age band and race, the largest equalized-odds difference IN
      EXCESS OF its permutation null mean. The first version compared raw ranges, which favours
      attributes with many small groups (race's null alone was ~0.13 in the review's check).
      The raw-range choice is still reported (leading_attribute_raw). In the reviewers'
      repeat-0 check age band led under both rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from readmission import cohort, metrics

MIN_POSITIVES = 50
MIN_NEGATIVES = 50
MIN_FLAGGED = 50
ATTRIBUTES = ("sex", "age_band", "race", "sex_x_age")
MAIN_ATTRIBUTES = ("sex", "age_band", "race")
NOT_A_GROUP = {"race": {"Unknown"}}  # missing-value codes: tabled, not used in gaps
ECE_BINS = 10
N_PERMUTATIONS = 200
N_ECE_SIMULATIONS = 50


def group_masks(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Boolean row masks per attribute and group (sex × age as its own attribute)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for attr in MAIN_ATTRIBUTES:
        col = df[attr].astype(str)
        out[attr] = {g: (col == g).to_numpy() for g in sorted(col.unique())}
    cell = df["sex"].astype(str) + " " + df["age_band"].astype(str)
    out["sex_x_age"] = {g: (cell == g).to_numpy() for g in sorted(cell.unique())}
    return out


def _flat(masks: dict, attributes=ATTRIBUTES) -> tuple[list[tuple[str, str]], np.ndarray]:
    """(attribute, group) pairs and a (G, n) float32 mask matrix in the same order."""
    pairs = [(a, g) for a in attributes for g in masks[a]]
    M = np.stack([masks[a][g] for a, g in pairs]).astype(np.float32)
    return pairs, M


def _rates(W: np.ndarray, M: np.ndarray, y: np.ndarray, flags: np.ndarray) -> dict:
    """Group rates per draw (rows of W), averaged over repeats (rows of flags)."""
    pos = W @ (M * y).T
    tot = W @ M.T
    neg = tot - pos
    acc = {"tpr": [], "fpr": [], "ppv": [], "selection_rate": [], "flagged": []}
    with np.errstate(invalid="ignore", divide="ignore"):
        for f in flags:
            # only the flagged encounters count: their weight per group, split by label (the
            # sums are of integer weights below 2**24, so float32 is exact either way)
            rows = np.flatnonzero(f)
            Wf, Mf = W[:, rows], M[:, rows]
            tp = Wf @ (Mf * y[rows]).T
            fp = Wf @ Mf.T - tp
            acc["tpr"].append(tp / pos)
            acc["fpr"].append(fp / neg)
            acc["ppv"].append(tp / (tp + fp))
            acc["selection_rate"].append((tp + fp) / tot)
            acc["flagged"].append(tp + fp)
    return {k: np.nanmean(v, axis=0) for k, v in acc.items()}


def _range(values: np.ndarray, idx: list[int]) -> np.ndarray:
    if len(idx) < 2:
        return np.full(values.shape[0], np.nan)
    sub = values[:, idx]
    return np.nanmax(sub, axis=1) - np.nanmin(sub, axis=1)


def _usable(pairs, sufficient, attr, metric, *, include_missing=False) -> list[int]:
    skip = set() if include_missing else NOT_A_GROUP.get(attr, set())
    return [i for i, (a, g) in enumerate(pairs)
            if a == attr and sufficient[metric][i] and g not in skip]


def _calibration(W: np.ndarray, M: np.ndarray, y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Observed rate − mean predicted risk per draw and group, averaged over repeats."""
    tot = W @ M.T
    obs = W @ (M * y).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean([(obs - W @ (M * s).T) / tot for s in scores], axis=0)


def _ece(y: np.ndarray, score: np.ndarray, order: np.ndarray | None = None,
         bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-count score bins (one group, unweighted)."""
    order = np.argsort(score, kind="mergesort") if order is None else order
    parts = np.array_split(order, bins)
    return float(sum(len(p) * abs(y[p].mean() - score[p].mean()) for p in parts if len(p))
                 / len(y))


def _patient_labels(df: pd.DataFrame, attr: str, codes: np.ndarray, n_pat: int) -> np.ndarray:
    """One label per patient (their first encounter's value) for patient-level shuffles."""
    col = (df["sex"].astype(str) + " " + df["age_band"].astype(str) if attr == "sex_x_age"
           else df[attr].astype(str)).to_numpy()
    order = np.argsort(df["encounter_id"].to_numpy()) if "encounter_id" in df else np.arange(len(df))
    _, first = np.unique(codes[order], return_index=True)  # first encounter of every patient
    return col[order[first]]


def _null_ranges(labels, y, flags, attr, codes, n_pat, n_perm, rng) -> np.ndarray:
    """EO difference under patient-level permutations of `attr` (same flags, no disparity).

    A shuffle moves whole patients between groups, so every group count is a sum of
    per-patient counts: positives, encounters, and true / false positives per repeat. Those
    are summed once per patient and re-summed by the shuffled label in every permutation
    (np.bincount): the same integers the mask matrix of _rates gives, without building that
    matrix n_perm times. The rates are then formed exactly as in _rates (float32).
    """
    groups, gid = np.unique(labels, return_inverse=True)  # group index per patient
    pairs = [(attr, g) for g in groups]

    def per_patient(v):
        return np.bincount(codes, weights=v, minlength=n_pat)

    def by_group(g, v):  # group totals when patient p is in group g[p]; one float32 row as in _rates
        return np.bincount(g, weights=v, minlength=len(groups)).astype(np.float32)[None]

    pat_pos, pat_all = per_patient(y), np.bincount(codes, minlength=n_pat)
    pat_tp = [per_patient(y * f) for f in flags]
    pat_fp = [per_patient(f - y * f) for f in flags]
    out = np.empty(n_perm)
    for b in range(n_perm):
        g = gid[rng.permutation(n_pat)]
        pos, tot = by_group(g, pat_pos), by_group(g, pat_all)
        neg = tot - pos
        with np.errstate(invalid="ignore", divide="ignore"):
            tpr = np.nanmean([by_group(g, v) / pos for v in pat_tp], axis=0)
            fpr = np.nanmean([by_group(g, v) / neg for v in pat_fp], axis=0)
        suff = {"tpr": pos[0] >= MIN_POSITIVES, "fpr": neg[0] >= MIN_NEGATIVES}
        t = _range(tpr, _usable(pairs, suff, attr, "tpr"))
        f = _range(fpr, _usable(pairs, suff, attr, "fpr"))
        out[b] = np.fmax(t, f)[0]
    return out


def audit(df: pd.DataFrame, oof: dict, *, keys: list[str], calibration_models: list[str],
          n_boot: int, rng: np.random.Generator, null_keys: list[str] | None = None,
          n_perm: int = N_PERMUTATIONS) -> dict:
    """Group metrics, gaps (range, fixed-pair signed, permutation null) and calibration.

    Args:
        df: cohort frame (target, patient id, sensitive columns, encounter_id).
        oof: readmission.evaluate.load_oof output.
        keys: flag keys "model__rule" to audit.
        calibration_models: models whose scores get calibration by group.
        null_keys: keys that get the permutation null (default: the "__topq" keys).
    """
    y = df[cohort.TARGET].to_numpy().astype(np.float32)
    masks = group_masks(df)
    pairs, M = _flat(masks)
    R = oof["q_ref"].shape[0]
    codes, n_pat = metrics.patient_codes(df[cohort.GROUP])
    null_keys = [k for k in keys if k.endswith("__topq")] if null_keys is None else null_keys
    flags = {k: oof["flags"][k].astype(np.float32) for k in keys}

    n_pos = (M * y).sum(axis=1)
    n_all = M.sum(axis=1)
    n_neg = n_all - n_pos

    ones = np.ones((1, len(y)), dtype=np.float32)
    point = {k: _rates(ones, M, y, flags[k]) for k in keys}
    sufficient = {k: {"tpr": n_pos >= MIN_POSITIVES, "fpr": n_neg >= MIN_NEGATIVES,
                      "ppv": point[k]["flagged"][0] >= MIN_FLAGGED} for k in keys}

    # fixed pairs (highest and lowest group at the point estimate) for the signed differences
    fixed = {}
    for k in keys:
        fixed[k] = {}
        for a in ATTRIBUTES:
            for m in ("tpr", "fpr"):
                idx = _usable(pairs, sufficient[k], a, m)
                if len(idx) >= 2:
                    vals = point[k][m][0, idx]
                    fixed[k][(a, m)] = (idx[int(np.nanargmax(vals))], idx[int(np.nanargmin(vals))])

    def gap_stats(rates: dict, k: str) -> dict:
        out = {}
        for a in ATTRIBUTES:
            t = _range(rates["tpr"], _usable(pairs, sufficient[k], a, "tpr"))
            f = _range(rates["fpr"], _usable(pairs, sufficient[k], a, "fpr"))
            d = {"tpr_gap": t, "fpr_gap": f, "eo_diff": np.fmax(t, f)}
            for m in ("tpr", "fpr"):
                if (a, m) in fixed[k]:
                    hi, lo = fixed[k][(a, m)]
                    d[f"pair_{m}"] = rates[m][:, hi] - rates[m][:, lo]
            if a in NOT_A_GROUP:
                ti = _range(rates["tpr"], _usable(pairs, sufficient[k], a, "tpr", include_missing=True))
                fi = _range(rates["fpr"], _usable(pairs, sufficient[k], a, "fpr", include_missing=True))
                d["eo_diff_incl_missing"] = np.fmax(ti, fi)
            out[a] = d
        return out

    point_gaps = {k: gap_stats(point[k], k) for k in keys}
    point_cal = {m: _calibration(ones, M, y, oof["scores"][m].astype(np.float32))
                 for m in calibration_models}
    point_base = (ones @ (M * y).T) / (ones @ M.T)

    draws = {"rates": {k: {m: [] for m in ("tpr", "fpr", "ppv", "selection_rate")} for k in keys},
             "gaps": {k: {a: {} for a in ATTRIBUTES} for k in keys},
             "cal": {m: [] for m in calibration_models}, "base": []}
    for W in metrics.bootstrap_weights(codes, n_pat, n_boot, rng):
        for k in keys:
            r = _rates(W, M, y, flags[k])
            for m in draws["rates"][k]:
                draws["rates"][k][m].append(r[m])
            for a, d in gap_stats(r, k).items():
                for name, v in d.items():
                    draws["gaps"][k][a].setdefault(name, []).append(v)
        for m in calibration_models:
            draws["cal"][m].append(_calibration(W, M, y, oof["scores"][m].astype(np.float32)))
        draws["base"].append((W @ (M * y).T) / (W @ M.T))

    def est_ci(estimate: float, parts: list) -> dict:
        return {"estimate": float(estimate), **metrics._interval(np.concatenate(parts))}

    out = {"min_positives": MIN_POSITIVES, "min_negatives": MIN_NEGATIVES,
           "min_flagged": MIN_FLAGGED, "not_a_group": {a: sorted(v) for a, v in NOT_A_GROUP.items()},
           "n_boot": n_boot, "n_repeats": R, "n_permutations": n_perm,
           "groups": {}, "gaps": {}, "calibration": {}}
    base_draws = np.concatenate(draws["base"])
    for i, (a, g) in enumerate(pairs):
        out["groups"].setdefault(a, {})[g] = {
            "n": int(n_all[i]), "n_pos": int(n_pos[i]), "n_neg": int(n_neg[i]),
            "base_rate": {"estimate": float(point_base[0, i]), **metrics._interval(base_draws[:, i])},
            "metrics": {}}
    for k in keys:
        for m, parts in draws["rates"][k].items():
            allm = np.concatenate(parts)
            for i, (a, g) in enumerate(pairs):
                insufficient = m in ("tpr", "fpr", "ppv") and not sufficient[k][m][i]
                out["groups"][a][g]["metrics"].setdefault(k, {})[m] = {
                    "estimate": float(point[k][m][0, i]), **metrics._interval(allm[:, i]),
                    "insufficient": bool(insufficient)}
        out["gaps"][k] = {}
        for a in ATTRIBUTES:
            g_out = {name: est_ci(point_gaps[k][a][name][0], draws["gaps"][k][a][name])
                     for name in point_gaps[k][a]}
            for m in ("tpr", "fpr"):
                if (a, m) in fixed[k]:
                    hi, lo = fixed[k][(a, m)]
                    g_out[f"pair_{m}"]["groups"] = [pairs[hi][1], pairs[lo][1]]
            g_out["excluded_tpr"] = [g for i, (aa, g) in enumerate(pairs)
                                     if aa == a and not sufficient[k]["tpr"][i]]
            g_out["excluded_fpr"] = [g for i, (aa, g) in enumerate(pairs)
                                     if aa == a and not sufficient[k]["fpr"][i]]
            g_out["not_a_group"] = sorted(NOT_A_GROUP.get(a, set()))
            out["gaps"][k][a] = g_out

    # permutation null of the EO difference (patient-level shuffles)
    labels = {a: _patient_labels(df, a, codes, n_pat) for a in ATTRIBUTES}
    for k in null_keys:
        for a in ATTRIBUTES:
            null = _null_ranges(labels[a], y, flags[k], a, codes, n_pat, n_perm, rng)
            obs = out["gaps"][k][a]["eo_diff"]["estimate"]
            out["gaps"][k][a]["null"] = {
                "mean": float(np.nanmean(null)), "p95": float(np.nanpercentile(null, 95)),
                "excess": float(obs - np.nanmean(null)),
                "p_value": float((1 + np.sum(null >= obs)) / (1 + np.isfinite(null).sum()))}

    for m in calibration_models:
        cd = np.concatenate(draws["cal"][m])
        s_all = oof["scores"][m]
        out["calibration"][m] = {}
        sim_rng = np.random.default_rng([len(y), R])
        for i, (a, g) in enumerate(pairs):
            rows = masks[a][g]
            yg = y[rows]
            ece, ece_null = [], []
            for r in range(R):
                sg = s_all[r, rows].astype(float)
                order = np.argsort(sg, kind="mergesort")
                ece.append(_ece(yg, sg, order))
                ece_null.append(np.mean([_ece((sim_rng.random(len(sg)) < sg).astype(float), sg, order)
                                         for _ in range(N_ECE_SIMULATIONS)]))
            out["calibration"][m].setdefault(a, {})[g] = {
                "citl": {"estimate": float(point_cal[m][0, i]), **metrics._interval(cd[:, i])},
                "mean_predicted": float(np.mean([s_all[r, rows].mean() for r in range(R)])),
                "observed": float(yg.mean()), "ece": float(np.mean(ece)),
                "ece_if_perfectly_calibrated": float(np.mean(ece_null))}
    return out


def choose_primary(summary: dict, approved: tuple[str, ...]) -> str:
    """Pre-registered: the approved model with the highest ROC-AUC point estimate."""
    aucs = {m: summary["ranking"][f"model:{m}"]["roc_auc"]["estimate"] for m in approved}
    return max(aucs, key=aucs.get)


def leading_attribute(result: dict, key: str, *, use_null: bool = True) -> str | None:
    """Largest EO difference above its permutation null (use_null=False: raw range, v1)."""
    vals = {}
    for a in MAIN_ATTRIBUTES:
        g = result["gaps"][key][a]
        v = g["null"]["excess"] if use_null and "null" in g else g["eo_diff"]["estimate"]
        if np.isfinite(v):
            vals[a] = v
    return max(vals, key=vals.get) if vals else None
