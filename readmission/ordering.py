"""
readmission/ordering.py — are HbA1c and glucose results recorded equally across groups?

docs/APPROVED_PROPOSAL.md: "Test ordering: Whether HbA1c and serum glucose were ordered is
modelled on sex, age band, race and primary-diagnosis group. The models use patient-clustered
standard errors and Benjamini–Hochberg correction, and are replicated on first encounters only."

This is an inferential model on the whole cohort, not a prediction task: no folds.

What the outcome is: the dataset holds a result range for the test ("A1Cresult",
"max_glu_serum") or "None". A result recorded during the encounter is the data's proxy for "the
test was ordered"; the approved wording says "ordered", the report says "result recorded". The
odds ratios are adjusted for primary-diagnosis group only, so they describe who has a recorded
result, not whether testing matched clinical need (no evidence of unequal care on their own).

    outcomes    hba1c_requested ("A1Cresult" not "None"), glucose_requested ("max_glu_serum"
                not "None") — readmission/cohort.py.
    model       logistic regression (statsmodels Logit, maximum likelihood) on indicator
                columns for sex, age band, race and primary-diagnosis group. Reference level of
                each = its largest group (Female, 60-79, Caucasian, circulatory), which gives
                the most precise contrasts; odds ratios compare each group with it, holding
                the other three variables fixed.
    SEs         cluster-robust by patient (a patient's encounters are not independent). The
                ordinary (model-based) SEs are kept for comparison.
    family      Benjamini–Hochberg over every sensitive-attribute coefficient of both outcomes
                (sex, age band and race terms; the diagnosis terms are adjustment, not tests).
                Race "Unknown" (missing code) is a level of the model so its rows stay in, but
                its coefficient is reported apart and left out of the family.
    replication the same models on each patient's first encounter only (one row per patient).
    crude rates recording rate per group with patient-bootstrap CIs.
    recording   where the results come from (added 23 Sep 2026 after an independent review):
                share of results in encounters with unknown admission type (the source's
                "not available / not mapped" codes), rates by encounter_id decile (the ids
                follow the order of the extraction), and persistence within a patient. For
                glucose these show a recording pattern of some sources / periods rather than a
                per-admission decision.
    sensitivity the approved model plus one indicator "admission type unknown" (not tested, not
                in any BH family): if a group's odds ratio moves towards 1, part of it came from
                where and when the field was filled in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from readmission import cohort, fairness, metrics

OUTCOMES = {"hba1c_requested": "HbA1c result recorded",
            "glucose_requested": "serum glucose result recorded"}
SOURCE_FLAG = ("admission_type", "unknown")  # the sensitivity indicator: column, value
TESTED = fairness.MAIN_ATTRIBUTES  # sex, age_band, race
COVARIATES = (*TESTED, "diag_1_group")


def reference_levels(df: pd.DataFrame) -> dict[str, str]:
    """Largest group of every covariate (ties broken by name, so the choice is stable)."""
    out = {}
    for c in COVARIATES:
        counts = df[c].astype(str).value_counts()
        out[c] = sorted(counts[counts == counts.max()].index)[0]
    return out


def design(df: pd.DataFrame, refs: dict[str, str], *, source_flag: bool = False) -> pd.DataFrame:
    """Intercept + one indicator per non-reference level, named 'covariate=level' (+ the
    sensitivity indicator 'admission_type=unknown' if source_flag)."""
    cols = {"const": np.ones(len(df))}
    for c in COVARIATES:
        values = df[c].astype(str).to_numpy()
        for level in sorted(set(values) - {refs[c]}):
            cols[f"{c}={level}"] = (values == level).astype(float)
    if source_flag:
        col, value = SOURCE_FLAG
        cols[f"{col}={value}"] = (df[col].astype(str) == value).to_numpy().astype(float)
    return pd.DataFrame(cols, index=df.index)


def fit(df: pd.DataFrame, outcome: str, refs: dict[str, str], *, source_flag: bool = False) -> dict:
    """Logit with cluster-robust (patient) and model-based SEs; odds ratios per term."""
    X = design(df, refs, source_flag=source_flag)
    y = df[outcome].astype(float).to_numpy()
    codes, _ = metrics.patient_codes(df[cohort.GROUP])
    model = sm.Logit(y, X)
    naive = model.fit(disp=0, maxiter=200)
    robust = model.fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": codes})
    if not (naive.mle_retvals["converged"] and robust.mle_retvals["converged"]):
        raise RuntimeError(f"{outcome}: logistic regression did not converge")
    ci = robust.conf_int()
    terms = {}
    for name in X.columns:
        if name == "const":
            continue
        covariate, level = name.split("=", 1)
        terms[name] = {
            "covariate": covariate, "level": level, "reference": refs.get(covariate, "other types"),
            "odds_ratio": float(np.exp(robust.params[name])),
            "ci": [float(np.exp(ci.loc[name, 0])), float(np.exp(ci.loc[name, 1]))],
            "p": float(robust.pvalues[name]),
            "p_model_based": float(naive.pvalues[name]),
            "se_ratio": float(robust.bse[name] / naive.bse[name]),
            "n_level": int((X[name] == 1).sum()),
            "tested": covariate in TESTED and level not in fairness.NOT_A_GROUP.get(covariate, set()),
            "q": None,  # set by correct() for the tested terms of a family
        }
    return {"outcome": outcome, "n": int(len(y)), "n_patients": int(codes.max() + 1),
            "rate": float(y.mean()), "events": int(y.sum()), "terms": terms}


def recording_pattern(df: pd.DataFrame) -> dict:
    """Where the recorded results come from, per outcome (all encounters)."""
    col, value = SOURCE_FLAG
    src = (df[col].astype(str) == value).to_numpy()
    ids = df["encounter_id"].to_numpy()
    decile = np.empty(len(df), int)
    decile[np.argsort(ids, kind="stable")] = np.arange(len(df)) * 10 // len(df)
    pid_all = df[cohort.GROUP].astype(str).to_numpy()
    by_patient = np.lexsort((ids, pid_all))  # each patient's encounters together, in id order
    pid = pid_all[by_patient]
    same = pid[1:] == pid[:-1]  # consecutive encounters of one patient
    out = {}
    for o in OUTCOMES:
        y = df[o].to_numpy().astype(bool)
        ys = y[by_patient]
        now, nxt = ys[:-1][same], ys[1:][same]
        out[o] = {
            "events": int(y.sum()), "share_of_encounters_source": float(src.mean()),
            "share_of_events_source": float(y[src].sum() / y.sum()),
            "rate_source": float(y[src].mean()), "rate_other": float(y[~src].mean()),
            "rate_by_decile": [float(y[decile == d].mean()) for d in range(10)],
            "next_if_now": float(nxt[now].mean()), "next_if_not_now": float(nxt[~now].mean()),
            "n_pairs": int(same.sum())}
    return out


def correct(fits: list[dict]) -> None:
    """Benjamini–Hochberg q-values over the tested terms of all fits (written in place)."""
    tested = [(f, name) for f in fits for name, t in f["terms"].items() if t["tested"]]
    q = multipletests([f["terms"][name]["p"] for f, name in tested], method="fdr_bh")[1]
    for (f, name), qv in zip(tested, q, strict=True):
        f["terms"][name]["q"] = float(qv)


def crude_rates(df: pd.DataFrame, *, n_boot: int, rng: np.random.Generator) -> dict:
    """Ordering rate per group and outcome with patient-bootstrap 95 % CIs."""
    codes, n_pat = metrics.patient_codes(df[cohort.GROUP])
    pairs = [(c, g) for c in COVARIATES for g in sorted(df[c].astype(str).unique())]
    M = np.stack([(df[c].astype(str) == g).to_numpy() for c, g in pairs]).astype(np.float32)
    Y = {o: df[o].to_numpy().astype(np.float32) for o in OUTCOMES}

    def rates(W):
        tot = W @ M.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return {o: (W @ (M * y).T) / tot for o, y in Y.items()}

    point = rates(np.ones((1, len(df)), dtype=np.float32))
    draws = [rates(W) for W in metrics.bootstrap_weights(codes, n_pat, n_boot, rng)]
    out = {}
    for o in OUTCOMES:
        allr = np.concatenate([d[o] for d in draws])
        for i, (c, g) in enumerate(pairs):
            out.setdefault(o, {}).setdefault(c, {})[g] = {
                "n": int(M[i].sum()), "estimate": float(point[o][0, i]),
                **metrics._interval(allr[:, i])}
    return out


def analyse(df: pd.DataFrame, first: pd.DataFrame, *, n_boot: int,
            rng: np.random.Generator) -> dict:
    """All encounters (clustered) and first encounters (one per patient), both BH-corrected
    within their own family, plus crude rates on all encounters."""
    refs = reference_levels(df)
    main = [fit(df, o, refs) for o in OUTCOMES]
    rep = [fit(first, o, refs) for o in OUTCOMES]
    correct(main)
    correct(rep)
    sens = {o: fit(df, o, refs, source_flag=True) for o in OUTCOMES}  # not corrected: not tests
    return {"references": refs, "all_encounters": {f["outcome"]: f for f in main},
            "first_encounters": {f["outcome"]: f for f in rep}, "source_adjusted": sens,
            "recording": recording_pattern(df),
            "crude": crude_rates(df, n_boot=n_boot, rng=rng), "n_boot": n_boot}
