"""Tests for readmission/fairness.py (group metrics, insufficient groups, gaps, calibration)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission import cohort, fairness


def _toy(seed=0, n_patients=1500):
    """Cohort with a deliberately tiny 'Asian' group (< 50 positives) and a known TPR gap."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 3, n_patients)
    pid = np.repeat([f"P{i}" for i in range(n_patients)], sizes)
    n = len(pid)
    race = rng.choice(["Caucasian", "AfricanAmerican", "Asian", "Unknown"], size=n,
                      p=[0.5, 0.35, 0.03, 0.12])
    df = pd.DataFrame({
        cohort.GROUP: pid,
        "sex": rng.choice(["Female", "Male"], size=n),
        "age_band": rng.choice(["20-39", "40-59", "60-79", "80+"], size=n),
        "race": race,
    })
    y = (rng.random(n) < 0.3).astype(int)
    df[cohort.TARGET] = y
    R = 2
    flags = np.zeros((R, n), bool)
    scores = np.zeros((R, n), np.float32)
    for r in range(R):
        # AfricanAmerican positives are caught more often than Caucasian ones -> TPR gap
        # "Unknown" (missing race) is caught worst: it would dominate the gap if it counted
        p_hit = np.select([race == "AfricanAmerican", race == "Unknown"], [0.7, 0.1], 0.4)
        p_flag = np.where(y == 1, p_hit, 0.1)
        flags[r] = rng.random(n) < p_flag
        scores[r] = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.05, n), 0, 1)
    df["encounter_id"] = np.arange(n)
    oof = {"flags": {"m__topq": flags}, "scores": {"m": scores}, "cw_scores": {},
           "q_ref": np.full((R, n), 0.3)}
    return df, oof


def test_group_rates_match_direct_computation():
    df, oof = _toy()
    res = fairness.audit(df, oof, keys=["m__topq"], calibration_models=["m"], n_boot=30,
                         rng=np.random.default_rng(1))
    y = df[cohort.TARGET].to_numpy()
    fl = oof["flags"]["m__topq"]
    for race in ("Caucasian", "AfricanAmerican"):
        rows = (df["race"] == race).to_numpy()
        want = np.mean([fl[r, rows & (y == 1)].mean() for r in range(2)])
        got = res["groups"]["race"][race]["metrics"]["m__topq"]["tpr"]
        assert got["estimate"] == pytest.approx(want, abs=1e-6)
        assert got["ci"][0] <= got["estimate"] <= got["ci"][1]
    # base rate and group sizes are plain counts
    g = res["groups"]["sex"]["Female"]
    rows = (df["sex"] == "Female").to_numpy()
    assert g["n"] == rows.sum() and g["n_pos"] == y[rows].sum()
    assert g["base_rate"]["estimate"] == pytest.approx(y[rows].mean())


def test_small_group_is_marked_and_left_out_of_the_gap():
    df, oof = _toy()
    res = fairness.audit(df, oof, keys=["m__topq"], calibration_models=[], n_boot=20,
                         rng=np.random.default_rng(2))
    asian = res["groups"]["race"]["Asian"]
    assert asian["n_pos"] < 50
    assert asian["metrics"]["m__topq"]["tpr"]["insufficient"]  # reported, not dropped
    assert np.isfinite(asian["metrics"]["m__topq"]["tpr"]["estimate"])
    gap = res["gaps"]["m__topq"]["race"]
    assert "Asian" in gap["excluded_tpr"]
    tprs = {r: res["groups"]["race"][r]["metrics"]["m__topq"]["tpr"]["estimate"]
            for r in ("Caucasian", "AfricanAmerican")}
    fprs = {r: res["groups"]["race"][r]["metrics"]["m__topq"]["fpr"]["estimate"]
            for r in ("Caucasian", "AfricanAmerican", "Asian")}
    # gaps come from the repeat-averaged rates in the table, so they can be recomputed from it
    assert gap["tpr_gap"]["estimate"] == pytest.approx(tprs["AfricanAmerican"] - tprs["Caucasian"])
    assert gap["eo_diff"]["estimate"] == pytest.approx(
        max(gap["tpr_gap"]["estimate"], gap["fpr_gap"]["estimate"]))
    assert gap["tpr_gap"]["estimate"] > 0.2  # the planted gap (0.7 vs 0.4) is found
    assert all(np.isfinite(v) for v in fprs.values())
    # "Unknown" is tabled, not used in the gap; including it makes the gap larger
    assert "Unknown" in res["groups"]["race"] and gap["not_a_group"] == ["Unknown"]
    assert gap["eo_diff_incl_missing"]["estimate"] > gap["eo_diff"]["estimate"]
    # the signed pair difference can be tested against 0; here it clearly excludes it
    assert gap["pair_tpr"]["groups"] == ["AfricanAmerican", "Caucasian"]
    assert gap["pair_tpr"]["ci"][0] > 0


def test_calibration_in_the_large_and_choice_rules():
    df, oof = _toy()
    res = fairness.audit(df, oof, keys=["m__topq"], calibration_models=["m"], n_boot=20,
                         rng=np.random.default_rng(3))
    y = df[cohort.TARGET].to_numpy()
    rows = (df["age_band"] == "80+").to_numpy()
    c = res["calibration"]["m"]["age_band"]["80+"]
    want = y[rows].mean() - oof["scores"]["m"][:, rows].mean()
    assert c["citl"]["estimate"] == pytest.approx(want, abs=1e-5)
    assert 0 <= c["ece"] <= 1
    assert fairness.leading_attribute(res, "m__topq") == "race"  # the planted gap
    summary = {"ranking": {"model:a": {"roc_auc": {"estimate": 0.60}},
                           "model:b": {"roc_auc": {"estimate": 0.66}}}}
    assert fairness.choose_primary(summary, ("a", "b")) == "b"


def test_report_builds_from_an_audit_result():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_fairness.py"
    spec = importlib.util.spec_from_file_location("run_fairness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df, oof = _toy()
    res = fairness.audit(df, oof, keys=["m__topq"], calibration_models=["m"], n_boot=10,
                         rng=np.random.default_rng(4))
    text = mod.build_report(res, "m", "race", ("m",), "0.1 min")
    assert "Asian" in text and "‡" in text  # the small group is shown and marked
    assert "sex × age intersections" in text


def test_null_separates_planted_gap_from_noise_and_signed_sex_ci_covers_zero():
    df, oof = _toy()
    res = fairness.audit(df, oof, keys=["m__topq"], calibration_models=[], n_boot=100,
                         rng=np.random.default_rng(5), n_perm=100)
    race, sex = res["gaps"]["m__topq"]["race"], res["gaps"]["m__topq"]["sex"]
    # race carries a planted gap: far above its permutation null
    assert race["null"]["p_value"] < 0.05 and race["null"]["excess"] > 0.15
    # sex carries none: the signed Female - Male difference has a CI that covers 0,
    # while the (always non-negative) range cannot
    lo, hi = sex["pair_tpr"]["ci"]
    assert lo < 0 < hi and sex["tpr_gap"]["ci"][0] >= 0
    assert sex["null"]["p_value"] > 0.05
    assert fairness.leading_attribute(res, "m__topq") == "race"
