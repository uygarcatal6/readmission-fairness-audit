"""Tests for readmission/ordering.py (test-ordering logit, clustered SEs, BH family)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from readmission import cohort, ordering


def _cohort(n_patients=6000, seed=3, patient_effect=1.5):
    """Planted: race 'B' has odds ratio 2 for HbA1c; a strong patient effect makes a
    patient's encounters alike, so model-based SEs are too small on all encounters."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 5, n_patients)
    pid = np.repeat(np.arange(n_patients), sizes)
    n = len(pid)
    u = rng.normal(0, patient_effect, n_patients)[pid]
    # patient-level attributes, constant within patient
    sex = rng.choice(["Female", "Male"], n_patients, p=[0.55, 0.45])[pid]
    race = rng.choice(["A", "B", "Unknown"], n_patients, p=[0.7, 0.25, 0.05])[pid]
    df = pd.DataFrame({
        cohort.GROUP: pid.astype(str), "encounter_id": np.arange(n), "sex": sex, "race": race,
        "age_band": rng.choice(["40-59", "60-79", "80+"], n, p=[0.3, 0.5, 0.2]),
        "diag_1_group": rng.choice(["circulatory", "other"], n, p=[0.6, 0.4]),
        "admission_type": rng.choice(["emergency", "unknown"], n, p=[0.9, 0.1]),
    })
    logit = -2 + np.log(2.0) * (race == "B") + u
    df["hba1c_requested"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["glucose_requested"] = rng.random(n) < 0.1
    return df


def test_design_uses_the_largest_group_as_reference():
    df = _cohort(n_patients=500)
    refs = ordering.reference_levels(df)
    assert refs == {"sex": "Female", "age_band": "60-79", "race": "A", "diag_1_group": "circulatory"}
    X = ordering.design(df, refs)
    assert "race=A" not in X and "race=B" in X and "race=Unknown" in X
    assert np.array_equal(X["race=B"].to_numpy(), (df["race"] == "B").to_numpy().astype(float))


def test_planted_odds_ratio_clustered_ses_and_bh_family():
    df = _cohort()
    first = df.sort_values("encounter_id").drop_duplicates(cohort.GROUP)
    res = ordering.analyse(df, first, n_boot=20, rng=np.random.default_rng(0))
    t = res["all_encounters"]["hba1c_requested"]["terms"]["race=B"]
    # the patient effect attenuates the population-average OR towards 1, but it stays clear
    assert 1.3 < t["odds_ratio"] < 2.2 and t["ci"][0] > 1
    # repeat encounters of one patient are alike: clustering must widen the SE
    assert t["se_ratio"] > 1.2
    # one encounter per patient: clustered and model-based SEs agree
    t1 = res["first_encounters"]["hba1c_requested"]["terms"]["race=B"]
    assert t1["se_ratio"] == pytest.approx(1.0, abs=0.1)
    # BH over the tested terms of both outcomes; Unknown and diagnosis terms are not tested
    fits = list(res["all_encounters"].values())
    tested = [(f, k) for f in fits for k, v in f["terms"].items() if v["tested"]]
    assert {k for _, k in tested} == {"sex=Male", "age_band=40-59", "age_band=80+", "race=B"}
    want = multipletests([f["terms"][k]["p"] for f, k in tested], method="fdr_bh")[1]
    assert [f["terms"][k]["q"] for f, k in tested] == pytest.approx(list(want))
    for f in fits:
        assert f["terms"]["race=Unknown"]["q"] is None
        assert f["terms"]["diag_1_group=other"]["q"] is None
    # the replication is its own BH family and keeps the all-encounter references
    rep = list(res["first_encounters"].values())
    rtested = [(f, k) for f in rep for k, v in f["terms"].items() if v["tested"]]
    want = multipletests([f["terms"][k]["p"] for f, k in rtested], method="fdr_bh")[1]
    assert [f["terms"][k]["q"] for f, k in rtested] == pytest.approx(list(want))
    for f in rep:
        for v in f["terms"].values():
            if v["covariate"] in res["references"]:
                assert v["reference"] == res["references"][v["covariate"]]
    # the sensitivity fit adds the source indicator, which is never tested
    s = res["source_adjusted"]["hba1c_requested"]["terms"]
    assert s["admission_type=unknown"]["tested"] is False and s["admission_type=unknown"]["q"] is None
    assert set(s) == set(res["all_encounters"]["hba1c_requested"]["terms"]) | {"admission_type=unknown"}


def test_crude_rates_are_plain_group_means():
    df = _cohort(n_patients=800)
    crude = ordering.crude_rates(df, n_boot=20, rng=np.random.default_rng(1))
    for g in ("A", "B"):
        rows = df["race"] == g
        d = crude["hba1c_requested"]["race"][g]
        assert d["estimate"] == pytest.approx(df.loc[rows, "hba1c_requested"].mean())
        assert d["n"] == rows.sum() and d["ci"][0] <= d["estimate"] <= d["ci"][1]


def test_report_builds():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ordering.py"
    spec = importlib.util.spec_from_file_location("run_ordering", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = _cohort(n_patients=1500)
    first = df.sort_values("encounter_id").drop_duplicates(cohort.GROUP)
    res = ordering.analyse(df, first, n_boot=10, rng=np.random.default_rng(2))
    text = mod.build_report(res, "0.1 min")
    assert "B vs A" in text and "(missing code, not tested)" in text
    assert "Replication on first encounters" in text and "Replicated:" in text
    assert "result recorded" in text and "Where the recorded results come from" in text


def test_recording_pattern_is_computed_from_the_rows():
    # patients interleave in id order (as in the real data): a=1,3,5  b=2,4  c=6
    df = pd.DataFrame({
        cohort.GROUP: ["a", "b", "a", "b", "a", "c"],
        "encounter_id": [1, 2, 3, 4, 5, 6],
        "admission_type": ["unknown", "emergency", "unknown", "emergency", "emergency", "unknown"],
        "hba1c_requested": [True, False, True, True, False, False],
        "glucose_requested": [True, False, True, False, True, True],
    }).sample(frac=1, random_state=0)  # row order must not matter
    r = ordering.recording_pattern(df)["glucose_requested"]
    assert r["events"] == 4
    assert r["share_of_events_source"] == pytest.approx(3 / 4)  # rows 1, 2, 6 of 1, 2, 3, 6
    assert r["rate_source"] == pytest.approx(1.0) and r["rate_other"] == pytest.approx(1 / 3)
    # consecutive pairs within a patient: a1->a3 (T,T), a3->a5 (T,T), b2->b4 (F,F)
    assert r["n_pairs"] == 3
    assert r["next_if_now"] == pytest.approx(1.0) and r["next_if_not_now"] == pytest.approx(0.0)
