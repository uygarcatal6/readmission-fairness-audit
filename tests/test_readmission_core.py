"""Tests for readmission/cohort.py, splits.py and preprocess.py (approved protocol)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission import cohort, preprocess, splits

# ── cohort ───────────────────────────────────────────────────────────────────


def test_age_band_uses_the_approved_bands():
    mid = pd.Series([15, 25, 35, 45, 55, 65, 75, 85, 95])
    got = cohort.age_band(mid).tolist()
    assert pd.isna(got[0])  # [10-20) is paediatric, excluded before this point
    assert got[1:] == ["20-39", "20-39", "40-59", "40-59", "60-79", "60-79", "80+", "80+"]


def test_feature_lists_cover_the_23_raw_attributes_once():
    assert len(cohort.FEATURES) == 23
    assert len(set(cohort.FEATURES)) == 23
    assert not set(cohort.NUMERIC) & set(cohort.CATEGORICAL)
    assert "y" not in cohort.FEATURES and "age_band" not in cohort.FEATURES
    assert not set(cohort.TEST_ORDERING) & set(cohort.FEATURES)


@pytest.mark.skipif(not cohort.DEFAULT_CSV.exists(), reason="UCI CSV not downloaded")
def test_real_cohort_matches_the_approved_numbers():
    df, rep = cohort.load_cohort(return_report=True)
    assert rep["n_encounters"] == 98_470
    assert rep["n_patients"] == 69_303
    assert round(rep["prevalence"], 3) == 0.114
    assert round(rep["no_information_rate"], 3) == 0.886
    assert set(df["sex"]) == {"Female", "Male"}
    assert set(df["age_band"]) == {"20-39", "40-59", "60-79", "80+"}
    assert df[cohort.FEATURES].isna().sum().sum() == 0

    first, rep1 = cohort.load_cohort(first_encounter_only=True, return_report=True)
    assert rep1["n_encounters"] == 69_303
    assert first[cohort.GROUP].is_unique


# ── splits ───────────────────────────────────────────────────────────────────


def _toy_patients(n_patients=400, seed=1):
    """1-3 encounters per patient, ~15 % positives."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, n_patients)
    groups = np.repeat(np.arange(n_patients), sizes)
    y = (rng.random(len(groups)) < 0.15).astype(int)
    return y, groups


def test_outer_splits_are_patient_disjoint_full_partitions():
    y, groups = _toy_patients()
    all_splits = list(splits.outer_splits(y, groups))
    assert len(all_splits) == 25
    for r in range(5):
        tests = [s.test for s in all_splits if s.repeat == r]
        assert np.array_equal(np.sort(np.concatenate(tests)), np.arange(len(y)))
    for s in all_splits:
        assert not set(groups[s.train]) & set(groups[s.test])


def test_outer_splits_keep_the_positive_rate():
    # Tight bound on purpose: sklearn's own shuffle=True breaks stratification (see the
    # splits.py docstring) and would fail this; plain GroupKFold would too.
    y, groups = _toy_patients(n_patients=2000)
    for s in splits.outer_splits(y, groups, n_repeats=3):
        assert abs(y[s.test].mean() - y.mean()) < 0.002


def test_repeats_use_different_partitions():
    y, groups = _toy_patients()
    first_test = {s.repeat: s.test for s in splits.outer_splits(y, groups) if s.fold == 0}
    assert not np.array_equal(np.sort(first_test[0]), np.sort(first_test[1]))


def test_inner_splits_stay_inside_the_training_fold():
    y, groups = _toy_patients()
    s = next(splits.outer_splits(y, groups))
    fit, val = splits.validation_split(y[s.train], groups[s.train])
    assert not set(groups[s.train][fit]) & set(groups[s.train][val])
    assert 0.1 < len(val) / len(s.train) < 0.3
    for fit, val in splits.inner_folds(y[s.train], groups[s.train]):
        assert set(s.train[val]) <= set(s.train)  # positions map back into train only
        assert not set(groups[s.train][fit]) & set(groups[s.train][val])


# ── preprocess ───────────────────────────────────────────────────────────────


def _toy_frame(n=1000, seed=2):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in cohort.NUMERIC})
    for c in cohort.CATEGORICAL:
        df[c] = rng.choice(["a", "b", "c"], size=n).astype(object)
    df["medical_specialty"] = np.where(np.arange(n) < 3, "rare", "common").astype(object)
    return df


def test_preprocessor_learns_only_from_training_rows():
    df = _toy_frame()
    pre = preprocess.make_preprocessor().fit(df)
    names = list(pre.get_feature_names_out())
    # 'rare' is 0.3 % of rows (< 0.5 %) -> pooled, not its own column
    assert "cat__medical_specialty_rare" not in names
    assert "cat__medical_specialty_infrequent_sklearn" in names

    test = df.head(5).copy()
    test["payer_code"] = "never_seen"  # payer_code has no infrequent level -> all zeros
    out = pre.transform(test)
    cols = [i for i, n in enumerate(names) if n.startswith("cat__payer_code_")]
    assert np.all(out[:, cols] == 0)


def test_preprocessor_accepts_subsets_and_rejects_unknown_names():
    df = _toy_frame()
    no_demo = [f for f in cohort.FEATURES if f not in cohort.DEMOGRAPHIC]
    names = preprocess.make_preprocessor(no_demo).fit(df).get_feature_names_out()
    assert not any(n.startswith(("num__age", "cat__sex_", "cat__race_")) for n in names)
    with pytest.raises(ValueError, match="not model features"):
        preprocess.make_preprocessor(["age", "y"])
