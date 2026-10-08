"""adapters/diabetes_uci: column mapping and row exclusions of the UCI loader.

All tests run on a small synthetic raw CSV shaped like the UCI file (50 columns); no real rows
are stored in the repository. The real-data cohort counts are checked in
tests/test_readmission_core.py when data/raw/diabetic_data.csv is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adapters import diabetes_uci as du

RAW_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
    "readmitted",
]
AGE_BANDS = ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)",
             "[90-100)"]
DIAG_CODES = ["250.83", "250", "428", "414", "786", "V57", "800", "715", "584", "162",
              "577", "276", "038", "401", "486"]
DRUG_LEVELS = ["No", "Steady", "Up", "Down"]


def make_raw(n_patients: int = 400, seed: int = 0) -> pd.DataFrame:
    """UCI-shaped synthetic raw table with injected edge cases (see comments)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    enc = 10_000
    for p in range(n_patients):
        n_enc = 1 + (rng.random() < 0.3) * int(rng.integers(1, 3))
        gender = rng.choice(["Female", "Male"], p=[0.55, 0.45])
        race = rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Other", "Asian", "?"],
                          p=[0.70, 0.20, 0.04, 0.03, 0.01, 0.02])
        age = rng.choice(AGE_BANDS)
        for _ in range(int(n_enc)):
            enc += int(rng.integers(1, 5))
            n_inpat = int(rng.integers(0, 4))
            p_lt30 = 0.06 + 0.06 * n_inpat
            readmitted = rng.choice(["<30", ">30", "NO"], p=[p_lt30, 0.35, 0.65 - p_lt30])
            rows.append({
                "encounter_id": enc, "patient_nbr": 100_000 + p, "race": race,
                "gender": gender, "age": age, "weight": "?",
                "admission_type_id": rng.choice([1, 2, 3, 5, 6]),
                "discharge_disposition_id": rng.choice([1, 3, 6, 18, 2, 22]),
                "admission_source_id": rng.choice([7, 1, 17, 4]),
                "time_in_hospital": int(rng.integers(1, 15)),
                "payer_code": rng.choice(["?", "MC", "HM", "SP"]),
                "medical_specialty": rng.choice(
                    ["?", "InternalMedicine", "Emergency/Trauma", "Cardiology", "Surgery-General"]),
                "num_lab_procedures": int(rng.integers(1, 100)),
                "num_procedures": int(rng.integers(0, 7)),
                "num_medications": int(rng.integers(1, 40)),
                "number_outpatient": int(rng.integers(0, 4)),
                "number_emergency": int(rng.integers(0, 3)),
                "number_inpatient": n_inpat,
                "diag_1": rng.choice(DIAG_CODES), "diag_2": rng.choice(DIAG_CODES),
                "diag_3": rng.choice(DIAG_CODES + ["?"]),
                "number_diagnoses": int(rng.integers(1, 10)),
                "max_glu_serum": rng.choice(["None", ">200", ">300", "Norm"],
                                            p=[0.9, 0.03, 0.03, 0.04]),
                "A1Cresult": rng.choice(["None", ">7", ">8", "Norm"], p=[0.8, 0.05, 0.1, 0.05]),
                **{d: rng.choice(DRUG_LEVELS, p=[0.7, 0.2, 0.05, 0.05]) for d in RAW_COLUMNS[24:47]},
                "change": rng.choice(["Ch", "No"]), "diabetesMed": rng.choice(["Yes", "No"], p=[0.75, 0.25]),
                "readmitted": readmitted,
            })
    df = pd.DataFrame(rows)[RAW_COLUMNS]
    # injected edge cases (row positions are fixed so tests can count them)
    df.loc[0, "gender"] = "Unknown/Invalid"                     # 1 unknown gender
    df.loc[[1, 2, 3], "discharge_disposition_id"] = 11           # 3 expired
    df.loc[4, "discharge_disposition_id"] = 13                   # 1 hospice
    df.loc[[5, 6], "diag_1"] = "?"                               # 2 missing primary dx
    df.loc[[7, 8, 9], "age"] = "[10-20)"                         # 3 minors (dropped later by readmission/cohort.py)
    df.loc[10, "age"] = "[0-10)"                                 # 1 minor
    return df


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("uci") / "diabetic_data_synthetic.csv"
    make_raw().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def adapted(raw_csv):
    return du.load(raw_csv, level="encounter", return_report=True)


# ── pure mapping helpers ─────────────────────────────────────────────────────


def test_age_band_midpoint():
    assert du.age_band_midpoint("[0-10)") == 5
    assert du.age_band_midpoint("[70-80)") == 75
    assert du.age_band_midpoint("[90-100)") == 95
    assert du.age_band_midpoint("garbage") is None


@pytest.mark.parametrize("code,group", [
    ("250.83", "diabetes"), ("250", "diabetes"), ("428", "circulatory"), ("785", "circulatory"),
    ("486", "respiratory"), ("786", "respiratory"), ("577", "digestive"), ("787", "digestive"),
    ("800", "injury"), ("999", "injury"), ("715", "musculoskeletal"), ("584", "genitourinary"),
    ("788", "genitourinary"), ("162", "neoplasms"), ("V57", "other"), ("E909", "other"),
    ("276", "other"), ("?", "other"), (None, "other"),
])
def test_diag1_group(code, group):
    assert du.diag1_group(code) == group


# ── output columns and row exclusions ────────────────────────────────────────


def test_sex_mapping_and_unknown_dropped(adapted):
    df, report = adapted
    assert set(df["sex"].unique()) == {"K", "E"}  # readmission/cohort.py maps K/E -> Female/Male
    assert report["excluded_gender_unknown"] == 1


def test_readmission_label_and_target_agree(adapted):
    df, _ = adapted
    assert set(df["clinical_diagnosis"].unique()) == {"readmit_lt30", "readmit_gt30", "no_readmit"}
    assert (df["readmission_30d"] == (df["clinical_diagnosis"] == "readmit_lt30")).all()


def test_exclusions_counted(adapted):
    df, report = adapted
    assert report["excluded_death_hospice"] == 4
    assert report["excluded_diag1_missing"] == 2
    assert report["n_adapted"] == report["n_raw"] - 1 - 4 - 2
    assert report["n_adapted"] == len(df)
    assert not df["discharge_group"].isna().any()


def test_requested_flags_match_raw(raw_csv, adapted):
    df, _ = adapted
    raw = pd.read_csv(raw_csv, keep_default_na=False, dtype=str)
    raw = raw.set_index(raw["encounter_id"].astype(int))
    sub = raw.loc[df["encounter_id"].to_numpy()]
    assert (df["hba1c_requested"].to_numpy() == (sub["A1Cresult"] != "None").to_numpy()).all()
    assert (df["glucose_requested"].to_numpy() == (sub["max_glu_serum"] != "None").to_numpy()).all()
    assert set(df["a1c_result"].unique()) <= {"not_measured", "gt7", "gt8", "norm"}
    assert set(df["max_glu_serum"].unique()) <= {"not_measured", "gt200", "gt300", "norm"}
    assert (df["prescription_recorded"].to_numpy() == (sub["diabetesMed"] == "Yes").to_numpy()).all()


def test_race_unknown_kept_as_subgroup(adapted):
    df, _ = adapted
    assert "Unknown" in set(df["race"].unique())
    assert "?" not in set(df["race"].unique())


def test_first_visit_date_is_ordering_only(adapted):
    df, _ = adapted
    d = df.sort_values("encounter_id")
    assert d["first_visit_date"].is_monotonic_increasing
    assert df["first_visit_date"].is_unique


def test_patient_level_keeps_first_encounter(raw_csv):
    enc_df = du.load(raw_csv, level="encounter")
    pat_df, rep = du.load(raw_csv, level="patient", return_report=True)
    assert pat_df["patient_id"].is_unique
    assert rep["excluded_repeat_encounters"] == len(enc_df) - len(pat_df)
    first = enc_df.groupby("patient_id")["encounter_id"].min()
    got = pat_df.set_index("patient_id")["encounter_id"]
    assert (got.reindex(first.index) == first).all()


def test_bad_level_rejected(raw_csv):
    with pytest.raises(ValueError):
        du.load(raw_csv, level="visit")
