"""
readmission/cohort.py — the analysis cohort of the approved protocol.

Steps (docs/APPROVED_PROPOSAL.md §2, "Data and target"):
    1. adapters.diabetes_uci.load drops unknown sex, death/hospice discharges and a
       missing primary diagnosis. Rare medical_specialty / payer_code levels are NOT
       collapsed here (min_category_n=1); readmission/preprocess.py does that inside each
       training fold.
    2. The two paediatric age bands [0-10) and [10-20) are dropped.
    3. Target y = 1 if readmitted within 30 days, else 0 ("everything else" = >30 or NO).
    4. Sensitive attributes: sex (Female/Male), age_band (20-39/40-59/60-79/80+), race.

On the UCI file this gives 98,470 encounters from 69,303 patients, 11.44 % positive
(tests/test_readmission_core.py::test_real_cohort_matches_the_approved_numbers).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adapters import diabetes_uci as du

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "raw" / "diabetic_data.csv"

# The adapter stores age as the band midpoint ([20-30) -> 25), so ">= 20" keeps [20-30) and
# drops [0-10) -> 5 and [10-20) -> 15.
MIN_AGE = 20
AGE_BANDS: tuple[tuple[int, int, str], ...] = (
    (20, 40, "20-39"),
    (40, 60, "40-59"),
    (60, 80, "60-79"),
    (80, 200, "80+"),
)
SEX_LABELS = {"K": "Female", "E": "Male"}  # the adapter's codes -> paper labels

TARGET = "y"
GROUP = "patient_id"
SENSITIVE: tuple[str, ...] = ("sex", "age_band", "race")

# Model inputs: 23 raw attributes. age is the band midpoint (numeric); the two yes/no
# attributes enter as 0/1 numbers.
DEMOGRAPHIC = ["age", "sex", "race"]
NUMERIC = ["age", *du.NUMERIC_FEATURES, *du.BOOL_FEATURES]
CATEGORICAL = ["sex", "race", *du.CATEGORICAL_FEATURES]
FEATURES = NUMERIC + CATEGORICAL

# Kept for the test-ordering analysis; never model inputs.
TEST_ORDERING = ["hba1c_requested", "glucose_requested"]


def age_band(age: pd.Series) -> pd.Series:
    """Band midpoint (25, 35, ..., 95) -> '20-39' / '40-59' / '60-79' / '80+'; else NA."""
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    for lo, hi, label in AGE_BANDS:
        out[(age >= lo) & (age < hi)] = label
    return out


def load_cohort(
    csv_path: str | Path = DEFAULT_CSV,
    *,
    first_encounter_only: bool = False,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Raw UCI CSV -> one row per encounter with features, y, patient_id and sensitive columns.

    Args:
        csv_path: path to diabetic_data.csv.
        first_encounter_only: keep each patient's earliest encounter (lowest encounter_id)
            after the exclusions; used to replicate the test-ordering analysis.
        return_report: also return the exclusion funnel and the class balance.
    """
    raw, rep = du.load(csv_path, level="encounter", return_report=True, min_category_n=1)

    keep = raw["age"] >= MIN_AGE
    n_paediatric = int((~keep).sum())
    raw = raw.loc[keep]

    n_repeat = 0
    if first_encounter_only:
        before = len(raw)
        raw = raw.sort_values("encounter_id").drop_duplicates(GROUP, keep="first")
        n_repeat = before - len(raw)
    raw = raw.reset_index(drop=True)

    # Plain numpy / object dtypes: scikit-learn transformers handle them without surprises.
    df = pd.DataFrame({
        "encounter_id": raw["encounter_id"].astype("int64"),
        GROUP: raw["patient_id"].astype(str).astype(object),
        TARGET: raw["readmission_30d"].astype(bool).astype("int8"),
        "age_band": age_band(raw["age"]),
    })
    for col in NUMERIC:
        df[col] = raw[col].astype("float64")
    for col in CATEGORICAL:
        df[col] = raw[col].astype(str).astype(object)
    df["sex"] = raw["sex"].map(SEX_LABELS).astype(object)
    for col in TEST_ORDERING:
        df[col] = raw[col].astype(bool)

    if not return_report:
        return df
    p = float(df[TARGET].mean())
    report = {
        "n_raw": rep["n_raw"],
        "excluded_unknown_sex": rep["excluded_gender_unknown"],
        "excluded_death_hospice": rep["excluded_death_hospice"],
        "excluded_missing_primary_diagnosis": rep["excluded_diag1_missing"],
        "excluded_paediatric": n_paediatric,
        "excluded_repeat_encounters": n_repeat,
        "n_encounters": len(df),
        "n_patients": int(df[GROUP].nunique()),
        "n_positive": int(df[TARGET].sum()),
        "prevalence": p,
        "no_information_rate": max(p, 1.0 - p),
    }
    return df, report
