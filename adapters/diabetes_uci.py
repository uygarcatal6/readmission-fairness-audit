"""
adapters/diabetes_uci.py — UCI "Diabetes 130-US Hospitals for Years 1999-2008" loader.

Source: UCI Machine Learning Repository, dataset id 296 (CC BY 4.0). Introductory
paper: Strack et al., BioMed Research International 2014, doi:10.1155/2014/781670.
Raw file: data/raw/diabetic_data.csv (101,766 encounters x 50 columns; 71,518 unique
patients). Column dictionary and aggregate profile: data/README_DATA.md, data/profile.md.

This module maps the raw CSV onto named, typed columns and applies the row exclusions;
readmission/cohort.py builds the analysis cohort from its output. No modelling
happens here.

Mapping table (raw -> output)
--------------------------------------------------------------------------------------
| raw column(s)             | output column         | rule                                |
|---------------------------|-----------------------|-------------------------------------|
| patient_nbr               | patient_id            | "D" + patient_nbr (string). Repeat  |
|                           |                       | visits share the id, so the         |
|                           |                       | patient-wise split groups them.     |
| encounter_id              | encounter_id          | int, kept as ordering key           |
|                           |                       | (assumed chronological; UCI gives   |
|                           |                       | no dates).                          |
| age "[a-b)"               | age                   | band midpoint (a+b)/2 as int:       |
|                           |                       | [0-10)->5 ... [90-100)->95. Bands   |
|                           |                       | below 20 are removed by             |
|                           |                       | readmission/cohort.py, not here.    |
| gender                    | sex                   | Female->"K", Male->"E" (cohort.py   |
|                           |                       | maps back); "Unknown/Invalid" out.  |
| race                      | race                  | as-is; "?"->"Unknown" (kept as its  |
|                           |                       | own subgroup, never dropped). Third |
|                           |                       | sensitive attribute (sex, age, race)|
| readmitted                | clinical_diagnosis    | "<30"->readmit_lt30,                |
|                           |                       | ">30"->readmit_gt30, "NO"->         |
|                           |                       | no_readmit (3-level label, kept for |
|                           |                       | reference; readmission/ uses        |
|                           |                       | readmission_30d). ASCII-safe tokens.|
| readmitted                | readmission_30d       | readmitted == "<30" (bool).         |
|                           |                       | Outcome column, never a feature.    |
| diag_1                    | icd10                 | raw ICD-9 code string. The column   |
|                           |                       | name is historical: the codes are   |
|                           |                       | ICD-9. Not a model input (the       |
|                           |                       | diag_1_group column is).            |
|                           |                       | "?" -> row dropped.                 |
| diag_1                    | diag_1_group          | Strack 2014 nine-group collapse     |
|                           |                       | (see diag1_group()).                |
| diabetesMed               | prescription_recorded | == "Yes".                           |
| (none)                    | imaging_ordered       | constant False: the dataset has no  |
|                           |                       | imaging information. Kept for a     |
|                           |                       | stable column set; not used by      |
|                           |                       | readmission/.                       |
| rank(encounter_id)        | first_visit_date      | synthetic monotone timestamp        |
|                           |                       | (2000-01-01 + rank minutes).        |
|                           |                       | ORDERING ONLY, NOT REAL DATES.      |
| A1Cresult                 | hba1c_requested       | != "None" (test-ordering flag).     |
| A1Cresult                 | a1c_result            | None->not_measured, >7->gt7,        |
|                           |                       | >8->gt8, Norm->norm.                |
| max_glu_serum             | glucose_requested     | != "None" (test-ordering flag).     |
| max_glu_serum             | max_glu_serum         | None->not_measured, >200->gt200,    |
|                           |                       | >300->gt300, Norm->norm.            |
| time_in_hospital,         | same names            | int.                                |
| num_lab_procedures,       |                       |                                     |
| num_procedures,           |                       |                                     |
| num_medications,          |                       |                                     |
| number_outpatient,        |                       |                                     |
| number_emergency,         |                       |                                     |
| number_inpatient,         |                       |                                     |
| number_diagnoses          |                       |                                     |
| admission_type_id         | admission_type        | IDS_mapping label (snake_case);     |
|                           |                       | 5/6/8 (Not Available / NULL / Not   |
|                           |                       | Mapped) -> unknown.                 |
| discharge_disposition_id  | discharge_group       | home / transfer_facility /          |
|                           |                       | left_ama / outpatient_pending /     |
|                           |                       | unknown. Death and hospice codes    |
|                           |                       | (11,13,14,19,20,21) are EXCLUDED    |
|                           |                       | (cannot be readmitted; Strack 2014).|
| admission_source_id       | admission_source      | emergency_room / referral /         |
|                           |                       | transfer / other_unknown.           |
| medical_specialty         | medical_specialty     | "?"->unknown; specialties with      |
|                           |                       | < MIN_CATEGORY_N encounters ->      |
|                           |                       | other; names sanitised to           |
|                           |                       | [0-9A-Za-z_].                       |
| payer_code                | payer_code            | "?"->unknown; codes with            |
|                           |                       | < MIN_CATEGORY_N -> other.          |
| insulin, metformin        | insulin, metformin    | no / steady / up / down.            |
| change                    | med_change            | == "Ch" (bool).                     |

Row exclusions, applied in this order (counts are returned by load(..., return_report=True)):
    1. gender == "Unknown/Invalid"
    2. discharge_disposition_id in DEATH_HOSPICE_CODES
    3. diag_1 == "?" (primary diagnosis missing)
    4. level="patient": keep only the first encounter (min encounter_id) per patient
    readmission/cohort.py additionally drops the age bands [0-10) and [10-20).

Levels:
    encounter  one row per encounter (default). Repeat visits of a patient are kept;
               readmission/splits.py keeps all encounters of a patient in one fold
               (StratifiedGroupKFold on patient_id).
    patient    first encounter per patient only (Strack 2014 style).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

READMITTED_TO_LABEL: dict[str, str] = {
    "<30": "readmit_lt30",
    ">30": "readmit_gt30",
    "NO": "no_readmit",
}

SEX_MAP: dict[str, str] = {"Female": "K", "Male": "E"}

# discharge_disposition_id values meaning death or hospice (IDS_mapping.csv):
# 11 Expired; 13 Hospice/home; 14 Hospice/medical facility; 19/20/21 Expired (Medicaid
# hospice variants). Such encounters cannot be followed by a readmission (Strack 2014).
DEATH_HOSPICE_CODES: frozenset[int] = frozenset({11, 13, 14, 19, 20, 21})

ADMISSION_TYPE: dict[int, str] = {
    1: "emergency",
    2: "urgent",
    3: "elective",
    4: "newborn",
    7: "trauma_center",
    # 5 Not Available, 6 NULL, 8 Not Mapped -> unknown
}
DISCHARGE_GROUP: dict[int, str] = {
    1: "home",
    6: "home",
    8: "home",
    7: "left_ama",
    12: "outpatient_pending",
    # 18 NULL, 25 Not Mapped, 26 Unknown/Invalid -> unknown
}
_TRANSFER_DISCHARGE = {2, 3, 4, 5, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30}
ADMISSION_SOURCE: dict[int, str] = {
    7: "emergency_room",
    1: "referral",
    2: "referral",
    3: "referral",
}
_TRANSFER_SOURCE = {4, 5, 6, 10, 18, 19, 22, 25, 26}

MIN_CATEGORY_N = 500  # rarer medical_specialty / payer_code levels -> "other"

NUMERIC_FEATURES: list[str] = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
CATEGORICAL_FEATURES: list[str] = [
    "admission_type",
    "discharge_group",
    "admission_source",
    "medical_specialty",
    "payer_code",
    "diag_1_group",
    "a1c_result",
    "max_glu_serum",
    "insulin",
    "metformin",
]
BOOL_FEATURES: list[str] = ["med_change", "prescription_recorded"]

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "encounter_id", "patient_nbr", "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "number_diagnoses", "max_glu_serum", "A1Cresult",
    "metformin", "insulin", "change", "diabetesMed", "readmitted",
)

_AGE_BAND_RE = re.compile(r"^\[(\d+)-(\d+)\)$")


# ── Small mapping helpers (pure functions; unit-tested) ─────────────────────


def age_band_midpoint(band: str) -> int | None:
    """'[70-80)' -> 75. Unparseable -> None."""
    m = _AGE_BAND_RE.match(str(band).strip())
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo + hi) // 2


def diag1_group(code: str | None) -> str:
    """ICD-9 diag_1 -> Strack et al. 2014 nine primary-diagnosis groups.

    circulatory 390-459, 785 | respiratory 460-519, 786 | digestive 520-579, 787 |
    diabetes 250.xx | injury 800-999 | musculoskeletal 710-739 |
    genitourinary 580-629, 788 | neoplasms 140-239 | other (incl. V/E codes, "?").
    """
    if code is None:
        return "other"
    c = str(code).strip()
    if not c or c == "?" or c[0] in "VEve":
        return "other"
    try:
        x = float(c)
    except ValueError:
        return "other"
    if 250.0 <= x < 251.0:
        return "diabetes"
    i = int(x)
    if 390 <= i <= 459 or i == 785:
        return "circulatory"
    if 460 <= i <= 519 or i == 786:
        return "respiratory"
    if 520 <= i <= 579 or i == 787:
        return "digestive"
    if 800 <= i <= 999:
        return "injury"
    if 710 <= i <= 739:
        return "musculoskeletal"
    if 580 <= i <= 629 or i == 788:
        return "genitourinary"
    if 140 <= i <= 239:
        return "neoplasms"
    return "other"


def _sanitize(value: str) -> str:
    """Category value -> [0-9A-Za-z_] token (safe as a one-hot feature name)."""
    v = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return v.lower() if v else "unknown"


def _map_admission_type(ids: pd.Series) -> pd.Series:
    return ids.map(ADMISSION_TYPE).fillna("unknown")


def _map_discharge_group(ids: pd.Series) -> pd.Series:
    out = ids.map(DISCHARGE_GROUP)
    out = out.where(~ids.isin(_TRANSFER_DISCHARGE), "transfer_facility")
    return out.fillna("unknown")


def _map_admission_source(ids: pd.Series) -> pd.Series:
    out = ids.map(ADMISSION_SOURCE)
    out = out.where(~ids.isin(_TRANSFER_SOURCE), "transfer")
    return out.fillna("other_unknown")


def _collapse_rare(s: pd.Series, *, min_n: int, missing_token: str = "?") -> pd.Series:
    """'?' -> unknown, rare levels -> other, names sanitised."""
    s = s.astype("string").fillna(missing_token)
    s = s.where(s != missing_token, "unknown")
    counts = s.value_counts()
    rare = set(counts[counts < min_n].index) - {"unknown"}
    s = s.where(~s.isin(rare), "other")
    return s.map(_sanitize).astype("string")


_LAB_TOKENS = {
    "None": "not_measured",
    "Norm": "norm",
    ">7": "gt7",
    ">8": "gt8",
    ">200": "gt200",
    ">300": "gt300",
}


def _lab_token(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    return s.map(_LAB_TOKENS).fillna(s.map(_sanitize)).astype("string")


# ── Public API ───────────────────────────────────────────────────────────────


def read_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the UCI CSV with the literal 'None' preserved (see data/profile.md note)."""
    raw = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"diabetes_uci: raw CSV lacks columns {missing}")
    return raw


def load(
    csv_path: str | Path,
    *,
    level: str = "encounter",
    return_report: bool = False,
    min_category_n: int = MIN_CATEGORY_N,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """diabetic_data.csv -> one row per encounter (or patient) with the mapped columns.

    Args:
        csv_path: path to the raw UCI CSV.
        level: "encounter" (all encounters) or "patient" (first encounter per patient).
        return_report: also return an exclusion/funnel dict with row counts.
        min_category_n: medical_specialty / payer_code levels rarer than this (counted
            on the WHOLE file) become "other". The approved protocol passes 1 (no
            collapse) and lets OneHotEncoder(min_frequency=...) group rare levels
            inside each training fold instead (readmission/preprocess.py).
    """
    if level not in ("encounter", "patient"):
        raise ValueError(f"level must be 'encounter' or 'patient', got {level!r}")

    raw = read_raw(csv_path)
    report: dict = {"level": level, "n_raw": int(len(raw))}

    # 1) gender Unknown/Invalid
    keep = raw["gender"].isin(SEX_MAP)
    report["excluded_gender_unknown"] = int((~keep).sum())
    raw = raw.loc[keep]

    # 2) death / hospice discharge
    disc = pd.to_numeric(raw["discharge_disposition_id"], errors="coerce")
    keep = ~disc.isin(DEATH_HOSPICE_CODES)
    report["excluded_death_hospice"] = int((~keep).sum())
    raw = raw.loc[keep]

    # 3) missing primary diagnosis
    keep = raw["diag_1"].astype(str).str.strip() != "?"
    report["excluded_diag1_missing"] = int((~keep).sum())
    raw = raw.loc[keep]

    raw = raw.assign(_enc=pd.to_numeric(raw["encounter_id"]).astype("int64"))
    raw = raw.sort_values("_enc").reset_index(drop=True)

    # 4) patient level: first encounter per patient
    if level == "patient":
        before = len(raw)
        raw = raw.drop_duplicates(subset=["patient_nbr"], keep="first").reset_index(drop=True)
        report["excluded_repeat_encounters"] = int(before - len(raw))
    else:
        report["excluded_repeat_encounters"] = 0

    n = len(raw)
    report["n_adapted"] = int(n)
    report["n_unique_patients"] = int(raw["patient_nbr"].nunique())

    def _int(col: str) -> pd.Series:
        return pd.to_numeric(raw[col], errors="coerce").astype("Int64")

    disc_id = pd.to_numeric(raw["discharge_disposition_id"], errors="coerce")
    adm_type_id = pd.to_numeric(raw["admission_type_id"], errors="coerce")
    adm_src_id = pd.to_numeric(raw["admission_source_id"], errors="coerce")
    readmitted = raw["readmitted"].astype("string")
    a1c = raw["A1Cresult"].astype("string")
    glu = raw["max_glu_serum"].astype("string")
    race = raw["race"].astype("string")

    rank = np.arange(n, dtype="int64")  # encounter_id order (chronology ASSUMED)
    df = pd.DataFrame(
        {
            # ── identifiers, demographics, target label ──
            "patient_id": ("D" + raw["patient_nbr"].astype(str)).astype("string"),
            "age": raw["age"].map(age_band_midpoint).astype("Int16"),
            "sex": raw["gender"].map(SEX_MAP).astype("string"),
            # ordering only, not real dates (dataset has none)
            "first_visit_date": pd.Timestamp("2000-01-01")
            + pd.to_timedelta(rank, unit="min"),
            "icd10": raw["diag_1"].astype("string"),  # ICD-9 code, see docstring
            "clinical_diagnosis": readmitted.map(READMITTED_TO_LABEL).astype("string"),
            "imaging_ordered": pd.Series([False] * n, dtype="boolean"),
            "prescription_recorded": (raw["diabetesMed"] == "Yes").astype("boolean"),
            # ── outcome ──
            "readmission_30d": (readmitted == "<30").astype("boolean"),
            # ── extra sensitive attribute ──
            "race": race.where(race != "?", "Unknown").astype("string"),
            # ── ids / ordering ──
            "encounter_id": raw["_enc"].to_numpy(),
            # ── numeric utilisation features ──
            **{c: _int(c) for c in NUMERIC_FEATURES},
            # ── categorical features ──
            "admission_type": _map_admission_type(adm_type_id).astype("string"),
            "discharge_group": _map_discharge_group(disc_id).astype("string"),
            "admission_source": _map_admission_source(adm_src_id).astype("string"),
            "medical_specialty": _collapse_rare(
                raw["medical_specialty"], min_n=min_category_n
            ),
            "payer_code": _collapse_rare(raw["payer_code"], min_n=min_category_n),
            "diag_1_group": raw["diag_1"].map(diag1_group).astype("string"),
            "a1c_result": _lab_token(a1c),
            "max_glu_serum": _lab_token(glu),
            "insulin": raw["insulin"].astype("string").str.lower(),
            "metformin": raw["metformin"].astype("string").str.lower(),
            "med_change": (raw["change"] == "Ch").astype("boolean"),
            # ── test-ordering flags (test-ordering analysis; not model features) ──
            "hba1c_requested": (a1c != "None").astype("boolean"),
            "glucose_requested": (glu != "None").astype("boolean"),
        }
    )

    if return_report:
        return df, report
    return df
