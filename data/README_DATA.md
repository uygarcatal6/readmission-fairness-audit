# Data — Diabetes 130-US Hospitals (1999-2008)

This folder holds the public clinical dataset used for the algorithmic-fairness
/ bias-audit project, plus everything needed to reproduce and verify it.

- **Dataset:** Diabetes 130-US Hospitals for Years 1999-2008
- **Source:** UCI Machine Learning Repository, dataset **id 296**
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**) —
  see [`LICENSE_CC-BY-4.0.md`](LICENSE_CC-BY-4.0.md)
- **Records:** 101,766 encounters × 50 columns; 71,518 unique patients
- **De-identified:** yes (public research dataset; contains age band, gender,
  race, but no direct identifiers)

## Contents

```
data/
├── raw/                      (git-ignored; written by download.py)
│   ├── diabetes+130-us+hospitals+for+years+1999-2008.zip   (source zip)
│   ├── diabetic_data.csv                                    (main table)
│   └── IDS_mapping.csv                                      (id → label maps)
├── download.py            # re-download + SHA-256 verification
├── profile.md             # human-readable aggregate profile
├── profile.json           # machine-readable profile (same figures)
├── README_DATA.md         # this file
└── LICENSE_CC-BY-4.0.md   # dataset license + required attribution
```

## Source & citation

- UCI dataset page:
  <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
- The UCI page distributes the data under **CC BY 4.0**.

Required attribution (the dataset's own introductory paper):

> Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian
> Ventura, Krzysztof J. Cios, and John N. Clore, "Impact of HbA1c Measurement
> on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient
> Records," *BioMed Research International*, vol. 2014, Article ID 781670,
> 11 pages, 2014.
> DOI: [10.1155/2014/781670](https://doi.org/10.1155/2014/781670)

DOI verified via Crossref (`https://api.crossref.org/works/10.1155/2014/781670`):
the title, journal (*BioMed Research International*), volume (2014), and full
author list all match.

## How to (re)download

Primary — direct UCI zip:

```bash
curl -L -o "raw/diabetes+130-us+hospitals+for+years+1999-2008.zip" \
  "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
unzip -o "raw/diabetes+130-us+hospitals+for+years+1999-2008.zip" -d raw/
```

Fallback — Python package `ucimlrepo`:

```bash
pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; d=fetch_ucirepo(id=296); print(d.data.features.shape)"
```

Or just run the bundled script (does download-if-missing + verification):

```bash
python download.py            # verify existing files
python download.py --force    # force a fresh download
```

## Integrity (SHA-256)

Recorded on 2026-09-16; `download.py` checks against these:

| File | Bytes | SHA-256 |
|---|---:|---|
| `diabetes+130-us+hospitals+for+years+1999-2008.zip` | 3,170,254 | `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a` |
| `diabetic_data.csv` | 19,159,383 | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| `IDS_mapping.csv` | 2,547 | `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e` |

## Aggregate profile (highlights)

Full detail in [`profile.md`](profile.md) / [`profile.json`](profile.json).

- **Target `readmitted`:** `NO` 54,864 (53.9%) · `>30` 35,545 (34.9%) ·
  `<30` 11,357 (11.2%). Early-readmission (`<30`) is the usual imbalanced
  binary label.
- **Patient recurrence:** 101,766 encounters for only 71,518 patients (mean
  1.42, max 40 encounters/patient). **Use a patient-wise (grouped) split** —
  a row-level split leaks patients across folds.
- **Sensitive attributes:** gender (Female 54,708 / Male 47,055 / Unknown 3),
  10-year age bands, race (Caucasian 74.8%, AfricanAmerican 18.9%,
  Hispanic 2.0%, Other 1.5%, Asian 0.6%, missing `?` 2.2%).
- **Test-ordering signal:** `A1Cresult` not ordered in 83.3% of encounters,
  `max_glu_serum` in 94.8%; the not-ordered rate varies by gender × race,
  giving a process-level fairness target beyond outcome metrics.
- **Missingness (`?`):** `weight` 96.9% (unusable), `medical_specialty` 49.1%,
  `payer_code` 39.6%, `race` 2.2%.

> Parsing caveat: `A1Cresult` / `max_glu_serum` store the literal string
> `None` for "test not ordered". Read the CSV with `keep_default_na=False`
> (or add `"None"` handling) so pandas does not turn it into `NaN`.

## Column dictionary

### Feature / target columns (from the UCI variable table, dataset id 296)

| Column | Role | Type | Missing | Description |
|---|---|---|---|---|
| `encounter_id` | ID | — | no | Unique identifier of an encounter |
| `patient_nbr` | ID | — | no | Unique identifier of a patient |
| `race` | Feature | Categorical | yes | Values: Caucasian, Asian, AfricanAmerican, Hispanic, Other |
| `gender` | Feature | Categorical | no | Values: Male, Female, Unknown/Invalid |
| `age` | Feature | Categorical | no | Grouped in 10-year intervals: [0,10), …, [90,100) |
| `weight` | Feature | Categorical | yes | Weight in pounds (≈97% missing) |
| `admission_type_id` | Feature | Categorical | no | Integer id, 9 distinct values (see id maps below) |
| `discharge_disposition_id` | Feature | Categorical | no | Integer id, 29 distinct values (see id maps) |
| `admission_source_id` | Feature | Categorical | no | Integer id, 21 distinct values (see id maps) |
| `time_in_hospital` | Feature | Integer | no | Days between admission and discharge (1–14) |
| `payer_code` | Feature | Categorical | yes | Integer id, 23 distinct values |
| `medical_specialty` | Feature | Categorical | yes | Specialty of the admitting physician |
| `num_lab_procedures` | Feature | Integer | no | Number of lab tests performed during the encounter |
| `num_procedures` | Feature | Integer | no | Number of procedures (other than lab tests) |
| `num_medications` | Feature | Integer | no | Number of distinct generic drug names administered |
| `number_outpatient` | Feature | Integer | no | Outpatient visits in the year before the encounter |
| `number_emergency` | Feature | Integer | no | Emergency visits in the year before the encounter |
| `number_inpatient` | Feature | Integer | no | Inpatient visits in the year before the encounter |
| `diag_1` | Feature | Categorical | yes | Primary diagnosis (first 3 digits of ICD-9); 848 distinct |
| `diag_2` | Feature | Categorical | yes | Secondary diagnosis (first 3 digits of ICD-9); 923 distinct |
| `diag_3` | Feature | Categorical | yes | Additional secondary diagnosis (first 3 ICD-9 digits) |
| `number_diagnoses` | Feature | Integer | no | Number of diagnoses entered to the system |
| `max_glu_serum` | Feature | Categorical | no | Glucose serum test range or "None" if not taken: >200, >300, Norm, None |
| `A1Cresult` | Feature | Categorical | no | HbA1c test range or "None" if not taken: >7, >8, Norm, None |
| 23 drug columns | Feature | Categorical | no | `metformin` … `metformin-pioglitazone`: each = Up / Down / Steady / No |
| `change` | Feature | Categorical | no | Change in diabetic medication: Ch / No |
| `diabetesMed` | Feature | Categorical | no | Any diabetic medication prescribed: Yes / No |
| `readmitted` | Target | Categorical | no | `<30`, `>30`, or `NO` (days to inpatient readmission) |

The 23 drug columns (each with values `Up`/`Down`/`Steady`/`No`):
`metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`,
`acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`,
`rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`,
`examide`, `citoglipton`, `insulin`, `glyburide-metformin`,
`glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`,
`metformin-pioglitazone`.

### ID → label maps (from `IDS_mapping.csv`)

`IDS_mapping.csv` decodes the three integer id columns. Selected entries:

**`admission_type_id`** (8 codes): 1 Emergency · 2 Urgent · 3 Elective ·
4 Newborn · 5 Not Available · 6 NULL · 7 Trauma Center · 8 Not Mapped.

**`discharge_disposition_id`** (29 codes) — e.g.: 1 Discharged to home ·
2 Transferred to another short-term hospital · 3 Transferred to SNF ·
11 Expired · 13 Hospice / home · 18 NULL · 25 Not Mapped ·
26 Unknown/Invalid. (Full list in `IDS_mapping.csv`.)

**`admission_source_id`** (25 codes) — e.g.: 1 Physician Referral ·
4 Transfer from a hospital · 7 Emergency Room · 9 Not Available ·
17 NULL · 20 Not Mapped · 21 Unknown/Invalid. (Full list in `IDS_mapping.csv`.)

> Codes 6/18 (NULL), 5/9/15 (Not Available), 8/20/25 (Not Mapped) and
> 26/21 (Unknown/Invalid) are all forms of missing/unknown metadata and should
> be treated as such during preprocessing.

## fairlearn's built-in version (for reference)

fairlearn 0.13 ships a convenience loader:

```python
from fairlearn.datasets import fetch_diabetes_hospital
d = fetch_diabetes_hospital(as_frame=True)   # d.frame
```

This works, but it is **not** the raw UCI file — it is a preprocessed
derivative the Fairlearn team prepared for their SciPy 2021 tutorial and
publishes via OpenML (`data_id=43874`). Differences vs. the raw UCI CSV:

- **Same row count** (101,766) but **25 columns** instead of 50.
- **`age`** is collapsed from 10 bands into 3: `'30 years or younger'`,
  `'30-60 years'`, `'Over 60 years'`.
- **`race`** uses `Unknown` instead of the raw `?`.
- **`diag_1`** is replaced by `primary_diagnosis`, grouped into readable
  categories (Diabetes, Other, Respiratory Issues, Musculoskeletal Issues,
  Genitourinary Issues, …) rather than raw ICD-9 codes.
- **Engineered columns added:** `medicare`, `medicaid` (booleans derived from
  `payer_code`); `had_emergency`, `had_inpatient_days`, `had_outpatient_days`
  (booleans from the `number_*` visit counts); `readmit_binary` (any
  readmission) and `readmit_30_days` (the tutorial's target).
- **Columns dropped:** `encounter_id`, `patient_nbr` (⚠ so a patient-wise
  split is no longer possible with this version), `weight`, most individual
  drug columns (only `insulin` is kept), the id-mapped columns, etc.
- **`readmit_30_days`** target: 90,409 negative / 11,357 positive (matches the
  raw `<30` count).

Preprocessing script (per fairlearn's dataset description):
<https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py>

For this project we use the **raw UCI CSV** in `raw/` so we keep `patient_nbr`
(needed for a leakage-free patient-wise split) and the full set of sensitive
and process features.

## Prior fairness uses

This dataset is a standard benchmark for fairness/bias auditing in healthcare:

- **Fairlearn documentation** — the "Diabetes 130-Hospitals" dataset is a
  built-in fairness example demonstrating subgroup disparity analysis and
  mitigation with race/gender/age as sensitive features:
  <https://fairlearn.org/main/user_guide/datasets/diabetes_hospital_data.html>
- **Le Quy et al., "A survey on datasets for fairness-aware machine learning"**
  (arXiv:2110.00530) lists this dataset among the canonical fairness benchmarks
  and characterises its sensitive attributes:
  <https://arxiv.org/abs/2110.00530>
- **Fabris et al., "Algorithmic Fairness Datasets: the Story so Far"**
  (arXiv:2202.01711) documents the dataset's provenance and its adoption in
  fairness research: <https://arxiv.org/abs/2202.01711>
