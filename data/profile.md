# Data Profile — Diabetes 130-US Hospitals (UCI id 296)

All figures below are **aggregates** computed from `raw/diabetic_data.csv`
(the public, de-identified UCI dataset). The machine-readable counterpart is
[`profile.json`](profile.json). `python download.py` downloads and verifies the file these
figures were computed from.

> Note on parsing: `A1Cresult` and `max_glu_serum` use the literal string
> `None` to mean "test not ordered". This value must be read with
> `keep_default_na=False`, otherwise pandas silently converts `None` to `NaN`
> and the test-ordering signal disappears.

## 1. Shape

| Property | Value |
|---|---|
| Rows (encounters) | **101,766** |
| Columns | **50** |
| Unique patients (`patient_nbr`) | **71,518** |

Confirms the expected 101,766 × 50 shape.

### Column list (50, original order)

`encounter_id`, `patient_nbr`, `race`, `gender`, `age`, `weight`,
`admission_type_id`, `discharge_disposition_id`, `admission_source_id`,
`time_in_hospital`, `payer_code`, `medical_specialty`, `num_lab_procedures`,
`num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`,
`number_inpatient`, `diag_1`, `diag_2`, `diag_3`, `number_diagnoses`,
`max_glu_serum`, `A1Cresult`, `metformin`, `repaglinide`, `nateglinide`,
`chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`,
`tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`,
`troglitazone`, `tolazamide`, `examide`, `citoglipton`, `insulin`,
`glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`,
`metformin-rosiglitazone`, `metformin-pioglitazone`, `change`, `diabetesMed`,
`readmitted`

## 2. Target: `readmitted`

| Value | Count | Share |
|---|---:|---:|
| `NO` | 54,864 | 53.91% |
| `>30` | 35,545 | 34.93% |
| `<30` | 11,357 | 11.16% |

The common binary framing is `<30` (early readmission) vs. the rest
(11,357 positives, ~11.2%) — an imbalanced target.

## 3. Sensitive attributes

### gender
| Value | Count |
|---|---:|
| Female | 54,708 |
| Male | 47,055 |
| Unknown/Invalid | 3 |

### age (10-year bands)
| Band | Count |
|---|---:|
| [0-10) | 161 |
| [10-20) | 691 |
| [20-30) | 1,657 |
| [30-40) | 3,775 |
| [40-50) | 9,685 |
| [50-60) | 17,256 |
| [60-70) | 22,483 |
| [70-80) | 26,068 |
| [80-90) | 17,197 |
| [90-100) | 2,793 |

### race
| Value | Count | Share |
|---|---:|---:|
| Caucasian | 76,099 | 74.78% |
| AfricanAmerican | 19,210 | 18.88% |
| `?` (missing) | 2,273 | 2.23% |
| Hispanic | 2,037 | 2.00% |
| Other | 1,506 | 1.48% |
| Asian | 641 | 0.63% |

## 4. Patient recurrence (matters for patient-wise split)

101,766 encounters belong to only **71,518 unique patients**, so the same
patient can appear in multiple rows. A naive row-level train/test split would
leak the same patient across folds; a **patient-wise (grouped) split** is
required.

| Metric | Value |
|---|---:|
| Encounters per patient — mean | 1.42 |
| Encounters per patient — median | 1 |
| Encounters per patient — max | 40 |
| Patients with exactly 1 encounter | 54,745 |
| Patients with >1 encounter | 16,773 |

Distribution of encounters per patient:

| # encounters | # patients |
|---|---:|
| 1 | 54,745 |
| 2 | 10,434 |
| 3 | 3,328 |
| 4 | 1,421 |
| 5+ | 1,590 |

## 5. Test-ordering equality signal: `A1Cresult` & `max_glu_serum`

`None` = the lab test was **not ordered** during the encounter.

| Column | None | >8 / >300 | >7 / >200 | Norm |
|---|---:|---:|---:|---:|
| `A1Cresult` | 84,748 (83.28%) | 8,216 (`>8`) | 3,812 (`>7`) | 4,990 |
| `max_glu_serum` | 96,420 (94.75%) | 1,264 (`>300`) | 1,485 (`>200`) | 2,597 |

**`None` rate by gender × race** (share of encounters where the test was NOT
ordered). Groups with n < 5 (`Unknown/Invalid` gender) omitted here; see
`profile.json` for the full table.

`A1Cresult` not-ordered %:

| gender \ race | AfricanAmerican | Asian | Caucasian | Hispanic | Other | ? |
|---|---:|---:|---:|---:|---:|---:|
| Female | 82.77 | 76.73 | 84.67 | 76.47 | 81.15 | 82.52 |
| Male | 80.10 | 81.11 | 83.33 | 75.24 | 78.34 | 80.32 |

`max_glu_serum` not-ordered %:

| gender \ race | AfricanAmerican | Asian | Caucasian | Hispanic | Other | ? |
|---|---:|---:|---:|---:|---:|---:|
| Female | 97.83 | 95.60 | 93.82 | 89.19 | 95.32 | 97.79 |
| Male | 97.27 | 96.28 | 94.24 | 92.80 | 95.38 | 97.28 |

The spread across subgroups (e.g. Hispanic patients have a noticeably lower
not-ordered rate for both tests) is exactly the kind of process-level
disparity a test-ordering / procedural-fairness analysis can target, alongside
the usual outcome-based (readmission) fairness metrics.

## 6. Primary diagnosis (`diag_1`, first 3 ICD-9 characters) — top 15

| ICD-9 (3) | Count | Typical meaning |
|---|---:|---|
| 250 | 8,757 | Diabetes mellitus |
| 428 | 6,862 | Heart failure |
| 414 | 6,581 | Chronic ischemic heart disease |
| 786 | 4,016 | Respiratory system / chest symptoms |
| 410 | 3,614 | Acute myocardial infarction |
| 486 | 3,508 | Pneumonia |
| 427 | 2,766 | Cardiac dysrhythmias |
| 491 | 2,275 | Chronic bronchitis |
| 715 | 2,151 | Osteoarthrosis |
| 682 | 2,042 | Other cellulitis / abscess |
| 434 | 2,028 | Occlusion of cerebral arteries |
| 780 | 2,019 | General symptoms |
| 996 | 1,967 | Complications of care |
| 276 | 1,889 | Fluid/electrolyte disorders |
| 038 | 1,688 | Septicemia |

(Meanings are the standard ICD-9 chapter readings, provided as a convenience;
the dataset stores only the codes.)

## 7. Numeric summaries

| Column | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `num_lab_procedures` | 43.10 | 19.67 | 1 | 31 | 44 | 57 | 132 |
| `num_medications` | 16.02 | 8.13 | 1 | 10 | 15 | 20 | 81 |
| `time_in_hospital` | 4.40 | 2.99 | 1 | 2 | 4 | 6 | 14 |

`time_in_hospital` is bounded to 1–14 days by the dataset's inclusion criteria.

## 8. Missingness — `?` code

The dataset encodes missing categorical values as the literal `?`. Columns with
any `?`, most-missing first:

| Column | `?` count | `?` % |
|---|---:|---:|
| `weight` | 98,569 | 96.86% |
| `medical_specialty` | 49,949 | 49.08% |
| `payer_code` | 40,256 | 39.56% |
| `race` | 2,273 | 2.23% |
| `diag_3` | 1,423 | 1.40% |
| `diag_2` | 358 | 0.35% |
| `diag_1` | 21 | 0.02% |

All other columns have no `?` values. `weight` is effectively unusable (~97%
missing); `medical_specialty` and `payer_code` are ~40-50% missing and need an
explicit missing-indicator strategy. Note that `race` missingness (2.23%) is a
fairness concern in itself — dropping those rows silently removes a subgroup.

## File integrity (SHA-256)

| File | Bytes | SHA-256 |
|---|---:|---|
| `diabetes+130-us+hospitals+for+years+1999-2008.zip` | 3,170,254 | `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a` |
| `diabetic_data.csv` | 19,159,383 | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| `IDS_mapping.csv` | 2,547 | `f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e` |
