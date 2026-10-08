# Approved protocol — test-ordering equity

Is a test result recorded equally often across groups once the other variables are held fixed? The dataset holds a result range for HbA1c and for serum glucose, or 'None'; a result recorded during the encounter is the data's proxy for the test being ordered (the approved wording). Logistic regression of each indicator on sex, age band, race and primary-diagnosis group (reference = the largest group of each); odds ratios with 95 % CIs from patient-clustered standard errors; Benjamini–Hochberg q over every sex, age-band and race term of both outcomes (diagnosis terms are adjustment; race 'Unknown' is the missing-value code and is not tested).

Events (rate): HbA1c result recorded 16,388 (0.166), serum glucose result recorded 5,136 (0.052). At these rates an odds ratio is larger than the matching risk ratio (read the crude rates in §1 for 'how often'). The models adjust for diagnosis group only: they describe who has a recorded result, not whether testing matched clinical need, and are not evidence of unequal care on their own. §4 shows that the glucose field in particular follows the data source.

## 1. Crude recording rates by group (all encounters, patient-bootstrap 95 % CI)

| Variable | Group | Encounters | HbA1c result recorded | serum glucose result recorded |
|---|---|---|---|---|
| sex | Female | 52,965 | 0.159 [0.156, 0.163] | 0.053 [0.050, 0.056] |
| sex | Male | 45,505 | 0.175 [0.171, 0.179] | 0.052 [0.049, 0.055] |
| age_band | 20-39 | 5,412 | 0.247 [0.235, 0.261] | 0.044 [0.035, 0.055] |
| age_band | 40-59 | 26,663 | 0.208 [0.203, 0.213] | 0.045 [0.041, 0.049] |
| age_band | 60-79 | 47,377 | 0.148 [0.145, 0.151] | 0.051 [0.048, 0.054] |
| age_band | 80+ | 19,018 | 0.131 [0.126, 0.136] | 0.067 [0.062, 0.072] |
| race | AfricanAmerican | 18,538 | 0.181 [0.175, 0.188] | 0.024 [0.021, 0.028] |
| race | Asian | 624 | 0.207 [0.176, 0.240] | 0.042 [0.026, 0.061] |
| race | Caucasian | 73,640 | 0.159 [0.156, 0.162] | 0.059 [0.057, 0.062] |
| race | Hispanic | 1,990 | 0.238 [0.219, 0.257] | 0.093 [0.074, 0.113] |
| race | Other | 1,457 | 0.202 [0.180, 0.222] | 0.047 [0.033, 0.063] |
| race | Unknown (missing code) | 2,221 | 0.188 [0.172, 0.206] | 0.024 [0.017, 0.032] |

## 2. Adjusted odds ratios, all encounters (n = 98,470, 69,303 patients)

| Variable | Contrast | HbA1c result recorded: OR [95 % CI] | p | BH q | serum glucose result recorded: OR [95 % CI] | p | BH q |
|---|---|---|---|---|---|---|---|
| sex | Male vs Female | 1.07 [1.03, 1.11] | 2.5e-04 | 5.1e-04 | 0.97 [0.89, 1.06] | 0.531 | 0.531 |
| age_band | 20-39 vs 60-79 | 1.60 [1.48, 1.73] | 3.6e-32 | 1.9e-31 | 0.82 [0.63, 1.07] | 0.145 | 0.179 |
| age_band | 40-59 vs 60-79 | 1.46 [1.40, 1.52] | 8.1e-69 | 1.3e-67 | 0.88 [0.79, 0.98] | 0.018 | 0.025 |
| age_band | 80+ vs 60-79 | 0.88 [0.84, 0.93] | 2.2e-06 | 7.2e-06 | 1.26 [1.14, 1.40] | 7.1e-06 | 1.9e-05 |
| race | AfricanAmerican vs Caucasian | 1.02 [0.98, 1.07] | 0.327 | 0.348 | 0.41 [0.35, 0.47] | 2.2e-34 | 1.7e-33 |
| race | Asian vs Caucasian | 1.42 [1.15, 1.75] | 9.6e-04 | 0.002 | 0.70 [0.45, 1.09] | 0.117 | 0.156 |
| race | Hispanic vs Caucasian | 1.47 [1.32, 1.65] | 1.3e-11 | 5.2e-11 | 1.66 [1.31, 2.12] | 3.6e-05 | 8.1e-05 |
| race | Other vs Caucasian | 1.25 [1.09, 1.43] | 0.001 | 0.002 | 0.81 [0.58, 1.13] | 0.218 | 0.249 |
| race | Unknown vs Caucasian | 1.20 [1.07, 1.34] (missing code, not tested) | 0.001 | — | 0.40 [0.30, 0.54] (missing code, not tested) | 1.5e-09 | — |

Clustering by patient changes the standard errors of the tested terms by a factor of HbA1c result recorded 1.04–1.11; serum glucose result recorded 1.13–1.85 relative to the model-based ones (repeat encounters of one patient are not independent). Significant (p < 0.05) under one kind of SE but not the other: serum glucose result recorded: age_band=20-39. Encounters may also cluster by hospital and period, which the public data does not identify; §4 shows this can move the estimates, not only their SEs.

## 3. Replication on first encounters (n = 69,303, one per patient)

Same model and references; BH within this family. With one encounter per patient the clustered SEs reduce to robust SEs, so this checks the repeat-encounter structure, not the hospital or period one.

| Variable | Contrast | HbA1c result recorded: OR [95 % CI] | p | BH q | serum glucose result recorded: OR [95 % CI] | p | BH q |
|---|---|---|---|---|---|---|---|
| sex | Male vs Female | 1.08 [1.04, 1.12] | 2.0e-04 | 3.9e-04 | 0.98 [0.91, 1.05] | 0.583 | 0.583 |
| age_band | 20-39 vs 60-79 | 1.55 [1.43, 1.69] | 1.6e-25 | 8.4e-25 | 0.78 [0.65, 0.93] | 0.005 | 0.007 |
| age_band | 40-59 vs 60-79 | 1.41 [1.35, 1.48] | 1.2e-48 | 1.9e-47 | 0.87 [0.80, 0.95] | 0.002 | 0.004 |
| age_band | 80+ vs 60-79 | 0.88 [0.83, 0.94] | 3.3e-05 | 7.6e-05 | 1.34 [1.23, 1.47] | 5.2e-11 | 2.1e-10 |
| race | AfricanAmerican vs Caucasian | 1.05 [1.00, 1.11] | 0.050 | 0.062 | 0.45 [0.40, 0.50] | 1.7e-38 | 1.4e-37 |
| race | Asian vs Caucasian | 1.36 [1.09, 1.70] | 0.007 | 0.009 | 0.84 [0.55, 1.30] | 0.442 | 0.471 |
| race | Hispanic vs Caucasian | 1.49 [1.32, 1.68] | 1.7e-10 | 5.5e-10 | 1.70 [1.40, 2.05] | 3.9e-08 | 1.0e-07 |
| race | Other vs Caucasian | 1.24 [1.07, 1.43] | 0.004 | 0.007 | 0.88 [0.67, 1.17] | 0.390 | 0.445 |
| race | Unknown vs Caucasian | 1.16 [1.04, 1.31] (missing code, not tested) | 0.011 | — | 0.47 [0.35, 0.62] (missing code, not tested) | 1.7e-07 | — |

Replicated: 15 of 16 tested terms have the same verdict (q < 0.05 and direction) in both analyses; different: serum glucose result recorded: age_band=20-39.

## 4. Where the recorded results come from, and a sensitivity check

Encounters with unknown admission type (the source's 'not available / not mapped' codes) and the encounter-id order (the order of the extraction) show whether a field was filled in by some sources or periods rather than per admission.

| Outcome | Results | From unknown admission type | Rate: unknown vs known type | Rate range over encounter-id deciles | Next encounter recorded: if this one is vs is not |
|---|---|---|---|---|---|
| HbA1c result recorded | 16,388 | 9.0% of results in 10.2% of encounters | 0.146 vs 0.169 | 0.089–0.203 | 0.238 vs 0.115 |
| serum glucose result recorded | 5,136 | 79.7% of results in 10.2% of encounters | 0.406 vs 0.012 | 0.010–0.151 | 0.888 vs 0.008 |

Sensitivity: the approved model plus one indicator 'admission_type = unknown' (not tested, not in a BH family). A group's odds ratio that moves towards 1 came partly from where and when the field was filled in.

| Variable | Contrast | HbA1c result recorded: approved | HbA1c result recorded: + source indicator | serum glucose result recorded: approved | serum glucose result recorded: + source indicator |
|---|---|---|---|---|---|
| sex | Male vs Female | 1.07 [1.03, 1.11] | 1.07 [1.03, 1.11] | 0.97 [0.89, 1.06] | 0.96 [0.87, 1.06] |
| age_band | 20-39 vs 60-79 | 1.60 [1.48, 1.73] | 1.60 [1.48, 1.73] | 0.82 [0.63, 1.07] | 0.75 [0.57, 0.97] |
| age_band | 40-59 vs 60-79 | 1.46 [1.40, 1.52] | 1.46 [1.40, 1.52] | 0.88 [0.79, 0.98] | 0.81 [0.72, 0.92] |
| age_band | 80+ vs 60-79 | 0.88 [0.84, 0.93] | 0.88 [0.84, 0.93] | 1.26 [1.14, 1.40] | 1.41 [1.25, 1.58] |
| race | AfricanAmerican vs Caucasian | 1.02 [0.98, 1.07] | 1.02 [0.97, 1.07] | 0.41 [0.35, 0.47] | 0.58 [0.50, 0.68] |
| race | Asian vs Caucasian | 1.42 [1.15, 1.75] | 1.42 [1.15, 1.75] | 0.70 [0.45, 1.09] | 0.62 [0.38, 1.00] |
| race | Hispanic vs Caucasian | 1.47 [1.32, 1.65] | 1.49 [1.33, 1.67] | 1.66 [1.31, 2.12] | 1.10 [0.84, 1.43] |
| race | Other vs Caucasian | 1.25 [1.09, 1.43] | 1.25 [1.09, 1.43] | 0.81 [0.58, 1.13] | 0.77 [0.54, 1.09] |
| race | Unknown vs Caucasian | 1.20 [1.07, 1.34] | 1.19 [1.06, 1.33] | 0.40 [0.30, 0.54] | 0.60 [0.44, 0.82] |
| admission_type | unknown vs other types | — | 0.85 [0.80, 0.90] | — | 57.70 [52.53, 63.38] |

## 5. Adjustment terms (primary-diagnosis group), all encounters

| Diagnosis group | Encounters | HbA1c result recorded: OR [95 % CI] | serum glucose result recorded: OR [95 % CI] |
|---|---|---|---|
| diabetes vs circulatory | 8,051 | 1.68 [1.58, 1.79] | 1.55 [1.36, 1.76] |
| digestive vs circulatory | 9,314 | 0.60 [0.56, 0.64] | 1.58 [1.40, 1.78] |
| genitourinary vs circulatory | 4,990 | 0.76 [0.70, 0.83] | 0.96 [0.82, 1.13] |
| injury vs circulatory | 6,825 | 0.58 [0.53, 0.63] | 0.96 [0.83, 1.10] |
| musculoskeletal vs circulatory | 4,932 | 0.51 [0.46, 0.56] | 0.78 [0.66, 0.93] |
| neoplasms vs circulatory | 3,128 | 0.48 [0.43, 0.55] | 0.93 [0.77, 1.14] |
| other vs circulatory | 17,677 | 0.88 [0.84, 0.93] | 1.33 [1.21, 1.46] |
| respiratory vs circulatory | 13,879 | 0.97 [0.92, 1.03] | 1.67 [1.51, 1.84] |

Run time: 0.2 min.
