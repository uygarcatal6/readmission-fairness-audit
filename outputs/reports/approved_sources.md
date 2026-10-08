# Approved protocol — where the disparities come from

Model: **rf** (the primary model of the audit), refitted on the same 25 outer splits without some of its inputs and flagged with the same rule (top q %). Leading attribute: **age_band**. Patient-bootstrap draws 1000, repeats 5, fitting 39.6 min (sum over splits), whole script 3.2 min.

Every row is the refitted model minus the full model, on the same bootstrap draw (95 % CI of the difference). The CIs cover patient sampling of the test folds given the fitted models; refit randomness is averaged over the splits, not resampled — the reference row 'all inputs, another random seed' shows its size. EO = equalized-odds difference with the audit's rules; with no disparity at all the audit's permutation null for age_band is 0.030. Confirmatory family: Δ EO of age_band at top q % for the 10 refitted arms, each with an unadjusted 95 % CI; everything else is descriptive.

Removing inputs also costs ROC-AUC, and a weaker ranking can shrink a gap by moving towards random flags (or widen one, as a model left with little but age does). The column 'EO age_band at the same ROC-AUC loss, random' is the gap the full model's ranking shows after losing the arm's ROC-AUC to random noise in every test fold (mean of 5 noise draws, their range in brackets). It is a descriptive reference, not a test: an arm well below its whole range removed inputs that carry the gap, not only predictive signal. On folds where an arm ranks no worse than the full model the reference keeps the full ranking, so for arms with no ROC-AUC loss it is the full model's gap (largest shortfall of the noise from a fold's target ROC-AUC: 0.0023). Base rates and calibration by group: `approved_fairness.md` §1 and §4.

Shown: top q % in §1–§3 and §5, point estimates for the other shares in §6; §3 for age_band, §5 for age_band and sex. In `sources.json` only: Δ EO for every attribute and rule, group rates and ranking for every attribute at top q %, per-class proxy CIs.

Feature blocks (every model input in exactly one block, fixed before the results):

| Block | Inputs |
|---|---|
| demographics | age, sex, race |
| prior_use | number_outpatient, number_emergency, number_inpatient |
| current_stay | time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_diagnoses |
| admission | admission_type, admission_source, discharge_group, medical_specialty, payer_code |
| diagnosis | diag_1_group |
| diabetes_care | a1c_result, max_glu_serum, insulin, metformin, med_change, prescription_recorded |

## 1. Feature-set ablation: leave one block out

| Model | Δ ROC-AUC | Δ accuracy | EO age_band (full → arm) | Δ EO age_band | EO age_band at the same ROC-AUC loss, random: mean (min–max) | Δ EO sex | Δ EO race |
|---|---|---|---|---|---|---|---|
| without demographics (age, sex, race) | -0.0032 [-0.0045, -0.0020] | -0.0005 [-0.0012, +0.0002] | 0.257 → 0.188 | -0.069 [-0.086, -0.051] | 0.252 (0.245–0.255) | +0.010 [+0.003, +0.015] | -0.034 [-0.064, +0.025] |
| without prior use (visits in the year before) | -0.0450 [-0.0495, -0.0402] | -0.0107 [-0.0124, -0.0090] | 0.257 → 0.080 | -0.177 [-0.236, -0.107] | 0.164 (0.158–0.168) | +0.012 [-0.007, +0.027] | -0.018 [-0.143, +0.058] |
| without current stay (length, labs, procedures, drugs, number of diagnoses) | -0.0080 [-0.0103, -0.0059] | -0.0008 [-0.0017, +0.0002] | 0.257 → 0.257 | +0.000 [-0.018, +0.018] | 0.236 (0.231–0.241) | -0.001 [-0.008, +0.006] | -0.012 [-0.061, +0.046] |
| without admission / discharge / specialty / payer | -0.0302 [-0.0339, -0.0266] | -0.0051 [-0.0064, -0.0038] | 0.257 → 0.303 | +0.046 [+0.025, +0.072] | 0.196 (0.184–0.204) | -0.014 [-0.024, +0.016] | -0.010 [-0.072, +0.068] |
| without primary-diagnosis group | -0.0033 [-0.0045, -0.0021] | -0.0004 [-0.0010, +0.0003] | 0.257 → 0.248 | -0.009 [-0.021, +0.004] | 0.246 (0.236–0.252) | +0.002 [-0.003, +0.007] | +0.004 [-0.044, +0.046] |
| without diabetes care (HbA1c, glucose, insulin, metformin, medication change, diabetes medication prescribed) | -0.0023 [-0.0040, -0.0006] | +0.0001 [-0.0010, +0.0011] | 0.257 → 0.264 | +0.007 [-0.011, +0.024] | 0.251 (0.249–0.255) | +0.002 [-0.004, +0.009] | -0.000 [-0.057, +0.040] |
| reference: all inputs, another random seed | -0.0002 [-0.0008, +0.0003] | +0.0001 [-0.0004, +0.0005] | 0.257 → 0.255 | -0.002 [-0.011, +0.007] | 0.258 (0.256–0.261) | +0.003 [-0.002, +0.006] | +0.006 [-0.020, +0.025] |

## 2. Removal of the sensitive attributes (unawareness)

age_band is not a model input; 'without age' removes the age it is derived from.

| Model | Δ ROC-AUC | Δ accuracy | EO age_band (full → arm) | Δ EO age_band | EO age_band at the same ROC-AUC loss, random: mean (min–max) | Δ EO sex | Δ EO race |
|---|---|---|---|---|---|---|---|
| without sex | -0.0005 [-0.0013, +0.0002] | +0.0001 [-0.0005, +0.0006] | 0.257 → 0.258 | +0.001 [-0.010, +0.011] | 0.255 (0.251–0.259) | +0.008 [+0.003, +0.012] | +0.003 [-0.029, +0.021] |
| without race | -0.0009 [-0.0016, -0.0002] | -0.0003 [-0.0008, +0.0003] | 0.257 → 0.260 | +0.003 [-0.005, +0.013] | 0.255 (0.253–0.256) | +0.004 [-0.001, +0.007] | -0.026 [-0.052, +0.026] |
| without age | -0.0027 [-0.0037, -0.0017] | -0.0004 [-0.0011, +0.0001] | 0.257 → 0.187 | -0.070 [-0.086, -0.052] | 0.254 (0.249–0.257) | +0.003 [-0.002, +0.007] | -0.018 [-0.043, +0.028] |
| without demographics (age, sex, race) | -0.0032 [-0.0045, -0.0020] | -0.0005 [-0.0012, +0.0002] | 0.257 → 0.188 | -0.069 [-0.086, -0.051] | 0.252 (0.245–0.255) | +0.010 [+0.003, +0.015] | -0.034 [-0.064, +0.025] |
| reference: all inputs, another random seed | -0.0002 [-0.0008, +0.0003] | +0.0001 [-0.0004, +0.0005] | 0.257 → 0.255 | -0.002 [-0.011, +0.007] | 0.258 (0.256–0.261) | +0.003 [-0.002, +0.006] | +0.006 [-0.020, +0.025] |

A model on age, sex and race alone reaches ROC-AUC 0.520 (full model 0.665); its age_band gap (0.346) is not interpreted: such a model can only rank whole demographic cells.

## 3. Which age_band groups move (TPR / FPR at top q %)

A gap can close because the low group rises or because the high group falls (levelling down). Point estimates, repeat-averaged.

| Model | 20-39 | 40-59 | 60-79 | 80+ |
|---|---|---|---|---|
| full model | 0.468 / 0.133 | 0.276 / 0.088 | 0.233 / 0.095 | 0.211 / 0.102 |
| without demographics (age, sex, race) | 0.408 / 0.110 | 0.272 / 0.089 | 0.233 / 0.094 | 0.220 / 0.113 |
| without prior use (visits in the year before) | 0.245 / 0.118 | 0.165 / 0.072 | 0.210 / 0.108 | 0.230 / 0.130 |
| without current stay (length, labs, procedures, drugs, number of diagnoses) | 0.469 / 0.131 | 0.266 / 0.085 | 0.231 / 0.097 | 0.212 / 0.104 |
| without admission / discharge / specialty / payer | 0.497 / 0.150 | 0.260 / 0.093 | 0.199 / 0.093 | 0.195 / 0.109 |
| without primary-diagnosis group | 0.460 / 0.129 | 0.268 / 0.088 | 0.234 / 0.096 | 0.212 / 0.103 |
| without diabetes care (HbA1c, glucose, insulin, metformin, medication change, diabetes medication prescribed) | 0.469 / 0.137 | 0.276 / 0.089 | 0.236 / 0.095 | 0.205 / 0.101 |
| without sex | 0.470 / 0.131 | 0.276 / 0.088 | 0.233 / 0.095 | 0.212 / 0.102 |
| without race | 0.468 / 0.135 | 0.274 / 0.089 | 0.233 / 0.095 | 0.208 / 0.101 |
| without age | 0.410 / 0.109 | 0.273 / 0.088 | 0.232 / 0.094 | 0.223 / 0.112 |
| reference: all inputs, another random seed | 0.466 / 0.132 | 0.274 / 0.088 | 0.235 / 0.095 | 0.211 / 0.102 |

## 4. How well the other inputs predict each attribute (proxy check)

A hgb classifier (default settings) trained on the training fold predicts the attribute from every other model input (sex: all but sex; age band: all but age; race: all but race, race 'Unknown' left out). ROC-AUC one class vs the rest. 0.5 = the other inputs carry no trace of the attribute; the further above 0.5, the less removing the attribute removes its information (an untuned model gives a lower bound).

| Attribute | ROC-AUC (mean over classes) [95 % CI] | Per class |
|---|---|---|
| age_band | 0.773 [0.770, 0.775] | 20-39 0.867, 40-59 0.766, 60-79 0.662, 80+ 0.795 |
| sex | 0.626 [0.622, 0.630] | Female 0.626, Male 0.626 |
| race | 0.716 [0.709, 0.723] | AfricanAmerican 0.755, Asian 0.690, Caucasian 0.742, Hispanic 0.715, Other 0.678 |

## 5. Ranking within groups versus the shared cut-off

For each group of the full model at top q %: ROC-AUC within the group; its TPR and FPR under the shared rule; and its TPR at its own cut-off that gives it the overall FPR (ranking quality at the operating point). Reading: TPR − 'TPR at the overall FPR' is what the shared cut-off adds (+) or takes away (−) from the group because of where its scores lie; differences in 'TPR at the overall FPR' between groups are differences in ranking quality at the operating point.

**age_band**

| Group | Readmitted | Within-group ROC-AUC | TPR | FPR | TPR at the overall FPR | Shared cut-off effect | Mean risk, readmitted / not |
|---|---|---|---|---|---|---|---|
| 20-39 | 660 | 0.755 [0.729, 0.783] | 0.468 | 0.133 | 0.383 | +0.085 | 0.193 / 0.109 |
| 40-59 | 2,690 | 0.692 [0.681, 0.703] | 0.276 | 0.088 | 0.294 | -0.018 | 0.148 / 0.100 |
| 60-79 | 5,545 | 0.651 [0.643, 0.659] | 0.233 | 0.095 | 0.236 | -0.003 | 0.144 / 0.112 |
| 80+ | 2,371 | 0.616 [0.603, 0.628] | 0.211 | 0.102 | 0.201 | +0.010 | 0.145 / 0.124 |

**sex**

| Group | Readmitted | Within-group ROC-AUC | TPR | FPR | TPR at the overall FPR | Shared cut-off effect | Mean risk, readmitted / not |
|---|---|---|---|---|---|---|---|
| Female | 6,102 | 0.668 [0.660, 0.676] | 0.260 | 0.100 | 0.253 | +0.006 | 0.150 / 0.112 |
| Male | 5,164 | 0.661 [0.652, 0.669] | 0.244 | 0.092 | 0.252 | -0.008 | 0.145 / 0.110 |

## 6. Ablation under the other capacities (sensitivity)

Δ EO age_band, refitted − full model, point estimates.

| Model | topq | q05 | q10 | q20 | q30 |
|---|---|---|---|---|---|
| without demographics (age, sex, race) | -0.069 | -0.045 | -0.072 | -0.035 | -0.010 |
| without prior use (visits in the year before) | -0.177 | -0.201 | -0.190 | -0.081 | +0.028 |
| without current stay (length, labs, procedures, drugs, number of diagnoses) | +0.000 | +0.001 | -0.003 | +0.004 | +0.017 |
| without admission / discharge / specialty / payer | +0.046 | +0.062 | +0.048 | +0.061 | +0.052 |
| without primary-diagnosis group | -0.009 | -0.008 | -0.010 | +0.000 | +0.004 |
| without diabetes care (HbA1c, glucose, insulin, metformin, medication change, diabetes medication prescribed) | +0.007 | +0.001 | +0.002 | +0.005 | -0.005 |
| without sex | +0.001 | +0.004 | +0.002 | +0.002 | +0.000 |
| without race | +0.003 | +0.005 | +0.005 | +0.007 | +0.001 |
| without age | -0.070 | -0.057 | -0.072 | -0.038 | -0.010 |
| age, sex and race only | +0.089 | +0.029 | +0.020 | +0.360 | +0.546 |
| reference: all inputs, another random seed | -0.002 | -0.003 | -0.005 | +0.003 | +0.005 |

† = more than 1 % of bootstrap draws undefined. ‡ = fewer than 50 readmitted encounters (TPR reported, not used in gaps).
