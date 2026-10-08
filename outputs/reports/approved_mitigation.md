# Approved protocol — mitigations

Model: **rf** (the primary model of the audit). Leading attribute: **age_band**; sex is shown for comparison and every mitigation is run for all three attributes. Patient-bootstrap draws 1000, repeats 5, fitting 21.0 min (sum over splits), whole script 4.2 min.

Every mitigated arm is compared with its unconstrained twin under the same decision rule, on the same bootstrap draw, so each CI is that of the difference itself (mitigated − unconstrained). EO = equalized-odds difference of the attribute with the audit's rules (groups with < 50 positives / negatives and race 'Unknown' left out). At a fixed number flagged, accuracy, TPR, FPR and PPV all move with the number of true positives alone (Δ accuracy = 2 ΔTP / n), so the tables show Δ accuracy and Δ TPR.

## 1. Reweighing at the capacity rule (top q %, q = training-fold rate)

Refitted on the whole training fold with Kamiran–Calders weights (attribute and label independent in the weighted data) or cell balancing (equal weight per group × label cell); flagged with the same rule and the same number per fold as the baseline.

| Attribute | Weights | EO (baseline → reweighed) | Δ EO [95 % CI] | Δ accuracy | Δ TPR | Δ ROC-AUC |
|---|---|---|---|---|---|---|
| age_band | Kamiran–Calders | 0.257 → 0.293 | +0.036 [+0.026, +0.045] | -0.0002 [-0.0007, +0.0004] | -0.001 [-0.003, +0.001] | -0.0012 [-0.0019, -0.0005] |
| age_band | cell balancing | 0.257 → 0.146 | -0.111 [-0.137, -0.087] | -0.0010 [-0.0019, -0.0000] | -0.004 [-0.008, -0.000] | -0.0005 [-0.0019, +0.0011] |
| sex | Kamiran–Calders | 0.016 → 0.013 | -0.003 [-0.006, +0.002] | +0.0003 [-0.0001, +0.0008] | +0.001 [-0.000, +0.003] | +0.0003 [-0.0002, +0.0008] |
| sex | cell balancing | 0.016 → 0.019 | +0.003 [-0.005, +0.010] | -0.0008 [-0.0017, +0.0002] | -0.003 [-0.007, +0.000] | +0.0006 [-0.0008, +0.0020] |
| race | Kamiran–Calders | 0.081 → 0.119 | +0.038 [-0.018, +0.051] | +0.0000 [-0.0005, +0.0005] | +0.000 [-0.002, +0.002] | -0.0006 [-0.0012, +0.0000] |
| race | cell balancing | 0.081 → 0.187 | +0.106 [-0.012, +0.153] | -0.0012 [-0.0023, -0.0002] | -0.005 [-0.009, -0.001] | -0.0023 [-0.0040, -0.0006] |

Kamiran–Calders − cell balancing (same rule):

| Attribute | Δ EO | Δ accuracy | Δ ROC-AUC |
|---|---|---|---|
| age_band | +0.147 [+0.125, +0.170] | +0.0008 [-0.0002, +0.0018] | -0.0007 [-0.0023, +0.0009] |
| sex | -0.005 [-0.013, +0.003] | +0.0011 [+0.0002, +0.0020] | -0.0003 [-0.0017, +0.0011] |
| race | -0.068 [-0.135, +0.030] | +0.0012 [+0.0002, +0.0024] | +0.0017 [-0.0001, +0.0036] |

## 2. Equalized-odds post-processing at the capacity rule

The model is refitted on 80 % of the training fold; the other 20 % (patient-level validation part) chooses the thresholds, which are applied to the test fold. Mitigated: every group gets the same TPR and FPR on the validation part while q % are flagged, with the most true positives possible (Hardt et al. 2016, solved on the capacity line; groups get their own randomised thresholds). On the test folds it flags 0.115 on average (per fold 0.105–0.124, q = 0.1144); at PPV < 0.5 flagging fewer alone raises accuracy and flagging more lowers it, so the unconstrained twin is the same model flagging its top scores, exactly as many per fold. The last column compares with the plain top q % rule. Levels pooled for fitting (table in section 3) are equalized only as a pool, while the EO here is the audit's, level by level: part of a remaining gap on such an attribute is by design.

| Attribute | EO (twin → EO post-proc.) | Δ EO [95 % CI] | Δ accuracy | Δ TPR | Share flagged | Δ EO vs top q % |
|---|---|---|---|---|---|---|
| age_band | 0.250 → 0.018 | -0.232 [-0.269, -0.146] | -0.0090 [-0.0100, -0.0081] | -0.039 [-0.045, -0.034] | 0.115 | -0.232 [-0.269, -0.146] |
| sex | 0.015 → 0.011 | -0.004 [-0.028, +0.024] | -0.0021 [-0.0026, -0.0016] | -0.009 [-0.011, -0.007] | 0.115 | -0.004 [-0.028, +0.024] |
| race | 0.077 → 0.060 | -0.017 [-0.086, +0.016] | -0.0035 [-0.0040, -0.0028] | -0.015 [-0.017, -0.013] | 0.115 | -0.020 [-0.085, +0.014] |

## 3. fairlearn ThresholdOptimizer (approved text) vs one threshold for everyone

Same 80 % model and validation part. Objective balanced accuracy for both: the single threshold is its unconstrained optimum (Youden's J), ThresholdOptimizer its optimum under equalized odds. Both flag far more than the capacity q %, and not the same number, so their shared objective (balanced accuracy) is the first comparison; the last two columns compare TO with the same model flagging its top scores, exactly as many per fold (accuracy moves with the number flagged). Levels pooled for fitting (table in section 3) are equalized only as a pool, while the EO here is the audit's, level by level: part of a remaining gap on such an attribute is by design.

| Attribute | Δ balanced accuracy | EO (single → TO) | Δ EO [95 % CI] | Selection rate | Δ accuracy vs same-number twin | Δ EO vs same-number twin |
|---|---|---|---|---|---|---|
| age_band | -0.031 [-0.033, -0.028] | 0.188 → 0.018 | -0.170 [-0.181, -0.137] | 0.364 → 0.365 | -0.012 [-0.014, -0.011] | -0.169 [-0.179, -0.134] |
| sex | -0.005 [-0.006, -0.004] | 0.030 → 0.008 | -0.022 [-0.040, +0.007] | 0.364 → 0.354 | -0.002 [-0.003, -0.001] | -0.019 [-0.038, +0.007] |
| race | -0.010 [-0.011, -0.009] | 0.076 → 0.091 | +0.014 [-0.070, +0.047] | 0.364 → 0.389 | -0.004 [-0.004, -0.003] | -0.008 [-0.064, +0.033] |

With fairlearn's default objective (accuracy) the largest selection rate over the three attributes is 0.0023: flagging only adds accuracy where PPV > 0.5 (acc − NIR = (2 TP − k) / n), so the accuracy optimum flags almost nobody and meets equalized odds trivially.

Cost of holding out 20 % for the thresholds (top q %, 80 % model − full model): Δ accuracy -0.0004 [-0.0010, +0.0002], Δ ROC-AUC -0.0017 [-0.0025, -0.0010].

Groups used to fit the post-processors (rule fixed before the full run: levels with < 50 validation positives, and race 'Unknown', are pooled):

| Attribute | Merged level → fitted level (splits) | Fewest validation positives in a fitted group | Binding group (splits) |
|---|---|---|---|
| age_band | none | 81 | eocap: 80+ 21, 60-79 4; to: 80+ 23, 60-79 2 |
| sex | none | 765 | eocap: Female 8, Male 17; to: Female 8, Male 17 |
| race | Asian → pooled (25/25), Hispanic → pooled (24/25), Other → pooled (25/25), Unknown → pooled (25/25) | 50 | eocap: Caucasian 7, AfricanAmerican 14, pooled 4; to: AfricanAmerican 14, Caucasian 10, pooled 1 |

Post-processing flags are random (coin flips inside groups); they are fixed per split and averaged over the repeats, and the bootstrap does not redraw them.

## 4. Spill-over: Δ EO of every attribute (mitigated − same-number twin)

Mitigating one attribute can move the gaps of the others. Paired differences, 95 % CI.

| Arm | Δ EO age_band | Δ EO sex | Δ EO race |
|---|---|---|---|
| K&C on age_band | +0.036 [+0.026, +0.045] | -0.003 [-0.007, +0.003] | +0.017 [-0.027, +0.036] |
| cell on age_band | -0.111 [-0.137, -0.087] | +0.005 [-0.003, +0.013] | +0.017 [-0.068, +0.070] |
| EO at capacity on age_band | -0.232 [-0.269, -0.146] | -0.008 [-0.018, +0.001] | -0.014 [-0.075, +0.034] |
| TO on age_band | -0.169 [-0.179, -0.134] | -0.018 [-0.027, -0.010] | +0.017 [-0.043, +0.041] |
| K&C on sex | -0.006 [-0.015, +0.003] | -0.003 [-0.006, +0.002] | -0.001 [-0.017, +0.019] |
| cell on sex | -0.088 [-0.108, -0.065] | +0.003 [-0.005, +0.010] | +0.014 [-0.086, +0.091] |
| EO at capacity on sex | -0.021 [-0.032, -0.009] | -0.004 [-0.028, +0.024] | -0.003 [-0.028, +0.024] |
| TO on sex | -0.009 [-0.011, -0.007] | -0.019 [-0.038, +0.007] | +0.005 [-0.017, +0.021] |
| K&C on race | -0.001 [-0.010, +0.008] | +0.001 [-0.003, +0.005] | +0.038 [-0.018, +0.051] |
| cell on race | -0.124 [-0.146, -0.088] | +0.012 [+0.002, +0.019] | +0.106 [-0.012, +0.153] |
| EO at capacity on race | -0.036 [-0.051, -0.020] | -0.001 [-0.006, +0.003] | -0.017 [-0.086, +0.016] |
| TO on race | -0.015 [-0.017, -0.013] | -0.005 [-0.009, -0.001] | -0.008 [-0.064, +0.033] |

## 5.1 Group rates by age_band (TPR / FPR)

| Group | Readmitted | baseline | K&C | cell | top q % (80 %) | EO at capacity | single thr. | TO |
|---|---|---|---|---|---|---|---|---|
| 20-39 | 660 | 0.468 / 0.133 | 0.473 / 0.136 | 0.347 / 0.091 | 0.457 / 0.129 | 0.222 / 0.106 | 0.668 / 0.290 | 0.510 / 0.348 |
| 40-59 | 2,690 | 0.276 / 0.088 | 0.303 / 0.103 | 0.260 / 0.085 | 0.274 / 0.089 | 0.213 / 0.100 | 0.540 / 0.261 | 0.522 / 0.345 |
| 60-79 | 5,545 | 0.233 / 0.095 | 0.231 / 0.094 | 0.251 / 0.106 | 0.234 / 0.095 | 0.215 / 0.102 | 0.559 / 0.342 | 0.521 / 0.345 |
| 80+ | 2,371 | 0.211 / 0.102 | 0.180 / 0.084 | 0.201 / 0.095 | 0.207 / 0.103 | 0.204 / 0.103 | 0.606 / 0.449 | 0.504 / 0.347 |
| share flagged |  | 0.114 | 0.114 | 0.114 | 0.114 | 0.115 | 0.364 | 0.365 |

The first five columns flag about q % (capacity); the last two flag 0.364–0.365 (balanced-accuracy objective): compare within each set.

## 5.2 Group rates by sex (TPR / FPR)

| Group | Readmitted | baseline | K&C | cell | top q % (80 %) | EO at capacity | single thr. | TO |
|---|---|---|---|---|---|---|---|---|
| Female | 6,102 | 0.260 / 0.100 | 0.260 / 0.099 | 0.258 / 0.101 | 0.258 / 0.100 | 0.239 / 0.098 | 0.584 / 0.348 | 0.547 / 0.329 |
| Male | 5,164 | 0.244 / 0.092 | 0.247 / 0.094 | 0.239 / 0.092 | 0.243 / 0.093 | 0.249 / 0.099 | 0.554 / 0.325 | 0.556 / 0.327 |
| share flagged |  | 0.114 | 0.114 | 0.114 | 0.114 | 0.115 | 0.364 | 0.354 |

The first five columns flag about q % (capacity); the last two flag 0.354–0.364 (balanced-accuracy objective): compare within each set.

‡ fewer than 50 positives: TPR reported, left out of the gaps. (not a group) = missing-value code, left out of the gaps.

## 6. Reweighing under the other capacities (sensitivity)

Δ EO of the reweighed attribute / Δ accuracy, mitigated − baseline, point estimates.

| Attribute | Weights | topq | q05 | q10 | q20 | q30 |
|---|---|---|---|---|---|---|
| age_band | kc | +0.036 / -0.0002 | +0.021 / -0.0001 | +0.028 / -0.0001 | +0.042 / -0.0003 | +0.007 / +0.0002 |
| age_band | cell | -0.111 / -0.0010 | -0.094 / -0.0012 | -0.111 / -0.0011 | -0.084 / -0.0007 | +0.002 / -0.0000 |
| sex | kc | -0.003 / +0.0003 | -0.006 / -0.0001 | -0.007 / +0.0001 | -0.007 / -0.0003 | -0.006 / +0.0002 |
| sex | cell | +0.003 / -0.0008 | -0.002 / -0.0010 | -0.001 / -0.0008 | +0.010 / -0.0008 | +0.005 / +0.0000 |
| race | kc | +0.038 / +0.0000 | +0.006 / +0.0001 | +0.036 / +0.0001 | +0.024 / -0.0002 | +0.019 / -0.0003 |
| race | cell | +0.106 / -0.0012 | +0.043 / -0.0016 | +0.097 / -0.0013 | +0.145 / -0.0013 | +0.177 / -0.0008 |

## 7. Usability gate for the arms

Lower CI of (accuracy − 0.886) above zero = usable (docs/APPROVED_PROPOSAL.md).

| Arm | Share flagged | Accuracy | Accuracy − NIR | Usable |
|---|---|---|---|---|
| baseline, top q % | 0.114 | 0.829 [0.826, 0.832] | -0.057 [-0.060, -0.054] | no |
| top q %, 80 % model | 0.114 | 0.829 [0.826, 0.831] | -0.057 [-0.060, -0.054] | no |
| single threshold | 0.364 | 0.652 [0.649, 0.655] | -0.234 [-0.237, -0.231] | no |
| K&C on age_band | 0.114 | 0.829 [0.826, 0.832] | -0.057 [-0.060, -0.054] | no |
| cell on age_band | 0.114 | 0.828 [0.825, 0.831] | -0.058 [-0.060, -0.055] | no |
| EO at capacity on age_band | 0.115 | 0.819 [0.817, 0.822] | -0.066 [-0.069, -0.064] | no |
| TO on age_band | 0.365 | 0.639 [0.636, 0.641] | -0.247 [-0.249, -0.244] | no |
| K&C on sex | 0.114 | 0.829 [0.826, 0.832] | -0.056 [-0.059, -0.054] | no |
| cell on sex | 0.114 | 0.828 [0.825, 0.831] | -0.057 [-0.060, -0.055] | no |
| EO at capacity on sex | 0.115 | 0.826 [0.823, 0.829] | -0.060 [-0.062, -0.057] | no |
| TO on sex | 0.354 | 0.658 [0.655, 0.661] | -0.228 [-0.230, -0.225] | no |
| K&C on race | 0.114 | 0.829 [0.826, 0.832] | -0.057 [-0.059, -0.054] | no |
| cell on race | 0.114 | 0.828 [0.825, 0.831] | -0.058 [-0.061, -0.055] | no |
| EO at capacity on race | 0.115 | 0.825 [0.822, 0.827] | -0.061 [-0.064, -0.058] | no |
| TO on race | 0.389 | 0.629 [0.626, 0.632] | -0.257 [-0.260, -0.254] | no |
