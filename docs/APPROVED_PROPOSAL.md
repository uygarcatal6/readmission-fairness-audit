# APPROVED PROPOSAL

This is the approved project proposal. It is the binding specification for the code, the notebook and the paper. Where the code disagrees, this file wins.

---

## 1. Proposal basics

Option: Option 2 – analysis of a dataset following CRISP-DM.

Problem: predicting which diabetic inpatients will be readmitted within 30 days, so that a hospital with limited follow-up capacity can prioritise them, and checking whether the model's errors differ by sex, age and race.

Dataset: Diabetes 130-US Hospitals for Years 1999–2008 (UCI Machine Learning Repository)
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
101,766 inpatient encounters from 130 US hospitals, 47 features (demographics, diagnoses, lab tests, medications, prior visits).

Preprocessing required:
- heavily missing features (weight ~97%; payer code and medical specialty roughly 40–50%);
- ICD-9 diagnosis codes that need to be grouped;
- exclusion of encounters ending in death or hospice discharge, with unknown sex, missing primary diagnosis or paediatric age;
- repeated encounters of the same patient, which require patient-level splits to avoid leakage.

## 1b. Project abstract

Project abstract:

> This project follows the CRISP-DM methodology to predict 30-day hospital readmission of diabetic inpatients, using the Diabetes 130-US Hospitals dataset (UCI; 101,766 encounters from 130 US hospitals, 1999–2008). The use case is a hospital with limited follow-up capacity that needs to prioritise high-risk patients. Preprocessing covers heavily missing features, grouping of ICD-9 diagnosis codes, exclusion of encounters ending in death or hospice discharge, and patient-level splitting to prevent leakage from repeated encounters. Decision tree, k-NN, logistic regression, random forest and gradient boosting models are compared under patient-level cross-validation and evaluated with ROC-AUC, PR-AUC and the Brier skill score, given the class imbalance (11.4% positive after cleaning). A fairness audit then compares TPR, FPR and PPV across sex, age band and race, and tests two mitigation methods: reweighing and equalized-odds post-processing.

Consistency check (23 Sep 2026): every statement in the abstract is covered by §2 and by the code; the test-ordering analysis (§2) and the MLP / bagging extensions are additions, not contradictions.

## 2. Methodology

Data and target: After excluding death/hospice discharges, unknown sex, missing primary diagnosis and the two paediatric age bands, the sample contains 98,470 encounters from 69,303 patients. The target is readmission within 30 days vs. everything else. 11.4% of encounters are positive, so the no-information rate is 0.886.

Validation: I use patient-level stratified group cross-validation with 5 folds, repeated 5 times with different shuffles. Encoding and scaling are fitted inside the training folds only. Confidence intervals come from a bootstrap over patients, not encounters.

Models: Decision tree with a pruning curve (max_depth, cost-complexity alpha), k-NN with a k sweep, logistic regression, random forest and gradient boosting. Depth and k are chosen with the one-standard-error rule on validation data. Test folds are never used for any choice.

Usability: A model counts as usable only if the lower confidence bound of (accuracy − 0.886) is above zero. I also report ROC-AUC, PR-AUC against prevalence and the Brier skill score. Operating point: in each test fold, the model flags the highest-risk q% of encounters, where q equals the training-fold readmission rate. This mimics a fixed follow-up capacity and uses no test labels. As sensitivity checks, I use q = 5/10/20/30%, an inner-CV Youden threshold and a class-weighted model at 0.5.

Fairness audit: By sex, age band (20–39, 40–59, 60–79, 80+) and race, I report TPR, FPR, PPV, selection rate, the equalized-odds difference and sex × age intersections. Groups with fewer than 50 positives (for TPR) or 50 negatives (for FPR) are reported as insufficient, not dropped. To find where disparities come from, I use feature-set ablation, removal of the sensitive attributes, how well the remaining features predict each attribute, and base rates and calibration by group.

Mitigations: Reweighing with Kamiran–Calders weights (compared with simple cell balancing), and ThresholdOptimizer with an equalized-odds constraint fitted on a separate patient-level validation split of each training fold. Each is compared with its unconstrained counterpart under the same decision rule, so the reported accuracy cost comes from the mitigation and not from a change of threshold. Both mitigations are run for all three attributes. The paper leads with the attribute that has the largest baseline gap, with sex for comparison.

Test ordering: Whether HbA1c and serum glucose were ordered is modelled on sex, age band, race and primary-diagnosis group. The models use patient-clustered standard errors and Benjamini–Hochberg correction, and are replicated on first encounters only.

---

## 3. Numbers in the proposal, checked against the raw data (23 Sep 2026)

`adapters/diabetes_uci.load(level="encounter")` then `age >= 20`, on `data/raw/diabetic_data.csv` (SHA-256 `0689e7ec…df97`, matches `data/download.py:40`):

| Claim | Measured | Status |
| :--- | :--- | :---: |
| 98,470 encounters | 98,470 | ✅ |
| 69,303 patients | 69,303 | ✅ |
| 11.4% positive | 0.1144 | ✅ |
| NIR 0.886 | 0.8856 | ✅ |

Positives per age band: 20–39 660 · 40–59 2,690 · 60–79 5,545 · 80+ 2,371. Positives per race: Caucasian 8,525 · AfricanAmerican 2,135 · Hispanic 210 · Unknown 188 · Other 143 · Asian 65. Consequence: in a single test fold (≈1/5), Hispanic, Unknown, Other and Asian fall below 50 positives, so group metrics are computed on the pooled out-of-fold predictions of each repeat, where every group clears the 50-positive gate.

---

## 4. Commitment ↔ code status

Legend: ✅ in the code as approved. Anchors verified 23 Sep 2026.

| # | Commitment | Status | Evidence / what changes |
| :---: | :--- | :---: | :--- |
| 1 | Exclusion funnel → 98,470 / 69,303 | ✅ | §3 above |
| 2 | Binary target, NIR 0.886 | ✅ | `readmission/cohort.py` (23 Sep); asserted in `tests/test_readmission_core.py` |
| 3 | Stratified group 5-fold | ✅ | `readmission/splits.py` (23 Sep); fold rates 0.1144–0.1145; works around a scikit-learn shuffle bug |
| 4 | 5 repeats with different shuffles | ✅ | `readmission/evaluate.py` outer loop, resumable checkpoints (23 Sep) |
| 5 | Encoding inside training folds | ✅ | `readmission/preprocess.py` (23 Sep); rare-level pooling also moved inside the fold |
| 6 | Patient bootstrap CIs | ✅ | `readmission/metrics.py`: patient-draw weights, same draw across repeats, NaN draws counted |
| 7 | Decision tree pruning curve | ✅ | depth by one-SE per split; max_depth and ccp_alpha curves in `evaluate.complexity_curves` |
| 8 | k-NN k sweep | ✅ | `tuning.knn_k_scores` (one neighbour search), grid 1–1501, grid-edge flag |
| 9 | LR, RF, gradient boosting | ✅ | `readmission/models.py` (HistGradientBoosting, early stopping off) |
| 10 | 1-SE rule on inner validation | ✅ | `tuning.one_se_choice`, inner 5-fold ROC-AUC |
| 11 | Test folds never used for choices | ✅ | q, depth, k, Youden thresholds and encoding all from the training fold (independent review, 23 Sep) |
| 12 | Gate: lower CI of (acc − 0.886) > 0 | ✅ | fixed NIR of the cohort (0.8856); per-draw version reported as secondary |
| 13 | ROC-AUC | ✅ | weighted, tie-aware, checked against scikit-learn |
| 14 | PR-AUC vs prevalence | ✅ | average precision, prevalence in the table header |
| 15 | Brier skill score | ✅ | reference = training-fold base rate |
| 16 | Top-q% rule, q = training-fold rate | ✅ | `decision.flag_top_q` (exact k, seeded tie-break) |
| 17 | q = 5/10/20/30% | ✅ | `evaluate.Q_RULES` |
| 18 | Inner-CV Youden threshold | ✅ | inner out-of-fold scores of every approved model |
| 19 | Class-weighted model at 0.5 | ✅ | logreg, tree, rf, hgb (`class_weight='balanced'`) |
| 20 | Age bands 20–39/40–59/60–79/80+ | ✅ | `readmission/cohort.py::age_band` (23 Sep) |
| 21 | TPR, FPR, PPV, EO difference | ✅ | `readmission/fairness.py` (23 Sep): pooled out-of-fold rates per repeat, patient-bootstrap CIs, permutation null, signed fixed-pair differences |
| 22 | Selection rate | ✅ | `metrics.weighted_decision` (overall; per group in the fairness step) |
| 23 | Sex × age intersections | ✅ | `fairness.ATTRIBUTES` includes `sex_x_age` with the approved bands and the same gates |
| 24 | Insufficient if <50 positives / <50 negatives, not dropped | ✅ | `fairness.MIN_POSITIVES/MIN_NEGATIVES` (+ PPV < 50 flagged); marked ‡, kept in tables, left out of gaps |
| 25 | Feature-set ablation | ✅ | `readmission/sources.py`: leave-one-block-out over six feature blocks, refit on the 25 splits, paired Δ against the audited model, seed-control arm and matched ROC-AUC-loss reference (`source_stats.matched_degradation`) |
| 26 | Removal of sensitive attributes | ✅ | `sources.UNAWARE`: without sex / race / age, and all three together |
| 27 | Proxy check: features → attribute | ✅ | `sources.py` proxy models (HistGradientBoosting, training fold) + `source_stats.proxy_auc` (one-vs-rest ROC-AUC, patient-bootstrap CI) |
| 28 | Base rates by group | ✅ | `fairness.audit` groups → base_rate with CI |
| 29 | Calibration by group | ✅ | `fairness.audit` calibration: observed − predicted with CI, ECE next to the ECE of a perfectly calibrated model of the same size |
| 30 | K&C vs cell reweighing | ✅ | `readmission/mitigation.py::reweigh_weights` (both modes, own arms, paired K&C − cell difference) |
| 31 | ThresholdOptimizer (EO) on a patient-level validation split | ✅ | fairlearn TO fitted on the 20 % patient-level validation part of each training fold (`splits.validation_split`); plus EO post-processing at the capacity rule (§5) |
| 32 | Compared with unconstrained model under the same decision rule | ✅ | `mitigation.comparisons` + `paired_effects`: same model, same rule / same number flagged, differences on the same bootstrap draw |
| 33 | Both mitigations for all three attributes | ✅ | sex, age band, race; leading attribute (age band) first in `outputs/reports/approved_mitigation.md` |
| 34 | Test-ordering logit on sex, age band, race, diagnosis group + BH | ✅ | `readmission/ordering.py` (age band, reference = largest group, BH over the sex/age/race terms of both outcomes) |
| 35 | Patient-clustered standard errors | ✅ | `ordering.fit`: statsmodels `cov_type="cluster"` by patient; model-based SEs kept for comparison |
| 36 | Replicated on first encounters | ✅ | `ordering.analyse` on `load_cohort(first_encounter_only=True)`, own BH family, same references; agreement line in the report |

All 36 commitments are in the code as approved (✅ 36 of 36, 23 Sep 2026).

---

## 5. Choices the approved text leaves open (written down before the results they affect)

| Choice | What was done | Why | Where |
| :--- | :--- | :--- | :--- |
| Primary model | highest ROC-AUC point estimate among the approved models (rf, 0.665) | a rule, not a judgement after seeing fairness results | `fairness.choose_primary` |
| Leading attribute | largest EO difference **in excess of its permutation null** (amended 23 Sep after an independent review, before the full results; the raw-range rule gives the same answer, age band) | a max − min range grows with the number of small groups | `fairness.leading_attribute` |
| PPV gate | PPV reported as insufficient if < 50 flagged | same logic as the TPR/FPR gates | `fairness.MIN_FLAGGED` |
| ThresholdOptimizer objective | balanced accuracy; the unconstrained twin is the Youden threshold (its unconstrained optimum) | fairlearn's default (accuracy) flags nobody at 11 % prevalence (max selection 0.0023 in the full run) | `mitigation.TO_OBJECTIVES` |
| Groups for fitting the post-processors | levels with < 50 validation positives, and race "Unknown", pooled (race: Asian, Hispanic, Other, Unknown) | otherwise one level with 11 validation positives set the race constraint (review of split r0 f0, before the full run) | `mitigation.fit_groups` |
| EO post-processing at the capacity rule (addition) | Hardt et al. 2016 solved on the line "q % flagged", compared with the same model flagging the same number | TO at balanced accuracy flags ~36 %, not the q % the hospital can follow up; the capacity rule is the protocol's main operating point | `mitigation.fit_eo_at_capacity` |
| Source analysis: feature blocks | six blocks covering the 23 inputs once (demographics, prior use, current stay, admission, diagnosis, diabetes care); leave-one-block-out | fixed before the results; one refit per block keeps the table readable | `sources.FEATURE_BLOCKS` |
| Source analysis: references | seed-control arm (all inputs, another random_state) and the gap the full ranking shows after losing the same ROC-AUC to random noise | removing inputs lowers ROC-AUC, and a weaker ranking can shrink or widen a gap by itself (independent review, before the full results) | `sources.SEED_OFFSET`, `source_stats.matched_degradation` |
| Source analysis: proxy model | HistGradientBoosting (default settings), one-vs-rest ROC-AUC; race 'Unknown' left out | an approved model; untuned, so a lower bound on the trace an attribute leaves | `sources.PROXY_MODEL` |
| Test ordering: outcome wording | "result recorded" (the dataset's proxy for "ordered") | the fields hold a result range or 'None' | `ordering.OUTCOMES` |
| Test ordering: references and family | reference = largest group; BH over the 16 sex/age/race terms of both outcomes (diagnosis terms and race 'Unknown' excluded); a separate family for the first-encounter replication | most precise contrasts; the diagnosis terms are adjustment, not hypotheses | `ordering.reference_levels`, `ordering.correct` |
| Test ordering: source sensitivity (addition) | the approved model plus an 'admission type unknown' indicator, reported next to it, not tested | 79.7 % of glucose results come from the 10.2 % of encounters with unknown admission type; glucose race odds ratios move towards 1 with it (independent review) | `ordering.recording_pattern`, `analyse(...)["source_adjusted"]` |

