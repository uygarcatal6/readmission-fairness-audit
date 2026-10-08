# Module map

One-to-two lines per module: what goes in, what comes out, what it does, and which
Machine Learning & Data Mining (ML&DM) course topic it illustrates. The package is a
flat layout so it runs unchanged in Colab or under `pytest` from the repo root.

## Approved protocol — `readmission/`

Implements `docs/APPROVED_PROPOSAL.md` step by step.

### `readmission/cohort.py`
- **In:** `data/raw/diabetic_data.csv`. **Out:** one row per encounter: 23 features, `y`, `patient_id`, `sex`/`age_band`/`race`, test-ordering flags; optional exclusion funnel.
- **Does:** adapter exclusions, drops the two paediatric bands, binary target (<30 vs rest), bands 20–39/40–59/60–79/80+; `first_encounter_only=True` keeps each patient's earliest encounter.
- **Why:** 98,470 / 69,303 / 11.44 % / NIR 0.886 are asserted in `tests/test_readmission_core.py`.
- **Course topic:** CRISP-DM data preparation; class imbalance (NIR as the baseline).

### `readmission/splits.py`
- **Does:** 5 folds × 5 repeats of patient-level stratified group K-fold; inner K folds and a single validation hold-out drawn from a training fold only.
- **Why not `StratifiedGroupKFold(shuffle=True)`:** in scikit-learn 1.6.1 it balances on another patient's labels (`_split.py:1042` vs `:1058`), fold rates 0.113–0.119. Random patient codes + `shuffle=False` gives 0.1144–0.1145 on all 25 folds.
- **Course topic:** cross-validation, leakage (patients, not rows, are the unit), model selection without the test set.

### `readmission/preprocess.py`
- **Does:** unfitted `ColumnTransformer` (StandardScaler + OneHotEncoder with `min_frequency=0.005`), always the first Pipeline step, so it is fitted on the training fold only.
- **Why:** encoding and rare-level pooling are learned on the training fold only; the loader's whole-file rare-level collapse is switched off for this path (`adapters/diabetes_uci.py`, `min_category_n=1`).
- **Course topic:** "Avoiding data leakage", "Use pipelines" (03-knn s.22–23).

### `readmission/decision.py`
- **Does:** `flag_top_q` (flag the top q % of risk scores: exactly k = round(q·n) encounters, ties at the cut-off broken at random with a required, seeded `rng`; contract in `tests/test_readmission_decision.py`) and `youden_threshold` (max TPR − FPR on inner-CV predictions).
- **Why:** q = training-fold readmission rate models fixed follow-up capacity and reads no test labels.

### `readmission/models.py`
- **Does:** one factory for the five approved models (logistic regression, decision tree, k-NN, random forest 300 trees, HistGradientBoosting) and two course-topic extensions (MLP (64, 32), bagged trees); class-weighted variants for the models that accept `class_weight`.
- **Why:** every run uses the same written-down settings; extensions never replace an approved model.
- **Course topic:** 02-dt, 03-knn, 04-neuralnetworks, 05-ensemble.

### `readmission/tuning.py`
- **Does:** one-SE rule (`one_se_choice`), tree-depth scores, k-NN scores for all k from one neighbour search, train-vs-validation complexity curves (max_depth and cost-complexity alpha).
- **Why:** "Depth and k are chosen with the one-standard-error rule on validation data" — the simplest candidate within one SE of the best inner-CV ROC-AUC (Occam's razor, 02-dt).

### `readmission/evaluate.py`
- **Does:** the outer loop — per split: q from the training fold, inner 5-fold tuning and Youden thresholds, final fits, decision rules (top q %, q = 5/10/20/30 %, Youden, class-weighted at 0.5); one resumable checkpoint per split; `load_oof` stacks them per repeat; `complexity_curves` for the pruning figure.
- **Why:** test folds are never used for a choice; a crash loses at most one split.

### `readmission/runtime.py`
- **Does:** `cap_threads(n_jobs)` — one knob for the CPU threads of a run: sets `LOKY_MAX_CPU_COUNT` (what joblib's `n_jobs=-1` in `models.py`/`tuning.py` resolves to) and caps the OpenMP and BLAS pools with threadpoolctl. `--n-jobs` flag of every run script > `READMISSION_N_JOBS` > default half the logical CPUs, at least 2.
- **Why:** `n_jobs=-1` occupies every CPU of a shared laptop; Colab's free tier has 2 vCPUs. The thread count changes run time, not results, so it is kept out of `Config` and every fingerprint: changing it never invalidates checkpoints (last-digit effects of BLAS: `docs/KNOWN_ISSUES.md` #3).

### `readmission/metrics.py`
- **Does:** weighted ROC-AUC / average precision (ties handled, checked against scikit-learn), Brier and Brier skill score, confusion-matrix metrics, patient-bootstrap weights, `summarize` (mean over repeats, 95 % percentile CI, usability gate = lower CI of accuracy − NIR > 0).
- **Why:** "Confidence intervals come from a bootstrap over patients, not encounters."
- **Speed (24 Sep 2026, results bit-identical):** `weighted_auc` in Mann-Whitney form (share of positive–negative pairs ranked correctly, ties ½) from one running sum over the sorted negatives (`_cumulative_weight`); average precision reads cumulative TP/FP from the same running sums; the four confusion counts in one matrix product; the reference Brier once per repeat. Exact because bootstrap weights are integers with sums far below 2**24.

### `readmission/fairness.py`
- **Does:** per group (sex, age band, race, sex × age): TPR, FPR, PPV, selection rate, base rate; equalized-odds difference per attribute; calibration-in-the-large and ECE per group; all with patient-bootstrap CIs on pooled out-of-fold predictions.
- **Why:** insufficient groups (< 50 positives for TPR, < 50 negatives for FPR, < 50 flagged for PPV) are reported and marked, never dropped, and left out of the gaps. Primary model and leading attribute are chosen by rules written down before the results (docstring).
- **Speed (24 Sep 2026, results bit-identical):** `_rates` multiplies only the flagged encounters (fp = flagged − tp); the permutation null works at patient level — a shuffle moves whole patients, so group counts are re-summed from per-patient counts with `np.bincount` instead of rebuilding the group-mask matrix 200 times (same `rng.permutation` sequence).
- **Course topic:** Module 2, "Bias and Fairness evaluation".

### `readmission/mitigation.py`
- **Does:** the two mitigations on the primary model, all three attributes, same 25 outer splits. Reweighing: refit on the whole training fold with Kamiran–Calders weights P(s)P(y)/P(s,y) or cell-balancing weights N/(cells·n(s,y)), flagged with the same top-q rule as the baseline. Post-processing on a model refit on 80 % of the training fold, thresholds fitted on the other 20 % (patient-level `splits.validation_split`): (a) fairlearn ThresholdOptimizer (equalized odds, objective balanced accuracy) vs one Youden threshold (its unconstrained optimum) and vs the same model flagging the same number; (b) `fit_eo_at_capacity`: Hardt et al. equalized-odds post-processing solved on the capacity line (ROC convex hull per group via `roc_hull`, common point on the lowest hull, `p_ignore` coin flip), vs the same model flagging the same number. `fit_groups` pools levels with < 50 validation positives and race 'Unknown' for fitting. `paired_effects` gives mitigated − counterpart differences with patient-bootstrap CIs computed on the same draw (accuracy, balanced accuracy, TPR, FPR, PPV, selection rate, ROC-AUC, EO difference of every attribute).
- **Why:** "Each is compared with its unconstrained counterpart under the same decision rule, so the reported accuracy cost comes from the mitigation and not from a change of threshold." The baseline arm is copied from the main run's checkpoint, so it is the audited baseline. fairlearn's default objective (accuracy) flags nobody at 11 % prevalence (acc − NIR > 0 needs PPV > 0.5; checked on real data: selection rate 0.000); it is kept as a diagnostic arm. Own checkpoint directory and fingerprint; never modifies the files hashed by the main run.
- **Course topic:** Module 2, "Bias and Fairness evaluation" (mitigation).

### `readmission/sources.py`
- **Does:** where the disparities come from, on the primary model and the same 25 outer splits: leave-one-block-out ablation over six feature blocks (demographics, prior use, current stay, admission, diagnosis, diabetes care), removal of each sensitive attribute (no_sex / no_race / no_age; all three = without demographics), a demographics-only model, the proxy check (HistGradientBoosting predicts each attribute from the other inputs; one-vs-rest ROC-AUC with patient-bootstrap CI) and ROC-AUC within each group (`group_ranking`, no refit).
- **Why:** "To find where disparities come from, I use feature-set ablation, removal of the sensitive attributes, how well the remaining features predict each attribute, and base rates and calibration by group" (the last two are in `fairness.py`). Differences against the main run's model use `mitigation.paired_effects` (same bootstrap draw).
- **Course topic:** feature selection / ablation (CRISP-DM modelling), Module 2 "Bias and Fairness evaluation" (proxies, fairness through unawareness).

### `readmission/source_stats.py`
- **Does:** the source-analysis statistics computed from the checkpoints (no fitting): `matched_degradation` (the gap the full ranking shows after losing an arm's ROC-AUC to random noise, per test fold), `group_rates` (TPR/FPR per group for any flag key), `proxy_auc`, `group_ranking` (within-group ROC-AUC, TPR/FPR under the shared rule, TPR at the group's own cut-off at the overall FPR).
- **Why separate:** the sources fingerprint hashes `sources.py` only, so fixing a statistic never forces the 25-split refit (same split as `metrics.py`/`fairness.py` vs the main run's prediction code).
- **Speed (24 Sep 2026, results bit-identical):** `matched_degradation` computes the base normal scores, their ROC-AUC and each arm's per-fold target once, uses `auc(roc_curve())` (the two calls `roc_auc_score` makes, without its repeated input checks) inside the bisection, and runs the (arm, draw) units in threads — each unit has its own seeded generators, so the order cannot change a draw; `group_ranking` computes the within-group AUC on the group's rows only.

### `readmission/ordering.py`
- **Does:** test-ordering equity: logistic regression of "HbA1c result recorded" and "serum glucose result recorded" on sex, age band, race and primary-diagnosis group (reference = largest group); patient-clustered SEs (model-based kept for comparison); Benjamini–Hochberg over the sex/age/race terms of both outcomes; replication on first encounters; crude rates with patient-bootstrap CIs; recording pattern of each field and a sensitivity fit with an "admission type unknown" indicator.
- **Why:** approved text, "Test ordering". The recording-pattern check was added after an independent review: 79.7 % of glucose results come from the 10.2 % of encounters with unknown admission type, so the glucose race gap is largely a source/period pattern.
- **Course topic:** Module 2 (bias in the data-generating process: measurement / recording bias), logistic regression inference.

### `scripts/run_ordering.py`
- **Does:** → `outputs/approved/ordering.json` + `outputs/reports/approved_ordering.md` (~0.2 min).

### `scripts/run_fairness.py`
- **Does:** audit of every approved model × decision rule → `outputs/approved*/fairness.json` + `outputs/reports/approved_fairness*.md`.

### `scripts/run_sources.py`
- **Does:** source-analysis run (resumable, `outputs/approved*/sources/`) → `outputs/approved*/sources.json` + `outputs/reports/approved_sources*.md`.

### `scripts/run_mitigation.py`
- **Does:** mitigation run (resumable, `outputs/approved*/mitigation/`) → `outputs/approved*/mitigation.json` + `outputs/reports/approved_mitigation*.md` (reweighing at capacity, TO vs single threshold, spill-over on the other attributes, group rates for the leading attribute and sex, sensitivity over q, usability gate).

### `scripts/run_approved.py`
- **Does:** end-to-end run (`--quick` = 1 repeat, 200 draws) → `outputs/approved*/summary.json` + `outputs/reports/approved_main*.md`.

### `scripts/make_figures.py`
- **Does:** writes the paper figures (`paper/figures/*.pdf`, `*.png`) from `outputs/approved/*.json` with `readmission/figures.py`; the notebook and the slides draw the same figures.

## Data loader — `adapters/`

### `adapters/diabetes_uci.py`
- **In:** the raw UCI "Diabetes 130-US Hospitals for Years 1999-2008" CSV (`data/raw/diabetic_data.csv`, 101,766 encounters × 50 columns).
- **Out:** one row per encounter with renamed and recoded columns (including the `hba1c_requested` / `glucose_requested` test-ordering flags) and a row-exclusion funnel.
- **Does:** drops unknown sex, death/hospice discharges and a missing primary diagnosis; groups the ICD-9 `diag_1` code into the nine groups of Strack et al. (2014); maps the id codes to labels. The mapping table is in the module docstring; `readmission/cohort.py` builds the analysis cohort from the output.
- **Course topic:** CRISP-DM data understanding and data preparation on a real public dataset.

## Tests

`tests/test_readmission_*.py` (86 tests) cover the approved protocol and `tests/test_adapter_diabetes_uci.py` (28 tests, on a synthetic UCI-shaped file) the loader. Tests that need the real CSV are skipped when it has not been downloaded (`tests/test_readmission_core.py`).
