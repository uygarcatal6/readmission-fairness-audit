# Known issues

Open defects and limitations of the code in this repository. None of them changes a reported
number.

## 1. `readmission/mitigation.py` fingerprint does not record `fairness.NOT_A_GROUP`

- **What:** the post-processors' fitting groups (`mitigation.fit_groups`) read
  `fairness.NOT_A_GROUP` (race "Unknown" is pooled), but the mitigation fingerprint hashes only
  `mitigation.py`, the main run's prediction code and the library versions. If `NOT_A_GROUP`
  changed, existing mitigation checkpoints would still load. `readmission/sources.py` records it.
- **Status:** *open, no effect on the current results* (NOT_A_GROUP has not changed since the
  mitigation run). Adding it would change `mitigation.py`'s hash and force the 25-split refit
  (~21 min); do it together with the next change that needs a refit anyway.

## 2. Checkpoint fingerprints hash raw file bytes, so line endings matter

- **What:** `evaluate.fingerprint` (and the mitigation / sources fingerprints) hash the raw
  bytes of the fingerprinted `.py` files. In the working copy that produced the saved
  checkpoints, `readmission/cohort.py`, `splits.py` and `preprocess.py` have LF line endings and
  the other six fingerprinted files CRLF. A fresh checkout with `core.autocrlf=true` (all CRLF)
  or a Colab clone (all LF) computes another `code_sha256` and refuses those checkpoints, although
  the code is the same.
- **Status:** *open, no effect on any result.* Checkpoints are git-ignored, so a fresh clone
  refits from scratch and writes its own fingerprint. The fix (hash the text with `\r\n` normalised to `\n`) changes
  `evaluate.py` and forces the full refit (about 2 h at 12 threads); do it together with #1.

## 3. Thread count and last digits

- **What:** the number of BLAS threads changes how sums of non-integer floats are split, so the
  logistic regression / MLP fits (~1e-13), the float32 calibration sums in `fairness.py`
  (~1e-8) and the ordering Logit (~1e-14) move in their last digits between thread counts.
  Flags, the integer-weight bootstrap sums and every number printed in the reports are the same
  at 2 and 12 threads (checked 24 Sep 2026). The saved outputs were made with 12 threads.
- **Status:** *by design, documented in `readmission/runtime.py`.* `--n-jobs 12` reruns under the
  saved condition; the default (half the CPUs) may rewrite those JSON leaves in their last digits.

## 4. Loader output that the approved protocol does not use

- **What:** `adapters/diabetes_uci.load` returns four columns that `readmission/cohort.py` does
  not read: `clinical_diagnosis` (the three-level `readmitted` label: `readmit_lt30` /
  `readmit_gt30` / `no_readmit`), `icd10` (the raw `diag_1` code; the codes are ICD-9 despite the
  name), `first_visit_date` (2000-01-01 plus the encounter's rank in minutes, an ordering key,
  not a date) and `imaging_ordered` (constant False). It codes sex as `K` (female) and `E`
  (male), which `readmission/cohort.py` maps to Female / Male, and `patient_id` is `"D"` +
  `patient_nbr`.
- **Status:** *by design.* `readmission/cohort.py` reads the sex codes and the `patient_id`
  strings enter the checkpoint fingerprint (`evaluate.fingerprint`, `rows_sha256`), so both
  stay as they are (the single-letter sex codes date from the loader's first version). The
  four unused columns are kept so that `load` returns the same DataFrame
  as when the saved results were made.
