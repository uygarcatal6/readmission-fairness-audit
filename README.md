# Fairness audit of hospital-readmission prediction (UCI Diabetes 130-US Hospitals)

Individual assignment for *Machine Learning and Data Mining* (University of Bologna, Cesena, A.Y. 2026/27, 6 CFU) — **Option 2: analysis of a dataset following CRISP-DM**.

**Question.** Can routinely collected admission data predict 30-day hospital readmission, and are the resulting errors distributed equally across gender, age and race? If not, where does the disparity come from (data, model, or measurement), and which mitigation is defensible at what cost in accuracy?

**Data.** UCI Machine Learning Repository, *Diabetes 130-US Hospitals for Years 1999–2008* (id 296; 101,766 inpatient encounters from 71,518 patients, 50 attributes; CC BY 4.0; Strack et al., 2014). Public, real, de-identified. Provenance, licence, download and a column dictionary: `data/README_DATA.md`; data-understanding profile: `data/profile.md`.

## Method (CRISP-DM, approved protocol)

The binding method is the proposal approved on 23 Sep 2026, `docs/APPROVED_PROPOSAL.md`; its §5 lists the choices the approved text leaves open.

1. **Business understanding** — a hospital with limited follow-up capacity flags the highest-risk q % of discharges (q = the training-fold readmission rate) and wants the flag's errors to be equal across sex, age band and race.
2. **Data understanding** — raw data: 101,766 encounters, 71,518 patients; `weight` is almost entirely missing and not used; payer code and medical specialty are about half missing (`data/profile.md`).
3. **Data preparation** — exclusion of death/hospice discharges, unknown sex, missing primary diagnosis and ages under 20; binary target "readmitted within 30 days"; ICD-9 grouping; one-hot encoding and scaling fitted inside each training fold (`adapters/diabetes_uci.py`, `readmission/cohort.py`, `readmission/preprocess.py`).
4. **Modelling** — decision tree (pruning curves), k-NN (k sweep), logistic regression, random forest and gradient boosting (+ MLP and bagging as extensions); depth and k by the one-standard-error rule; patient-level stratified group 5-fold cross-validation, repeated 5 times (`readmission/splits.py`, `models.py`, `tuning.py`, `evaluate.py`).
5. **Evaluation** — usability gate (lower 95 % CI of accuracy − no-information rate > 0), ROC-AUC, PR-AUC, Brier skill score with patient-bootstrap CIs; fairness audit (TPR, FPR, PPV, selection rate, equalized-odds difference, sex × age, permutation null); sources of the disparity (feature-block ablation, removal of the sensitive attributes, proxy check, ranking within groups); mitigations (Kamiran–Calders vs cell reweighing, equalized-odds post-processing); HbA1c / glucose recording model (patient-clustered SEs, Benjamini–Hochberg, first-encounter replication).
6. **Deployment** — the decision and its limits: `outputs/reports/approved_*.md`, the notebook's §6 and the paper.

## Repository layout

```
readmission/      approved-protocol package (cohort, splits, preprocess, models, tuning, evaluate,
                  decision, metrics, fairness, mitigation, sources, source_stats, ordering, figures, runtime)
adapters/         diabetes_uci.py: reads the UCI CSV, recodes the columns, applies the row exclusions
scripts/          run_{approved,fairness,mitigation,sources,ordering}.py (+ make_figures.py)
notebooks/        readmission_fairness_audit.ipynb (Colab)
paper/            four-page IEEE paper (LaTeX) and its figures
slides/           10-minute talk (deck, PDF, build script and figures)
outputs/          reports/approved_*.md (results) · approved/*.json (summary statistics; checkpoints are git-ignored)
tests/            pytest suite (test_readmission_*.py for the approved protocol, test_adapter_diabetes_uci.py for the loader)
data/             download/verify script, profile, licence (raw CSV is git-ignored)
docs/             APPROVED_PROPOSAL (the approved method) · MODULE_MAP · KNOWN_ISSUES
```

Per-module description and course-topic mapping: `docs/MODULE_MAP.md`.

## Run

Python 3.11+ (developed on 3.13). Everything runs on CPU.

Approved protocol (`readmission/`, method in `docs/APPROVED_PROPOSAL.md`), in this order; each fitting script resumes per outer split:

```bash
pip install -r requirements.txt
python data/download.py                        # UCI archive → data/raw/, sha256-verified
python scripts/run_approved.py                 # 5×5 patient-level CV, all models, top-q rule → approved_main.md
python scripts/run_fairness.py                 # equalized-odds audit + permutation null → approved_fairness.md
python scripts/run_mitigation.py               # reweighing, post-processing → approved_mitigation.md
python scripts/run_sources.py                  # ablation, unawareness, proxy check → approved_sources.md
python scripts/run_ordering.py                 # HbA1c / glucose recording model → approved_ordering.md
python -m pytest -q                            # 86 approved-protocol tests + 28 loader tests
```

Notebook: `notebooks/readmission_fairness_audit.ipynb` (Google Colab: *Runtime → Run all*) fits one outer split live and shows every table and figure from the saved summary JSONs in `outputs/approved/`; `python scripts/make_figures.py` writes the same figures to `paper/figures/`. Open it in Colab: <https://colab.research.google.com/github/uygarcatal6/readmission-fairness-audit/blob/main/notebooks/readmission_fairness_audit.ipynb>. The notebook clones the repository; alternatively, download it on GitHub (*Code → Download ZIP*) and in Colab either drop the zip into `/content` (Files pane) before *Runtime → Run all*, or upload it when the setup cell asks; the cell unpacks it and continues. The setup cell downloads the UCI data from archive.ics.uci.edu and stops if that site is unreachable.

CPU threads: every script takes `--n-jobs N` (or the environment variable `READMISSION_N_JOBS`); the default is half of the logical CPUs, at least 2 (`readmission/runtime.py`). The thread count changes run time, not the reported results. `--quick` (the first four scripts) gives a 1-repeat, 200-draw check run.

## Provenance

All code in this repository was written for this project, and no data or results from other work are used. The approved proposal (23 Sep 2026) replaced an earlier three-class design; that earlier code is not part of this repository.

## Use of AI assistants

The author chose the topic and the dataset within the course scope, set the research question and designed the method, which the professor approved. The AI assistant Claude (Anthropic; Opus 5, Opus 5.5 and Fable 5.1) was used, under the author's direction, to write and refactor code, run the analyses, review the code and the results independently, draft the documentation, the paper and the slides, and look up literature. The author reviewed and checked every analysis and result, can explain every line and remains responsible for every part of the submission (course rule: "You must understand and defend your code").

## References

Strack B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*, 781670. doi:10.1155/2014/781670 · Kamiran F., Calders T. (2012). Data preprocessing techniques for classification without discrimination. *Knowl. Inf. Syst.* 33(1):1–33 · Hardt M., Price E., Srebro N. (2016). Equality of opportunity in supervised learning. *NeurIPS* · Mitchell M. et al. (2019). Model cards for model reporting. *FAT\** · Gebru T. et al. (2021). Datasheets for datasets. *CACM* 64(12) · Weerts H. et al. (2023). Fairlearn: assessing and improving fairness of AI systems. arXiv:2303.16626.

## License

MIT (see `LICENSE`). Dataset: CC BY 4.0 (see `data/LICENSE_CC-BY-4.0.md`).
