"""
readmission — the approved protocol (docs/APPROVED_PROPOSAL.md), one module per step.

cohort.py        analysis cohort, binary target, age bands, feature lists
splits.py        repeated patient-level stratified group K-fold + inner validation splits
preprocess.py    one-hot + scaling, fitted inside each training fold
decision.py      decision rules: top-q% flagging, Youden threshold
models.py        the approved models and the two course extensions
tuning.py        one-SE rule, depth and k sweeps, complexity curves
evaluate.py      outer loop, resumable checkpoints, fingerprint
metrics.py       weighted metrics, patient bootstrap, usability gate
fairness.py      group rates, equalized-odds differences, permutation null, calibration
mitigation.py    reweighing and equalized-odds post-processing
sources.py       ablation, unawareness, proxy check (refits)
source_stats.py  statistics of the source analysis (no fitting)
ordering.py      test-ordering logistic regression
figures.py       figures of the paper, slides and notebook
runtime.py       CPU thread cap
"""
