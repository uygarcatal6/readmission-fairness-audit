"""Tests for readmission/models.py, tuning.py, evaluate.py and metrics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from readmission import cohort, evaluate, metrics, models, tuning

# ── metric kernels vs scikit-learn ──────────────────────────────────────────────────────────


def _toy_scores(n=400, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(float)
    s = rng.random(n) + 0.3 * y
    if ties:
        s = np.round(s, 1)  # big tie blocks, like a shallow tree or k-NN
    w = rng.integers(0, 4, n).astype(float)  # bootstrap-like weights, some zero
    return y, s, w


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_auc_matches_sklearn(ties):
    y, s, w = _toy_scores(ties=ties)
    got = metrics.weighted_auc(w[None, :], y, s)[0]
    assert got == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_average_precision_matches_sklearn(ties):
    y, s, w = _toy_scores(ties=ties)
    got = metrics.weighted_average_precision(w[None, :], y, s)[0]
    assert got == pytest.approx(average_precision_score(y, s, sample_weight=w), abs=1e-12)


@pytest.mark.parametrize("ties", [False, True])
def test_ranking_kernels_per_row_of_a_weight_batch_and_zero_weights_drop_out(ties):
    y, s, _ = _toy_scores(ties=ties)
    rng = np.random.default_rng(1)
    W = rng.integers(0, 4, (5, len(y))).astype(np.float32)  # a batch of bootstrap draws
    W[2, : len(y) // 2] = 0  # one draw leaves out half the rows (a subgroup)
    auc, ap = metrics.weighted_auc(W, y, s), metrics.weighted_average_precision(W, y, s)
    for b in range(len(W)):
        assert auc[b] == pytest.approx(roc_auc_score(y, s, sample_weight=W[b]), abs=1e-12)
        assert ap[b] == pytest.approx(average_precision_score(y, s, sample_weight=W[b]), abs=1e-12)
    # rows with weight 0 can be dropped: bit-identical (the sums are exact integers)
    keep = W[2] > 0
    assert metrics.weighted_auc(W[2:3, keep], y[keep], s[keep])[0] == auc[2]


def test_decision_metrics_and_acc_minus_nir():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], float)
    flag = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool)  # TP=1 FN=1 FP=1 TN=7
    d = {k: v[0] for k, v in metrics.weighted_decision(np.ones((1, 10)), y, flag).items()}
    assert d["tpr"] == 0.5 and d["ppv"] == 0.5 and d["fpr"] == pytest.approx(1 / 8)
    assert d["accuracy"] == 0.8 and d["acc_minus_nir"] == pytest.approx(0.0)
    # identity from the decision-rule analysis: acc - NIR = (2 TP - k) / n
    assert d["acc_minus_nir"] == pytest.approx((2 * 1 - 2) / 10)
    # the approved gate uses a FIXED NIR; the per-draw one is kept as "paired"
    fixed = metrics.weighted_decision(np.ones((1, 10)), y, flag, nir=0.886)
    assert fixed["acc_minus_nir"][0] == pytest.approx(0.8 - 0.886)
    assert fixed["acc_minus_nir_paired"][0] == pytest.approx(0.0)


def test_interval_counts_undefined_draws():
    ok = metrics._interval(np.r_[np.linspace(0, 1, 200)])
    assert ok["n_nan"] == 0 and ok["ci_reliable"]
    bad = metrics._interval(np.r_[np.linspace(0, 1, 190), [np.nan] * 10])
    assert bad["n_nan"] == 10 and not bad["ci_reliable"]


def test_bootstrap_weights_are_constant_within_a_patient():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    codes, n_pat = metrics.patient_codes(groups)
    W = next(metrics.bootstrap_weights(codes, n_pat, 50, np.random.default_rng(0)))
    assert W.shape == (50, 6)
    assert np.all(W[:, 0] == W[:, 1]) and np.all(W[:, 3] == W[:, 5])
    per_patient = W[:, [0, 2, 3]]
    assert np.all(per_patient.sum(axis=1) == n_pat)  # P patients drawn per replicate


# ── tuning ──────────────────────────────────────────────────────────────────────────────────


def test_one_se_rule_picks_the_simplest_within_one_se():
    depths = [1, 2, 4, 8, None]
    scores = np.array([
        [0.60, 0.61, 0.59],   # clearly worse
        [0.650, 0.656, 0.653],  # mean 0.653 >= 0.655 - SE(0.0029) -> simplest eligible
        [0.66, 0.65, 0.655],  # best mean 0.655
        [0.65, 0.66, 0.65],
        [0.55, 0.56, 0.54],
    ])
    sel = tuning.one_se_choice(depths, scores, prefer="smaller")
    assert sel["best_value"] == 4 and sel["value"] == 2
    ks = [1, 5, 25, 101]
    scores_k = np.array([[0.55, 0.56], [0.60, 0.61], [0.65, 0.66], [0.645, 0.655]])
    sel_k = tuning.one_se_choice(ks, scores_k, prefer="larger")
    assert sel_k["value"] == 101 and not sel_k["at_grid_edge"]  # best is 25, inside the grid
    rising = np.array([[0.55, 0.56], [0.60, 0.61], [0.65, 0.66], [0.67, 0.68]])
    assert tuning.one_se_choice(ks, rising, prefer="larger")["at_grid_edge"]


def test_knn_sweep_matches_kneighbors_classifier():
    rng = np.random.default_rng(3)
    X_fit, X_val = rng.normal(size=(300, 4)), rng.normal(size=(50, 4))
    y_fit = (X_fit[:, 0] + rng.normal(size=300) > 0).astype(int)
    ks = [1, 5, 15]
    got = tuning.knn_k_scores(X_fit, y_fit, X_val, ks)
    for j, k in enumerate(ks):
        want = KNeighborsClassifier(k).fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
        assert np.allclose(got[:, j], want)


def test_class_weight_only_where_supported():
    assert models.make_estimator("rf", class_weight="balanced").class_weight == "balanced"
    with pytest.raises(ValueError):
        models.make_estimator("knn", class_weight="balanced")


# ── end to end on a synthetic cohort ────────────────────────────────────────────────────────


def _synthetic_cohort(n_patients=250, seed=5):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, n_patients)
    pid = np.repeat([f"P{i}" for i in range(n_patients)], sizes)
    n = len(pid)
    df = pd.DataFrame({c: rng.normal(size=n) for c in cohort.NUMERIC})
    for c in cohort.CATEGORICAL:
        df[c] = rng.choice(["a", "b", "c"], size=n).astype(object)
    logit = -2.0 + 1.2 * df["number_inpatient"] + 0.5 * (df["race"] == "a")
    df[cohort.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    df[cohort.GROUP] = pid.astype(object)
    return df


def test_outer_loop_runs_resumes_and_keeps_capacity(tmp_path):
    df = _synthetic_cohort()
    cfg = evaluate.Config(models=("logreg", "tree", "knn"), extensions=(),
                          youden_models=("logreg", "tree", "knn"), class_weighted=("logreg",),
                          n_splits=3, n_repeats=2, n_inner=3,
                          tree_depths=(1, 2, 4, None), knn_ks=(1, 5, 15))
    logs = []
    evaluate.run(df, cfg, tmp_path, log=logs.append)
    assert len(list(tmp_path.glob("r*_f*.npz"))) == 6
    evaluate.run(df, cfg, tmp_path, log=logs.append)  # second call only skips
    assert sum(line.startswith("skip") for line in logs) == 6
    changed = evaluate.Config(**{**cfg.__dict__, "n_inner": 2})
    with pytest.raises(ValueError, match="different configuration"):
        evaluate.run(df, changed, tmp_path, log=logs.append)
    other_cohort = df.copy()
    other_cohort.loc[0, cohort.TARGET] = 1 - other_cohort.loc[0, cohort.TARGET]
    with pytest.raises(ValueError, match="different configuration"):
        evaluate.load_oof(tmp_path, other_cohort, cfg)
    assert not list(tmp_path.glob("*.tmp.*"))  # atomic writes leave no temp files

    oof = evaluate.load_oof(tmp_path, df, cfg)
    assert oof["flags"]["tree__topq"].shape == (2, len(df))
    for meta in oof["meta"]:
        assert meta["tuning"]["tree"]["value"] in cfg.tree_depths
        assert meta["tuning"]["knn"]["value"] in cfg.knn_ks
    # capacity: in every fold the number flagged equals round(q * n_test)
    y = df[cohort.TARGET].to_numpy()
    for r in range(2):
        flagged = oof["flags"]["logreg__topq"][r]
        assert abs(flagged.sum() - round(y.mean() * len(y))) <= 3  # one rounding per fold

    res = metrics.summarize(y, df[cohort.GROUP], oof, n_boot=20, rng=np.random.default_rng(0))
    auc = res["ranking"]["model:logreg"]["roc_auc"]
    assert 0.5 < auc["estimate"] < 1 and auc["ci"][0] <= auc["estimate"] <= auc["ci"][1]
    assert "usable" in res["decision"]["logreg__topq"]
    assert set(res["decision"]) >= {"logreg__youden", "logreg__cw05", "knn__q30"}
