"""Tests for readmission/mitigation.py (weights, EO post-processing, arms, paired differences)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from readmission import cohort, evaluate, fairness, metrics, mitigation, splits


def _dependent(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    s = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1])
    y = (rng.random(n) < np.select([s == "A", s == "B"], [0.1, 0.3], 0.5)).astype(int)
    return s, y


def _cell_totals(s, y, w):
    return pd.Series(w).groupby([s, y]).sum()


def test_kc_weights_make_attribute_and_label_independent():
    s, y = _dependent()
    w = mitigation.reweigh_weights(y, s, "kc")
    assert w.mean() == pytest.approx(1.0)
    tot = _cell_totals(s, y, w) / w.sum()
    p_s = pd.Series(s).value_counts(normalize=True)
    p_y = pd.Series(y).value_counts(normalize=True)
    for (g, label), share in tot.items():  # weighted P(s, y) = P(s) P(y)
        assert share == pytest.approx(p_s[g] * p_y[label])


def test_cell_weights_give_every_cell_the_same_total():
    s, y = _dependent()
    w = mitigation.reweigh_weights(y, s, "cell")
    assert w.mean() == pytest.approx(1.0)
    tot = _cell_totals(s, y, w)
    assert np.allclose(tot, tot.iloc[0])


def test_kc_weights_edge_cases():
    s = np.repeat(["A", "B"], 10)
    y = np.tile([1, 0, 0, 0, 0], 4)  # 20 % positives in both groups: already independent
    assert np.allclose(mitigation.reweigh_weights(y, s, "kc"), 1.0)
    with pytest.raises(ValueError, match="mode"):
        mitigation.reweigh_weights(y, s, "balanced")
    with pytest.raises(ValueError, match="empty"):  # group B without positives
        mitigation.reweigh_weights(np.r_[np.tile([1, 0], 5), np.zeros(10)], s, "kc")


def test_fit_groups_pools_small_levels_and_the_missing_code():
    v = np.array(["big"] * 300 + ["mid"] * 200 + ["tiny"] * 30 + ["Unknown"] * 200)
    y = np.r_[np.ones(100), np.zeros(200), np.ones(60), np.zeros(140), np.ones(10), np.zeros(20),
              np.ones(60), np.zeros(140)]
    mapping, default = mitigation.fit_groups(v, y, min_positives=50, not_a_group={"Unknown"})
    assert mapping == {"big": "big", "mid": "mid", "tiny": "pooled", "Unknown": "pooled"}
    assert default == "pooled"
    # a pool below the minimum joins the largest level
    mapping, default = mitigation.fit_groups(v[:530], y[:530], min_positives=50)
    assert mapping["tiny"] == "big" and default == "big"
    # nothing small: every level kept, unseen levels go to the largest
    mapping, default = mitigation.fit_groups(v[:500], y[:500], min_positives=50)
    assert mapping == {"big": "big", "mid": "mid"} and default == "big"


def test_eo_at_capacity_equalizes_groups_and_keeps_the_share_on_its_fit_data():
    rng = np.random.default_rng(0)
    n = 20000
    g = rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1])
    y = (rng.random(n) < 0.12).astype(int)
    s = rng.normal(size=n) + y * np.select([g == "A", g == "B"], [1.0, 0.5], 0.2)
    fit = mitigation.fit_eo_at_capacity(y, s, g, 0.12)
    p = mitigation.eo_flag_probability(fit, s, g)
    assert p.mean() == pytest.approx(0.12, abs=1e-6)  # expected share flagged = q
    for k in "ABC":
        r = g == k
        assert p[r & (y == 1)].mean() == pytest.approx(fit["tpr"], abs=1e-6)
        assert p[r & (y == 0)].mean() == pytest.approx(fit["fpr"], abs=1e-6)
    # the weakest group binds (uses its scores fully); the others mix in a coin flip
    assert fit["groups"]["C"]["p_ignore"] == pytest.approx(0, abs=1e-6)
    assert fit["groups"]["A"]["p_ignore"] > 0
    # on the capacity line no common point has more true positives: C's hull is the limit
    hull_c = mitigation.roc_hull(y[g == "C"], s[g == "C"])
    assert np.interp(fit["fpr"], hull_c[:, 0], hull_c[:, 1]) == pytest.approx(fit["tpr"], abs=1e-6)


@pytest.mark.parametrize("case", ["continuous", "ties", "positive_on_top", "all_tied"])
def test_roc_hull_is_concave_above_every_roc_point_and_reproducible(case):
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.2).astype(int)
    s = rng.normal(size=400) + y
    if case == "ties":
        s = np.round(s, 0)
    elif case == "positive_on_top":
        s[np.flatnonzero(y)[0]] = 99.0
    elif case == "all_tied":
        s = np.zeros(400)
    h = mitigation.roc_hull(y, s)
    assert np.all(np.diff(h[:, 0]) > 0) and h[-1, 0] == 1 and h[-1, 1] == 1
    slopes = np.diff(h[:, 1]) / np.diff(h[:, 0])
    assert np.all(np.diff(slopes) <= 1e-12)  # concave
    fpr, tpr, _ = roc_curve(y, s)
    assert np.all(np.interp(fpr, h[:, 0], h[:, 1]) >= tpr - 1e-12)  # on or above every point
    for f, t, thr in h:  # every vertex is a real threshold on the fit data
        flag = s >= thr
        assert flag[y == 0].mean() == pytest.approx(f) and flag[y == 1].mean() == pytest.approx(t)


def test_eo_at_capacity_matches_a_brute_force_optimum():
    rng = np.random.default_rng(4)
    n = 6000
    g = rng.choice(["A", "B"], n, p=[0.7, 0.3])
    y = (rng.random(n) < 0.12).astype(int)
    s = rng.normal(size=n) + y * np.where(g == "A", 1.0, 0.4)
    hulls = [mitigation.roc_hull(y[g == k], s[g == k]) for k in "AB"]
    p1 = y.mean()
    grid = np.linspace(0, 1, 200001)
    for q in (0.05, 0.1144, 0.3):
        fit = mitigation.fit_eo_at_capacity(y, s, g, q)
        line = (q - (1 - p1) * grid) / p1
        feasible = np.minimum(*[np.interp(grid, h[:, 0], h[:, 1]) for h in hulls]) >= line
        assert fit["tpr"] == pytest.approx(line[feasible].max(), abs=1e-4)


def _cohort(n_patients=1500, seed=7):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 3, n_patients)
    pid = np.repeat([f"P{i}" for i in range(n_patients)], sizes)
    n = len(pid)
    df = pd.DataFrame({c: rng.normal(size=n) for c in cohort.NUMERIC})
    for c in cohort.CATEGORICAL:
        df[c] = rng.choice(["a", "b", "c"], size=n).astype(object)
    df["sex"] = rng.choice(["Female", "Male"], size=n).astype(object)
    df["race"] = rng.choice(["Caucasian", "AfricanAmerican", "Asian", "Unknown"], size=n,
                            p=[0.55, 0.33, 0.02, 0.10]).astype(object)
    df["age_band"] = rng.choice(["40-59", "60-79", "80+"], size=n).astype(object)
    # race enters the risk and is a feature: the unmitigated model has a race gap
    logit = -1.8 + 0.9 * df["number_inpatient"] + 1.0 * (df["race"] == "AfricanAmerican")
    df[cohort.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    df[cohort.GROUP] = pid.astype(object)
    df["encounter_id"] = np.arange(n)
    return df


CFG = evaluate.Config(models=("logreg",), extensions=(), youden_models=("logreg",),
                      class_weighted=(), n_splits=3, n_repeats=1, n_inner=2)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mit")
    df = _cohort(n_patients=4000)  # large enough that the main groups reach 50 validation positives
    evaluate.run(df, CFG, tmp / "main", log=lambda s: None)
    mcfg = mitigation.MitigationConfig(primary="logreg")
    logs = []
    mitigation.run(df, CFG, mcfg, tmp / "main", tmp / "mit", log=logs.append)
    return df, mcfg, tmp, logs


def test_run_resumes_guards_and_copies_the_baseline(small_run):
    df, mcfg, tmp, logs = small_run
    mitigation.run(df, CFG, mcfg, tmp / "main", tmp / "mit", log=logs.append)
    assert sum(line.startswith("skip") for line in logs) == 3
    with pytest.raises(ValueError, match="different configuration"):
        mitigation.run(df, CFG, mitigation.MitigationConfig(primary="logreg", val_fraction=0.25),
                       tmp / "main", tmp / "mit", log=logs.append)
    assert not list((tmp / "mit").glob("*.tmp.*"))
    oof = mitigation.load(tmp / "mit", df, CFG, mcfg)
    main = evaluate.load_oof(tmp / "main", df, CFG)
    assert np.array_equal(oof["flags"]["base__topq"], main["flags"]["logreg__topq"])
    assert np.allclose(oof["scores"]["base"], main["scores"]["logreg"])
    # race: Asian (tiny) and Unknown (missing code) are merged for fitting in every split; the
    # pool stands alone when it has >= 50 validation positives, else it joins the largest level
    for meta in oof["meta"]:
        diag = meta["tuning"]["postprocessing"]["race"]
        fg = diag["fit_groups"]
        assert fg["Asian"] == fg["Unknown"] and fg["Caucasian"] == "Caucasian"
        assert fg["Asian"] in ("pooled", "Caucasian")
        if fg["Asian"] == "pooled":
            assert diag["val_positives"]["pooled"] >= 50
        assert min(diag["val_positives"].values()) >= 50  # every fitted group meets the rule


def test_capacity_is_kept_per_fold_and_counterparts_share_the_model(small_run):
    df, mcfg, tmp, _ = small_run
    oof = mitigation.load(tmp / "mit", df, CFG, mcfg)
    y = df[cohort.TARGET].to_numpy()
    for meta, split in zip(oof["meta"], splits.outer_splits(y, df[cohort.GROUP].to_numpy(),
                                                            n_splits=3, n_repeats=1), strict=True):
        te = split.test
        k = oof["flags"]["base__topq"][0, te].sum()
        assert k == round(meta["q"] * len(te))
        for attr in mcfg.attributes:
            for mode in ("kc", "cell"):
                assert oof["flags"][f"rw_{mode}_{attr}__topq"][0, te].sum() == k
        assert oof["flags"]["val__topq"][0, te].sum() == k
        # the TO twin is the single threshold on the SAME 80 % model's scores
        thr = meta["thresholds"]["val__youden"]
        assert np.array_equal(oof["flags"]["val__youden"][0, te], oof["scores"]["val"][0, te] >= thr)
    # accuracy objective: (almost) nobody flagged; balanced accuracy and capacity: many
    assert oof["flags"]["toacc_race__eo"].mean() < 0.05 < oof["flags"]["to_race__eo"].mean()
    assert abs(oof["flags"]["eocap_race__eo"].mean() - oof["flags"]["val__topq"].mean()) < 0.05
    for meta, split in zip(oof["meta"], splits.outer_splits(y, df[cohort.GROUP].to_numpy(),
                                                            n_splits=3, n_repeats=1), strict=True):
        for attr in mcfg.attributes:  # the twins flag exactly as many as their EO arm, per fold
            for name in ("eocap", "to"):
                assert (oof["flags"][f"{name}twin_{attr}__samek"][0, split.test].sum()
                        == oof["flags"][f"{name}_{attr}__eo"][0, split.test].sum())


def test_paired_effects_zero_for_identical_arms_and_match_the_audit(small_run):
    df, mcfg, tmp, _ = small_run
    oof = mitigation.load(tmp / "mit", df, CFG, mcfg)
    comps = mitigation.comparisons() | {"same": ("base__topq", "base__topq")}
    res = mitigation.paired_effects(df, oof, comps, n_boot=30, rng=np.random.default_rng(0))
    assert res["same"]["accuracy"]["difference"] == 0 and res["same"]["accuracy"]["ci"] == [0.0, 0.0]
    audit = fairness.audit(df, oof, keys=["base__topq", "to_race__eo"], calibration_models=[],
                           n_boot=5, rng=np.random.default_rng(1), null_keys=[])
    for a in fairness.MAIN_ATTRIBUTES:  # the paired EO uses exactly the audit's EO rules
        assert res["to_race"][f"eo_{a}"]["mitigated"] == pytest.approx(
            audit["gaps"]["to_race__eo"][a]["eo_diff"]["estimate"], nan_ok=True)
        assert res["rw_kc_race@topq"][f"eo_{a}"]["counterpart"] == pytest.approx(
            audit["gaps"]["base__topq"][a]["eo_diff"]["estimate"], nan_ok=True)
    # EO post-processing at capacity on race narrows the race gap of its twin (fairlearn's TO is
    # too noisy on this ~300-row validation part for a directional check)
    assert res["eocap_race"]["eo_race"]["mitigated"] < res["eocap_race"]["eo_race"]["counterpart"]
    assert "roc_auc" not in res["to_race"] and "roc_auc" in res["rw_kc_race@topq"]


def test_test_fold_labels_do_not_reach_any_decision(small_run):
    df, mcfg, tmp, _ = small_run
    y = df[cohort.TARGET].to_numpy()
    split = next(splits.outer_splits(y, df[cohort.GROUP].to_numpy(), n_splits=3, n_repeats=1))
    a = mitigation.run_split(df, split, CFG, mcfg, tmp / "main")
    changed = df.copy()
    rows = changed.index[split.test[:60]]
    changed.loc[rows, cohort.TARGET] = 1 - changed.loc[rows, cohort.TARGET]
    b = mitigation.run_split(changed, split, CFG, mcfg, tmp / "main")
    assert a["thresholds"] == b["thresholds"]
    for k in a["flags"]:
        assert np.array_equal(a["flags"][k], b["flags"][k]), k
    for k in a["scores"]:
        assert np.array_equal(a["scores"][k], b["scores"][k]), k


def test_knn_primary_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sample weights"):
        mitigation.run(pd.DataFrame(), evaluate.Config(), mitigation.MitigationConfig(primary="knn"),
                       tmp_path, tmp_path / "mit")


def test_report_builds_from_a_small_run(small_run):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mitigation.py"
    spec = importlib.util.spec_from_file_location("run_mitigation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df, mcfg, tmp, _ = small_run
    oof = mitigation.load(tmp / "mit", df, CFG, mcfg)
    rng = np.random.default_rng(1)
    keys = mod.audit_keys(mcfg.attributes)
    effects = mitigation.paired_effects(
        df, oof, mitigation.comparisons(rules=mitigation.SCORE_RULES), n_boot=10, rng=rng)
    audit = fairness.audit(df, oof, keys=keys, calibration_models=[], n_boot=10, rng=rng,
                           null_keys=[])
    audit["overall_selection"] = {k: float(oof["flags"][k].mean()) for k in keys}
    overall = metrics.summarize(df[cohort.TARGET].to_numpy(), df[cohort.GROUP].to_numpy(),
                                {**oof, "flags": {k: oof["flags"][k] for k in keys}},
                                n_boot=10, rng=rng)
    res = {"n_boot": 10, "n_repeats": 1, "effects": effects, "audit": audit, "summary": overall,
           "config": {"min_fit_positives": 50},
           "eocap_share": {"mean": 0.11, "min": 0.10, "max": 0.12, "q": 0.11},
           "postprocessing": mod.postprocessing_summary(oof["meta"], mcfg.attributes)}
    text = mod.build_report(res, "logreg", "race", "0.1 min")
    assert text.index("| race |") < text.index("| sex |")  # the leading attribute comes first
    assert "Group rates by race" in text and "Group rates by sex" in text
    assert "Unknown (not a group)" in text and "Asian → " in text
    assert "Kamiran–Calders − cell balancing" in text and "Usable" in text
