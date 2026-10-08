"""Tests for readmission/sources.py (blocks, ablation arms, proxy check, ranking by group)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from readmission import cohort, evaluate, fairness, mitigation, source_stats, sources, splits


def test_blocks_cover_every_input_once_and_arms_drop_the_right_columns():
    flat = sum(sources.FEATURE_BLOCKS.values(), [])
    assert sorted(flat) == sorted(cohort.FEATURES) and len(flat) == len(set(flat))
    arms = sources.arm_features()
    for block, cols in sources.FEATURE_BLOCKS.items():
        assert set(arms[f"without_{block}"]) == set(cohort.FEATURES) - set(cols)
    assert set(cohort.FEATURES) - set(arms["no_age"]) == {"age"}
    assert set(cohort.FEATURES) - set(arms["no_race"]) == {"race"}
    assert set(cohort.FEATURES) - set(arms["no_sex"]) == {"sex"}
    assert set(arms["only_demographics"]) == {"age", "sex", "race"}
    assert arms["refit_seed"] == list(cohort.FEATURES) and sources.SEED_OFFSET["refit_seed"] != 0
    assert set(sources.comparisons()) == {f"{a}@topq" for a in arms}


BANDS = {"40-59": 50, "60-79": 70, "80+": 85}


def _cohort(n_patients=3000, seed=11):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 3, n_patients)
    pid = np.repeat([f"P{i}" for i in range(n_patients)], sizes)
    n = len(pid)
    df = pd.DataFrame({c: rng.normal(size=n) for c in cohort.NUMERIC})
    for c in cohort.CATEGORICAL:
        df[c] = rng.choice(["a", "b", "c"], size=n).astype(object)
    df["sex"] = rng.choice(["Female", "Male"], size=n).astype(object)
    df["race"] = rng.choice(["Caucasian", "AfricanAmerican", "Unknown"], size=n,
                            p=[0.6, 0.3, 0.1]).astype(object)
    df["age_band"] = rng.choice(list(BANDS), size=n).astype(object)
    df["age"] = df["age_band"].map(BANDS).astype(float) + rng.normal(0, 3, n)
    # a planted proxy: emergency visits carry race; nothing carries sex
    df["number_emergency"] = rng.normal(size=n) + 1.2 * (df["race"] == "AfricanAmerican")
    logit = -1.8 + 0.9 * df["number_inpatient"] + 0.5 * df["number_emergency"]
    df[cohort.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    df[cohort.GROUP] = pid.astype(object)
    df["encounter_id"] = np.arange(n)
    return df


CFG = evaluate.Config(models=("logreg",), extensions=(), youden_models=("logreg",),
                      class_weighted=(), n_splits=3, n_repeats=1, n_inner=2)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("src")
    df = _cohort()
    evaluate.run(df, CFG, tmp / "main", log=lambda s: None)
    scfg = sources.SourcesConfig(primary="logreg")
    logs = []
    sources.run(df, CFG, scfg, tmp / "main", tmp / "src", log=logs.append)
    return df, scfg, tmp, logs


def test_run_resumes_guards_and_keeps_every_rule(small_run):
    df, scfg, tmp, logs = small_run
    sources.run(df, CFG, scfg, tmp / "main", tmp / "src", log=logs.append)
    assert sum(line.startswith("skip") for line in logs) == 3
    with pytest.raises(ValueError, match="different configuration"):
        sources.run(df, CFG, sources.SourcesConfig(primary="logreg", attributes=("sex",)),
                    tmp / "main", tmp / "src", log=logs.append)
    oof = sources.load(tmp / "src", df, CFG, scfg)
    main = evaluate.load_oof(tmp / "main", df, CFG)
    assert np.array_equal(oof["flags"]["base__topq"], main["flags"]["logreg__topq"])
    y = df[cohort.TARGET].to_numpy()
    for (r, f, te, q), split in zip(oof["folds"], splits.outer_splits(
            y, df[cohort.GROUP].to_numpy(), n_splits=3, n_repeats=1), strict=True):
        assert np.array_equal(te, split.test)
        shares = {"topq": q, **evaluate.Q_RULES}
        for arm in sources.arm_features():
            for rule, share in shares.items():  # capacity kept for every arm and rule
                assert oof["flags"][f"{arm}__{rule}"][0, te].sum() == round(share * len(te))
    # race 'Unknown' is not a class of the proxy task
    assert "proxy_race__Unknown" not in oof["scores"]
    assert {"proxy_race__Caucasian", "proxy_race__AfricanAmerican"} <= set(oof["scores"])
    # the seed control is the full pipeline refitted: logistic regression ignores the seed, so
    # it reproduces the main run's scores (for the forest it is another realisation)
    assert np.allclose(oof["scores"]["refit_seed"], oof["scores"]["base"], atol=1e-6)


def test_a_main_checkpoint_from_another_split_is_refused(small_run, tmp_path):
    import shutil

    df, scfg, tmp, _ = small_run
    bad = tmp_path / "main"
    shutil.copytree(tmp / "main", bad)
    z = dict(np.load(bad / "r0_f0.npz"))
    z["test"] = z["test"][::-1].copy()  # same rows, other order: not the same split
    np.savez_compressed(bad / "r0_f0.npz", **z)
    y = df[cohort.TARGET].to_numpy()
    split = next(splits.outer_splits(y, df[cohort.GROUP].to_numpy(), n_splits=3, n_repeats=1))
    with pytest.raises(ValueError, match="outer split differs"):
        sources.run_split(df, split, CFG, scfg, bad)


def test_matched_degradation_hits_the_arm_auc_and_group_rates_match(small_run):
    df, scfg, tmp, _ = small_run
    oof = sources.load(tmp / "src", df, CFG, scfg)
    deg = source_stats.matched_degradation(df, oof, ["without_prior_use", "only_demographics"],
                                      n_draws=2)
    y = df[cohort.TARGET].to_numpy()
    for arm in ("without_prior_use", "only_demographics"):
        want = np.mean([roc_auc_score(y[te], oof["scores"][arm][r, te]) for r, f, te, q in oof["folds"]])
        # the noise brings the base ranking down to the arm's fold ROC-AUC (or leaves it if the
        # arm is not worse)
        base_auc = np.mean([roc_auc_score(y[te], oof["scores"]["base"][r, te])
                            for r, f, te, q in oof["folds"]])
        assert deg[arm]["mean_fold_roc_auc"] == pytest.approx(min(want, base_auc), abs=2e-3)
        assert set(deg[arm]) >= {f"eo_{a}" for a in fairness.MAIN_ATTRIBUTES}
    # identity: an exact copy of the base ranking loses nothing, so its reference is the base's
    # own EO (the rank transform and the sigma = 0 path change no decision)
    copy = {**oof, "scores": {**oof["scores"], "copy": oof["scores"]["base"]}}
    ident = source_stats.matched_degradation(df, copy, ["copy"], n_draws=1)["copy"]
    eff = mitigation.paired_effects(df, oof, {"b": ("base__topq", "base__topq")}, n_boot=2,
                                    rng=np.random.default_rng(0))["b"]
    for a in fairness.MAIN_ATTRIBUTES:
        assert ident[f"eo_{a}"] == pytest.approx(eff[f"eo_{a}"]["mitigated"], abs=1e-9, nan_ok=True)
    assert ident["max_auc_miss"] <= 1e-9
    # the noise draws have their own generators: the number of worker threads changes nothing
    seq = source_stats.matched_degradation(df, oof, ["without_prior_use", "only_demographics"],
                                           n_draws=2, n_jobs=1)
    for arm in seq:
        for k, v in seq[arm].items():
            assert v == pytest.approx(deg[arm][k], abs=0, nan_ok=True)
    rates = source_stats.group_rates(df, oof, ["base__topq"], "race")["base__topq"]
    rows = (df["race"] == "AfricanAmerican").to_numpy()
    fl = oof["flags"]["base__topq"][0]
    assert rates["AfricanAmerican"]["tpr"] == pytest.approx(fl[rows & (y == 1)].mean())
    assert rates["AfricanAmerican"]["fpr"] == pytest.approx(fl[rows & (y == 0)].mean())


def test_proxy_check_finds_the_planted_proxy_and_matches_sklearn(small_run):
    df, scfg, tmp, _ = small_run
    oof = sources.load(tmp / "src", df, CFG, scfg)
    rng = np.random.default_rng(0)
    race = source_stats.proxy_auc(df, oof, "race", n_boot=40, rng=rng)
    sex = source_stats.proxy_auc(df, oof, "sex", n_boot=40, rng=rng)
    assert race["macro"]["ci"][0] > 0.6  # emergency visits carry race
    assert sex["macro"]["ci"][0] < 0.5 < sex["macro"]["ci"][1]  # nothing carries sex
    keep = df["race"].to_numpy() != "Unknown"
    s = oof["scores"]["proxy_race__AfricanAmerican"][0]
    want = roc_auc_score((df["race"] == "AfricanAmerican").to_numpy()[keep], s[keep])
    assert race["AfricanAmerican"]["estimate"] == pytest.approx(want, abs=1e-9)


def test_group_ranking_is_the_auc_inside_each_group(small_run):
    df, scfg, tmp, _ = small_run
    oof = sources.load(tmp / "src", df, CFG, scfg)
    flags = oof["flags"]["base__topq"]
    res = source_stats.group_ranking(df, oof["scores"]["base"], flags, ("age_band",), n_boot=20,
                                rng=np.random.default_rng(1))
    y = df[cohort.TARGET].to_numpy()
    s = oof["scores"]["base"][0].astype(float)
    fpr_all = flags[0][y == 0].mean()
    for band in BANDS:
        rows = (df["age_band"] == band).to_numpy()
        want = roc_auc_score(y[rows], s[rows])
        got = res["age_band"][band]
        assert got["roc_auc"]["estimate"] == pytest.approx(want, abs=1e-9)
        assert got["n_pos"] == y[rows].sum()
        assert got["tpr"] == pytest.approx(flags[0][rows & (y == 1)].mean())
        assert got["fpr"] == pytest.approx(flags[0][rows & (y == 0)].mean())
        assert 0 <= got["tpr_at_overall_fpr"] <= 1
    # everyone in one group: its own cut-off IS the shared one, so the shared cut-off adds nothing
    one = df.copy()
    one["age_band"] = "all"
    whole = source_stats.group_ranking(one, oof["scores"]["base"], flags, ("age_band",), n_boot=5,
                                       rng=np.random.default_rng(3))["age_band"]["all"]
    assert whole["tpr_at_overall_fpr"] == pytest.approx(whole["tpr"], abs=0.005)
    assert fpr_all == pytest.approx(whole["fpr"])


def test_test_fold_labels_do_not_reach_any_output(small_run):
    df, scfg, tmp, _ = small_run
    y = df[cohort.TARGET].to_numpy()
    split = next(splits.outer_splits(y, df[cohort.GROUP].to_numpy(), n_splits=3, n_repeats=1))
    a = sources.run_split(df, split, CFG, scfg, tmp / "main")

    flipped = df.copy()  # readmission labels of test rows
    rows = flipped.index[split.test[:80]]
    flipped.loc[rows, cohort.TARGET] = 1 - flipped.loc[rows, cohort.TARGET]
    b = sources.run_split(flipped, split, CFG, scfg, tmp / "main")
    for part in ("flags", "scores"):
        for k in a[part]:
            assert np.array_equal(a[part][k], b[part][k]), k

    # race of test rows: the label of the race proxy task and an input of some arms. Arms and
    # proxy tasks without race as an input must not move (the proxy's labels come from training)
    swapped = df.copy()
    swap = {"Caucasian": "AfricanAmerican", "AfricanAmerican": "Caucasian", "Unknown": "Unknown"}
    swapped.loc[rows, "race"] = swapped.loc[rows, "race"].map(swap)
    c = sources.run_split(swapped, split, CFG, scfg, tmp / "main")
    unaffected = ["no_race", "without_demographics"] + [k for k in a["scores"]
                                                          if k.startswith("proxy_race__")]
    for k in unaffected:
        assert np.array_equal(a["scores"][k], c["scores"][k]), k
    assert not np.array_equal(a["scores"]["no_sex"], c["scores"]["no_sex"])  # race is its input


def test_report_builds_and_prints_the_right_numbers(small_run):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sources.py"
    spec = importlib.util.spec_from_file_location("run_sources", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df, scfg, tmp, _ = small_run
    oof = sources.load(tmp / "src", df, CFG, scfg)
    rng = np.random.default_rng(2)
    effects = mitigation.paired_effects(df, oof, sources.comparisons(rules=sources.SCORE_RULES),
                                        n_boot=10, rng=rng)
    keys = [f"{a}__topq" for a in ["base", *sources.arm_features()]]
    res = {"n_boot": 10, "n_repeats": 1, "n_splits": 3, "blocks": sources.FEATURE_BLOCKS,
           "effects": effects, "null_mean": 0.03, "fairness_report": "approved_fairness.md",
           "degradation_draws": 1,
           "degradation": source_stats.matched_degradation(df, oof, list(sources.arm_features()), n_draws=1),
           "proxy": {a: source_stats.proxy_auc(df, oof, a, n_boot=10, rng=rng) for a in scfg.attributes},
           "ranking": source_stats.group_ranking(df, oof["scores"]["base"], oof["flags"]["base__topq"],
                                            scfg.attributes, n_boot=10, rng=rng),
           "group_rates": source_stats.group_rates(df, oof, keys, "age_band")}
    text = mod.build_report(res, "logreg", "age_band", "0.1 min")
    assert "## 1. Feature-set ablation" in text and "proxy check" in text and "3 outer splits" in text
    assert text.index("EO age_band (full → arm)") < text.index("Δ EO sex")
    e = effects["without_prior_use@topq"]["eo_age_band"]
    row = next(line for line in text.splitlines() if line.startswith("| without prior use"))
    assert f"{e['counterpart']:.3f} → {e['mitigated']:.3f}" in row  # full → arm, not swapped
    assert f"{e['difference']:+.3f} [" in row  # signed difference
    d = res["ranking"]["age_band"]["80+"]
    row80 = next(line for line in text.splitlines() if line.startswith("| 80+ |"))
    assert f"{d['tpr']:.3f}" in row80 and f"{d['tpr'] - d['tpr_at_overall_fpr']:+.3f}" in row80
    assert "another random seed" in text and "permutation null" in text


def test_lead_is_taken_only_from_an_audit_of_the_same_flags(small_run):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sources.py"
    spec = importlib.util.spec_from_file_location("run_sources", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df, scfg, tmp, _ = small_run
    oof = sources.load(tmp / "src", df, CFG, scfg)
    eff = mitigation.paired_effects(df, oof, {"refit_seed@topq": ("refit_seed__topq", "base__topq")},
                                    n_boot=2, rng=np.random.default_rng(0))
    audit = fairness.audit(df, {**oof, "flags": {"logreg__topq": oof["flags"]["base__topq"]}},
                           keys=["logreg__topq"], calibration_models=[], n_boot=2,
                           rng=np.random.default_rng(1), null_keys=[])
    fair = {"primary_model": "logreg", "leading_attribute": "race", "gaps": audit["gaps"]}
    assert mod.lead_from_audit(fair, eff, "logreg") == "race"
    assert mod.lead_from_audit({**fair, "primary_model": "rf"}, eff, "logreg") is None
    stale = {**fair, "gaps": {"logreg__topq": {a: {"eo_diff": {"estimate": 0.5}}
                                              for a in fairness.MAIN_ATTRIBUTES}}}
    assert mod.lead_from_audit(stale, eff, "logreg") is None
