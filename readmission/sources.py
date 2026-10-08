"""
readmission/sources.py — where the disparities come from (source analysis).

docs/APPROVED_PROPOSAL.md: "To find where disparities come from, I use feature-set ablation,
removal of the sensitive attributes, how well the remaining features predict each attribute,
and base rates and calibration by group." Base rates and calibration by group are in
readmission/fairness.py; this module does the other three, plus diagnostics that need no refit
(ranking quality within each group, group rates per arm).

Model and splits: the primary model of the audit with the main run's settings, on the same
outer splits, flagged with the same rules (top q %, q05..q30). Every arm is compared with the
main run's model (`base`, copied from its checkpoint) on the same bootstrap draw
(mitigation.paired_effects). A difference is the effect of removing the features INCLUDING a
new random-forest realisation; the `refit_seed` arm measures the second part alone. The CIs
cover patient sampling of the test folds given the fitted models; refit variability is
averaged over the splits, not resampled. Confirmatory comparison: Δ EO of the leading
attribute at top q %; the other attributes and rules are descriptive.

Feature blocks (all 23 model inputs, each in exactly one block; fixed before the results):
    demographics    age, sex, race
    prior_use       outpatient, emergency and inpatient visits in the year before
    current_stay    length of stay, lab tests, procedures, medications, number of diagnoses
    admission       admission type and source, discharge group, admitting specialty, payer
    diagnosis       primary-diagnosis group
    diabetes_care   HbA1c and glucose results, insulin, metformin, medication change,
                    diabetes medication prescribed

Arms (flag keys "arm__rule", as in readmission/evaluate.py)
    base                  the main run's scores and flags for the primary model
    without_{block}       refitted without one block (leave-one-block-out ablation)
    no_sex, no_race, no_age
                          refitted without one sensitive attribute ("unawareness"; the three
                          together are without_demographics). age_band is not a model input:
                          no_age removes the age it is derived from.
    only_demographics     refitted on age, sex and race alone (how much of the risk the
                          attributes carry by themselves; its gap is large by construction,
                          because it can only rank whole demographic cells)
    refit_seed            refitted on all inputs with another random_state: how much the gap
                          moves with no input removed (reference row)

Reading the ablation: removing inputs also lowers ROC-AUC. A weaker ranking can shrink a gap by
moving towards random flags, and it can also widen one (a model left with little but age).
Two references are reported next to every arm: `refit_seed` (no input removed) and
`matched_degradation` — the gap the base ranking shows after losing the SAME ROC-AUC to random
noise in each test fold. An arm whose gap is far below that reference removed features that
carry the gap, not just predictive signal.

Proxy check: for each attribute, a classifier (HistGradientBoosting, the approved boosting model,
default settings) predicts the attribute from the other model inputs (sex: all but sex; age
band: all but age; race: all but race), fitted on the training fold and scored on the test fold.
Race "Unknown" (missing code) is left out of the race task. Reported as ROC-AUC (one class vs
the rest, averaged over classes for age band and race). 0.5 = the other inputs carry no trace of
the attribute; the further above 0.5, the less removing the attribute removes its information.
An untuned model gives a lower bound on that trace.

The statistics computed from the checkpoints (matched degradation, group rates, proxy
ROC-AUC, ranking within groups) are in readmission/source_stats.py, so that fixing a
statistic never invalidates the fitted checkpoints: the fingerprint covers only this file.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from readmission import cohort, evaluate, fairness, splits
from readmission.decision import flag_top_q
from readmission.models import TUNED, make_estimator
from readmission.preprocess import make_preprocessor

FEATURE_BLOCKS: dict[str, list[str]] = {
    "demographics": ["age", "sex", "race"],
    "prior_use": ["number_outpatient", "number_emergency", "number_inpatient"],
    "current_stay": ["time_in_hospital", "num_lab_procedures", "num_procedures",
                     "num_medications", "number_diagnoses"],
    "admission": ["admission_type", "admission_source", "discharge_group",
                  "medical_specialty", "payer_code"],
    "diagnosis": ["diag_1_group"],
    "diabetes_care": ["a1c_result", "max_glu_serum", "insulin", "metformin", "med_change",
                      "prescription_recorded"],
}
UNAWARE: dict[str, list[str]] = {"no_sex": ["sex"], "no_race": ["race"], "no_age": ["age"]}
ONLY: dict[str, list[str]] = {"only_demographics": FEATURE_BLOCKS["demographics"]}
SEED_OFFSET: dict[str, int] = {"refit_seed": 1}  # arms refitted with random_state = seed + offset
PROXY_DROP: dict[str, list[str]] = {"sex": ["sex"], "age_band": ["age"], "race": ["race"]}
PROXY_MODEL = "hgb"
SCORE_RULES = ("topq", *evaluate.Q_RULES)


@dataclass
class SourcesConfig:
    primary: str
    attributes: tuple[str, ...] = fairness.MAIN_ATTRIBUTES


def arm_features(features=cohort.FEATURES) -> dict[str, list[str]]:
    """Model inputs of every refitted arm, in the cohort's column order."""
    features = list(features)
    out = {f"without_{b}": [f for f in features if f not in cols] for b, cols in FEATURE_BLOCKS.items()}
    out |= {arm: [f for f in features if f not in cols] for arm, cols in UNAWARE.items()}
    out |= {arm: [f for f in features if f in cols] for arm, cols in ONLY.items()}
    out |= {arm: list(features) for arm in SEED_OFFSET}
    return out


def _x(df: pd.DataFrame, rows: np.ndarray, features) -> pd.DataFrame:
    return df.iloc[rows][list(features)]


def proxy_classes(df: pd.DataFrame, attr: str) -> list[str]:
    """Classes of the proxy task (race 'Unknown' is a missing code, not a class)."""
    skip = fairness.NOT_A_GROUP.get(attr, set())
    return sorted(c for c in df[attr].astype(str).unique() if c not in skip)


def run_split(df: pd.DataFrame, split: splits.Split, cfg: evaluate.Config,
              scfg: SourcesConfig, main_dir: Path) -> dict:
    """Ablation, unawareness, seed-control and proxy models for one outer split."""
    t0 = time.time()
    m = scfg.primary
    y_all = df[cohort.TARGET].to_numpy()
    tr, te = split.train, split.test
    y_tr = y_all[tr]
    main = np.load(main_dir / f"r{split.repeat}_f{split.fold}.npz")
    meta = json.loads((main_dir / f"r{split.repeat}_f{split.fold}.json").read_text(encoding="utf-8"))
    if not np.array_equal(main["test"], te):
        raise ValueError("outer split differs from the main run's checkpoint")
    q = meta["q"]
    params = {TUNED[m]: meta["tuning"][m]["value"]} if m in meta["tuning"] else None
    rng = np.random.default_rng([cfg.seed, split.repeat, split.fold, 2])  # its own stream

    scores = {"base": main[f"score__{m}"].astype(float)}
    flags = {f"base__{rule}": main[f"flag__{m}__{rule}"] for rule in SCORE_RULES}
    for arm, feats in arm_features(cfg.features).items():
        pre = make_preprocessor(feats).fit(_x(df, tr, feats))
        est = make_estimator(m, params=params, random_state=cfg.seed + SEED_OFFSET.get(arm, 0))
        est.fit(pre.transform(_x(df, tr, feats)), y_tr)
        s = est.predict_proba(pre.transform(_x(df, te, feats)))[:, 1]
        scores[arm] = s
        flags[f"{arm}__topq"] = flag_top_q(s, q, rng=rng)
        for rule, share in evaluate.Q_RULES.items():
            flags[f"{arm}__{rule}"] = flag_top_q(s, share, rng=rng)

    for attr in scfg.attributes:
        feats = [f for f in cfg.features if f not in PROXY_DROP[attr]]
        target = df[attr].astype(str).to_numpy()
        classes = proxy_classes(df, attr)
        fit_rows = tr[np.isin(target[tr], classes)]
        pre = make_preprocessor(feats).fit(_x(df, fit_rows, feats))
        est = make_estimator(PROXY_MODEL, random_state=cfg.seed)
        est.fit(pre.transform(_x(df, fit_rows, feats)), target[fit_rows])
        proba = est.predict_proba(pre.transform(_x(df, te, feats)))
        for j, c in enumerate(est.classes_):
            scores[f"proxy_{attr}__{c}"] = proba[:, j]

    return {"repeat": split.repeat, "fold": split.fold, "test": te, "q": q,
            "scores": scores, "cw_scores": {}, "flags": flags, "tuning": {}, "thresholds": {},
            "seconds": round(time.time() - t0, 1)}


def fingerprint(df: pd.DataFrame, cfg: evaluate.Config, scfg: SourcesConfig) -> dict:
    """The main run's fingerprint plus this module's settings, blocks, code and versions."""
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"main": evaluate.fingerprint(df, cfg), "sources": asdict(scfg),
            "blocks": FEATURE_BLOCKS, "seed_offset": SEED_OFFSET,
            "not_a_group": {a: sorted(v) for a, v in fairness.NOT_A_GROUP.items()},
            "sources_code_sha256": code, "versions": {"scikit-learn": sklearn.__version__}}


def run(df: pd.DataFrame, cfg: evaluate.Config, scfg: SourcesConfig, main_dir: Path,
        out_dir: Path, *, log=print) -> None:
    """Run (or resume) every outer split; one checkpoint pair per split, as the main run."""
    if sorted(sum(FEATURE_BLOCKS.values(), [])) != sorted(cfg.features):
        raise ValueError("the feature blocks must cover every model input exactly once")
    main_dir, out_dir = Path(main_dir), Path(out_dir)
    evaluate._check_fingerprint(main_dir, evaluate.fingerprint(df, cfg), write=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate._check_fingerprint(out_dir, fingerprint(df, cfg, scfg), write=True)
    y, g = df[cohort.TARGET].to_numpy(), df[cohort.GROUP].to_numpy()
    for split in splits.outer_splits(y, g, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats,
                                     seed=cfg.seed):
        path = out_dir / f"r{split.repeat}_f{split.fold}"
        if path.with_suffix(".npz").exists() and path.with_suffix(".json").exists():
            log(f"skip r{split.repeat} f{split.fold} (checkpoint exists)")
            continue
        res = run_split(df, split, cfg, scfg, main_dir)
        evaluate._save(res, path)
        log(f"done r{split.repeat} f{split.fold} in {res['seconds']}s")


def load(out_dir: Path, df: pd.DataFrame, cfg: evaluate.Config, scfg: SourcesConfig) -> dict:
    """Per-repeat out-of-fold arrays, in the layout of evaluate.load_oof (same checks), plus
    `folds`: (repeat, fold, test rows, q) of every split."""
    out_dir = Path(out_dir)
    evaluate._check_fingerprint(out_dir, fingerprint(df, cfg, scfg), write=False)
    R, n = cfg.n_repeats, len(df)
    out = {"scores": {}, "cw_scores": {}, "flags": {}, "q_ref": np.full((R, n), np.nan),
           "meta": [], "folds": []}
    keys = None
    for r in range(R):
        for f in range(cfg.n_splits):
            z = np.load(out_dir / f"r{r}_f{f}.npz")
            if keys is None:
                keys = set(z.files)
            elif set(z.files) != keys:
                raise ValueError(f"r{r}_f{f}.npz holds different arrays than the first split")
            meta = json.loads((out_dir / f"r{r}_f{f}.json").read_text(encoding="utf-8"))
            out["meta"].append(meta)
            te = z["test"]
            out["folds"].append((r, f, te, meta["q"]))
            out["q_ref"][r, te] = meta["q"]
            for key in z.files:
                if key == "test":
                    continue
                kind, name = key.split("__", 1)
                bucket = "flags" if kind == "flag" else "scores"
                if name not in out[bucket]:
                    out[bucket][name] = (np.zeros((R, n), bool) if kind == "flag"
                                         else np.full((R, n), np.nan, np.float32))
                out[bucket][name][r, te] = z[key]
    if np.isnan(out["q_ref"]).any():
        raise ValueError("some rows were never tested: incomplete checkpoints")
    for name, arr in out["scores"].items():
        if np.isnan(arr).any():
            raise ValueError(f"scores[{name}] has rows without a prediction")
    return out


def comparisons(rules=("topq",)) -> dict:
    """{name: (arm key, base key)} for every refitted arm and rule."""
    return {f"{arm}@{rule}": (f"{arm}__{rule}", f"base__{rule}")
            for arm in arm_features() for rule in rules}
