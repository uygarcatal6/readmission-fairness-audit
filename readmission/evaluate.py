"""
readmission/evaluate.py — the outer loop of the approved protocol.

For each of the 25 outer splits (5 folds x 5 repeats, readmission/splits.py):
    1. q = readmission rate of the training fold (never the test fold).
    2. Inner loop on the training fold only (5 patient-level folds). For every inner fold the
       preprocessor is fitted on the inner-fit rows, then:
         - decision tree: one tree per candidate depth      -> depth by the one-SE rule
         - k-NN: one neighbour search for all candidate k   -> k by the one-SE rule
         - the other Youden models: inner out-of-fold scores
       The inner out-of-fold scores give each model its Youden threshold.
    3. Final fit on the whole training fold (preprocessor refitted there), scores on the test
       fold, then the decision rules:
         topq      flag the top q %  (main operating point)
         q05..q30  flag the top 5 / 10 / 20 / 30 %  (sensitivity)
         youden    score >= inner-CV Youden threshold  (sensitivity)
         cw05      class-weighted model, score >= 0.5  (sensitivity)
    4. Everything for the split is written to <checkpoint_dir>/r{repeat}_f{fold}.npz (+ .json)
       so an interrupted run resumes where it stopped.

Randomness: one generator per split, np.random.default_rng([seed, repeat, fold]); it breaks
ties in flag_top_q. The models get random_state = seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from readmission import cohort, splits
from readmission.decision import SENSITIVITY_Q, flag_top_q, youden_threshold
from readmission.models import (
    APPROVED,
    CLASS_WEIGHTED,
    EXTENSIONS,
    KNN_KS,
    TREE_DEPTHS,
    make_estimator,
)
from readmission.preprocess import make_preprocessor
from readmission.tuning import (
    fold_aucs,
    knn_k_scores,
    one_se_choice,
    tree_complexity_curve,
    tree_depth_scores,
)

Q_RULES = {"q05": 0.05, "q10": 0.10, "q20": 0.20, "q30": 0.30}
assert tuple(Q_RULES.values()) == SENSITIVITY_Q


@dataclass
class Config:
    models: tuple[str, ...] = APPROVED
    extensions: tuple[str, ...] = EXTENSIONS
    youden_models: tuple[str, ...] = APPROVED
    class_weighted: tuple[str, ...] = CLASS_WEIGHTED
    features: tuple[str, ...] = tuple(cohort.FEATURES)
    n_splits: int = splits.N_SPLITS
    n_repeats: int = splits.N_REPEATS
    n_inner: int = 5
    seed: int = 0
    tree_depths: tuple = TREE_DEPTHS
    knn_ks: tuple = KNN_KS
    notes: dict = field(default_factory=dict)


def _xy(df: pd.DataFrame, rows: np.ndarray, features):
    return df.iloc[rows][list(features)]


def run_split(df: pd.DataFrame, split: splits.Split, cfg: Config) -> dict:
    """Inner tuning + final fits + decision rules for one outer split."""
    t0 = time.time()
    y_all = df[cohort.TARGET].to_numpy()
    g_all = df[cohort.GROUP].to_numpy()
    tr, te = split.train, split.test
    y_tr = y_all[tr]
    q = float(y_tr.mean())
    all_models = tuple(cfg.models) + tuple(cfg.extensions)

    # ── 2. inner loop (training fold only) ──
    inner = splits.inner_folds(y_tr, g_all[tr], n_splits=cfg.n_inner,
                               seed=cfg.seed * 1000 + split.repeat * 10 + split.fold)
    n_tr = len(tr)
    tree_oof = np.full((n_tr, len(cfg.tree_depths)), np.nan) if "tree" in cfg.models else None
    knn_oof = np.full((n_tr, len(cfg.knn_ks)), np.nan) if "knn" in cfg.models else None
    other = [m for m in cfg.youden_models if m not in ("tree", "knn")]
    oof = {m: np.full(n_tr, np.nan) for m in other}
    for fit, val in inner:
        pre = make_preprocessor(cfg.features).fit(_xy(df, tr[fit], cfg.features))
        X_fit = pre.transform(_xy(df, tr[fit], cfg.features))
        X_val = pre.transform(_xy(df, tr[val], cfg.features))
        if tree_oof is not None:
            tree_oof[val] = tree_depth_scores(X_fit, y_tr[fit], X_val, cfg.tree_depths,
                                              random_state=cfg.seed)
        if knn_oof is not None:
            knn_oof[val] = knn_k_scores(X_fit, y_tr[fit], X_val, cfg.knn_ks)
        for m in other:
            est = make_estimator(m, random_state=cfg.seed).fit(X_fit, y_tr[fit])
            oof[m][val] = est.predict_proba(X_val)[:, 1]

    tuning, params = {}, {}
    if tree_oof is not None:
        sel = one_se_choice(cfg.tree_depths, fold_aucs(y_tr, tree_oof, inner), prefer="smaller")
        tuning["tree"], params["tree"] = sel, {"max_depth": sel["value"]}
        oof["tree"] = tree_oof[:, sel["index"]]
    if knn_oof is not None:
        sel = one_se_choice(cfg.knn_ks, fold_aucs(y_tr, knn_oof, inner), prefer="larger")
        tuning["knn"], params["knn"] = sel, {"n_neighbors": sel["value"]}
        oof["knn"] = knn_oof[:, sel["index"]]
    thresholds = {m: youden_threshold(y_tr, oof[m]) for m in cfg.youden_models if m in oof}

    # ── 3. final fits on the whole training fold, scores on the test fold ──
    pre = make_preprocessor(cfg.features).fit(_xy(df, tr, cfg.features))
    X_tr = pre.transform(_xy(df, tr, cfg.features))
    X_te = pre.transform(_xy(df, te, cfg.features))
    scores, cw_scores = {}, {}
    for m in all_models:
        est = make_estimator(m, params=params.get(m), random_state=cfg.seed).fit(X_tr, y_tr)
        scores[m] = est.predict_proba(X_te)[:, 1]
    for m in cfg.class_weighted:
        est = make_estimator(m, params=params.get(m), class_weight="balanced",
                             random_state=cfg.seed).fit(X_tr, y_tr)
        cw_scores[m] = est.predict_proba(X_te)[:, 1]

    rng = np.random.default_rng([cfg.seed, split.repeat, split.fold])
    flags = {}
    for m, s in scores.items():
        flags[f"{m}__topq"] = flag_top_q(s, q, rng=rng)
        for rule, share in Q_RULES.items():
            flags[f"{m}__{rule}"] = flag_top_q(s, share, rng=rng)
        if m in thresholds:
            flags[f"{m}__youden"] = s >= thresholds[m]
    for m, s in cw_scores.items():
        flags[f"{m}__cw05"] = s >= 0.5

    return {
        "repeat": split.repeat, "fold": split.fold, "test": te, "q": q,
        "scores": scores, "cw_scores": cw_scores, "flags": flags,
        "tuning": tuning, "thresholds": thresholds, "seconds": round(time.time() - t0, 1),
    }


# Files whose code decides the predictions; a change in any of them invalidates checkpoints.
_PREDICTION_CODE = ("cohort", "splits", "preprocess", "decision", "models", "tuning", "evaluate")


def fingerprint(df: pd.DataFrame, cfg: Config) -> dict:
    """What a checkpoint directory was produced with: settings, code and cohort."""
    here = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for name in _PREDICTION_CODE:
        code.update((here / f"{name}.py").read_bytes())
    rows = hashlib.sha256()
    rows.update(df[cohort.GROUP].astype(str).str.cat(sep="|").encode())
    rows.update(df[cohort.TARGET].to_numpy().astype(np.int8).tobytes())
    return {"config": asdict(cfg), "code_sha256": code.hexdigest(),
            "cohort": {"n_rows": len(df), "n_positive": int(df[cohort.TARGET].sum()),
                       "rows_sha256": rows.hexdigest()}}


def _check_fingerprint(checkpoint_dir: Path, fp: dict, *, write: bool) -> None:
    path = checkpoint_dir / "config.json"
    text = json.dumps(fp, indent=1, default=str)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"{path} was written with a different configuration, code or "
                             "cohort; use a new checkpoint directory instead of mixing results")
    elif write:
        path.write_text(text, encoding="utf-8")
    else:
        raise FileNotFoundError(f"{path} is missing")


def _save(res: dict, path: Path) -> None:
    """Write the npz first and the json last, each via a temp file + os.replace: an
    interrupted write never leaves a checkpoint that looks complete (the json is the marker)."""
    arrays = {"test": res["test"]}
    arrays |= {f"score__{m}": s.astype(np.float32) for m, s in res["scores"].items()}
    arrays |= {f"cwscore__{m}": s.astype(np.float32) for m, s in res["cw_scores"].items()}
    arrays |= {f"flag__{k}": v for k, v in res["flags"].items()}
    tmp_npz = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, path.with_suffix(".npz"))
    meta = {k: res[k] for k in ("repeat", "fold", "q", "tuning", "thresholds", "seconds")}
    tmp_json = path.with_name(path.name + ".tmp.json")
    tmp_json.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    os.replace(tmp_json, path.with_suffix(".json"))


def run(df: pd.DataFrame, cfg: Config, checkpoint_dir: Path, *, log=print) -> None:
    """Run (or resume) all outer splits; one checkpoint pair per split."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _check_fingerprint(checkpoint_dir, fingerprint(df, cfg), write=True)
    y = df[cohort.TARGET].to_numpy()
    g = df[cohort.GROUP].to_numpy()
    for split in splits.outer_splits(y, g, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats,
                                     seed=cfg.seed):
        path = checkpoint_dir / f"r{split.repeat}_f{split.fold}"
        if path.with_suffix(".npz").exists() and path.with_suffix(".json").exists():
            log(f"skip r{split.repeat} f{split.fold} (checkpoint exists)")
            continue
        res = run_split(df, split, cfg)
        _save(res, path)
        tuned = {m: t["value"] for m, t in res["tuning"].items()}
        log(f"done r{split.repeat} f{split.fold} in {res['seconds']}s  q={res['q']:.4f}  tuned={tuned}")


def load_oof(checkpoint_dir: Path, df: pd.DataFrame, cfg: Config) -> dict:
    """Stack the checkpoints into per-repeat out-of-fold arrays (repeat, row).

    Refuses checkpoints written with other settings, code or cohort, checkpoints whose array
    keys differ, and scores that were never filled.

    Returns scores / cw_scores {model: (R, n)}, flags {model__rule: (R, n) bool},
    q_ref (R, n) = training-fold readmission rate of the fold that tested the row.
    """
    checkpoint_dir = Path(checkpoint_dir)
    _check_fingerprint(checkpoint_dir, fingerprint(df, cfg), write=False)
    R, n_rows = cfg.n_repeats, len(df)
    out = {"scores": {}, "cw_scores": {}, "flags": {}, "q_ref": np.full((R, n_rows), np.nan),
           "meta": []}
    keys = None
    for r in range(R):
        for f in range(cfg.n_splits):
            path = checkpoint_dir / f"r{r}_f{f}"
            z = np.load(path.with_suffix(".npz"))
            if keys is None:
                keys = set(z.files)
            elif set(z.files) != keys:
                raise ValueError(f"{path.name}.npz holds different arrays than the first split")
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            out["meta"].append(meta)
            te = z["test"]
            out["q_ref"][r, te] = meta["q"]
            for key in z.files:
                if key == "test":
                    continue
                kind, name = key.split("__", 1)
                bucket = {"score": "scores", "cwscore": "cw_scores", "flag": "flags"}[kind]
                if name not in out[bucket]:
                    dtype = bool if kind == "flag" else np.float32
                    fill = False if kind == "flag" else np.nan
                    out[bucket][name] = np.full((R, n_rows), fill, dtype=dtype)
                out[bucket][name][r, te] = z[key]
    if np.isnan(out["q_ref"]).any():
        raise ValueError("some rows were never tested: incomplete checkpoints")
    for bucket in ("scores", "cw_scores"):
        for name, arr in out[bucket].items():
            if np.isnan(arr).any():
                raise ValueError(f"{bucket}[{name}] has rows without a prediction")
    return out


# 0 (no pruning) plus a log grid over the region where validation AUC moves (the review found
# 1e-6..3e-6 identical to 0 and 1e-2 already a stump).
PRUNING_ALPHAS = (0.0, *np.geomspace(1e-5, 3e-3, 14).round(7).tolist())


def complexity_curves(df: pd.DataFrame, cfg: Config) -> dict:
    """Decision-tree complexity curves for the report (02-dt: overfitting and pruning).

    Uses the first outer split and the first inner fold of its training part, so the test
    fold is not touched. Two knobs: max_depth (pre-pruning) and cost-complexity alpha on a
    fully grown tree (post-pruning). Returns train and validation ROC-AUC along each.
    """
    y = df[cohort.TARGET].to_numpy()
    g = df[cohort.GROUP].to_numpy()
    split = next(splits.outer_splits(y, g, n_splits=cfg.n_splits, n_repeats=1, seed=cfg.seed))
    tr = split.train
    fit, val = splits.inner_folds(y[tr], g[tr], n_splits=cfg.n_inner, seed=cfg.seed * 1000)[0]
    pre = make_preprocessor(cfg.features).fit(_xy(df, tr[fit], cfg.features))
    X_fit = pre.transform(_xy(df, tr[fit], cfg.features))
    X_val = pre.transform(_xy(df, tr[val], cfg.features))
    return {
        "depth": tree_complexity_curve(X_fit, y[tr][fit], X_val, y[tr][val], param="max_depth",
                                       values=cfg.tree_depths, random_state=cfg.seed),
        "ccp_alpha": tree_complexity_curve(X_fit, y[tr][fit], X_val, y[tr][val],
                                           param="ccp_alpha", values=PRUNING_ALPHAS,
                                           random_state=cfg.seed),
    }

