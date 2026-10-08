"""
readmission/mitigation.py — the two mitigations of the approved protocol.

docs/APPROVED_PROPOSAL.md: "Reweighing with Kamiran–Calders weights (compared with simple cell
balancing), and ThresholdOptimizer with an equalized-odds constraint fitted on a separate
patient-level validation split of each training fold. Each is compared with its unconstrained
counterpart under the same decision rule, so the reported accuracy cost comes from the
mitigation and not from a change of threshold. Both mitigations are run for all three
attributes."

Model: the primary model of the fairness audit (fairness.choose_primary), with the settings and
random_state of the main run, on the same 25 outer splits. This module never changes the main
run: it reads its checkpoints and writes its own directory.

Arms per outer split (flag keys "arm__rule", as in readmission/evaluate.py)

  base             the main run's scores and flags for the primary model, copied from its
                   checkpoint, so the baseline here IS the audited baseline. Rules topq, q05..q30.
  rw_kc_{attr}     refitted on the whole training fold with Kamiran–Calders weights
                   w(s, y) = P(s) P(y) / P(s, y): in the weighted training data the attribute and
                   the label are independent. Same rules as base -> compared with base.
  rw_cell_{attr}   the same with cell balancing w(s, y) = N / (n_cells · n(s, y)): every
                   group × label cell gets the same total weight (this also balances the classes).
  val              the model refitted on 80 % of the training fold (patient-level split,
                   splits.validation_split); the other 20 % is the validation part.
                   val__topq    the capacity rule on this model (context for eocap).
                   val__youden  one threshold for everyone that maximises balanced accuracy on the
                                validation part (= Youden's J): the counterpart of to.
  eocap_{attr}     equalized-odds post-processing AT THE CAPACITY RULE (Hardt et al. 2016, the
                   method behind fairlearn's ThresholdOptimizer, solved on the line "q % flagged"):
                   on the validation part, the (FPR, TPR) point that every group can reach and
                   that flags q % with the most true positives; each group reaches it with its
                   own randomised thresholds. Rule "eo".
  to_{attr}        fairlearn ThresholdOptimizer on the `val` model: constraint equalized odds,
                   objective balanced accuracy, fitted on the validation part. Rule "eo".
  toacc_{attr}     the same with fairlearn's default objective, accuracy (diagnostic arm).
  eocaptwin_{attr}, totwin_{attr}
                   the `val` model flagging its top scores, exactly as many per fold as the
                   eocap / to arm flagged. Rule "samek". eocap flags q % in expectation on the
                   validation part but somewhat fewer on test folds (r0 f0: 0.106 vs 0.114), and
                   with PPV < 0.5 flagging fewer alone raises accuracy; the twin removes the
                   number flagged from the comparison, leaving only who is flagged.

Why two post-processing arms: fairlearn has no fixed-capacity objective. With its default
(accuracy) it flags nobody here: acc − NIR = (2 TP − k) / n, so flagging only adds accuracy where
PPV > 0.5, which no score band reaches at 11 % prevalence (checked on split r0 f0: 0 flagged).
With balanced accuracy it flags ~28–30 %, not the q % the hospital can follow up. `to` is the
approved text taken literally (counterpart: the Youden threshold, the unconstrained
balanced-accuracy optimum); `eocap` is the same constraint at the protocol's operating point.

Groups used to FIT the post-processors (rule written down on 23 Sep 2026, after an independent
review of split r0 f0 and before the full run): a level with fewer than 50 positives in the
validation part, and the missing-value code (race "Unknown"), is pooled into one level; if the
pool itself has fewer than 50 positives it joins the largest level. Without this, race's
constraint was set by Asian with 11 validation positives. The audit still reports every level;
the pooled levels are equalized only as a pool. Reweighing uses the raw levels (weights come from
training-fold counts, where every cell is populated).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import fairlearn
import numpy as np
import pandas as pd
import sklearn
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import roc_curve

from readmission import cohort, evaluate, fairness, metrics, splits
from readmission.decision import flag_top_q, youden_threshold
from readmission.models import TUNED, make_estimator
from readmission.preprocess import make_preprocessor

REWEIGH_MODES = ("kc", "cell")
TO_OBJECTIVES = {"to": "balanced_accuracy_score", "toacc": "accuracy_score"}
SCORE_RULES = ("topq", *evaluate.Q_RULES)
POOLED = "pooled"


@dataclass
class MitigationConfig:
    primary: str
    attributes: tuple[str, ...] = fairness.MAIN_ATTRIBUTES
    val_fraction: float = 0.2
    min_fit_positives: int = fairness.MIN_POSITIVES


# ── reweighing ────────────────────────────────────────────────────────────────────────────


def reweigh_weights(y, s, mode: str) -> np.ndarray:
    """Sample weights from the attribute × label cells of the training rows (mean 1).

    kc:   n(s) n(y) / (N n(s, y)) = P(s) P(y) / P(s, y)   (Kamiran & Calders 2012)
    cell: N / (n_cells n(s, y))                          (equal total weight per cell)
    """
    if mode not in REWEIGH_MODES:
        raise ValueError(f"mode must be one of {REWEIGH_MODES}, got {mode!r}")
    cells = pd.DataFrame({"s": np.asarray(s).astype(str), "y": np.asarray(y)})
    table = pd.crosstab(cells["s"], cells["y"])
    if (table == 0).to_numpy().any():  # an empty cell has no weight to carry
        raise ValueError(f"empty attribute × label cell:\n{table}")
    n = len(cells)
    n_sy = cells.groupby(["s", "y"])["y"].transform("size").to_numpy(float)
    if mode == "kc":
        n_s = cells.groupby("s")["y"].transform("size").to_numpy(float)
        n_y = cells.groupby("y")["y"].transform("size").to_numpy(float)
        return n_s * n_y / (n * n_sy)
    return n / (table.size * n_sy)


# ── post-processing ───────────────────────────────────────────────────────────────────────


def fit_groups(values, y, *, min_positives: int, not_a_group=()) -> tuple[dict, str]:
    """Level -> level used for fitting (see the module docstring), and the level for unseen ones."""
    v = pd.Series(np.asarray(values).astype(str))
    pos = pd.Series(np.asarray(y)).groupby(v).sum()
    small = [g for g in pos.index if pos[g] < min_positives or g in not_a_group]
    if len(small) == len(pos):  # nothing to equalize: one level for everyone
        return {g: POOLED for g in pos.index}, POOLED
    largest = v[~v.isin(small)].value_counts().idxmax()
    if not small:
        return {g: g for g in pos.index}, largest
    target = POOLED if pos[small].sum() >= min_positives else largest
    return {g: (target if g in small else g) for g in pos.index}, target


def _map(values, mapping: dict, default: str) -> np.ndarray:
    return pd.Series(np.asarray(values).astype(str)).map(mapping).fillna(default).to_numpy()


def roc_hull(y, score) -> np.ndarray:
    """Upper convex hull of a group's ROC points: rows (FPR, TPR, threshold), FPR increasing.

    Randomising between two thresholds reaches every point on the segment between them, so the
    hull is what the group can achieve. "Flag if score >= threshold"; threshold inf = flag nobody.
    """
    fpr, tpr, thr = roc_curve(y, score, drop_intermediate=False)
    hull: list[tuple[float, float, float]] = []
    for p in sorted(zip(fpr, tpr, thr, strict=True), key=lambda p: (p[0], p[1])):
        # drop the last vertex while it lies on or below the line from its neighbour to p
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(p)
    h = np.array(hull)
    return h[np.r_[np.diff(h[:, 0]) > 0, True]]  # one vertex per FPR (the highest TPR)


def fit_eo_at_capacity(y, score, groups, q: float) -> dict:
    """Equalized-odds post-processing that flags a share q (Hardt et al. 2016, on the line
    p1·TPR + p0·FPR = q).

    Every group must reach the same (FPR, TPR) point; a group can reach any point on or below
    its ROC hull. The best point on the capacity line is where the line meets the lowest of the
    group hulls (more FPR means fewer true positives on that line). Groups whose hull lies above
    that point get there by ignoring the score with probability p_ignore and flagging at the
    common FPR instead (a coin flip that lowers TPR without changing FPR).
    """
    y, score, groups = np.asarray(y), np.asarray(score, float), np.asarray(groups)
    hulls = {g: roc_hull(y[groups == g], score[groups == g]) for g in np.unique(groups)}
    p1 = y.mean()

    def envelope(f):
        return min(np.interp(f, h[:, 0], h[:, 1]) for h in hulls.values())

    lo, hi = 0.0, min(1.0, q / (1 - p1))  # the line's TPR is q/p1 at FPR 0, 0 at FPR q/p0
    for _ in range(60):
        mid = (lo + hi) / 2
        if envelope(mid) < (q - (1 - p1) * mid) / p1:
            lo = mid
        else:
            hi = mid
    f_star = hi
    t_star = (q - (1 - p1) * f_star) / p1
    out = {"fpr": f_star, "tpr": t_star, "groups": {}}
    for g, h in hulls.items():
        j = int(np.clip(np.searchsorted(h[:, 0], f_star, side="right") - 1, 0, len(h) - 2))
        (f_lo, t_lo, thr_lo), (f_hi, t_hi, thr_hi) = h[j], h[j + 1]
        lam = (f_star - f_lo) / (f_hi - f_lo)
        t_hull = t_lo + lam * (t_hi - t_lo)
        p_ignore = (t_hull - t_star) / (t_hull - f_star) if t_hull > f_star else 0.0
        out["groups"][g] = {"thr_lo": thr_lo, "thr_hi": thr_hi, "lam": lam,
                            "p_ignore": float(np.clip(p_ignore, 0, 1))}
    return out


def eo_flag_probability(fit: dict, score, groups) -> np.ndarray:
    """Probability that each encounter is flagged under a fit_eo_at_capacity solution."""
    score, groups = np.asarray(score, float), np.asarray(groups)
    p = np.zeros(len(score))
    for g, d in fit["groups"].items():
        rows = groups == g
        s = score[rows]
        by_score = (1 - d["lam"]) * (s >= d["thr_lo"]) + d["lam"] * (s >= d["thr_hi"])
        p[rows] = (1 - d["p_ignore"]) * by_score + d["p_ignore"] * fit["fpr"]
    return p


# ── one outer split ───────────────────────────────────────────────────────────────────────


def _x(df: pd.DataFrame, rows: np.ndarray, features) -> pd.DataFrame:
    return df.iloc[rows][list(features)]


def run_split(df: pd.DataFrame, split: splits.Split, cfg: evaluate.Config,
              mcfg: MitigationConfig, main_dir: Path) -> dict:
    """All mitigation arms for one outer split (see the module docstring)."""
    t0 = time.time()
    m = mcfg.primary
    y_all = df[cohort.TARGET].to_numpy()
    g_all = df[cohort.GROUP].to_numpy()
    tr, te = split.train, split.test
    y_tr = y_all[tr]
    main = np.load(main_dir / f"r{split.repeat}_f{split.fold}.npz")
    meta = json.loads((main_dir / f"r{split.repeat}_f{split.fold}.json").read_text(encoding="utf-8"))
    if not np.array_equal(main["test"], te):
        raise ValueError("outer split differs from the main run's checkpoint")
    q = meta["q"]
    params = {TUNED[m]: meta["tuning"][m]["value"]} if m in meta["tuning"] else None
    rng = np.random.default_rng([cfg.seed, split.repeat, split.fold, 1])  # not the main run's stream
    to_seed = cfg.seed * 1000 + split.repeat * 10 + split.fold

    scores = {"base": main[f"score__{m}"].astype(float)}
    flags = {f"base__{rule}": main[f"flag__{m}__{rule}"] for rule in SCORE_RULES}

    def score_rules(arm: str, s: np.ndarray) -> None:
        scores[arm] = s
        flags[f"{arm}__topq"] = flag_top_q(s, q, rng=rng)
        for rule, share in evaluate.Q_RULES.items():
            flags[f"{arm}__{rule}"] = flag_top_q(s, share, rng=rng)

    # ── reweighing: whole training fold, one refit per attribute and weight mode ──
    pre = make_preprocessor(cfg.features).fit(_x(df, tr, cfg.features))
    X_tr, X_te = pre.transform(_x(df, tr, cfg.features)), pre.transform(_x(df, te, cfg.features))
    for attr in mcfg.attributes:
        s_tr = df[attr].to_numpy()[tr]
        for mode in REWEIGH_MODES:
            est = make_estimator(m, params=params, random_state=cfg.seed)
            est.fit(X_tr, y_tr, sample_weight=reweigh_weights(y_tr, s_tr, mode))
            score_rules(f"rw_{mode}_{attr}", est.predict_proba(X_te)[:, 1])

    # ── post-processing: fit on 80 % of the training fold, thresholds on the other 20 % ──
    fit_pos, val_pos = splits.validation_split(
        y_tr, g_all[tr], val_fraction=mcfg.val_fraction,
        seed=cfg.seed * 1000 + split.repeat * 10 + split.fold + 500)
    fit_rows, val_rows = tr[fit_pos], tr[val_pos]
    y_val = y_all[val_rows]
    pre = make_preprocessor(cfg.features).fit(_x(df, fit_rows, cfg.features))
    X_fit, X_val, X_te = (pre.transform(_x(df, rows, cfg.features)) for rows in (fit_rows, val_rows, te))
    est = make_estimator(m, params=params, random_state=cfg.seed).fit(X_fit, y_all[fit_rows])
    s_val, s_te = est.predict_proba(X_val)[:, 1], est.predict_proba(X_te)[:, 1]
    threshold = youden_threshold(y_val, s_val)
    scores["val"] = s_te
    flags["val__youden"] = s_te >= threshold
    flags["val__topq"] = flag_top_q(s_te, q, rng=rng)

    diagnostics = {}
    for attr in mcfg.attributes:
        mapping, default = fit_groups(df[attr].to_numpy()[val_rows], y_val,
                                      min_positives=mcfg.min_fit_positives,
                                      not_a_group=fairness.NOT_A_GROUP.get(attr, ()))
        grp_val = _map(df[attr].to_numpy()[val_rows], mapping, default)
        grp_te = _map(df[attr].to_numpy()[te], mapping, default)
        cap = fit_eo_at_capacity(y_val, s_val, grp_val, q)
        flags[f"eocap_{attr}__eo"] = rng.random(len(te)) < eo_flag_probability(cap, s_te, grp_te)
        diag = {"fit_groups": mapping,
                "val_positives": {g: int(y_val[grp_val == g].sum()) for g in np.unique(grp_val)},
                "eocap": {"fpr": cap["fpr"], "tpr": cap["tpr"],
                          "p_ignore": {g: d["p_ignore"] for g, d in cap["groups"].items()}}}
        for name, objective in TO_OBJECTIVES.items():
            to = ThresholdOptimizer(estimator=est, constraints="equalized_odds", objective=objective,
                                    prefit=True, predict_method="predict_proba")
            to.fit(X_val, y_val, sensitive_features=grp_val)
            flags[f"{name}_{attr}__eo"] = to.predict(X_te, sensitive_features=grp_te,
                                                     random_state=to_seed).astype(bool)
            interp = to.interpolated_thresholder_.interpolation_dict
            diag[name] = {"p_ignore": {str(g): float(d["p_ignore"]) for g, d in interp.items()}}
        for name in ("eocap", "to"):  # same number flagged, unconstrained model
            flags[f"{name}twin_{attr}__samek"] = flag_top_q(s_te, flags[f"{name}_{attr}__eo"].mean(),
                                                           rng=rng)
        diagnostics[attr] = diag

    return {"repeat": split.repeat, "fold": split.fold, "test": te, "q": q,
            "scores": scores, "cw_scores": {}, "flags": flags,
            # "tuning" = what was fitted on the validation part (json meta of the checkpoint)
            "tuning": {"postprocessing": diagnostics},
            "thresholds": {"val__youden": threshold}, "seconds": round(time.time() - t0, 1)}


# ── run, resume, load ─────────────────────────────────────────────────────────────────────


def fingerprint(df: pd.DataFrame, cfg: evaluate.Config, mcfg: MitigationConfig) -> dict:
    """The main run's fingerprint plus this module's settings, code and library versions."""
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"main": evaluate.fingerprint(df, cfg), "mitigation": asdict(mcfg),
            "mitigation_code_sha256": code,
            "versions": {"fairlearn": fairlearn.__version__, "scikit-learn": sklearn.__version__}}


def run(df: pd.DataFrame, cfg: evaluate.Config, mcfg: MitigationConfig, main_dir: Path,
        out_dir: Path, *, log=print) -> None:
    """Run (or resume) every outer split; one checkpoint pair per split, as the main run."""
    if mcfg.primary == "knn":
        raise ValueError("k-NN takes no sample weights, so reweighing is undefined for it")
    main_dir, out_dir = Path(main_dir), Path(out_dir)
    evaluate._check_fingerprint(main_dir, evaluate.fingerprint(df, cfg), write=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate._check_fingerprint(out_dir, fingerprint(df, cfg, mcfg), write=True)
    y, g = df[cohort.TARGET].to_numpy(), df[cohort.GROUP].to_numpy()
    for split in splits.outer_splits(y, g, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats,
                                     seed=cfg.seed):
        path = out_dir / f"r{split.repeat}_f{split.fold}"
        if path.with_suffix(".npz").exists() and path.with_suffix(".json").exists():
            log(f"skip r{split.repeat} f{split.fold} (checkpoint exists)")
            continue
        res = run_split(df, split, cfg, mcfg, main_dir)
        evaluate._save(res, path)
        log(f"done r{split.repeat} f{split.fold} in {res['seconds']}s")


def load(out_dir: Path, df: pd.DataFrame, cfg: evaluate.Config, mcfg: MitigationConfig) -> dict:
    """Per-repeat out-of-fold arrays, in the layout of evaluate.load_oof (same checks)."""
    out_dir = Path(out_dir)
    evaluate._check_fingerprint(out_dir, fingerprint(df, cfg, mcfg), write=False)
    R, n = cfg.n_repeats, len(df)
    out = {"scores": {}, "cw_scores": {}, "flags": {}, "q_ref": np.full((R, n), np.nan), "meta": []}
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


# ── paired differences ────────────────────────────────────────────────────────────────────


def comparisons(attributes=fairness.MAIN_ATTRIBUTES, rules=("topq",)) -> dict:
    """{name: (mitigated key, counterpart key)} — each arm against its unconstrained twin."""
    out = {}
    for attr in attributes:
        for mode in REWEIGH_MODES:
            for rule in rules:
                out[f"rw_{mode}_{attr}@{rule}"] = (f"rw_{mode}_{attr}__{rule}", f"base__{rule}")
        out[f"kc_vs_cell_{attr}@topq"] = (f"rw_kc_{attr}__topq", f"rw_cell_{attr}__topq")
        out[f"eocap_{attr}"] = (f"eocap_{attr}__eo", f"eocaptwin_{attr}__samek")
        out[f"eocap_vs_topq_{attr}"] = (f"eocap_{attr}__eo", "val__topq")
        out[f"to_{attr}"] = (f"to_{attr}__eo", "val__youden")
        out[f"to_samek_{attr}"] = (f"to_{attr}__eo", f"totwin_{attr}__samek")
    out["holdout@topq"] = ("val__topq", "base__topq")  # context: the cost of the 80 % refit
    return out


def paired_effects(df: pd.DataFrame, oof: dict, comps: dict, *, n_boot: int,
                   rng: np.random.Generator) -> dict:
    """Mitigated − counterpart differences with patient-bootstrap CIs.

    Both sides of a difference are computed on the same bootstrap draw, so the CI is that of the
    difference itself. Per comparison: the overall decision metrics, ROC-AUC when both sides have
    scores, and the equalized-odds difference of every main attribute (the targeted one and the
    spill-over on the others). The EO difference uses the audit's rules: repeat-averaged group
    rates, groups with < 50 positives (TPR) / < 50 negatives (FPR) and race "Unknown" left out.
    Randomised arms (eocap, to) are fixed per split: their coin flips are averaged over the
    repeats but not resampled by the bootstrap.
    """
    y = df[cohort.TARGET].to_numpy().astype(np.float32)
    pairs, M = fairness._flat(fairness.group_masks(df), fairness.MAIN_ATTRIBUTES)
    n_pos = (M * y).sum(axis=1)
    usable = {"tpr": n_pos >= fairness.MIN_POSITIVES,
              "fpr": M.sum(axis=1) - n_pos >= fairness.MIN_NEGATIVES}
    idx = {(a, m): fairness._usable(pairs, usable, a, m)
           for a in fairness.MAIN_ATTRIBUTES for m in ("tpr", "fpr")}
    codes, n_pat = metrics.patient_codes(df[cohort.GROUP])
    keys = sorted({k for pair in comps.values() for k in pair})
    flags = {k: oof["flags"][k].astype(np.float32) for k in keys}
    arms = {k: k.split("__")[0] for k in keys}
    scored = sorted({a for a in arms.values() if a in oof["scores"]})
    y64 = y.astype(float)

    def stats(W: np.ndarray) -> dict:
        auc = {a: np.mean([metrics.weighted_auc(W, y64, s.astype(float)) for s in oof["scores"][a]],
                          axis=0) for a in scored}
        out = {}
        for k in keys:
            per_rep = [metrics.weighted_decision(W, y, f) for f in flags[k]]
            d = {name: np.nanmean([p[name] for p in per_rep], axis=0)
                 for name in ("accuracy", "tpr", "fpr", "ppv", "selection_rate")}
            d["balanced_accuracy"] = (d["tpr"] + 1 - d["fpr"]) / 2
            r = fairness._rates(W, M, y, flags[k])
            for a in fairness.MAIN_ATTRIBUTES:
                d[f"eo_{a}"] = np.fmax(fairness._range(r["tpr"], idx[(a, "tpr")]),
                                       fairness._range(r["fpr"], idx[(a, "fpr")]))
            if arms[k] in auc:
                d["roc_auc"] = auc[arms[k]]
            out[k] = d
        return out

    point = stats(np.ones((1, len(y)), dtype=np.float32))
    draws = [stats(W) for W in metrics.bootstrap_weights(codes, n_pat, n_boot, rng)]
    result = {}
    for name, (k, c) in comps.items():
        result[name] = {"mitigated": k, "counterpart": c}
        for stat in point[k]:
            if stat not in point[c]:
                continue
            diff = np.concatenate([d[k][stat] - d[c][stat] for d in draws])
            result[name][stat] = {"mitigated": float(point[k][stat][0]),
                                  "counterpart": float(point[c][stat][0]),
                                  "difference": float(point[k][stat][0] - point[c][stat][0]),
                                  **metrics._interval(diff)}
    return result
