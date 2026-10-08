"""
scripts/run_mitigation.py — reweighing and equalized-odds post-processing on the primary model.

Needs a finished `python scripts/run_approved.py` and `python scripts/run_fairness.py` (the
latter for the leading attribute; without it the report keeps the order sex, age band, race).

    python scripts/run_mitigation.py            # 25 splits, 1000 patient-bootstrap draws
    python scripts/run_mitigation.py --quick    # on the --quick main run, 200 draws
    python scripts/run_mitigation.py --n-jobs 4 # cap the CPU threads (default: half)

Writes
    outputs/approved[_quick]/mitigation/r*_f*.npz|json   one pair per outer split (resumable)
    outputs/approved[_quick]/mitigation.json             paired differences, group metrics
    outputs/reports/approved_mitigation[_quick].md       tables for the paper (aggregates only)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from readmission import cohort, evaluate, fairness, metrics, mitigation, runtime  # noqa: E402


def audit_keys(attributes) -> list[str]:
    """The flag keys that get group metrics and the overall summary."""
    keys = ["base__topq", "val__topq", "val__youden"]
    for a in attributes:
        keys += [f"rw_kc_{a}__topq", f"rw_cell_{a}__topq", f"eocap_{a}__eo", f"to_{a}__eo",
                 f"toacc_{a}__eo", f"eocaptwin_{a}__samek", f"totwin_{a}__samek"]
    return keys


def postprocessing_summary(meta: list[dict], attributes) -> dict:
    """Across splits: which levels were pooled for fitting, and which fitted group bound."""
    out = {}
    for a in attributes:
        diags = [m["tuning"]["postprocessing"][a] for m in meta]
        pooled = Counter(f"{g} → {t}" for d in diags for g, t in d["fit_groups"].items() if t != g)
        binding = {name: Counter(min(d[name]["p_ignore"], key=d[name]["p_ignore"].get) for d in diags)
                   for name in ("eocap", "to")}
        out[a] = {"n_splits": len(diags), "pooled": dict(pooled),
                  "binding": {k: dict(v) for k, v in binding.items()},
                  "min_val_positives": min(min(d["val_positives"].values()) for d in diags)}
    return out


def _ci(d: dict, digits: int = 3, key: str = "estimate") -> str:
    lo, hi = d["ci"]
    mark = " †" if not d.get("ci_reliable", True) else ""
    return f"{d[key]:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{mark}"


def _pt(d: dict, digits: int = 3) -> str:
    lo, hi = d["ci"]
    return f"{d['estimate']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _order(lead: str | None) -> list[str]:
    attrs = list(fairness.MAIN_ATTRIBUTES)
    if lead in attrs:
        attrs.remove(lead)
        attrs.insert(0, lead)
    return attrs


def build_report(res: dict, primary: str, lead: str | None, timing: str) -> str:
    eff, audit, summ = res["effects"], res["audit"], res["summary"]["decision"]
    sel = audit["overall_selection"]
    attrs = _order(lead)

    def diff(name, stat, digits=3):
        return _ci(eff[name][stat], digits=digits, key="difference")

    def arrow(name, stat, digits=3):
        e = eff[name][stat]
        return f"{e['counterpart']:.{digits}f} → {e['mitigated']:.{digits}f}"

    lines = [
        "# Approved protocol — mitigations",
        "",
        f"Model: **{primary}** (the primary model of the audit). Leading attribute: "
        f"**{lead or 'not available'}**; sex is shown for comparison and every mitigation is run "
        f"for all three attributes. Patient-bootstrap draws {res['n_boot']}, repeats "
        f"{res['n_repeats']}, {timing}.",
        "",
        "Every mitigated arm is compared with its unconstrained twin under the same decision "
        "rule, on the same bootstrap draw, so each CI is that of the difference itself "
        "(mitigated − unconstrained). EO = equalized-odds difference of the attribute with the "
        "audit's rules (groups with < 50 positives / negatives and race 'Unknown' left out). "
        "At a fixed number flagged, accuracy, TPR, FPR and PPV all move with the number of true "
        "positives alone (Δ accuracy = 2 ΔTP / n), so the tables show Δ accuracy and Δ TPR.",
        "",
        "## 1. Reweighing at the capacity rule (top q %, q = training-fold rate)",
        "",
        "Refitted on the whole training fold with Kamiran–Calders weights (attribute and label "
        "independent in the weighted data) or cell balancing (equal weight per group × label "
        "cell); flagged with the same rule and the same number per fold as the baseline.",
        "",
    ]
    rows = []
    for a in attrs:
        for mode, label in (("kc", "Kamiran–Calders"), ("cell", "cell balancing")):
            n = f"rw_{mode}_{a}@topq"
            rows.append([a, label, arrow(n, f"eo_{a}"), diff(n, f"eo_{a}"), diff(n, "accuracy", 4),
                         diff(n, "tpr"), diff(n, "roc_auc", 4)])
    lines += _table(["Attribute", "Weights", "EO (baseline → reweighed)", "Δ EO [95 % CI]",
                     "Δ accuracy", "Δ TPR", "Δ ROC-AUC"], rows)
    lines += ["", "Kamiran–Calders − cell balancing (same rule):", ""]
    rows = [[a, diff(f"kc_vs_cell_{a}@topq", f"eo_{a}"), diff(f"kc_vs_cell_{a}@topq", "accuracy", 4),
             diff(f"kc_vs_cell_{a}@topq", "roc_auc", 4)] for a in attrs]
    lines += _table(["Attribute", "Δ EO", "Δ accuracy", "Δ ROC-AUC"], rows)

    pp = res["postprocessing"]
    cap = res["eocap_share"]
    pooled_note = (" Levels pooled for fitting (table in section 3) are equalized only as a pool, "
                   "while the EO here is the audit's, level by level: part of a remaining gap "
                   "on such an attribute is by design.")
    lines += ["", "## 2. Equalized-odds post-processing at the capacity rule", "",
              "The model is refitted on 80 % of the training fold; the other 20 % (patient-level "
              "validation part) chooses the thresholds, which are applied to the test fold. "
              "Mitigated: every group gets the same TPR and FPR on the validation part while q % "
              "are flagged, with the most true positives possible (Hardt et al. 2016, solved on "
              "the capacity line; groups get their own randomised thresholds). On the test folds "
              f"it flags {cap['mean']:.3f} on average (per fold {cap['min']:.3f}–{cap['max']:.3f}, "
              f"q = {cap['q']:.4f}); at PPV < 0.5 flagging fewer alone raises accuracy and flagging "
              "more lowers it, so the unconstrained twin is the same model flagging its top "
              "scores, exactly as many per fold. The last column compares with the plain top q % "
              "rule." + pooled_note, ""]
    rows = []
    for a in attrs:
        n = f"eocap_{a}"
        rows.append([a, arrow(n, f"eo_{a}"), diff(n, f"eo_{a}"), diff(n, "accuracy", 4),
                     diff(n, "tpr"), f"{eff[n]['selection_rate']['mitigated']:.3f}",
                     diff(f"eocap_vs_topq_{a}", f"eo_{a}")])
    lines += _table(["Attribute", "EO (twin → EO post-proc.)", "Δ EO [95 % CI]", "Δ accuracy",
                     "Δ TPR", "Share flagged", "Δ EO vs top q %"], rows)

    lines += ["", "## 3. fairlearn ThresholdOptimizer (approved text) vs one threshold for everyone",
              "", "Same 80 % model and validation part. Objective balanced accuracy for both: the "
              "single threshold is its unconstrained optimum (Youden's J), ThresholdOptimizer its "
              "optimum under equalized odds. Both flag far more than the capacity q %, and not the "
              "same number, so their shared objective (balanced accuracy) is the first comparison; "
              "the last two columns compare TO with the same model flagging its top scores, "
              "exactly as many per fold (accuracy moves with the number flagged)." + pooled_note,
              ""]
    rows = []
    for a in attrs:
        n, k = f"to_{a}", f"to_samek_{a}"
        rows.append([a, diff(n, "balanced_accuracy"), arrow(n, f"eo_{a}"), diff(n, f"eo_{a}"),
                     arrow(n, "selection_rate"), diff(k, "accuracy"), diff(k, f"eo_{a}")])
    lines += _table(["Attribute", "Δ balanced accuracy", "EO (single → TO)", "Δ EO [95 % CI]",
                     "Selection rate", "Δ accuracy vs same-number twin",
                     "Δ EO vs same-number twin"], rows)
    toacc = max(sel[f"toacc_{a}__eo"] for a in attrs)
    ho = eff["holdout@topq"]
    lines += ["",
              f"With fairlearn's default objective (accuracy) the largest selection rate over the "
              f"three attributes is {toacc:.4f}: flagging only adds accuracy where PPV > 0.5 "
              f"(acc − NIR = (2 TP − k) / n), so the accuracy optimum flags "
              f"{'nobody' if toacc == 0 else 'almost nobody'} and meets equalized odds trivially.",
              "",
              f"Cost of holding out 20 % for the thresholds (top q %, 80 % model − full model): "
              f"Δ accuracy {diff('holdout@topq', 'accuracy', 4)}, Δ ROC-AUC "
              f"{_ci(ho['roc_auc'], digits=4, key='difference')}.",
              "",
              "Groups used to fit the post-processors (rule fixed before the full run: levels with "
              f"< {res['config']['min_fit_positives']} validation positives, and race 'Unknown', "
              "are pooled):", ""]
    rows = []
    for a in attrs:
        d = pp[a]
        pooled = ", ".join(f"{g} ({c}/{d['n_splits']})" for g, c in sorted(d["pooled"].items())) or "none"
        bind = "; ".join(f"{k}: " + ", ".join(f"{g} {c}" for g, c in v.items())
                         for k, v in d["binding"].items())
        rows.append([a, pooled, d["min_val_positives"], bind])
    lines += _table(["Attribute", "Merged level → fitted level (splits)",
                     "Fewest validation positives in a "
                     "fitted group", "Binding group (splits)"], rows)
    lines += ["", "Post-processing flags are random (coin flips inside groups); they are fixed "
              "per split and averaged over the repeats, and the bootstrap does not redraw them."]

    lines += ["", "## 4. Spill-over: Δ EO of every attribute (mitigated − same-number twin)", "",
              "Mitigating one attribute can move the gaps of the others. Paired differences, "
              "95 % CI.", ""]
    rows = []
    for a in attrs:
        for label, n in ((f"K&C on {a}", f"rw_kc_{a}@topq"), (f"cell on {a}", f"rw_cell_{a}@topq"),
                         (f"EO at capacity on {a}", f"eocap_{a}"), (f"TO on {a}", f"to_samek_{a}")):
            rows.append([label] + [diff(n, f"eo_{b}") for b in attrs])
    lines += _table(["Arm"] + [f"Δ EO {b}" for b in attrs], rows)

    for i, a in enumerate(dict.fromkeys([attrs[0], "sex"])):
        lines += ["", f"## 5.{i + 1} Group rates by {a} (TPR / FPR)", ""]
        keys = [("baseline", "base__topq"), ("K&C", f"rw_kc_{a}__topq"),
                ("cell", f"rw_cell_{a}__topq"), ("top q % (80 %)", "val__topq"),
                ("EO at capacity", f"eocap_{a}__eo"), ("single thr.", "val__youden"),
                ("TO", f"to_{a}__eo")]
        rows = []
        for g, d in audit["groups"][a].items():
            note = " (not a group)" if g in fairness.NOT_A_GROUP.get(a, ()) else ""
            cells = []
            for _, k in keys:
                mm = d["metrics"][k]
                mark = " ‡" if mm["tpr"]["insufficient"] else ""
                cells.append(f"{mm['tpr']['estimate']:.3f} / {mm['fpr']['estimate']:.3f}{mark}")
            rows.append([g + note, f"{d['n_pos']:,}"] + cells)
        rows.append(["share flagged", ""] + [f"{sel[k]:.3f}" for _, k in keys])
        lines += _table(["Group", "Readmitted"] + [lbl for lbl, _ in keys], rows)
        lo, hi = sorted(sel[k] for _, k in keys[-2:])
        lines += ["", f"The first five columns flag about q % (capacity); the last two flag "
                  f"{lo:.3f}–{hi:.3f} (balanced-accuracy objective): compare within each set."]
    lines += ["", "‡ fewer than 50 positives: TPR reported, left out of the gaps. (not a group) = "
              "missing-value code, left out of the gaps."]

    lines += ["", "## 6. Reweighing under the other capacities (sensitivity)", "",
              "Δ EO of the reweighed attribute / Δ accuracy, mitigated − baseline, point "
              "estimates.", ""]
    rules = list(mitigation.SCORE_RULES)
    rows = []
    for a in attrs:
        for mode in mitigation.REWEIGH_MODES:
            rows.append([a, mode] + [f"{eff[f'rw_{mode}_{a}@{r}'][f'eo_{a}']['difference']:+.3f} / "
                                     f"{eff[f'rw_{mode}_{a}@{r}']['accuracy']['difference']:+.4f}"
                                     for r in rules])
    lines += _table(["Attribute", "Weights"] + rules, rows)

    lines += ["", "## 7. Usability gate for the arms", "",
              "Lower CI of (accuracy − 0.886) above zero = usable (docs/APPROVED_PROPOSAL.md).", ""]
    rows = []
    gate = [("baseline, top q %", "base__topq"), ("top q %, 80 % model", "val__topq"),
            ("single threshold", "val__youden")]
    for a in attrs:
        gate += [(f"K&C on {a}", f"rw_kc_{a}__topq"), (f"cell on {a}", f"rw_cell_{a}__topq"),
                 (f"EO at capacity on {a}", f"eocap_{a}__eo"), (f"TO on {a}", f"to_{a}__eo")]
    for label, k in gate:
        d = summ[k]
        rows.append([label, f"{sel[k]:.3f}", _pt(d["accuracy"]), _ci(d["acc_minus_nir"]),
                     "yes" if d["usable"] else "no"])
    lines += _table(["Arm", "Share flagged", "Accuracy", "Accuracy − NIR", "Usable"], rows)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--primary", default=None, help="override the audit's primary model")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="CPU threads for fitting and the bootstrap; changes the speed, not the "
                         "reported results (default: half of the CPUs, see readmission/runtime.py)")
    args = ap.parse_args(argv)
    print(f"using {runtime.cap_threads(args.n_jobs)} CPU threads", flush=True)

    tag = "approved_quick" if args.quick else "approved"
    out_dir = ROOT / "outputs" / tag
    main_ckpt, mit_ckpt = out_dir / "checkpoints", out_dir / "mitigation"
    saved = json.loads((main_ckpt / "config.json").read_text(encoding="utf-8"))
    cfg = evaluate.Config(**saved["config"])
    n_boot = args.n_boot or (200 if args.quick else 1000)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    primary = args.primary or fairness.choose_primary(summary, tuple(cfg.models))
    lead = None
    fair_path = out_dir / "fairness.json"
    if fair_path.exists():
        fair = json.loads(fair_path.read_text(encoding="utf-8"))
        if fair.get("primary_model") == primary:  # the lead was found on this model's gaps
            lead = fair.get("leading_attribute")
    mcfg = mitigation.MitigationConfig(primary=primary)

    t0 = time.time()
    df = cohort.load_cohort()
    if not args.summarize_only:
        print(f"mitigations on {primary}; leading attribute {lead}", flush=True)
        mitigation.run(df, cfg, mcfg, main_ckpt, mit_ckpt, log=lambda s: print(s, flush=True))
    oof = mitigation.load(mit_ckpt, df, cfg, mcfg)
    fit_min = sum(m["seconds"] for m in oof["meta"]) / 60

    keys = audit_keys(mcfg.attributes)
    y, groups = df[cohort.TARGET].to_numpy(), df[cohort.GROUP].to_numpy()
    rng = np.random.default_rng([cfg.seed, 4040])
    print(f"paired differences with {n_boot} draws ...", flush=True)
    effects = mitigation.paired_effects(df, oof, mitigation.comparisons(rules=mitigation.SCORE_RULES),
                                        n_boot=n_boot, rng=rng)
    print("group metrics ...", flush=True)
    audit = fairness.audit(df, oof, keys=keys, calibration_models=[], n_boot=n_boot,
                           rng=rng, null_keys=[])
    audit["overall_selection"] = {k: float(oof["flags"][k].mean()) for k in keys}
    print("overall metrics ...", flush=True)
    overall = metrics.summarize(y, groups, {**oof, "flags": {k: oof["flags"][k] for k in keys}},
                                n_boot=n_boot, rng=rng)

    shares = []
    for meta in oof["meta"]:
        te = np.load(main_ckpt / f"r{meta['repeat']}_f{meta['fold']}.npz")["test"]
        shares += [oof["flags"][f"eocap_{a}__eo"][meta["repeat"], te].mean() for a in mcfg.attributes]
    eocap_share = {"mean": float(np.mean(shares)), "min": float(np.min(shares)),
                   "max": float(np.max(shares)), "q": float(np.mean([m["q"] for m in oof["meta"]]))}
    res = {"primary": primary, "leading_attribute": lead, "config": dataclasses.asdict(mcfg),
           "eocap_share": eocap_share,
           "n_boot": n_boot, "n_repeats": cfg.n_repeats, "effects": effects, "audit": audit,
           "summary": overall, "postprocessing": postprocessing_summary(oof["meta"], mcfg.attributes),
           "split_seconds": [m["seconds"] for m in oof["meta"]]}
    (out_dir / "mitigation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    timing = (f"fitting {fit_min:.1f} min (sum over splits), whole script "
              f"{(time.time() - t0) / 60:.1f} min")
    report = ROOT / "outputs" / "reports" / f"{tag.replace('approved', 'approved_mitigation')}.md"
    report.write_text(build_report(res, primary, lead, timing), encoding="utf-8")
    print(f"wrote {out_dir / 'mitigation.json'} and {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
