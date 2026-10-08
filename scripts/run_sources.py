"""
scripts/run_sources.py — source analysis: feature-set ablation, unawareness, proxy check,
ranking within groups (readmission/sources.py).

Needs a finished `python scripts/run_approved.py` and `python scripts/run_fairness.py`.

    python scripts/run_sources.py            # all splits, 1000 patient-bootstrap draws
    python scripts/run_sources.py --quick    # on the --quick main run, 200 draws
    python scripts/run_sources.py --n-jobs 4 # cap the CPU threads (default: half)

Writes
    outputs/approved[_quick]/sources/r*_f*.npz|json   one pair per outer split (resumable)
    outputs/approved[_quick]/sources.json             paired differences, references, proxy
                                                      AUCs, ranking and group rates
    outputs/reports/approved_sources[_quick].md       tables for the paper (aggregates only)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from readmission import (  # noqa: E402
    cohort,
    evaluate,
    fairness,
    mitigation,
    runtime,
    source_stats,
    sources,
)

ARM_LABEL = {
    "without_demographics": "without demographics (age, sex, race)",
    "without_prior_use": "without prior use (visits in the year before)",
    "without_current_stay": "without current stay (length, labs, procedures, drugs, number of diagnoses)",
    "without_admission": "without admission / discharge / specialty / payer",
    "without_diagnosis": "without primary-diagnosis group",
    "without_diabetes_care": "without diabetes care (HbA1c, glucose, insulin, metformin, "
                             "medication change, diabetes medication prescribed)",
    "no_sex": "without sex",
    "no_race": "without race",
    "no_age": "without age",
    "only_demographics": "age, sex and race only",
    "refit_seed": "reference: all inputs, another random seed",
}
UNAWARE_ROWS = ("no_sex", "no_race", "no_age", "without_demographics")


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


def lead_from_audit(fair: dict, effects: dict, primary: str) -> str | None:
    """The audit's leading attribute, but only if the audit was run on these very flags: same
    primary model and the same baseline EO for every attribute (to 1e-9)."""
    key = f"{primary}__topq"
    if fair.get("primary_model") != primary or key not in fair.get("gaps", {}):
        return None
    for a in fairness.MAIN_ATTRIBUTES:
        if abs(effects["refit_seed@topq"][f"eo_{a}"]["counterpart"]
               - fair["gaps"][key][a]["eo_diff"]["estimate"]) >= 1e-9:
            return None
    return fair.get("leading_attribute")


def build_report(res: dict, primary: str, lead: str | None, timing: str) -> str:
    eff, deg, attrs = res["effects"], res["degradation"], _order(lead)
    first = attrs[0]

    def diff(name, stat, digits=3):
        return _ci(eff[name][stat], digits=digits, key="difference")

    def arm_rows(arms):
        rows = []
        for arm in arms:
            n = f"{arm}@topq"
            e = eff[n][f"eo_{first}"]
            rows.append([ARM_LABEL[arm], diff(n, "roc_auc", 4), diff(n, "accuracy", 4),
                         f"{e['counterpart']:.3f} → {e['mitigated']:.3f}", diff(n, f"eo_{first}"),
                         f"{deg[arm][f'eo_{first}']:.3f} ({deg[arm][f'eo_{first}_min']:.3f}–"
                         f"{deg[arm][f'eo_{first}_max']:.3f})"]
                        + [diff(n, f"eo_{b}") for b in attrs[1:]])
        return rows

    headers = (["Model", "Δ ROC-AUC", "Δ accuracy", f"EO {first} (full → arm)", f"Δ EO {first}",
                f"EO {first} at the same ROC-AUC loss, random: mean (min–max)"]
               + [f"Δ EO {b}" for b in attrs[1:]])
    n_arms = len(sources.arm_features()) - len(sources.SEED_OFFSET)  # refitted arms, not the seed control
    confirm = (f"Confirmatory family: Δ EO of {first} at top q % for the {n_arms} refitted arms, "
               "each with an unadjusted 95 % CI; everything else is descriptive." if lead else
               "No leading attribute (the audit does not match these checkpoints): all rows are "
               "descriptive.")
    miss = max(d["max_auc_miss"] for d in deg.values())
    null = res.get("null_mean")
    lines = [
        "# Approved protocol — where the disparities come from",
        "",
        f"Model: **{primary}** (the primary model of the audit), refitted on the same "
        f"{res['n_splits']} outer splits without some of its inputs and flagged with the same "
        f"rule (top q %). Leading attribute: **{lead or 'not available'}**. Patient-bootstrap "
        f"draws {res['n_boot']}, repeats {res['n_repeats']}, {timing}.",
        "",
        "Every row is the refitted model minus the full model, on the same bootstrap draw (95 % "
        "CI of the difference). The CIs cover patient sampling of the test folds given the fitted "
        "models; refit randomness is averaged over the splits, not resampled — the reference row "
        "'all inputs, another random seed' shows its size. EO = equalized-odds difference with the "
        "audit's rules" + (f"; with no disparity at all the audit's permutation null for {first} "
                           f"is {null:.3f}" if null is not None else "") + ". " + confirm,
        "",
        "Removing inputs also costs ROC-AUC, and a weaker ranking can shrink a gap by moving "
        "towards random flags (or widen one, as a model left with little but age does). The "
        f"column 'EO {first} at the same ROC-AUC loss, random' is the gap the full model's ranking "
        "shows after losing the arm's ROC-AUC to random noise in every test fold (mean of "
        f"{res['degradation_draws']} noise draws, their range in brackets). It is a descriptive "
        "reference, not a test: an arm well below its whole range removed inputs that carry the "
        "gap, not only predictive signal. On folds where an arm ranks no worse than the full model "
        "the reference keeps the full ranking, so for arms with no ROC-AUC loss it is the full "
        "model's gap" + (f" (largest shortfall of the noise from a fold's target ROC-AUC: {miss:.4f})"
                          if miss > 1e-3 else "") + ". Base rates and calibration by group: "
        f"`{res['fairness_report']}` §1 and §4.",
        "",
        f"Shown: top q % in §1–§3 and §5, point estimates for the other shares in §6; §3 for "
        f"{first}, §5 for {first}" + (" and sex" if first != "sex" else "") + ". In "
        "`sources.json` only: Δ EO for every attribute and rule, group rates and ranking for every "
        "attribute at top q %, per-class proxy CIs.",
        "",
        "Feature blocks (every model input in exactly one block, fixed before the results):",
        "",
    ]
    lines += _table(["Block", "Inputs"], [[b, ", ".join(c)] for b, c in res["blocks"].items()])

    lines += ["", "## 1. Feature-set ablation: leave one block out", ""]
    lines += _table(headers, arm_rows([f"without_{b}" for b in res["blocks"]] + ["refit_seed"]))

    lines += ["", "## 2. Removal of the sensitive attributes (unawareness)", "",
              "age_band is not a model input; 'without age' removes the age it is derived from.",
              ""]
    lines += _table(headers, arm_rows([*UNAWARE_ROWS, "refit_seed"]))
    od = eff["only_demographics@topq"]
    lines += ["", f"A model on age, sex and race alone reaches ROC-AUC {od['roc_auc']['mitigated']:.3f} "
              f"(full model {od['roc_auc']['counterpart']:.3f}); its {first} gap "
              f"({od[f'eo_{first}']['mitigated']:.3f}) is not interpreted: such a model can only "
              "rank whole demographic cells."]

    gr = res["group_rates"]
    arms = ["base", *(f"without_{b}" for b in res["blocks"]), *UNAWARE_ROWS[:3], "refit_seed"]
    lines += ["", f"## 3. Which {first} groups move (TPR / FPR at top q %)", "",
              "A gap can close because the low group rises or because the high group falls "
              "(levelling down). Point estimates, repeat-averaged.", ""]
    groups = list(gr["base__topq"].keys())
    rows = [[ARM_LABEL.get(a, "full model")] + [f"{gr[f'{a}__topq'][g]['tpr']:.3f} / "
                                                f"{gr[f'{a}__topq'][g]['fpr']:.3f}" for g in groups]
            for a in arms]
    lines += _table(["Model"] + groups, rows)

    lines += ["", "## 4. How well the other inputs predict each attribute (proxy check)", "",
              f"A {sources.PROXY_MODEL} classifier (default settings) trained on the training fold "
              "predicts the attribute from every other model input (sex: all but sex; age band: "
              "all but age; race: all but race, race 'Unknown' left out). ROC-AUC one class vs the "
              "rest. 0.5 = the other inputs carry no trace of the attribute; the further above "
              "0.5, the less removing the attribute removes its information (an untuned model "
              "gives a lower bound).", ""]
    rows = []
    for a in attrs:
        p = res["proxy"][a]
        per = ", ".join(f"{c} {v['estimate']:.3f}" for c, v in p.items() if c != "macro")
        rows.append([a, _pt(p["macro"]), per])
    lines += _table(["Attribute", "ROC-AUC (mean over classes) [95 % CI]", "Per class"], rows)

    lines += ["", "## 5. Ranking within groups versus the shared cut-off", "",
              "For each group of the full model at top q %: ROC-AUC within the group; its TPR and "
              "FPR under the shared rule; and its TPR at its own cut-off that gives it the "
              "overall FPR (ranking quality at the operating point). Reading: TPR − 'TPR at the "
              "overall FPR' is what the shared cut-off adds (+) or takes away (−) from the group "
              "because of where its scores lie; differences in 'TPR at the overall FPR' between "
              "groups are differences in ranking quality at the operating point.", ""]
    rk = res["ranking"]
    for a in dict.fromkeys([first, "sex"]):
        lines += [f"**{a}**", ""]
        rows = []
        for g, d in rk[a].items():
            mark = " ‡" if d["n_pos"] < fairness.MIN_POSITIVES else ""
            rows.append([g, f"{d['n_pos']:,}", _pt(d["roc_auc"]), f"{d['tpr']:.3f}{mark}",
                         f"{d['fpr']:.3f}", f"{d['tpr_at_overall_fpr']:.3f}",
                         f"{d['tpr'] - d['tpr_at_overall_fpr']:+.3f}",
                         f"{d['mean_risk_readmitted']:.3f} / {d['mean_risk_not_readmitted']:.3f}"])
        lines += _table(["Group", "Readmitted", "Within-group ROC-AUC", "TPR", "FPR",
                         "TPR at the overall FPR", "Shared cut-off effect",
                         "Mean risk, readmitted / not"], rows)
        lines += [""]

    lines += ["## 6. Ablation under the other capacities (sensitivity)", "",
              f"Δ EO {first}, refitted − full model, point estimates.", ""]
    rules = list(sources.SCORE_RULES)
    rows = [[ARM_LABEL[arm]] + [f"{eff[f'{arm}@{r}'][f'eo_{first}']['difference']:+.3f}" for r in rules]
            for arm in sources.arm_features()]
    lines += _table(["Model"] + rules, rows)
    lines += ["", "† = more than 1 % of bootstrap draws undefined. ‡ = fewer than 50 readmitted "
              "encounters (TPR reported, not used in gaps)."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="CPU threads for fitting and the bootstrap; changes the speed, not the "
                         "reported results (default: half of the CPUs, see readmission/runtime.py)")
    args = ap.parse_args(argv)
    print(f"using {runtime.cap_threads(args.n_jobs)} CPU threads", flush=True)

    tag = "approved_quick" if args.quick else "approved"
    out_dir = ROOT / "outputs" / tag
    main_ckpt, src_ckpt = out_dir / "checkpoints", out_dir / "sources"
    cfg = evaluate.Config(**json.loads((main_ckpt / "config.json").read_text(encoding="utf-8"))["config"])
    n_boot = args.n_boot or (200 if args.quick else 1000)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    primary = fairness.choose_primary(summary, tuple(cfg.models))
    fair = json.loads((out_dir / "fairness.json").read_text(encoding="utf-8"))
    scfg = sources.SourcesConfig(primary=primary)

    t0 = time.time()
    df = cohort.load_cohort()
    if not args.summarize_only:
        print(f"source analysis on {primary}", flush=True)
        sources.run(df, cfg, scfg, main_ckpt, src_ckpt, log=lambda s: print(s, flush=True))
    oof = sources.load(src_ckpt, df, cfg, scfg)
    fit_min = sum(m["seconds"] for m in oof["meta"]) / 60

    rng = np.random.default_rng([cfg.seed, 5050])
    print(f"paired differences with {n_boot} draws ...", flush=True)
    effects = mitigation.paired_effects(df, oof, sources.comparisons(rules=sources.SCORE_RULES),
                                        n_boot=n_boot, rng=rng)
    lead = lead_from_audit(fair, effects, primary)
    if lead is None:
        print("fairness.json does not match these checkpoints: no leading attribute", flush=True)
    null_mean = fair["gaps"][f"{primary}__topq"][lead]["null"]["mean"] if lead else None
    print("matched-degradation reference ...", flush=True)
    degradation = source_stats.matched_degradation(df, oof, list(sources.arm_features()), seed=cfg.seed)
    print("proxy check ...", flush=True)
    proxy = {a: source_stats.proxy_auc(df, oof, a, n_boot=n_boot, rng=rng) for a in scfg.attributes}
    print("ranking within groups ...", flush=True)
    ranking = source_stats.group_ranking(df, oof["scores"]["base"], oof["flags"]["base__topq"],
                                    scfg.attributes, n_boot=n_boot, rng=rng)
    rate_keys = [f"{a}__topq" for a in ["base", *sources.arm_features()]]
    group_rates = {a: source_stats.group_rates(df, oof, rate_keys, a) for a in scfg.attributes}

    first = lead or fairness.MAIN_ATTRIBUTES[0]
    res = {"primary": primary, "leading_attribute": lead, "config": dataclasses.asdict(scfg),
           "blocks": sources.FEATURE_BLOCKS, "n_boot": n_boot, "n_repeats": cfg.n_repeats,
           "n_splits": cfg.n_splits * cfg.n_repeats, "null_mean": null_mean,
           "fairness_report": f"{tag.replace('approved', 'approved_fairness')}.md",
           "degradation_draws": source_stats.N_DEGRADATION_DRAWS, "effects": effects,
           "degradation": degradation, "proxy": proxy, "ranking": ranking,
           "group_rates_all": group_rates, "group_rates": group_rates[first],
           "split_seconds": [m["seconds"] for m in oof["meta"]]}
    (out_dir / "sources.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    timing = (f"fitting {fit_min:.1f} min (sum over splits), whole script "
              f"{(time.time() - t0) / 60:.1f} min")
    report = ROOT / "outputs" / "reports" / f"{tag.replace('approved', 'approved_sources')}.md"
    report.write_text(build_report(res, primary, lead, timing), encoding="utf-8")
    print(f"wrote {out_dir / 'sources.json'} and {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
