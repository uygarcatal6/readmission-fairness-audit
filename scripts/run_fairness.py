"""
scripts/run_fairness.py — the fairness audit on the approved-protocol predictions.

Needs a finished `python scripts/run_approved.py` (checkpoints + summary.json).

    python scripts/run_fairness.py            # 1000 patient-bootstrap draws
    python scripts/run_fairness.py --quick    # the --quick run of run_approved.py, 200 draws
    python scripts/run_fairness.py --n-jobs 4 # cap the CPU threads (default: half)

Writes
    outputs/approved[_quick]/fairness.json           all group metrics, gaps, calibration (+ CIs)
    outputs/reports/approved_fairness[_quick].md     tables for the paper (aggregates only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from readmission import cohort, evaluate, fairness, runtime  # noqa: E402

RULES = ("topq", "q05", "q10", "q20", "q30", "youden", "cw05")


def _fmt(m: dict, digits: int = 3) -> str:
    lo, hi = m["ci"]
    mark = " ‡" if m.get("insufficient") else ""
    mark += " †" if not m.get("ci_reliable", True) else ""
    return f"{m['estimate']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]{mark}"


def _table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def build_report(res: dict, primary: str, lead: str, approved, timing: str) -> str:
    key = f"{primary}__topq"
    lines = [
        "# Approved protocol — fairness audit",
        "",
        f"Primary model (pre-registered: highest ROC-AUC among approved models): **{primary}**. "
        f"Leading attribute (largest equalized-odds difference in excess of its permutation "
        f"null, rule amended 23 Sep before the full results): **{lead}**; by the raw range "
        f"(first rule): {res.get('leading_attribute_raw')}. Patient-bootstrap draws "
        f"{res['n_boot']}, repeats {res['n_repeats']}, {timing}.",
        "",
        f"‡ = insufficient (TPR: < {res['min_positives']} positives; FPR: < "
        f"{res['min_negatives']} negatives; PPV: < {res['min_flagged']} flagged) — reported, "
        "left out of the gaps. † = more than 1 % of bootstrap draws undefined.",
        "",
        "## 1. Groups and base rates",
        "",
    ]
    rows = []
    for a in fairness.ATTRIBUTES:
        for g, d in res["groups"][a].items():
            rows.append([a, g, f"{d['n']:,}", f"{d['n_pos']:,}", f"{d['n_neg']:,}",
                         _fmt(d["base_rate"], 4)])
    lines += _table(["Attribute", "Group", "Encounters", "Readmitted", "Not readmitted",
                     "Base rate [95 % CI]"], rows)

    lines += ["", "## 2. Equalized-odds difference at top q % (all approved models)", "",
              "Range (max − min over usable groups) of the repeat-averaged group rates. A range "
              "is never below 0, so its CI cannot show 'no gap': read it with the permutation "
              "null (same flags, attribute shuffled across patients) and the signed pair "
              "differences in section 3.", ""]
    rows = []
    for a in fairness.ATTRIBUTES:
        rows.append([a] + [_fmt(res["gaps"][f"{m}__topq"][a]["eo_diff"]) for m in approved])
    lines += _table(["Attribute"] + [m + (" (primary)" if m == primary else "") for m in approved], rows)
    lines += ["", f"Permutation null for {primary} ({res['n_permutations']} patient-level shuffles):", ""]
    rows = []
    for a in fairness.ATTRIBUTES:
        g = res["gaps"][key][a]
        nl = g.get("null")
        if nl:
            rows.append([a, f"{g['eo_diff']['estimate']:.3f}", f"{nl['mean']:.3f}", f"{nl['p95']:.3f}",
                         f"{nl['excess']:.3f}", f"{nl['p_value']:.3f}"])
    lines += _table(["Attribute", "EO difference", "Null mean", "Null 95th pct",
                     "Excess over null", "Permutation p"], rows)
    if "race" in res["gaps"][key] and "eo_diff_incl_missing" in res["gaps"][key]["race"]:
        lines += ["", "Race 'Unknown' (missing race) is shown in the tables but left out of the race "
                  f"gap; with it included the race EO difference is "
                  f"{_fmt(res['gaps'][key]['race']['eo_diff_incl_missing'])}."]

    for a in fairness.ATTRIBUTES:
        title = "sex × age intersections" if a == "sex_x_age" else a
        lines += ["", f"## 3.{fairness.ATTRIBUTES.index(a) + 1} {title} — {primary}, top q %", ""]
        rows = []
        for g, d in res["groups"][a].items():
            m = d["metrics"][key]
            rows.append([g, f"{d['n_pos']:,}", _fmt(m["tpr"]), _fmt(m["fpr"]), _fmt(m["ppv"]),
                         f"{m['selection_rate']['estimate']:.3f}"])
        gap = res["gaps"][key][a]
        lines += _table(["Group", "Readmitted", "TPR", "FPR", "PPV", "Selection rate"], rows)
        notes = [f"TPR gap {_fmt(gap['tpr_gap'])}", f"FPR gap {_fmt(gap['fpr_gap'])}",
                 f"equalized-odds difference {_fmt(gap['eo_diff'])}"]
        for m in ("tpr", "fpr"):
            if f"pair_{m}" in gap:
                hi, lo = gap[f"pair_{m}"]["groups"]
                notes.append(f"signed {m.upper()} {hi} − {lo}: {_fmt(gap[f'pair_{m}'])}")
        if gap["excluded_tpr"]:
            notes.append("left out of the TPR gap (< 50 positives): " + ", ".join(gap["excluded_tpr"]))
        if gap["excluded_fpr"]:
            notes.append("left out of the FPR gap (< 50 negatives): " + ", ".join(gap["excluded_fpr"]))
        if gap.get("not_a_group"):
            notes.append("missing-value code, not a group: " + ", ".join(gap["not_a_group"]))
        lines += [""] + [f"- {n}" for n in notes]

    lines += ["", f"## 4. Calibration by group — {primary}", ""]
    rows = []
    for a in fairness.ATTRIBUTES:
        for g, c in res["calibration"][primary][a].items():
            rows.append([a, g, f"{c['observed']:.4f}", f"{c['mean_predicted']:.4f}",
                         _fmt(c["citl"], 4), f"{c['ece']:.4f}",
                         f"{c['ece_if_perfectly_calibrated']:.4f}"])
    lines += _table(["Attribute", "Group", "Observed rate", "Mean predicted risk",
                     "Observed − predicted [95 % CI]", "ECE (10 bins)",
                     "ECE if perfectly calibrated"], rows)
    lines += ["", "ECE depends on group size: compare each value with the ECE a perfectly "
              "calibrated model would show in a group of that size (last column)."]

    lines += ["", f"## 5. Equalized-odds difference under the sensitivity rules — {primary}", ""]
    rules = [r for r in RULES if f"{primary}__{r}" in res["gaps"]]
    rows = [[a] + [_fmt(res["gaps"][f"{primary}__{r}"][a]["eo_diff"]) for r in rules]
            for a in fairness.ATTRIBUTES]
    lines += _table(["Attribute"] + list(rules), rows)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="CPU threads for the bootstrap; changes the speed, not the reported "
                         "results (default: half of the CPUs, see readmission/runtime.py)")
    args = ap.parse_args(argv)
    print(f"using {runtime.cap_threads(args.n_jobs)} CPU threads", flush=True)

    tag = "approved_quick" if args.quick else "approved"
    out_dir = ROOT / "outputs" / tag
    saved = json.loads((out_dir / "checkpoints" / "config.json").read_text(encoding="utf-8"))
    cfg = evaluate.Config(**saved["config"])  # the settings the checkpoints were made with
    n_boot = args.n_boot or (200 if args.quick else 1000)

    t0 = time.time()
    df = cohort.load_cohort()
    oof = evaluate.load_oof(out_dir / "checkpoints", df, cfg)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    primary = fairness.choose_primary(summary, tuple(cfg.models))
    keys = [f"{m}__{r}" for m in cfg.models for r in RULES if f"{m}__{r}" in oof["flags"]]
    keys += [f"{m}__topq" for m in cfg.extensions if f"{m}__topq" in oof["flags"]]
    print(f"primary model: {primary}; auditing {len(keys)} model/rule pairs with {n_boot} draws",
          flush=True)
    res = fairness.audit(df, oof, keys=keys, calibration_models=list(cfg.models),
                         n_boot=n_boot, rng=np.random.default_rng([cfg.seed, 3030]))
    (out_dir / "fairness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lead = fairness.leading_attribute(res, f"{primary}__topq") or "undetermined"
    res["leading_attribute_raw"] = fairness.leading_attribute(res, f"{primary}__topq", use_null=False)
    res["primary_model"], res["leading_attribute"] = primary, lead
    timing = f"{(time.time() - t0) / 60:.1f} min"
    (out_dir / "fairness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    report = ROOT / "outputs" / "reports" / f"{tag.replace('approved', 'approved_fairness')}.md"
    report.write_text(build_report(res, primary, lead, cfg.models, timing), encoding="utf-8")
    print(f"leading attribute: {lead}; wrote {out_dir / 'fairness.json'} and {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
