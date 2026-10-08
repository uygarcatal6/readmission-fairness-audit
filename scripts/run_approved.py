"""
scripts/run_approved.py — the approved protocol end to end: models, evaluation, report.

    python scripts/run_approved.py --quick            # 1 repeat, 200 bootstrap draws
    python scripts/run_approved.py                    # 5 x 5 folds, 1000 bootstrap draws
    python scripts/run_approved.py --summarize-only   # rebuild the report from checkpoints
    python scripts/run_approved.py --n-jobs 4         # cap the CPU threads (default: half)

Writes (outputs/approved*/ is git-ignored: it holds per-encounter predictions)
    outputs/approved[_quick]/checkpoints/r*_f*.npz|json   one pair per outer split (resumable)
    outputs/approved[_quick]/summary.json                 all metrics with patient-bootstrap CIs
    outputs/reports/approved_main[_quick].md              tables for the paper (aggregates only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from readmission import cohort, evaluate, metrics, runtime  # noqa: E402

RULE_LABEL = {
    "topq": "top q % (q = training rate)",
    "q05": "top 5 %", "q10": "top 10 %", "q20": "top 20 %", "q30": "top 30 %",
    "youden": "inner-CV Youden threshold",
    "cw05": "class-weighted model at 0.5",
}


def _fmt(m: dict, digits: int = 3) -> str:
    lo, hi = m["ci"]
    flag = "" if m.get("ci_reliable", True) else " †"
    return f"{m['estimate']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]{flag}"


def _table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _ranking_rows(summary: dict, names) -> list[list]:
    rows = []
    for key, m in summary["ranking"].items():
        kind, name = key.split(":")
        if name not in names:
            continue
        label = name + (" (class-weighted)" if kind == "class_weighted" else "")
        rows.append([label, _fmt(m["roc_auc"]), _fmt(m["pr_auc"]),
                     f"{m['brier']['estimate']:.4f}", _fmt(m["brier_skill"])])
    return rows


def _gate_rows(summary: dict, names, rules) -> list[list]:
    rows = []
    for key, m in summary["decision"].items():
        name, rule = key.split("__")
        if name not in names or rule not in rules:
            continue
        rows.append([name, RULE_LABEL[rule], f"{m['accuracy']['estimate']:.4f}",
                     _fmt(m["acc_minus_nir"], 4), "yes" if m["usable"] else "no",
                     _fmt(m["tpr"]), _fmt(m["ppv"]), _fmt(m["fpr"]),
                     f"{m['selection_rate']['estimate']:.4f}"])
    return rows


def build_report(summary: dict, cfg: evaluate.Config, timing: str) -> str:
    p, nir = summary["prevalence"], summary["nir"]
    approved, extensions = tuple(cfg.models), tuple(cfg.extensions)
    rank_head = ["Model", "ROC-AUC", f"PR-AUC (prevalence {p:.3f})", "Brier", "Brier skill score"]
    gate_head = ["Model", "Decision rule", "Accuracy", f"Accuracy − {nir:.4f} (NIR)",
                 "Usable (lower CI > 0)", "TPR", "PPV", "FPR", "Selection rate"]
    sensitivity = tuple(r for r in RULE_LABEL if r != "topq")
    lines = [
        "# Approved protocol — main results",
        "",
        f"Encounters {summary['n_encounters']:,} · readmitted within 30 days {p:.4f} · "
        f"no-information rate {nir:.4f} · outer splits {cfg.n_splits} × {cfg.n_repeats} · "
        f"patient-bootstrap draws {summary['n_boot']} · {timing}.",
        "",
        "Point estimate = mean over repeats of the pooled out-of-fold metric; [95 % CI] from "
        "the patient bootstrap (readmission/metrics.py). † = more than 1 % of bootstrap draws "
        "undefined for that metric.",
        "",
        "## 1. Ranking quality (approved models)",
        "",
        *_table(rank_head, _ranking_rows(summary, approved)),
        "",
        "## 2. Main operating point and usability gate (approved models)",
        "",
        *_table(gate_head, _gate_rows(summary, approved, ("topq",))),
        "",
        "## 3. Sensitivity checks (approved models)",
        "",
        *_table(gate_head, _gate_rows(summary, approved, sensitivity)),
    ]
    curves = summary.get("complexity_curves")
    if curves:
        lines += ["", "## 4. Decision-tree complexity (first outer split, first inner fold)", ""]
        for knob, label in (("depth", "max_depth"), ("ccp_alpha", "cost-complexity alpha")):
            c = curves[knob]
            rows = [[str(v), f"{tr:.3f}", f"{va:.3f}"]
                    for v, tr, va in zip(c["values"], c["train_auc"], c["val_auc"], strict=True)]
            lines += [f"**{label}**", "",
                      *_table([label, "train ROC-AUC", "validation ROC-AUC"], rows), ""]
    lines += ["## 5. Hyperparameters chosen by the one-SE rule (per outer split)", ""]
    rows = [[m, ", ".join(f"{v}: {n}" for v, n in sorted(t["best"].items())),
             ", ".join(f"{v}: {n}" for v, n in sorted(t["chosen"].items())),
             f"{t['at_grid_edge']} of {t['n']}", f"{t['chosen_at_grid_edge']} of {t['n']}"]
            for m, t in summary["tuning"].items()]
    lines += _table(["Model", "best mean (value: splits)", "one-SE choice (value: splits)",
                     "best at grid edge", "choice at grid edge"], rows)
    if extensions:
        lines += ["", "## 6. Course-topic extensions (not part of the approved protocol)", "",
                  *_table(rank_head, _ranking_rows(summary, extensions)), "",
                  *_table(gate_head, _gate_rows(summary, extensions, ("topq",)))]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1 repeat, 200 bootstrap draws")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--no-extensions", action="store_true", help="skip MLP and bagging")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="CPU threads for fitting and the bootstrap; changes the speed, not the "
                         "reported results (default: half of the CPUs, see readmission/runtime.py)")
    args = ap.parse_args(argv)
    print(f"using {runtime.cap_threads(args.n_jobs)} CPU threads", flush=True)

    tag = "approved_quick" if args.quick else "approved"
    out_dir = ROOT / "outputs" / tag
    ckpt = out_dir / "checkpoints"
    cfg = evaluate.Config(n_repeats=1 if args.quick else 5,
                          extensions=() if args.no_extensions else evaluate.EXTENSIONS)
    n_boot = args.n_boot or (200 if args.quick else 1000)

    t0 = time.time()
    df = cohort.load_cohort()
    if not args.summarize_only:
        evaluate.run(df, cfg, ckpt, log=lambda s: print(s, flush=True))
    oof = evaluate.load_oof(ckpt, df, cfg)
    print(f"summarizing with {n_boot} patient-bootstrap draws ...", flush=True)
    summary = metrics.summarize(df[cohort.TARGET].to_numpy(), df[cohort.GROUP].to_numpy(), oof,
                                n_boot=n_boot, rng=np.random.default_rng([cfg.seed, 2026]))
    summary["tuning"] = {}
    grids = {"tree": list(cfg.tree_depths), "knn": list(cfg.knn_ks)}
    for m in ("tree", "knn"):
        picks = [meta["tuning"][m] for meta in oof["meta"] if m in meta["tuning"]]
        if picks:
            ends = (str(grids[m][0]), str(grids[m][-1]))
            summary["tuning"][m] = {
                "chosen": dict(Counter(str(t["value"]) for t in picks)),
                "best": dict(Counter(str(t["best_value"]) for t in picks)),
                "at_grid_edge": sum(bool(t.get("at_grid_edge")) for t in picks),
                # the one-SE pick itself at the end of the grid: a wider grid could pick an
                # even simpler model, at most one SE below the best mean
                "chosen_at_grid_edge": sum(str(t["value"]) in ends for t in picks),
                "n": len(picks)}
    print("decision-tree complexity curves ...", flush=True)
    summary["complexity_curves"] = evaluate.complexity_curves(df, cfg)
    summary["split_seconds"] = [meta["seconds"] for meta in oof["meta"]]
    fit_min = sum(summary["split_seconds"]) / 60
    timing = (f"model fitting {fit_min:.1f} min (sum over splits), "
              f"whole script {(time.time() - t0) / 60:.1f} min")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    report = ROOT / "outputs" / "reports" / f"{tag.replace('approved', 'approved_main')}.md"
    report.write_text(build_report(summary, cfg, timing), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.json'} and {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
