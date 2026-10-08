"""
scripts/run_ordering.py — test-ordering equity (readmission/ordering.py).

    python scripts/run_ordering.py            # 1000 patient-bootstrap draws for the crude rates
    python scripts/run_ordering.py --n-boot 200
    python scripts/run_ordering.py --n-jobs 4  # cap the CPU threads (default: half)

Writes
    outputs/approved/ordering.json            odds ratios, CIs, p, BH q, crude rates, recording
                                              pattern, source-adjusted sensitivity fit
    outputs/reports/approved_ordering.md      tables for the paper (aggregates only)
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

from readmission import cohort, fairness, ordering, runtime  # noqa: E402

SIG = 0.05


def _p(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"


def _or(u: dict) -> str:
    return f"{u['odds_ratio']:.2f} [{u['ci'][0]:.2f}, {u['ci'][1]:.2f}]"


def _table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _tested_names(fits: dict) -> list[str]:
    first = fits[next(iter(fits))]["terms"]
    return [n for n, t in first.items() if t["covariate"] in ordering.TESTED]


def _or_rows(fits: dict) -> list[list[str]]:
    rows = []
    for name in _tested_names(fits):
        t = fits[next(iter(fits))]["terms"][name]
        row = [t["covariate"], f"{t['level']} vs {t['reference']}"]
        for o in ordering.OUTCOMES:
            u = fits[o]["terms"][name]
            mark = "" if u["tested"] else " (missing code, not tested)"
            row += [_or(u) + mark, _p(u["p"]), _p(u["q"])]
        rows.append(row)
    return rows


def _verdicts(fits: dict) -> dict:
    """(outcome, term) -> (significant at q < 0.05, direction) for the tested terms."""
    return {(o, n): (t["q"] < SIG, t["odds_ratio"] > 1)
            for o, f in fits.items() for n, t in f["terms"].items() if t["tested"]}


def build_report(res: dict, timing: str) -> str:
    allf, first, sens = res["all_encounters"], res["first_encounters"], res["source_adjusted"]
    labels = ordering.OUTCOMES
    heads = ["Variable", "Contrast"]
    for label in labels.values():
        heads += [f"{label}: OR [95 % CI]", "p", "BH q"]
    events = ", ".join(f"{labels[o]} {allf[o]['events']:,} ({allf[o]['rate']:.3f})" for o in labels)
    lines = [
        "# Approved protocol — test-ordering equity",
        "",
        "Is a test result recorded equally often across groups once the other variables are held "
        "fixed? The dataset holds a result range for HbA1c and for serum glucose, or 'None'; a "
        "result recorded during the encounter is the data's proxy for the test being ordered (the "
        "approved wording). Logistic regression of each indicator on sex, age band, race and "
        "primary-diagnosis group (reference = the largest group of each); odds ratios with 95 % CIs "
        "from patient-clustered standard errors; Benjamini–Hochberg q over every sex, age-band and "
        "race term of both outcomes (diagnosis terms are adjustment; race 'Unknown' is the "
        "missing-value code and is not tested).",
        "",
        f"Events (rate): {events}. At these rates an odds ratio is larger than the matching risk "
        "ratio (read the crude rates in §1 for 'how often'). The models adjust for diagnosis "
        "group only: they describe who has a recorded result, not whether testing matched "
        "clinical need, and are not evidence of unequal care on their own. §4 shows that the "
        "glucose field in particular follows the data source.",
        "",
        "## 1. Crude recording rates by group (all encounters, patient-bootstrap 95 % CI)",
        "",
    ]
    rows = []
    crude = res["crude"]
    o0 = next(iter(labels))
    for c in ordering.TESTED:
        for g, d in crude[o0][c].items():
            note = " (missing code)" if g in fairness.NOT_A_GROUP.get(c, ()) else ""
            rows.append([c, g + note, f"{d['n']:,}"] + [
                f"{crude[o][c][g]['estimate']:.3f} [{crude[o][c][g]['ci'][0]:.3f}, "
                f"{crude[o][c][g]['ci'][1]:.3f}]" for o in labels])
    lines += _table(["Variable", "Group", "Encounters"] + list(labels.values()), rows)

    lines += ["", f"## 2. Adjusted odds ratios, all encounters (n = {allf[o0]['n']:,}, "
              f"{allf[o0]['n_patients']:,} patients)", ""]
    lines += _table(heads, _or_rows(allf))
    parts, flips = [], []
    for o in labels:
        tested = [t for t in allf[o]["terms"].values() if t["tested"]]
        r = [t["se_ratio"] for t in tested]
        parts.append(f"{labels[o]} {min(r):.2f}–{max(r):.2f}")
        flips += [f"{labels[o]}: {t['covariate']}={t['level']}" for t in tested
                  if (t["p"] < SIG) != (t["p_model_based"] < SIG)]
    lines += ["", "Clustering by patient changes the standard errors of the tested terms by a "
              f"factor of {'; '.join(parts)} relative to the model-based ones (repeat encounters "
              "of one patient are not independent). " + (
                  f"Significant (p < {SIG}) under one kind of SE but not the other: "
                  f"{', '.join(flips)}." if flips else
                  f"No tested term changes its p < {SIG} verdict between the two kinds of SE.")
              + " Encounters may also cluster by hospital and period, which the public data does "
              "not identify; §4 shows this can move the estimates, not only their SEs."]

    lines += ["", f"## 3. Replication on first encounters (n = {first[o0]['n']:,}, one per "
              "patient)", "", "Same model and references; BH within this family. With one "
              "encounter per patient the clustered SEs reduce to robust SEs, so this checks the "
              "repeat-encounter structure, not the hospital or period one.", ""]
    lines += _table(heads, _or_rows(first))
    va, vf = _verdicts(allf), _verdicts(first)
    same = [k for k in va if va[k] == vf[k]]
    diff = [f"{labels[o]}: {n}" for (o, n) in va if va[(o, n)] != vf[(o, n)]]
    lines += ["", f"Replicated: {len(same)} of {len(va)} tested terms have the same verdict (q < "
              f"{SIG} and direction) in both analyses" + (f"; different: {', '.join(diff)}."
                                                          if diff else ".")]

    rec = res["recording"]
    lines += ["", "## 4. Where the recorded results come from, and a sensitivity check", "",
              "Encounters with unknown admission type (the source's 'not available / not mapped' "
              "codes) and the encounter-id order (the order of the extraction) show whether a "
              "field was filled in by some sources or periods rather than per admission.", ""]
    rows = []
    for o in labels:
        r = rec[o]
        dec = r["rate_by_decile"]
        rows.append([labels[o], f"{r['events']:,}",
                     f"{r['share_of_events_source']:.1%} of results in "
                     f"{r['share_of_encounters_source']:.1%} of encounters",
                     f"{r['rate_source']:.3f} vs {r['rate_other']:.3f}",
                     f"{min(dec):.3f}–{max(dec):.3f}",
                     f"{r['next_if_now']:.3f} vs {r['next_if_not_now']:.3f}"])
    lines += _table(["Outcome", "Results", "From unknown admission type",
                     "Rate: unknown vs known type", "Rate range over encounter-id deciles",
                     "Next encounter recorded: if this one is vs is not"], rows)
    col, value = ordering.SOURCE_FLAG
    lines += ["", f"Sensitivity: the approved model plus one indicator '{col} = {value}' (not "
              "tested, not in a BH family). A group's odds ratio that moves towards 1 came partly "
              "from where and when the field was filled in.", ""]
    rows = []
    for name in _tested_names(allf):
        t = allf[o0]["terms"][name]
        row = [t["covariate"], f"{t['level']} vs {t['reference']}"]
        for o in labels:
            row += [_or(allf[o]["terms"][name]), _or(sens[o]["terms"][name])]
        rows.append(row)
    src = f"{col}={value}"
    rows.append([col, f"{value} vs other types"] + sum(
        [["—", _or(sens[o]["terms"][src])] for o in labels], []))
    sh = []
    for label in labels.values():
        sh += [f"{label}: approved", f"{label}: + source indicator"]
    lines += _table(["Variable", "Contrast"] + sh, rows)

    lines += ["", "## 5. Adjustment terms (primary-diagnosis group), all encounters", ""]
    rows = []
    for name, t in allf[o0]["terms"].items():
        if t["covariate"] != "diag_1_group":
            continue
        rows.append([f"{t['level']} vs {t['reference']}", f"{t['n_level']:,}"]
                    + [_or(allf[o]["terms"][name]) for o in labels])
    lines += _table(["Diagnosis group", "Encounters"] + [f"{v}: OR [95 % CI]" for v in labels.values()],
                    rows)
    lines += ["", f"Run time: {timing}."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="CPU threads for the bootstrap; changes the speed, not the reported "
                         "results (default: half of the CPUs, see readmission/runtime.py)")
    args = ap.parse_args(argv)
    print(f"using {runtime.cap_threads(args.n_jobs)} CPU threads", flush=True)

    t0 = time.time()
    df = cohort.load_cohort()
    first = cohort.load_cohort(first_encounter_only=True)
    res = ordering.analyse(df, first, n_boot=args.n_boot, rng=np.random.default_rng([0, 6060]))
    out = ROOT / "outputs" / "approved" / "ordering.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    report = ROOT / "outputs" / "reports" / "approved_ordering.md"
    report.write_text(build_report(res, f"{(time.time() - t0) / 60:.1f} min"), encoding="utf-8")
    print(f"wrote {out} and {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
