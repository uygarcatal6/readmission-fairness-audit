"""
readmission/figures.py — the figures of the paper, the slides and the notebook.

Every function takes the loaded result JSON(s) (load_results) and returns a matplotlib Figure;
nothing is recomputed and no number is typed by hand. Each docstring names the report section
(outputs/reports/approved_*.md) whose numbers the figure shows.

Style: one column of an IEEE paper is 3.5 in wide, so every figure is 3.5 in wide with 7 pt
text. Colours are the Okabe–Ito palette, which stays distinguishable with the common forms of
colour blindness; where a series could still be confused, the marker shape differs too.

Figures are built with matplotlib.figure.Figure (not pyplot), so creating many of them in a
test or a script never piles up open windows. In a notebook, show one by making it the last
expression of a cell (or with display(fig)).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

RESULT_FILES = ("summary", "fairness", "mitigation", "sources", "ordering")
APPROVED = ("logreg", "tree", "knn", "rf", "hgb")  # readmission/models.py APPROVED
EXTENSIONS = ("mlp", "bagging")
MAIN_ATTRIBUTES = ("sex", "age_band", "race")
NOT_A_GROUP = {"race": {"Unknown"}}  # readmission/fairness.py NOT_A_GROUP

WIDTH = 3.5  # inches, one IEEE column
OKABE_ITO = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
             "#CC79A7")
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE = OKABE_ITO
GREY = "#7F7F7F"
STYLE = {
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "lines.markersize": 4, "errorbar.capsize": 0, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
}


def use_style() -> None:
    """Apply the shared style (like seaborn.set_theme, it changes matplotlib's defaults)."""
    mpl.rcParams.update(STYLE)


def load_results(root: str | Path) -> dict[str, dict]:
    """The five result JSONs of the approved run: {"summary": ..., "fairness": ..., ...}.

    `root` is the repository root (the files are in outputs/approved/) or that folder itself.
    Raises FileNotFoundError naming the missing files.
    """
    root = Path(root)
    folder = root if (root / "summary.json").exists() else root / "outputs" / "approved"
    missing = [n for n in RESULT_FILES if not (folder / f"{n}.json").exists()]
    if missing:
        raise FileNotFoundError(f"missing in {folder}: {', '.join(m + '.json' for m in missing)}")
    return {n: json.loads((folder / f"{n}.json").read_text(encoding="utf-8")) for n in RESULT_FILES}


def _err(d: dict, key: str = "estimate") -> tuple[float, list[list[float]]]:
    """Point and asymmetric error bar (for errorbar's xerr / yerr) from {estimate|difference, ci}."""
    v = d[key]
    lo, hi = d["ci"]
    return v, [[v - lo], [hi - v]]


def _label(attr: str, group: str) -> str:
    return group + (" (missing)" if group in NOT_A_GROUP.get(attr, ()) else "")


# ── main results ──────────────────────────────────────────────────────────────────────────


def fig_models(summary: dict, *, rule: str = "topq") -> Figure:
    """Ranking and usability gate per model (approved_main.md §1–§2, §6 for the extensions).

    Panels: ROC-AUC (chance = 0.5, off the axis), PR-AUC (dashed line = prevalence, the PR-AUC
    of random scores) and accuracy − NIR at the decision rule (the gate: usable only if the
    whole CI is above 0). Approved models in black, course-topic extensions in grey.
    """
    use_style()
    models = [m for m in APPROVED + EXTENSIONS if f"model:{m}" in summary["ranking"]]
    fig = Figure(figsize=(WIDTH, 1.9), layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    ypos = np.arange(len(models))[::-1]
    panels = (("ROC-AUC", lambda m: summary["ranking"][f"model:{m}"]["roc_auc"], None),
              ("PR-AUC", lambda m: summary["ranking"][f"model:{m}"]["pr_auc"], summary["prevalence"]),
              (f"accuracy − NIR\n(NIR {summary['nir']:.3f})",
               lambda m: summary["decision"][f"{m}__{rule}"]["acc_minus_nir"], 0.0))
    for ax, (title, get, ref) in zip(axes, panels, strict=True):
        for y, m in zip(ypos, models, strict=True):
            v, e = _err(get(m))
            colour = BLACK if m in APPROVED else GREY
            ax.errorbar(v, y, xerr=e, fmt="o", color=colour, ms=3.5, elinewidth=1.2)
        if ref is not None:
            ax.axvline(ref, color=VERMILLION, ls="--", lw=0.8)
        ax.set_title(title)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(2))
    axes[1].text(summary["prevalence"], ypos[0] + 0.6, "prevalence", color=VERMILLION,
                 fontsize=6, ha="left")
    axes[2].text(0, ypos[0] + 0.6, "gate", color=VERMILLION, fontsize=6, ha="right")
    axes[0].set_yticks(ypos, models)
    axes[0].set_ylim(-0.7, len(models) - 0.1)
    return fig


def fig_complexity(summary: dict) -> Figure:
    """Decision-tree complexity: train vs validation ROC-AUC (approved_main.md §4).

    Left: max_depth (pre-pruning), "none" = unlimited. Right: cost-complexity alpha on a fully
    grown tree (post-pruning; symmetric-log axis so alpha = 0 fits). First outer split, first
    inner fold. The one-SE picks over the 25 splits are in the title (approved_main.md §5).
    """
    use_style()
    curves = summary["complexity_curves"]
    fig = Figure(figsize=(WIDTH, 1.7), layout="constrained")
    ax_d, ax_a = fig.subplots(1, 2, sharey=True)

    d = curves["depth"]
    x = np.arange(len(d["values"]))
    ax_d.plot(x, d["train_auc"], "s-", color=BLUE, ms=2.5, label="train")
    ax_d.plot(x, d["val_auc"], "o-", color=ORANGE, ms=2.5, label="validation")
    ax_d.set_xticks(x, ["none" if v is None else str(v) for v in d["values"]], rotation=90)
    ax_d.set_xlabel("max_depth")
    ax_d.set_ylabel("ROC-AUC")
    ax_d.set_title("pre-pruning")
    picks = summary.get("tuning", {}).get("tree", {}).get("chosen", {})
    if picks:
        n = summary["tuning"]["tree"]["n"]
        text = "one-SE pick over splits:\n" + ", ".join(f"depth {v} ({c}/{n})"
                                                        for v, c in picks.items())
        ax_d.text(0.02, 0.97, text, transform=ax_d.transAxes, va="top", fontsize=5.5)

    a = curves["ccp_alpha"]
    ax_a.plot(a["values"], a["train_auc"], "s-", color=BLUE, ms=2.5, label="train")
    ax_a.plot(a["values"], a["val_auc"], "o-", color=ORANGE, ms=2.5, label="validation")
    ax_a.legend(loc="upper right")
    positive = [v for v in a["values"] if v > 0]
    ax_a.set_xscale("symlog", linthresh=min(positive) if positive else 1e-5)
    ax_a.set_xlabel("cost-complexity alpha")
    ax_a.set_title("post-pruning")
    # inside the axes, like the note on the left panel: a long title overflows the figure
    ax_a.text(0.02, 0.03, "alpha 0 = unpruned", transform=ax_a.transAxes, fontsize=5.5)
    return fig


# ── fairness audit ────────────────────────────────────────────────────────────────────────


def fig_fairness(fairness: dict, *, key: str | None = None,
                 attributes=MAIN_ATTRIBUTES, overall: dict | None = None) -> Figure:
    """TPR and FPR by group with 95 % CIs (approved_fairness.md §3.1–§3.3).

    key defaults to the primary model at top q % ("rf__topq"). Hollow markers = insufficient
    group (< 50 positives for TPR, < 50 negatives for FPR; kept, left out of the gaps).
    "(missing)" = race Unknown, a missing-value code. Each attribute's EO difference and the
    mean of its permutation null (no-disparity reference) are printed on its header row.
    `overall` (summary.json "decision" entry of the same key) adds the overall rates as
    dashed lines.
    """
    use_style()
    key = key or f"{fairness['primary_model']}__topq"
    n_rows = sum(len(fairness["groups"][a]) + 1 for a in attributes)  # + one header row each
    fig = Figure(figsize=(WIDTH, 0.5 + 0.16 * n_rows), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    colours = dict(zip(attributes, (BLUE, VERMILLION, GREEN, PURPLE), strict=False))
    ticks, labels, header, y = [], [], [], n_rows
    for a in attributes:
        y -= 1  # header row: attribute name; its EO difference and null mean on the right
        header.append((y, a))
        gap = fairness["gaps"][key][a]
        note = f"EO {gap['eo_diff']['estimate']:.3f}"
        if "null" in gap:
            note += f" (null {gap['null']['mean']:.3f})"
        axes[1].text(0.98, y, note, transform=mpl.transforms.blended_transform_factory(
            axes[1].transAxes, axes[1].transData), color=colours[a], fontsize=6,
            ha="right", va="center", bbox={"fc": "white", "ec": "none", "pad": 0.5})
        for g in fairness["groups"][a]:
            y -= 1
            m = fairness["groups"][a][g]["metrics"][key]
            for ax, metric in zip(axes, ("tpr", "fpr"), strict=True):
                v, e = _err(m[metric])
                hollow = m[metric]["insufficient"]
                ax.errorbar(v, y, xerr=e, fmt="o", color=colours[a], elinewidth=1.1,
                            mfc="white" if hollow else colours[a], ms=3.5)
            ticks.append(y)
            labels.append(_label(a, g))
    if overall is not None:
        for ax, metric in zip(axes, ("tpr", "fpr"), strict=True):
            ax.axvline(overall[metric]["estimate"], color=GREY, ls="--", lw=0.7)
    axes[0].set_yticks(ticks + [h for h, _ in header], labels + [a for _, a in header])
    for tick, (_, a) in zip(axes[0].get_yticklabels()[len(ticks):], header, strict=True):
        tick.set_color(colours[a])
        tick.set_fontweight("bold")
    axes[0].set_ylim(-0.7, n_rows - 0.4)
    axes[0].set_xlabel("TPR")
    axes[1].set_xlabel("FPR")
    for ax in axes:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    model, rule = key.split("__", 1)
    rule = "top q %" if rule == "topq" else rule
    fig.suptitle(f"{model}, {rule}; 95 % CI; hollow = insufficient group"
                 + ("; dashed = overall" if overall is not None else ""), fontsize=6.5)
    return fig


# ── mitigation ────────────────────────────────────────────────────────────────────────────

MITIGATION_ARMS = (  # (effect name pattern, label, colour, marker)
    ("rw_kc_{a}@topq", "K&C reweighing", ORANGE, "o"),
    ("rw_cell_{a}@topq", "cell balancing", SKY, "s"),
    ("eocap_{a}", "EO at capacity", GREEN, "D"),
    ("to_samek_{a}", "ThresholdOptimizer", VERMILLION, "^"),
)
RATE_ARMS = (  # (audit key pattern, label, colour, marker, flags about q %)
    ("base__topq", "baseline", BLACK, "o", True),
    ("rw_kc_{a}__topq", "K&C", ORANGE, "o", True),
    ("rw_cell_{a}__topq", "cell", SKY, "s", True),
    ("val__topq", "top q % ({fit} model)", GREY, "o", True),  # model fitted without the hold-out
    ("eocap_{a}__eo", "EO at capacity", GREEN, "D", True),
    ("val__youden", "single threshold", BLUE, "v", False),
    ("to_{a}__eo", "ThresholdOptimizer", VERMILLION, "^", False),
)


def fig_mitigation(mitigation: dict, *, attribute: str | None = None) -> Figure:
    """Mitigation trade-off and who moves (approved_mitigation.md §1–§3, §5.1).

    Left: Δ EO against Δ accuracy per arm, mitigated − its unconstrained twin (95 % CIs of the
    paired difference). Reweighing and EO at capacity flag q %; ThresholdOptimizer is compared
    with the same model flagging exactly as many (same-number twin). Right: TPR per group per
    arm; filled = arms that flag about q %, hollow = arms that flag about a third (balanced-
    accuracy objective) — compare within each set. `attribute` defaults to the leading one.
    """
    use_style()
    a = attribute or mitigation["leading_attribute"]
    eff, audit = mitigation["effects"], mitigation["audit"]
    fig = Figure(figsize=(WIDTH, 2.6), layout="constrained")
    ax_t, ax_r = fig.subplots(1, 2, width_ratios=(1, 1.2))

    for pattern, label, colour, marker in MITIGATION_ARMS:
        e = eff[pattern.format(a=a)]
        x, xe = _err(e["accuracy"], "difference")
        y, ye = _err(e[f"eo_{a}"], "difference")
        ax_t.errorbar(x, y, xerr=xe, yerr=ye, fmt=marker, color=colour, ms=3.5, elinewidth=0.9,
                      label=label)
    ax_t.axhline(0, color=GREY, lw=0.5)
    ax_t.axvline(0, color=GREY, lw=0.5)
    ax_t.set_xlabel("Δ accuracy")
    ax_t.set_ylabel(f"Δ EO difference ({a})")
    ax_t.xaxis.set_major_locator(mpl.ticker.MaxNLocator(2))
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), handletextpad=0.2,
                fontsize=5.5, labelspacing=0.2)

    groups = [g for g in audit["groups"][a] if g not in NOT_A_GROUP.get(a, ())]
    x = np.arange(len(groups))
    width = 0.8 / len(RATE_ARMS)
    for i, (pattern, label, colour, marker, capacity) in enumerate(RATE_ARMS):
        k = pattern.format(a=a)
        vals = [audit["groups"][a][g]["metrics"][k]["tpr"]["estimate"] for g in groups]
        fit = f"{1 - mitigation['config']['val_fraction']:.0%}"
        ax_r.plot(x - 0.4 + width * (i + 0.5), vals, marker, color=colour, ms=3,
                  mfc=colour if capacity else "white", ls="none", label=label.format(fit=fit))
    ax_r.set_xticks(x, groups, rotation=30)
    ax_r.set_xlabel(a)
    ax_r.set_ylabel("TPR")
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=2, handletextpad=0.1,
                columnspacing=0.6, fontsize=5.5, labelspacing=0.2)
    ax_r.set_ylim(0, None)
    return fig


# ── sources ───────────────────────────────────────────────────────────────────────────────

SOURCE_ARMS = (  # (arm, short label)
    ("without_demographics", "− demographics"),
    ("without_prior_use", "− prior use"),
    ("without_current_stay", "− current stay"),
    ("without_admission", "− admission block"),
    ("without_diagnosis", "− diagnosis"),
    ("without_diabetes_care", "− diabetes care"),
    ("no_sex", "− sex"),
    ("no_race", "− race"),
    ("no_age", "− age"),
    ("refit_seed", "other seed (ref.)"),
)


def fig_sources(sources: dict, *, attribute: str | None = None) -> Figure:
    """Δ EO of the leading attribute per refit arm (approved_sources.md §1–§2).

    Dots: refit without the block − full model, top q %, 95 % CI. Grey diamonds: the change the
    full model's gap shows when its ranking loses the same ROC-AUC to random noise (matched
    degradation; bar = min–max over the noise draws). An arm well below its diamond removed
    inputs that carry the gap, not only predictive signal. The age-sex-race-only model is not
    shown (its gap is not interpreted, §2).
    """
    use_style()
    a = attribute or sources["leading_attribute"]
    eff, deg = sources["effects"], sources["degradation"]
    arms = [(arm, lbl) for arm, lbl in SOURCE_ARMS if f"{arm}@topq" in eff]
    fig = Figure(figsize=(WIDTH, 0.5 + 0.18 * len(arms)), layout="constrained")
    ax = fig.subplots()
    ypos = np.arange(len(arms))[::-1]
    for y, (arm, _) in zip(ypos, arms, strict=True):
        e = eff[f"{arm}@topq"][f"eo_{a}"]
        full = e["counterpart"]
        d = deg[arm]
        ref = d[f"eo_{a}"] - full
        ax.errorbar(ref, y - 0.18, xerr=[[ref - (d[f"eo_{a}_min"] - full)],
                                         [(d[f"eo_{a}_max"] - full) - ref]],
                    fmt="D", color=GREY, ms=3, elinewidth=0.9,
                    label="same ROC-AUC loss, random" if y == ypos[0] else None)
        v, err = _err(e, "difference")
        ax.errorbar(v, y + 0.1, xerr=err, fmt="o", color=BLUE, ms=3.5, elinewidth=1.1,
                    label="refit − full model" if y == ypos[0] else None)
    ax.axvline(0, color=BLACK, lw=0.5)
    ax.set_yticks(ypos, [lbl for _, lbl in arms])
    ax.set_xlabel(f"Δ EO difference ({a}), top q %")
    ax.legend(loc="lower left", borderaxespad=0.1)
    null = sources.get("null_mean")
    if null is not None:
        full = eff["refit_seed@topq"][f"eo_{a}"]["counterpart"]
        ax.axvline(null - full, color=VERMILLION, ls="--", lw=0.8)
        ax.text(null - full, ypos[0] + 0.55, " gap down to its null", color=VERMILLION,
                fontsize=5.5, ha="left", va="bottom")
    ax.set_ylim(-0.7, len(arms) - 0.1)
    return fig


# ── test ordering ─────────────────────────────────────────────────────────────────────────

ORDERING_SERIES = (  # (analysis, outcome, label, colour, marker)
    ("all_encounters", "hba1c_requested", "HbA1c", BLUE, "o"),
    ("all_encounters", "glucose_requested", "glucose", ORANGE, "s"),
    ("source_adjusted", "glucose_requested", "glucose + source indicator", GREEN, "D"),
)


def fig_ordering(ordering: dict) -> Figure:
    """Adjusted odds ratios of a recorded result, clustered 95 % CI (approved_ordering.md §2, §4).

    Tested terms only (sex, age band, race; race Unknown is a missing code and not tested),
    against the reference group of each variable. Log scale; 1 = same odds as the reference.
    "glucose + source indicator" adds the admission-type-unknown indicator (post hoc
    sensitivity fit, not tested).
    """
    use_style()
    first = ordering[ORDERING_SERIES[0][0]][ORDERING_SERIES[0][1]]["terms"]
    terms = [n for n, t in first.items() if t["tested"]]
    fig = Figure(figsize=(WIDTH, 0.6 + 0.2 * len(terms)), layout="constrained")
    ax = fig.subplots()
    ypos = np.arange(len(terms))[::-1]
    offsets = np.linspace(0.22, -0.22, len(ORDERING_SERIES))
    for (analysis, outcome, label, colour, marker), off in zip(ORDERING_SERIES, offsets, strict=True):
        tt = ordering[analysis][outcome]["terms"]
        for y, n in zip(ypos, terms, strict=True):
            t = tt[n]
            lo, hi = t["ci"]
            ax.errorbar(t["odds_ratio"], y + off,
                        xerr=[[t["odds_ratio"] - lo], [hi - t["odds_ratio"]]],
                        fmt=marker, color=colour, ms=3, elinewidth=0.9,
                        label=label if y == ypos[0] else None)
    ax.axvline(1, color=BLACK, lw=0.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    labels = [f"{first[n]['level']} vs {first[n]['reference']}" for n in terms]
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("adjusted odds ratio (log scale)")  # of a recorded result: see the caption
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.2), ncols=3, handletextpad=0.1,
              columnspacing=0.8, fontsize=5.5)
    ax.set_ylim(-0.7, len(terms) - 0.3)
    return fig


FIGURES = {  # name -> (function, JSON names it takes)
    "models": (fig_models, ("summary",)),
    "complexity": (fig_complexity, ("summary",)),
    "fairness": (fig_fairness, ("fairness",)),
    "mitigation": (fig_mitigation, ("mitigation",)),
    "sources": (fig_sources, ("sources",)),
    "ordering": (fig_ordering, ("ordering",)),
}


def all_figures(results: dict) -> dict[str, Figure]:
    """Every figure of FIGURES from load_results output: {name: Figure}."""
    fair = results["fairness"]
    extra = {"fairness": {"overall": results["summary"]["decision"].get(
        f"{fair['primary_model']}__topq")}}  # dashed overall rates on the fairness figure
    return {name: fn(*(results[n] for n in needs), **extra.get(name, {}))
            for name, (fn, needs) in FIGURES.items()}
