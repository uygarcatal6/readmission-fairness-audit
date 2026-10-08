"""Tests for readmission/figures.py — every figure builds from the saved result JSONs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from readmission import figures

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def results():
    try:
        return figures.load_results(ROOT)
    except FileNotFoundError as err:
        pytest.skip(f"no saved results: {err}")


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_each_figure_builds(results, name):
    fn, needs = figures.FIGURES[name]
    fig = fn(*(results[n] for n in needs))
    assert isinstance(fig, Figure)
    assert fig.axes, "the figure has no axes"
    assert fig.get_figwidth() == pytest.approx(figures.WIDTH)  # one IEEE column


def test_all_figures_and_saving(results, tmp_path):
    out = figures.all_figures(results)
    assert set(out) == set(figures.FIGURES)
    for name, fig in out.items():
        fig.savefig(tmp_path / f"{name}.png", dpi=50)
        assert (tmp_path / f"{name}.png").stat().st_size > 0


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_figure_text_stays_inside(results, name):
    # Axis labels, titles and legend entries must lie inside the figure, or the saved PNG/PDF
    # cuts them off (as happened once with a long x-axis label).
    fn, needs = figures.FIGURES[name]
    fig = fn(*(results[n] for n in needs))
    fig.set_dpi(100)
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()
    texts = [fig._suptitle] if fig._suptitle is not None else []
    for ax in fig.axes:
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title]
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
    for legend in fig.legends:
        texts += legend.get_texts()
    for text in texts:
        if not text.get_text() or not text.get_visible():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.x1 <= fig.bbox.x1, f"{text.get_text()!r} is cut off"
        assert box.y0 >= 0 and box.y1 <= fig.bbox.y1, f"{text.get_text()!r} is cut off"


def test_fairness_figure_marks_insufficient_groups(results):
    # With pooled out-of-fold flags no group of the real run is below 50 positives or
    # negatives, so mark one by hand: its TPR marker must be drawn hollow (white face).
    fair = copy.deepcopy(results["fairness"])
    key = f"{fair['primary_model']}__topq"
    fair["groups"]["race"]["Asian"]["metrics"][key]["tpr"]["insufficient"] = True
    fig = figures.fig_fairness(fair, key=key)
    faces = [line.get_markerfacecolor() for line in fig.axes[0].lines]
    assert faces.count("white") == 1


def test_load_results_names_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="summary.json"):
        figures.load_results(tmp_path)
