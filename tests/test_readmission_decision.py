"""Contract of readmission/decision.py (flag_top_q, youden_threshold)."""

from __future__ import annotations

import numpy as np
import pytest

from readmission.decision import flag_top_q, youden_threshold


def test_flags_exactly_k_highest():
    score = np.array([0.9, 0.1, 0.5, 0.7, 0.3, 0.2, 0.8, 0.4, 0.6, 0.0])
    flags = flag_top_q(score, 0.3, rng=np.random.default_rng(0))
    assert flags.dtype == bool and flags.sum() == 3
    assert set(np.flatnonzero(flags)) == {0, 6, 3}  # 0.9, 0.8, 0.7


def test_capacity_is_fixed_even_with_ties():
    score = np.array([0.5] * 8 + [0.9, 0.1])  # cut-off falls inside the 0.5 block
    flags = flag_top_q(score, 0.4, rng=np.random.default_rng(0))
    assert flags.sum() == 4
    assert flags[8] and not flags[9]
    assert score[flags].min() >= score[~flags].max()


def test_ties_are_not_broken_by_row_order():
    score = np.full(1000, 0.5)
    picked = [flag_top_q(score, 0.1, rng=np.random.default_rng(s)) for s in range(2)]
    assert not np.array_equal(picked[0], picked[1])
    assert picked[0][:100].sum() < 100  # not simply the first 100 rows


@pytest.mark.parametrize("q, expected", [(0.0, 0), (1.0, 10)])
def test_edge_shares(q, expected):
    assert flag_top_q(np.linspace(0, 1, 10), q, rng=np.random.default_rng(0)).sum() == expected


@pytest.mark.parametrize("q", [-0.1, 1.5])
def test_rejects_invalid_share(q):
    with pytest.raises(ValueError):
        flag_top_q(np.linspace(0, 1, 10), q, rng=np.random.default_rng(0))


def test_rng_is_required_and_lists_are_accepted():
    with pytest.raises(TypeError):
        flag_top_q(np.full(10, 0.5), 0.3)  # no silent row-order fallback
    assert flag_top_q([0.2, 0.9, 0.5], 1 / 3, rng=np.random.default_rng(0)).tolist() == [
        False, True, False]


def test_youden_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert youden_threshold(y, score) == 0.8
