"""Tests for readmission/runtime.py — the thread cap is a runtime knob, not part of the method."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from joblib import effective_n_jobs
from threadpoolctl import threadpool_info, threadpool_limits

from readmission import evaluate, runtime


def test_default_is_half_the_cpus_at_least_two(monkeypatch):
    for cpus, want in ((1, 1), (2, 2), (3, 2), (4, 2), (8, 4), (12, 6)):
        monkeypatch.setattr(os, "cpu_count", lambda cpus=cpus: cpus)
        assert runtime.default_n_jobs() == want


def test_resolve_order_flag_then_env_then_default(monkeypatch):
    monkeypatch.setenv(runtime.ENV_VAR, "3")
    assert runtime.resolve_n_jobs(5) == 5
    assert runtime.resolve_n_jobs() == 3
    monkeypatch.delenv(runtime.ENV_VAR)
    assert runtime.resolve_n_jobs() == runtime.default_n_jobs()
    with pytest.raises(ValueError):
        runtime.resolve_n_jobs(0)


def test_cap_reaches_joblib_and_the_native_pools(monkeypatch):
    monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
    before = [p["num_threads"] for p in threadpool_info()]
    try:
        assert runtime.cap_threads(1) == 1
        assert effective_n_jobs(-1) == 1  # what RandomForest(n_jobs=-1) will use
        assert all(p["num_threads"] == 1 for p in threadpool_info())  # BLAS / OpenMP
    finally:  # restore the pools and joblib for the other tests
        monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
        for pool, n in zip(threadpool_info(), before, strict=True):
            threadpool_limits(limits=n, user_api=pool["user_api"])


def test_cap_works_before_numpy_and_sklearn_are_imported():
    """A notebook may call cap_threads first thing; the pools loaded later must be capped too."""
    code = ("from readmission import runtime; runtime.cap_threads(1); "
            "from threadpoolctl import threadpool_info; "
            "print(sorted({p['num_threads'] for p in threadpool_info()}))")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True)
    assert out.stdout.strip() == "[1]"


def test_knob_is_not_part_of_the_fingerprint():
    assert "n_jobs" not in asdict(evaluate.Config())
    assert "runtime" not in evaluate._PREDICTION_CODE
