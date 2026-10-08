"""
readmission/runtime.py — how many CPU threads a run may use.

A runtime knob, not part of the method: the number of threads changes how long a run takes,
not the reported results. It is therefore NOT part of evaluate.Config and NOT in the
checkpoint fingerprint, so changing it never invalidates saved checkpoints.

Last digits (checked 24 Sep 2026, 2 vs 12 threads): BLAS splits a sum of non-integer numbers
differently for another thread count, so a few floating-point values move in their last
digits: the logistic regression / MLP fits (~1e-13), the float32 calibration sums of
fairness.py (~1e-8) and the ordering Logit (~1e-14). Flags, the bootstrap's integer-weight
sums and every number printed in the reports stayed the same. The saved outputs were made
with 12 threads; --n-jobs 12 reruns them under that same condition.

Three thread pools take part in a run, each with its own switch, all set by cap_threads:
    joblib   random forest, bagging and the inner-loop trees ask for n_jobs=-1 ("all CPUs").
             joblib counts CPUs with loky.cpu_count(), which honours the environment variable
             LOKY_MAX_CPU_COUNT, so that variable caps every n_jobs=-1 without touching the
             model code.
    OpenMP   gradient boosting and the brute-force k-NN search (scikit-learn's pairwise
             distance kernels) — threadpoolctl.
    BLAS     logistic regression, the MLP and the bootstrap matrix products — threadpoolctl.

Choose with the --n-jobs flag of the run scripts or the environment variable
READMISSION_N_JOBS. Default: half of the logical CPUs, at least 2. On the 12-thread laptop
that is 6 (about its physical cores: the machine stays usable and cool); on Google Colab's
2 vCPUs it is 2 (full speed).
"""

from __future__ import annotations

import os

from threadpoolctl import threadpool_limits

ENV_VAR = "READMISSION_N_JOBS"


def default_n_jobs() -> int:
    """Half of the logical CPUs, at least 2, never more than there are."""
    cpus = os.cpu_count() or 1
    return min(cpus, max(2, cpus // 2))


def resolve_n_jobs(n_jobs: int | None = None) -> int:
    """`n_jobs` if given, else READMISSION_N_JOBS if set, else the default; always >= 1."""
    if n_jobs is None:
        env = os.environ.get(ENV_VAR)
        n_jobs = int(env) if env else default_n_jobs()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1, got {n_jobs}")
    return n_jobs


def cap_threads(n_jobs: int | None = None) -> int:
    """Cap joblib, OpenMP and BLAS at `n_jobs` threads for the rest of the process.

    Call it once at the start of a script or notebook, before any model is fitted.
    Returns the cap that was applied.
    """
    n = resolve_n_jobs(n_jobs)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(n)  # what joblib's n_jobs=-1 resolves to
    # threadpoolctl caps only the pools already loaded into the process, so load the
    # libraries that own them first (numpy / scipy: BLAS; scikit-learn: OpenMP).
    import sklearn.ensemble  # noqa: F401

    threadpool_limits(limits=n)
    return n
