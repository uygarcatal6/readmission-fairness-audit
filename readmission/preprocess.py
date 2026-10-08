"""
readmission/preprocess.py — encoding and scaling, fitted inside each training fold.

make_preprocessor returns an UNFITTED ColumnTransformer. It is always the first step of a
Pipeline, so pipeline.fit(X_train, y_train) learns the category lists, the rare-level
grouping and the means / standard deviations from the training fold alone; the test fold
is only transformed (docs/APPROVED_PROPOSAL.md §2: "Encoding and scaling are fitted inside
the training folds only").

    numeric      StandardScaler — needed by logistic regression and k-NN (distances),
                 harmless for trees.
    categorical  OneHotEncoder — levels seen in fewer than MIN_FREQUENCY of the training
                 rows are pooled into one "infrequent" column. A level never seen in
                 training maps to that column, or to all zeros if the fold has none.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from readmission.cohort import CATEGORICAL, FEATURES, NUMERIC

# 0.5 % of the training rows (~390 of a 78.8k-encounter fold). The old adapter pooled levels
# with < 500 encounters in the whole file, i.e. about the same share.
MIN_FREQUENCY = 0.005


def make_preprocessor(features: Sequence[str] | None = None) -> ColumnTransformer:
    """Unfitted transformer for `features` (default: all 23 model inputs).

    Passing a subset (e.g. without sex/age/race) is how the ablation and unawareness runs
    drop attributes; unknown names raise instead of being silently ignored.
    """
    features = list(FEATURES if features is None else features)
    unknown = sorted(set(features) - set(NUMERIC) - set(CATEGORICAL))
    if unknown:
        raise ValueError(f"not model features: {unknown}")
    numeric = [c for c in NUMERIC if c in features]
    categorical = [c for c in CATEGORICAL if c in features]

    steps = []
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    if categorical:
        steps.append((
            "cat",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=MIN_FREQUENCY,
                sparse_output=False,
            ),
            categorical,
        ))
    return ColumnTransformer(steps, remainder="drop")
