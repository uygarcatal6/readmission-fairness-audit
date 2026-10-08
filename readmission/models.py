"""
readmission/models.py — the classifiers of the approved protocol (docs/APPROVED_PROPOSAL.md).

Approved models: decision tree, k-NN, logistic regression, random forest, gradient boosting.
Course-topic extensions (reported separately, never replacing an approved model):
    mlp      04-neuralnetworks: a small feed-forward network (scikit-learn, CPU)
    bagging  05-ensemble: bagged decision trees, the step between one tree and a forest

make_estimator returns a bare scikit-learn estimator. The evaluation loop pairs it with a
preprocessor fitted on the training rows only (readmission/preprocess.py); make_pipeline
does the same pairing in one object for notebooks.

Class-weighted variants ("class_weight='balanced'") exist for the models that support it;
they are the "class-weighted model at 0.5" sensitivity check. k-NN and MLP have no class
weights in scikit-learn, so they have no such variant.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.ensemble import (
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from readmission.preprocess import make_preprocessor

APPROVED = ("logreg", "tree", "knn", "rf", "hgb")
EXTENSIONS = ("mlp", "bagging")
CLASS_WEIGHTED = ("logreg", "tree", "rf", "hgb")  # models that accept class_weight

# Hyperparameters chosen by the one-SE rule on inner validation folds (readmission/tuning.py).
TUNED = {"tree": "max_depth", "knn": "n_neighbors"}
TREE_DEPTHS: tuple[int | None, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None)
# Up to 1501: validation AUC was still rising at k = 201 (review, 23 Sep), and a grid that
# stops too early lets the grid edge, not the one-SE rule, choose k.
KNN_KS: tuple[int, ...] = (1, 3, 5, 11, 21, 51, 101, 201, 301, 501, 751, 1001, 1501)

# Fixed settings (not tuned; written down so every run uses the same ones).
RF_TREES = 300
RF_MIN_LEAF = 5
BAGGING_TREES = 100
# The 23 Sep inner-CV probe (training part of outer split r0 f0 only) stopped after 13-20
# epochs with early stopping; 20 fixed epochs reproduces that without an encounter-level
# split. scikit-learn warns that it did not converge; that is expected here.
MLP_EPOCHS = 20


def make_estimator(
    name: str,
    *,
    params: dict | None = None,
    class_weight: str | None = None,
    random_state: int = 0,
):
    """Bare estimator for `name`; `params` overrides the defaults (e.g. the tuned depth)."""
    params = dict(params or {})
    if class_weight is not None and name not in CLASS_WEIGHTED:
        raise ValueError(f"{name} does not support class weights")
    cw = {"class_weight": class_weight} if class_weight else {}
    if name == "logreg":
        est = LogisticRegression(max_iter=3000, **cw)
    elif name == "tree":
        est = DecisionTreeClassifier(random_state=random_state, **cw)
    elif name == "knn":
        est = KNeighborsClassifier(n_jobs=-1)
    elif name == "rf":
        est = RandomForestClassifier(
            n_estimators=RF_TREES, min_samples_leaf=RF_MIN_LEAF, n_jobs=-1,
            random_state=random_state, **cw)
    elif name == "hgb":
        # Early stopping off: its internal validation split would be encounter-level, and a
        # fixed number of boosting rounds is easier to explain.
        est = HistGradientBoostingClassifier(early_stopping=False, random_state=random_state, **cw)
    elif name == "mlp":
        # Same reason as HGB: early stopping would hold out random ENCOUNTERS, so a patient's
        # other stays leak into its validation part. A fixed MLP_EPOCHS is used instead.
        est = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=False,
                            max_iter=MLP_EPOCHS, random_state=random_state)
    elif name == "bagging":
        est = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=RF_MIN_LEAF),
                                n_estimators=BAGGING_TREES, n_jobs=-1, random_state=random_state)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {APPROVED + EXTENSIONS}")
    return est.set_params(**params) if params else est


def make_pipeline(
    name: str,
    *,
    features: Sequence[str] | None = None,
    params: dict | None = None,
    class_weight: str | None = None,
    random_state: int = 0,
) -> Pipeline:
    """Preprocessor + estimator in one object; fit it on training rows only."""
    return Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", make_estimator(name, params=params, class_weight=class_weight,
                                 random_state=random_state)),
    ])
