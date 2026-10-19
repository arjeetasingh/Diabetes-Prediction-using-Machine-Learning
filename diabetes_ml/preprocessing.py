"""Leak-free preprocessing, built as scikit-learn pipelines.

Everything that learns from data (imputation values, scaling) lives *inside* the
pipeline, so cross-validation never lets test rows influence training.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, ZERO_MEANS_MISSING


class HiddenMissingToNaN(BaseEstimator, TransformerMixin):
    """Turn impossible zeros (and optionally extreme insulin) into NaN.

    This step is stateless (it learns nothing), so it cannot leak information.
    """

    def __init__(self, insulin_cap: float | None = None):
        self.insulin_cap = insulin_cap

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            X = X[FEATURES]
        A = np.array(X, dtype=float, copy=True)
        if A.ndim != 2 or A.shape[1] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} feature columns, got shape {A.shape}.")
        for col in ZERO_MEANS_MISSING:
            j = FEATURES.index(col)
            A[A[:, j] == 0, j] = np.nan
        if self.insulin_cap is not None:
            j = FEATURES.index("Insulin")
            A[A[:, j] > self.insulin_cap, j] = np.nan
        return A


def penalised_logreg(kind: str, C: float) -> LogisticRegression:
    """Logistic Regression with an L1 ('l1') or L2 ('l2') penalty, on any scikit-learn version.

    scikit-learn 1.8 deprecated the `penalty=` argument in favour of `l1_ratio`
    (1.0 = pure L1, 0.0 = pure L2); older versions only understand `penalty=`.
    """
    if kind not in ("l1", "l2"):
        raise ValueError("kind must be 'l1' or 'l2'.")
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=C, l1_ratio=1.0 if kind == "l1" else 0.0, solver="liblinear", max_iter=5000)
    return LogisticRegression(penalty=kind, C=C, solver="liblinear", max_iter=5000)


def build_pipeline(
    estimator=None,
    insulin_cap: float | None = None,
    add_indicator: bool = False,
    scale: bool = True,
) -> Pipeline:
    """Missing-value handling -> median imputation -> scaling -> model."""
    if estimator is None:
        estimator = LogisticRegression(max_iter=5000)
    steps = [
        ("missing", HiddenMissingToNaN(insulin_cap=insulin_cap)),
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)
