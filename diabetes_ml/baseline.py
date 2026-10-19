"""Reproduction of the original course project (the part shown in the slides).

Original recipe: insulin > 425 -> NaN, fill every NaN with the column mean
(computed on the WHOLE dataset), 80/20 split, Logistic Regression, forward
feature selection, L1/L2 curves. It is kept as-is so the extension can be
compared against it honestly.

Note: the mean imputation here uses all 768 rows *before* the split, a small
form of data leakage. The extension fixes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, ORIGINAL_INSULIN_CAP, RANDOM_STATE, TARGET
from .preprocessing import penalised_logreg


def original_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Original cleaning: insulin > 425 -> NaN, then fill NaN with the full-data mean."""
    d = df.copy()
    d.loc[d["Insulin"] > ORIGINAL_INSULIN_CAP, "Insulin"] = np.nan
    d = d.apply(pd.to_numeric, errors="coerce")
    return d.fillna(d.mean())


def run_baseline(df: pd.DataFrame, random_state: int = RANDOM_STATE, test_size: float = 0.2) -> dict:
    """One 80/20 split with Logistic Regression, exactly as in the slides."""
    d = original_clean(df)
    X, y = d[FEATURES], d[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "auc": roc_auc_score(y_te, proba),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "confusion": confusion_matrix(y_te, pred),
        "fpr": fpr,
        "tpr": tpr,
        "n_test": len(y_te),
    }


def split_variability(df: pd.DataFrame, n_splits: int = 200) -> pd.DataFrame:
    """Repeat the same 80/20 experiment with many different random splits.

    A single split gives a single number. This shows how much that number moves
    just because of *which* rows landed in the test set.
    """
    rows = []
    for seed in range(n_splits):
        r = run_baseline(df, random_state=seed)
        rows.append({"seed": seed, "accuracy": r["accuracy"], "auc": r["auc"], "recall": r["recall"]})
    return pd.DataFrame(rows)


def forward_selection(df: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    """Greedy forward selection by cross-validated ROC-AUC (as in the slides)."""
    d = original_clean(df)
    y = d[TARGET]
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    chosen: list[str] = []
    remaining = list(FEATURES)
    rows = []
    while remaining:
        scores = {}
        for f in remaining:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
            scores[f] = cross_val_score(model, d[chosen + [f]], y, cv=skf, scoring="roc_auc").mean()
        best = max(scores, key=scores.get)
        chosen.append(best)
        remaining.remove(best)
        rows.append({"step": len(chosen), "added": best, "cv_auc": scores[best], "features": ", ".join(chosen)})
    return pd.DataFrame(rows)


def regularization_curves(df: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    """Cross-validated AUC against regularisation strength (lambda = 1 / C), L1 and L2."""
    d = original_clean(df)
    X, y = d[FEATURES], d[TARGET]
    skf = StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for lam in np.round(np.linspace(0.02, 2.0, 25), 4):
        for penalty in ("l1", "l2"):
            model = make_pipeline(StandardScaler(), penalised_logreg(penalty, 1.0 / lam))
            auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc").mean()
            rows.append({"lambda": lam, "penalty": penalty.upper(), "cv_auc": auc})
    return pd.DataFrame(rows)
