"""Fair model comparison, preprocessing ablation and subgroup checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .baseline import original_clean
from .config import FEATURES, ORIGINAL_INSULIN_CAP, RANDOM_STATE, TARGET
from .preprocessing import build_pipeline

SCORING = {"accuracy": "accuracy", "auc": "roc_auc", "recall": "recall", "precision": "precision", "brier": "neg_brier_score"}


def model_zoo() -> dict:
    """The four models from the original project, plus gradient boosting."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
        # SVC(probability=True) is deprecated; this is the recommended replacement (Platt scaling).
        "SVM (RBF)": CalibratedClassifierCV(SVC(kernel="rbf"), method="sigmoid", cv=5, ensemble=False),
        "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=RANDOM_STATE),
    }


def _cv(n_repeats: int = 5) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)


def _summarise(res: dict) -> dict:
    out = {}
    for key in SCORING:
        vals = res[f"test_{key}"] * (-1 if key == "brier" else 1)
        out[f"{key}_mean"] = vals.mean()
        out[f"{key}_sd"] = vals.std(ddof=1)
    return out


def compare_models(df: pd.DataFrame, n_repeats: int = 5) -> pd.DataFrame:
    """Repeated 5-fold CV on identical folds for every model, with leak-free preprocessing."""
    X, y = df[FEATURES], df[TARGET]
    rows = []
    for name, est in model_zoo().items():
        res = cross_validate(build_pipeline(est), X, y, cv=_cv(n_repeats), scoring=SCORING, n_jobs=1)
        rows.append({"model": name, **_summarise(res)})
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


def preprocessing_ablation(df: pd.DataFrame, n_repeats: int = 5) -> pd.DataFrame:
    """Same model, same folds, different data cleaning. Isolates what cleaning is worth."""
    y = df[TARGET]
    raw = df[FEATURES]
    lr = lambda: LogisticRegression(max_iter=5000)  # noqa: E731
    variants = {
        "A. Original recipe (insulin>425 removed, full-data mean fill)": (
            make_pipeline(StandardScaler(), lr()),
            original_clean(df)[FEATURES],
        ),
        "B. Zeros kept as real values (no cleaning)": (make_pipeline(StandardScaler(), lr()), raw),
        "C. Hidden zeros -> missing, median fill inside CV (used here)": (build_pipeline(lr()), raw),
        "D. C plus insulin>425 removed": (build_pipeline(lr(), insulin_cap=ORIGINAL_INSULIN_CAP), raw),
        "E. C plus missing-value indicator columns": (build_pipeline(lr(), add_indicator=True), raw),
    }
    rows = []
    for name, (pipe, X) in variants.items():
        res = cross_validate(pipe, X, y, cv=_cv(n_repeats), scoring=SCORING, n_jobs=1)
        rows.append({"variant": name, **_summarise(res)})
    return pd.DataFrame(rows)


def oof_probabilities(pipeline, X, y, n_splits: int = 10, seed: int = RANDOM_STATE) -> np.ndarray:
    """Out-of-fold probability of class 1: every row is predicted by a model that never saw it."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]


def subgroup_report(df: pd.DataFrame, p_oof: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Out-of-fold performance in age groups. Small groups, so read with caution."""
    y = df[TARGET].to_numpy()
    bins = pd.cut(df["Age"], [20, 29, 39, 49, 100], labels=["21-29", "30-39", "40-49", "50+"])
    rows = []
    for grp in bins.cat.categories:
        m = (bins == grp).to_numpy()
        yt, pt = y[m], p_oof[m]
        pos = yt == 1
        rows.append(
            {
                "age_group": grp,
                "n": int(m.sum()),
                "prevalence_pct": 100 * yt.mean(),
                "auc": roc_auc_score(yt, pt) if 0 < yt.sum() < len(yt) else np.nan,
                "sensitivity_pct": 100 * (pt[pos] >= threshold).mean() if pos.any() else np.nan,
                "specificity_pct": 100 * (pt[~pos] < threshold).mean() if (~pos).any() else np.nan,
                "brier": brier_score_loss(yt, pt),
            }
        )
    return pd.DataFrame(rows)
