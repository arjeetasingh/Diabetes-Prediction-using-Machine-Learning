"""Are the predicted probabilities honest? (A "0.7" should mean about 70 %.)"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .config import FEATURES, RANDOM_STATE, TARGET
from .evaluation import oof_probabilities
from .preprocessing import build_pipeline


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Average gap between predicted and observed risk, weighted by bin size (quantile bins)."""
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(p)
    ece = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        ece += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def reliability_points(y, p, n_bins: int = 8):
    """(mean predicted, observed fraction) per quantile bin, for a reliability diagram."""
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac


def calibration_study(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold probabilities for LR, RF, and RF after sigmoid calibration."""
    X, y = df[FEATURES], df[TARGET]
    rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)  # noqa: E731
    models = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=5000)),
        "Random Forest (raw)": build_pipeline(rf()),
        "Random Forest (sigmoid-calibrated)": build_pipeline(CalibratedClassifierCV(rf(), method="sigmoid", cv=5)),
    }
    probs, rows = {}, []
    for name, pipe in models.items():
        p = np.clip(oof_probabilities(pipe, X, y), 1e-6, 1 - 1e-6)
        probs[name] = p
        rows.append(
            {
                "model": name,
                "brier": brier_score_loss(y, p),
                "log_loss": log_loss(y, p),
                "ece": expected_calibration_error(y, p),
            }
        )
    return pd.DataFrame(rows), probs
