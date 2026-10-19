"""Choosing a decision threshold for screening instead of defaulting to 0.5."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_at(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "accuracy": (tp + tn) / len(y),
    }


def threshold_table(y, p, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return pd.DataFrame([confusion_at(y, p, t) for t in thresholds])


def threshold_for_sensitivity(y, p, target: float = 0.85) -> float:
    """Highest threshold that still catches at least `target` of the diabetic patients."""
    grid = np.round(np.arange(0.02, 0.98, 0.01), 2)
    ok = [t for t in grid if confusion_at(y, p, t)["sensitivity"] >= target]
    return float(max(ok)) if ok else float(grid[0])
