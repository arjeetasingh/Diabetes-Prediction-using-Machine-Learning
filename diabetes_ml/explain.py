"""Explaining a prediction.

For a logistic regression on standardised features, the exact contribution of
feature j to a patient's log-odds, relative to the average training patient, is

    contribution_j = coef_j * z_j        with z_j = (x_j - mean_j) / sd_j

These are the same numbers a SHAP analysis gives for a linear model, without
needing the SHAP library.  intercept + sum(contributions) = the model's log-odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, RANDOM_STATE, TARGET
from .preprocessing import build_pipeline


def contributions(pipeline, row: pd.DataFrame) -> pd.DataFrame:
    """Per-feature push (in log-odds) for one patient. Positive = raises diabetes risk."""
    if len(row) != 1:
        raise ValueError("Pass exactly one patient row.")
    raw_after_missing = pipeline.named_steps["missing"].transform(row)[0]
    used = pipeline[:2].transform(row)[0]
    z = pipeline.named_steps["scale"].transform(pipeline[:2].transform(row))[0]
    coef = pipeline.named_steps["model"].coef_[0]
    out = pd.DataFrame(
        {
            "feature": FEATURES,
            "value": raw_after_missing,       # NaN where the value was not measured
            "value_used": used,               # after median imputation
            "z_score": z,
            "coef_per_sd": coef,
            "contribution": coef * z,
        }
    )
    out["was_missing"] = np.isnan(raw_after_missing)
    return out


def intercept_and_logit(pipeline, row: pd.DataFrame) -> tuple[float, float]:
    """(intercept, total log-odds) so that intercept + sum(contributions) == total."""
    model = pipeline.named_steps["model"]
    return float(model.intercept_[0]), float(pipeline.decision_function(row)[0])


def odds_ratios(df: pd.DataFrame, n_boot: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Odds ratio per +1 SD of each feature, with a bootstrap 95 % interval.

    An odds ratio of 1.5 means: a patient 1 SD higher on this feature has 1.5x the
    odds of diabetes, other features held equal. An interval that crosses 1.0 means
    the effect is not clearly different from zero.
    """
    rng = np.random.default_rng(seed)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    full = build_pipeline().fit(X, y).named_steps["model"].coef_[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        if y[idx].min() == y[idx].max():
            continue
        boots.append(build_pipeline().fit(X.iloc[idx], y[idx]).named_steps["model"].coef_[0])
    boots = np.exp(np.array(boots))
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "odds_ratio_per_sd": np.exp(full),
            "ci_low": np.percentile(boots, 2.5, axis=0),
            "ci_high": np.percentile(boots, 97.5, axis=0),
        }
    ).sort_values("odds_ratio_per_sd", ascending=False).reset_index(drop=True)


def global_importance(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute contribution across a dataset (how much each feature moves predictions)."""
    Z = pipeline[:-1].transform(X)
    coef = pipeline.named_steps["model"].coef_[0]
    imp = np.abs(Z * coef).mean(axis=0)
    return pd.DataFrame({"feature": FEATURES, "mean_abs_contribution": imp}).sort_values(
        "mean_abs_contribution", ascending=False
    ).reset_index(drop=True)
