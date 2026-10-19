"""The end-to-end screening model: probability + uncertainty + explanation + what-if.

    model = ScreeningModel().fit(load_pima())
    result = model.predict({"Glucose": 160, "BMI": 35, ...})
    result.decision          # 'High risk' | 'Low risk' | 'Uncertain'
    result.probability       # calibrated-by-construction logistic probability
    result.contributions     # why
    model.what_if(record)    # what would have to change
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .conformal import MondrianConformal, cross_conformal_fit
from .config import ALPHA, FEATURES, RANDOM_STATE, TARGET, VALID_RANGES, ZERO_MEANS_MISSING
from .counterfactual import minimal_changes
from .explain import contributions, intercept_and_logit
from .preprocessing import build_pipeline

DECISION_LABELS = {
    "high": "High risk",
    "low": "Low risk",
    "uncertain": "Uncertain",
}
DECISION_ADVICE = {
    "high": "The model is confident this profile looks like diabetic patients. Confirm with a lab test.",
    "low": "The model is confident this profile looks like non-diabetic patients.",
    "uncertain": "The model cannot separate this profile from both groups. A lab test (fasting glucose / HbA1c / OGTT) is needed.",
}


@dataclass
class Prediction:
    probability: float
    decision: str                 # 'High risk' | 'Low risk' | 'Uncertain'
    decision_key: str             # 'high' | 'low' | 'uncertain'
    prediction_set: tuple         # e.g. (0,), (1,), (0, 1) or ()
    advice: str
    contributions: pd.DataFrame
    intercept: float
    log_odds: float
    not_measured: list


class ScreeningModel:
    """Logistic regression + class-conditional conformal calibration."""

    def __init__(self, alpha: float = ALPHA, random_state: int = RANDOM_STATE, C: float = 1.0):
        self.alpha = alpha
        self.random_state = random_state
        self.C = C
        self.pipeline_ = None
        self.conformal_: MondrianConformal | None = None
        self.n_train_: int | None = None
        self.train_means_: pd.Series | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame) -> "ScreeningModel":
        X, y = df[FEATURES], df[TARGET].to_numpy()
        pipe = build_pipeline(LogisticRegression(C=self.C, max_iter=5000))
        self.conformal_ = cross_conformal_fit(pipe, X, y, alpha=self.alpha, seed=self.random_state)
        self.pipeline_ = build_pipeline(LogisticRegression(C=self.C, max_iter=5000)).fit(X, y)
        self.n_train_ = len(df)
        self.train_means_ = X.replace(0, np.nan).mean()
        return self

    def _check_fitted(self):
        if self.pipeline_ is None or self.conformal_ is None:
            raise RuntimeError("Call fit() first.")

    # ------------------------------------------------------------------ input
    @staticmethod
    def to_row(record: dict) -> pd.DataFrame:
        """Validate one patient and return a 1-row frame. 0 / None means 'not measured' where allowed."""
        unknown = set(record) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown fields: {sorted(unknown)}")
        values = {}
        for f in FEATURES:
            v = record.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                if f not in ZERO_MEANS_MISSING:
                    raise ValueError(f"{f} is required.")
                values[f] = np.nan
                continue
            v = float(v)
            if f in ZERO_MEANS_MISSING and v == 0:
                values[f] = np.nan
                continue
            lo, hi = VALID_RANGES[f]
            if not lo <= v <= hi:
                raise ValueError(f"{f}={v} is outside the plausible range [{lo}, {hi}].")
            values[f] = v
        return pd.DataFrame([values], columns=FEATURES)

    # ------------------------------------------------------------------ predict
    def predict(self, record: dict) -> Prediction:
        self._check_fitted()
        row = self.to_row(record)
        p = float(self.pipeline_.predict_proba(row)[0, 1])
        sets = self.conformal_.predict_sets([p])[0]
        key = str(self.conformal_.decisions([p])[0])
        intercept, logit = intercept_and_logit(self.pipeline_, row)
        contrib = contributions(self.pipeline_, row)
        return Prediction(
            probability=p,
            decision=DECISION_LABELS[key],
            decision_key=key,
            prediction_set=tuple(int(k) for k in np.flatnonzero(sets)),
            advice=DECISION_ADVICE[key],
            contributions=contrib,
            intercept=intercept,
            log_odds=logit,
            not_measured=[f for f, m in zip(FEATURES, contrib["was_missing"]) if m],
        )

    # ------------------------------------------------------------------ what-if
    def zone_edges(self) -> tuple[float, float]:
        """(low_below, high_above): probability below the first is 'Low risk', above the second is
        'High risk', and anything in between is 'Uncertain'."""
        self._check_fitted()
        a, b = self.conformal_.zone_edges
        return min(a, b), max(a, b)

    def what_if(self, record: dict, goal: str = "uncertain") -> dict:
        """Smallest change in Glucose / BMI / BloodPressure that reaches the goal zone.

        goal='uncertain': stop being 'High risk'.  goal='low': become 'Low risk'.
        """
        self._check_fitted()
        if goal not in ("uncertain", "low"):
            raise ValueError("goal must be 'uncertain' or 'low'.")
        low_below, high_above = self.zone_edges()
        # a hair inside the boundary so the new probability is safely in the goal zone
        target = (high_above if goal == "uncertain" else low_below) - 1e-4
        target = float(np.clip(target, 1e-4, 1 - 1e-4))
        row = self.to_row(record)
        return minimal_changes(self.pipeline_, row, target)
