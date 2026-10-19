"""Class-conditional (Mondrian) conformal prediction for a binary classifier.

Idea in plain words
-------------------
A normal classifier always answers "diabetic" or "not diabetic", even when it is
unsure. Conformal prediction instead returns a *set* of plausible labels:

    {1}      -> confident: diabetic
    {0}      -> confident: not diabetic
    {0, 1}   -> cannot tell -> "uncertain, send for a lab test"

and it comes with a guarantee: for each class separately, the true label lands
inside the returned set at least (1 - alpha) of the time (here 90 %). Doing it per
class matters because diabetic patients are the minority (35 %) and are the ones
we must not miss.

The guarantee is distribution-free (it needs only exchangeable data, no model
assumptions). Here the calibration scores come from out-of-fold predictions
("cross-conformal"), which is a standard, slightly approximate version; the
coverage is checked empirically in `evaluate_coverage`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .config import ALPHA, FEATURES, RANDOM_STATE, TARGET


class MondrianConformal:
    """Per-class conformal threshold on the score s = 1 - P(true class)."""

    def __init__(self, alpha: float = ALPHA):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.qhat_: dict[int, float] | None = None

    def fit(self, y, p1) -> "MondrianConformal":
        y = np.asarray(y).astype(int)
        p1 = np.asarray(p1, dtype=float)
        scores = np.where(y == 1, 1.0 - p1, p1)  # 1 - probability given to the true class
        self.qhat_ = {}
        for k in (0, 1):
            s = scores[y == k]
            n = len(s)
            if n == 0:
                raise ValueError(f"No calibration samples for class {k}.")
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.qhat_[k] = float(np.quantile(s, level, method="higher"))
        return self

    def predict_sets(self, p1) -> np.ndarray:
        """Boolean array (n, 2): column k is True if label k is in the prediction set."""
        if self.qhat_ is None:
            raise RuntimeError("Call fit() first.")
        p1 = np.asarray(p1, dtype=float)
        in0 = p1 <= self.qhat_[0]          # score for class 0 is 1 - P(0) = p1
        in1 = (1.0 - p1) <= self.qhat_[1]  # score for class 1 is 1 - p1
        return np.column_stack([in0, in1])

    def decisions(self, p1) -> np.ndarray:
        """Turn prediction sets into 'high', 'low' or 'uncertain'."""
        sets = self.predict_sets(p1)
        out = np.full(len(sets), "uncertain", dtype=object)
        out[sets[:, 1] & ~sets[:, 0]] = "high"
        out[sets[:, 0] & ~sets[:, 1]] = "low"
        return out

    @property
    def zone_edges(self) -> tuple[float, float]:
        """Raw boundaries (q0, 1 - q1): label 0 stays plausible up to q0, label 1 from 1 - q1 up.

        When q0 >= 1 - q1 the two ranges overlap and the overlap is the 'uncertain' zone.
        `ScreeningModel.zone_edges` returns the same two numbers sorted as (low_below, high_above)."""
        if self.qhat_ is None:
            raise RuntimeError("Call fit() first.")
        return self.qhat_[0], 1.0 - self.qhat_[1]


def cross_conformal_fit(pipeline, X, y, alpha: float = ALPHA, n_splits: int = 10, seed: int = RANDOM_STATE) -> MondrianConformal:
    """Calibrate on out-of-fold probabilities so that all rows are used for training AND calibration."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    p_oof = cross_val_predict(clone(pipeline), X, y, cv=skf, method="predict_proba")[:, 1]
    return MondrianConformal(alpha).fit(y, p_oof)


def evaluate_coverage(df: pd.DataFrame, pipeline, alpha: float = ALPHA, n_repeats: int = 100) -> pd.DataFrame:
    """Honest test of the whole procedure.

    Repeat: split 80/20, calibrate on the training part only, then score the unseen
    test part. Report per-class coverage, how often the answer is 'uncertain', and
    accuracy when the model does commit.
    """
    X, y = df[FEATURES], df[TARGET].to_numpy()
    rows = []
    for r in range(n_repeats):
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=1000 + r)
        pipe = clone(pipeline)
        conf = cross_conformal_fit(pipe, X.iloc[idx_tr], y[idx_tr], alpha=alpha, seed=r)
        pipe.fit(X.iloc[idx_tr], y[idx_tr])
        p = pipe.predict_proba(X.iloc[idx_te])[:, 1]
        yt = y[idx_te]
        sets = conf.predict_sets(p)
        dec = conf.decisions(p)
        committed = dec != "uncertain"
        pred_label = (dec == "high").astype(int)
        rows.append(
            {
                "repeat": r,
                "coverage_class0": sets[yt == 0, 0].mean(),
                "coverage_class1": sets[yt == 1, 1].mean(),
                "pct_uncertain": 100 * (~committed).mean(),
                "pct_low": 100 * (dec == "low").mean(),
                "pct_high": 100 * (dec == "high").mean(),
                "accuracy_when_committed": (pred_label[committed] == yt[committed]).mean() if committed.any() else np.nan,
                "pct_flagged_for_followup": 100 * sets[:, 1].mean(),  # set contains 'diabetic'
                "ppv_of_flagged": yt[sets[:, 1]].mean() if sets[:, 1].any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
