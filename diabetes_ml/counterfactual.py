"""'What would have to change?' for a logistic-regression screening model.

Given a patient and a target risk, find the SMALLEST change (measured in standard
deviations, so BMI and glucose are on a comparable footing) to a few adjustable
measurements that brings the predicted probability down to the target.

Because the model is linear in the scaled features this has an exact answer:
minimise sum(dz_j^2) subject to sum(w_j * dz_j) <= -needed_drop, with each dz_j
between its lower limit and 0. The solution is dz_j = -lambda * w_j, clipped to the
limits ("water-filling"), and lambda is found by bisection.

IMPORTANT: this describes the MODEL, which learned associations. It is not proof
that changing a value would change a real person's health.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, MODIFIABLE_FLOORS


def _floor_1dp(x: float) -> float:
    return float(np.floor(x * 10.0 + 1e-9) / 10.0)


def minimal_changes(pipeline, row: pd.DataFrame, target_p1: float, floors: dict[str, float] | None = None) -> dict:
    """Smallest decrease in adjustable features that brings P(diabetes) to <= target_p1."""
    floors = MODIFIABLE_FLOORS if floors is None else floors
    if not 0 < target_p1 < 1:
        raise ValueError("target_p1 must be between 0 and 1.")
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]

    measured_row = pipeline.named_steps["missing"].transform(row)[0]
    used = pipeline[:2].transform(row)[0]
    w = model.coef_[0]
    sd = scaler.scale_
    p_before = float(pipeline.predict_proba(row)[0, 1])
    logit_before = float(pipeline.decision_function(row)[0])
    need = logit_before - np.log(target_p1 / (1 - target_p1)) + 1e-6

    result = {
        "target": float(target_p1),
        "p_before": p_before,
        "p_after": p_before,
        "feasible": p_before <= target_p1,
        "already_met": p_before <= target_p1,
        "changes": [],
    }
    if result["already_met"]:
        return result

    # Candidates: adjustable, actually measured, lowering helps (positive weight), room above floor.
    cand = []
    for name, floor in floors.items():
        j = FEATURES.index(name)
        if np.isnan(measured_row[j]) or w[j] <= 0 or used[j] <= floor:
            continue
        lo = (floor - used[j]) / sd[j]  # most negative allowed change in z units
        cand.append((j, name, float(w[j]), float(lo)))
    if not cand:
        result["feasible"] = False
        return result

    def drop(lam: float) -> float:
        return sum(wj * min(lam * wj, -lo) for _, _, wj, lo in cand)

    max_drop = sum(wj * (-lo) for _, _, wj, lo in cand)
    if max_drop < need:
        lam = np.inf  # everything goes to its floor, still not enough
        result["feasible"] = False
    else:
        a, b = 0.0, 1.0
        while drop(b) < need:
            b *= 2.0
        for _ in range(100):
            mid = (a + b) / 2.0
            if drop(mid) < need:
                a = mid
            else:
                b = mid
        lam = b
        result["feasible"] = True

    new_row = row.copy()
    changes = []
    for j, name, wj, lo in cand:
        dz = max(-lam * wj, lo) if np.isfinite(lam) else lo
        new_val = used[j] + dz * sd[j]
        new_val = max(floors[name], _floor_1dp(new_val))  # round DOWN: never raises risk
        if new_val < used[j]:
            new_row[name] = new_val
            changes.append({"feature": name, "from": float(used[j]), "to": float(new_val), "change": float(new_val - used[j])})
    result["changes"] = changes
    result["p_after"] = float(pipeline.predict_proba(new_row)[0, 1])
    result["feasible"] = result["p_after"] <= target_p1 + 1e-9
    return result


def changes_table(result: dict) -> pd.DataFrame:
    return pd.DataFrame(result["changes"], columns=["feature", "from", "to", "change"])
