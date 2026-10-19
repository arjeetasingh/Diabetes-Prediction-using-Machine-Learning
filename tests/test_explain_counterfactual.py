import numpy as np
import pytest

from diabetes_ml.config import FEATURES, MODIFIABLE_FLOORS
from diabetes_ml.counterfactual import minimal_changes
from diabetes_ml.explain import contributions, intercept_and_logit, odds_ratios


def test_contributions_add_up_exactly(model, df):
    for i in (0, 5, 100, 500):
        row = model.to_row(df.iloc[i][FEATURES].to_dict())
        intercept, logit = intercept_and_logit(model.pipeline_, row)
        total = intercept + contributions(model.pipeline_, row)["contribution"].sum()
        assert total == pytest.approx(logit, abs=1e-9)


def test_odds_ratios_glucose_first_and_ordered_ci(df):
    t = odds_ratios(df, n_boot=60)
    assert t.iloc[0]["feature"] == "Glucose"
    assert (t["ci_low"] < t["ci_high"]).all()
    # the point estimate should sit inside its own bootstrap interval
    assert ((t["ci_low"] <= t["odds_ratio_per_sd"]) & (t["odds_ratio_per_sd"] <= t["ci_high"])).all()


def test_what_if_reaches_target_and_never_raises_risk(model, df):
    checked = 0
    for i in range(0, 768, 7):
        rec = df.iloc[i][FEATURES].to_dict()
        for goal in ("uncertain", "low"):
            res = model.what_if(rec, goal)
            assert res["p_after"] <= res["p_before"] + 1e-12
            if res["feasible"] and not res["already_met"]:
                assert res["p_after"] <= res["target"] + 1e-9
            for ch in res["changes"]:
                assert ch["to"] < ch["from"]                       # only decreases
                assert ch["to"] >= MODIFIABLE_FLOORS[ch["feature"]] - 1e-9  # never below the healthy floor
            checked += 1
    assert checked > 100


def test_already_met_returns_no_changes(model):
    low_risk = {"Pregnancies": 0, "Glucose": 80, "BloodPressure": 66, "SkinThickness": 20, "Insulin": 60, "BMI": 22.0, "DiabetesPedigreeFunction": 0.2, "Age": 22}
    res = model.what_if(low_risk, "uncertain")
    assert res["already_met"] and res["changes"] == []


def test_infeasible_target_is_reported(model, df):
    rec = df.iloc[0][FEATURES].to_dict()
    res = minimal_changes(model.pipeline_, model.to_row(rec), target_p1=0.001)
    assert res["feasible"] is False
    assert res["p_after"] < res["p_before"]  # still moves as far as it can


def test_unmeasured_feature_is_never_changed(model):
    rec = {"Pregnancies": 4, "Glucose": 170, "BloodPressure": 0, "SkinThickness": 30, "Insulin": 0, "BMI": 0, "DiabetesPedigreeFunction": 0.8, "Age": 45}
    res = model.what_if(rec, "uncertain")
    changed = {c["feature"] for c in res["changes"]}
    assert changed <= {"Glucose"}
