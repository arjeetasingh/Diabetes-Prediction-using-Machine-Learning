import pytest

from diabetes_ml.screening import ScreeningModel

GOOD = {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 25, "Insulin": 80, "BMI": 30.0, "DiabetesPedigreeFunction": 0.4, "Age": 35}


def test_predict_basic(model):
    p = model.predict(GOOD)
    assert 0 <= p.probability <= 1
    assert p.decision in {"High risk", "Low risk", "Uncertain"}
    assert set(p.prediction_set) <= {0, 1}
    assert len(p.contributions) == 8


def test_zero_means_not_measured(model):
    p = model.predict({**GOOD, "Insulin": 0, "SkinThickness": None})
    assert set(p.not_measured) == {"Insulin", "SkinThickness"}


def test_validation_errors(model):
    with pytest.raises(ValueError):
        model.predict({**GOOD, "Glucose": 900})
    with pytest.raises(ValueError):
        model.predict({**GOOD, "Age": None})
    with pytest.raises(ValueError):
        model.predict({**GOOD, "Shoe size": 40})


def test_monotone_in_glucose(model):
    lo = model.predict({**GOOD, "Glucose": 90}).probability
    hi = model.predict({**GOOD, "Glucose": 180}).probability
    assert hi > lo


def test_zone_edges_sorted_and_consistent(model):
    low_below, high_above = model.zone_edges()
    assert 0 < low_below <= high_above < 1
    assert model.predict({**GOOD, "Glucose": 70, "BMI": 20}).decision == "Low risk"
    assert model.predict({**GOOD, "Glucose": 199, "BMI": 45, "Age": 55, "Pregnancies": 8}).decision == "High risk"


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        ScreeningModel().predict(GOOD)
