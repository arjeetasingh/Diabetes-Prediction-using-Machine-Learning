import numpy as np
import pandas as pd

from diabetes_ml.config import FEATURES, TARGET
from diabetes_ml.preprocessing import HiddenMissingToNaN, build_pipeline


def test_zeros_become_nan_only_where_impossible(df):
    out = HiddenMissingToNaN().fit_transform(df[FEATURES])
    j = {f: i for i, f in enumerate(FEATURES)}
    assert np.isnan(out[:, j["Insulin"]]).sum() == 374
    assert np.isnan(out[:, j["Glucose"]]).sum() == 5
    # zero pregnancies is real and must be kept
    assert (out[:, j["Pregnancies"]] == 0).sum() == 111
    assert not np.isnan(out[:, j["Pregnancies"]]).any()


def test_insulin_cap_optional(df):
    base = HiddenMissingToNaN().fit_transform(df[FEATURES])
    capped = HiddenMissingToNaN(insulin_cap=425).fit_transform(df[FEATURES])
    j = FEATURES.index("Insulin")
    assert np.isnan(capped[:, j]).sum() > np.isnan(base[:, j]).sum()


def test_column_order_is_enforced(df):
    shuffled = df[FEATURES[::-1]]
    a = HiddenMissingToNaN().transform(df[FEATURES])
    b = HiddenMissingToNaN().transform(shuffled)
    assert np.allclose(a, b, equal_nan=True)


def test_imputation_uses_training_data_only(df):
    """The fill value must come from the training rows, never from the test rows."""
    train, test = df.iloc[:600], df.iloc[600:].copy()
    pipe = build_pipeline().fit(train[FEATURES], train[TARGET])
    train_median = np.nanmedian(HiddenMissingToNaN().transform(train[FEATURES])[:, FEATURES.index("Insulin")])
    test.loc[test.index[0], "Insulin"] = 0  # force a missing value
    filled = pipe[:2].transform(test[FEATURES])[0, FEATURES.index("Insulin")]
    assert filled == train_median


def test_pipeline_has_no_nan_after_imputation(df):
    Xt = build_pipeline().fit(df[FEATURES], df[TARGET])[:-1].transform(df[FEATURES])
    assert not np.isnan(Xt).any()
