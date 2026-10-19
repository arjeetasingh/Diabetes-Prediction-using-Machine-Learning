import numpy as np

from diabetes_ml.baseline import forward_selection, original_clean, run_baseline


def test_original_clean_has_no_nan(df):
    d = original_clean(df)
    assert not d.isna().any().any()
    assert d["Insulin"].max() <= 425


def test_baseline_split_size_and_reproducible(df):
    a = run_baseline(df, random_state=7)
    b = run_baseline(df, random_state=7)
    assert a["confusion"].sum() == 154  # 20 % of 768, as on the slides
    assert np.array_equal(a["confusion"], b["confusion"])
    assert 0.6 < a["accuracy"] < 0.9 and 0.7 < a["auc"] < 0.95


def test_forward_selection_starts_with_glucose(df):
    fs = forward_selection(df, cv=3)
    assert fs.iloc[0]["added"] == "Glucose"
    assert len(fs) == 8


def test_l1_really_zeroes_coefficients_and_l2_does_not(df):
    """Guards against a scikit-learn API change silently turning L1 into L2."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from diabetes_ml.config import FEATURES, TARGET
    from diabetes_ml.preprocessing import penalised_logreg

    d = original_clean(df)
    l1 = make_pipeline(StandardScaler(), penalised_logreg("l1", C=0.03)).fit(d[FEATURES], d[TARGET])
    l2 = make_pipeline(StandardScaler(), penalised_logreg("l2", C=0.03)).fit(d[FEATURES], d[TARGET])
    assert (l1[-1].coef_ == 0).sum() >= 2
    assert (l2[-1].coef_ == 0).sum() == 0
