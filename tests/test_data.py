import pandas as pd
import pytest

from diabetes_ml.config import FEATURES, TARGET
from diabetes_ml.data import hidden_missing_report, load_pima


def test_shape_and_balance(df):
    assert df.shape == (768, 9)
    assert list(df.columns) == FEATURES + [TARGET]
    assert int(df[TARGET].sum()) == 268  # 34.9 % positive


def test_known_hidden_zero_counts(df):
    rep = hidden_missing_report(df).set_index("Feature")["Hidden_missing_n"].to_dict()
    assert rep == {"Glucose": 5, "BloodPressure": 35, "SkinThickness": 227, "Insulin": 374, "BMI": 11}


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pima(tmp_path / "nope.csv")


def test_bad_columns_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_pima(p)
