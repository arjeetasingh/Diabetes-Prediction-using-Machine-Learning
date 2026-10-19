import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from diabetes_ml.data import load_pima  # noqa: E402
from diabetes_ml.screening import ScreeningModel  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return load_pima()


@pytest.fixture(scope="session")
def model(df):
    return ScreeningModel().fit(df)
