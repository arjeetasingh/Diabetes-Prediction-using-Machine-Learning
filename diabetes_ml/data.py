"""Loading and validating the PIMA Indians Diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, FEATURES, TARGET, ZERO_MEANS_MISSING


def load_pima(path: str | Path | None = None) -> pd.DataFrame:
    """Load diabetes.csv and check that it has the expected shape and columns."""
    path = Path(path) if path is not None else DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for how to get diabetes.csv."
        )
    df = pd.read_csv(path)
    expected = FEATURES + [TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df[expected].apply(pd.to_numeric, errors="raise")
    if df.isna().any().any():
        raise ValueError("Dataset contains NaN values; the PIMA file should have none.")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Outcome column must contain only 0 and 1.")
    return df


def hidden_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count the zeros that really mean 'not measured' in each column."""
    rows = []
    for col in ZERO_MEANS_MISSING:
        n = int((df[col] == 0).sum())
        rows.append({"Feature": col, "Hidden_missing_n": n, "Hidden_missing_pct": 100 * n / len(df)})
    return pd.DataFrame(rows)
