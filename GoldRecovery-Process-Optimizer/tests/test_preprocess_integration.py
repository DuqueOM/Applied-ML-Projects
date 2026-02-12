"""Integration tests for GoldRecovery preprocessing module.

Tests the actual data/preprocess.py functions with synthetic data
matching the expected metallurgical process schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data.preprocess import (
    basic_clean,
    compute_recovery,
    create_features,
    fill_missing_with_median,
)


@pytest.fixture
def process_df() -> pd.DataFrame:
    """Create a realistic metallurgical process DataFrame."""
    n = 30
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2016-01-15", periods=n, freq="h"),
            "rougher.input.feed_au": rng.uniform(7, 9, n),
            "rougher.output.concentrate_au": rng.uniform(22, 28, n),
            "rougher.output.tail_au": rng.uniform(2.5, 3.5, n),
            "rougher.output.concentrate_ag": rng.uniform(9, 11, n),
            "primary_cleaner.output.concentrate_au": rng.uniform(36, 42, n),
            "primary_cleaner.output.concentrate_ag": rng.uniform(7, 8.5, n),
            "secondary_cleaner.output.concentrate_au": rng.uniform(44, 49, n),
            "secondary_cleaner.output.concentrate_ag": rng.uniform(4.5, 6, n),
            "rougher.output.recovery": rng.uniform(60, 72, n),
            "final.output.recovery": rng.uniform(48, 58, n),
        }
    )


def test_compute_recovery_basic() -> None:
    """Recovery formula should produce reasonable percentages."""
    feed = pd.Series([8.0, 7.5])
    concentrate = pd.Series([25.0, 24.0])
    tail = pd.Series([3.0, 3.5])
    recovery = compute_recovery(feed, concentrate, tail)
    assert len(recovery) == 2
    assert recovery.between(0, 100).all()


def test_compute_recovery_handles_zero_denominator() -> None:
    """Should return NaN when denominator is near zero."""
    feed = pd.Series([0.0])
    concentrate = pd.Series([0.0])
    tail = pd.Series([0.0])
    recovery = compute_recovery(feed, concentrate, tail)
    assert pd.isna(recovery.iloc[0])


def test_basic_clean_filters_invalid_recovery() -> None:
    """Rows with recovery outside [0, 100] should be removed."""
    df = pd.DataFrame(
        {
            "final.output.recovery": [50.0, -5.0, 110.0, 75.0],
            "feature1": [1, 2, 3, 4],
        }
    )
    cleaned = basic_clean(df)
    assert len(cleaned) == 2
    assert cleaned["final.output.recovery"].between(0, 100).all()


def test_basic_clean_sorts_by_date(process_df: pd.DataFrame) -> None:
    """Data should be sorted by date after cleaning."""
    # Shuffle first
    shuffled = process_df.sample(frac=1, random_state=0)
    cleaned = basic_clean(shuffled)
    dates = cleaned["date"].values
    assert (dates[:-1] <= dates[1:]).all()


def test_basic_clean_drops_high_null_columns() -> None:
    """Columns with >60% nulls should be dropped."""
    df = pd.DataFrame(
        {
            "good_col": [1, 2, 3, 4, 5],
            "bad_col": [np.nan, np.nan, np.nan, np.nan, 1],
            "final.output.recovery": [50, 60, 70, 80, 90],
        }
    )
    cleaned = basic_clean(df)
    assert "good_col" in cleaned.columns
    assert "bad_col" not in cleaned.columns


def test_fill_missing_with_median() -> None:
    """NaN values should be filled with column medians."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
        }
    )
    filled = fill_missing_with_median(df)
    assert not filled.isnull().any().any()
    assert filled["a"].iloc[2] == pytest.approx(3.0)  # median of [1,2,4,5]
    assert filled["b"].iloc[1] == pytest.approx(35.0)  # median of [10,30,40,50]


def test_create_features_adds_au_recovery_ratio(process_df: pd.DataFrame) -> None:
    """Should create au_recovery_ratio from rougher and cleaner concentrations."""
    featured = create_features(process_df)
    assert "au_recovery_ratio" in featured.columns
    assert featured["au_recovery_ratio"].notna().all()


def test_create_features_adds_ag_recovery_ratio(process_df: pd.DataFrame) -> None:
    """Should create ag_recovery_ratio from rougher and cleaner Ag concentrations."""
    featured = create_features(process_df)
    assert "ag_recovery_ratio" in featured.columns


def test_create_features_adds_temporal_features(process_df: pd.DataFrame) -> None:
    """Should add hour, day_of_week, month from date column."""
    featured = create_features(process_df)
    assert "hour" in featured.columns
    assert "day_of_week" in featured.columns
    assert "month" in featured.columns
    assert featured["hour"].between(0, 23).all()
    assert featured["day_of_week"].between(0, 6).all()
    assert featured["month"].between(1, 12).all()


def test_create_features_without_date() -> None:
    """Should work without date column (no temporal features added)."""
    df = pd.DataFrame(
        {
            "rougher.output.concentrate_au": [25.0],
            "primary_cleaner.output.concentrate_au": [38.0],
        }
    )
    featured = create_features(df)
    assert "hour" not in featured.columns
    assert "au_recovery_ratio" in featured.columns


def test_process_df_schema(process_df: pd.DataFrame) -> None:
    """Verify the test fixture has the expected schema."""
    required = [
        "date",
        "rougher.input.feed_au",
        "rougher.output.concentrate_au",
        "final.output.recovery",
    ]
    for col in required:
        assert col in process_df.columns
    assert len(process_df) == 30
