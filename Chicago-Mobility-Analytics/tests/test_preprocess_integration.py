"""Integration tests for the Chicago Mobility preprocessing pipeline.

These tests exercise the actual data/preprocess.py module with synthetic data
that matches the expected schema (start_ts, duration_seconds, weather_conditions).
"""

from __future__ import annotations

import pandas as pd
import pytest
from data.preprocess import engineer_features, load_raw_weather_data, setup_logging


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Create a realistic raw DataFrame matching the expected schema."""
    return pd.DataFrame(
        {
            "start_ts": [
                "2017-07-01 06:00:00",
                "2017-07-01 18:00:00",
                "2017-07-02 10:00:00",
                "2017-07-02 22:00:00",
                "2017-07-03 08:00:00",
                "2017-07-03 14:00:00",
                "2017-07-04 12:00:00",
                "2017-07-04 20:00:00",
                "2017-07-05 09:00:00",
                "2017-07-05 16:00:00",
            ],
            "duration_seconds": [420, 1320, 600, 540, 780, 900, 720, 1020, 660, 1080],
            "weather_conditions": [
                "Good",
                "Bad",
                "Good",
                "Good",
                "Bad",
                "Good",
                "Good",
                "Bad",
                "Good",
                "Good",
            ],
        }
    )


def test_engineer_features_returns_correct_columns(raw_df: pd.DataFrame) -> None:
    """engineer_features should produce hour, day_of_week, is_weekend, weather_is_bad."""
    X, y, _ = engineer_features(raw_df)
    expected_cols = {"hour", "day_of_week", "is_weekend", "weather_is_bad"}
    assert expected_cols == set(X.columns)


def test_engineer_features_target_is_positive(raw_df: pd.DataFrame) -> None:
    """All duration_seconds values should be positive after filtering."""
    _, y, _ = engineer_features(raw_df)
    assert (y > 0).all()


def test_engineer_features_shapes_match(raw_df: pd.DataFrame) -> None:
    """X and y should have the same number of rows."""
    X, y, _ = engineer_features(raw_df)
    assert len(X) == len(y)


def test_engineer_features_hour_range(raw_df: pd.DataFrame) -> None:
    """Hour feature should be between 0 and 23."""
    X, _, _ = engineer_features(raw_df)
    assert X["hour"].between(0, 23).all()


def test_engineer_features_day_of_week_range(raw_df: pd.DataFrame) -> None:
    """day_of_week feature should be between 0 and 6."""
    X, _, _ = engineer_features(raw_df)
    assert X["day_of_week"].between(0, 6).all()


def test_engineer_features_weekend_flag(raw_df: pd.DataFrame) -> None:
    """is_weekend should be 0 or 1."""
    X, _, _ = engineer_features(raw_df)
    assert set(X["is_weekend"].unique()).issubset({0, 1})


def test_engineer_features_weather_flag(raw_df: pd.DataFrame) -> None:
    """weather_is_bad should be 0 or 1."""
    X, _, _ = engineer_features(raw_df)
    assert set(X["weather_is_bad"].unique()).issubset({0, 1})


def test_engineer_features_filters_non_positive_duration() -> None:
    """Rows with duration <= 0 should be removed."""
    df = pd.DataFrame(
        {
            "start_ts": ["2017-07-01 06:00:00", "2017-07-01 07:00:00", "2017-07-01 08:00:00"],
            "duration_seconds": [420, 0, -100],
            "weather_conditions": ["Good", "Good", "Bad"],
        }
    )
    X, y, _ = engineer_features(df)
    assert len(X) == 1
    assert y.iloc[0] == 420


def test_engineer_features_bad_weather_encoding() -> None:
    """'Bad' weather should map to 1, anything else to 0."""
    df = pd.DataFrame(
        {
            "start_ts": ["2017-07-01 06:00:00", "2017-07-01 07:00:00"],
            "duration_seconds": [420, 540],
            "weather_conditions": ["Bad", "Good"],
        }
    )
    X, _, _ = engineer_features(df)
    assert X["weather_is_bad"].iloc[0] == 1
    assert X["weather_is_bad"].iloc[1] == 0


def test_load_raw_weather_data_raises_on_missing_file() -> None:
    """Should raise FileNotFoundError for nonexistent path."""
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        load_raw_weather_data(Path("/nonexistent/path.csv"))


def test_setup_logging_does_not_crash() -> None:
    """setup_logging should not raise for valid levels."""
    setup_logging("DEBUG")
    setup_logging("INFO")
    setup_logging("WARNING")


def test_engineer_features_no_nulls(raw_df: pd.DataFrame) -> None:
    """Feature matrix should have no null values."""
    X, y, _ = engineer_features(raw_df)
    assert not X.isnull().any().any()
    assert not y.isnull().any()
