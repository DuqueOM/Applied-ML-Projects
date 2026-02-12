"""Integration tests for Gaming Market Intelligence preprocessing module.

Tests the actual data/preprocess.py functions with synthetic data
matching the expected schema (name, platform, year_of_release, genre, sales, scores, rating).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data.preprocess import (
    PreprocessConfig,
    build_preprocessor,
    load_raw_dataset,
    make_features_and_target,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Create a realistic raw games DataFrame."""
    return pd.DataFrame(
        {
            "name": ["Game A", "Game B", "Game C", "Game D", "Game E", "Game F"],
            "platform": ["PS4", "XOne", "PC", "PS4", "3DS", "PC"],
            "year_of_release": [2015, 2014, 2016, 2015, 2013, 2016],
            "genre": ["Action", "Shooter", "RPG", "Sports", "RPG", "Action"],
            "na_sales": [3.5, 2.1, 0.5, 1.8, 0.3, 0.1],
            "eu_sales": [2.8, 1.5, 0.8, 1.2, 0.1, 0.2],
            "jp_sales": [0.3, 0.1, 0.0, 0.1, 2.5, 0.0],
            "other_sales": [1.0, 0.5, 0.2, 0.4, 0.1, 0.05],
            "critic_score": [92, 85, 88, 78, 86, 72],
            "user_score": [8.5, 7.0, 8.2, 6.5, 8.0, 5.5],
            "rating": ["M", "T", "M", "E", "E", "T"],
        }
    )


def test_load_raw_dataset_computes_total_sales(tmp_path) -> None:
    """load_raw_dataset should compute total_sales from regional sales columns."""
    csv_path = tmp_path / "games.csv"
    df = pd.DataFrame(
        {
            "Name": ["G1"],
            "Platform": ["PS4"],
            "Year_of_Release": [2015],
            "Genre": ["Action"],
            "NA_Sales": [3.0],
            "EU_Sales": [2.0],
            "JP_Sales": [1.0],
            "Other_Sales": [0.5],
            "Critic_Score": [90],
            "User_Score": [8.0],
            "Rating": ["M"],
        }
    )
    df.to_csv(csv_path, index=False)
    result = load_raw_dataset(str(csv_path))
    assert "total_sales" in result.columns
    assert result["total_sales"].iloc[0] == pytest.approx(6.5)


def test_load_raw_dataset_handles_tbd_user_score(tmp_path) -> None:
    """User scores of 'tbd' should be converted to NaN."""
    csv_path = tmp_path / "games.csv"
    df = pd.DataFrame(
        {
            "Name": ["G1", "G2"],
            "Platform": ["PS4", "PC"],
            "Year_of_Release": [2015, 2016],
            "Genre": ["Action", "RPG"],
            "NA_Sales": [1.0, 0.5],
            "EU_Sales": [0.5, 0.3],
            "JP_Sales": [0.1, 0.0],
            "Other_Sales": [0.1, 0.05],
            "Critic_Score": [85, 80],
            "User_Score": ["8.5", "tbd"],
            "Rating": ["M", "T"],
        }
    )
    df.to_csv(csv_path, index=False)
    result = load_raw_dataset(str(csv_path))
    assert result["user_score"].iloc[0] == pytest.approx(8.5)
    assert pd.isna(result["user_score"].iloc[1])


def test_make_features_and_target_binary_threshold(raw_df) -> None:
    """Target should be 1 if total_sales >= threshold, 0 otherwise."""
    raw_df["total_sales"] = raw_df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum(axis=1)
    config = PreprocessConfig(target_threshold_million=2.0)
    X, y = make_features_and_target(raw_df, config=config)
    # Game A: 7.6M -> 1, Game F: 0.35M -> 0
    assert y.iloc[0] == 1  # Game A
    assert y.iloc[5] == 0  # Game F


def test_make_features_and_target_excludes_sales_columns(raw_df) -> None:
    """Sales columns should be excluded from features to prevent leakage."""
    raw_df["total_sales"] = raw_df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum(axis=1)
    config = PreprocessConfig()
    X, y = make_features_and_target(raw_df, config=config)
    sales_cols = {"na_sales", "eu_sales", "jp_sales", "other_sales", "total_sales"}
    assert sales_cols.isdisjoint(set(X.columns))


def test_make_features_and_target_shapes(raw_df) -> None:
    """X and y should have matching lengths."""
    raw_df["total_sales"] = raw_df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum(axis=1)
    config = PreprocessConfig()
    X, y = make_features_and_target(raw_df, config=config)
    assert len(X) == len(y) == len(raw_df)


def test_build_preprocessor_fits(raw_df) -> None:
    """Preprocessor should fit without errors on valid data."""
    raw_df["total_sales"] = raw_df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum(axis=1)
    config = PreprocessConfig()
    X, _ = make_features_and_target(raw_df, config=config)
    preprocessor = build_preprocessor(X, config=config)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_build_preprocessor_handles_missing_values() -> None:
    """Preprocessor should handle NaN values via imputation."""
    df = pd.DataFrame(
        {
            "platform": ["PS4", "PC", None],
            "year_of_release": [2015, None, 2016],
            "genre": ["Action", "RPG", "Action"],
            "critic_score": [90.0, np.nan, 85.0],
            "user_score": [8.5, 7.0, np.nan],
            "rating": ["M", "T", "E"],
        }
    )
    config = PreprocessConfig()
    preprocessor = build_preprocessor(df, config=config)
    transformed = preprocessor.fit_transform(df)
    assert not np.isnan(transformed).any()


def test_preprocess_config_defaults() -> None:
    """PreprocessConfig defaults should be sensible."""
    config = PreprocessConfig()
    assert config.numeric_imputer_strategy == "median"
    assert config.categorical_imputer_strategy == "most_frequent"
    assert config.scale_numeric is True
    assert config.target_threshold_million == 1.0
