"""Integration tests for OilWell data preprocessing module.

Tests the actual data/preprocess.py functions with synthetic data
matching the expected schema (id, f0, f1, f2, product).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data.preprocess import clean_deduplicate_and_shuffle, compute_profit, split_features_target


@pytest.fixture
def region_df() -> pd.DataFrame:
    """Create a realistic region DataFrame."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "id": [f"well_{i:04d}" for i in range(n)],
            "f0": rng.normal(0, 1, n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "product": rng.exponential(50, n),
        }
    )


def test_clean_dedup_removes_duplicates() -> None:
    """Duplicate IDs should be deduplicated, keeping the highest product."""
    df = pd.DataFrame(
        {
            "id": ["w1", "w1", "w2", "w3", "w3"],
            "f0": [1.0, 1.0, 2.0, 3.0, 3.0],
            "f1": [0.5, 0.5, 1.0, 1.5, 1.5],
            "f2": [0.2, 0.2, 0.4, 0.6, 0.6],
            "product": [100.0, 50.0, 75.0, 200.0, 150.0],
        }
    )
    result = clean_deduplicate_and_shuffle(df, id_col="id", target_col="product")
    assert len(result) == 3
    assert result[result["id"] == "w1"]["product"].iloc[0] == 100.0
    assert result[result["id"] == "w3"]["product"].iloc[0] == 200.0


def test_clean_dedup_shuffles_deterministically() -> None:
    """Same random_state should produce same order."""
    df = pd.DataFrame(
        {
            "id": [f"w{i}" for i in range(20)],
            "f0": range(20),
            "f1": range(20),
            "f2": range(20),
            "product": range(20),
        }
    )
    r1 = clean_deduplicate_and_shuffle(df, "id", "product", random_state=7)
    r2 = clean_deduplicate_and_shuffle(df, "id", "product", random_state=7)
    pd.testing.assert_frame_equal(r1, r2)


def test_clean_dedup_raises_on_missing_columns() -> None:
    """Should raise ValueError if id or target column is missing."""
    df = pd.DataFrame({"f0": [1.0], "f1": [2.0]})
    with pytest.raises(ValueError, match="Missing columns"):
        clean_deduplicate_and_shuffle(df, "id", "product")


def test_split_features_target_correct_split(region_df: pd.DataFrame) -> None:
    """split_features_target should separate features from target correctly."""
    X, y = split_features_target(region_df, ["f0", "f1", "f2"], "product")
    assert list(X.columns) == ["f0", "f1", "f2"]
    assert len(X) == len(y) == len(region_df)


def test_split_features_target_raises_on_missing_col() -> None:
    """Should raise KeyError if a feature column is missing."""
    df = pd.DataFrame({"f0": [1.0], "product": [10.0]})
    with pytest.raises(KeyError, match="f1"):
        split_features_target(df, ["f0", "f1", "f2"], "product")


def test_compute_profit_positive() -> None:
    """Profit should be positive when revenue exceeds cost."""
    profit = compute_profit(
        predicted_units_sum=50000.0,
        revenue_per_unit=4500.0,
        total_cost=100_000_000.0,
    )
    assert profit == (50000.0 * 4500.0) - 100_000_000.0


def test_compute_profit_negative() -> None:
    """Profit should be negative when cost exceeds revenue."""
    profit = compute_profit(
        predicted_units_sum=10.0,
        revenue_per_unit=4500.0,
        total_cost=100_000_000.0,
    )
    assert profit < 0


def test_compute_profit_zero_units() -> None:
    """Zero production should yield negative profit (pure cost)."""
    profit = compute_profit(0.0, 4500.0, 100_000_000.0)
    assert profit == -100_000_000.0


@pytest.mark.parametrize("n_wells", [10, 50, 100])
def test_region_df_has_correct_size(n_wells: int) -> None:
    """Verify DataFrame construction at various sizes."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": [f"w{i}" for i in range(n_wells)],
            "f0": rng.normal(size=n_wells),
            "f1": rng.normal(size=n_wells),
            "f2": rng.normal(size=n_wells),
            "product": rng.exponential(50, n_wells),
        }
    )
    assert len(df) == n_wells
    assert set(df.columns) == {"id", "f0", "f1", "f2", "product"}
