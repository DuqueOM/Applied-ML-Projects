"""Tests for common_utils.seed module."""

from __future__ import annotations

import os

import numpy as np
import pytest
from common_utils.seed import set_seed


def test_set_seed_returns_default():
    """Default seed should be 42 when no arg and no env var."""
    env_backup = os.environ.pop("SEED", None)
    try:
        result = set_seed()
        assert result == 42
    finally:
        if env_backup is not None:
            os.environ["SEED"] = env_backup


def test_set_seed_explicit_override():
    """Explicit seed arg should override everything."""
    result = set_seed(123)
    assert result == 123


def test_set_seed_from_env(monkeypatch):
    """SEED env var should be used when no explicit arg."""
    monkeypatch.setenv("SEED", "99")
    result = set_seed()
    assert result == 99


def test_set_seed_explicit_overrides_env(monkeypatch):
    """Explicit arg should take priority over env var."""
    monkeypatch.setenv("SEED", "99")
    result = set_seed(7)
    assert result == 7


def test_set_seed_reproducibility():
    """Same seed should produce same random numbers."""
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_different_seeds_differ():
    """Different seeds should produce different random numbers."""
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 42, 12345, 999999])
def test_set_seed_accepts_various_values(seed):
    """set_seed should accept various integer values without error."""
    result = set_seed(seed)
    assert result == seed
