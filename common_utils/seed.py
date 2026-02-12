"""Reproducibility utilities shared across all ML projects.

Usage:
    from common_utils.seed import set_seed

    seed = set_seed()        # Uses SEED env var or defaults to 42
    seed = set_seed(123)     # Explicit override
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int | None = None) -> int:
    """Set random seed for reproducibility across all libraries.

    Priority: explicit arg > SEED env var > default (42).

    Args:
        seed: Optional explicit seed value.

    Returns:
        The seed that was actually used.
    """
    if seed is None:
        seed = int(os.environ.get("SEED", 42))

    random.seed(seed)
    np.random.seed(seed)

    # Optional: set torch seed if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
