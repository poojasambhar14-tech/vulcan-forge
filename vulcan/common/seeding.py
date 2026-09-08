"""Centralized seeding utilities for reproducibility."""
from __future__ import annotations

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_rng(seed: int) -> np.random.Generator:
    """Return an isolated numpy Generator (does not touch global state)."""
    return np.random.default_rng(seed)
