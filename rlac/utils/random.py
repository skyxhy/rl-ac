"""Centralised RNG seeding.

The old pipeline only seeded ``numpy``/``torch`` and never ``random``, so
replay-buffer minibatch ordering differed between runs of the *same* seed
(breaking bit-level reproducibility). ``seed_all`` covers every RNG the
pipeline draws from.
"""
from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed python ``random``, ``numpy`` and ``torch`` (and cudnn if present)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch optional at import time in some tooling
        pass
