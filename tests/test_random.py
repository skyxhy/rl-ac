"""RNG determinism: same seed -> same draw sequence across python/numpy/torch."""
import random

import numpy as np

from rlac.utils.random import seed_all


def test_seed_all_deterministic():
    seed_all(7)
    py_seq = [random.random() for _ in range(10)]
    np_seq = np.random.rand(10)
    try:
        import torch
        torch_seq = torch.rand(10).tolist()
    except Exception:
        torch_seq = None

    seed_all(7)
    assert py_seq == [random.random() for _ in range(10)]
    assert np.array_equal(np_seq, np.random.rand(10))
    if torch_seq is not None:
        import torch
        assert torch_seq == torch.rand(10).tolist()
