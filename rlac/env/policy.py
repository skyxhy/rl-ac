"""Access-level policy map (pseudo -> access level)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class PolicyManager:
    def __init__(self, init_access_level: int = 1):
        self.init_access_level = int(init_access_level)
        self.level_map: Dict[int, int] = {}

    def reset(self):
        self.level_map = {}

    def set(self, pseudo: int, level: int):
        self.level_map[pseudo] = int(level)

    def get(self, pseudo: int) -> int:
        return self.level_map.get(pseudo, self.init_access_level)

    def batch_set(self, pseudo_map: Dict[int, int], levels: np.ndarray):
        for pseudo, i in pseudo_map.items():
            self.set(pseudo, levels[i])

    def batch_get(self, pseudos: List[int]) -> List[int]:
        return [self.get(p) for p in pseudos]
