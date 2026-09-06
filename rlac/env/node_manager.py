"""Per-pseudonym state history.

Optimisation: the old implementation scanned every known pseudonym on every
``get_state`` call to find the active window. Active sets are static (each
pseudo has a fixed [start, end] time range), so we cache ``time -> actives``
lazily and only scan O(P) once per distinct time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class NodeInfo:
    node_id: int
    window_size: int
    feature_dim: int
    start: int
    end: int
    node_states: deque = field(default_factory=deque)
    current_trust: float = 0.5
    prev_trust: float = 0.5
    trust_history: deque = field(default_factory=deque)

    def __post_init__(self):
        self.node_states = deque(maxlen=self.window_size)
        self.trust_history = deque(maxlen=self.window_size)


class NodeManager:
    def __init__(self, df: pd.DataFrame, window_size: int = 10, feature_dim: int = 5):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.node_info: Dict[Any, NodeInfo] = {}
        self._active_cache: Dict[int, List[Any]] = {}

        if df is None or df.empty:
            return
        for pseudo, row in df.groupby("sender_pseudo")["time"].agg(["min", "max"]).iterrows():
            self.node_info[pseudo] = NodeInfo(
                node_id=pseudo,
                window_size=window_size,
                feature_dim=feature_dim,
                start=int(row["min"]),
                end=int(row["max"]),
            )

    def reset(self):
        self._active_cache.clear()
        for info in self.node_info.values():
            info.node_states.clear()
            info.trust_history.clear()
            info.current_trust = 0.5
            info.prev_trust = 0.5

    def _actives(self, time: int) -> List[Any]:
        cached = self._active_cache.get(time)
        if cached is None:
            cached = [p for p, info in self.node_info.items() if info.start <= time <= info.end]
            self._active_cache[time] = cached
        return cached

    def _as_numpy(self, x) -> Optional[np.ndarray]:
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def update_state(self, node_map, features, trust_map: Optional[Dict[Any, float]] = None):
        features = self._as_numpy(features)
        if features is None or len(features) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        for pseudo, idx in node_map.items():
            info = self.node_info.get(pseudo)
            if info is None or idx >= len(features):
                continue
            info.node_states.append(features[idx])
            if trust_map is not None and pseudo in trust_map:
                info.prev_trust = info.current_trust
                info.current_trust = float(trust_map[pseudo])
                info.trust_history.append(info.current_trust)

    def get_state(self, time: int, include_trust: bool = False) -> Tuple[Dict[int, int], Optional[np.ndarray]]:
        add = 1 if include_trust else 0
        width = self.feature_dim + add
        node_map: Dict[int, int] = {}
        res = []
        for pseudo in self._actives(time):
            info = self.node_info[pseudo]
            state = np.zeros((self.window_size, width), dtype=np.float32)
            hist = list(info.node_states)
            if hist:
                history = np.stack(hist)
                state[self.window_size - len(history):, : self.feature_dim] = history
            if include_trust:
                state[:, self.feature_dim:] = float(info.current_trust)
            node_map[pseudo] = len(res)
            res.append(state)
        if not res:
            return {}, None
        return node_map, np.stack(res)

    def get_trust_map(self) -> Dict[Any, float]:
        return {p: float(i.current_trust) for p, i in self.node_info.items()}
