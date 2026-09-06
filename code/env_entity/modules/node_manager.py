import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any



# =========================================================
# Node_manager
# =========================================================
@dataclass
class Node_info:
    node_id: int
    node_states: deque
    trust_history: deque
    current_trust: float
    prev_trust: float
    start: int
    end: int


class Node_manager:
    def __init__(self, df: pd.DataFrame, window_size: int = 10, feature_dim: int = 5):
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.node_info: Dict[Any, Node_info] = {}

        if df is None or df.empty:
            return

        pseudo_df = df.groupby("sender_pseudo").agg({"time": ["min", "max"]})
        for pseudo, pseudo_info in pseudo_df.iterrows():
            self.node_info[pseudo] = Node_info(
                node_id=pseudo,
                node_states=deque(maxlen=self.window_size),
                trust_history=deque(maxlen=self.window_size),
                current_trust=0.5,
                prev_trust=0.5,
                start=int(pseudo_info["time"]["min"]),
                end=int(pseudo_info["time"]["max"])
            )

    def reset(self):
        for _, node_info in self.node_info.items():
            node_info.node_states.clear()
            node_info.trust_history.clear()
            node_info.current_trust = 0.5
            node_info.prev_trust = 0.5

    def _to_numpy(self, x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def update_state(self, node_map, features, trust_map: Optional[Dict[Any, float]] = None):
        features = self._to_numpy(features)

        if features is None or len(features) == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        for pseudo in node_map:
            if pseudo not in self.node_info:
                continue

            idx = node_map[pseudo]
            if idx < len(features):
                self.node_info[pseudo].node_states.append(features[idx])

            if trust_map is not None and pseudo in trust_map:
                info = self.node_info[pseudo]
                info.prev_trust = info.current_trust
                info.current_trust = float(trust_map[pseudo])
                info.trust_history.append(info.current_trust)

    def get_state(self, time, include_trust: bool = False) -> Tuple[Dict, np.ndarray]:
        node_map = {}
        res = []

        for pseudo, node_info in self.node_info.items():
            if node_info.start <= time <= node_info.end:
                state = np.zeros(
                    (self.window_size, self.feature_dim),
                    dtype=np.float32
                )

                if len(node_info.node_states) > 0:
                    history = np.stack(list(node_info.node_states))
                    start_idx = self.window_size - len(history)
                    state[start_idx:] = history

                if include_trust:
                    trust_scalar = float(node_info.current_trust)
                    trust_channel = np.full(
                        (self.window_size, 1),
                        trust_scalar,
                        dtype=np.float32
                    )
                    state = np.concatenate([state, trust_channel], axis=1)

                res.append(state)
                node_map[pseudo] = len(res) - 1

        if len(res) == 0:
            return {}, None

        return node_map, np.stack(res)

    def get_node_info(self) -> List[Node_info]:
        return list(self.node_info.values())

    def get_trust_map(self) -> Dict[Any, float]:
        return {
            pseudo: float(info.current_trust)
            for pseudo, info in self.node_info.items()
        }
