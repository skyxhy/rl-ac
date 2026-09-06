"""Replay buffers.

``ReplayBuffer`` stores 2-step delayed returns (``r_t + gamma*r_{t+1}``) keyed
per pseudonym; ``SimpleReplayBuffer`` stores single-step per-node transitions
for the no-bootstrapping contextual-bandit baseline.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=10000, n_step=2, gamma=0.85, seed: int = 0):
        assert n_step == 2, "only the 2-step delayed buffer is implemented"
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self._rng = random.Random(seed)
        self._pending: Dict[Any, Tuple[Any, Any, float, Any]] = {}

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        k = min(len(self.buffer), batch_size)
        return self._rng.sample(self.buffer, k)

    def flush(self):
        self._pending.clear()

    def _extract(self, batch, pseudo, idx):
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], dict):
            mapping, values = batch
            return values[mapping[pseudo]]
        if isinstance(batch, dict):
            return batch[pseudo]
        if np.isscalar(batch):
            return float(batch)
        arr = np.asarray(batch)
        return float(arr.item()) if arr.ndim == 0 else arr[idx]

    def batch_push(self, state, action, reward, next_state):
        node_map = state[0]
        pseudo_set = set(node_map) & set(action[0]) & set(next_state[0])
        if isinstance(reward, tuple) and len(reward) == 2 and isinstance(reward[0], dict):
            pseudo_set &= set(reward[0])
        current = {}
        for p in pseudo_set:
            i, a_i, n_i = node_map[p], action[0][p], next_state[0][p]
            current[p] = (state[1][i], action[1][a_i], float(self._extract(reward, p, i)), next_state[1][n_i])
        prev = set(self._pending)
        for p, cur in current.items():
            if p in self._pending:
                ps, pa, pr, _ = self._pending[p]
                self.buffer.append((ps, pa, pr + self.gamma * cur[2], cur[3]))
            self._pending[p] = cur
        for p in prev - set(current):
            self._pending.pop(p, None)


class SimpleReplayBuffer:
    """Per-node single-step buffer for the contextual-bandit ablation."""

    def __init__(self, capacity=20000, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward):
        if state is None:
            return
        node_map = state[0]
        r_map = reward[0] if isinstance(reward, tuple) and len(reward) == 2 and isinstance(reward[0], dict) else None
        r_vals = reward[1] if r_map is not None else None
        if r_map is None:
            # scalar reward: store for every node that acted
            for p, i in node_map.items():
                self.buffer.append((state[1][i], int(action[1][action[0].get(p, i)]),
                                    float(reward) if np.isscalar(reward) else float(np.mean(reward))))
            return
        # reward is per-node for the *next* window: only store nodes that acted
        # AND reappear in the reward map (indexed by the reward's own node order)
        for p, i in node_map.items():
            if p not in r_map:
                continue
            self.buffer.append((state[1][i], int(action[1][action[0].get(p, i)]),
                                float(r_vals[r_map[p]])))

    def sample(self, batch_size):
        k = min(len(self.buffer), batch_size)
        return self._rng.sample(self.buffer, k)
