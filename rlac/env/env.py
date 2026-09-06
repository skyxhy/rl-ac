"""Windowed replay environment for the adaptive access-control game.

Ported from ``code/env_entity/env.py``. Optimisations:
* ``apply_weight`` uses ``np.take`` (no per-row Python lambda).
* ``reward_function`` aggregates with two group-bys instead of per-node scans.
Interface is unchanged for agents: ``state = (node_map: dict, X: np.ndarray)``
and ``reward = (node_map, np.ndarray)``.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from rlac.env.attacker import AttackManager
from rlac.env.data_module import DataModule
from rlac.env.graph_module import GraphModule
from rlac.env.node_manager import NodeManager
from rlac.env.policy import PolicyManager
from rlac.env.risk import RiskManager


class Env:
    def __init__(
        self,
        graph_feature_builder: GraphModule,
        attack_manager: AttackManager,
        policy_manager: PolicyManager,
        risk_manager: RiskManager,
        data_module: DataModule,
        start_time: int = 29800,
        window_size: int = 1,
        max_step: int = 100,
        alpha: float = 0.5,
        action_dim: int = 2,
        include_trust_in_state: bool = False,
        sight: int = 5,
        beta: float = 2,
    ):
        self.graph_feature_builder = graph_feature_builder
        self.attack_manager = attack_manager
        self.policy_manager = policy_manager
        self.risk_manager = risk_manager
        self.data_module = data_module
        self.include_trust_in_state = include_trust_in_state
        self.node_manager = NodeManager(
            data_module.df, window_size=sight, feature_dim=risk_manager.embed_dim
        )
        self.start_time = start_time
        self.window_size = window_size
        self.max_step = max_step

        end_time = self.start_time + self.window_size * self.max_step
        if end_time > data_module.max_time:
            raise ValueError("window size too large", end_time, data_module.max_time)

        self.action_dim = action_dim
        self.level_weight = self.level_weight_cal(action_dim)
        self._weight_arr = np.asarray(self.level_weight, dtype=np.float64)
        self.state_dim = risk_manager.embed_dim + (1 if include_trust_in_state else 0)
        self.alpha = alpha
        self.beta = beta
        self.window_history: deque = deque(maxlen=3)
        self.n_ref_floor = 5.0
        self.time = start_time
        self.steps = 0

    # ------------------------------------------------------------ lifecycle
    def reset(self):
        self.time = self.start_time
        self.policy_manager.reset()
        self.attack_manager.reset()
        self.node_manager.reset()
        self.steps = 0
        self.window_history.clear()
        intensity, sender_map = self.attack_manager.attack_intensity_cal()
        self.window_df = self.data_module.read_data(
            attack_intensity=intensity, sender_intensity_map=sender_map,
            start_time=self.time, end_time=self.time + self.window_size,
        )
        graph = self.graph_feature_builder.build_snapshot(self.window_df)
        state = self._observe(graph)
        self.access_info = self.access_info_cal(self.window_df)
        self.window_history.append(self.access_info.copy())
        return state

    def step(self, action: Tuple[Dict[int, int], np.ndarray]):
        self.policy_manager.batch_set(action[0], action[1])
        intensity, sender_map = self.attack_manager.attack_intensity_cal()
        self.time += self.window_size
        self.window_df = self.data_module.read_data(
            attack_intensity=intensity, sender_intensity_map=sender_map,
            start_time=self.time, end_time=self.time + self.window_size,
        )
        graph = self.graph_feature_builder.build_snapshot(self.window_df)
        state = self._observe(graph)
        self.access_info = self.access_info_cal(self.window_df)
        self.window_history.append(self.access_info.copy())

        df = self.apply_weight(self.access_info)
        self.attack_manager.update_state(df)
        reward = self.reward_function(state[0], self.access_info)
        metrics = self.metrics_cal(self.access_info, reward_arr=reward[1])
        self.steps += 1
        done = self.steps >= self.max_step
        return state, reward, done, metrics

    def _observe(self, graph):
        if graph is None or graph.feature is None or graph.feature.shape[0] == 0:
            self.node_manager.update_state({}, np.zeros((0, self.risk_manager.embed_dim), dtype=np.float32), {})
            return {}, None
        embed = self.risk_manager.embed(graph)
        self.node_manager.update_state(graph.node_map, embed, trust_map={})
        return self.node_manager.get_state(self.time, include_trust=self.include_trust_in_state)

    # ------------------------------------------------------------ access info
    def access_info_cal(self, window_df: pd.DataFrame) -> pd.DataFrame:
        if window_df is None or window_df.empty:
            return window_df.copy() if window_df is not None else pd.DataFrame()
        out = window_df.copy()
        out["access_level"] = (
            out["sender_pseudo"].map(self.policy_manager.level_map)
            .fillna(self.policy_manager.init_access_level)
            .astype(int)
        )
        return out

    def level_weight_cal(self, action_dim: int):
        if action_dim == 2:
            return [1.0, 0.0]
        if action_dim == 3:
            return [1.0, 0.5, 0.0]
        return [1.0] * action_dim

    def apply_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "access_level" not in df.columns:
            return df
        lvl = np.clip(df["access_level"].to_numpy(), 0, len(self.level_weight) - 1).astype(np.int64)
        return df.assign(w=self._weight_arr[lvl])

    # ------------------------------------------------------------ reward
    def reward_function(self, node_map: Dict[int, int], df: pd.DataFrame) -> Tuple[Dict[int, int], np.ndarray]:
        reward = np.zeros(len(node_map), dtype=np.float32)
        if not self.window_history:
            return node_map, reward

        combined = self.apply_weight(pd.concat(list(self.window_history), ignore_index=True))
        current = self.window_history[-1]
        normal_current = current[current["attack_type"] == 0]
        if not normal_current.empty:
            N_ref = float(np.percentile(normal_current.groupby("sender_pseudo").size().values, 90))
        else:
            N_ref = self.n_ref_floor
        N_ref = max(N_ref, self.n_ref_floor)

        normal = combined[combined["attack_type"] == 0].groupby("sender_pseudo")["w"].mean()
        atk = combined[combined["attack_type"] != 0].groupby("sender_pseudo")["w"]
        present = set(normal.index) | set(atk.groups.keys())

        for pseudo, idx in node_map.items():
            if pseudo not in present:
                reward[idx] = 0.0  # matches legacy: no rows in 3-window history
                continue
            A = float(normal.get(pseudo, 1.0))
            if pseudo in atk.groups:
                g = atk.get_group(pseudo)
                L = float(g.mean())
                N_atk = int(len(g))
                phi = 1.0 + self.beta * (np.log(1.0 + N_atk) / np.log(1.0 + N_ref))
            else:
                L, phi = 0.0, 1.0
            reward[idx] = float(A - self.alpha * L * phi)
        return node_map, reward

    # ------------------------------------------------------------ metrics
    def _attack_state_code(self) -> int:
        mapping = {"latent": 0, "approach": 1, "burst": 2, "cooldown": 3, "fixed": 4}
        return int(mapping.get(getattr(self.attack_manager, "current_mode", None), -1))

    def metrics_cal(self, df: pd.DataFrame, reward_arr: np.ndarray | None = None) -> dict:
        avg_intensity = float(self.attack_manager.current_intensity)
        avg_rej = float(np.mean(self.attack_manager.rejection_history)) if self.attack_manager.rejection_history else 0.0
        total_reward = float(np.sum(reward_arr)) if reward_arr is not None else 0.0

        normal_avg_success = 1.0
        attack_ratios = []
        attack_ratio_pos_sum = 0.0
        attack_success_rates = []
        attack_success_counts = []

        if df is not None and not df.empty:
            w = self.apply_weight(df)
            normal_df = w[w["attack_type"] == 0]
            if not normal_df.empty:
                per = normal_df.groupby("sender_id")["w"].sum()
                if len(per):
                    normal_avg_success = float(per.mean())
            atk_df = w[w["attack_type"] != 0]
            if not atk_df.empty:
                if normal_avg_success > 1e-9:
                    per = atk_df.groupby("sender_id")["w"].sum()
                    ratios = (per / normal_avg_success - 1).tolist()
                    attack_ratio_pos_sum = float(sum(r for r in ratios if r > 0))
                    attack_ratios = ratios
                attack_success_rates = atk_df.groupby("sender_id")["w"].mean().tolist()
                attack_success_counts = atk_df.groupby("sender_id")["w"].sum().tolist()

        return {
            "utility": self.utility(df),
            "avg_leakage_risk_ratio": float(np.mean(attack_ratios)) if attack_ratios else 0.0,
            "max_leakage_risk_ratio": float(np.max(attack_ratios)) if attack_ratios else 0.0,
            "leakage_risk_ratio_sum": float(attack_ratio_pos_sum),
            "avg_intensity": avg_intensity,
            "avg_rejection_rate": avg_rej,
            "total_reward": total_reward,
            "attack_state_code": self._attack_state_code(),
            "avg_attack_success_rate_per_sender": float(np.mean(attack_success_rates)) if attack_success_rates else 0.0,
            "avg_attack_success_count_per_sender": float(np.mean(attack_success_counts)) if attack_success_counts else 0.0,
            "max_attack_success_count_per_sender": float(np.max(attack_success_counts)) if attack_success_counts else 0.0,
        }

    def utility(self, df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        normal_df = df[df["attack_type"] == 0]
        if len(normal_df) == 0:
            return 0.0
        return float(self.apply_weight(normal_df)["w"].mean())
