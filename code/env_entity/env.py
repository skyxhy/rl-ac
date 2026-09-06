from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Tuple, Any

from .modules.feature_graph_module import Graph_Module
from .modules.policy_manager import PolicyManager
from .modules.risk_manager import Risk_Manager
from .modules.data_module import DataModule
from .modules.node_manager import Node_manager
from .modules.attack_manager import Attack_Manager


class Env():
    def __init__(
        self,
        graph_feature_builder: Graph_Module,
        attack_manager: Attack_Manager,
        policy_manager: PolicyManager,
        risk_manager: Risk_Manager,
        data_module: DataModule,
        start_time: int = 29800,
        window_size: int = 1,
        max_step: int = 100,
        alpha: float = 0.5,
        action_dim: int = 2,
        include_trust_in_state: bool = False,
        sight: int = 5,
        beta: float = 2  # 规模敏感系数 Φ 的增益因子
    ):
        self.graph_feature_builder = graph_feature_builder
        self.attack_manager = attack_manager
        self.policy_manager = policy_manager
        self.risk_manager = risk_manager
        self.data_module = data_module
        self.include_trust_in_state = include_trust_in_state

        self.node_manager = Node_manager(
            self.data_module.df,
            window_size=sight,
            feature_dim=self.risk_manager.embed_dim
        )

        self.start_time = start_time
        self.window_size = window_size
        self.max_step = max_step

        end_time = self.start_time + self.window_size * self.max_step
        if end_time > self.data_module.max_time:
            raise ValueError("window size is too large", end_time, self.data_module.max_time)

        self.state_dim = self.risk_manager.embed_dim + (1 if self.include_trust_in_state else 0)
        self.action_dim = action_dim
        self.level_weight = self.level_weight_cal(self.action_dim)

        self.alpha = alpha
        self.beta = beta  # 规模惩罚放大系数
        
        # 新增：3窗口滚动缓存 & N_ref下限
        self.window_history = deque(maxlen=3)
        self.n_ref_floor = 5.0  # 防止分母过小导致梯度爆炸

    def reset(self):
        self.time = self.start_time
        self.policy_manager.reset()
        self.attack_manager.reset()
        self.node_manager.reset()
        self.steps = 0
        self.window_history.clear()  # 清空历史

        attack_intensity, sender_intensity_map = self.attack_manager.attack_intensity_cal()
        self.window_df = self.data_module.read_data(
            attack_intensity=attack_intensity,
            sender_intensity_map=sender_intensity_map,
            start_time=self.time,
            end_time=self.time + self.window_size
        )

        graph = self.graph_feature_builder.build_snapshot(self.window_df)
        risk_embed = self.risk_manager.embed(graph)

        # 不再使用节点级 trust 记录，直接传空 dict
        self.node_manager.update_state(graph.node_map, risk_embed, trust_map={})

        state = self.node_manager.get_state(self.time, include_trust=self.include_trust_in_state)
        self.access_info = self.access_info_cal(self.window_df)
        
        # 初始化滚动窗口
        self.window_history.append(self.access_info.copy())
        
        return state

    def step(self, action: Tuple[Dict[int, int], np.ndarray]):
        self.policy_manager.batch_set(action[0], action[1])

        attack_intensity, sender_intensity_map = self.attack_manager.attack_intensity_cal()

        self.time += self.window_size
        self.window_df = self.data_module.read_data(
            attack_intensity=attack_intensity,
            sender_intensity_map=sender_intensity_map,
            start_time=self.time,
            end_time=self.time + self.window_size
        )

        graph = self.graph_feature_builder.build_snapshot(self.window_df)
        risk_embed = self.risk_manager.embed(graph)

        self.node_manager.update_state(graph.node_map, risk_embed, trust_map={})
        state = self.node_manager.get_state(self.time, include_trust=self.include_trust_in_state)

        self.access_info = self.access_info_cal(self.window_df)
        # 维护最近3个窗口的数据
        self.window_history.append(self.access_info.copy())

        # 更新攻击模型所需的观测：这里只保留全局拒绝率，不保留节点级状态
        df = self.apply_weight(self.access_info)
        self.attack_manager.update_state(df)

        # 奖励计算使用内部维护的3窗口历史
        reward = self.reward_function(state[0], self.access_info)
        metrics = self.metrics_cal(self.access_info, reward_arr=reward[1])

        self.steps += 1
        done = self.done_cal()
        return state, reward, done, metrics

    def access_info_cal(self, window_df: pd.DataFrame) -> pd.DataFrame:
        if window_df.empty:
            return window_df.copy()

        access_info = window_df.copy()
        access_info["access_level"] = (
            access_info["sender_pseudo"]
            .map(self.policy_manager.level_map)
            .fillna(self.policy_manager.init_access_level)
            .astype(int)
        )
        return access_info

    def state_cal(self, risk_embed: torch.Tensor, global_info: Dict):
        if global_info is None:
            return risk_embed.squeeze().numpy()
        global_info_tensor = torch.tensor([global_info[pseudo] for pseudo in self.policy_manager.pseudos])
        global_info_tensor = global_info_tensor.view(1, -1)
        return torch.cat([risk_embed, global_info_tensor], dim=1).squeeze().numpy()

    def done_cal(self):
        return self.steps >= self.max_step

    # =========================================================
    # reward (已重构为规模感知的3窗口帕累托奖励)
    # =========================================================
    def _pseudo_success_rate(self, pseudo_df: pd.DataFrame) -> float:
        if pseudo_df.empty:
            return 0.0
        pseudo_df = self.apply_weight(pseudo_df)
        return float(pseudo_df["w"].mean())

    def _allocate_sender_reward_to_pseudos(self, sender_df: pd.DataFrame, sender_reward: float) -> Dict[Any, float]:
        if sender_df.empty:
            return {}

        pseudo_groups = sender_df.groupby("sender_pseudo", sort=False)

        pseudo_rates = {}
        for pseudo, group in pseudo_groups:
            pseudo_rates[pseudo] = self._pseudo_success_rate(group)

        total_rate = float(sum(pseudo_rates.values()))
        pseudos = list(pseudo_rates.keys())

        if len(pseudos) == 0:
            return {}

        if total_rate <= 0:
            equal_share = float(sender_reward) / len(pseudos)
            return {pseudo: equal_share for pseudo in pseudos}

        return {
            pseudo: float(sender_reward) * (rate / total_rate)
            for pseudo, rate in pseudo_rates.items()
        }

    def reward_function(self, node_map: Dict[int, int], df: pd.DataFrame) -> Tuple[Dict[int, int], np.ndarray]:
        """
        规模感知的帕累托奖励函数：
        R_i = A_i - α * L_base_i * Φ(N_atk_i)
        其中 Φ = 1 + β * ln(1+N_atk) / ln(1+N_ref)
        A_i: 3窗口内正常流量平均通过率 (归一化)
        L_base_i: 3窗口内攻击流量平均通过率 (归一化)
        N_ref: 当前窗口正常流量规模的90%分位数
        """
        reward = np.zeros(len(node_map), dtype=np.float32)
        if not self.window_history:
            return node_map, reward

        # 1. 聚合最近3个窗口的数据
        combined_df = pd.concat(list(self.window_history), ignore_index=True)
        combined_df = self.apply_weight(combined_df)

        # 2. 计算 N_ref：当前窗口正常流量的90%分位数
        current_df = self.window_history[-1]
        normal_current = current_df[current_df["attack_type"] == 0]
        if not normal_current.empty:
            normal_counts = normal_current.groupby("sender_pseudo").size().values
            N_ref = float(np.percentile(normal_counts, 90))
        else:
            N_ref = self.n_ref_floor
        N_ref = max(N_ref, self.n_ref_floor)  # 防除零/对数极小

        # 3. 节点级奖励计算
        for pseudo, idx in node_map.items():
            p_df = combined_df[combined_df["sender_pseudo"] == pseudo]
            if p_df.empty:
                reward[idx] = 0.0
                continue

            # 可用性 A (归一化均值)
            norm_mask = p_df["attack_type"] == 0
            A = float(p_df.loc[norm_mask, "w"].mean()) if norm_mask.any() else 1.0

            # 基础泄漏率 L_base (归一化均值)
            atk_mask = p_df["attack_type"] != 0
            L_base = float(p_df.loc[atk_mask, "w"].mean()) if atk_mask.any() else 0.0

            # 攻击规模 N_atk (3窗口累计计数)
            N_atk = int(atk_mask.sum())

            # 规模调制因子 Φ
            if N_atk > 0:
                phi = 1.0 + self.beta * (np.log(1.0 + N_atk) / np.log(1.0 + N_ref))
            else:
                phi = 1.0

            # 帕累托标量化奖励
            reward[idx] = float(A - self.alpha * L_base * phi)

        return node_map, reward
    # def reward_function(self, node_map: Dict[int, int], df: pd.DataFrame) -> Tuple[Dict[int, int], np.ndarray]:
    #     """
    #     规模感知的帕累托奖励函数（已修正：可用性 A 基于全量流量）：
    #     R_i = A_i - α * L_base_i * Φ(N_atk_i)
        
    #     关键修改：
    #     - A_i 现为 3窗口内该节点所有请求的平均接受率/权重（含正常+攻击）
    #     - 接受攻击节点可提升 A，但会触发 L_base 惩罚，形成真实 Pareto 权衡
    #     """
    #     reward = np.zeros(len(node_map), dtype=np.float32)
    #     if not self.window_history:
    #         return node_map, reward

    #     # 1. 聚合最近3个窗口的数据
    #     combined_df = pd.concat(list(self.window_history), ignore_index=True)
    #     combined_df = self.apply_weight(combined_df)

    #     # 2. 计算 N_ref：当前窗口正常流量的90%分位数（规模基准，保持不变）
    #     current_df = self.window_history[-1]
    #     normal_current = current_df[current_df["attack_type"] == 0]
    #     if not normal_current.empty:
    #         normal_counts = normal_current.groupby("sender_pseudo").size().values
    #         N_ref = float(np.percentile(normal_counts, 90))
    #     else:
    #         N_ref = self.n_ref_floor
    #     N_ref = max(N_ref, self.n_ref_floor)

    #     # 3. 节点级奖励计算
    #     for pseudo, idx in node_map.items():
    #         p_df = combined_df[combined_df["sender_pseudo"] == pseudo]
    #         if p_df.empty:
    #             reward[idx] = 0.0
    #             continue

    #         # ✅ 【核心修正】可用性 A：全量请求的平均权重/接受率
    #         # 之前仅统计正常流量 → 攻击节点无奖励 → RL 直接拒绝所有攻击 → Pareto 坍缩
    #         # 现在全量统计 → 接受攻击可提升 A，但会触发泄漏惩罚 → 形成真实权衡
    #         A = float(p_df["w"].mean()) if not p_df.empty else 1.0

    #         # 基础泄漏率 L_base：仅针对攻击流量（保持不变）
    #         atk_mask = p_df["attack_type"] != 0
    #         L_base = float(p_df.loc[atk_mask, "w"].mean()) if atk_mask.any() else 0.0

    #         # 攻击规模 N_atk (3窗口累计攻击请求数)
    #         N_atk = int(atk_mask.sum())

    #         # 规模调制因子 Φ
    #         phi = 1.0 + self.beta * (np.log(1.0 + N_atk) / np.log(1.0 + N_ref)) if N_atk > 0 else 1.0

    #         # 帕累托标量化奖励
    #         reward[idx] = float(A - self.alpha * L_base * phi)

    #     return node_map, reward


    # =========================================================
    # metrics
    # =========================================================
    def _attack_state_code(self) -> int:
        """
        攻击者状态硬编码：
            latent    -> 0
            approach  -> 1
            burst     -> 2
            cooldown  -> 3
            fixed     -> 4
            unknown   -> -1
        """
        mode = getattr(self.attack_manager, "current_mode", None)
        if mode is None:
            mode = getattr(self.attack_manager, "attack_mode", None)

        mapping = {
            "latent": 0,
            "approach": 1,
            "burst": 2,
            "cooldown": 3,
            "fixed": 4,
        }
        return int(mapping.get(mode, -1))

    def metrics_cal(self, df: pd.DataFrame, reward_arr: np.ndarray | None = None) -> dict:
        avg_intensity = float(self.attack_manager.current_intensity)
        avg_rejection_rate = (
            float(np.mean(self.attack_manager.rejection_history))
            if len(self.attack_manager.rejection_history) > 0
            else 0.0
        )
        total_reward = float(np.sum(reward_arr)) if reward_arr is not None else 0.0

        normal_avg_success = 1.0
        attack_ratios = []
        attack_ratio_pos_sum = 0.0

        # ===== 新增指标容器 =====
        attack_success_rates = []        # 每个攻击节点的平均成功率
        attack_success_counts = []       # ✅ 每个攻击节点的成功总量

        if not df.empty:
            df_w = self.apply_weight(df)

            # =========================
            # 1. 正常节点平均成功量
            # =========================
            normal_df = df_w[df_w["attack_type"] == 0]
            if not normal_df.empty:
                normal_success_per_sender = normal_df.groupby("sender_id")["w"].sum()
                if len(normal_success_per_sender) > 0:
                    normal_avg_success = float(normal_success_per_sender.mean())

            # =========================
            # 2. 攻击节点统计
            # =========================
            attack_df = df_w[df_w["attack_type"] != 0]
            if not attack_df.empty:

                # --- 原有：risk ratio ---
                if normal_avg_success > 1e-9:
                    attack_success_per_sender = attack_df.groupby("sender_id")["w"].sum()
                    attack_ratios = (attack_success_per_sender / normal_avg_success - 1).tolist()
                    attack_ratio_pos_sum = float(sum(r for r in attack_ratios if r > 0))

                # --- 新增1：攻击成功率 ---
                attacker_success_rate_series = attack_df.groupby("sender_id")["w"].mean()
                if not attacker_success_rate_series.empty:
                    attack_success_rates = attacker_success_rate_series.tolist()

                # --- 新增2：攻击成功总量（关键）---
                attacker_success_count_series = attack_df.groupby("sender_id")["w"].sum()
                if not attacker_success_count_series.empty:
                    attack_success_counts = attacker_success_count_series.tolist()

        # =========================
        # 聚合统计
        # =========================
        avg_attack_success_rate = float(np.mean(attack_success_rates)) if attack_success_rates else 0.0

        # ✅ 新指标
        avg_attack_success_count = float(np.mean(attack_success_counts)) if attack_success_counts else 0.0
        max_attack_success_count = float(np.max(attack_success_counts)) if attack_success_counts else 0.0

        return {
            "utility": self.utility(df),

            "avg_leakage_risk_ratio": float(np.mean(attack_ratios)) if attack_ratios else 0.0,
            "max_leakage_risk_ratio": float(np.max(attack_ratios)) if attack_ratios else 0.0,
            "leakage_risk_ratio_sum": float(attack_ratio_pos_sum),

            "avg_intensity": avg_intensity,
            "avg_rejection_rate": avg_rejection_rate,
            "total_reward": total_reward,
            "attack_state_code": self._attack_state_code(),

            # ===== 原有 =====
            "avg_attack_success_rate_per_sender": avg_attack_success_rate,

            # ===== 新增（重点）=====
            "avg_attack_success_count_per_sender": avg_attack_success_count,
            "max_attack_success_count_per_sender": max_attack_success_count,
        }
    # =========================================================
    # utils
    # =========================================================
    def level_weight_cal(self, action_dim: int) -> list:
        if action_dim == 2:
            return [1.0, 0.0]
        elif action_dim == 3:
            return [1.0, 0.5, 0.0]
        return [1.0] * self.action_dim

    def apply_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            w=df["access_level"].map(lambda x: self.level_weight[x] if x < len(self.level_weight) else 0.0)
        )

    def utility(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0

        normal_df = df[df["attack_type"] == 0]
        if len(normal_df) == 0:
            return 0.0

        normal_df = self.apply_weight(normal_df)
        return float(normal_df["w"].mean())

    def leakage(self, df: pd.DataFrame, mode="sum", alpha=0.2) -> float:
        """
        泄漏量：攻击成功次数（按 sender 汇总）
        注：此方法保留供外部或 Reward 调用，不再用于 metrics_cal
        """
        if df.empty:
            return 0.0

        df = self.apply_weight(df)
        attack_df = df[df["attack_type"] != 0]

        if len(attack_df) == 0:
            return 0.0

        attack_success = attack_df.groupby("sender_id")["w"].sum()
        values = attack_success.values

        if len(values) == 0:
            return 0.0

        if mode == "mean":
            return float(values.mean())
        elif mode == "max":
            return float(values.max())
        elif mode == "sum":
            return float(values.sum())
        elif mode == "cvar":
            k = max(1, int(len(values) * alpha))
            topk = np.sort(values)[-k:]
            return float(np.mean(topk))
        else:
            raise ValueError("Unknown leakage mode")

    def leakage_entity_cal(self, sender_id, df: pd.DataFrame) -> float:
        sender_df = df[df["sender_id"] == sender_id]
        if sender_df.empty:
            return 0.0

        sender_df = self.apply_weight(sender_df)
        sender_df = sender_df[sender_df["attack_type"] != 0]
        if sender_df.empty:
            return 0.0

        return float(sender_df["w"].sum())

    def utility_entity_cal(self, sender_id, df: pd.DataFrame) -> float:
        sender_df = df[df["sender_id"] == sender_id]
        if sender_df.empty:
            return 0.0

        if sender_df["attack_type"].sum() > 0:
            return 0.0

        sender_df = self.apply_weight(sender_df)
        return float(sender_df["w"].sum())