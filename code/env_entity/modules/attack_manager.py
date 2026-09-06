from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, Any, Tuple, Optional


class Attack_Manager:
    """
    Global control attack manager.

    Phases:
        latent -> approach -> burst -> cooldown -> latent

    Key design:
        - global control only
        - fixed mode supported
        - adaptive mode uses normalized phase efficiencies + softmax
        - burst and cooldown have probabilistic transitions
        - cooldown exits deterministically when rejection rate < 0.38
        - cooldown max duration = 20 steps
    """

    MODE_NAMES = ("latent", "approach", "burst", "cooldown")

    def __init__(
        self,
        global_intensity: float = 0.15,
        min_intensity: float = 0.01,
        max_intensity: float = 1.0,
        history_len: int = 8,
        col_sender: str = "sender_id",
        col_access: str = "access_level",
        col_weight: str = "w",
        attack_mode: str = "adaptive",   # "fixed" | "adaptive"
        fixed_intensity: float = 0.5,

        # thresholds
        burst_reject_trigger: float = 0.7,
        cooldown_reject_exit: float = 0.38,
        latent_reject_split: float = 0.4,

        # timing
        cooldown_max_steps: int = 1,
        burst_soft_max_steps: int = 2,

        # intensities
        latent_intensity: Tuple[float, float] = (0.1, 0.2),
        burst_intensity: float = 0.9,
        cooldown_max_intensity: float = 0.13,
        cooldown_decay: float = 0.82,

        # probabilities
        latent_keep_prob_low: float = 0.3,
        latent_keep_prob_high: float = 0.8,
        approach_keep_prob: float = 0.9,
        cooldown_keep_prob: float = 0.3,

        # softmax
        softmax_temperature: float = 0.7,

        random_seed: Optional[int] = None,
    ):
        self.attack_mode = attack_mode
        self.fixed_intensity = float(np.clip(fixed_intensity, min_intensity, max_intensity))

        self.global_intensity = float(np.clip(global_intensity, min_intensity, max_intensity))
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

        self.history_len = history_len
        self.col_sender = col_sender
        self.col_access = col_access
        self.col_weight = col_weight

        self.burst_reject_trigger = burst_reject_trigger
        self.cooldown_reject_exit = cooldown_reject_exit
        self.latent_reject_split = latent_reject_split

        self.cooldown_max_steps = cooldown_max_steps
        self.burst_soft_max_steps = burst_soft_max_steps

        self.latent_intensity = latent_intensity
        self.burst_intensity = float(np.clip(burst_intensity, min_intensity, max_intensity))
        self.cooldown_max_intensity = cooldown_max_intensity
        self.cooldown_decay = cooldown_decay

        self.latent_keep_prob_low = latent_keep_prob_low
        self.latent_keep_prob_high = latent_keep_prob_high
        self.approach_keep_prob = approach_keep_prob
        self.cooldown_keep_prob = cooldown_keep_prob

        self.softmax_temperature = max(1e-6, float(softmax_temperature))
        self.rng = np.random.default_rng(random_seed)

        # global state
        self.observed_senders: set = set()
        self.current_mode: str = "latent"
        self.pending_mode: Optional[str] = None
        self.mode_step: int = 0

        self.current_intensity: float = self.global_intensity
        self.current_rejection_rate: float = 0.0
        self.current_trust: float = 0.5

        self.rejection_ema: float = 0.0
        self.trust_ema: float = 0.5

        self.global_step: int = 0

        # histories for stable-state estimation
        self.latent_rej_history: deque[float] = deque(maxlen=history_len)
        self.latent_intensity_history: deque[float] = deque(maxlen=history_len)

        self.approach_rej_history: deque[float] = deque(maxlen=history_len)
        self.approach_intensity_history: deque[float] = deque(maxlen=history_len)

        self.cooldown_rej_history: deque[float] = deque(maxlen=history_len)
        self.cooldown_intensity_history: deque[float] = deque(maxlen=history_len)

        self.burst_duration_hist: deque[float] = deque(maxlen=history_len)
        self.cooldown_duration_hist: deque[float] = deque(maxlen=history_len)

        # debugging / plotting
        self.rejection_history: deque[float] = deque(maxlen=history_len)
        self.pressure_history: deque[float] = deque(maxlen=history_len)

        # auxiliary rolling pairs for approach fit
        self.global_rej_intensity_hist: deque[Tuple[float, float]] = deque(maxlen=history_len)

    # =========================================================
    # reset
    # =========================================================
    def reset(self):
        self.observed_senders.clear()
        self.current_mode = "latent"
        self.pending_mode = None
        self.mode_step = 0
        self.current_intensity = self.global_intensity
        self.current_rejection_rate = 0.0
        self.current_trust = 0.5
        self.rejection_ema = 0.0
        self.trust_ema = 0.5
        self.global_step = 0

        self.latent_rej_history.clear()
        self.latent_intensity_history.clear()
        self.approach_rej_history.clear()
        self.approach_intensity_history.clear()
        self.cooldown_rej_history.clear()
        self.cooldown_intensity_history.clear()
        self.burst_duration_hist.clear()
        self.cooldown_duration_hist.clear()
        self.rejection_history.clear()
        self.pressure_history.clear()
        self.global_rej_intensity_hist.clear()

    # =========================================================
    # helpers
    # =========================================================
    def _clip(self, x: float) -> float:
        return float(np.clip(x, self.min_intensity, self.max_intensity))

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _safe_mean(self, values, default: float) -> float:
        if values is None or len(values) == 0:
            return float(default)
        return float(np.mean(values))

    def _normalize_efficiencies(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        mn = float(np.min(scores))
        mx = float(np.max(scores))
        if mx - mn < 1e-8:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - mn) / (mx - mn)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float) / self.softmax_temperature
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        if not np.isfinite(s) or s <= 0:
            return np.ones_like(x, dtype=float) / len(x)
        return e / s

    def _apply_pending_transition(self):
        """
        Apply the mode decided in the previous step.
        """
        if self.pending_mode is None or self.pending_mode == self.current_mode:
            self.pending_mode = None
            return

        prev_mode = self.current_mode
        prev_len = max(1, self.mode_step)

        if prev_mode == "burst":
            self.burst_duration_hist.append(float(prev_len))
        elif prev_mode == "cooldown":
            self.cooldown_duration_hist.append(float(prev_len))

        self.current_mode = self.pending_mode
        self.mode_step = 0
        self.pending_mode = None

    def _stable_use_current(self, mode: str) -> bool:
        return self.current_mode == mode and self.mode_step >= 1

    def _estimate_cooldown_steps_to_target(self, target_rej: float = 0.3) -> float:
        """
        Estimate how many steps are needed for rejection rate to drop to target_rej.
        Use recent cooldown rejection trend if available; otherwise fallback to global trend.
        """
        if self.current_rejection_rate <= target_rej:
            return 1.0

        hist = list(self.cooldown_rej_history)
        if len(hist) >= 2:
            slope = (hist[-1] - hist[0]) / max(1, len(hist) - 1)
        else:
            hist2 = list(self.rejection_history)
            if len(hist2) >= 2:
                slope = (hist2[-1] - hist2[0]) / max(1, len(hist2) - 1)
            else:
                slope = -0.05

        if slope >= 0:
            return float(self.cooldown_max_steps)

        steps = (self.current_rejection_rate - target_rej) / max(1e-6, -slope)
        return float(np.clip(np.ceil(steps), 1, self.cooldown_max_steps))

    def _fit_approach_optimal_intensity(self) -> Tuple[float, float]:
        """
        Fit rejection ~= a + b * intensity using recent global history,
        then solve max intensity * (1 - rej(intensity)).
        Returns:
            I_star, score_star
        """
        pairs = list(self.global_rej_intensity_hist)
        if len(pairs) < 2:
            I_star = self.current_intensity
            rej_star = self.current_rejection_rate
            return float(I_star), float(np.clip(I_star * (1.0 - rej_star), 0.0, 1.0))

        I = np.array([p[1] for p in pairs], dtype=float)
        R = np.array([p[0] for p in pairs], dtype=float)

        I_mean = float(np.mean(I))
        R_mean = float(np.mean(R))
        var_I = float(np.var(I))

        if var_I < 1e-8:
            I_star = float(np.clip(self.current_intensity, self.min_intensity, self.max_intensity))
            rej_star = self.current_rejection_rate
            return float(I_star), float(np.clip(I_star * (1.0 - rej_star), 0.0, 1.0))

        cov = float(np.mean((I - I_mean) * (R - R_mean)))
        b = cov / var_I
        a = R_mean - b * I_mean

        # maximize f(I) = I * (1 - (a + bI))
        # derivative: 1 - a - 2bI = 0 -> I* = (1-a)/(2b), if b > 0
        if b > 1e-6:
            I_star = (1.0 - a) / (2.0 * b)
        else:
            I_star = self.max_intensity

        I_star = float(np.clip(I_star, self.min_intensity, self.max_intensity))
        rej_star = float(np.clip(a + b * I_star, 0.0, 1.0))
        score_star = float(np.clip(I_star * (1.0 - rej_star), 0.0, 1.0))
        return I_star, score_star

    # =========================================================
    # update feedback
    # =========================================================
    def update_state(self, df: pd.DataFrame):
        """
        Global control only.
        """
        if df is None or df.empty:
            self.current_rejection_rate = 0.0
            self.current_trust = 1.0
            self.rejection_ema = 0.9 * self.rejection_ema + 0.1 * self.current_rejection_rate
            self.trust_ema = 0.9 * self.trust_ema + 0.1 * self.current_trust
            self.rejection_history.append(self.current_rejection_rate)
            self.pressure_history.append(self.get_global_attack_pressure())
            return

        if "attack_type" in df.columns:
            feedback_df = df[df["attack_type"] != 0]
            if feedback_df.empty:
                feedback_df = df
        else:
            feedback_df = df

        total = len(feedback_df)
        if self.col_weight in feedback_df.columns:
            # 统一口径：被拒比例 = 攻击流量实际未通过的权重占比 = mean(1 - w)。
            # 这样对 action_dim=2 ([1.0,0.0]) 与 action_dim=3 ([1.0,0.5,0.0])
            # 都成立，避免硬编码 “level==2 全拒、level==1 半拒” 在 2 档动作下失真。
            rejected = float((1.0 - feedback_df[self.col_weight]).sum())
        else:
            # 兜底：无权重列时退化为原访问等级计数
            rejected = (feedback_df[self.col_access] == 2).sum()
            rejected += (feedback_df[self.col_access] == 1).sum() * 0.5

        self.current_rejection_rate = float(rejected / total) if total > 0 else 0.0
        self.current_trust = float(np.clip(1.0 - self.current_rejection_rate, 0.0, 1.0))

        self.rejection_ema = 0.8 * self.rejection_ema + 0.2 * self.current_rejection_rate
        self.trust_ema = 0.8 * self.trust_ema + 0.2 * self.current_trust

        self.rejection_history.append(self.current_rejection_rate)
        self.pressure_history.append(self.get_global_attack_pressure())

        if self.col_sender in feedback_df.columns:
            self.observed_senders.update(feedback_df[self.col_sender].unique().tolist())

        # record global history for stability/fit
        self.global_rej_intensity_hist.append((self.current_rejection_rate, self.current_intensity))

        if self.current_mode == "latent":
            self.latent_rej_history.append(self.current_rejection_rate)
            self.latent_intensity_history.append(self.current_intensity)
        elif self.current_mode == "approach":
            self.approach_rej_history.append(self.current_rejection_rate)
            self.approach_intensity_history.append(self.current_intensity)
        elif self.current_mode == "cooldown":
            self.cooldown_rej_history.append(self.current_rejection_rate)
            self.cooldown_intensity_history.append(self.current_intensity)

    # =========================================================
    # efficiencies
    # =========================================================
    def _latent_efficiency(self) -> float:
        if self._stable_use_current("latent"):
            rej = self.current_rejection_rate
            inten = self.current_intensity
        else:
            rej = self._safe_mean(self.latent_rej_history,self.current_intensity)
            inten = self._safe_mean(self.latent_intensity_history,self.current_intensity)

        return float(np.clip((rej) * inten, 0.0, 1.0))

    def _approach_efficiency(self) -> float:
        # stable: use current value directly
        if self._stable_use_current("approach"):
            rej = self.current_rejection_rate
            inten = self.current_intensity
            score = rej * inten
            return float(np.clip(score, 0.0, 1.0))

        # otherwise: solve max (1-rej) * intensity by fitting recent history
        I_star, score_star = self._fit_approach_optimal_intensity()
        # keep the idea explicit: the score is the optimum of (1-rej) * intensity
        return float(np.clip(score_star, 0.0, 1.0))

    def _burst_efficiency(self) -> float:
        est_burst_duration = self._safe_mean(self.burst_duration_hist, 2.0)
        est_cooldown_duration = self._estimate_cooldown_steps_to_target(target_rej=0.3)

        planned_intensity = self.burst_intensity
        expected_benefit = est_burst_duration * planned_intensity * max(0.0, 1.0 - self.current_rejection_rate)
        eff = expected_benefit / (est_burst_duration + est_cooldown_duration + 1e-6)

        return float(np.clip(eff, 0.0, 1.0))

    def _cooldown_efficiency(self) -> float:
        if self._stable_use_current("cooldown"):
            rej = self.current_rejection_rate
            inten = self.current_intensity
        else:
            rej = self._safe_mean(self.cooldown_rej_history, self.current_rejection_rate)
            inten = self._safe_mean(self.cooldown_intensity_history, self.current_intensity)

        # cooling is attractive when rejection is high and intensity is low
        score = rej * max(0.0, 1.0 - inten)
        return float(np.clip(score, 0.0, 1.0))

    def _phase_scores(self) -> np.ndarray:
        return np.array(
            [
                self._latent_efficiency(),
                self._approach_efficiency(),
                self._burst_efficiency(),
                self._cooldown_efficiency(),
            ],
            dtype=float,
        )

    # =========================================================
    # mode transition
    # =========================================================
    def _sample_next_mode(self, current_mode: str) -> str:
        """
        Global mode transition.
        Burst and cooldown are probabilistic, not hard-cut.
        """
        scores = self._phase_scores()
        norm_scores = self._normalize_efficiencies(scores)
        probs = self._softmax(norm_scores)

        mode_idx = {m: i for i, m in enumerate(self.MODE_NAMES)}

        # cooldown: deterministic exit when rejection is low enough
        if current_mode == "cooldown":
            if self.current_rejection_rate < self.cooldown_reject_exit:
                return "latent"

            # no probabilistic jump before max steps
            if self.mode_step < self.cooldown_max_steps:
                return "cooldown"

            # after max duration, choose by scores
            return str(self.rng.choice(list(self.MODE_NAMES), p=probs))

        # burst -> cooldown probabilistic
        if current_mode == "burst":
            p_cool = self._sigmoid(12.0 * (self.current_rejection_rate - self.burst_reject_trigger))

            if self.mode_step >= self.burst_soft_max_steps:
                p_cool = min(0.95, p_cool + 0.15)

            return "cooldown" if self.rng.random() < p_cool else "burst"

        # latent / approach: keep current mode with base probability
        keep_prob_map = {
            "latent": self.latent_keep_prob_high if self.current_rejection_rate > self.latent_reject_split else self.latent_keep_prob_low,
            "approach": self.approach_keep_prob,
        }
        keep_prob = keep_prob_map.get(current_mode, 0.1)

        p = (1.0 - keep_prob) * probs
        p[mode_idx[current_mode]] += keep_prob
        p = p / p.sum()

        next_mode = self.rng.choice(list(self.MODE_NAMES), p=p)
        return str(next_mode)

    # =========================================================
    # intensity policy
    # =========================================================
    def _current_mode_intensity(self) -> float:
        if self.current_mode == "latent":
            return float(self.rng.uniform(self.latent_intensity[0], self.latent_intensity[1]))

        elif self.current_mode == "approach":
            # ==================== 新增：线性拟合 + 比例控制 ====================
            I_star, _ = self._fit_approach_optimal_intensity()
            
            # 比例控制系数（可调，越大越激进）
            approach_kp = 0.35          # 推荐范围 0.25 ~ 0.5
            
            # 向最优强度靠近
            intensity = self.current_intensity + approach_kp * (I_star - self.current_intensity)
            
            # 加入轻微下限保护，防止卡在太低的位置
            intensity = max(intensity, self.current_intensity * 0.92)
            
            # 上限保护（接近 burst 但留一定安全 margin）
            target_max = 0.92 * self.burst_intensity
            
            return float(np.clip(intensity, self.min_intensity, target_max))
        
        elif self.current_mode == "burst":
            return self.burst_intensity
        
        elif self.current_mode == "cooldown":
            intensity = self.current_intensity * self.cooldown_decay
            intensity = min(intensity, self.cooldown_max_intensity)
            return self._clip(intensity)
        
        return self.global_intensity

    # =========================================================
    # main interface
    # =========================================================
    def attack_intensity_cal(self) -> Tuple[float, Dict[Any, float]]:
        """
        Returns:
            global_intensity: float
            sender_intensity_map: all observed senders share the same intensity
        """
        self.global_step += 1

        # apply previous decision
        self._apply_pending_transition()

        # fixed mode
        if self.attack_mode == "fixed":
            self.current_mode = "fixed"
            self.current_intensity = self.fixed_intensity
            sender_map = {sid: self.current_intensity for sid in self.observed_senders}
            return self.current_intensity, sender_map

        # current step in current mode
        self.mode_step += 1

        # choose intensity for current mode
        self.current_intensity = self._current_mode_intensity()
        self.current_intensity = self._clip(self.current_intensity)

        # decide next mode
        next_mode = self._sample_next_mode(self.current_mode)
        self.pending_mode = next_mode

        sender_map = {sid: self.current_intensity for sid in self.observed_senders}
        return float(self.current_intensity), sender_map

    # =========================================================
    # compatibility accessors
    # =========================================================
    def get_sender_trust_map(self) -> Dict[Any, float]:
        """
        Global control: return the same trust value for all observed senders.
        """
        return {sid: float(self.current_trust) for sid in self.observed_senders}

    def get_global_attack_pressure(self) -> float:
        """
        A simple global pressure proxy.
        """
        return float(np.clip(self.current_rejection_rate, 0.0, 1.0))

    def get_sender_record_table(self) -> pd.DataFrame:
        """
        Global summary table (one-row table), compatible with Env metrics.
        """
        return pd.DataFrame([{
            "sender_id": "global",
            "current_mode": self.current_mode,
            "pending_mode": self.pending_mode if self.pending_mode is not None else self.current_mode,
            "phase_step": int(self.mode_step),
            "current_intensity": float(self.current_intensity),
            "current_rejection_rate": float(self.current_rejection_rate),
            "rolling_rejection_rate": float(np.mean(self.rejection_history)) if len(self.rejection_history) > 0 else 0.0,
            "trust_score": float(self.current_trust),
            "trust_ema": float(self.trust_ema),
            "burst_est_duration": float(self._safe_mean(self.burst_duration_hist, 2.0)),
            "cooldown_est_duration": float(self._estimate_cooldown_steps_to_target(target_rej=0.3)),
            "observed_sender_count": int(len(self.observed_senders)),
            "global_step": int(self.global_step),
        }])

    # =========================================================
    # plotting
    # =========================================================
    def plot_rejection_history(self, save_path: str = "rejection_history.png"):
        if len(self.rejection_history) < 2:
            return

        plt.figure(figsize=(10, 5))
        history = list(self.rejection_history)
        plt.plot(history, "b-o", linewidth=2, markersize=5, label="Global Rejection Rate")
        plt.axhline(y=self.burst_reject_trigger, color="red", linestyle="--",
                    label=f"Burst Trigger ({self.burst_reject_trigger})", alpha=0.7)
        plt.axhline(y=self.cooldown_reject_exit, color="green", linestyle="--",
                    label=f"Cooldown Exit ({self.cooldown_reject_exit})", alpha=0.7)
        plt.title("Rejection Rate History")
        plt.xlabel("Recent Steps")
        plt.ylabel("Rejection Rate")
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_pressure_history(self, save_path: str = "pressure_history.png"):
        if len(self.pressure_history) < 2:
            return

        plt.figure(figsize=(10, 5))
        history = list(self.pressure_history)
        plt.plot(history, "m-o", linewidth=2, markersize=5, label="Global Pressure Proxy")
        plt.title("Pressure History")
        plt.xlabel("Recent Steps")
        plt.ylabel("Pressure")
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()