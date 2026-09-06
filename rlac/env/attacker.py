"""Adaptive finite-state attacker (latent -> approach -> burst -> cooldown).

Ported from ``code/env_entity/modules/attack_manager.py``. Changes:
* RNG is created once from an explicit ``random_seed`` (reproducible).
* Feedback rejection rate is derived from the *admitted weight* ``1-w`` so the
  semantics hold for both 2- and 3-level action spaces.
* matplotlib plotting removed (kept in the run layer instead).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class AttackManager:
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
        attack_mode: str = "adaptive",
        fixed_intensity: float = 0.5,
        burst_reject_trigger: float = 0.7,
        cooldown_reject_exit: float = 0.38,
        latent_reject_split: float = 0.4,
        cooldown_max_steps: int = 1,
        burst_soft_max_steps: int = 2,
        latent_intensity: Tuple[float, float] = (0.1, 0.2),
        burst_intensity: float = 0.9,
        cooldown_max_intensity: float = 0.13,
        cooldown_decay: float = 0.82,
        latent_keep_prob_low: float = 0.3,
        latent_keep_prob_high: float = 0.8,
        approach_keep_prob: float = 0.9,
        cooldown_keep_prob: float = 0.3,
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

        def _dq(): return deque(maxlen=history_len)
        self.latent_rej_history = _dq()
        self.latent_intensity_history = _dq()
        self.approach_rej_history = _dq()
        self.approach_intensity_history = _dq()
        self.cooldown_rej_history = _dq()
        self.cooldown_intensity_history = _dq()
        self.burst_duration_hist = _dq()
        self.cooldown_duration_hist = _dq()
        self.rejection_history = _dq()
        self.pressure_history = _dq()
        self.global_rej_intensity_hist = _dq()

    # ------------------------------------------------------------ reset
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
        for q in (self.latent_rej_history, self.latent_intensity_history,
                  self.approach_rej_history, self.approach_intensity_history,
                  self.cooldown_rej_history, self.cooldown_intensity_history,
                  self.burst_duration_hist, self.cooldown_duration_hist,
                  self.rejection_history, self.pressure_history,
                  self.global_rej_intensity_hist):
            q.clear()

    # ------------------------------------------------------------ helpers
    def _clip(self, x: float) -> float:
        return float(np.clip(x, self.min_intensity, self.max_intensity))

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _safe_mean(self, values, default: float) -> float:
        return float(np.mean(values)) if len(values) else float(default)

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = float(np.min(scores)), float(np.max(scores))
        if hi - lo < 1e-8:
            return np.full_like(scores, 0.5)
        return (scores - lo) / (hi - lo)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float) / self.softmax_temperature
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        if not np.isfinite(s) or s <= 0:
            return np.ones_like(x) / len(x)
        return e / s

    def _apply_pending(self):
        if self.pending_mode is None or self.pending_mode == self.current_mode:
            self.pending_mode = None
            return
        prev, prev_len = self.current_mode, max(1, self.mode_step)
        if prev == "burst":
            self.burst_duration_hist.append(float(prev_len))
        elif prev == "cooldown":
            self.cooldown_duration_hist.append(float(prev_len))
        self.current_mode = self.pending_mode
        self.mode_step = 0
        self.pending_mode = None

    def _stable(self, mode: str) -> bool:
        return self.current_mode == mode and self.mode_step >= 1

    def _cooldown_steps_to_target(self, target: float = 0.3) -> float:
        if self.current_rejection_rate <= target:
            return 1.0
        hist = list(self.cooldown_rej_history)
        if len(hist) >= 2:
            slope = (hist[-1] - hist[0]) / max(1, len(hist) - 1)
        else:
            hist = list(self.rejection_history)
            slope = (hist[-1] - hist[0]) / max(1, len(hist) - 1) if len(hist) >= 2 else -0.05
        if slope >= 0:
            return float(self.cooldown_max_steps)
        steps = (self.current_rejection_rate - target) / max(1e-6, -slope)
        return float(np.clip(np.ceil(steps), 1, self.cooldown_max_steps))

    def _fit_approach(self) -> Tuple[float, float]:
        pairs = list(self.global_rej_intensity_hist)
        if len(pairs) < 2:
            return float(self.current_intensity), float(np.clip(
                self.current_intensity * (1.0 - self.current_rejection_rate), 0.0, 1.0))
        I = np.array([p[1] for p in pairs], float)
        R = np.array([p[0] for p in pairs], float)
        var = float(np.var(I))
        if var < 1e-8:
            return float(np.clip(self.current_intensity, self.min_intensity, self.max_intensity)), float(
                np.clip(self.current_intensity * (1.0 - self.current_rejection_rate), 0.0, 1.0))
        b = float(np.mean((I - I.mean()) * (R - R.mean()))) / var
        a = float(R.mean() - b * I.mean())
        I_star = (1.0 - a) / (2.0 * b) if b > 1e-6 else self.max_intensity
        I_star = float(np.clip(I_star, self.min_intensity, self.max_intensity))
        rej = float(np.clip(a + b * I_star, 0.0, 1.0))
        return I_star, float(np.clip(I_star * (1.0 - rej), 0.0, 1.0))

    # ------------------------------------------------------------ update
    def update_state(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.current_rejection_rate = 0.0
            self.current_trust = 1.0
            self.rejection_ema = 0.9 * self.rejection_ema + 0.1 * self.current_rejection_rate
            self.trust_ema = 0.9 * self.trust_ema + 0.1 * self.current_trust
            self.rejection_history.append(self.current_rejection_rate)
            self.pressure_history.append(self.get_global_attack_pressure())
            return
        if "attack_type" in df.columns:
            feedback = df[df["attack_type"] != 0]
            if feedback.empty:
                feedback = df
        else:
            feedback = df
        total = len(feedback)
        if self.col_weight in feedback.columns:
            # rejection = mean(1 - admitted weight), consistent across action dims
            rejected = float((1.0 - feedback[self.col_weight]).sum())
        else:
            rejected = (feedback[self.col_access] == 2).sum()
            rejected += (feedback[self.col_access] == 1).sum() * 0.5
        self.current_rejection_rate = float(rejected / total) if total > 0 else 0.0
        self.current_trust = float(np.clip(1.0 - self.current_rejection_rate, 0.0, 1.0))
        self.rejection_ema = 0.8 * self.rejection_ema + 0.2 * self.current_rejection_rate
        self.trust_ema = 0.8 * self.trust_ema + 0.2 * self.current_trust
        self.rejection_history.append(self.current_rejection_rate)
        self.pressure_history.append(self.get_global_attack_pressure())
        if self.col_sender in feedback.columns:
            self.observed_senders.update(feedback[self.col_sender].unique().tolist())
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

    # ------------------------------------------------------------ efficiencies
    def _latent_eff(self):
        if self._stable("latent"):
            rej, inten = self.current_rejection_rate, self.current_intensity
        else:
            rej = self._safe_mean(self.latent_rej_history, self.current_intensity)
            inten = self._safe_mean(self.latent_intensity_history, self.current_intensity)
        return float(np.clip(rej * inten, 0.0, 1.0))

    def _approach_eff(self):
        if self._stable("approach"):
            return float(np.clip(self.current_rejection_rate * self.current_intensity, 0.0, 1.0))
        _, score = self._fit_approach()
        return float(np.clip(score, 0.0, 1.0))

    def _burst_eff(self):
        est_b = self._safe_mean(self.burst_duration_hist, 2.0)
        est_c = self._cooldown_steps_to_target(0.3)
        benefit = est_b * self.burst_intensity * max(0.0, 1.0 - self.current_rejection_rate)
        return float(np.clip(benefit / (est_b + est_c + 1e-6), 0.0, 1.0))

    def _cooldown_eff(self):
        if self._stable("cooldown"):
            rej, inten = self.current_rejection_rate, self.current_intensity
        else:
            rej = self._safe_mean(self.cooldown_rej_history, self.current_rejection_rate)
            inten = self._safe_mean(self.cooldown_intensity_history, self.current_intensity)
        return float(np.clip(rej * max(0.0, 1.0 - inten), 0.0, 1.0))

    def _phase_scores(self):
        return np.array([self._latent_eff(), self._approach_eff(), self._burst_eff(), self._cooldown_eff()], dtype=float)

    def _sample_next_mode(self, mode: str) -> str:
        probs = self._softmax(self._normalize(self._phase_scores()))
        names = list(self.MODE_NAMES)
        idx = {m: i for i, m in enumerate(names)}
        if mode == "cooldown":
            if self.current_rejection_rate < self.cooldown_reject_exit:
                return "latent"
            if self.mode_step < self.cooldown_max_steps:
                return "cooldown"
            return str(self.rng.choice(names, p=probs))
        if mode == "burst":
            p = self._sigmoid(12.0 * (self.current_rejection_rate - self.burst_reject_trigger))
            if self.mode_step >= self.burst_soft_max_steps:
                p = min(0.95, p + 0.15)
            return "cooldown" if self.rng.random() < p else "burst"
        keep = {
            "latent": self.latent_keep_prob_high if self.current_rejection_rate > self.latent_reject_split else self.latent_keep_prob_low,
            "approach": self.approach_keep_prob,
        }.get(mode, 0.1)
        p = (1.0 - keep) * probs
        p[idx[mode]] += keep
        p = p / p.sum()
        return str(self.rng.choice(names, p=p))

    def _current_intensity(self) -> float:
        if self.current_mode == "latent":
            return float(self.rng.uniform(*self.latent_intensity))
        if self.current_mode == "approach":
            I_star, _ = self._fit_approach()
            intensity = self.current_intensity + 0.35 * (I_star - self.current_intensity)
            intensity = max(intensity, self.current_intensity * 0.92)
            return float(np.clip(intensity, self.min_intensity, 0.92 * self.burst_intensity))
        if self.current_mode == "burst":
            return self.burst_intensity
        if self.current_mode == "cooldown":
            intensity = min(self.current_intensity * self.cooldown_decay, self.cooldown_max_intensity)
            return self._clip(intensity)
        return self.global_intensity

    # ------------------------------------------------------------ main
    def attack_intensity_cal(self) -> Tuple[float, Dict[Any, float]]:
        self.global_step += 1
        self._apply_pending()
        if self.attack_mode == "fixed":
            self.current_mode = "fixed"
            self.current_intensity = self.fixed_intensity
            return float(self.current_intensity), {s: self.current_intensity for s in self.observed_senders}
        self.mode_step += 1
        self.current_intensity = self._clip(self._current_intensity())
        self.pending_mode = self._sample_next_mode(self.current_mode)
        return float(self.current_intensity), {s: self.current_intensity for s in self.observed_senders}

    def get_global_attack_pressure(self) -> float:
        return float(np.clip(self.current_rejection_rate, 0.0, 1.0))

    def get_sender_trust_map(self) -> Dict[Any, float]:
        return {s: float(self.current_trust) for s in self.observed_senders}
