"""Windowed access to a VeReMi-derived message CSV.

Old implementation re-masked the whole 3.3M-row frame on every ``read_data``
call (O(N) per environment step). The message stream is time-sorted (produced
by ``test-go``), so we precompute a time -> row-slice index once and then read
each window in O(window rows). If the data turns out unsorted we fall back to
the original boolean mask so behaviour never silently changes.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TIME_COL = "time"


def load_window_with_attack_intensity(
    df: pd.DataFrame,
    start_time: int,
    end_time: int,
    attack_intensity: float = 1.0,
    attack_col: str = "attack_type",
    sender_col: str = "sender_id",
    time_col: str = TIME_COL,
    random_seed: Optional[int] = 42,
    sender_intensity_map: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """Slice ``[start_time, end_time)`` and downsample attacker requests.

    Each attacker sender keeps ``ceil(n * intensity)`` rows chosen without
    replacement. ``random_seed`` should be derived from the run seed so that a
    fixed config reproduces the same windows.
    """
    assert 0 < attack_intensity <= 1.0, "attack_intensity must be in (0, 1]"
    if sender_intensity_map is not None:
        for intensity in sender_intensity_map.values():
            assert 0 < intensity <= 1.0, "per-sender intensity must be in (0, 1]"

    rng = np.random.default_rng(random_seed)

    window_df = df[(df[time_col] >= start_time) & (df[time_col] < end_time)].copy()
    if window_df.empty:
        return window_df

    attack_mask = window_df[attack_col] > 0
    benign_df = window_df[~attack_mask]
    attack_df = window_df[attack_mask]
    if attack_df.empty:
        return window_df.reset_index(drop=True)

    parts = []
    for sender_id, group in attack_df.groupby(sender_col):
        intensity = (
            sender_intensity_map.get(sender_id, attack_intensity)
            if sender_intensity_map
            else attack_intensity
        )
        n_total = len(group)
        n_keep = max(1, int(np.ceil(n_total * intensity)))
        pool_idx = group.index.to_numpy()
        keep_idx = rng.choice(pool_idx, size=n_keep, replace=False)
        parts.append(group.loc[keep_idx])

    sampled = pd.concat(parts, axis=0) if parts else pd.DataFrame(columns=attack_df.columns)
    out = pd.concat([benign_df, sampled], axis=0)
    return out.sort_values(time_col).reset_index(drop=True)


class DataModule:
    """In-memory CSV store with an O(window) time-window reader."""

    def __init__(self, data_path: str, subsample_seed: int = 42):
        self.data_path = data_path
        self.subsample_seed = int(subsample_seed)
        self.df = pd.read_csv(data_path)
        times = self.df[TIME_COL].to_numpy()
        self.min_time, self.max_time = float(times.min()), float(times.max())
        # Sorted-message fast path: time -> (start_row, end_row) slice.
        self._sorted = bool(np.all(times[1:] >= times[:-1])) if len(times) > 1 else True
        self._times = times if self._sorted else None

    def _seed_for(self, start_time: int, attack_intensity: float) -> int:
        """Deterministic per-(run-seed, time, intensity) RNG for subsampling."""
        s = self.subsample_seed
        seed = (s * 1_000_003 + int(start_time) * 1_000_033 + int(round(attack_intensity * 1e4))) % (2**32)
        return int(seed)

    def _raw_window(self, start_time: int, end_time: int) -> pd.DataFrame:
        if self._sorted:
            lo = int(np.searchsorted(self._times, start_time, side="left"))
            hi = int(np.searchsorted(self._times, end_time, side="left"))
            return self.df.iloc[lo:hi]
        mask = (self.df[TIME_COL] >= start_time) & (self.df[TIME_COL] < end_time)
        return self.df[mask]

    def read_data(
        self,
        attack_intensity: float | int = 1,
        sender_intensity_map: Dict[Any, float] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        if start_time is None or end_time is None:
            start_time, end_time = int(self.min_time), int(self.max_time)
        if start_time < self.min_time or end_time > self.max_time:
            raise ValueError("Invalid time range")

        window_df = self._raw_window(int(start_time), int(end_time))
        seed = self._seed_for(int(start_time), float(attack_intensity))
        return load_window_with_attack_intensity(
            window_df,
            int(start_time),
            int(end_time),
            attack_intensity,
            sender_intensity_map=sender_intensity_map,
            random_seed=seed,
        )
