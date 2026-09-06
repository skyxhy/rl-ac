"""Feature-name registry (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

FEATURE_NAMES: List[str] = [
    "mean_x",
    "mean_y",
    "std_x",
    "std_y",
    "msg_count",
    "uniq_recv",
    "recv_entropy",
    "active_span",
    "gap_mean",
    "gap_std",
    "gap_cv",
    "radius_mean",
    "sync_ratio",
    "shared_recv_ratio",
    "mean_recv_pop",
    "max_recv_pop",
    "deg",
    "wdeg",
]


@dataclass
class FeatureConfig:
    window: int = 1
    step: int = 1
    num_windows: int = 100
    start_offset: int = 100
    time_dep_len: int = 3
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
