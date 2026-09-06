"""Typed, validated configuration for the simulation pipeline.

Replaces the ad-hoc ``yaml.safe_load`` + ``cfg.get(...)`` pattern scattered
through the old ``code.train.simulation`` module. A single YAML file fully
describes one run; unknown keys are tolerated (warned) so legacy configs still
load, but missing required keys raise a clear error.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

#: Sections/keys we intentionally ignore when reading legacy files (dead or
#: superseded fields left over from hand-edited configs).
_IGNORED_TOP = {
    "num_episodes",  # never consumed; run length is driven by max_step
}
_IGNORED_ENV = {"gamma", "attack_intensity"}  # historical strays
_IGNORED_AGENT = {"checkpoint_path"}  # kept only for test-mode configs


@dataclass
class EnvConfig:
    data_path: str
    scaler_path: str
    gnn_path: str
    window_size: int = 1
    alpha: float = 1.0
    action_dim: int = 3
    start_time: int = 30000
    max_step: int = 100
    attack_mode: str = "adaptive"  # "adaptive" | "fixed"
    fixed_intensity: float = 0.5
    sight: int = 10


@dataclass
class AgentConfig:
    device: str = "cpu"  # "cpu" | "cuda"
    hidden_dim: int = 256
    gamma: float = 0.9
    checkpoint_path: Optional[str] = None
    lr: float = 1e-3
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.997
    sync_freq: int = 10
    n_step: int = 2


@dataclass
class ThresholdConfig:
    risk_threshold: float = 0.5


@dataclass
class SimConfig:
    seed: int = 42
    mode: str = "train"  # "train" | "test"
    train_mode: str = "rl"  # "rl" | "cb"
    test_mode: str = "rl"  # "rl" | "cb" | "threshold"
    train_episodes: int = 20
    batch_size: int = 64
    warmup_size: int = 200
    buffer_size: int = 20000
    save_dir: str = "outputs/simulation"
    save_name: Optional[str] = None
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)

    # ---- convenience ----
    @property
    def method(self) -> str:
        return self.train_mode if self.mode == "train" else self.test_mode

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _require(mapping: Dict[str, Any], section: str, key: str) -> None:
    if key not in mapping:
        raise KeyError(f"config missing required key `{section}.{key}`")


def load_config(path: str | Path) -> SimConfig:
    """Parse and validate a YAML config into :class:`SimConfig`.

    Unknown keys are collected and printed as warnings so we can progressively
    prune stale configs without hard-failing.
    """
    path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    for key in _IGNORED_TOP:
        raw.pop(key, None)

    top_warn: List[str] = []
    allowed_top = {f.name for f in dataclasses.fields(SimConfig)} | {"env", "agent", "threshold"}
    for k in raw:
        if k not in allowed_top:
            top_warn.append(k)

    # env section
    env_raw: Dict[str, Any] = raw.get("env", {})
    for key in _IGNORED_ENV:
        env_raw.pop(key, None)
    _require(env_raw, "env", "data_path")
    _require(env_raw, "env", "scaler_path")
    _require(env_raw, "env", "gnn_path")
    env_allowed = {f.name for f in dataclasses.fields(EnvConfig)}
    env = EnvConfig(**{k: v for k, v in env_raw.items() if k in env_allowed})

    # agent section (mostly optional with defaults)
    agent_raw: Dict[str, Any] = raw.get("agent", {})
    for key in _IGNORED_AGENT:
        agent_raw.pop(key, None)
    agent_allowed = {f.name for f in dataclasses.fields(AgentConfig)}
    agent_kw = {k: v for k, v in agent_raw.items() if k in agent_allowed}
    if "checkpoint_path" in raw.get("agent", {}):
        agent_kw["checkpoint_path"] = raw["agent"]["checkpoint_path"]
    agent = AgentConfig(**agent_kw)

    th_raw = raw.get("threshold", {})
    threshold = ThresholdConfig(**{k: v for k, v in th_raw.items() if k in {f.name for f in dataclasses.fields(ThresholdConfig)}})

    # top-level scalar options with legacy alias `episodes` -> train_episodes
    cfg = SimConfig(
        seed=int(raw.get("seed", 42)),
        mode=str(raw.get("mode", "train")),
        train_mode=str(raw.get("train_mode", "rl")),
        test_mode=str(raw.get("test_mode", "rl")),
        train_episodes=int(raw.get("train_episodes", raw.get("episodes", 20))),
        batch_size=int(raw.get("batch_size", 64)),
        warmup_size=int(raw.get("warmup_size", 200)),
        buffer_size=int(raw.get("buffer_size", 20000)),
        save_dir=str(raw.get("save_dir", "outputs/simulation")),
        save_name=raw.get("save_name"),
        env=env,
        agent=agent,
        threshold=threshold,
    )

    if env.attack_mode not in ("fixed", "adaptive"):
        raise ValueError(f"env.attack_mode must be 'fixed'|'adaptive', got {env.attack_mode!r}")
    if cfg.mode not in ("train", "test"):
        raise ValueError(f"mode must be 'train'|'test', got {cfg.mode!r}")

    for k in top_warn:
        print(f"[config] warning: ignoring unknown top-level key `{k}` in {path}")
    return cfg
