"""Config schema tests: every shipped YAML must load and validate."""
from pathlib import Path

import pytest

from rlac.config import EnvConfig, SimConfig, load_config

CFG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.mark.parametrize("yaml_name", sorted(p.name for p in CFG_DIR.glob("*.yaml")))
def test_load_every_config(yaml_name):
    cfg = load_config(CFG_DIR / yaml_name)
    assert isinstance(cfg, SimConfig)
    assert isinstance(cfg.env, EnvConfig)
    assert cfg.mode in ("train", "test")
    assert cfg.env.action_dim in (2, 3)
    assert cfg.seed > 0


def test_missing_required_env_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\nmode: train\nenv: {}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(bad)


def test_unknown_top_key_warns(capsys):
    cfg = load_config(CFG_DIR / "train_rl.yaml")
    assert cfg.train_mode == "rl"
