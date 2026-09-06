"""Train / test entry point for the RL access-control pipeline.

Usage (from the repo root)::

    python -m rlac.run.simulate --config config/train_rl.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlac.agents.cb import CBAgent  # noqa: E402
from rlac.agents.drqn import DRQNAgent  # noqa: E402
from rlac.agents.replay import ReplayBuffer, SimpleReplayBuffer  # noqa: E402
from rlac.agents.threshold import GNNThresholdAgent  # noqa: E402
from rlac.config import SimConfig, load_config  # noqa: E402
from rlac.data.features import FeatureConfig  # noqa: E402
from rlac.env.attacker import AttackManager  # noqa: E402
from rlac.env.data_module import DataModule  # noqa: E402
from rlac.env.env import Env  # noqa: E402
from rlac.env.graph_module import GraphModule  # noqa: E402
from rlac.env.policy import PolicyManager  # noqa: E402
from rlac.env.risk import RiskManager  # noqa: E402
from rlac.models.gnn import GNNModel  # noqa: E402
from rlac.utils.random import seed_all  # noqa: E402


# ----------------------------------------------------------------------
def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _plot_curves(curves: dict, save_dir: str, x_label: str):
    png_dir, csv_dir, npy_dir = (os.path.join(save_dir, d) for d in ("png", "csv", "npy"))
    for d in (png_dir, csv_dir, npy_dir):
        os.makedirs(d, exist_ok=True)
    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        base = name.replace(".png", "")
        np.savetxt(os.path.join(csv_dir, base + ".csv"), values, delimiter=",")
        np.save(os.path.join(npy_dir, base + ".npy"), values)
        plt.figure(figsize=(6, 4))
        plt.plot(values, marker=".", linestyle="-", linewidth=1.5)
        plt.xlabel(x_label)
        plt.ylabel(base.replace("episode_", "").replace("step_", "").replace("_", " ").title())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(png_dir, base + ".png"), dpi=150)
        plt.close()


def _plot_history(history, ylabel, out, threshold=None):
    if len(history) < 2:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(list(history), "o-", linewidth=2, markersize=4, label=ylabel)
    if threshold is not None:
        plt.axhline(threshold, color="red", ls="--", alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def build_env(cfg: SimConfig):
    data_module = DataModule(cfg.env.data_path, subsample_seed=cfg.seed)
    feat_cfg = FeatureConfig()
    scaler = joblib.load(cfg.env.scaler_path)
    graph_module = GraphModule(feat_cfg.feature_names, scaler=scaler)
    gnn = GNNModel(in_dim=graph_module.feature_dim)
    gnn.load_state_dict(torch.load(cfg.env.gnn_path, map_location=cfg.agent.device))
    gnn.eval()
    risk = RiskManager(gnn, device=cfg.agent.device)
    attack = AttackManager(
        attack_mode=cfg.env.attack_mode,
        fixed_intensity=cfg.env.fixed_intensity,
        random_seed=cfg.seed,
    )
    env = Env(
        graph_feature_builder=graph_module,
        attack_manager=attack,
        policy_manager=PolicyManager(),
        risk_manager=risk,
        data_module=data_module,
        window_size=cfg.env.window_size,
        alpha=cfg.env.alpha,
        action_dim=cfg.env.action_dim,
        start_time=cfg.env.start_time,
        max_step=cfg.env.max_step,
        sight=cfg.env.sight,
    )
    return env


def build_agent(cfg: SimConfig, env: Env, mode: str):
    dev = cfg.agent.device
    if mode == "rl":
        agent = DRQNAgent(
            feature_dim=env.state_dim, action_dim=env.action_dim,
            hidden_dim=cfg.agent.hidden_dim, gamma=cfg.agent.gamma, device=dev,
        )
        return agent
    if mode == "cb":
        return CBAgent(feature_dim=env.state_dim, action_dim=env.action_dim,
                       hidden_dim=cfg.agent.hidden_dim, device=dev)
    if mode == "threshold":
        return GNNThresholdAgent(env.risk_manager.model, threshold=cfg.threshold.risk_threshold,
                                 action_dim=env.action_dim, device=dev)
    raise ValueError(mode)


def _save_dir(cfg: SimConfig, mode: str) -> str:
    name = cfg.save_name or f"{mode}_{cfg.method}"
    d = os.path.join(cfg.save_dir, name)
    os.makedirs(d, exist_ok=True)
    return d


def run_train(env, agent, cfg: SimConfig):
    is_cb = cfg.train_mode == "cb"
    replay = (SimpleReplayBuffer(capacity=cfg.buffer_size, seed=cfg.seed) if is_cb
              else ReplayBuffer(capacity=cfg.buffer_size, gamma=agent.gamma, seed=cfg.seed))
    save_dir = _save_dir(cfg, "train")
    curves = {"episode_return.png": [], "episode_mean_reward.png": [], "episode_length.png": []}
    losses = []
    metrics = defaultdict(list)

    for ep in range(cfg.train_episodes):
        state = env.reset()
        done = False
        ep_ret = ep_rew = 0.0
        ep_len = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, m = env.step(action)
            if is_cb:
                replay.push(state, action, reward)
            else:
                replay.batch_push(state, action, reward, next_state)
            if hasattr(agent, "update") and len(replay) >= cfg.warmup_size:
                batch = replay.sample(cfg.batch_size)
                if is_cb:
                    bs, ba, br = zip(*batch)
                    bn = np.zeros(len(bs), dtype=np.float32)
                else:
                    bs, ba, br, bn = zip(*batch)
                losses.append(agent.update(np.array(bs), np.array(ba), np.array(br), np.array(bn)))
            r_val = reward[1].mean() if isinstance(reward, tuple) and reward[1] is not None else float(np.mean(reward))
            ep_ret += r_val
            ep_rew += r_val
            for k, v in m.items():
                metrics[k].append(float(v))
            state = next_state
            ep_len += 1
        curves["episode_return.png"].append(ep_ret)
        curves["episode_mean_reward.png"].append(ep_rew / max(1, ep_len))
        curves["episode_length.png"].append(ep_len)
        print(f"Ep {ep+1}/{cfg.train_episodes} | Len={ep_len} | R_sum={ep_ret:.3f} | "
              f"Loss={losses[-1]:.3f}" if losses else f"Ep {ep+1}/{cfg.train_episodes} | Len={ep_len} | R_sum={ep_ret:.3f}")

    agent.save(os.path.join(save_dir, "final_agent.pt"))
    _plot_history(env.attack_manager.rejection_history, "Rejection Rate",
                  os.path.join(save_dir, "rejection_history.png"), threshold=0.7)
    _plot_history(env.attack_manager.pressure_history, "Pressure",
                  os.path.join(save_dir, "pressure_history.png"))
    for k, v in metrics.items():
        curves.setdefault(f"episode_{k}.png", []).append(float(np.mean(v)))
    _plot_curves(curves, save_dir, "Episode")
    _write_meta(save_dir, cfg, method=cfg.train_mode)
    return curves


def run_test(env, agent, cfg: SimConfig, mode: str):
    if hasattr(agent, "eval"):
        agent.eval()
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    save_dir = _save_dir(cfg, "test")
    state = env.reset()
    curves = {"step_reward.png": []}
    metrics = defaultdict(list)
    done = False
    step = 0
    while not done:
        action = agent.take_action(state)
        state, reward, done, m = env.step(action)
        r_val = reward[1].mean() if isinstance(reward, tuple) and reward[1] is not None else float(np.mean(reward))
        curves["step_reward.png"].append(r_val)
        for k, v in m.items():
            metrics[k].append(float(v))
        step += 1
    _plot_history(env.attack_manager.rejection_history, "Rejection Rate",
                  os.path.join(save_dir, "rejection_history.png"), threshold=0.7)
    _plot_history(env.attack_manager.pressure_history, "Pressure",
                  os.path.join(save_dir, "pressure_history.png"))
    for k, v in metrics.items():
        curves.setdefault(f"step_{k}.png", []).append(float(np.mean(v)))
    _plot_curves(curves, save_dir, "Step")
    _write_meta(save_dir, cfg, method=mode)
    return curves


def _write_meta(save_dir: str, cfg: SimConfig, method: str):
    meta = {"config": cfg.to_dict(), "method": method, "git_rev": git_rev()}
    with open(os.path.join(save_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.seed)
    env = build_env(cfg)
    if cfg.mode == "train":
        agent = build_agent(cfg, env, cfg.train_mode)
        run_train(env, agent, cfg)
    else:
        mode = cfg.test_mode
        env2 = build_env(cfg)
        if mode == "threshold":
            agent = build_agent(cfg, env2, "threshold")
        else:
            agent = build_agent(cfg, env2, mode)
            if not cfg.agent.checkpoint_path or not os.path.exists(cfg.agent.checkpoint_path):
                raise FileNotFoundError(cfg.agent.checkpoint_path)
            agent.load(cfg.agent.checkpoint_path, device=cfg.agent.device)
        run_test(env2, agent, cfg, mode)
    print("done")


if __name__ == "__main__":
    main()
