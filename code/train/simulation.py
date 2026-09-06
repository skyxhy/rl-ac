import os
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from collections import defaultdict

# ================= PROJECT IMPORTS =================
from ..env_entity.modules.data_module import DataModule
from ..env_entity.modules.attack_manager import Attack_Manager
from ..env_entity.modules.feature_graph_module import Graph_Module
from ..env_entity.modules.policy_manager import PolicyManager
from ..env_entity.modules.risk_manager import Risk_Manager
from ..env_entity.env import Env
from ..entity_agent.agent import DRQNAgent, ReplayBuffer, SimpleReplayBuffer
from ..entity_agent.cb_agent import CBAgent
from ..model.gnn_model import GNN_model
from ..data.feature_config import FeatureConfig


class GNNThresholdAgent:
    """基于 GNN 风险分数的阈值决策 Agent"""
    def __init__(self, gnn_model: GNN_model, threshold: float = 0.5, action_dim: int = 2, device: str = "cpu"):
        self.gnn = gnn_model
        self.threshold = threshold
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.gnn.to(self.device)
        self.gnn.eval()

        self.defensive_action = action_dim - 1
        self.epsilon = 0

    def take_action(self, state):
        """
        期望 state 格式: (node_map, x, edge_index, edge_attr)
        返回格式: (node_map, np.ndarray)
        """
        node_map = state[0]
        x = state[1]

        x = torch.tensor(x[:, -1, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            embedding = x[:, :self.gnn.hidden * self.gnn.heads]
            scores = self.gnn.out_head(embedding)
            probs = torch.sigmoid(scores).squeeze(-1)

        actions = []
        for pseudo, idx in node_map.items():
            risk_prob = probs[idx].item()
            action = 0 if risk_prob < self.threshold else self.defensive_action
            actions.append(action)

        return (node_map, np.array(actions))

    def eval(self):
        pass


def _to_scalar(v):
    if isinstance(v, (int, float, np.number)):
        return float(v)
    try:
        arr = np.asarray(v, dtype=float)
        return float(np.mean(arr))
    except Exception:
        return float(v)


def _reward_to_scalar(reward):
    if isinstance(reward, (list, tuple)) and len(reward) > 1:
        r_val = reward[1]
        return _to_scalar(r_val)
    return _to_scalar(reward)


def _make_save_dir(cfg: dict, mode: str, submode: str) -> str:
    """
    mode: train / test
    submode: rl / cb / threshold
    """
    save_root = cfg["save_dir"]

    # 优先使用用户自定义名字
    save_name = (
        cfg.get("save_name")
        or cfg.get("save_dir_name")
        or cfg.get("exp_name")
    )

    # 没指定就用默认名字
    if not save_name:
        save_name = f"{mode}_{submode}"

    save_dir = os.path.join(save_root, save_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _plot_curves(curves: dict, save_dir: str, x_label: str = "Episode"):
    png_dir = os.path.join(save_dir, "png")
    csv_dir = os.path.join(save_dir, "csv")
    npy_dir = os.path.join(save_dir, "npy")

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        base_name = name.replace(".png", "")

        csv_path = os.path.join(csv_dir, base_name + ".csv")
        np.savetxt(csv_path, values, delimiter=",")

        npy_path = os.path.join(npy_dir, base_name + ".npy")
        np.save(npy_path, values)

        plt.figure(figsize=(6, 4))
        plt.plot(values, marker='.', linestyle='-', linewidth=1.5)
        plt.xlabel(x_label, fontsize=12)
        clean_name = base_name.replace("step_", "").replace("episode_", "").replace("_", " ").title()
        plt.ylabel(clean_name, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        png_path = os.path.join(png_dir, base_name + ".png")
        plt.savefig(png_path, dpi=150)
        plt.close()


def run_train(env: Env, agent, cfg: dict) -> dict:
    """
    训练按 episode 记录：
    - episode return
    - episode mean reward
    - episode mean loss
    - episode mean metrics
    """
    is_cb = cfg.get("train_mode", "rl") == "cb"
    if is_cb:
        # CB 消融：无自举 → 用单步即时奖励的逐节点缓冲（避免“偷看未来”）
        replay_buffer = SimpleReplayBuffer(capacity=cfg.get("buffer_size", 20000))
    else:
        # RL：2 步缓冲，其折现 gamma 必须与 agent 的 gamma 一致
        agent_gamma = getattr(agent, "gamma", None)
        if agent_gamma is None:
            agent_gamma = cfg.get("agent", {}).get("gamma", 0.85)
        replay_buffer = ReplayBuffer(capacity=cfg.get("buffer_size", 20000), gamma=agent_gamma)

    save_dir = _make_save_dir(cfg, mode="train", submode=cfg.get("train_mode", "rl"))

    step_histories = {
        "episode_return.png": [],
        "episode_mean_reward.png": [],
        "episode_length.png": [],
    }

    record_loss = hasattr(agent, "update")
    if record_loss:
        step_histories["episode_mean_loss.png"] = []

    num_episodes = int(cfg.get("train_episodes", cfg.get("episodes", 1)))
    warmup_size = int(cfg.get("warmup_size", 1))
    batch_size = int(cfg.get("batch_size", 32))

    print("🚀 Starting Training...")

    for ep in tqdm(range(num_episodes), desc="Training Episodes", unit="episode"):
        state = env.reset()
        done = False
        ep_step = 0

        ep_rewards = []
        ep_losses = []
        ep_metrics = defaultdict(list)

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, metrics = env.step(action)

            if is_cb:
                replay_buffer.push(state, action, reward)
            else:
                replay_buffer.batch_push(state, action, reward, next_state)

            # ---------- update ----------
            loss_val = np.nan
            if record_loss and len(replay_buffer) >= warmup_size:
                batch = replay_buffer.sample(batch_size)
                if is_cb:
                    bs, ba, br = zip(*batch)
                    bn = np.zeros(len(bs), dtype=np.float32)  # CB 不使用，仅为接口一致
                else:
                    bs, ba, br, bn = zip(*batch)
                loss = agent.update(np.array(bs), np.array(ba), np.array(br), np.array(bn))
                loss_val = _to_scalar(loss)

            # ---------- reward ----------
            r_val = _reward_to_scalar(reward)
            ep_rewards.append(r_val)

            if record_loss:
                ep_losses.append(loss_val)

            # ---------- metrics ----------
            for k, v in metrics.items():
                ep_metrics[k].append(_to_scalar(v))

            state = next_state
            ep_step += 1

        # ===== episode statistics =====
        episode_return = float(np.sum(ep_rewards)) if len(ep_rewards) > 0 else 0.0
        episode_mean_reward = float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else 0.0
        episode_length = int(ep_step)

        step_histories["episode_return.png"].append(episode_return)
        step_histories["episode_mean_reward.png"].append(episode_mean_reward)
        step_histories["episode_length.png"].append(episode_length)

        if record_loss:
            valid_losses = np.asarray(ep_losses, dtype=float)
            episode_mean_loss = float(np.nanmean(valid_losses)) if np.any(~np.isnan(valid_losses)) else np.nan
            step_histories["episode_mean_loss.png"].append(episode_mean_loss)

        for k, vals in ep_metrics.items():
            step_histories.setdefault(f"episode_{k}.png", []).append(float(np.mean(vals)))

        # ===== logging =====
        log_msg = (
            f"Ep {ep + 1}/{num_episodes} | "
            f"Len={episode_length} | "
            f"R_sum={episode_return:.3f} | "
            f"R_mean={episode_mean_reward:.3f}"
        )
        if record_loss:
            log_msg += f" | Loss={episode_mean_loss:.3f}" if not np.isnan(episode_mean_loss) else " | Loss=nan"
        print(log_msg)

    print(f"✅ Training complete: {num_episodes} episodes")

    # ✅ 保存模型
    if hasattr(agent, "save"):
        agent.save(os.path.join(save_dir, "final_agent.pt"))

    # ✅ 保存 attack 曲线
    env.attack_manager.plot_rejection_history(os.path.join(save_dir, "rejection_history.png"))
    env.attack_manager.plot_pressure_history(os.path.join(save_dir, "pressure_history.png"))

    # ✅ 保存所有曲线（episode 级别）
    _plot_curves(step_histories, save_dir, x_label="Episode")

    return step_histories


def run_test(env: Env, agent, cfg: dict, mode: str = "rl") -> dict:
    if hasattr(agent, 'eval'):
        agent.eval()
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0

    save_dir = _make_save_dir(cfg, mode="test", submode=mode)

    step_histories = {"step_reward.png": []}

    state = env.reset()
    done = False
    step = 0

    print(f"🚀 Testing ({mode})...")

    with tqdm(desc="Testing", unit="step") as pbar:
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, metrics = env.step(action)

            r_val = _reward_to_scalar(reward)
            step_histories["step_reward.png"].append(r_val)

            for k, v in metrics.items():
                val = _to_scalar(v)
                step_histories.setdefault(f"step_{k}.png", []).append(val)

            pbar.update(1)
            pbar.set_postfix({
                "Step": step,
                "R": f"{r_val:.3f}",
                "U": f"{_to_scalar(metrics.get('utility', 0)):.3f}"
            })

            state = next_state
            step += 1

    print(f"✅ Test complete: {step} steps")

    # ✅ attack 曲线
    env.attack_manager.plot_rejection_history(os.path.join(save_dir, "rejection_history.png"))
    env.attack_manager.plot_pressure_history(os.path.join(save_dir, "pressure_history.png"))

    _plot_curves(step_histories, save_dir, x_label="Step")
    return step_histories


def main():
    parser = argparse.ArgumentParser(description="Unified RL Simulation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # ================= 1. MODULE INITIALIZATION =================
    data_module = DataModule(cfg["env"]["data_path"])
    feat_cfg = FeatureConfig()
    scaler = joblib.load(cfg["env"]["scaler_path"])

    graph_module = Graph_Module(feature_names=feat_cfg.feature_names, scaler=scaler)
    gnn_model = GNN_model(in_dim=graph_module.feature_dim)
    gnn_model.load_state_dict(torch.load(cfg["env"]["gnn_path"], map_location=cfg["agent"]["device"]))
    gnn_model.eval()

    risk_manager = Risk_Manager(gnn_model)
    policy_manager = PolicyManager(data_module.df)
    attack_manager = Attack_Manager(
        attack_mode=cfg["env"]["attack_mode"],
        fixed_intensity=cfg["env"].get("fixed_intensity", 0.5),
        random_seed=seed  # 让自适应攻击者 RNG 受 seed 控制，保证可复现
    )

    env = Env(
        graph_feature_builder=graph_module,
        attack_manager=attack_manager,
        policy_manager=policy_manager,
        risk_manager=risk_manager,
        data_module=data_module,
        window_size=cfg["env"]["window_size"],
        alpha=cfg["env"]["alpha"],
        action_dim=cfg["env"]["action_dim"],
        start_time=cfg["env"]["start_time"],
        max_step=cfg["env"]["max_step"],
        sight=cfg["env"]["sight"]
    )

    # ================= 2. AGENT SETUP =================
    agent = None
    if cfg["mode"] == "train":
        train_mode = cfg.get("train_mode", "rl")

        if train_mode == "rl":
            agent = DRQNAgent(
                feature_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=cfg["agent"].get("hidden_dim", 256),
                device=cfg["agent"]["device"],
                gamma=cfg["agent"].get("gamma", 0.85)
            )

        elif train_mode == "cb":
            agent = CBAgent(
                feature_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=cfg["agent"].get("hidden_dim", 256),
                device=cfg["agent"]["device"]
            )
            print("✅ Using Contextual Bandit Agent")

        else:
            raise ValueError(f"Unsupported train_mode: {train_mode}")

    elif cfg["mode"] == "test":
        test_mode = cfg.get("test_mode", "rl")

        if test_mode == "rl":
            agent = DRQNAgent(
                feature_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=cfg["agent"].get("hidden_dim", 256),
                device=cfg["agent"]["device"]
            )
            ckpt_path = cfg["agent"].get("checkpoint_path")
            if ckpt_path and os.path.exists(ckpt_path):
                agent.load(ckpt_path, device=agent.device)
                print(f"✅ Loaded RL agent from {ckpt_path}")
            else:
                raise FileNotFoundError(f"Agent checkpoint not found: {ckpt_path}")

        elif test_mode == "threshold":
            agent = GNNThresholdAgent(
                gnn_model=gnn_model,
                threshold=cfg["threshold"].get("risk_threshold", 0.5),
                action_dim=env.action_dim,
                device=cfg["agent"]["device"]
            )
            print(f"✅ Initialized GNNThresholdAgent (thresh={cfg['threshold']['risk_threshold']}, action_dim={env.action_dim})")

        elif test_mode == "cb":
            agent = CBAgent(
                feature_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=cfg["agent"].get("hidden_dim", 256),
                device=cfg["agent"]["device"]
            )

            ckpt_path = cfg["agent"].get("checkpoint_path")
            if ckpt_path and os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=agent.device)
                agent.q_net.load_state_dict(checkpoint["model_state_dict"])
                print(f"✅ Loaded CB agent from {ckpt_path}")
            else:
                raise FileNotFoundError(f"CB checkpoint not found: {ckpt_path}")
        else:
            raise ValueError(f"Unsupported test_mode: {test_mode}")

    # ================= 3. EXECUTION =================
    if cfg["mode"] == "train":
        run_train(env, agent, cfg)
    elif cfg["mode"] == "test":
        run_test(env, agent, cfg, mode=cfg.get("test_mode", "rl"))
    else:
        raise ValueError(f"Unsupported mode: {cfg['mode']}")


if __name__ == "__main__":
    main()