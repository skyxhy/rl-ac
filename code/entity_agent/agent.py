from copy import deepcopy
from collections import deque
import random
import os
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# DRQN Model
# =========================================================
class DRQNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, num_actions=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        """
        x: (B, T, F)
        output: (B, A)
        """
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)


# =========================================================
# Replay Buffer with delayed 2-step storage
# =========================================================
class ReplayBuffer:
    """
    2-step delayed replay buffer.

    外部接口不变：
        push(state, action, reward, next_state)
        sample(batch_size)
        batch_push(state, action, reward, next_state)

    内部逻辑：
        - 对每个 pseudo node 单独维护 pending transition
        - 当同一 pseudo node 在下一次 batch_push 中再次出现时，
          才把“前一步”转成 2-step sample 入池
        - 若节点消失，则直接丢弃该节点 pending 数据
    """

    def __init__(self, capacity=10000, n_step=2, gamma=0.85):
        # 该缓冲只实现 2 步延迟回报（r_t + gamma * r_{t+1}），n_step 固定为 2。
        # 若扩展多步，需同步修改 batch_push 的回报累加与 agent.update 的折现幂次。
        assert n_step == 2, "This buffer is designed for n-step == 2 only."
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma

        # pseudo_id -> (s, a, r, next_s)
        self._pending: Dict[Any, Tuple[Any, Any, float, Any]] = {}

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch_size = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, batch_size)

    @staticmethod
    def _extract_value(batch, pseudo, idx):
        """
        从 tuple(dict, array) / dict / scalar 中提取某个 pseudo 对应的数据。
        """
        # 常见格式: (mapping, values)
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], dict):
            mapping, values = batch
            return values[mapping[pseudo]]

        # dict: pseudo -> value
        if isinstance(batch, dict):
            return batch[pseudo]

        # 标量
        if np.isscalar(batch):
            return float(batch)

        arr = np.asarray(batch)
        if arr.ndim == 0:
            return float(arr.item())

        # fallback: 按 idx 取
        return arr[idx]

    def _build_current_transitions(self, state, action, reward, next_state):
        """
        把当前 batch 对齐成:
            pseudo -> (s_t, a_t, r_t, s_{t+1})
        只保留 state/action/reward/next_state 的交集。
        """
        # 交集：只保留当前时刻都存在的节点
        pseudo_set = set(state[0]) & set(action[0]) & set(next_state[0])

        # reward 可能是 tuple(dict, array)，也可能是标量
        if isinstance(reward, tuple) and len(reward) == 2 and isinstance(reward[0], dict):
            pseudo_set = pseudo_set & set(reward[0])

        pseudo_set = sorted(pseudo_set)

        cur = {}
        for pseudo in pseudo_set:
            s_i = state[0][pseudo]
            a_i = action[0][pseudo]
            n_i = next_state[0][pseudo]

            s = state[1][s_i]
            a = action[1][a_i]
            n = next_state[1][n_i]

            r = self._extract_value(reward, pseudo, s_i)
            cur[pseudo] = (s, a, float(r), n)

        return cur

    def batch_push(
        self,
        state: tuple[Dict, np.ndarray],
        action: tuple[Dict, np.ndarray],
        reward: tuple[Dict, np.ndarray],
        next_state: tuple[Dict, np.ndarray]
    ):
        """
        延迟 2-step 入池。

        对于同一 pseudo node:
            t 时刻:     (s_t, a_t, r_t, s_{t+1})
            t+1 时刻:   (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})
        则写入经验池:
            (s_t, a_t, r_t + gamma * r_{t+1}, s_{t+2})

        若某个 pseudo node 在下一步消失，则其 pending 样本直接丢弃。
        """
        current = self._build_current_transitions(state, action, reward, next_state)

        current_pseudos = set(current.keys())
        previous_pseudos = set(self._pending.keys())

        # 1) 对于当前仍然存在的节点，若它之前有 pending，就构造 2-step transition 入池
        for pseudo, cur_tran in current.items():
            if pseudo in self._pending:
                prev_s, prev_a, prev_r, prev_next_s = self._pending[pseudo]
                cur_s, cur_a, cur_r, cur_next_s = cur_tran

                # 2-step return
                two_step_reward = prev_r + self.gamma * cur_r

                # 入池样本：s_t, a_t, R_t^(2), s_{t+2}
                self.push(prev_s, prev_a, two_step_reward, cur_next_s)

            # 当前步作为下一次的 pending
            self._pending[pseudo] = cur_tran

        # 2) 节点消失：直接删除 pending，不再保留
        disappeared = previous_pseudos - current_pseudos
        for pseudo in disappeared:
            self._pending.pop(pseudo, None)

    def flush(self):
        """
        结束一个 episode / 一个连续段时调用：
        清空所有 pending，不做最后一步补偿。
        因为没有下一步，2-step 无法成立。
        """
        self._pending.clear()


class SimpleReplayBuffer:
    """
    逐节点单步（即时奖励）缓冲，供 Contextual Bandit 消融使用。

    与 ReplayBuffer 的关键区别：
      - CB 无自举（无 TD target），因此回归目标应是即时奖励 r_t，
        而不是 ReplayBuffer 里延迟累加的 2 步回报（后者会让 CB 偷看未来奖励，
        削弱“无时序信用分配”这一消融的意义）。

    接口：
        push(state, action, reward)   # 每步把每个节点拆成一条 (s, a, r)
        sample(batch_size)            # 返回 [(s, a, r), ...]
    """

    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward):
        if state is None:
            return
        node_map = state[0]
        # reward 可能是 (node_map, values) 元组，也可能是标量
        if isinstance(reward, tuple) and len(reward) == 2 and isinstance(reward[0], dict):
            reward_map, reward_vals = reward
        else:
            reward_map, reward_vals = None, None

        for pseudo, idx in node_map.items():
            s = state[1][idx]
            a = int(action[1][action[0].get(pseudo, idx)])
            if reward_vals is not None:
                r = float(reward_vals[reward_map.get(pseudo, idx)])
            else:
                r = float(reward) if np.isscalar(reward) else float(np.mean(reward))
            self.buffer.append((s, a, r))

    def sample(self, batch_size):
        batch_size = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, batch_size)


# =========================================================
# DRQN Agent
# =========================================================
class DRQNAgent:
    def __init__(
        self,
        feature_dim,
        hidden_dim=12,
        action_dim=2,
        lr=1e-3,
        gamma=0.85,
        n_step=2,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        sync_freq=10,
        device="cpu"
    ):
        self.device = torch.device(device)

        self.action_dim = action_dim
        self.gamma = gamma
        self.n_step = n_step

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.sync_freq = sync_freq
        self.update_step = 0

        self.q_net = DRQNNet(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_actions=action_dim
        ).to(self.device)

        self.target_net = deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=lr
        )

        self.loss_func = nn.SmoothL1Loss()

    def take_action(self, state):
        if state is None:
            return {}

        node_map = state[0]
        state_tensor = torch.tensor(
            state[1], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            q_values = self.q_net(state_tensor)   # (N, A)

        actions = []
        for pseudo, idx in node_map.items():
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = torch.argmax(q_values[idx]).item()
            actions.append(action)

        actions = np.array(actions, dtype=np.int64)
        return (node_map, actions)

    def update(self, bs, ba, br, bn):
        bs = torch.tensor(bs, dtype=torch.float32, device=self.device)
        ba = torch.tensor(ba, dtype=torch.long, device=self.device).view(-1)
        br = torch.tensor(br, dtype=torch.float32, device=self.device).view(-1)
        bn = torch.tensor(bn, dtype=torch.float32, device=self.device)

        q_values = self.q_net(bs)
        q_sa = q_values.gather(1, ba.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_net(bn)
            next_action = next_q_online.argmax(dim=1)

            next_q_target = self.target_net(bn)
            next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1)

            # 注意：buffer 里已经是 2-step reward:
            # R_t^(2) = r_t + gamma * r_{t+1}
            # 所以 bootstrap 折现到 s_{t+2} 需乘 gamma^n_step = gamma^2。
            # n_step 必须与 ReplayBuffer 的实现一致（2），否则折现口径错乱。
            target_q = br + (self.gamma ** self.n_step) * next_q

        loss = self.loss_func(q_sa, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.sync_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    # ==================== Save & Load ====================
    def save(self, path: str):
        """保存完整训练状态：模型、目标网络、优化器、epsilon、步数"""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        torch.save({
            "model_state_dict": self.q_net.state_dict(),
            "target_model_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_step": self.update_step,
            "gamma": self.gamma,
            "n_step": self.n_step
        }, path)

    def load(self, path: str, device: str = None):
        """加载训练状态并自动映射设备"""
        target_device = torch.device(device if device is not None else self.device)
        checkpoint = torch.load(path, map_location=target_device)

        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.update_step = checkpoint.get("update_step", 0)

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.n_step = checkpoint.get("n_step", self.n_step)

        self.q_net.to(target_device)
        self.target_net.to(target_device)
        self.device = target_device

    def eval(self):
        self.q_net.eval()
        self.target_net.eval()


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=10000, n_step=2, gamma=0.85)

    # 仅示意：state/action/reward/next_state 最好都是 (mapping, values)
    # values 的索引位置由 mapping 指定
    s = np.ones(shape=(2, 3, 4), dtype=np.float32)
    s2 = 2 * np.ones(shape=(2, 3, 4), dtype=np.float32)
    s3 = 3 * np.ones(shape=(2, 3, 4), dtype=np.float32)

    node_map1 = {1: 0, 2: 1}
    node_map2 = {2: 0, 1: 1}
    node_map3 = {1: 0}  # 节点 2 消失，只保留交集

    a1 = np.array([0, 1], dtype=np.int64)
    a2 = np.array([1, 0], dtype=np.int64)
    a3 = np.array([0], dtype=np.int64)

    r1 = np.array([1.0, 2.0], dtype=np.float32)
    r2 = np.array([3.0, 4.0], dtype=np.float32)
    r3 = np.array([5.0], dtype=np.float32)

    # step t
    buffer.batch_push(
        (node_map1, s),
        (node_map1, a1),
        (node_map1, r1),
        (node_map2, s2)
    )

    print("buffer len after step1:", len(buffer))

    # step t+1
    buffer.batch_push(
        (node_map2, s2),
        (node_map2, a2),
        (node_map2, r2),
        (node_map3, s3)
    )

    print("buffer len after step2:", len(buffer))

    # step t+2
    buffer.batch_push(
        (node_map3, s3),
        (node_map3, a3),
        (node_map3, r3),
        (node_map3, s3)
    )

    print("buffer len after step3:", len(buffer))

    sample = buffer.sample(2)
    print(sample)
    print(list(zip(*sample)))