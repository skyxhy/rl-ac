"""DRQN agent: LSTM Q-network + n-step (2) delayed replay."""
from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from rlac.agents.replay import ReplayBuffer


class DRQNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, num_actions=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):  # x: (B, T, F) -> (B, A)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class DRQNAgent:
    def __init__(
        self,
        feature_dim,
        action_dim=2,
        hidden_dim=256,
        lr=1e-3,
        gamma=0.9,
        n_step=2,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        sync_freq=10,
        device="cpu",
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

        self.q_net = DRQNNet(feature_dim, hidden_dim, action_dim).to(self.device)
        self.target_net = deepcopy(self.q_net).to(self.device)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def eval(self):
        self.q_net.eval()
        self.target_net.eval()

    def take_action(self, state):
        if state is None:
            return {}
        node_map, X = state
        q = self.q_net(torch.tensor(X, dtype=torch.float32, device=self.device))
        q = q.detach().cpu().numpy()
        greedy = q.argmax(1)
        acts = np.where(np.random.rand(len(node_map)) < self.epsilon,
                        np.random.randint(self.action_dim, size=len(node_map)), greedy)
        return node_map, acts

    def update(self, bs, ba, br, bn):
        bs = torch.tensor(np.asarray(bs), dtype=torch.float32, device=self.device)
        ba = torch.tensor(np.asarray(ba), dtype=torch.long, device=self.device).view(-1)
        br = torch.tensor(np.asarray(br), dtype=torch.float32, device=self.device).view(-1)
        bn = torch.tensor(np.asarray(bn), dtype=torch.float32, device=self.device)

        q = self.q_net(bs)
        q_sa = q.gather(1, ba.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target_net(bn)
            nq = nq.gather(1, self.q_net(bn).argmax(1).unsqueeze(1)).squeeze(1)
            target = br + (self.gamma ** self.n_step) * nq
        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        self.update_step += 1
        if self.update_step % self.sync_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.q_net.state_dict(),
            "target_model_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_step": self.update_step,
            "gamma": self.gamma,
            "n_step": self.n_step,
        }, path)

    def load(self, path: str, device: str = None):
        dev = torch.device(device or self.device)
        ckpt = torch.load(path, map_location=dev)
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.load_state_dict(ckpt["target_model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.update_step = ckpt.get("update_step", 0)
        self.gamma = ckpt.get("gamma", self.gamma)
        self.n_step = ckpt.get("n_step", self.n_step)
        self.q_net.to(dev)
        self.target_net.to(dev)
        self.device = dev
