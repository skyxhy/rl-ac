"""Contextual-bandit baseline: LSTM net trained with single-step supervised
regression (no TD target, no bootstrapping)."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


class CBNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, num_actions=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CBAgent:
    def __init__(
        self,
        feature_dim,
        action_dim=2,
        hidden_dim=256,
        lr=1e-3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_net = CBNet(feature_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def eval(self):
        self.q_net.eval()

    def take_action(self, state):
        if state is None:
            return {}
        node_map, X = state
        q = self.q_net(torch.tensor(X, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
        greedy = q.argmax(1)
        acts = np.where(np.random.rand(len(node_map)) < self.epsilon,
                        np.random.randint(self.action_dim, size=len(node_map)), greedy)
        return node_map, acts

    def update(self, bs, ba, br, bn=None):
        bs = torch.tensor(np.asarray(bs), dtype=torch.float32, device=self.device)
        ba = torch.tensor(np.asarray(ba), dtype=torch.long, device=self.device).view(-1)
        br = torch.tensor(np.asarray(br), dtype=torch.float32, device=self.device).view(-1)
        q = self.q_net(bs)
        q_sa = q.gather(1, ba.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_sa, br)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state_dict": self.q_net.state_dict(), "epsilon": self.epsilon}, path)
