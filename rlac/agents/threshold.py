"""Threshold defender: reject when the frozen GNN risk score >= threshold."""
from __future__ import annotations

import numpy as np
import torch

from rlac.models.gnn import GNNModel


class GNNThresholdAgent:
    def __init__(self, gnn_model: GNNModel, threshold: float = 0.5, action_dim: int = 3, device: str = "cpu"):
        self.gnn = gnn_model.to(device)
        self.gnn.eval()
        self.threshold = threshold
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.defensive_action = action_dim - 1
        self.epsilon = 0

    def eval(self):
        self.gnn.eval()

    def take_action(self, state):
        if state is None:
            return {}
        node_map, X = state
        # X: (N, sight, embed); last-step embedding's node-half carries the risk signal
        x = torch.tensor(X[:, -1, :self.gnn.hidden * self.gnn.heads], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.gnn.out_head(x)).squeeze(-1).cpu().numpy()
        acts = np.where(probs < self.threshold, 0, self.defensive_action)
        return node_map, acts
