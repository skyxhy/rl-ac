"""MLP baseline (node-feature only, no graph structure)."""
from __future__ import annotations

import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed_dim = hidden_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def embed(self, x):
        return self.network[:-1](x)
