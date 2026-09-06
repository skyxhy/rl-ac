"""TransformerConv GNN risk encoder/classifier."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden=16, heads=2, dropout=0.35):
        super().__init__()
        self.dropout = dropout
        self.hidden = hidden
        self.heads = heads
        self.embed_dim = hidden * heads
        hdim = hidden * heads

        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, edge_dim=1, dropout=dropout, beta=True)
        self.bn1 = nn.BatchNorm1d(hdim)
        self.skip1 = nn.Linear(in_dim, hdim)
        self.conv2 = TransformerConv(hdim, hidden, heads=heads, edge_dim=1, dropout=dropout, beta=True)
        self.bn2 = nn.BatchNorm1d(hdim)
        self.skip2 = nn.Linear(hdim, hdim)
        self.mlp = nn.Sequential(
            nn.Linear(hdim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def embed(self, x, edge_index, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = F.relu(self.bn1(x1) + self.skip1(x))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = F.relu(self.bn2(x2) + self.skip2(x1))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        return x2

    def forward(self, x, edge_index, edge_attr):
        return self.mlp(self.embed(x, edge_index, edge_attr))

    def out_head(self, embedding):
        return self.mlp(embedding)
