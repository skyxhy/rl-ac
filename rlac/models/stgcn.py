"""Masked temporal ST-GNN for sequence-level misbehavior detection."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class MaskedTemporalSTGNN(nn.Module):
    def __init__(self, in_dim, hidden=12, heads=2, dropout=0.35):
        super().__init__()
        self.dropout = dropout
        self.hdim = hidden * heads
        self.embed_dim = self.hdim + 1
        self.spatial1 = TransformerConv(in_dim, hidden, heads=heads, edge_dim=1, dropout=dropout, beta=True)
        self.bn1 = nn.BatchNorm1d(self.hdim)
        self.skip1 = nn.Linear(in_dim, self.hdim)
        self.gru_cell = nn.GRUCell(self.hdim, self.hdim)
        self.time_attn = nn.Linear(self.hdim, 1)
        self.cls = nn.Sequential(nn.Linear(self.hdim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.risk_head = nn.Sequential(nn.Linear(self.hdim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def encode_step(self, x_t, ei, ew):
        ea = ew if ew.dim() == 2 else ew.unsqueeze(-1)
        h = self.spatial1(x_t, ei, ea)
        h = F.relu(self.bn1(h) + self.skip1(x_t))
        return F.dropout(h, p=self.dropout, training=self.training)

    def forward(self, sample, return_state=False):
        x_seq = sample["x_seq"]
        dev = x_seq.device
        mask = sample["active_mask_seq"].to(dev)
        T, N, _ = x_seq.shape
        hidden = torch.zeros((N, self.hdim), device=dev)
        traj = []
        for t in range(T):
            sp = self.encode_step(x_seq[t], sample["edge_index_seq"][t].to(dev), sample["edge_weight_seq"][t].to(dev))
            active = mask[t].float().unsqueeze(-1)
            hidden = active * self.gru_cell(sp, hidden) + (1.0 - active) * hidden
            traj.append(hidden)
        traj = torch.stack(traj, 0)
        score = self.time_attn(traj).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, 0) * mask.float()
        alpha = alpha / (alpha.sum(0, keepdim=True) + 1e-6)
        ctx = (alpha.unsqueeze(-1) * traj).sum(0)
        logits = self.cls(ctx)
        risk = self.risk_head(ctx)
        state = torch.cat([ctx, risk], -1)
        return (logits, risk, state) if return_state else logits
