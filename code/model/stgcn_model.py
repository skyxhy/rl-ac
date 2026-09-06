import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class MaskedTemporalSTGNN(nn.Module):
    def __init__(self, in_dim, hidden=12, heads=2, dropout=0.35):
        super().__init__()
        self.dropout = dropout
        self.hdim = hidden * heads

        self.spatial1 = TransformerConv(
            in_channels=in_dim,
            out_channels=hidden,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
            beta=True,
        )

        self.bn1 = nn.BatchNorm1d(self.hdim)
        self.skip1 = nn.Linear(in_dim, self.hdim)

        self.gru_cell = nn.GRUCell(self.hdim, self.hdim)
        self.time_attn = nn.Linear(self.hdim, 1)

        # 分类头
        self.cls = nn.Sequential(
            nn.Linear(self.hdim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # 风险感知头
        self.risk_head = nn.Sequential(
            nn.Linear(self.hdim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def encode_step(self, x_t, edge_index_t, edge_weight_t):
        edge_attr_t = (
            edge_weight_t
            if edge_weight_t.dim() == 2
            else edge_weight_t.unsqueeze(-1)
        )

        h = self.spatial1(x_t, edge_index_t, edge_attr_t)
        h = self.bn1(h)
        h = F.relu(h + self.skip1(x_t))
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def forward(self, sample, return_state=False):
        x_seq = sample["x_seq"]
        device = x_seq.device

        mask_seq = sample["active_mask_seq"].to(device)
        edge_index_seq = sample["edge_index_seq"]
        edge_weight_seq = sample["edge_weight_seq"]

        T, N, _ = x_seq.shape
        hidden = torch.zeros((N, self.hdim), device=device)

        traj = []

        for t in range(T):
            spatial = self.encode_step(
                x_seq[t],
                edge_index_seq[t].to(device),
                edge_weight_seq[t].to(device),
            )

            active = mask_seq[t].float().unsqueeze(-1)

            hidden_new = self.gru_cell(spatial, hidden)
            hidden = active * hidden_new + (1.0 - active) * hidden

            traj.append(hidden)

        traj = torch.stack(traj, dim=0)  # [T, N, H]

        score = self.time_attn(traj).squeeze(-1)
        score = score.masked_fill(~mask_seq, -1e9)

        alpha = torch.softmax(score, dim=0)
        alpha = alpha * mask_seq.float()
        alpha = alpha / (alpha.sum(dim=0, keepdim=True) + 1e-6)

        temporal_ctx = (alpha.unsqueeze(-1) * traj).sum(dim=0)  # [N,H]

        # 分类输出
        logits = self.cls(temporal_ctx)

        # 风险分数
        risk_score = self.risk_head(temporal_ctx)

        # RL状态表示：隐藏层 + 风险分数
        state = torch.cat([temporal_ctx, risk_score], dim=-1)

        if return_state:
            return logits, risk_score, state

        return logits