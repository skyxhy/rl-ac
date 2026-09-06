"""Risk embedding wrapper around a frozen encoder."""
from __future__ import annotations

import torch

from rlac.env.graph_module import Snapshot


class RiskManager:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        # we concatenate node embedding with the graph-mean vector -> 2x node dim
        node_dim = getattr(model, "embed_dim", model.hidden * model.heads)
        self.embed_dim = int(node_dim) * 2

    def embed(self, snap: Snapshot) -> torch.Tensor:
        with torch.no_grad():
            x = snap.feature.to(self.device)
            ei = snap.edge_index.to(self.device)
            ew = snap.edge_weight.to(self.device)
            node = self.model.embed(x, ei, ew)
            graph = node.mean(dim=0, keepdim=True)
            return torch.cat([node, graph.expand(node.shape[0], -1)], dim=1)
