"""Scaled per-window graph snapshots consumed by the GNN risk encoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from rlac.data.builder import GraphFeatureBuilder


@dataclass
class Snapshot:
    node_map: Optional[Dict[int, int]] = None
    feature: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_weight: Optional[torch.Tensor] = None


class GraphModule:
    def __init__(self, feature_names: List[str], scaler: Optional[StandardScaler] = None):
        self.builder = GraphFeatureBuilder(feature_names)
        self.feature_names = feature_names
        self.feature_dim = len(feature_names)
        self.scaler = scaler

    def build_snapshot(self, window_df: pd.DataFrame) -> Snapshot:
        _, _, snap = self.builder.build(window_df)
        if snap is None:
            return Snapshot()
        nodes = sorted(snap["node_map"].items(), key=lambda x: x[1])
        raw = np.stack([snap["feat_map"][nid].numpy() for nid, _ in nodes])
        x = self.scaler.transform(raw) if self.scaler is not None else raw
        return Snapshot(
            node_map=snap["node_map"],
            edge_index=snap["edge_index"].long(),
            edge_weight=snap["edge_weight"].float(),
            feature=torch.tensor(x, dtype=torch.float32),
            label=torch.tensor([snap["label_map"][nid] for nid, _ in nodes], dtype=torch.long),
        )
