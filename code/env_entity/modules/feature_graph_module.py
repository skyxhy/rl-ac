from ...data.feature_utils import GraphFeatureBuilder
from dataclasses import dataclass
import torch
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Snapshot:
    node_map: Dict[int, int] = None
    feature: torch.Tensor = None
    label: torch.Tensor = None
    edge_index: torch.Tensor = None
    edge_weight: torch.Tensor = None


class Graph_Module:
    def __init__(self, feature_names: List[str], scaler: Optional[StandardScaler] = None):
        self.graph_feature_builder = GraphFeatureBuilder(feature_names)

        self.feature_names = feature_names
        self.feature_dim = len(feature_names)

        # ⭐ 统一 scaler（外部 fit）
        self.scaler = scaler

    def set_scaler(self, scaler: StandardScaler):
        self.scaler = scaler

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return x
        return self.scaler.transform(x)

    def build_snapshot(self, window_df: pd.DataFrame) -> Snapshot:
        _, _, snapshot_dict = self.graph_feature_builder.build(window_df)

        nodes = sorted(snapshot_dict["node_map"].items(), key=lambda x: x[1])

        # =========================
        # feature matrix
        # =========================
        raw_features = np.stack([
            snapshot_dict["feat_map"][nid].numpy()
            for nid, _ in nodes
        ])

        scaled_features = self._scale(raw_features)

        snapshot = Snapshot(
            node_map=snapshot_dict["node_map"],
            edge_index=snapshot_dict["edge_index"].detach().clone().long(),
            edge_weight=snapshot_dict["edge_weight"].detach().clone().float(),
            feature=torch.tensor(scaled_features, dtype=torch.float32),
            label=torch.tensor(
                [snapshot_dict["label_map"][nid] for nid, _ in nodes],
                dtype=torch.long
            )
        )

        return snapshot