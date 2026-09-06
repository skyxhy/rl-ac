from ...model.gnn_model import GNN_model
import torch
from .feature_graph_module import Snapshot
from typing import List, Dict
import numpy as np
class Risk_Manager:
    def __init__(self, gnn_model: GNN_model):
        self.model = gnn_model
        self.model.eval()
        self.embed_dim = self.model.hidden*self.model.heads*2#*0+1
    def embed(self, feature_graph:Snapshot)->np.ndarray:
        feature_graph = feature_graph
        with torch.no_grad():
            x,edge_index,edge_weight = feature_graph.feature, feature_graph.edge_index, feature_graph.edge_weight
            embed_of_nodes = self.model.embed(x, edge_index, edge_weight)
            embed_of_graph = embed_of_nodes.mean(dim=0)
            embed_of_nodes_graph = torch.cat([embed_of_nodes, embed_of_graph.unsqueeze(0).repeat(embed_of_nodes.shape[0],1)], dim=1)
            # score = self.model(x, edge_index, edge_weight)
        return embed_of_nodes_graph.numpy()
        