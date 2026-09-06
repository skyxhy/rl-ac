"""Per-window graph-snapshot builder.

Ported from ``code/data/feature_utils.py`` with the following clean-ups:
* node groups fetched once per window and reused (no triple ``get_group``);
* ``list.index`` O(V) lookups replaced by a position dict;
* label computation performed once and reused for ``label_map``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class GraphFeatureBuilder:
    def __init__(self, feature_names: list):
        self.feature_names = feature_names
        self._cache: Dict[str, object] = {}
        self._df: Optional[pd.DataFrame] = None
        self._groups: Dict[object, pd.DataFrame] = {}
        self._node_ids: List[object] = []
        self._pos: Dict[object, int] = {}
        self._feature_funcs = {
            "mean_x": self._calc_mean_x,
            "mean_y": self._calc_mean_y,
            "std_x": self._calc_std_x,
            "std_y": self._calc_std_y,
            "msg_count": self._calc_msg_count,
            "uniq_recv": self._calc_uniq_recv,
            "recv_entropy": self._calc_recv_entropy,
            "active_span": self._calc_active_span,
            "gap_mean": self._calc_gap_mean,
            "gap_std": self._calc_gap_std,
            "gap_cv": self._calc_gap_cv,
            "radius_mean": self._calc_radius_mean,
            "sync_ratio": self._calc_sync_ratio,
            "shared_recv_ratio": self._calc_shared_recv_ratio,
            "mean_recv_pop": self._calc_mean_recv_pop,
            "max_recv_pop": self._calc_max_recv_pop,
            "deg": self._calc_deg,
            "wdeg": self._calc_wdeg,
        }

    # ---------- lazy intermediate values ----------
    def _get(self, key, compute_fn):
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def _group_by_pseudo(self):
        return self._df.groupby("sender_pseudo", sort=False)

    def _calc_recv_to_senders(self):
        d = defaultdict(set)
        for row in self._df.itertuples(index=False):
            d[row.receiver_id].add(row.sender_pseudo)
        return d

    def _calc_recv_sets(self):
        recv_sets = {}
        for nid, g in self._groups.items():
            recv_sets[nid] = set(g["receiver_id"].tolist())
        return recv_sets

    def _calc_edges(self):
        pairs, weights = [], []
        recv_sets = self._get("recv_sets", self._calc_recv_sets)
        node_ids = self._node_ids
        n = len(node_ids)
        for i in range(n):
            for j in range(i + 1, n):
                common = len(recv_sets[node_ids[i]] & recv_sets[node_ids[j]])
                if common == 0:
                    continue
                w = common / (len(recv_sets[node_ids[i]] | recv_sets[node_ids[j]]) + 1e-6)
                if w <= 0:
                    continue
                pairs.extend([(i, j), (j, i)])
                weights.extend([w, w])
        for i in range(n):
            pairs.append((i, i))
            weights.append(1.0)
        ei = np.array(pairs).T if pairs else np.empty((2, 0), dtype=int)
        ew = np.array(weights, dtype=float)
        return ei, ew

    def _calc_degrees(self):
        ei, ew = self._get("edges", self._calc_edges)
        deg = np.zeros(len(self._node_ids), dtype=float)
        wdeg = np.zeros_like(deg)
        for src, dst, w in zip(ei[0], ei[1], ew):
            deg[src] += 1.0
            wdeg[src] += w
        return deg, wdeg

    def _calc_time_to_senders(self):
        d = defaultdict(set)
        for row in self._df.itertuples(index=False):
            d[row.time].add(row.sender_pseudo)
        return d

    # ---------- per-node features ----------
    def _calc_mean_x(self, p, g): return float(g["pos_x"].mean())
    def _calc_mean_y(self, p, g): return float(g["pos_y"].mean())
    def _calc_std_x(self, p, g): return float(g["pos_x"].std()) if len(g) > 1 else 0.0
    def _calc_std_y(self, p, g): return float(g["pos_y"].std()) if len(g) > 1 else 0.0
    def _calc_msg_count(self, p, g): return float(len(g))
    def _calc_uniq_recv(self, p, g): return float(len(set(g["receiver_id"].tolist())))

    def _calc_recv_entropy(self, p, g):
        items = g["receiver_id"].tolist()
        if not items:
            return 0.0
        c = Counter(items)
        p_arr = np.array(list(c.values()), dtype=np.float64) / len(items)
        return float(-(p_arr * np.log(p_arr + 1e-12)).sum())

    def _calc_active_span(self, p, g):
        t = g["time"].to_numpy(dtype=float)
        return float(t.max() - t.min()) if len(t) > 1 else 0.0

    def _gap_arr(self, g) -> np.ndarray:
        t = np.sort(g["time"].to_numpy(dtype=float))
        return np.diff(t) if len(t) > 1 else np.array([])

    def _calc_gap_mean(self, p, g):
        gaps = self._gap_arr(g)
        return float(gaps.mean()) if len(gaps) else 0.0

    def _calc_gap_std(self, p, g):
        gaps = self._gap_arr(g)
        return float(gaps.std()) if len(gaps) else 0.0

    def _calc_gap_cv(self, p, g):
        m = self._calc_gap_mean(p, g)
        s = self._calc_gap_std(p, g)
        return float(s / (m + 1e-6))

    def _calc_radius_mean(self, p, g):
        pos = g[["pos_x", "pos_y"]].to_numpy(dtype=float)
        if len(pos) <= 1:
            return 0.0
        center = pos.mean(axis=0)
        return float(np.linalg.norm(pos - center, axis=1).mean())

    def _calc_sync_ratio(self, p, g):
        times = g["time"].to_numpy(dtype=float)
        if len(times) == 0:
            return 0.0
        time_to_senders = self._get("time_to_senders", self._calc_time_to_senders)
        return float(np.mean([1.0 if len(time_to_senders[t]) > 1 else 0.0 for t in times]))

    def _calc_shared_recv_ratio(self, p, g):
        recv_sets = self._get("recv_sets", self._calc_recv_sets)
        r2s = self._get("recv_to_senders", self._calc_recv_to_senders)
        my_recv = recv_sets[p]
        if not my_recv:
            return 0.0
        return float(np.mean([1.0 if len(r2s[r]) > 1 else 0.0 for r in my_recv]))

    def _calc_mean_recv_pop(self, p, g):
        recv_sets = self._get("recv_sets", self._calc_recv_sets)
        r2s = self._get("recv_to_senders", self._calc_recv_to_senders)
        pops = [len(r2s[r]) for r in recv_sets[p]]
        return float(np.mean(pops)) if pops else 0.0

    def _calc_max_recv_pop(self, p, g):
        recv_sets = self._get("recv_sets", self._calc_recv_sets)
        r2s = self._get("recv_to_senders", self._calc_recv_to_senders)
        pops = [len(r2s[r]) for r in recv_sets[p]]
        return float(max(pops)) if pops else 0.0

    def _calc_deg(self, p, g):
        deg, _ = self._get("degrees", self._calc_degrees)
        return float(deg[self._pos[p]])

    def _calc_wdeg(self, p, g):
        _, wdeg = self._get("degrees", self._calc_degrees)
        return float(wdeg[self._pos[p]])

    # ---------- build ----------
    def build(self, window_df: pd.DataFrame, window_idx: int = None):
        if window_df is None or window_df.empty:
            return None
        self._cache = {}
        self._df = window_df
        self._groups = {pseudo: g for pseudo, g in self._df.groupby("sender_pseudo", sort=False)}
        self._node_ids = sorted(self._groups.keys())
        self._pos = {nid: i for i, nid in enumerate(self._node_ids)}
        if len(self._node_ids) < 2:
            return None

        node_map = dict(self._pos)
        feats, labels, rows = [], [], []
        for pseudo in self._node_ids:
            g = self._groups[pseudo]
            label = int(g["attack_type"].max())
            row = {"window_idx": window_idx, "sender_pseudo": pseudo, "label": label}
            feat_vec = []
            for name in self.feature_names:
                val = self._feature_funcs[name](pseudo, g)
                feat_vec.append(val)
                row[name] = val
            feats.append(feat_vec)
            labels.append(label)
            rows.append(row)

        ei, ew = self._get("edges", self._calc_edges)
        edge_index = torch.tensor(ei, dtype=torch.long).contiguous()
        edge_weight = torch.tensor(ew, dtype=torch.float).view(-1) if ew.ndim == 1 else torch.tensor(ew, dtype=torch.float)
        data = Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_weight.view(-1, 1),
            y=torch.tensor(labels, dtype=torch.long),
        )
        snapshot_dict = {
            "window_idx": window_idx,
            "node_ids": self._node_ids,
            "node_map": node_map,
            "feat_map": {nid: torch.tensor(f, dtype=torch.float) for nid, f in zip(self._node_ids, feats)},
            "label_map": {nid: lab for nid, lab in zip(self._node_ids, labels)},
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }
        return data, rows, snapshot_dict


def build_sequence_sample(seq_snapshots: list, feature_names: list):
    """Build a temporal sequence sample for ST-GCN style encoders."""
    if not seq_snapshots:
        return None
    node_id_set = set()
    for snap in seq_snapshots:
        node_id_set.update(snap["node_ids"])
    node_ids = sorted(node_id_set)
    if not node_ids:
        return None
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    N, F, T = len(node_ids), len(feature_names), len(seq_snapshots)
    x_seq = torch.zeros((T, N, F), dtype=torch.float)
    active_mask_seq = torch.zeros((T, N), dtype=torch.bool)
    edge_index_seq, edge_weight_seq = [], []

    for t, snap in enumerate(seq_snapshots):
        for nid in snap["node_ids"]:
            idx = node_map[nid]
            x_seq[t, idx] = torch.as_tensor(snap["feat_map"][nid], dtype=torch.float).contiguous()
            active_mask_seq[t, idx] = True
        ei, ew = snap["edge_index"], snap["edge_weight"]
        if ei.numel() == 0:
            edge_index_seq.append(torch.empty((2, 0), dtype=torch.long))
            edge_weight_seq.append(torch.empty((0,), dtype=torch.float))
        else:
            src_new, dst_new, w_new = [], [], []
            for s_old, d_old, w in zip(ei[0].tolist(), ei[1].tolist(), ew.tolist()):
                s_nid, d_nid = snap["node_ids"][s_old], snap["node_ids"][d_old]
                if s_nid in node_map and d_nid in node_map:
                    src_new.append(node_map[s_nid])
                    dst_new.append(node_map[d_nid])
                    w_new.append(float(w))
            if not src_new:
                edge_index_seq.append(torch.empty((2, 0), dtype=torch.long))
                edge_weight_seq.append(torch.empty((0,), dtype=torch.float))
            else:
                edge_index_seq.append(torch.tensor([src_new, dst_new], dtype=torch.long).contiguous())
                edge_weight_seq.append(torch.tensor(w_new, dtype=torch.float).contiguous())

    target_snap = seq_snapshots[-1]
    y = torch.full((N,), -1, dtype=torch.long)
    target_mask = torch.zeros((N,), dtype=torch.bool)
    for nid in target_snap["node_ids"]:
        if nid in node_map:
            idx = node_map[nid]
            y[idx] = int(target_snap["label_map"][nid])
            target_mask[idx] = True
    if target_mask.sum().item() == 0:
        return None

    return {
        "seq_len": T,
        "target_window_idx": target_snap["window_idx"],
        "seq_window_idxs": list(range(target_snap["window_idx"] - T + 1, target_snap["window_idx"] + 1)),
        "node_ids": node_ids,
        "x_seq": x_seq,
        "active_mask_seq": active_mask_seq,
        "edge_index_seq": edge_index_seq,
        "edge_weight_seq": edge_weight_seq,
        "y": y,
        "target_mask": target_mask,
    }
