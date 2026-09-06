"""Dataset builders (graphs.pt / samples.csv for the GNN encoder).

Ported from ``code/data/build_gnn_dataset.py`` onto the ``rlac`` stack.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd
import torch

from rlac.data.builder import GraphFeatureBuilder
from rlac.data.features import FeatureConfig
from rlac.env.data_module import DataModule


def build_graph_dataset(
    data_path: str,
    out_dir: str = "data",
    start_time: int = 31000,
    window: int = 1,
    step: int = 10,
    num_windows: int = 100,
    intensities: Optional[List[float]] = None,
    seed: int = 42,
) -> int:
    intensities = intensities or [1.0]
    os.makedirs(out_dir, exist_ok=True)
    feat_cfg = FeatureConfig(window=window)
    dm = DataModule(data_path, subsample_seed=seed)
    df = dm.df.sort_values("time").reset_index(drop=True)

    builder = GraphFeatureBuilder(feat_cfg.feature_names)
    graphs, rows, meta = [], [], []
    t_end = int(df["time"].max())
    starts = list(range(start_time, t_end - window + 1, step))[:num_windows]

    gidx = 0
    for intensity in intensities:
        for t0 in starts:
            wdf = dm.read_data(attack_intensity=float(intensity), start_time=t0, end_time=t0 + window)
            if wdf.empty:
                continue
            out = builder.build(wdf, window_idx=gidx)
            if out is None:
                continue
            data, sample_rows, md = out
            md["attack_intensity"] = intensity
            md["start_time"] = t0
            graphs.append(data)
            rows.extend(sample_rows)
            meta.append(md)
            gidx += 1

    torch.save(graphs, os.path.join(out_dir, "graphs.pt"))
    torch.save(meta, os.path.join(out_dir, "graph_metadata.pt"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "samples.csv"), index=False)
    with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feat_cfg.feature_names))
    return len(graphs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="output.csv")
    ap.add_argument("--start_time", type=int, default=31000)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--num_windows", type=int, default=100)
    ap.add_argument("--attack_intensity_list", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out_dir", default="data/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = build_graph_dataset(
        args.data_path, args.out_dir, args.start_time, args.window,
        args.step, args.num_windows, args.attack_intensity_list, args.seed,
    )
    print(f"saved {n} graphs to {args.out_dir}")


if __name__ == "__main__":
    main()
