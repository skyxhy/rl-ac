import pandas as pd
import torch
import numpy as np
import os
from .feature_config import FeatureConfig
from .feature_utils import GraphFeatureBuilder, build_sequence_sample
from tqdm import tqdm


def main():
    cfg = FeatureConfig(
        window=1, step=10, num_windows=200, start_offset=100,
        time_dep_len=5
    )
    np.random.seed(42)
    torch.manual_seed(42)

    df = pd.read_csv("output.csv").sort_values("time").reset_index(drop=True)
    builder = GraphFeatureBuilder(cfg.feature_names)
    snapshots = []

    t_start = int(df["time"].min()) + cfg.start_offset
    t_end = int(df["time"].max())
    window_starts = list(range(t_start, t_end - cfg.window + 1, cfg.step))[:cfg.num_windows]

    print("Building single window snapshots...")
    with tqdm(total=len(window_starts)) as pbar:
        for idx, t0 in enumerate(window_starts):
            window_df = df[(df["time"] >= t0) & (df["time"] < t0 + cfg.window)].copy()
            out = builder.build(window_df, window_idx=idx)
            # 🔑 关键：原代码即使窗口为空/单节点也保留占位，此处保持兼容
            if out is not None:
                snapshots.append(out[2])
            else:
                # 若 builder 过滤了 <2 节点的窗口，需插入空占位符维持时间轴对齐
                snapshots.append({
                    "window_idx": idx, "node_ids": [], "feat_map": {}, "label_map": {},
                    "edge_index": torch.empty((2, 0), dtype=torch.long),
                    "edge_weight": torch.empty((0,), dtype=torch.float)
                })
            pbar.update(1)

    print(f"Total snapshots: {len(snapshots)} (Aligned with original timeline)")

    # 2. 滑动窗口构建序列
    samples = []
    T = cfg.time_dep_len
    print(f"Building sequences with seq_len={T}...")
    with tqdm(total=len(snapshots) - T + 1) as pbar:
        for i in range(T - 1, len(snapshots)):
            seq_slice = snapshots[i - T + 1 : i + 1]
            sample = build_sequence_sample(seq_slice, cfg.feature_names)
            if sample is not None:
                samples.append(sample)
            pbar.update(1)

    OUT_PATH = "data/stgcn_samples_masked.pt"
    FEATURE_NAME_PATH = "data/stgcn_feature_names.txt"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(samples, OUT_PATH)
    with open(FEATURE_NAME_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(cfg.feature_names))
    print(f"Saved {len(samples)} sequence samples (Original logic equivalent)")

if __name__ == "__main__":
    main()