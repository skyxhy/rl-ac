import pandas as pd
import torch
import os
import argparse
from typing import List, Dict, Optional

from .feature_config import FeatureConfig
from .feature_utils import GraphFeatureBuilder
from ..env_entity.modules.data_module import DataModule


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="output.csv")

    parser.add_argument("--start_time", type=int, default=31000)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--num_windows", type=int, default=100)

    parser.add_argument(
        "--attack_intensity_list",
        type=float,
        nargs="+",
        default=[1.0],
        help="e.g. 1.0 0.8 0.5"
    )

    parser.add_argument("--out_dir", type=str, default="data/")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # =========================
    # 1. 初始化模块
    # =========================
    cfg = FeatureConfig(window=args.window)

    data_module = DataModule(args.data_path)
    df = data_module.df.sort_values("time").reset_index(drop=True)

    builder = GraphFeatureBuilder(cfg.feature_names)

    graphs = []
    all_rows = []
    graph_metadata = []

    # =========================
    # 2. 时间窗口生成
    # =========================
    if args.start_time is None:
        t_start = int(df["time"].min())
    else:
        t_start = args.start_time

    t_end = int(df["time"].max())

    window_starts = list(
        range(t_start, t_end - args.window + 1, args.step)
    )[:args.num_windows]

    print(f"Base windows: {len(window_starts)}")
    print(f"Attack intensities: {args.attack_intensity_list}")

    global_idx = 0

    # =========================
    # 3. 核心：攻击强度 × 窗口
    # =========================
    for intensity in args.attack_intensity_list:

        print(f"\n🚀 Processing intensity = {intensity}")

        for t0 in window_starts:
            window_df = data_module.read_data(
                attack_intensity=float(intensity),
                sender_intensity_map=None,
                start_time=t0,
                end_time=t0 + args.window
            )

            if window_df.empty:
                continue

            out = builder.build(window_df, window_idx=global_idx)
            if out is None:
                continue

            data, rows, metadata = out

            # 记录额外信息（关键）
            metadata["attack_intensity"] = intensity
            metadata["start_time"] = t0

            graphs.append(data)
            all_rows.extend(rows)
            graph_metadata.append(metadata)

            global_idx += 1

    # =========================
    # 4. 保存
    # =========================
    torch.save(graphs, os.path.join(args.out_dir, "graphs.pt"))
    torch.save(graph_metadata, os.path.join(args.out_dir, "graph_metadata.pt"))

    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.out_dir, "samples.csv"),
        index=False
    )

    with open(os.path.join(args.out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cfg.feature_names))

    print("\n==============================")
    print(f"✅ Total graphs: {len(graphs)}")
    print(f"✅ Total samples: {len(all_rows)}")
    print(f"✅ Expected graphs: {len(window_starts) * len(args.attack_intensity_list)}")
    print("==============================")


if __name__ == "__main__":
    main()