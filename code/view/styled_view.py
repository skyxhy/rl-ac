# -*- coding: utf-8 -*-
"""
毕业设计研究绘图脚本
- 保留原有绘图/分析函数
- 统一期刊风格
- 所有图保存到 ../../figure/
"""

import os
import sys
import json
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import joblib


# =========================================================
# 全局路径
# =========================================================
# 以本文件位置推导仓库根（data_view/），不再依赖 cwd / 硬编码 Linux 路径
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_run_dir_env = os.environ.get("SIMULATION_OUTPUT_DIR")
run_dir = _run_dir_env or os.path.join(_REPO_ROOT, "outputs", "simulation")
FIGURE_DIR = os.environ.get("FIGURE_DIR") or os.path.join(_REPO_ROOT, "figure")
os.makedirs(FIGURE_DIR, exist_ok=True)


# =========================================================
# 统一期刊风格
# =========================================================
def set_journal_style():
    """统一成常见期刊/论文绘图风格。"""
    plt.rcParams.update({
        # 版面与输出
        "figure.dpi": 1500,
        "savefig.dpi": 1500,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # 字体
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.unicode_minus": False,

        # 坐标轴
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # 刻度
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # 网格
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,

        # 图例
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",

        # 线条
        "lines.linewidth": 1.8,
        "lines.markersize": 5,

        # PDF/PS 输出
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


set_journal_style()


# =========================================================
# 保存工具
# =========================================================
def save_figure(fig, filename_stem, close=True):
    """
    将图保存到 ../../figure/ 下，同时输出 PNG 和 PDF。
    高清输出
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    png_path = os.path.join(FIGURE_DIR, f"{filename_stem}.png")
    pdf_path = os.path.join(FIGURE_DIR, f"{filename_stem}.pdf")
    svg_path = os.path.join(FIGURE_DIR, f"{filename_stem}.svg")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    if close:
        plt.close(fig)
    return png_path, pdf_path


# =========================================================
# 路径生成函数 (严格对齐新规范)
# =========================================================
# 规范说明：
# - train/: 仅含 a, thr, s, tr, tt, ep, seed  (剔除 ts, t)
# - test/:  仅含 a, thr, s, ts, t, seed        (剔除 tr, tt, ep)
# - 参数映射: ai→a, th→thr, st→ts
# - 路径前缀: "train/" 或 "test/" 作为目录前缀
# =========================================================

def norm_num(x):
    """数值标准化: 0.2→0p2, -1→m1"""
    return str(x).replace('-', 'm').replace('.', 'p')


# ================= Threshold 方法 (纯 Test) =================

def th_fixed_save_name_test(type_, atk, seed=None):
    """
    test/test_th_fixed_{type}_a{atk}_thr0p5_ts{start}_t{time}[_seed{seed}]
    """
    base = (
        f"test/test_th_fixed_{type_}"
        f"_a{norm_num(atk)}"
        f"_thr{norm_num(0.5)}"
        f"_ts{30000}"
        f"_t{100}"
    )
    return f"{base}_seed{seed}" if seed is not None else base


def th_adapt_save_name_test(type_, thr=0.5, seed=None):
    """
    test/test_th_adaptive_{type}_thr{thr}_ts{start}_t{time}[_seed{seed}]
    """
    base = (
        f"test/test_th_adaptive_{type_}"
        f"_thr{norm_num(thr)}"
        f"_ts{30000}"
        f"_t{100}"
    )
    return f"{base}_seed{seed}" if seed is not None else base


# ================= RL Fixed 方法 (Train + Test 分离) =================

def rl_fixed_save_name_train(type_, atk, alpha, sight):
    """
    train/train_rl_fixed_{type}_a{atk}_al{alpha}_s{sight}_tr{start}_tt{time}_ep{ep}
    注意: 学习率参数用 al 区分攻击强度 a，避免冲突
    """
    return (
        f"train/train_rl_fixed_{type_}"
        f"_a{norm_num(atk)}"
        f"_al{norm_num(alpha)}"
        f"_s{sight}"
        f"_tr{30000}"
        f"_tt{100}"
        f"_ep{20}"
    )


def rl_fixed_save_name_test(type_, atk, alpha, sight, seed=None):
    """
    test/test_rl_fixed_{type}_a{atk}_al{alpha}_s{sight}_ts{start}_t{time}[_seed{seed}]
    """
    base = (
        f"test/test_rl_fixed_{type_}"
        f"_a{norm_num(atk)}"
        f"_al{norm_num(alpha)}"
        f"_s{sight}"
        f"_ts{30000}"
        f"_t{100}"
    )
    return f"{base}_seed{seed}" if seed is not None else base


# ================= RL Adaptive 方法 (Train + Test 分离) =================

def rl_adapt_save_name_train(type_, alpha, sight):
    """
    train/train_rl_adaptive_{type}_al{alpha}_s{sight}_tr{start}_tt{time}_ep{ep}
    """
    return (
        f"train/train_rl_adaptive_{type_}"
        f"_al{norm_num(alpha)}"
        f"_s{norm_num(sight)}"
        f"_tr{30000}"
        f"_tt{100}"
        f"_ep{20}"
    )


def rl_adapt_save_name_test(type_, alpha, sight, seed=None):
    """
    test/test_rl_adaptive_{type}_al{alpha}_s{sight}_ts{start}_t{time}[_seed{seed}]
    """
    base = (
        f"test/test_rl_adaptive_{type_}"
        f"_al{norm_num(alpha)}"
        f"_s{norm_num(sight)}"
        f"_ts{30000}"
        f"_t{100}"
    )
    return f"{base}_seed{seed}" if seed is not None else base


# ================= CB Adaptive 方法 (Train + Test 分离) =================

def cb_adapt_save_name_train(type_, alpha, sight):
    """
    train/train_cb_adaptive_{type}_a{alpha}_s{sight}_tr{start}_tt{time}_ep{ep}
    """
    return (
        f"train/train_cb_adaptive_{type_}"
        f"_al{norm_num(alpha)}"
        f"_s{norm_num(sight)}"
        f"_tr{30000}"
        f"_tt{100}"
        f"_ep{20}"
    )


def cb_adapt_save_name_test(type_, alpha, sight, seed=None):
    """
    test/test_cb_adaptive_{type}_a{alpha}_s{sight}_ts{start}_t{time}[_seed{seed}]
    """
    base = (
        f"test/test_cb_adaptive_{type_}"
        f"_al{norm_num(alpha)}"
        f"_s{norm_num(sight)}"
        f"_ts{30000}"
        f"_t{100}"
    )
    return f"{base}_seed{seed}" if seed is not None else base


def load_npy(base_dir, metric_name):
    npy_path = os.path.join(base_dir, "npy", f"{metric_name}.npy")
    data = np.load(npy_path)
    if data is None:
        raise ValueError(f"{npy_path} not found")
    return data

def get_pareto_front(points):
    """
    2D Pareto front for:
      - utility: maximize
      - leakage: minimize

    points: list of (utility, leakage)
    return: sorted front points by utility ascending, suitable for plt.plot
    """
    if not points:
        return []

    # utility 高优先，leakage 低优先
    pts = sorted(points, key=lambda p: (-p[0], p[1]))

    front = []
    best_leak = float("inf")

    for u, l in pts:
        # 只保留当前“最低 leakage”点，形成非支配前沿
        if l < best_leak - 1e-12:
            front.append((u, l))
            best_leak = l

    # 画线时按 utility 从小到大连接
    return sorted(front, key=lambda p: p[0])

def select_seed_pareto_points(seed_points, show_all=True):
    """
    在同一个 alpha / sight 下，从多个 seeds 的点里选支配解。

    参数
    ----
    seed_points : list of (utility, leakage, reward)
    show_all    : True  -> 返回该 alpha 下所有 Pareto 支配解
                  False -> 只返回 Pareto 解里 leakage 最小的那个

    返回
    ----
    selected_points : list of (utility, leakage, reward)
    pareto_ul       : list of (utility, leakage)
    """
    if not seed_points:
        return [], []

    ul_points = [(u, l) for u, l, _ in seed_points]
    pareto_ul = get_pareto_front(ul_points)

    if not pareto_ul:
        return [], []

    def _find_reward(u, l):
        for uu, ll, rr in seed_points:
            if np.isclose(uu, u) and np.isclose(ll, l):
                return rr
        return None

    if show_all:
        selected_points = []
        for u, l in pareto_ul:
            r = _find_reward(u, l)
            selected_points.append((u, l, r))
    else:
        u, l = max(pareto_ul, key=lambda p: p[0])  # leakage 最小
        r = _find_reward(u, l)
        selected_points = [(u, l, r)]

    return selected_points, pareto_ul


# =========================================================
# 图形统一设置补充
# =========================================================
COLOR_THR = "tab:blue"
COLOR_RL = "tab:orange"
COLOR_ATTACK = "tab:red"
COLOR_NORMAL = "tab:gray"


# =========================================================
# Task 1
# =========================================================
def task1_1():
    sight = 10
    alpha = 1
    attack_types = ["dos", "sybil"]

    fig = plt.figure(figsize=(10, 4))
    for i, type_ in enumerate(attack_types):
        th_path = th_adapt_save_name_test(type_, thr=0.5, seed=1)
        rl_path = rl_adapt_save_name_train(type_, alpha=alpha, sight=sight)

        # 仅加载奖励指标
        th_reward = load_npy(os.path.join(run_dir, th_path), "step_reward")
        rl_reward = load_npy(os.path.join(run_dir, rl_path), "episode_return")

        # th step 取平均作为 episode 对齐指标
        th_reward = np.repeat(th_reward.sum(), len(rl_reward))

        # 绘制子图 (左dos, 右sybil)
        ax = plt.subplot(1, 2, i + 1)
        ax.plot(th_reward, label="Threshold Adaptive Test", color=COLOR_THR)
        ax.plot(rl_reward, label="RL Adaptive Train", color=COLOR_RL)
        ax.set_title(f"{type_.upper()} Attack - Return")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

    plt.suptitle("Threshold Test vs RL Train", y=1.02)
    plt.tight_layout()
    save_figure(fig, "task1_1_threshold_vs_rl_train")
    # plt.show()

def task1_3():
    """Fig.3: 攻击强度 Sweep - Threshold Fixed vs RL (RL多seed均值)"""
    sight = 10
    alpha = 1
    attack_intensitys = [0.2, 0.4, 0.6, 0.8, 1, "adp"]
    attack_types = ["dos", "sybil"]

    # RL 采用多个 seed 求均值
    rl_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    metrics_names_step = {
        "utility": "step_utility",
        "leakage": "step_avg_attack_success_count_per_sender",
        "reward": "step_reward"
    }

    def agg(x):
        return np.mean(x)

    def mean_over_seeds(load_fn, paths, metric_key):
        vals = []
        for p in paths:
            arr = load_fn(p, metric_key)
            vals.append(agg(arr))
        return float(np.mean(vals))

    for type_ in attack_types:
        th_utility_list, th_leakage_list, th_reward_list = [], [], []
        rl_utility_list, rl_leakage_list, rl_reward_list = [], [], []

        for atti in attack_intensitys:
            # -------------------------
            # Threshold: 单 seed
            # -------------------------
            if atti == "adp":
                th_path = th_adapt_save_name_test(type_, thr=0.5, seed=1)
                th_full_path = os.path.join(run_dir, th_path)
                th_utility = load_npy(th_full_path, metrics_names_step["utility"])
                th_leakage = load_npy(th_full_path, metrics_names_step["leakage"])
                th_reward = load_npy(th_full_path, metrics_names_step["reward"])
            else:
                th_path = th_fixed_save_name_test(type_, atk=atti, seed=1)
                th_full_path = os.path.join(run_dir, th_path)
                th_utility = load_npy(th_full_path, metrics_names_step["utility"])
                th_leakage = load_npy(th_full_path, metrics_names_step["leakage"])
                th_reward = load_npy(th_full_path, metrics_names_step["reward"])

            th_utility_list.append(agg(th_utility))
            th_leakage_list.append(agg(th_leakage))
            th_reward_list.append(agg(th_reward))

            # -------------------------
            # RL: 多 seed 均值
            # -------------------------
            rl_paths = []
            for seed in rl_seeds:
                if atti == "adp":
                    rl_path = rl_adapt_save_name_test(type_, alpha=alpha, sight=sight, seed=seed)
                else:
                    rl_path = rl_fixed_save_name_test(type_, atk=atti, alpha=alpha, sight=sight, seed=seed)
                rl_paths.append(os.path.join(run_dir, rl_path))

            rl_utility_list.append(mean_over_seeds(load_npy, rl_paths, metrics_names_step["utility"]))
            rl_leakage_list.append(mean_over_seeds(load_npy, rl_paths, metrics_names_step["leakage"]))
            rl_reward_list.append(mean_over_seeds(load_npy, rl_paths, metrics_names_step["reward"]))

        # 绘图：3个子图 (Utility / Leakage / Reward)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
        x = np.arange(len(attack_intensitys))
        width = 0.35

        # ---- Utility ----
        axes[0].bar(x - width / 2, th_utility_list, width, label="Threshold", edgecolor="black", linewidth=0.6)
        axes[0].bar(x + width / 2, rl_utility_list, width, label="RL (mean over seeds)", edgecolor="black", linewidth=0.6)
        axes[0].set_xlabel("Attack Intensity")
        axes[0].set_ylabel("Mean Utility")
        axes[0].set_title(f"{type_.upper()} - Utility")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(attack_intensitys)
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", linestyle="--", alpha=0.35)
        y_min, y_max = min(min(th_utility_list), min(rl_utility_list)), max(max(th_utility_list), max(rl_utility_list))
        delta = (y_max - y_min) * 0.2 if y_max > y_min else 0.05
        axes[0].set_ylim(max(y_min - delta, 0), min(y_max + delta, 1))

        # ---- Leakage ----
        axes[1].bar(x - width / 2, th_leakage_list, width, label="Threshold", edgecolor="black", linewidth=0.6)
        axes[1].bar(x + width / 2, rl_leakage_list, width, label="RL (mean over seeds)", edgecolor="black", linewidth=0.6)
        axes[1].set_xlabel("Attack Intensity")
        axes[1].set_ylabel("Mean Leakage")
        axes[1].set_title(f"{type_.upper()} - Leakage")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(attack_intensitys)
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", linestyle="--", alpha=0.35)
        y_min, y_max = min(min(th_leakage_list), min(rl_leakage_list)), max(max(th_leakage_list), max(rl_leakage_list))
        delta = (y_max - y_min) * 0.2 if y_max > y_min else 0.05
        axes[1].set_ylim(max(y_min - delta, 0), y_max + delta)

        # ---- Reward ----
        axes[2].bar(x - width / 2, th_reward_list, width, label="Threshold", edgecolor="black", linewidth=0.6)
        axes[2].bar(x + width / 2, rl_reward_list, width, label="RL (mean over seeds)", edgecolor="black", linewidth=0.6)
        axes[2].set_xlabel("Attack Intensity")
        axes[2].set_ylabel("Mean Reward")
        axes[2].set_title(f"{type_.upper()} - Reward")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(attack_intensitys)
        axes[2].legend(fontsize=8)
        axes[2].grid(axis="y", linestyle="--", alpha=0.35)
        y_min, y_max = min(min(th_reward_list), min(rl_reward_list)), max(max(th_reward_list), max(rl_reward_list))
        delta = (y_max - y_min) * 0.2 if y_max > y_min else 0.05
        axes[2].set_ylim(max(y_min - delta, 0), y_max + delta)

        plt.suptitle(f"{type_.upper()} Attack: RL vs Threshold", fontsize=12, y=1.02)
        plt.tight_layout()
        # 保存图片高清晰
        save_figure(fig, f"task1_3_{type_}_attack_sweep")
        plt.show()

# =========================================================
# Task 2
# =========================================================
def task2_1():
    sight = 20
    attack_types = ["dos", "sybil"]
    alphas = [0, 0.2, 0.5, 1, 2]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    metrics_names_step = {
        "utility": "step_utility",
        "leakage": "step_avg_attack_success_count_per_sender",
        "reward": "step_reward"
    }

    def find_central_point(points):
        """
        找到距离其他点欧氏距离之和最小的点作为中心点（几何中位数的近似）
        points: list of tuples [(utility, leakage), ...]
        returns: (utility, leakage) of the central point
        """
        if len(points) == 1:
            return points[0]

        min_total_dist = float("inf")
        central_point = points[0]

        for i, (u_i, l_i) in enumerate(points):
            total_dist = sum(
                np.sqrt((u_i - u_j) ** 2 + (l_i - l_j) ** 2)
                for j, (u_j, l_j) in enumerate(points) if i != j
            )
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                central_point = (u_i, l_i)

        return central_point

    for type_ in attack_types:
        rl_utility_list, rl_leakage_list, rl_reward_list = [], [], []

        for alpha in alphas:
            seed_points = []

            for seed in seeds:
                try:
                    rl_path = rl_adapt_save_name_test(type_, alpha=alpha, sight=sight, seed=seed)
                    rl_utility = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["utility"])
                    rl_leakage = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["leakage"])
                    rl_reward = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["reward"])

                    seed_points.append((
                        rl_utility.mean(),
                        rl_leakage.mean(),
                        rl_reward.mean()
                    ))
                except FileNotFoundError:
                    print(f"⚠️  Warning: seed={seed} not found for {type_}, alpha={alpha}")
                    continue

            if not seed_points:
                print(f"❌ No valid data for {type_}, alpha={alpha}")
                continue

            central_util, central_leak = find_central_point([(p[0], p[1]) for p in seed_points])
            central_reward = next(
                p[2] for p in seed_points
                if np.isclose(p[0], central_util) and np.isclose(p[1], central_leak)
            )

            rl_utility_list.append(central_util)
            rl_leakage_list.append(central_leak)
            rl_reward_list.append(central_reward)

        fig = plt.figure(figsize=(4.6, 3.6))
        plt.scatter(
            rl_utility_list,
            rl_leakage_list,
            s=34,
            c=COLOR_THR,
            edgecolors="black",
            linewidth=0.6,
            zorder=3,
            label="Central points"
        )

        # 画 Pareto 前沿线
        pareto_points = get_pareto_front(list(zip(rl_utility_list, rl_leakage_list)))
        if len(pareto_points) >= 2:
            px, py = zip(*pareto_points)
            plt.plot(
                px, py,
                color=COLOR_THR,
                linewidth=1.8,
                linestyle="-",
                alpha=0.9,
                zorder=2,
                label="Pareto front"
            )

        for i, alpha in enumerate(alphas):
            if i < len(rl_utility_list):
                plt.annotate(
                    f"α={alpha:.1f}",
                    (rl_utility_list[i], rl_leakage_list[i]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=8
                )

        plt.title(f"{type_.upper()} Attack - Pareto Front")
        plt.xlabel("Mean Utility")
        plt.ylabel("Mean Leakage")
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=True)
        plt.tight_layout()
        save_figure(fig, f"task2_1_{type_}_pareto_front")
        plt.show()
def task2_2(show_all_pareto=True, annotate_all=False):
    """
    Task 2-2:
    与 task2_1 类似，但每个 alpha 下不再取 central point，
    而是取该 alpha 下多个 seeds 的 Pareto 支配解。

    参数
    ----
    show_all_pareto : bool
        True  -> 一个 alpha 下所有 Pareto 解都画出来
        False -> 只画 leakage 最小的那个 Pareto 解
    annotate_all : bool
        True  -> 对所有选出的点都标注 alpha
        False -> 只标注每个 alpha 的第一个选中点
    """
    sight = 20
    attack_types = ["dos", "sybil"]
    alphas = [0, 0.2, 0.5, 1, 1.5, 2]
    seeds = [1, 2, 3, 4, 5, 6,7,8 ,9 ,10]

    metrics_names_step = {
        "utility": "step_utility",
        "leakage": "step_avg_attack_success_count_per_sender",
        "reward": "step_reward"
    }

    for type_ in attack_types:
        all_points = []          # 所有 alpha 选出的点
        all_alpha_tags = []      # 每个点对应的 alpha

        fig = plt.figure(figsize=(4.8, 3.8))

        for alpha in alphas:
            seed_points = []

            for seed in seeds:
                try:
                    rl_path = rl_adapt_save_name_test(type_, alpha=alpha, sight=sight, seed=seed)
                    rl_utility = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["utility"])
                    rl_leakage = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["leakage"])
                    rl_reward = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["reward"])

                    seed_points.append((
                        float(rl_utility.mean()),
                        float(rl_leakage.mean()),
                        float(rl_reward.mean())
                    ))
                except FileNotFoundError:
                    print(f"⚠️  Warning: seed={seed} not found for {type_}, alpha={alpha}")
                    continue

            if not seed_points:
                print(f"❌ No valid data for {type_}, alpha={alpha}")
                continue

            selected_points, pareto_ul = select_seed_pareto_points(
                seed_points,
                show_all=show_all_pareto
            )

            if not selected_points:
                continue

            for (u, l, r) in selected_points:
                all_points.append((u, l))
                all_alpha_tags.append(alpha)

        if not all_points:
            print(f"❌ No valid plotted points for {type_}")
            continue

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        plt.scatter(
            xs,
            ys,
            s=34,
            c=COLOR_THR,
            edgecolors="black",
            linewidth=0.6,
            zorder=3,
            label="Pareto solutions"
        )

        pareto_points = get_pareto_front(all_points)
        if len(pareto_points) >= 2:
            px, py = zip(*pareto_points)
            plt.plot(
                px, py,
                color=COLOR_THR,
                linewidth=1.8,
                linestyle="-",
                alpha=0.9,
                zorder=2,
                label="Pareto front"
            )

        # 标注 alpha
        if annotate_all:
            for (u, l), alpha in zip(all_points, all_alpha_tags):
                plt.annotate(
                    f"α={alpha:.1f}",
                    (u, l),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=8
                )
        else:
            # 每个 alpha 只标注第一个点，避免太乱
            seen = set()
            for (u, l), alpha in zip(all_points, all_alpha_tags):
                if alpha in seen:
                    continue
                plt.annotate(
                    f"α={alpha:.1f}",
                    (u, l),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=8
                )
                seen.add(alpha)

        plt.title(f"{type_.upper()} Attack - Pareto Front")
        plt.xlabel("Mean Utility")
        plt.ylabel("Mean Leakage")
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=True)
        plt.tight_layout()
        save_figure(fig, f"task2_2_{type_}_pareto_front_seed_pareto")
        plt.show()
# =========================================================
# Task 3
# =========================================================
def task3_1():
    sight_list = [1, 20]
    attack_types = ["dos", "sybil"]
    alpha = 1
    thr = 0.5

    metrics_names_step = {
        "attack_intensity": "step_avg_intensity",
        "reject_rate": "step_avg_rejection_rate",
    }

    for type_ in attack_types:
        fig, axs = plt.subplots(1, len(sight_list), figsize=(4 * (len(sight_list) + 1), 3.2), constrained_layout=False)

        # ===== RL 部分: 每个 sight 一个子图 =====
        for i, sight in enumerate(sight_list):
            ax = axs[i]
            rl_path = rl_adapt_save_name_test(type_, alpha=alpha, sight=sight, seed=5)
            rl_intensity = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["attack_intensity"])
            rl_reject = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["reject_rate"])

            ax.plot(rl_intensity, color=COLOR_THR, label="Intensity")
            ax_twin = ax.twinx()
            ax_twin.plot(rl_reject, color=COLOR_RL, label="Reject Rate")
            ax.set_title(f"RL s={sight}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Intensity", color=COLOR_THR, fontsize=9)
            ax_twin.set_ylabel("Reject Rate", color=COLOR_RL, fontsize=9)
            ax.tick_params(axis="y", labelcolor=COLOR_THR, labelsize=8)
            ax_twin.tick_params(axis="y", labelcolor=COLOR_RL, labelsize=8)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(0, 1)
            ax_twin.set_ylim(0, 1)

        # # ===== Threshold 部分: 最后一个子图 =====
        # ax_th = axs[0]
        # th_path = th_adapt_save_name_test(type_, thr=thr, seed=8)
        # th_intensity = load_npy(os.path.join(run_dir, th_path), metrics_names_step["attack_intensity"])
        # th_reject = load_npy(os.path.join(run_dir, th_path), metrics_names_step["reject_rate"])

        # ax_th.plot(th_intensity, color=COLOR_THR, label="Intensity")
        # ax_th_twin = ax_th.twinx()
        # ax_th_twin.plot(th_reject, color=COLOR_RL, label="Reject Rate")
        # ax_th.set_title(f"Threshold thr={thr}")
        # ax_th.set_xlabel("Step")
        # ax_th.set_ylabel("Intensity", color=COLOR_THR, fontsize=9)
        # ax_th_twin.set_ylabel("Reject Rate", color=COLOR_RL, fontsize=9)
        # ax_th.tick_params(axis="y", labelcolor=COLOR_THR, labelsize=8)
        # ax_th_twin.tick_params(axis="y", labelcolor=COLOR_RL, labelsize=8)
        # ax_th.grid(True, alpha=0.25)
        # ax_th.set_ylim(0, 1)
        # ax_th_twin.set_ylim(0, 1)

        plt.suptitle(f"{type_.upper()} Attack - RL(sight sweep) vs Threshold", y=1.02)
        plt.tight_layout()
        save_figure(fig, f"task3_1_{type_}_sight_sweep")
        plt.show()


# =========================================================
# Task 4
# =========================================================
def task4_1():
    attack_types = ["sybil"]
    alphas = [0,0.2, 0.5, 1, 2]
    sight_list = [1, 20]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    sight_colors = {1: "tab:blue", 5: "tab:green", 10: "tab:orange", 20: "tab:red"}
    sight_markers = {1: "o", 5: "s", 10: "^", 20: "D"}

    metrics_names_step = {
        "utility": "step_utility",
        "leakage": "step_avg_attack_success_count_per_sender",
        "reward": "step_reward"
    }

    def find_central_point(points):
        """
        找到欧氏距离和最小的点作为中心代表点
        points: list of (utility, leakage) tuples
        """
        if len(points) == 1:
            return points[0]

        min_total_dist = float("inf")
        central = points[0]

        for i, (u_i, l_i) in enumerate(points):
            total_dist = sum(
                np.sqrt((u_i - u_j) ** 2 + (l_i - l_j) ** 2)
                for j, (u_j, l_j) in enumerate(points) if i != j
            )
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                central = (u_i, l_i)
        return central

    for type_ in attack_types:
        fig = plt.figure(figsize=(6.4, 4.8))

        for sight in sight_list:
            rl_utility_list, rl_leakage_list = [], []
            alpha_valid_flags = []

            for alpha in alphas:
                seed_points = []

                for seed in seeds:
                    try:
                        rl_path = rl_adapt_save_name_test(type_, alpha=alpha, sight=sight, seed=seed)
                        rl_utility = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["utility"])
                        rl_leakage = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["leakage"])

                        seed_points.append((
                            rl_utility.mean(),
                            rl_leakage.mean()
                        ))
                    except FileNotFoundError:
                        continue

                if not seed_points:
                    print(f"⚠️  No data for {type_}, sight={sight}, alpha={alpha}")
                    alpha_valid_flags.append(False)
                    continue

                central_util, central_leak = find_central_point(seed_points)
                rl_utility_list.append(central_util)
                rl_leakage_list.append(central_leak)
                alpha_valid_flags.append(True)

            if not rl_utility_list:
                continue

            color = sight_colors[sight]
            marker = sight_markers.get(sight, "o")

            plt.scatter(
                rl_utility_list,
                rl_leakage_list,
                c=color,
                marker=marker,
                s=78,
                label=f"sight={sight} (n={len(seeds)} seeds)",
                edgecolors="black",
                linewidth=0.6,
                alpha=0.95,
                zorder=3
            )

            # 画 Pareto 前沿线
            pareto_points = get_pareto_front(list(zip(rl_utility_list, rl_leakage_list)))
            if len(pareto_points) >= 2:
                px, py = zip(*pareto_points)
                plt.plot(
                    px, py,
                    color=color,
                    linewidth=2.0,
                    linestyle="-",
                    alpha=0.9,
                    zorder=2
                )

            valid_idx = 0
            for i, alpha in enumerate(alphas):
                if alpha_valid_flags[i]:
                    plt.annotate(
                        f"α={alpha:.1f}",
                        (rl_utility_list[valid_idx], rl_leakage_list[valid_idx]),
                        textcoords="offset points",
                        xytext=(7, 7),
                        ha="left",
                        fontsize=8,
                        color=color,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor=color,
                            alpha=0.85,
                            linewidth=0.5
                        )
                    )
                    valid_idx += 1

        plt.title(f"{type_.upper()} Attack - Pareto Front ", fontsize=12, pad=12)
        plt.xlabel("Mean Utility", fontsize=10)
        plt.ylabel("Mean Leakage", fontsize=10)
        plt.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        plt.legend(title="Observation Sight", loc="best", frameon=True, fancybox=False)
        plt.tight_layout()
        save_figure(fig, f"task4_1_{type_}_pareto_front_multiseed")
        plt.show()

def task4_2(show_all_pareto=True, annotate_all=False):
    """
    Task 4-2:
    与 task4_1 类似，但每个 sight / alpha 下不再取 central point，
    而是取该组 seeds 中的 Pareto 支配解。

    参数
    ----
    show_all_pareto : bool
        True  -> 一个 alpha 下所有 Pareto 解都画出来
        False -> 只画 leakage 最小的那个 Pareto 解
    annotate_all : bool
        True  -> 对所有选出的点都标注 alpha
        False -> 只标注每个 alpha 的第一个选中点
    """
    attack_types = ["sybil"]
    alphas = [0, 0.2, 0.5, 1, 1.5, 2]
    sight_list = [1, 20]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]

    sight_colors = {1: "tab:blue", 5: "tab:green", 10: "tab:orange", 20: "tab:red"}
    sight_markers = {1: "o", 5: "s", 10: "^", 20: "D"}

    metrics_names_step = {
        "utility": "step_utility",
        "leakage": "step_avg_attack_success_count_per_sender",
        "reward": "step_reward"
    }

    for type_ in attack_types:
        fig = plt.figure(figsize=(6.4, 4.8))

        for sight in sight_list:
            all_points = []
            all_alpha_tags = []

            for alpha in alphas:
                seed_points = []

                for seed in seeds:
                    try:
                        rl_path = rl_adapt_save_name_test(type_, alpha=alpha, sight=sight, seed=seed)
                        rl_utility = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["utility"])
                        rl_leakage = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["leakage"])
                        rl_reward = load_npy(os.path.join(run_dir, rl_path), metrics_names_step["reward"])

                        seed_points.append((
                            float(rl_utility.mean()),
                            float(rl_leakage.mean()),
                            float(rl_reward.mean())
                        ))
                    except FileNotFoundError:
                        continue

                if not seed_points:
                    print(f"⚠️  No data for {type_}, sight={sight}, alpha={alpha}")
                    continue

                selected_points, pareto_ul = select_seed_pareto_points(
                    seed_points,
                    show_all=show_all_pareto
                )

                if not selected_points:
                    continue

                for (u, l, r) in selected_points:
                    all_points.append((u, l))
                    all_alpha_tags.append(alpha)

            if not all_points:
                continue

            color = sight_colors.get(sight, "tab:blue")
            marker = sight_markers.get(sight, "o")

            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]

            plt.scatter(
                xs,
                ys,
                c=color,
                marker=marker,
                s=78,
                label=f"sight={sight} (Pareto solutions)",
                edgecolors="black",
                linewidth=0.6,
                alpha=0.95,
                zorder=3
            )

            pareto_points = get_pareto_front(all_points)
            if len(pareto_points) >= 2:
                px, py = zip(*pareto_points)
                plt.plot(
                    px, py,
                    color=color,
                    linewidth=2.0,
                    linestyle="-",
                    alpha=0.9,
                    zorder=2
                )

            if annotate_all:
                for (u, l), alpha in zip(all_points, all_alpha_tags):
                    plt.annotate(
                        f"α={alpha:.1f}",
                        (u, l),
                        textcoords="offset points",
                        xytext=(7, 7),
                        ha="left",
                        fontsize=8,
                        color=color,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor=color,
                            alpha=0.85,
                            linewidth=0.5
                        )
                    )
            else:
                seen = set()
                for (u, l), alpha in zip(all_points, all_alpha_tags):
                    if alpha in seen:
                        continue
                    plt.annotate(
                        f"α={alpha:.1f}",
                        (u, l),
                        textcoords="offset points",
                        xytext=(7, 7),
                        ha="left",
                        fontsize=8,
                        color=color,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor=color,
                            alpha=0.85,
                            linewidth=0.5
                        )
                    )
                    seen.add(alpha)

        plt.title(f"{type_.upper()} Attack - Pareto Front", fontsize=12, pad=12)
        plt.xlabel("Mean Utility", fontsize=10)
        plt.ylabel("Mean Leakage", fontsize=10)
        plt.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        plt.legend(title="Observation Sight", loc="best", frameon=True, fancybox=False)
        plt.tight_layout()
        save_figure(fig, f"task4_2_{type_}_pareto_front_seed_pareto")
        plt.show()
# =========================================================
# Task 5
# =========================================================
def task5_1():
    attack_types = ["dos", "sybil"]
    methods = ["gnn", "mlp"]

    gnn_path_cal = lambda at: f"../../outputs/gnn_outputs/single_{at}_gnn/"
    mlp_path_cal = lambda at: f"../../outputs/mlp_outputs/single_{at}_mlp/"
    path_map = {"gnn": gnn_path_cal, "mlp": mlp_path_cal}

    rows = []
    for at in attack_types:
        for method in methods:
            base_path = path_map[method](at)
            metrics_path = os.path.join(base_path, "metrics.json")

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            rows.append({
                "Attack": at.upper(),
                "Method": method.upper(),
                "Test F1": f"{metrics['test_f1']:.4f}",
                "Test AUC": f"{metrics['test_auc']:.4f}",
                "Best Val F1": f"{metrics['best_val_f1']:.4f}",
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Attack", columns="Method", values=["Test F1", "Test AUC", "Best Val F1"])

    print("\n📊 GNN vs MLP Performance Comparison")
    print("=" * 60)
    print(pivot)


def task5_2_extract_tsne(attack_type="sybil"):
    # 路径配置
    DATA_PATH = f"../../data/graphs_{attack_type}.pt"
    GNN_CKPT = f"../../outputs/gnn_outputs/single_{attack_type}_gnn/best_model.pt"
    MLP_CKPT = f"../../outputs/mlp_outputs/single_{attack_type}_mlp/best_model.pt"
    SCALER_PATH = f"../../outputs/gnn_outputs/single_{attack_type}_gnn/scaler.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/3] Loading data & models...")
    graphs = torch.load(DATA_PATH, weights_only=False)
    scaler = joblib.load(SCALER_PATH)
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)

    X_all = torch.cat([g.x for g in graphs], dim=0)
    y_all = torch.cat([g.y for g in graphs], dim=0).view(-1).numpy()
    in_dim = X_all.shape[1]

    gnn = GNN_model(in_dim=in_dim).to(device)
    gnn.load_state_dict(torch.load(GNN_CKPT, map_location=device))
    gnn.eval()

    mlp = MLP_model(in_dim=in_dim).to(device)
    mlp.load_state_dict(torch.load(MLP_CKPT, map_location=device))
    mlp.eval()

    print("[2/3] Extracting embeddings...")

    @torch.no_grad()
    def get_gnn_emb():
        embs = []
        for g in graphs:
            g = g.to(device)
            embs.append(gnn.embed(g.x, g.edge_index, g.edge_attr).cpu())
        return torch.cat(embs, dim=0).numpy()

    @torch.no_grad()
    def get_mlp_emb():
        embs = []
        loader = DataLoader(TensorDataset(X_all), batch_size=2048, shuffle=False)
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            feat = batch_x
            for layer in list(mlp.network.children())[:-1]:
                feat = layer(feat)
            embs.append(feat.cpu())
        return torch.cat(embs, dim=0).numpy()

    gnn_emb = get_gnn_emb()
    mlp_emb = get_mlp_emb()

    print("[3/3] Running t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, init="pca", n_jobs=-1)
    gnn_2d = tsne.fit_transform(gnn_emb)
    mlp_2d = tsne.fit_transform(mlp_emb)

    save_path = os.path.join(FIGURE_DIR, f"{attack_type}_tsne_results.npz")
    np.savez(save_path, gnn_2d=gnn_2d, mlp_2d=mlp_2d, y_all=y_all)
    print(f"✅ Saved to {save_path}")


def task5_2_plot_tsne(attack_type="sybil"):
    load_path = os.path.join(f"{attack_type}_tsne_results.npz")

    print(f"Loading from {load_path}...")
    data = np.load(load_path)
    gnn_2d = data["gnn_2d"]
    mlp_2d = data["mlp_2d"]
    y_all = data["y_all"]

    fig = plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    for label in np.unique(y_all)[::-1]:
        mask = y_all == label
        attack_label = "Attack" if label == 1 else "Normal"
        plt.scatter(
            gnn_2d[mask, 0],
            gnn_2d[mask, 1],
            s=5,
            alpha=0.75,
            label=attack_label,
            c=COLOR_ATTACK if label == 1 else COLOR_NORMAL
        )
    plt.title("GNN Embedding")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=True)

    plt.subplot(1, 2, 2)
    for label in np.unique(y_all)[::-1]:
        mask = y_all == label
        attack_label = "Attack" if label == 1 else "Normal"
        plt.scatter(
            mlp_2d[mask, 0],
            mlp_2d[mask, 1],
            s=5,
            alpha=0.75,
            label=attack_label,
            c=COLOR_ATTACK if label == 1 else COLOR_NORMAL
        )
    plt.title("MLP Embedding")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=True)

    plt.suptitle(f"{attack_type} Attack - 2D t-SNE Comparison", y=1.02)
    plt.tight_layout()
    save_figure(fig, f"task5_2_{attack_type}_tsne")
    plt.show()


# =========================================================
# Model definitions
# =========================================================
class MLP_model(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def embed(self, x):
        return self.network[:-1](x)


class GNN_model(nn.Module):
    def __init__(self, in_dim, hidden=16, heads=2, dropout=0.35):
        super().__init__()

        self.dropout = dropout
        self.hidden = hidden
        self.heads = heads

        hdim = hidden * heads

        self.conv1 = TransformerConv(
            in_dim,
            hidden,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
            beta=True
        )
        self.bn1 = nn.BatchNorm1d(hdim)
        self.skip1 = nn.Linear(in_dim, hdim)

        self.conv2 = TransformerConv(
            hdim,
            hidden,
            heads=heads,
            edge_dim=1,
            dropout=dropout,
            beta=True
        )
        self.bn2 = nn.BatchNorm1d(hdim)
        self.skip2 = nn.Linear(hdim, hdim)

        self.mlp = nn.Sequential(
            nn.Linear(hdim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def embed(self, x, edge_index, edge_attr):
        """
        输出节点 embedding
        shape = [N, hidden*heads]
        """
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = F.relu(x1 + self.skip1(x))
        x1 = F.dropout(
            x1,
            p=self.dropout,
            training=self.training
        )

        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = F.relu(x2 + self.skip2(x1))
        x2 = F.dropout(
            x2,
            p=self.dropout,
            training=self.training
        )

        return x2

    def forward(self, x, edge_index, edge_attr):
        """
        监督学习输出
        """
        embedding = self.embed(
            x,
            edge_index,
            edge_attr
        )

        return self.mlp(embedding)

    def global_embed(self, x, edge_index, edge_attr):
        """
        图嵌入输出
        """
        embedding_for_nodes = self.embed(
            x,
            edge_index,
            edge_attr
        )
        embedding = torch.mean(embedding_for_nodes, dim=0)

        score_for_nodes = self.mlp(embedding_for_nodes)
        score = torch.mean(score_for_nodes, dim=0)

        return embedding

    def out_head(self, embedding):
        return self.mlp(embedding)


# =========================================================
# 运行入口（按需取消注释）
# =========================================================
if __name__ == "__main__":
    task1_1()
    task1_3()
    task2_1()
    # task2_2(show_all_pareto=True)     # 每个 alpha 下所有支配解都画出来
    task2_2(show_all_pareto=False)    # 每个 alpha 只保留 leakage 最小的支配解

    # task4_2(show_all_pareto=True)
    task4_2(show_all_pareto=False)
    task3_1()
    task4_1()
    task5_1()
    # task5_2_extract_tsne(attack_type="sybil")
    task5_2_plot_tsne(attack_type="sybil")
    pass
