import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 平滑函数
# =========================
def smooth(x, k=5):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k)/k, mode='valid')


# =========================
# 读取单条曲线
# =========================
def load_curve(path):
    return np.loadtxt(path, delimiter=",")


# =========================
# 多 run 平均（可选）
# =========================
def load_multi_runs(dir_path):
    curves = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f == "reward_curve.csv":
                p = os.path.join(root, f)
                curves.append(load_curve(p))

    if len(curves) == 0:
        raise ValueError(f"No reward_curve.csv found in {dir_path}")

    # 对齐长度（取最短）
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]

    return np.mean(curves, axis=0), np.std(curves, axis=0)


# =========================
# 主函数
# =========================
def plot_rl_vs_cb(
    rl_path,
    cb_path,
    save_path="rl_vs_cb.png",
    smooth_k=5,
    multi_run=False
):
    if multi_run:
        rl_mean, rl_std = load_multi_runs(rl_path)
        cb_mean, cb_std = load_multi_runs(cb_path)
    else:
        rl_mean = load_curve(rl_path)[-150:]  # 取最后150步，代表稳定状态
        cb_mean = load_curve(cb_path)[-150:]  # 取最后150步，代表稳定状态
        rl_std = cb_std = None
    mean_rl = np.mean(rl_mean)
    mean_cb = np.mean(cb_mean)
    print(f"RL Mean Reward: {mean_rl:.4f}")
    print(f"CB Mean Reward: {mean_cb:.4f}")
    diff = mean_rl - mean_cb
    print(f"RL - CB: {diff:.4f}")
    # 平滑
    rl_s = smooth(rl_mean, smooth_k)
    cb_s = smooth(cb_mean, smooth_k)

    x = np.arange(len(rl_s))

    plt.figure(figsize=(7, 5))

    # RL
    plt.plot(x, rl_s, label="RL", linewidth=2)
    if rl_std is not None:
        rl_std = rl_std[:len(rl_s)]
        plt.fill_between(x, rl_s - rl_std, rl_s + rl_std, alpha=0.2)

    # CB
    plt.plot(x, cb_s, label="CB", linewidth=2)
    if cb_std is not None:
        cb_std = cb_std[:len(cb_s)]
        plt.fill_between(x, cb_s - cb_std, cb_s + cb_std, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("RL vs CB (Smoothed Reward)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axhline(np.mean(rl_mean), linestyle="--",color="blue", label="RL Mean")
    plt.axhline(np.mean(cb_mean), linestyle="--",color="orange", label="CB Mean")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# =========================
# 示例调用
# =========================
if __name__ == "__main__":
    # 单次实验
    plot_rl_vs_cb(
        rl_path="outputs/simulation/test_rl/csv/step_reward.csv",
        cb_path="outputs/simulation/test_cb/csv/step_reward.csv",
        save_path="compare.png",
        smooth_k=10,
        multi_run=False
    )

    # 多次实验（自动找所有 reward_curve.csv）
    # plot_rl_vs_cb(
    #     rl_path="logs/rl/",
    #     cb_path="logs/cb/",
    #     save_path="compare_multi.png",
    #     smooth_k=10,
    #     multi_run=True
    # )