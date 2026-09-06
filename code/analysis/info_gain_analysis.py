import pandas as pd

CSV_PATH = "output.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    required_cols = ["sender_id", "receiver_id", "attack_type"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    # =========================
    # 每个真实 sender 和 receiver 的交互次数
    # =========================
    interaction_counts = (
        df.groupby(["sender_id", "receiver_id"])
        .size()
        .reset_index(name="count")
    )

    # =========================
    # sender 标签
    # =========================
    sender_labels = (
        df.groupby("sender_id")["attack_type"]
        .max()
        .reset_index()
    )

    sender_labels["is_attack"] = sender_labels["attack_type"] > 0

    # =========================
    # 每个 sender 的最大交互次数
    # =========================
    sender_max_counts = (
        interaction_counts.groupby("sender_id")["count"]
        .max()
        .reset_index(name="max_interaction_count")
    )

    result = sender_max_counts.merge(
        sender_labels[["sender_id", "is_attack"]],
        on="sender_id",
        how="left"
    )

    # =========================
    # 分组统计
    # =========================
    attack_df = result[result["is_attack"]]
    benign_df = result[~result["is_attack"]]

    print("=" * 60)
    print("攻击者 sender 最大交互次数统计")
    print("=" * 60)
    print(f"sender 数量: {len(attack_df)}")
    print(f"平均最大交互次数: {attack_df['max_interaction_count'].mean():.4f}")
    print(f"中位数: {attack_df['max_interaction_count'].median():.4f}")
    print(f"最大值: {attack_df['max_interaction_count'].max()}")
    print(f"最小值: {attack_df['max_interaction_count'].min()}")

    print()

    print("=" * 60)
    print("正常节点 sender 最大交互次数统计")
    print("=" * 60)
    print(f"sender 数量: {len(benign_df)}")
    print(f"平均最大交互次数: {benign_df['max_interaction_count'].mean():.4f}")
    print(f"中位数: {benign_df['max_interaction_count'].median():.4f}")
    print(f"最大值: {benign_df['max_interaction_count'].max()}")
    print(f"最小值: {benign_df['max_interaction_count'].min()}")

    print()

    ratio = (
        attack_df["max_interaction_count"].mean()
        / benign_df["max_interaction_count"].mean()
    )

    print("=" * 60)
    print("攻击者相对收益能力")
    print("=" * 60)
    print(f"攻击/正常 平均最大交互次数比值: {ratio:.4f}")


if __name__ == "__main__":
    main()