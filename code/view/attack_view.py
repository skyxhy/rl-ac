import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# ========== 数据加载 ==========
attack_path = os.path.join(os.path.dirname(__file__), '../../sybil.csv')
df = pd.read_csv(attack_path)
df = df[df["attack_type"] == 1]

sender_id = 23853
sender_df = df[df['sender_id'] == sender_id].copy()

if sender_df.empty:
    raise ValueError(f"❌ 未找到 sender_id={sender_id} 的数据！")

# ========== 🕰️ 交互式时间范围选择 ==========
TIME_COL = 'time'  
min_t = sender_df[TIME_COL].min()
max_t = sender_df[TIME_COL].max()

print(f"\n📊 {TIME_COL} 原始范围: [{min_t}, {max_t}]")

while True:
    start_in = input(f"👉 输入起始 {TIME_COL} (直接回车 = {min_t}): ").strip()
    end_in   = input(f"👉 输入结束 {TIME_COL} (直接回车 = {max_t}): ").strip()

    try:
        start_t = float(start_in) if start_in else min_t
        end_t   = float(end_in) if end_in else max_t

        if start_t > end_t:
            print("❌ 起始时间不能大于结束时间，请重新输入。\n")
            continue

        # 过滤数据
        sender_df = sender_df[(sender_df[TIME_COL] >= start_t) & (sender_df[TIME_COL] <= end_t)]
        if sender_df.empty:
            print("⚠️ 该时间范围内无交互记录，请重新输入。\n")
            continue

        print(f"✅ 成功筛选 {len(sender_df)} 条交互记录。区间: [{start_t}, {end_t}]\n")
        break
    except ValueError:
        print("❌ 请输入有效的数字。\n")

# ========== 构建邻接矩阵 ==========
senders = sorted(sender_df['sender_pseudo'].unique())
receivers = sorted(sender_df['receiver_id'].unique())

sender_to_idx = {s: i for i, s in enumerate(senders)}
receiver_to_idx = {r: j for j, r in enumerate(receivers)}

adj_matrix = np.zeros((len(senders), len(receivers)), dtype=int)

for _, row in sender_df.iterrows():
    src = row['sender_pseudo']
    dst = row['receiver_id']
    if src in sender_to_idx and dst in receiver_to_idx:
        adj_matrix[sender_to_idx[src], receiver_to_idx[dst]] += 1

print(f"📦 Matrix shape: {adj_matrix.shape} (senders × receivers)")

# ========== 可视化 ==========
plt.figure(figsize=(12, 8))

def shorten_labels(labels, max_len=12):
    return [f"{str(l)[:max_len]}.." if len(str(l)) > max_len else str(l) for l in labels]

x_labels = shorten_labels(receivers)
y_labels = shorten_labels(senders)

sns.heatmap(adj_matrix,
            cmap='YlOrRd',
            xticklabels=x_labels,
            yticklabels=y_labels,
            cbar_kws={'label': 'Interaction Count'},
            linewidths=0.2)

plt.title(f'Interaction Adjacency Matrix\nsender_id={sender_id} | Time: {start_t:.1f} ~ {end_t:.1f}', fontsize=14)
plt.xlabel('receiver_id', fontsize=10)
plt.ylabel('sender_pseudo', fontsize=10)

plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()