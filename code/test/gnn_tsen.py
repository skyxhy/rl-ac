import os
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from ..model.gnn_model import GNN_model

# =========================
# 配置
# =========================
INPUT_PATH = "data/graphs.pt"
MODEL_PATH = "outputs/gnn_outputs/run_20260418_115612/best_model.pt"
SCALER_PATH = "outputs/scaler.pkl"

NUM_SAMPLES = 90
PERPLEXITY = 5
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# 加载数据
# =========================
graphs = torch.load(INPUT_PATH, weights_only=False)
graphs = graphs[:NUM_SAMPLES]

# =========================
# 加载 scaler（关键）
# =========================
print("[Scaler] Loading...")
scaler = joblib.load(SCALER_PATH)

def scale_graphs(graphs):
    scaled = []
    for g in graphs:
        x = g.x.cpu().numpy()
        x = scaler.transform(x)

        g.x = torch.tensor(x, dtype=torch.float32)
        scaled.append(g)
    return scaled

graphs = scale_graphs(graphs)

loader = DataLoader(graphs, batch_size=1, shuffle=False)

# =========================
# 加载模型
# =========================
in_dim = graphs[0].x.shape[1]

model = GNN_model(in_dim=in_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =========================
# 提取 embedding
# =========================
@torch.no_grad()
def extract_embeddings(loader):
    all_embeds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        # ⚠️ 必须带 edge_attr（你之前报错的根因）
        emb = model.embed(
            batch.x,
            batch.edge_index,
            batch.edge_attr
        )

        all_embeds.append(emb.cpu())
        all_labels.append(batch.y.cpu())

    X = torch.cat(all_embeds).numpy()
    y = torch.cat(all_labels).numpy()

    return X, y


print("[TSNE] Extracting embeddings...")
X, y = extract_embeddings(loader)

print(f"[TSNE] samples={X.shape}")

# =========================
# TSNE 降维
# =========================
print("[TSNE] Running TSNE...")

tsne = TSNE(
    n_components=2,
    perplexity=PERPLEXITY,
    random_state=RANDOM_SEED,
    init="pca",
    learning_rate=50,   # 🔥 比你原来更稳定
)

X_2d = tsne.fit_transform(X)

# =========================
# 可视化
# =========================
print("[TSNE] Plotting...")

plt.figure(figsize=(6, 5))

normal_idx = (y == 0)
attack_idx = (y != 0)

plt.scatter(
    X_2d[normal_idx, 0],
    X_2d[normal_idx, 1],
    s=10,
    alpha=0.6,
    label="Normal"
)

plt.scatter(
    X_2d[attack_idx, 0],
    X_2d[attack_idx, 1],
    s=10,
    alpha=0.6,
    label="Attack"
)

plt.legend()
plt.title("t-SNE of GNN Embeddings")
plt.tight_layout()

plt.savefig("tsne_visualization.png", dpi=300)
plt.show()

print("✅ TSNE figure saved to tsne_visualization.png")