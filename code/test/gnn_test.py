import os
import json
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score
)

from ..model.gnn_model import GNN_model


# =========================================================
# 参数
# =========================================================
INPUT_PATH = "data/graphs.pt"
MODEL_PATH = "outputs/gnn_outputs/run_20260415_222249/best_model.pt"
THRESHOLD = 0.5
BATCH_SIZE = 4

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 数据加载
# =========================================================
print("[Data] Loading graphs...")

graphs = torch.load(INPUT_PATH, weights_only=False)

n = len(graphs)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_graphs = graphs[:n_train]
test_graphs = graphs[n_train + n_val:]

print(f"[Data] train={len(train_graphs)}, test={len(test_graphs)}")


# =========================================================
# 标准化（必须和训练一致）
# =========================================================
print("[Scaler] Fitting on train split...")

scaler = StandardScaler()

train_pool = np.vstack([g.x.numpy() for g in train_graphs])
scaler.fit(train_pool)
import joblib

joblib.dump(scaler, "outputs/scaler.pkl")

def scale_graphs(graph_list):
    scaled = []

    for g in graph_list:
        g.x = torch.tensor(
            scaler.transform(g.x.numpy()),
            dtype=torch.float32
        )
        scaled.append(g)

    return scaled


test_graphs = scale_graphs(test_graphs)

test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================================================
# 模型加载
# =========================================================
print("[Model] Loading...")

in_dim = test_graphs[0].x.shape[1]

model = GNN_model(in_dim=in_dim).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(state_dict)

model.eval()


# =========================================================
# 测试函数
# =========================================================
@torch.no_grad()
def evaluate_next_snapshot(graphs, metadata_list, threshold=0.5):
    """
    用 t 时刻预测结果去评估 t+1
    基于 pseudo(node_ids) 对齐

    新出现节点默认预测正常
    """

    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for i in range(len(graphs) - 1):
        g_prev = graphs[i]
        g_next = graphs[i + 1]

        meta_prev = metadata_list[i]
        meta_next = metadata_list[i + 1]

        # ==========================
        # 1. 前一快照输出
        # ==========================
        g_prev = g_prev.to(device)

        logits = model(
            g_prev.x,
            g_prev.edge_index,
            g_prev.edge_attr
        ).squeeze(-1)

        prob_prev = torch.sigmoid(logits).cpu().numpy()
        pred_prev = (prob_prev > threshold).astype(int)

        # ==========================
        # 2. 用 node_ids 对齐
        # ==========================
        prev_nodes = meta_prev["node_ids"]
        next_nodes = meta_next["node_ids"]

        # 长度安全检查
        assert len(prev_nodes) == len(pred_prev), (
            f"prev nodes={len(prev_nodes)} "
            f"!= pred={len(pred_prev)}"
        )

        assert len(next_nodes) == len(g_next.y), (
            f"next nodes={len(next_nodes)} "
            f"!= label={len(g_next.y)}"
        )

        # ==========================
        # 3. 建立上一时刻映射
        # ==========================
        prev_pred_map = {
            nid: pred
            for nid, pred in zip(prev_nodes, pred_prev)
        }

        prev_prob_map = {
            nid: prob
            for nid, prob in zip(prev_nodes, prob_prev)
        }

        # ==========================
        # 4. 用 t 预测 t+1
        # ==========================
        next_labels = g_next.y.cpu().numpy()

        for nid, true_label in zip(next_nodes, next_labels):
            pred = prev_pred_map.get(nid, 0)
            prob = prev_prob_map.get(nid, 0.0)

            all_true.append(int(true_label))
            all_pred.append(int(pred))
            all_prob.append(float(prob))

    # ==========================
    # 5. metrics
    # ==========================
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return macro_f1, auc, y_true, y_pred, y_prob


@torch.no_grad()
def evaluate(loader, threshold=0.5):
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for batch in loader:
        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr
        ).squeeze(-1)

        prob = torch.sigmoid(logits)

        pred = (prob > threshold).long()

        all_true.append(batch.y.cpu())
        all_pred.append(pred.cpu())
        all_prob.append(prob.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_prob = torch.cat(all_prob).numpy()

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    return macro_f1, auc, y_true, y_pred, y_prob


# =========================================================
# 开始测试
# =========================================================
print(f"[Test] threshold={THRESHOLD}")

# test_f1, test_auc, y_true, y_pred, y_prob = evaluate(
#     test_loader,
#     threshold=THRESHOLD
# )
METADATA_PATH = "data/graph_metadata.pt"
graph_metadata = torch.load(METADATA_PATH, weights_only=False)
test_f1, test_auc, y_true, y_pred, y_prob = evaluate_next_snapshot(
    test_graphs,
    graph_metadata[n_train + n_val:],
    threshold=THRESHOLD
)

report = classification_report(
    y_true,
    y_pred,
    digits=4,
    zero_division=0
)

print("\n========== TEST RESULT ==========")
print(f"Threshold : {THRESHOLD}")
print(f"Macro F1  : {test_f1:.4f}")
print(f"ROC AUC   : {test_auc:.4f}")
print(report)


# =========================================================
# 保存测试结果
# =========================================================
save_result = {
    "threshold": THRESHOLD,
    "macro_f1": float(test_f1),
    "roc_auc": float(test_auc),
    "report": report
}

with open("gnn_test_result.json", "w") as f:
    json.dump(save_result, f, indent=2)

print("\n✅ Test result saved to gnn_test_result.json")