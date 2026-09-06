import os
import json
import torch
import numpy as np
import datetime
import argparse
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import torch.nn as nn
from ..model.mlp_model import MLP_model


# =========================================================
# 参数
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="mlp_baseline")
parser.add_argument("--use_timestamp", type=int, default=1)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=256)  # ✅ MLP通常可用更大batch
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--clip_norm", type=float, default=2.0)
parser.add_argument("--input_path", type=str, default="data/graphs.pt")
parser.add_argument("--output_dir", type=str, default="outputs/mlp_outputs")
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--test_ratio", type=float, default=0.2)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# =========================================================
# 随机种子 & 设备
# =========================================================
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 输出目录
# =========================================================
EXP_DIR = os.path.join(args.output_dir, args.exp_name)
if args.use_timestamp:
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = os.path.join(EXP_DIR, f"run_{TIMESTAMP}")
else:
    RUN_DIR = EXP_DIR
os.makedirs(RUN_DIR, exist_ok=True)

with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)
print(f"[Run Dir] {RUN_DIR}")

# =========================================================
# 数据加载 & 划分
# =========================================================
print("[Data] Loading...")
graphs = torch.load(args.input_path, weights_only=False)
if len(graphs) < 10:
    raise RuntimeError("图样本太少")

n = len(graphs)
n_train = int(n * args.train_ratio)
n_val = int(n * args.val_ratio)
train_graphs = graphs[:n_train]
val_graphs = graphs[n_train:n_train + n_val]
test_graphs = graphs[n_train + n_val:]
print(f"[Split] train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

# =========================================================
# 标准化
# =========================================================
scaler = StandardScaler()
train_pool = np.vstack([g.x.numpy() for g in train_graphs])
scaler.fit(train_pool)
joblib.dump(scaler, os.path.join(RUN_DIR, "scaler.pkl"))

def scale_graphs(g_list):
    for g in g_list:
        g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float)
    return g_list

train_graphs = scale_graphs(train_graphs)
val_graphs = scale_graphs(val_graphs)
test_graphs = scale_graphs(test_graphs)

# ✅ MLP适配: 将图数据展平为 (X, y) 张量
def graphs_to_tensors(g_list):
    X = torch.cat([g.x for g in g_list], dim=0)
    y = torch.cat([g.y for g in g_list], dim=0).view(-1).long()  # 确保1D
    return X, y

X_train, y_train = graphs_to_tensors(train_graphs)
X_val, y_val = graphs_to_tensors(val_graphs)
X_test, y_test = graphs_to_tensors(test_graphs)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

# =========================================================
# 类别权重
# =========================================================
counts = torch.bincount(y_train, minlength=2).float()
pos_weight = counts[0] / counts[1].clamp(min=1.0)
print(f"[Counts] {counts}")
print(f"[Pos Weight] {pos_weight.item():.4f}")

# =========================================================
# 模型 & 优化器
# =========================================================
model = MLP_model(in_dim=X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

# =========================================================
# 评估函数
# =========================================================
@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x).squeeze(-1)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).long()
        all_true.append(batch_y.cpu())
        all_pred.append(pred.cpu())
        all_prob.append(prob.cpu())

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_prob = torch.cat(all_prob).numpy()

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")
    return f1, auc, y_true, y_pred

# =========================================================
# 训练循环
# =========================================================
best_val_f1 = -1
best_state = None
bad_epochs = 0
curves = {"epoch": [], "train_loss": [], "val_f1": [], "val_auc": []}

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x).squeeze(-1)
        loss = criterion(logits, batch_y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        total_loss += loss.item()

    val_f1, val_auc, _, _ = evaluate(val_loader)
    scheduler.step(val_f1)

    curves["epoch"].append(epoch)
    curves["train_loss"].append(total_loss)
    curves["val_f1"].append(val_f1)
    curves["val_auc"].append(val_auc)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= args.patience:
        print(f"Early stopping @ epoch {epoch}")
        break

# =========================================================
# 测试 & 保存
# =========================================================
torch.save(best_state, os.path.join(RUN_DIR, "best_model.pt"))
model.load_state_dict(best_state)
test_f1, test_auc, y_true, y_pred = evaluate(test_loader)
report = classification_report(y_true, y_pred, digits=4, zero_division=0)

print(f"\nF1={test_f1:.4f} AUC={test_auc:.4f}")
print(report)

with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
    json.dump({"test_f1": float(test_f1), "test_auc": float(test_auc), "best_val_f1": float(best_val_f1), "report": report}, f, indent=2)
with open(os.path.join(RUN_DIR, "curves.json"), "w") as f:
    json.dump(curves, f, indent=2)

print(f"\n✅ Saved to {RUN_DIR}")