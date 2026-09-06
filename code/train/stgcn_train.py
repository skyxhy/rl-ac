import os
import json
import torch
import torch.nn as nn
import numpy as np
import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from ..model.stgcn_model import MaskedTemporalSTGNN

# =========================================================
# 参数
# =========================================================
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

SEQ_LEN = 5

EPOCHS = 300
PATIENCE = 15

BATCH_SIZE = 1

LR = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_NORM = 2.0

INPUT_PATH = "data/stgcn_samples_masked.pt"
BASE_OUT_DIR = "outputs/stgcn_outputs_masked"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 输出目录
# =========================================================
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(BASE_OUT_DIR, f"run_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

# =========================================================
# 工具函数
# =========================================================
def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj

# =========================================================
# 数据加载
# =========================================================
samples = torch.load(INPUT_PATH, weights_only=False)

print(f"[Data] Loaded {len(samples)} samples.")

samples = sorted(samples, key=lambda s: s["target_window_idx"])

assert samples[0]["seq_len"] == SEQ_LEN, "SEQ_LEN 与数据不一致"

n = len(samples)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

gap = SEQ_LEN - 1

train_samples = samples[:n_train]

val_samples = samples[
    min(n_train + gap, n):
    min(n_train + gap + n_val, n)
]

test_samples = samples[
    min(val_samples[-1]["target_window_idx"] + gap + 1, n):
] if val_samples else []

print(f"[Split] train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

# =========================================================
# 标准化
# =========================================================
def collect_pool(s_list):
    pool = []
    for s in s_list:
        if s["active_mask_seq"].sum() > 0:
            pool.append(
                s["x_seq"][s["active_mask_seq"]].numpy()
            )
    return np.vstack(pool)

scaler = StandardScaler()
scaler.fit(collect_pool(train_samples))

def apply_scaler(s):
    x = s["x_seq"].clone()
    m = s["active_mask_seq"]

    if m.sum() > 0:
        x[m] = torch.tensor(
            scaler.transform(x[m].numpy()),
            dtype=torch.float
        )

    s = dict(s)
    s["x_seq"] = x
    return s

train_samples = [apply_scaler(s) for s in train_samples]
val_samples = [apply_scaler(s) for s in val_samples]
test_samples = [apply_scaler(s) for s in test_samples]

# =========================================================
# Dataset
# =========================================================
class TemporalDataset(Dataset):
    def __init__(self, s_list):
        self.samples = s_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_single(batch):
    return batch[0]

train_loader = DataLoader(
    TemporalDataset(train_samples),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_single
)

val_loader = DataLoader(
    TemporalDataset(val_samples),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_single
)

test_loader = DataLoader(
    TemporalDataset(test_samples),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_single
)

# =========================================================
# 类别权重（单输出版）
# =========================================================
def collect_labels(s_list):
    return torch.cat([
        s["y"][s["target_mask"]]
        for s in s_list
        if s["target_mask"].sum() > 0
    ], dim=0)

labels = collect_labels(train_samples)

counts = torch.bincount(labels.long(), minlength=2).float()

neg_count = counts[0]
pos_count = counts[1]

pos_weight = neg_count / pos_count.clamp(min=1.0)

print(f"[Class Counts] neg={neg_count}, pos={pos_count}")
print(f"[Pos Weight] {pos_weight.item():.4f}")

# =========================================================
# 模型
# =========================================================
model = MaskedTemporalSTGNN(
    in_dim=train_samples[0]["x_seq"].shape[-1],
    hidden=12,
    heads=2,
    dropout=0.15
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(device)
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=8
)

# =========================================================
# 评估函数（单输出版）
# =========================================================
@torch.no_grad()
def evaluate(loader):
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for sample in loader:
        sample = move_to_device(sample, device)

        logits = model(sample).squeeze(-1)

        mask = sample["target_mask"].to(device)

        if mask.sum() == 0:
            continue

        prob = torch.sigmoid(logits[mask])

        pred = (prob > 0.5).long()

        all_true.append(sample["y"][mask].cpu())
        all_pred.append(pred.cpu())
        all_prob.append(prob.cpu())

    if not all_true:
        return float("nan"), float("nan"), np.array([]), np.array([]), np.array([])

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
    except:
        auc = float("nan")

    return macro_f1, auc, y_true, y_pred, y_prob

# =========================================================
# 训练
# =========================================================
best_val_f1 = -1.0
best_state = None
bad_epochs = 0

curves = {
    "epoch": [],
    "train_loss": [],
    "val_f1": [],
    "val_auc": []
}

print(f"[Run Dir] {RUN_DIR}")

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0

    for sample in train_loader:
        sample = move_to_device(sample, device)

        optimizer.zero_grad()

        logits = model(sample).squeeze(-1)

        mask = sample["target_mask"].to(device)

        if mask.sum() == 0:
            continue

        target = sample["y"][mask].float()

        loss = criterion(
            logits[mask],
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CLIP_NORM
        )

        optimizer.step()

        total_loss += float(loss.item())

    val_f1, val_auc, _, _, _ = evaluate(val_loader)

    scheduler.step(val_f1)

    curves["epoch"].append(epoch)
    curves["train_loss"].append(total_loss)
    curves["val_f1"].append(val_f1)
    curves["val_auc"].append(val_auc)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        bad_epochs = 0
    else:
        bad_epochs += 1

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(
            f"epoch={epoch:03d} "
            f"loss={total_loss:.4f} "
            f"val_f1={val_f1:.4f} "
            f"val_auc={val_auc:.4f}"
        )

    if bad_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch}. Best val_f1={best_val_f1:.4f}")
        break

# =========================================================
# 保存最佳模型
# =========================================================
torch.save(
    best_state,
    os.path.join(RUN_DIR, "best_model.pt")
)

# =========================================================
# 测试
# =========================================================
model.load_state_dict(best_state)

test_f1, test_auc, y_true, y_pred, y_prob = evaluate(test_loader)

report = classification_report(
    y_true,
    y_pred,
    digits=4,
    zero_division=0
)

print("\n=== TEST RESULT ===")
print(f"macro_f1 = {test_f1:.4f} | roc_auc = {test_auc:.4f}")
print(report)

# =========================================================
# 保存指标
# =========================================================
metrics = {
    "seq_len": SEQ_LEN,
    "test_macro_f1": float(test_f1),
    "test_roc_auc": float(test_auc),
    "best_val_f1": float(best_val_f1),
    "class_counts": counts.tolist(),
    "pos_weight": float(pos_weight),
    "classification_report": report
}

with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(RUN_DIR, "curves.json"), "w") as f:
    json.dump(curves, f, indent=2)

print(f"\n✅ Saved to {RUN_DIR}")