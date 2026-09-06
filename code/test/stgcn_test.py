import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import argparse

from ..model.stgcn_model import MaskedTemporalSTGNN

# =========================================================
# 参数
# =========================================================
RANDOM_SEED = 42
SEQ_LEN = 10
BATCH_SIZE = 1
INPUT_PATH = "data/stgcn_samples_masked.pt"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def collect_pool(s_list):
    pool = []
    for s in s_list:
        if s["active_mask_seq"].sum() > 0:
            pool.append(
                s["x_seq"][s["active_mask_seq"]].numpy()
            )

    if not pool:
        return np.empty((0, s_list[0]["x_seq"].shape[-1]))

    return np.vstack(pool)

def apply_scaler(s, scaler):
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

class TemporalDataset(Dataset):
    def __init__(self, s_list):
        self.samples = s_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_single(batch):
    return batch[0]

# =========================================================
# 单输出评估函数
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
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

        target_logits = logits[mask]

        prob = torch.sigmoid(target_logits)

        pred = (prob > threshold).long()

        all_true.append(sample["y"][mask].cpu())
        all_pred.append(pred.cpu())
        all_prob.append(prob.cpu())

    if not all_true:
        print("[Warning] No valid targets found.")
        return (
            float("nan"),
            float("nan"),
            np.array([]),
            np.array([]),
            np.array([])
        )

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
    except ValueError:
        auc = float("nan")

    return macro_f1, auc, y_true, y_pred, y_prob

# =========================================================
# 主函数
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="run directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="classification threshold"
    )

    args = parser.parse_args()

    run_dir = args.run_dir
    threshold = args.threshold

    model_path = os.path.join(run_dir, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    print(f"[Info] Loading data from {INPUT_PATH}")

    # =====================================================
    # 加载数据
    # =====================================================
    samples = torch.load(INPUT_PATH, weights_only=False)

    print(f"[Data] Loaded {len(samples)} samples.")

    samples = sorted(
        samples,
        key=lambda s: s["target_window_idx"]
    )

    assert samples[0]["seq_len"] == SEQ_LEN

    # =====================================================
    # 重建测试集
    # =====================================================
    n = len(samples)

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    gap = SEQ_LEN - 1

    train_samples_raw = samples[:n_train]

    val_samples_raw = samples[
        min(n_train + gap, n):
        min(n_train + gap + n_val, n)
    ]

    if val_samples_raw:
        test_start_idx = min(
            val_samples_raw[-1]["target_window_idx"] + gap + 1,
            n
        )
    else:
        test_start_idx = n

    test_samples_raw = samples[test_start_idx:]

    print(f"[Split] test={len(test_samples_raw)}")

    # =====================================================
    # 标准化
    # =====================================================
    scaler = StandardScaler()

    scaler.fit(
        collect_pool(train_samples_raw)
    )

    test_samples = [
        apply_scaler(s, scaler)
        for s in test_samples_raw
    ]

    test_loader = DataLoader(
        TemporalDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_single
    )

    # =====================================================
    # 初始化模型
    # =====================================================
    in_dim = test_samples[0]["x_seq"].shape[-1]

    model = MaskedTemporalSTGNN(
        in_dim=in_dim,
        hidden=12,
        heads=2,
        dropout=0.15
    ).to(device)

    print(f"[Model] Loading {model_path}")

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    # =====================================================
    # 测试
    # =====================================================
    print(f"[Eval] threshold={threshold}")

    test_f1, test_auc, y_true, y_pred, y_prob = evaluate(
        model,
        test_loader,
        device,
        threshold=threshold
    )

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")
    print(f"Macro F1 : {test_f1:.4f}")
    print(f"ROC AUC  : {test_auc:.4f}")
    print(report)

    # =====================================================
    # 保存
    # =====================================================
    result = {
        "threshold": threshold,
        "test_macro_f1": float(test_f1),
        "test_roc_auc": float(test_auc),
        "classification_report": report
    }

    result_path = os.path.join(
        run_dir,
        "test_result_final.json"
    )

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Saved to {result_path}")

if __name__ == "__main__":
    main()