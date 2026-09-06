import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb

# =========================================================
# 0. 参数
# =========================================================
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

INPUT_CSV = "data/samples.csv"
OUT_DIR = "outputs/xgb_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# =========================================================
# 1. 读取数据
# =========================================================
df = pd.read_csv(INPUT_CSV)

required_cols = {"window_idx", "sender_pseudo", "label"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"samples.csv 缺少必要列: {missing}")

# 自动识别特征列：排除元信息列
meta_cols = {"window_idx", "sender_pseudo", "label"}
feature_cols = [c for c in df.columns if c not in meta_cols]

# 只保留数值列
feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

if len(feature_cols) == 0:
    raise ValueError("没有找到可用特征列。")

# 清理无穷/缺失
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

print(f"num samples: {len(df)}")
print(f"num features: {len(feature_cols)}")
print("features:", feature_cols)

# =========================================================
# 2. 按 window_idx 时间顺序切分，避免泄漏
# =========================================================
unique_windows = sorted(df["window_idx"].unique())
n_w = len(unique_windows)
n_train_w = int(n_w * TRAIN_RATIO)
n_val_w = int(n_w * VAL_RATIO)
n_test_w = n_w - n_train_w - n_val_w

train_windows = unique_windows[:n_train_w]
val_windows = unique_windows[n_train_w:n_train_w + n_val_w]
test_windows = unique_windows[n_train_w + n_val_w:]

train_df = df[df["window_idx"].isin(train_windows)].copy()
val_df = df[df["window_idx"].isin(val_windows)].copy()
test_df = df[df["window_idx"].isin(test_windows)].copy()

print(f"windows: total={n_w}, train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")
print(f"samples: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# =========================================================
# 3. 构造数据矩阵
# =========================================================
X_train = train_df[feature_cols]
y_train = train_df["label"].astype(int).to_numpy()

X_val = val_df[feature_cols]
y_val = val_df["label"].astype(int).to_numpy()

X_test = test_df[feature_cols]
y_test = test_df["label"].astype(int).to_numpy()

# 类别权重
classes = np.array([0, 1])
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_map = {0: float(cw[0]), 1: float(cw[1])}
scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

print("class_weight_map:", class_weight_map)
print("scale_pos_weight:", scale_pos_weight)

# =========================================================
# 4. 训练 XGBoost
# =========================================================
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=1,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=1.0,
    colsample_bynode=1.0,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="auc",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

# =========================================================
# 5. 阈值搜索：在验证集上选择最优阈值
# =========================================================
def macro_f1_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

val_prob = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.05, 0.95, 181)
scores = []
for t in thresholds:
    scores.append(macro_f1_at_threshold(y_val, val_prob, t))

best_idx = int(np.argmax(scores))
best_threshold = float(thresholds[best_idx])
best_val_macro_f1 = float(scores[best_idx])

print("\n=== Threshold search on validation set ===")
print(f"best_threshold = {best_threshold:.4f}")
print(f"best_val_macro_f1 = {best_val_macro_f1:.4f}")

# =========================================================
# 6. 测试集评估
# =========================================================
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, zero_division=0)
rec = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
macro_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)

roc_auc = roc_auc_score(y_test, test_prob)
pr_auc = average_precision_score(y_test, test_prob)

print("\n=== TEST RESULT ===")
print(f"accuracy  = {acc:.4f}")
print(f"precision = {prec:.4f}")
print(f"recall    = {rec:.4f}")
print(f"f1        = {f1:.4f}")
print(f"macro_f1  = {macro_f1:.4f}")
print(f"roc_auc   = {roc_auc:.4f}")
print(f"pr_auc    = {pr_auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, test_pred, digits=4, zero_division=0))

# 保存测试指标
metrics = {
    "threshold": best_threshold,
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "macro_f1": float(macro_f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "best_val_macro_f1": float(best_val_macro_f1),
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

# =========================================================
# 7. 混淆矩阵、ROC、PR 曲线
# =========================================================
cm = confusion_matrix(y_test, test_pred)
cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["pred_0", "pred_1"])
plt.yticks([0, 1], ["true_0", "true_1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, test_prob)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, test_prob)
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=200)
plt.close()

# =========================================================
# 8. 特征重要性：gain / weight
# =========================================================
booster = model.get_booster()

gain_map = booster.get_score(importance_type="gain")
weight_map = booster.get_score(importance_type="weight")
cover_map = booster.get_score(importance_type="cover")

importance_df = pd.DataFrame({"feature": feature_cols})
importance_df["gain"] = importance_df["feature"].map(gain_map).fillna(0.0)
importance_df["weight"] = importance_df["feature"].map(weight_map).fillna(0.0)
importance_df["cover"] = importance_df["feature"].map(cover_map).fillna(0.0)

importance_df = importance_df.sort_values("gain", ascending=False).reset_index(drop=True)
importance_df.to_csv(os.path.join(OUT_DIR, "xgb_importance_gain_weight_cover.csv"), index=False)

print("\n=== Top feature importance by gain ===")
print(importance_df.head(15).to_string(index=False))

# 画 top20 gain
top_k = 20
plot_df = importance_df.head(top_k).iloc[::-1]

plt.figure(figsize=(10, 7))
plt.barh(plot_df["feature"], plot_df["gain"])
plt.xlabel("Gain")
plt.title(f"XGBoost Feature Importance (Top {top_k}, gain)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "xgb_gain_top20.png"), dpi=200)
plt.close()

# weight 画图
weight_df = importance_df.sort_values("weight", ascending=False).head(top_k).iloc[::-1]
plt.figure(figsize=(10, 7))
plt.barh(weight_df["feature"], weight_df["weight"])
plt.xlabel("Weight")
plt.title(f"XGBoost Feature Importance (Top {top_k}, weight)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "xgb_weight_top20.png"), dpi=200)
plt.close()

# =========================================================
# 9. Permutation Importance
# =========================================================
print("\n=== Permutation importance on test set (scoring=roc_auc) ===")
perm_auc = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_SEED,
    scoring="roc_auc",
    n_jobs=-1,
)

perm_auc_df = pd.DataFrame({
    "feature": feature_cols,
    "perm_auc_mean": perm_auc.importances_mean,
    "perm_auc_std": perm_auc.importances_std,
}).sort_values("perm_auc_mean", ascending=False).reset_index(drop=True)

print(perm_auc_df.head(15).to_string(index=False))
perm_auc_df.to_csv(os.path.join(OUT_DIR, "permutation_importance_auc.csv"), index=False)

plt.figure(figsize=(10, 7))
tmp = perm_auc_df.head(top_k).iloc[::-1]
plt.barh(tmp["feature"], tmp["perm_auc_mean"])
plt.xlabel("Permutation Importance (AUC drop)")
plt.title(f"Permutation Importance (Top {top_k}, roc_auc)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "perm_auc_top20.png"), dpi=200)
plt.close()

print("\n=== Permutation importance on test set (scoring=f1) ===")
perm_f1 = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=RANDOM_SEED,
    scoring="f1",
    n_jobs=-1,
)

perm_f1_df = pd.DataFrame({
    "feature": feature_cols,
    "perm_f1_mean": perm_f1.importances_mean,
    "perm_f1_std": perm_f1.importances_std,
}).sort_values("perm_f1_mean", ascending=False).reset_index(drop=True)

print(perm_f1_df.head(15).to_string(index=False))
perm_f1_df.to_csv(os.path.join(OUT_DIR, "permutation_importance_f1.csv"), index=False)

plt.figure(figsize=(10, 7))
tmp = perm_f1_df.head(top_k).iloc[::-1]
plt.barh(tmp["feature"], tmp["perm_f1_mean"])
plt.xlabel("Permutation Importance (F1 drop)")
plt.title(f"Permutation Importance (Top {top_k}, f1)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "perm_f1_top20.png"), dpi=200)
plt.close()

# =========================================================
# 10. 合并重要性结果
# =========================================================
merged_imp = importance_df.merge(perm_auc_df, on="feature", how="left").merge(perm_f1_df, on="feature", how="left")
merged_imp.to_csv(os.path.join(OUT_DIR, "xgb_importance_merged.csv"), index=False)

print("\nSaved files to:", OUT_DIR)
print("- metrics.json / metrics.csv")
print("- confusion_matrix.csv / confusion_matrix.png")
print("- roc_curve.png / pr_curve.png")
print("- xgb_importance_gain_weight_cover.csv")
print("- xgb_gain_top20.png / xgb_weight_top20.png")
print("- permutation_importance_auc.csv / permutation_importance_f1.csv")
print("- perm_auc_top20.png / perm_f1_top20.png")
print("- xgb_importance_merged.csv")