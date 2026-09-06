import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ..data.feature_config import FEATURE_NAMES

samples = pd.read_csv("data/samples.csv")

OUT_DIR = "outputs/correlation_outputs/"

feature_df = samples[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
label = samples["label"]

constant_cols = [c for c in feature_df.columns if feature_df[c].nunique(dropna=False) <= 1]
if constant_cols:
    print("Constant features:", constant_cols)

pearson_corr = feature_df.corr(method="pearson")
spearman_corr = feature_df.corr(method="spearman")

def plot_heatmap(corr_df, title, out_path):
    mat = corr_df.fillna(0.0).values
    plt.figure(figsize=(14, 12))
    im = plt.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, label="correlation")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_df.index)), corr_df.index, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

plot_heatmap(pearson_corr, "Pearson Correlation Heatmap", OUT_DIR+"pearson_corr.png")
plot_heatmap(spearman_corr, "Spearman Correlation Heatmap", OUT_DIR+"spearman_corr.png")

# 与 label 的相关性
label_corr = feature_df.copy()
label_corr["label"] = label
label_corr = label_corr.corr(method="pearson")["label"].sort_values(ascending=False)

print("\n=== Correlation with Label ===")
print(label_corr)

label_corr.to_csv(OUT_DIR+"label_corr.csv")

plt.figure(figsize=(8, 10))
label_corr.drop("label").sort_values().plot(kind="barh")
plt.title("Feature Correlation with Label")
plt.tight_layout()
plt.savefig(OUT_DIR+"label_corr.png", dpi=200)
plt.close()

# 高相关特征对
def top_corr_pairs(corr_df, threshold=0.85, top_n=50):
    pairs = []
    cols = list(corr_df.columns)
    mat = corr_df.fillna(0.0).values
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = mat[i, j]
            if abs(r) >= threshold:
                pairs.append((cols[i], cols[j], r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_n]

pearson_pairs = top_corr_pairs(pearson_corr, threshold=0.85)
spearman_pairs = top_corr_pairs(spearman_corr, threshold=0.85)

print("\n=== Highly Correlated Feature Pairs |Pearson| >= 0.85 ===")
if not pearson_pairs:
    print("None")
else:
    for a, b, r in pearson_pairs:
        print(f"{a:20s} <-> {b:20s}  corr={r:+.4f}")

print("\n=== Highly Correlated Feature Pairs |Spearman| >= 0.85 ===")
if not spearman_pairs:
    print("None")
else:
    for a, b, r in spearman_pairs:
        print(f"{a:20s} <-> {b:20s}  corr={r:+.4f}")

# VIF
def compute_vif(df_features: pd.DataFrame):
    X = df_features.values
    n_features = X.shape[1]
    vifs = []

    for i in range(n_features):
        y_i = X[:, i]
        X_other = np.delete(X, i, axis=1)

        if np.std(y_i) < 1e-12:
            vifs.append(np.inf)
            continue

        model = LinearRegression()
        model.fit(X_other, y_i)
        r2 = model.score(X_other, y_i)

        if r2 >= 1.0 - 1e-12:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)
        vifs.append(vif)

    return pd.DataFrame({"feature": df_features.columns, "VIF": vifs}).sort_values("VIF", ascending=False).reset_index(drop=True)

vif_df = compute_vif(feature_df)
print("\n=== Top VIF features ===")
print(vif_df.head(15).to_string(index=False))

vif_df.to_csv(OUT_DIR+"vif.csv", index=False)
pearson_corr.to_csv(OUT_DIR+"pearson_corr.csv")
spearman_corr.to_csv(OUT_DIR+"spearman_corr.csv")

# VIF 可视化
top_vif = vif_df.head(20).iloc[::-1]
plt.figure(figsize=(10, 8))
plt.barh(top_vif["feature"], top_vif["VIF"])
plt.xlabel("VIF")
plt.title("Top 20 VIF Features")
plt.tight_layout()
plt.savefig(OUT_DIR+"vif_top20.png", dpi=200)
plt.close()

# 自动分析文本
high_vif = vif_df[vif_df["VIF"] >= 10.0]
summary = []
summary.append("Correlation analysis summary")
summary.append(f"Constant features: {len(constant_cols)}")
summary.append(f"High Pearson pairs (|corr|>=0.85): {len(pearson_pairs)}")
summary.append(f"High Spearman pairs (|corr|>=0.85): {len(spearman_pairs)}")
summary.append(f"High VIF features (VIF>=10): {len(high_vif)}")
summary.append("")
summary.append("Top label correlations:")
for k, v in label_corr.drop("label").head(10).items():
    summary.append(f"{k}: {v:.4f}")

with open(OUT_DIR+"correlation_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\nSaved:")
print("- pearson_corr.png")
print("- spearman_corr.png")
print("- label_corr.png")
print("- label_corr.csv")
print("- pearson_corr.csv")
print("- spearman_corr.csv")
print("- vif.csv")
print("- vif_top20.png")
print("- correlation_summary.txt")