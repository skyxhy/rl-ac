import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ..data.feature_config import FEATURE_NAMES

samples = pd.read_csv("data/samples.csv")

OUT_DIR = "outputs/pca_outputs/"

feature_df = samples[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = samples["label"].to_numpy()

constant_cols = [c for c in feature_df.columns if feature_df[c].nunique(dropna=False) <= 1]
if constant_cols:
    print("Constant features:", constant_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df.values)

n_components = min(10, X_scaled.shape[1])
pca = PCA(n_components=n_components, random_state=42)
Z = pca.fit_transform(X_scaled)

print("\n=== PCA explained variance ratio ===")
for i, r in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {r:.4f}")
print("cumulative:", np.cumsum(pca.explained_variance_ratio_))

loadings = pd.DataFrame(
    pca.components_.T,
    index=FEATURE_NAMES,
    columns=[f"PC{i}" for i in range(1, n_components + 1)]
)

print("\n=== Top loading features ===")
for pc in loadings.columns[:3]:
    s = loadings[pc].abs().sort_values(ascending=False).head(8)
    print(f"\n[{pc}]")
    for feat_name, _ in s.items():
        sign = loadings.loc[feat_name, pc]
        print(f"{feat_name:20s} loading={sign:+.4f}")

def plot_pca_scatter(Z2, y2, title, out_path):
    plt.figure(figsize=(8, 6))
    labels = sorted(np.unique(y2))
    for lab in labels:
        idx = y2 == lab
        plt.scatter(Z2[idx, 0], Z2[idx, 1], s=10, alpha=0.55, label=f"class {lab}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_variance(pca_obj, out_path):
    ratios = pca_obj.explained_variance_ratio_
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(ratios) + 1), ratios)
    plt.plot(range(1, len(ratios) + 1), np.cumsum(ratios), marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if Z.shape[1] >= 2:
    plot_pca_scatter(Z, y, "PCA Scatter", OUT_DIR+"pca_scatter.png")
plot_variance(pca, OUT_DIR+"pca_variance.png")

loadings.reset_index().rename(columns={"index": "feature"}).to_csv(OUT_DIR+"pca_loadings.csv", index=False)
pd.DataFrame(Z, columns=[f"PC{i}" for i in range(1, n_components + 1)]).assign(label=y).to_csv(OUT_DIR+"pca_scores.csv", index=False)
feature_df.assign(label=y).to_csv(OUT_DIR+"feature_matrix_scaled_like_input.csv", index=False)

print("\nSaved:")
print("- pca_loadings.csv")
print("- pca_scores.csv")
print("- pca_scatter.png")
print("- pca_variance.png")

print("\n=== Train-like class centroids in PCA space ===")
for lab in sorted(np.unique(y)):
    c = Z[y == lab, :2].mean(axis=0)
    print(f"class {lab}: PC1={c[0]:+.4f}, PC2={c[1]:+.4f}")