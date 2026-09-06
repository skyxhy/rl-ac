# view.py
import os, sys, json, argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main():
    parser = argparse.ArgumentParser(description="Visualize GNN training results")
    parser.add_argument("run_dir", type=str, help="Path to the timestamped run directory")
    args = parser.parse_args()

    curves_path = os.path.join(args.run_dir, "curves.json")
    metrics_path = os.path.join(args.run_dir, "metrics.json")

    if not (os.path.exists(curves_path) and os.path.exists(metrics_path)):
        print("❌ Error: curves.json or metrics.json not found in the directory.")
        sys.exit(1)

    with open(curves_path) as f: curves = json.load(f)
    with open(metrics_path) as f: metrics = json.load(f)

    epochs = curves["epoch"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    # 1. Loss
    ax1.plot(epochs, curves["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax1.set_title("Training Dynamics", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Val Metrics
    ax2.plot(epochs, curves["val_f1"], "r-o", linewidth=2, markersize=4, label="Val Macro F1")
    ax2.plot(epochs, curves["val_auc"], "g-s", linewidth=2, markersize=4, label="Val ROC AUC")
    ax2.set_ylabel("Metric Value")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 保存图像
    plot_path = os.path.join(args.run_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"📈 Saved plot to {plot_path}")
    plt.show()

    # 打印终端报告
    print("\n" + "="*40)
    print("📊 FINAL METRICS SUMMARY")
    print("="*40)
    for k, v in metrics.items():
        if k != "classification_report":
            print(f"  {k:<20}: {v}")
    print("\n📋 CLASSIFICATION REPORT")
    print(metrics.get("classification_report", "Not available."))
    print("="*40)

if __name__ == "__main__":
    main()