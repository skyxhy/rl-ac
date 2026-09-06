import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置参数
root_dir = "outputs/simulation/"

path = [
    "single_sybil_adaptive/csv/",
    "single_sybil_test_rl_alpha0/csv/",
    "single_sybil_test_rl_alpha0.1/csv/",
    "single_sybil_test_rl_alpha0.2/csv/",
    "single_sybil_test_rl_alpha0.4/csv/",
    "single_sybil_test_rl_alpha0.8/csv/",
    "single_sybil_test_rl_alpha2/csv/",
]

csv_file_names = [
    "step_leakage_risk_ratio_sum.csv",
    "step_utility.csv",
]

# 为每个CSV文件创建一个图表
for csv_idx, csv_file in enumerate(csv_file_names):
    plt.figure(figsize=(12, 6))
    
    # 遍历每个路径，读取数据并绘制曲线
    for path_idx, sub_path in enumerate(path):
        full_path = os.path.join(root_dir, sub_path, csv_file)
        
        try:
            df = pd.read_csv(full_path)
            y = df.iloc[:, 0]
            x = range(len(y))
            
            # 平滑处理
            y_smooth = y.rolling(window=10, min_periods=1).mean()
            
            # 绘制曲线，使用不同的标签区分不同路径
            label_name = sub_path.replace("single_sybil_", "").replace("/csv/", "")
            plt.plot(x, y_smooth, label=label_name, linewidth=1.5)
            
            print(f"已读取: {full_path}, 数据点数: {len(y)}, 均值: {y.mean():.4f}")
            
        except FileNotFoundError:
            print(f"文件未找到: {full_path}")
        except Exception as e:
            print(f"读取文件出错 {full_path}: {e}")
    
    # 添加标题和标签
    plt.title(f'Comparison of {csv_file.replace(".csv", "")}', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    output_filename = f"comparison_{csv_file.replace('.csv', '.png')}"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_filename}\n")
    
    # 显示图表
    plt.show()

print("所有图表绘制完成！")

