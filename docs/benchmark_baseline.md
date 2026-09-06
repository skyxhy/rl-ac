# 性能基线（重构前，`code/` 包）

> 用于量化重构收益；重构后用同一脚本/数据重测并对比。

| 项 | 值 |
|---|---|
| 日期 | 2026-09-06 |
| 机器 | 本机 Windows（myenv py3.13，CPU） |
| 数据 | `sybil.csv`（3.34M 行） |
| 场景 | fixed 攻击, action_dim=3, sight=10, window=1s, GNN embed on CPU |
| 指标 | 单次 `env.step` 平均耗时 |
| **旧代码（code/）** | **~237 ms/step**（30 步实测，含读窗 O(N) 扫描 + 图构建 + GNN embed） |
| 新 `rlac` 环境（首轮） | **~193 ms/step**（同机同数据 30 步；读窗 O(N)→O(窗)，向量化权重，groupby reward；剩余主要在图特征构建 + CPU GNN embed） |

复现：`code/` 与 `rlac/` 各构造 `Env`（同 `simulation.main` 装配）→ warmup 1 步 → 连续 30 步计时。
等价性：新/旧 Env 在 fixed intensity=1 下 20 步 reward/utility **逐位一致**（max diff = 0）。
进一步提速方向：图特征向量化（`rlac.data.builder` 待做）、GNN embed 用 GPU（device=cuda）。
