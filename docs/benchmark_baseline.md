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

复现：`code/` 下构造 `Env`（同 `simulation.main`）→ warmup 1 步 → 连续 30 步计时。
预期优化后数量级下降（时间索引使读窗由 O(N)→O(窗)，图特征向量化，GPU device 可选）。
