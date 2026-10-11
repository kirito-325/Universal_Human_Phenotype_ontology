# 研究状态

快照日期：2026-10-11。本轮只重构代码组织、导入、配置与运行入口，没有执行付费调研或重新计算真实数据结果。

| 模块 | 状态与限制 |
| --- | --- |
| CPO 构建 | 骨架、GO–CL 与文献 Marker 组件分别维护；真实资源不上传 |
| HPO/MPO–CPO | 当前 agent 主线；两端独立调研并保留来源与条件 |
| MPO–HPO | 旧探索仍在；结构嵌入明确弃用；尚未接入最新 agent |
| 双端先导 | 20 个术语的调研暂停，完整对接未运行；候选机制不是 gold positive |
| 下游评估 | 历史快照结果与初步验证，不代表当前 agent 的效果 |

无训练语义/CPO 开发实验仍建议纯语义 P0 为默认；探索性 MRR 增量 +0.000476 的置信区间跨零，不能据此宣称机制收益。

源实验说明见 [基因排序结果](../04_downstream_evaluation/gene_prioritization/CellPhenoRank_current_results.md)、[单细胞验证结果](../04_downstream_evaluation/single_cell_validation/docs/CellPhenoRank_single_cell_validation_results.md) 和 [小鼠模型排序状态](../04_downstream_evaluation/mouse_model_prioritization/REAL_DATA_STATUS.md)。
