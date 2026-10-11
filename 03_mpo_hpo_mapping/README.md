# 基于 CPO 的 MPO–HPO 映射

**本模块尚未同步最新 HPO/MPO agent 的完整输出。** 当前保留以前探索过的方法、对照实验，以及新 agent 的先导调研脚手架。

| 任务 | 方法 | 状态 |
| --- | --- | --- |
| [structural_embedding](structural_embedding/README.md) | Poincare CL/GO 嵌入、语义/结构双塔及晚融合 | **方法落后，已弃用，不采用，待更新** |
| [cpo_profile_similarity](cpo_profile_similarity/README.md) | CPO profile、方向、继承与术语相似度矩阵 | 既有探索性实现 |
| [semantic_cpo_experiments](semantic_cpo_experiments/README.md) | 无训练语义/CPO 融合及预设对照 | 探索验证；不支持总体机制收益声明 |
| [agent_alignment](agent_alignment/README.md) | 独立两端输入、调研运行与状态报告 | 先导调研暂停；完整对接尚未完成 |

术语映射算法放在本模块；消费映射结果的疾病模型评分、候选基因预测和单细胞验证放在第四模块。重构调整组织与接口，没有重新训练历史模型或重算真实映射。
