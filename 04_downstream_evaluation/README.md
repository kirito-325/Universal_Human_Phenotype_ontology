# 下游任务与评估

| 任务 | 实际实现 | 目标 |
| --- | --- | --- |
| [gene_prioritization](gene_prioritization/README.md) | `src/cellphenorank/` | 罕见病候选基因排序、无参数 CPO 重排及基线 |
| [mouse_model_prioritization](mouse_model_prioritization/README.md) | `src/mission3/` | 具体 MGI 基因型优先级、冻结队列与机制消融 |
| [single_cell_validation](single_cell_validation/README.md) | `src/cellphenorank_validation/` | 候选病例、metadata 预检、冻结终点与供体层验证 |
| [cpo_disease_gene_prediction](cpo_disease_gene_prediction/README.md) | `src/uhpo_prediction/` | 消费第三模块矩阵，计算疾病模型与基因排名并评估 |

`cellphenorank` 与 `mission3` 是保留的 Python 包名，不再是仓库顶层目录。单细胞代码、配置、数据准备脚本、测试和研究摘要已从 CellPhenoRank 主目录拆出。

冻结疾病队列与 gold 必须独立于中间层构建。缺少 CPO 证据保留 unknown，不能作为阴性；旧实验摘要对应原数据快照，不代表最新 agent 的效果。
