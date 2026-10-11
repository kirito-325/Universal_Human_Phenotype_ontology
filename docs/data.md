# 数据与配置

代码仓库不包含真实本体、论文、评测队列、模型权重、索引或运行产物。小型合成 `examples/` 和明确的规则表是代码契约。

| 资源 | 默认位置 |
| --- | --- |
| CL 原始本体与目录生成物 | `01_cpo_construction/cl_skeleton/cell_ontology/` |
| 文献数据库、审核目录、名称映射和 Marker | `01_cpo_construction/cell_marker_mining/` 下的忽略目录 |
| GO/CL 证据及链接产物 | `01_cpo_construction/cl_go_linking/` 各组件的 `work/`、`results/` |
| HPO、MP、GO 本体 | 根目录 `data/ontologies/hp.obo`、`mp.obo`、`go-basic.obo` |
| GOA、HGNC、MGI、专家映射 | 根目录 `data/reference/`，各 CLI 支持覆盖 |
| SapBERT 本地快照 | 根目录 `data/models/sapbert/` |
| 两个 agent 的版本化索引与新结果 | 各自任务的 `data/` 与 `runs/` |
| 旧 HPO/MPO-CPO TSV | 各自任务的 `data/hpo_cpo/`、`data/mpo_cpo/` |
| 既有冻结模型队列 | `04_downstream_evaluation/mouse_model_prioritization/data/benchmark/` |
| 下游真实数据与结果 | 各任务自己的 `data/`、`outputs/` |

数据来源为 HPO、MGI、Cell Ontology、Gene Ontology、GOA、HGNC 和原论文。必须自行准备并固定版本、SHA-256 及许可；当前在线版本不自动等于原实验快照。

## 配置规则

根目录 editable 安装提供稳定的 Python 导入；默认跨任务资源使用 `shared/uhpo_common/paths.py` 的路径。相似度和无训练实验的 `project_root` 相对配置所属任务目录解析，换工作目录不会改变输入位置。CellPhenoRank 配置相对其 `configs/` 的上级任务目录解析。

配置中残留的 `/path/to/home` 等模型缓存占位符需要按本地环境设置；CLI 参数可显式覆盖输入。根目录 `.env` 保存模型服务配置，Agent 从其任务目录运行时使用 `--env-file ../../.env`。不要提交任何真实密钥。

`dataset_tools/` 的 `data/legacy_inputs/` 保存旧流程的本地输入。它们只处理旧 TSV 数据，不自动消费 agent 的丰富 JSONL。第三和第四模块仍需明确的转换器与重新评测，不能省略来源、方向、条件、模型身份或覆盖状态。
