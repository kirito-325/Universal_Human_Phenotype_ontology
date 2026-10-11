# 数据与配置

本次不发布真实数据。保留的小型 `examples/`、模板和决策规则属于运行契约；不包含真实文献全文、模型数据库或评测队列。

| 模块 | 本地需要准备 | 读取方式 |
| --- | --- | --- |
| HPO Agent | CL/GO OBO、HPO OBO、`phenotype.hpoa` | `build-index` 和 `build-hpo-context`，见模块 README |
| MPO Agent | CL/GO OBO、MP OBO、`MGI_PhenoGenoMP.rpt` | `build-index` 和 `build-mp-context`，见模块 README |
| CL 目录 | `CPO_construction/cell_lineage/cell_ontology/cl-basic.obo` | 骨架脚本与文献目录构建器 |
| 文献 Marker | PubMed/论文元信息、补充表格、原始 CL、审核目录 | `marker_literuture/config/` 与 CLI 参数 |
| GO–CL 构建 | GO/CL、GOA、HGNC、审核目录、语义及审核中间表 | 各模块 `--help`；`GO_link` 汇总依赖前序证据，不是从零下载入口 |
| 下游验证 | 冻结疾病队列、模型/表型、CPO 关系、基线矩阵、orthology、独立 gold | 各模块 `config/` / `configs/` |
| 历史嵌入 | HPO/MPO 机制 TSV、专家映射、本体、CL 目录 | `legacy/MPO_HPO_mapping/scripts/run_pipeline.py --help` |

参考源为 [HPO](https://hpo.jax.org/)、[MGI 数据报告](https://www.informatics.jax.org/downloads/reports/index.html)、[Cell Ontology](https://github.com/obophenotype/cell-ontology)、[Gene Ontology](https://geneontology.org/)、[HGNC](https://www.genenames.org/) 与 [GOA](https://www.ebi.ac.uk/GOA/)。请固定所用版本和 SHA-256，遵守各数据源许可；当前在线最新版本不自动等于原实验快照。

## 本地路径

整理时已移除原机器的用户名和绝对路径，未能在无数据条件下迁移的实验路径统一表示为 `/path/to/workspace` 或 `/path/to/home`。这些是需要配置的占位路径，不能直接用于完整实验。

优先配置 CLI 参数或配置文件；`CELL_LINEAGE_SOURCE_ROOT` 可覆盖目录构建代码位置。模型密钥放在忽略的 `.env` 或导出的环境变量中；Agent 示例读取 `CPO_mapping/.env`。共享 ECNU 客户端读取 `ECNU_API_KEY`；公开模型服务地址不是凭据。

`data/`、`database/`、`work/`、`results/`、`outputs/`、`runs/`、缓存、权重及常见数据文件格式均被 `.gitignore` 排除。文档中提到的结果文件是本地生成物，未上传。
