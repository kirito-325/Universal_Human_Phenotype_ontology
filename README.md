# Universal Human Phenotype Ontology (UHPO)

UHPO 通过细胞表型中间层 CPO，连接人类 HPO、哺乳动物 MPO 与细胞机制证据。仓库按研究任务分为以下 **四个模块**，每个任务的实现、配置、脚本和测试放在同一目录。

```text
01_cpo_construction/                  CPO 构建
├── cl_skeleton/                      CL 骨架与细胞目录生成
├── cl_go_linking/                     CL–GO 关系链接与范围判定
└── cell_marker_mining/                单细胞文献 Cell Marker 挖掘与特征分析

02_phenotype_cpo_mapping/              表型与 CPO 映射
├── hpo_cpo/                          HPO–CPO：人类表型文献研究 agent
├── mpo_cpo/                          MPO–CPO：逐 MGI 基因型文献研究 agent
└── review_browser/                   两端关系与来源的联合审核工具

03_mpo_hpo_mapping/                   基于 CPO 中间层的 MPO–HPO 映射
├── structural_embedding/             已弃用的结构嵌入与晚融合探索
├── cpo_profile_similarity/           既有 CPO profile 相似度与术语召回
├── semantic_cpo_experiments/         无训练语义/CPO 融合与对照实验
└── agent_alignment/                  新 agent 的先导输入与调研脚手架

04_downstream_evaluation/             下游任务与评估
├── gene_prioritization/              CellPhenoRank 罕见病候选基因排序
├── mouse_model_prioritization/       具体小鼠模型优先级与冻结基准
├── single_cell_validation/           单细胞病例选择、供体层验证与结果分析
└── cpo_disease_gene_prediction/       CPO 矩阵驱动的疾病模型/基因预测与评估
```

| 模块 | 内容与入口 | 当前状态 |
| --- | --- | --- |
| [01 CPO 构建](01_cpo_construction/README.md) | CL 骨架、GO–CL 判定、文献 Marker | 三个任务分别维护 |
| [02 表型–CPO 映射](02_phenotype_cpo_mapping/README.md) | HPO、MPO 两个独立 agent | 当前术语调研主线 |
| [03 MPO–HPO 映射](03_mpo_hpo_mapping/README.md) | 中间层比较与历史探索 | **尚未同步最新 agent 的端到端输出** |
| [04 下游任务](04_downstream_evaluation/README.md) | 基因排序、模型排序、单细胞验证和 CPO 预测 | 保留代码与初步结果说明 |

`shared/` 仅存放跨任务复用的客户端与路径定义；`docs/` 保存数据契约、研究状态及重构记录；`scripts/` 提供仓库检查和测试入口。根目录不再保留 `CPO_generize`、`mission3`、`CPO_mapping` 或 `legacy` 等混合工作区。

## 安装

科学计算模块通过仓库根目录的 editable 安装共享 Python 包，不依赖旧目录名或散落的 `PYTHONPATH` 配置。Agent 独立安装，需 Python 3.12+。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[scientific,test]'
python -m pip install -e ./02_phenotype_cpo_mapping/hpo_cpo -e ./02_phenotype_cpo_mapping/mpo_cpo
cp .env.example .env
```

在本地 `.env` 配置模型服务。真实文献调研可能产生 API 费用；离线单元测试使用替身。模型编码或历史嵌入实验另外安装 `.[embedding]`。

## 验证与复现

```bash
python scripts/check_repository.py
python scripts/run_tests.py --group scientific
python scripts/run_tests.py --group agents
```

首次体验可运行 [基因排序合成示例](04_downstream_evaluation/gene_prioritization/README.md#合成示例)；完整任务命令见对应 README。根目录 `pyproject.toml` 是跨任务 Python 包的统一注册表，原科学包名保留为稳定导入接口。本轮重构后 **578 项离线测试通过**。

本仓库只保存代码、配置、schema、测试、小型合成示例和研究摘要；真实数据、文献数据库、索引、模型权重及运行结果不上传。数据位置和版本要求见 [数据指南](docs/data.md)。

当前 agent 的丰富证据记录还没有转换并重新接入第三、四模块的历史 TSV 接口。共享 CL/GO ID 不代表术语等价，旧实验数值也不代表最新 agent 的效果。详见 [研究状态](docs/status.md)、[验证记录](docs/validation.md) 和 [重构说明](docs/provenance.md)。

第三方代码保留原许可与 notices；数据许可由相应来源规定，本次未新增统一开源许可证。
