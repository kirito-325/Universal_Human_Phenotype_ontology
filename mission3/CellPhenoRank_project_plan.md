# CellPhenoRank 项目计划

> 文档状态：原始设计稿。当前正式无参数重排请以
> `CellPhenoRank_parameter_free_reranking_revision_plan.md` 为准；连续加权旧实现见
> `CellPhenoRank_project_description_v2.md`。

> 基于跨物种表型相似性、HPO–CPO 映射和细胞功能基因集的罕见病致病基因预测与单细胞验证

---

## 1. 项目概述

### 1.1 项目背景

现有的小鼠模型致病基因预测方法通常采用以下思路：

1. 将罕见病表示为一组 HPO 表型；
2. 将小鼠敲除或突变模型表示为一组 MPO 表型；
3. 通过 HPO–MPO 跨物种相似度矩阵计算疾病与小鼠模型的表型相似度；
4. 将高相似度小鼠模型对应的人类同源基因作为候选致病基因。

传统 HPO–MPO 映射通常依赖表型术语的解剖实体、异常性质和方向等语义成分，能够完成整体表型层面的跨物种对齐，但其输出通常只能说明：

> 某个小鼠模型与某种人类疾病在表型上相似。

它较难进一步解释：

- 该疾病具体涉及哪些细胞类型；
- 哪些细胞功能发生异常；
- 为什么某个候选基因在对应细胞和功能背景下具有更强的致病合理性。

本项目已有以下资源：

1. HPO 与 CPO 之间的映射关系；
2. CPO 中的细胞类型信息；
3. 每种细胞类型的 marker gene set；
4. 每种“细胞类型–细胞功能”组合对应的 function gene set；
5. 正在构建的、基于“细胞类型 + 细胞表型异常”表示的 HPO–MPO 新相似度矩阵。

由于新的 HPO–MPO 相似度矩阵尚未完全实现，本项目首先使用现有 HPO–MPO 相似度矩阵完成其余流程，并将相似度矩阵封装为可替换模块。待新矩阵完成后，只需替换输入，不需要重写下游算法。

---

### 1.2 项目核心目标

构建一个可解释的罕见病致病基因优先级排序系统，完成以下三个层次的任务：

#### 目标 A：复现跨物种表型相似性基线

使用现有 HPO–MPO 相似度矩阵，根据疾病 HPO 集合与小鼠模型 MPO 集合的相似程度，对小鼠模型及其对应基因进行排序。

#### 目标 B：利用细胞表型知识重排候选基因

将疾病 HPO 投影至 CPO，获得疾病相关的细胞类型和细胞功能画像，并利用 marker gene set 与 function gene set 对候选基因进行机制化重排。

#### 目标 C：利用独立单细胞数据验证预测机制

选择具有公开患者与对照单细胞数据的罕见病，验证系统预测的关键细胞类型和细胞功能是否确实在患者样本中发生异常。

---

### 1.3 项目核心假设

本项目需要验证以下假设：

> 在传统 HPO–MPO 表型相似性预测的基础上，加入 HPO–CPO 映射、细胞 marker 和细胞功能 gene set，可以为候选致病基因提供额外的细胞背景和功能机制证据，从而提高已知致病基因的排序，并生成可被单细胞数据验证的机制解释。

需要特别区分三个贡献来源：

1. **原始 HPO–MPO 相似度矩阵的贡献**；
2. **HPO–CPO 与 gene set 重排模块的贡献**；
3. **新的细胞表型 HPO–MPO 相似度矩阵的贡献**。

因此，项目必须采用模块化和消融实验设计，避免将三类贡献混在一起。

---

## 2. 项目任务定义

### 2.1 输入

对于每个罕见病，输入包括：

- disease ID；
- disease name；
- 一组 HPO 表型；
- 可选的 HPO 频率、核心表型、起病时间和方向信息。

对于每个小鼠模型，输入包括：

- model ID；
- mouse gene；
- human ortholog；
- allele type；
- zygosity；
- conditional status；
- 一组异常 MPO 表型；
- 可选的正常检测、未检测、性别和年龄信息。

额外知识资源包括：

- HPO–MPO 相似度矩阵；
- HPO–CPO 映射；
- CPO 细胞类型；
- CPO 细胞功能；
- cell marker gene set；
- cell-function gene set；
- 人鼠同源基因映射；
- 独立疾病单细胞数据。

---

### 2.2 输出

系统输出应包括：

1. 疾病–小鼠模型相似度排名；
2. 疾病–候选基因基础排名；
3. 加入 CPO、marker 和 function 证据后的最终基因排名；
4. 每个候选基因的可解释证据链；
5. 基准评估、消融实验和随机对照结果；
6. 单细胞案例中的细胞和功能验证结果；
7. 可追溯的运行配置、数据版本和输入哈希。

建议的解释链格式：

```text
Disease
→ HPO
→ matched MPO in mouse model
→ HPO-linked CPO
→ predicted cell type
→ predicted cell function abnormality
→ marker/function gene evidence
→ candidate gene
```

---

## 3. 总体技术架构

```text
                         ┌────────────────────────────┐
                         │ Rare disease HPO profile   │
                         └──────────────┬─────────────┘
                                        │
                                        ▼
                         ┌────────────────────────────┐
                         │ HPO–MPO SimilarityProvider │
                         │ current: existing matrix   │
                         │ future: cell-based matrix  │
                         └──────────────┬─────────────┘
                                        │
                                        ▼
                    ┌──────────────────────────────────────┐
                    │ Disease–mouse-model phenotype score │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ Mouse-model to human-gene ranking │
                     └─────────────────┬─────────────────┘
                                       │
                         Base candidate gene ranking
                                       │
               ┌───────────────────────┴────────────────────────┐
               │                                                │
               ▼                                                ▼
   ┌─────────────────────────┐                   ┌─────────────────────────┐
   │ HPO → CPO projection    │                   │ Marker/function genes   │
   │ cell + abnormal process │                   │ of each cell/CPO        │
   └────────────┬────────────┘                   └────────────┬────────────┘
                │                                             │
                └──────────────────┬──────────────────────────┘
                                   ▼
                   ┌────────────────────────────────┐
                   │ Candidate-specific CellPathScore│
                   └────────────────┬───────────────┘
                                    │
                                    ▼
                       Final candidate gene ranking
                                    │
                                    ▼
                  Independent single-cell case validation
```

---

## 4. 关键设计原则

1. **矩阵可替换**：当前使用原始矩阵，未来可直接替换为新矩阵或混合矩阵。
2. **候选生成与重排分离**：基础表型相似度负责召回候选，CPO 负责机制化重排。
3. **模型级结果优先**：先计算具体小鼠基因型模型的得分，再聚合到基因，避免丢失等位基因和表型信息。
4. **缺失证据保持中性**：缺少 marker、function 或 CPO 证据时，不应直接惩罚候选基因。
5. **function 权重高于 marker**：marker 主要表示细胞定位，function 更接近致病机制。
6. **不要求致病基因差异表达**：单细胞验证重点是细胞和功能模块异常，而非致病基因 mRNA 必须显著变化。
7. **避免循环验证**：测试疾病的已知致病基因不能通过构建 HPO–CPO 映射的同一证据直接泄漏到重排结果中。
8. **所有结果可审计**：每个总分必须能够分解为 HPO–MPO、HPO–CPO、marker 和 function 等具体贡献。
9. **固定随机种子和配置**：所有随机对照和参数搜索均必须可复现。
10. **不以单个案例代替基准评估**：单细胞案例用于解释机制，整体性能仍需通过疾病集合评估。

---

## 5. 软件实现形式

建议实现为 Python 包和命令行工具。

### 5.1 环境

- Python 3.10+
- Linux 和 macOS
- 使用 `pyproject.toml` 管理依赖
- 支持 TSV、JSON、JSONL、Parquet、PNG、SVG、HTML 和 Markdown 输出

### 5.2 推荐依赖

核心依赖：

- `pandas`
- `numpy`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `pydantic`
- `pyyaml`
- `typer`
- `networkx`
- `matplotlib`

本体相关：

- `pronto` 或 `owlready2`

单细胞可选依赖：

- `anndata`
- `scanpy`

测试：

- `pytest`
- `pytest-cov`

---

## 6. 推荐仓库结构

```text
CellPhenoRank/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile
├── configs/
│   ├── baseline.yaml
│   ├── rerank.yaml
│   ├── benchmark.yaml
│   ├── strict_masking.yaml
│   ├── single_cell_case.yaml
│   └── schema_description.md
├── data/
│   ├── raw/
│   │   ├── diseases/
│   │   ├── mouse_models/
│   │   ├── orthologs/
│   │   ├── similarity_matrices/
│   │   ├── hpo_cpo/
│   │   ├── gene_sets/
│   │   └── single_cell/
│   ├── processed/
│   └── example/
├── src/
│   └── cellphenorank/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── schemas.py
│       ├── io.py
│       ├── gene_ids.py
│       ├── ontology.py
│       ├── qc.py
│       ├── similarity/
│       │   ├── base.py
│       │   ├── matrix_provider.py
│       │   ├── cell_phenotype_provider.py
│       │   └── hybrid_provider.py
│       ├── phenotype_scoring/
│       │   ├── pair_scoring.py
│       │   ├── model_scoring.py
│       │   └── normalization.py
│       ├── gene_ranking/
│       │   ├── model_to_gene.py
│       │   ├── base_ranker.py
│       │   └── rank_fusion.py
│       ├── cpo/
│       │   ├── projection.py
│       │   ├── disease_profile.py
│       │   └── candidate_profile.py
│       ├── gene_sets/
│       │   ├── marker_score.py
│       │   ├── function_score.py
│       │   ├── specificity.py
│       │   └── network_extension.py
│       ├── reranking/
│       │   ├── cellpath_score.py
│       │   ├── reranker.py
│       │   └── explanations.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── cohort.py
│       │   ├── ablation.py
│       │   ├── stratified.py
│       │   └── statistics.py
│       ├── leakage_control/
│       │   ├── gene_masking.py
│       │   ├── evidence_audit.py
│       │   └── temporal_split.py
│       ├── null_models/
│       │   ├── shuffle_hpo_cpo.py
│       │   ├── shuffle_gene_sets.py
│       │   └── shuffle_cell_labels.py
│       ├── single_cell/
│       │   ├── label_mapping.py
│       │   ├── pseudobulk.py
│       │   ├── composition.py
│       │   ├── module_scoring.py
│       │   └── validation.py
│       ├── plotting/
│       ├── report.py
│       └── utils.py
├── scripts/
│   ├── prepare_data.py
│   ├── generate_example_data.py
│   └── validate_inputs.py
├── workflows/
│   ├── run_baseline.sh
│   ├── run_reranking.sh
│   ├── run_benchmark.sh
│   ├── run_null_models.sh
│   └── run_single_cell_case.sh
├── tests/
│   ├── test_schemas.py
│   ├── test_similarity_provider.py
│   ├── test_model_score.py
│   ├── test_gene_aggregation.py
│   ├── test_cpo_projection.py
│   ├── test_gene_set_scores.py
│   ├── test_cellpath_score.py
│   ├── test_gene_masking.py
│   ├── test_metrics.py
│   └── test_end_to_end.py
└── outputs/
```

---

## 7. 数据接口设计

所有输入表必须经过 schema 校验，缺少必要字段时中止运行；缺少可选字段时记录 warning 并采用明确的降级策略。

### 7.1 疾病 HPO 表

文件：`disease_hpo.tsv`

```text
disease_id	disease_name	hpo_id	hpo_name	weight	is_core	frequency	onset	direction	source
```

字段说明：

- `disease_id`：OMIM、ORPHA 或其他统一疾病标识；
- `weight`：可选，默认 1.0；
- `is_core`：是否为核心表型；
- `frequency`：可保存原始频率或标准频率类别；
- `direction`：可选，表示增加、减少、缺失或未知；
- negated HPO 不进入正向匹配，应单独存储。

---

### 7.2 疾病–已知基因表

文件：`disease_genes.tsv`

```text
disease_id	gene_id	gene_symbol	evidence_level	relation_type	source	is_evaluation_positive
```

用途：

- 生成金标准；
- 严格遮蔽；
- 时间留出评估；
- 不允许在推理阶段直接作为排序特征使用。

---

### 7.3 小鼠模型表

文件：`mouse_models.tsv`

```text
model_id	model_name	mouse_gene_id	mouse_gene_symbol	human_gene_id	human_gene_symbol	allele_id	allele_type	zygosity	conditional	cre_driver	sex	life_stage	genetic_background	source
```

模型必须以具体 genotype 为单位，而不是只用 gene–MPO 关联。

---

### 7.4 小鼠模型表型表

文件：`model_mpo.tsv`

```text
model_id	mpo_id	mpo_name	status	direction	sex	life_stage	procedure_id	parameter_id	source
```

`status` 支持：

- `abnormal`；
- `tested_normal`；
- `untested`。

第一版基础相似度只使用 `abnormal`，但必须保留其他状态以支持后续缺口分析。

---

### 7.5 HPO–MPO 相似度矩阵

文件：`hpo_mpo_similarity.tsv`

```text
hpo_id	mpo_id	score	score_type	source	version	evidence
```

约束：

- `score` 建议统一归一化至 0–1；
- 必须记录矩阵来源和版本；
- 当前来源为原始矩阵；
- 后续可接入细胞表型矩阵或混合矩阵。

---

### 7.6 HPO–CPO 映射表

文件：`hpo_cpo.tsv`

```text
hpo_id	cpo_id	cell_id	cell_name	function_id	function_name	direction	relation_type	confidence	evidence_type	source_gene_ids	source_record_ids
```

必须保留：

- 映射置信度；
- 映射是否由基因证据产生；
- 用于构建映射的源基因；
- 是否经过人工审核；
- 是否允许进入严格评估模式。

---

### 7.7 细胞 marker gene set

文件：`cell_markers.tsv`

```text
cell_id	cell_name	gene_id	gene_symbol	marker_weight	specificity	reproducibility	expression_fraction	source
```

第一版允许仅提供二元关系，但建议预留连续权重字段。

---

### 7.8 Cell–function gene set

文件：`cell_function_genes.tsv`

```text
cpo_id	cell_id	function_id	gene_id	gene_symbol	function_weight	evidence_type	source
```

需要区分：

- 直接功能注释；
- 表达与功能交集；
- 文献支持；
- 网络扩展；
- 推断证据。

---

### 7.9 单细胞标签映射表

文件：`single_cell_label_map.tsv`

```text
dataset_id	dataset_label	cell_id	mapping_type	confidence	note
```

`mapping_type` 可包括：

- exact；
- synonym；
- child_to_parent；
- parent_to_child；
- manual；
- marker_supported。

---

## 8. 模块一：相似度矩阵接口

### 8.1 统一接口

```python
from abc import ABC, abstractmethod

class SimilarityProvider(ABC):
    @abstractmethod
    def get_score(self, hpo_id: str, mpo_id: str) -> float:
        """Return a normalized score in [0, 1]."""

    @abstractmethod
    def get_evidence(self, hpo_id: str, mpo_id: str) -> dict:
        """Return source, version and interpretable evidence."""
```

### 8.2 第一阶段实现

- `MatrixSimilarityProvider`
- 从 TSV 或 Parquet 加载已有 HPO–MPO 矩阵；
- 对不存在的 pair 返回 0，并标记 `not_mapped`；
- 支持按需稀疏加载；
- 检查重复 pair 和非法 score。

### 8.3 后续实现

- `CellPhenotypeSimilarityProvider`
- `HybridSimilarityProvider`

混合矩阵可定义为：

\[
q_{hybrid}(h,p)=
\eta q_{cell}(h,p)+(1-\eta)q_{base}(h,p)
\]

当新矩阵中不存在某个 HPO–MPO pair 时，允许退回原始矩阵。

---

## 9. 模块二：疾病–小鼠模型表型相似度

设疾病 \(d\) 的 HPO 集合为：

\[
H_d=\{h_1,h_2,\ldots,h_n\}
\]

小鼠模型 \(m\) 的 MPO 集合为：

\[
P_m=\{p_1,p_2,\ldots,p_k\}
\]

HPO \(h\) 和 MPO \(p\) 的相似度为：

\[
q(h,p)\in[0,1]
\]

---

### 9.1 单向疾病覆盖分数

\[
S_{H\rightarrow P}(d,m)
=
\frac{\sum_{h\in H_d}w_h\max_{p\in P_m}q(h,p)}{\sum_{h\in H_d}w_h}
\]

它表示疾病中的表型有多少能够被小鼠模型解释。

---

### 9.2 单向模型一致性分数

\[
S_{P\rightarrow H}(d,m)
=
\frac{1}{|P_m|}\sum_{p\in P_m}\max_{h\in H_d}q(h,p)
\]

它表示小鼠模型出现的表型有多少与目标疾病一致。

---

### 9.3 双向最佳匹配平均分

\[
S_{BMA}(d,m)
=
\frac{
\sum_{h\in H_d}w_h\max_{p\in P_m}q(h,p)
+
\alpha\sum_{p\in P_m}\max_{h\in H_d}q(h,p)
}{
\sum_{h\in H_d}w_h+\alpha|P_m|
}
\]

其中 \(\alpha\) 控制是否惩罚小鼠模型中的额外表型。

默认：

```yaml
model_scoring:
  reverse_match_weight: 1.0
```

---

### 9.4 最大配对分数

\[
S_{max}(d,m)=\max_{h\in H_d,p\in P_m}q(h,p)
\]

用于捕捉一个非常强的特征性表型匹配，但不能单独作为总分。

---

### 9.5 基础模型总分

推荐第一版：

\[
S_{model}(d,m)
=
\gamma S_{BMA}(d,m)+(1-\gamma)S_{max}(d,m)
\]

默认：

```yaml
model_scoring:
  bma_weight: 0.8
  max_pair_weight: 0.2
```

权重必须通过开发集选择，并在独立测试集上固定。

---

### 9.6 每个 HPO 的模型解释强度

对于候选模型 \(m\)，定义：

\[
E_{d,m}(h)=w_h\max_{p\in P_m}q(h,p)
\]

并记录产生最大值的 MPO：

\[
p^*(h,m)=\arg\max_{p\in P_m}q(h,p)
\]

该结果将用于后续候选特异性的 CPO 投影。

---

## 10. 模块三：小鼠模型聚合为候选基因

同一个人类基因可能对应多个小鼠模型。

设基因 \(g\) 的模型集合为 \(M_g\)。

### 10.1 主聚合分数

\[
S_{base}(d,g)=\max_{m\in M_g}S_{model}(d,m)
\]

选择最大值是为了保留最符合疾病机制的具体模型。

### 10.2 辅助聚合特征

同时输出：

- 最佳模型；
- Top 3 模型平均分；
- 模型数量；
- 异常 MPO 数量；
- 被解释的 HPO 比例；
- 最佳 HPO–MPO pair；
- 有效匹配 pair 数量；
- 模型间得分稳定性。

### 10.3 候选空间

默认候选空间：

> 所有具有异常 MPO 表型且可映射到人类同源基因的小鼠模型对应基因。

不得只使用已经经过官方阈值筛选的候选列表，否则无法构建完整排名和公平评估。

---

## 11. 模块四：疾病 HPO 投影到 CPO

### 11.1 疾病级 CPO 画像

设 HPO \(h\) 与 CPO \(c\) 的映射置信度为：

\[
R(h,c)\in[0,1]
\]

疾病级 CPO 权重：

\[
C_d(c)=\sum_{h\in H_d}w_hR(h,c)
\]

该画像回答：

> 从疾病整体 HPO 看，疾病可能涉及哪些细胞和功能异常。

---

### 11.2 候选特异性的 CPO 画像

候选基因 \(g\) 先选择最佳小鼠模型：

\[
m_g^*=\arg\max_{m\in M_g}S_{model}(d,m)
\]

然后定义：

\[
C_{d,g}(c)
=
\sum_{h\in H_d}E_{d,m_g^*}(h)R(h,c)
\]

这一步使 CPO 重排与候选基因实际匹配的 HPO–MPO 表型相连接。

候选特异性画像比直接使用疾病整体 CPO 画像更合理，因为不同候选模型可能解释疾病的不同部分。

---

### 11.3 CPO 覆盖度

疾病覆盖度：

\[
Coverage_{d}
=
\frac{
\sum_{h\in H_d}w_h\mathbf{1}[h\text{ has valid CPO mapping}]
}{
\sum_{h\in H_d}w_h
}
\]

候选可比较覆盖度：

\[
Coverage_{d,g}
=
\frac{
\sum_{h\in H_d}w_h\mathbf{1}[h\text{ has CPO and MPO match}]
}{
\sum_{h\in H_d}w_h
}
\]

覆盖度只控制 CPO 证据对总分的影响，不作为候选基因的直接正负证据。

---

## 12. 模块五：marker gene score

marker 证据表示候选基因是否具有相应细胞背景。

### 12.1 二元版本

\[
M(g,t)=\mathbf{1}[g\in Marker(t)]
\]

### 12.2 加权版本

若有连续 marker 指标：

\[
M(g,t)=
\operatorname{Norm}(
marker\_weight
\times specificity
\times reproducibility
)
\]

### 12.3 高频基因校正

设总细胞类型数为 \(N_T\)，包含基因 \(g\) 的细胞 marker 集合数为 \(n_g^M\)：

\[
IDF_M(g)=\log\frac{N_T+1}{n_g^M+1}
\]

校正后：

\[
M'(g,t)=M(g,t)\times IDF_M(g)
\]

### 12.4 使用规则

- 在正确细胞 marker 中：正向加分；
- 不在 marker 中：中性；
- 不允许将 marker 缺失作为硬过滤或强负分；
- marker 仅作为细胞定位先验，其权重低于 function 证据。

---

## 13. 模块六：function gene score

function 证据表示候选基因是否参与 CPO 对应的细胞功能。

### 13.1 二元版本

\[
F(g,c)=\mathbf{1}[g\in FunctionGenes(c)]
\]

### 13.2 加权版本

\[
F(g,c)=function\_weight(g,c)
\]

### 13.3 高频 gene set 校正

设 CPO gene set 总数为 \(N_C\)，包含基因 \(g\) 的 gene set 数量为 \(n_g^F\)：

\[
IDF_F(g)=\log\frac{N_C+1}{n_g^F+1}
\]

\[
F'(g,c)=F(g,c)\times IDF_F(g)
\]

### 13.4 可选网络扩展

严格遮蔽模式下，已知致病基因可能被从 function gene set 中移除。此时可计算其与剩余功能基因的间接关系：

- 蛋白互作网络距离；
- 功能相似性；
- 共表达网络距离；
- 调控网络邻近度。

定义：

\[
F_{indirect}(g,c)
=
\frac{1}{K}\sum_{u\in TopK(FunctionGenes(c))}sim(g,u)
\]

第一版可以暂不实现网络扩展，但代码接口必须预留。

---

## 14. 模块七：候选基因–CPO 相容性

设 CPO \(c\) 对应的细胞类型为 \(t_c\)。

候选基因与 CPO 的相容性定义为：

\[
A(g,c)=
\beta_FF'(g,c)
+
\beta_MM'(g,t_c)
+
\beta_J\sqrt{F'(g,c)M'(g,t_c)}
\]

推荐初始权重：

```yaml
gene_set_scoring:
  function_weight: 0.65
  marker_weight: 0.25
  interaction_weight: 0.10
```

必须通过消融实验验证：

- marker only；
- function only；
- marker + function；
- marker + function + interaction。

---

## 15. 模块八：CellPathScore

最终的细胞机制路径分数为：

\[
S_{cellpath}(d,g)
=
\frac{
\sum_c C_{d,g}(c)A(g,c)
}{
\sum_c C_{d,g}(c)+\epsilon
}
\]

等价展开：

\[
S_{cellpath}(d,g)
=
\frac{
\sum_{h\in H_d}\sum_c
E_{d,m_g^*}(h)R(h,c)A(g,c)
}{
\sum_{h\in H_d}\sum_c
E_{d,m_g^*}(h)R(h,c)+\epsilon
}
\]

需要输出 Top evidence paths：

```text
hpo_id
matched_mpo_id
hpo_mpo_score
cpo_id
cell_id
function_id
hpo_cpo_confidence
marker_score
function_score
path_contribution
```

---

## 16. 模块九：基础排名与 CPO 排名融合

由于基础表型分数和 CellPathScore 的尺度可能不同，建议先在每个疾病内部进行 percentile 或 rank normalization。

### 16.1 基础分数归一化

\[
B_{d,g}=PercentileRank(S_{base}(d,g))
\]

### 16.2 CPO 分数归一化

\[
P_{d,g}=PercentileRank(S_{cellpath}(d,g))
\]

### 16.3 最终分数

\[
S_{final}(d,g)
=
B_{d,g}
+
\lambda\cdot Coverage_d\cdot P_{d,g}
\]

可选使用候选可比较覆盖度：

\[
S_{final}(d,g)
=
B_{d,g}
+
\lambda\cdot Coverage_{d,g}\cdot P_{d,g}
\]

默认参数搜索：

```yaml
rank_fusion:
  lambda_grid: [0.05, 0.10, 0.20, 0.30, 0.50]
  normalization: percentile
  missing_cpo_policy: neutral
```

不允许直接删除没有 CPO 证据的候选基因。

---

## 17. 实验设计

### 17.1 基础消融实验

| 实验 | HPO–MPO 矩阵 | CPO 重排 | 目的 |
|---|---|---|---|
| A0 | 原始矩阵 | 无 | 跨物种表型基线 |
| A1 | 原始矩阵 | marker only | 验证细胞 marker 的独立贡献 |
| A2 | 原始矩阵 | function only | 验证功能 gene set 的独立贡献 |
| A3 | 原始矩阵 | marker + function | 验证完整重排模块 |
| A4 | 新细胞矩阵 | 无 | 验证新矩阵本身的贡献 |
| A5 | 新细胞矩阵 | marker + function | 最终完整系统 |
| A6 | 混合矩阵 | marker + function | 验证两类矩阵互补性 |

现阶段优先完成 A0–A3。

关键比较：

- `A3 - A0`：HPO–CPO 和 gene set 的独立贡献；
- `A4 - A0`：新矩阵的独立贡献；
- `A5 - A4`：在新矩阵基础上的重排贡献；
- `A6 - A5`：原始语义矩阵是否仍提供互补信息。

---

### 17.2 Benchmark 疾病纳入标准

基础 benchmark 疾病建议满足：

1. 至少 3 个非否定 HPO；
2. 至少 1 个已知致病基因；
3. 已知基因存在可靠人鼠同源映射；
4. 对应小鼠基因至少有一个带 MPO 异常的具体模型；
5. 至少一个疾病 HPO 能够与候选 MPO 比较。

CPO-covered benchmark 额外要求：

- 至少 1 个 HPO 有有效 CPO 映射；
- 推荐 `Coverage_d >= 0.2`；
- 同时报告全体疾病和高覆盖子集，禁止只选择对方法有利的疾病。

---

### 17.3 主要评估指标

以疾病为单位，对全部候选基因排序。

主要指标：

- Recall@1；
- Recall@5；
- Recall@10；
- Recall@50；
- MRR；
- NDCG@10；
- median rank；
- median rank percentile；
- mean rank change；
- median rank change；
- 排名提升、持平、下降的疾病比例。

定义：

\[
\Delta Rank(d)=Rank_{base}(d)-Rank_{final}(d)
\]

正值表示排名提升。

对于多基因疾病，同时输出：

- 最佳已知基因排名；
- 所有已知基因平均排名；
- Recall of any known gene；
- Recall of all known genes。

不建议以 ROC-AUC 作为主要指标，因为未标注基因不是真正负样本。

---

### 17.4 统计检验

对于疾病级排名变化，建议使用：

- paired Wilcoxon signed-rank test；
- bootstrap 置信区间；
- permutation test；
- 多重比较时使用 FDR 校正。

除 P 值外必须报告：

- median effect；
- mean effect；
- 95% CI；
- 改善疾病比例。

---

### 17.5 分层评估

按照以下因素分层：

- HPO 数量；
- HPO–MPO 覆盖率；
- HPO–CPO 覆盖率；
- 已知基因数量；
- 小鼠模型数量；
- MPO 数量；
- 细胞大类；
- 器官系统；
- marker gene set 大小；
- function gene set 大小；
- 是否有方向信息；
- 是否为单基因疾病；
- 是否为细胞机制明确的疾病。

需要检验方法是否主要在高 CPO 覆盖、细胞机制明确的疾病中获益。

---

## 18. 防止数据泄漏与循环验证

### 18.1 风险

若满足以下情况：

1. HPO–CPO 映射由疾病已知致病基因产生；
2. function gene set 中直接包含该已知致病基因；
3. 评估时再次利用该关系提升该基因；

则排名提高可能只是已知答案泄漏，而非方法具有泛化能力。

---

### 18.2 三种评估模式

#### 模式 1：Practical 模式

允许使用当前全部知识。

目的：评价系统在真实知识库环境中的实际排序能力。

该结果可以说明应用价值，但不能独立证明新基因预测能力。

#### 模式 2：Strict gene-masking 模式

对于每个测试疾病–已知基因 pair \((d,g^*)\)：

1. 从用于该疾病 HPO–CPO 映射的源基因中删除 \(g^*\)；
2. 若 HPO–CPO confidence 依赖基因重叠，重新计算映射分数；
3. 从相关 function gene set 的直接成员证据中删除 \(g^*\)；
4. marker 直接成员关系也单独遮蔽并进行消融；
5. 输出遮蔽前后排名和分数变化。

结果需区分：

```text
direct_gene_evidence_score
indirect_functional_support_score
```

#### 模式 3：Temporal holdout 模式

使用较早版本数据构建系统，并用后续新增的疾病–基因关联验证。

该模式作为后续增强任务，第一版不强制完成。

---

### 18.3 证据审计表

输出：`evidence_leakage_audit.tsv`

```text
disease_id	test_gene	hpo_id	cpo_id	gene_used_in_hpo_cpo	gene_in_marker_set	gene_in_function_set	masked	remaining_evidence_count	warning
```

---

## 19. 随机对照与 Null Models

真实重排效果必须与结构保持的随机对照比较。

### 19.1 HPO–CPO 随机化

保持：

- 每个 HPO 的 CPO 数量；
- CPO 被连接次数；
- 可选的器官系统约束。

随机交换 HPO–CPO 边。

### 19.2 Gene set 随机化

保持：

- 每个 gene set 大小；
- 可选的基因出现频率；
- 可选的表达水平分箱。

随机替换 gene set 成员。

### 19.3 细胞标签随机化

保持 CPO function 不变，随机交换 cell ID，检验细胞上下文是否真正有贡献。

### 19.4 错误细胞对照

将正确 function gene set 与：

- 同一器官的错误细胞；
- 不同器官的错误细胞；
- 父级或兄弟细胞；

进行对照。

### 19.5 Null 输出

```text
null_model_summary.tsv
observed_vs_null_metrics.tsv
permutation_pvalues.tsv
```

默认至少 500 次随机化；正式结果建议 1000 次。

---

## 20. 单细胞案例验证

### 20.1 单细胞案例的作用

单细胞案例用于验证：

> 排名模型预测的细胞类型和细胞功能模块，是否在真实患者组织中出现疾病相关异常。

单细胞数据不直接证明候选基因具有因果致病性。

---

### 20.2 案例筛选评分

定义：

\[
CaseSuitability
=
CPOCoverage
\times TissueMatch
\times scRNAAvailability
\times MouseModelCoverage
\times SampleQuality
\]

优先选择：

1. 单基因罕见病；
2. 患者与对照来自相同组织；
3. 至少有多个患者和多个对照供体；
4. 已知基因有小鼠模型；
5. 疾病 HPO 有较高 CPO 覆盖；
6. 系统预测指向明确细胞和功能；
7. 单细胞数据未用于当前 marker/function gene set 构建。

---

### 20.3 单细胞分析流程

#### 步骤 1：数据质控

- 细胞和基因过滤；
- 双细胞检测；
- ambient RNA 检查；
- donor、condition、batch 字段检查；
- 不以疾病状态为依据过度校正。

#### 步骤 2：细胞注释映射

将数据集标签映射到 CPO/CL：

```text
dataset label
→ exact/synonym mapping
→ ontology parent or child mapping
→ marker-supported manual review
→ final standardized cell ID
```

#### 步骤 3：细胞组成分析

输出每个 donor 的细胞比例，比较疾病与对照。

组成变化与功能变化必须分别分析。

#### 步骤 4：donor-level pseudobulk

分析单位：

```text
donor × cell type
```

不能把每个细胞当作独立生物学重复。

对每个细胞类型构建 gene × donor pseudobulk 矩阵。

#### 步骤 5：细胞类型特异差异表达

对每个可比较细胞类型进行疾病与对照差异分析，模型中尽可能纳入：

- condition；
- batch；
- sex；
- age；
- library size；
- 其他数据可用协变量。

#### 步骤 6：function gene set 验证

针对预测 CPO function gene set，计算：

- pseudobulk GSEA；
- gene-set mean z-score；
- AUCell 或 module score；
- leading-edge genes；
- effect direction；
- gene coverage。

#### 步骤 7：细胞特异性比较

核心检验：

\[
Effect(predicted\ cell,predicted\ function)
>
Effect(other\ cells,predicted\ function)
\]

并与以下对照比较：

- 错误细胞 + 正确 function；
- 正确细胞 + 随机 function；
- 错误细胞 + 随机 function。

#### 步骤 8：候选基因检查

检查候选基因是否：

- 在预测细胞中表达；
- 与异常功能模块相关；
- 与 leading-edge genes 存在功能或网络关系；
- 位于疾病相关细胞状态中。

不得要求候选基因本身必须显著差异表达。

---

### 20.4 单细胞验证指标

- predicted cell rank；
- predicted function enrichment score；
- predicted cell vs other cells effect difference；
- function gene coverage；
- leading-edge gene overlap；
- candidate gene expression coverage；
- marker identity consistency；
- random gene set percentile；
- wrong-cell contrast；
- donor-level effect stability。

---

## 21. 输出目录规范

```text
outputs/<run_id>/
├── run_manifest.json
├── resolved_config.yaml
├── logs/
├── qc/
│   ├── input_summary.tsv
│   ├── duplicate_summary.tsv
│   ├── id_mapping_summary.tsv
│   └── coverage_summary.tsv
├── baseline/
│   ├── disease_model_scores.tsv
│   ├── hpo_mpo_evidence.tsv
│   ├── model_gene_aggregation.tsv
│   └── base_gene_rankings.tsv
├── cpo/
│   ├── disease_cpo_profiles.tsv
│   ├── candidate_cpo_profiles.tsv
│   └── cpo_coverage.tsv
├── gene_sets/
│   ├── marker_specificity.tsv
│   ├── function_specificity.tsv
│   └── gene_frequency_weights.tsv
├── reranking/
│   ├── candidate_cellpath_scores.tsv
│   ├── evidence_paths.tsv
│   ├── final_gene_rankings.tsv
│   └── rank_changes.tsv
├── benchmark/
│   ├── cohort_manifest.tsv
│   ├── benchmark_metrics.tsv
│   ├── disease_level_metrics.tsv
│   ├── ablation_metrics.tsv
│   ├── stratified_metrics.tsv
│   └── statistical_tests.tsv
├── leakage_control/
│   ├── evidence_leakage_audit.tsv
│   ├── masked_rankings.tsv
│   └── masking_summary.tsv
├── null_models/
│   ├── null_model_summary.tsv
│   ├── permutation_metrics.tsv
│   └── observed_vs_null.tsv
├── single_cell/
│   ├── cell_label_mapping.tsv
│   ├── cell_composition.tsv
│   ├── pseudobulk_de.tsv
│   ├── function_enrichment.tsv
│   ├── cell_specificity.tsv
│   └── case_summary.tsv
├── figures/
└── report/
    ├── report.md
    └── report.html
```

---

## 22. 关键结果表设计

### 22.1 最终候选基因排名

文件：`final_gene_rankings.tsv`

```text
disease_id
disease_name
gene_id
gene_symbol
best_model_id
base_score
base_rank
bma_score
max_pair_score
hpo_mpo_coverage
cpo_coverage
candidate_cpo_coverage
marker_score
function_score
interaction_score
cellpath_score
cellpath_percentile
final_score
final_rank
rank_change
predicted_cell_id
predicted_cell_name
predicted_function_id
predicted_function_name
top_hpo_id
top_mpo_id
top_cpo_id
direct_gene_evidence_score
indirect_functional_support_score
is_known_gene
evaluation_mode
```

### 22.2 Evidence path 表

文件：`evidence_paths.tsv`

```text
disease_id
gene_symbol
model_id
hpo_id
hpo_weight
matched_mpo_id
hpo_mpo_score
cpo_id
hpo_cpo_confidence
cell_id
function_id
marker_score
function_score
interaction_score
path_contribution
source_record_ids
```

---

## 23. 图表设计

第一版至少生成以下图表：

1. benchmark 疾病和候选基因规模图；
2. HPO–MPO 与 HPO–CPO 覆盖度分布；
3. 基线与重排后已知基因 rank scatter plot；
4. 每个疾病的 rank change waterfall plot；
5. Recall@K 和 MRR 消融比较；
6. marker only、function only、完整模型比较；
7. CPO coverage 与 rank improvement 的关系；
8. observed vs randomized gene set 性能分布；
9. 单个案例的 Disease–HPO–MPO–CPO–Gene 网络；
10. 单细胞 predicted cell × function heatmap；
11. 正确细胞与错误细胞的 function effect 比较；
12. 预测候选基因的证据路径瀑布图。

绘图要求：

- 显示样本量和覆盖度；
- 不用 P 值替代效应量；
- 低覆盖结果必须显式标记；
- 所有图对应的原始数据均输出 TSV；
- 同一疾病在不同实验中的颜色和顺序保持一致。

---

## 24. 命令行设计

### 24.1 数据校验

```bash
cellphenorank validate-inputs \
  --config configs/baseline.yaml
```

### 24.2 运行基础排名

```bash
cellphenorank run-baseline \
  --diseases data/processed/disease_hpo.tsv \
  --models data/processed/mouse_models.tsv \
  --model-phenotypes data/processed/model_mpo.tsv \
  --similarity data/processed/hpo_mpo_similarity.tsv \
  --config configs/baseline.yaml \
  --output-dir outputs/baseline_run
```

### 24.3 运行 CPO 重排

```bash
cellphenorank run-rerank \
  --base-rankings outputs/baseline_run/baseline/base_gene_rankings.tsv \
  --hpo-cpo data/processed/hpo_cpo.tsv \
  --cell-markers data/processed/cell_markers.tsv \
  --cell-functions data/processed/cell_function_genes.tsv \
  --config configs/rerank.yaml \
  --output-dir outputs/rerank_run
```

### 24.4 运行 benchmark

```bash
cellphenorank benchmark \
  --rankings outputs/rerank_run/reranking/final_gene_rankings.tsv \
  --gold-standard data/processed/disease_genes.tsv \
  --config configs/benchmark.yaml \
  --output-dir outputs/benchmark_run
```

### 24.5 运行严格遮蔽

```bash
cellphenorank benchmark-masked \
  --config configs/strict_masking.yaml \
  --output-dir outputs/masked_run
```

### 24.6 运行单细胞案例

```bash
cellphenorank validate-single-cell \
  --adata data/processed/single_cell/case.h5ad \
  --case-ranking outputs/rerank_run/reranking/final_gene_rankings.tsv \
  --label-map data/processed/single_cell_label_map.tsv \
  --config configs/single_cell_case.yaml \
  --output-dir outputs/single_cell_case
```

---

## 25. 配置文件示例

```yaml
project:
  name: CellPhenoRank
  random_seed: 2026
  evaluation_mode: practical

similarity:
  provider: matrix
  score_min: 0.0
  score_max: 1.0
  missing_pair_score: 0.0

model_scoring:
  hpo_weight_mode: uniform
  reverse_match_weight: 1.0
  bma_weight: 0.8
  max_pair_weight: 0.2
  min_hpo_count: 3
  min_model_mpo_count: 1

model_to_gene:
  aggregation: max
  top_k_mean: 3
  require_one_to_one_ortholog: true

hpo_cpo:
  min_confidence: 0.5
  allowed_relation_types:
    - causal
    - equivalent
    - constituent
  include_unreviewed: false
  candidate_profile_mode: matched_hpo_weighted

marker_scoring:
  mode: weighted
  missing_policy: neutral
  use_idf: true

function_scoring:
  mode: weighted
  missing_policy: neutral
  use_idf: true
  network_extension: false

cellpath:
  function_weight: 0.65
  marker_weight: 0.25
  interaction_weight: 0.10
  epsilon: 1.0e-8

rank_fusion:
  normalization: percentile
  lambda_grid: [0.05, 0.10, 0.20, 0.30, 0.50]
  coverage_mode: disease
  missing_cpo_policy: neutral

benchmark:
  metrics:
    - recall_at_1
    - recall_at_5
    - recall_at_10
    - recall_at_50
    - mrr
    - ndcg_at_10
    - median_rank
    - median_rank_percentile
  bootstrap_iterations: 1000
  fdr_method: fdr_bh

null_models:
  iterations: 1000
  shuffle_hpo_cpo: true
  shuffle_gene_sets: true
  shuffle_cell_labels: true

report:
  top_genes_per_disease: 50
  top_paths_per_gene: 10
  formats:
    - markdown
    - html
```

---

## 26. 测试要求

### 26.1 单元测试

必须覆盖：

- 相似度矩阵读取和缺失 pair；
- BMA 计算；
- 最大匹配计算；
- 模型到基因聚合；
- HPO–CPO 投影；
- marker/function IDF；
- CellPathScore；
- percentile normalization；
- rank fusion；
- known gene masking；
- Recall@K、MRR 和 NDCG；
- 随机化后结构约束是否保持。

### 26.2 合成数据测试

构建一个最小合成示例，使：

- Gene A 的小鼠模型具有最高整体表型相似度；
- Gene B 的整体相似度略低，但拥有更一致的细胞和功能证据；
- 开启重排后 Gene B 排名上升；
- 打乱 CPO 或 gene set 后该提升消失。

该例仅用于软件正确性测试，不作为生物学结论。

### 26.3 端到端测试

从原始示例输入运行到：

- `base_gene_rankings.tsv`；
- `final_gene_rankings.tsv`；
- `benchmark_metrics.tsv`；
- `report.html`。

测试必须检查：

- 文件存在；
- 行数合理；
- 分数无 NaN/Inf；
- 排名连续；
- 同一 seed 结果完全一致。

---

## 27. 实施阶段与里程碑

### 阶段 0：数据审计与接口冻结

任务：

- 整理现有 HPO–MPO 矩阵格式；
- 整理 HPO–CPO、marker 和 function gene set；
- 明确基因 ID、细胞 ID 和本体版本；
- 建立 schema 和 QC；
- 冻结所有核心输入接口。

交付物：

- `schema_description.md`；
- 示例数据；
- 输入校验程序；
- 数据覆盖度报告。

验收标准：

- 所有输入均可通过 schema 校验；
- 重复、缺失和无法映射 ID 均有明确报告；
- 相似度矩阵可由统一 provider 加载。

---

### 阶段 1：完成基础跨物种表型排名

任务：

- 实现疾病–小鼠模型 BMA 和最大匹配分数；
- 输出每个 HPO 的最佳 MPO；
- 聚合为基因排名；
- 实现基础 benchmark。

交付物：

- `disease_model_scores.tsv`；
- `hpo_mpo_evidence.tsv`；
- `base_gene_rankings.tsv`；
- 基线指标报告。

验收标准：

- 合成测试结果符合预期；
- 所有 benchmark 疾病均能生成完整候选排名；
- 每个基因分数可追溯到最佳模型和 HPO–MPO pair。

---

### 阶段 2：完成 HPO–CPO 与 gene set 重排

任务：

- 构建疾病级和候选特异性 CPO 画像；
- 实现 marker score；
- 实现 function score；
- 实现 CellPathScore；
- 实现 rank fusion 和 evidence paths。

交付物：

- `candidate_cpo_profiles.tsv`；
- `candidate_cellpath_scores.tsv`；
- `evidence_paths.tsv`；
- `final_gene_rankings.tsv`。

验收标准：

- 无 CPO 证据时最终排名退化为基线；
- marker 缺失不会导致候选被删除；
- 每个重排变化可被具体路径解释；
- 权重全部来自配置文件。

---

### 阶段 3：完成消融、严格遮蔽和随机对照

任务：

- 完成 A0–A3；
- 完成 known gene masking；
- 完成 HPO–CPO、gene set 和 cell label 随机化；
- 完成分层评估。

交付物：

- `ablation_metrics.tsv`；
- `masked_rankings.tsv`；
- `observed_vs_null.tsv`；
- `stratified_metrics.tsv`。

验收标准：

- 能分别报告 marker 和 function 的贡献；
- 实际映射的性能优于大多数随机映射；
- 严格遮蔽结果与 practical 结果明确分开；
- 报告中不存在将测试基因直接作为输入特征的隐藏路径。

---

### 阶段 4：完成单细胞案例

任务：

- 筛选至少一个合适罕见病案例；
- 完成细胞标签映射；
- donor-level pseudobulk；
- function gene set 分析；
- predicted cell vs wrong cell 对照；
- 生成案例机制图。

交付物：

- 单细胞分析结果表；
- 案例报告；
- Disease–HPO–MPO–CPO–Gene 网络图；
- 细胞功能异常 heatmap。

验收标准：

- 分析单位为 donor 而非 cell；
- 组成变化和功能变化分开报告；
- 预测 function 在预测细胞中的异常强于随机或错误细胞对照，或明确报告未验证成功；
- 不将候选基因未差异表达解释为模型失败。

---

### 阶段 5：接入新的细胞表型 HPO–MPO 矩阵

任务：

- 实现 `CellPhenotypeSimilarityProvider`；
- 完成 A4–A6；
- 比较原始矩阵、新矩阵和混合矩阵；
- 分析新矩阵在哪些疾病类型中最有优势。

交付物：

- 新矩阵 benchmark；
- 三类矩阵比较报告；
- 新矩阵覆盖度和错误分析。

验收标准：

- 下游代码无需修改，仅更换 provider 和配置；
- 能独立量化新矩阵贡献；
- 新矩阵覆盖不足时有明确回退策略；
- 不将新矩阵和 CPO 重排的贡献混为一个指标。

---

## 28. 第一版最小可行产品

第一版 MVP 必须实现：

1. 读取疾病 HPO、小鼠模型 MPO 和原始 HPO–MPO 矩阵；
2. 生成疾病–模型分数；
3. 聚合为疾病–基因完整排名；
4. 读取 HPO–CPO、marker 和 function gene set；
5. 生成候选特异性 CPO 画像；
6. 计算 marker、function 和 CellPathScore；
7. 输出重排后的候选基因；
8. 完成 A0–A3 消融；
9. 完成至少一种 gene masking；
10. 完成至少一种 gene set 随机对照；
11. 自动生成 Markdown 报告。

第一版不强制实现：

- 新 HPO–MPO 矩阵；
- 深度学习模型；
- learning-to-rank；
- 大规模网络传播；
- 自动下载所有官方数据；
- 完整时间留出 benchmark；
- 多疾病单细胞批量验证。

---

## 29. 最终验收标准

项目完成时应满足：

### 工程层面

- 一条命令可从标准化输入运行到最终报告；
- 相似度矩阵可插拔；
- 所有参数配置化；
- 所有随机过程可复现；
- 单元测试和端到端测试通过；
- 输出包含输入哈希、版本和 warning。

### 方法层面

- 完成基础表型预测；
- 完成 marker/function 机制化重排；
- 完成消融；
- 完成数据泄漏审计；
- 完成随机对照；
- 完成覆盖度分层。

### 生物学层面

- 至少在一批 CPO 高覆盖疾病中评估重排效果；
- 能解释哪些 HPO、MPO、细胞和功能推动了候选基因排名；
- 至少完成一个独立单细胞案例；
- 不夸大单细胞数据对致病因果关系的证明能力。

---

## 30. 预期结果与结论边界

### 30.1 当前使用原始矩阵时可以支持的结论

> 在保持传统 HPO–MPO 跨物种表型相似度不变的条件下，HPO–CPO 映射以及细胞 marker/function gene sets 能够为罕见病候选致病基因提供额外的细胞背景和功能机制证据，并在部分疾病中改善已知致病基因排序。

当前阶段不能声称：

> 新构建的细胞表型 HPO–MPO 矩阵优于原始矩阵。

---

### 30.2 新矩阵完成后可以进一步支持的结论

若 A4/A5 显著优于 A0/A3，可以支持：

> 将 HPO 和 MPO 表示为“细胞类型 + 细胞表型异常”，能够在特定疾病类别中改善跨物种表型对齐和致病基因优先级预测。

---

### 30.3 单细胞案例可以支持的结论

> 系统预测的关键细胞类型和功能模块在独立患者单细胞数据中表现出疾病相关异常，从而为 HPO–CPO 映射和 gene set 的生物学合理性提供支持。

单细胞案例不能单独证明：

- 新候选基因已经被确认致病；
- 预测细胞是疾病发生的唯一细胞来源；
- mRNA 无差异代表基因无功能作用。

---

## 31. 推荐论文或汇报叙事

本项目可按照以下逻辑组织：

1. 传统跨物种表型方法能够根据疾病 HPO 和小鼠 MPO 检索相似模型，但缺乏细胞机制解释；
2. 本项目将疾病表型进一步投影到 CPO 所表示的细胞类型和细胞功能异常；
3. 利用细胞 marker 和 cell-function gene set，构建候选基因特异性的机制路径分数；
4. 在固定原始 HPO–MPO 矩阵的情况下，通过消融实验证明 CPO 和 gene set 的独立贡献；
5. 在严格基因遮蔽和随机对照下检验结果是否由已知基因泄漏导致；
6. 利用独立患者单细胞数据验证预测的细胞和功能模块；
7. 新矩阵完成后，将其作为可替换的相似度模块接入，并与原始矩阵及混合矩阵比较。

项目中最关键的逻辑边界为：

\[
\boxed{
候选基因生成
\neq
细胞机制重排
\neq
单细胞机制验证
}
\]

只有将三者分开评估，才能清楚说明 HPO–CPO、gene set 和新矩阵分别提供了什么价值。

---

## 32. Codex 实现优先级

建议 Codex 按以下顺序开发：

```text
P0  Schema、数据读取、ID 统一和 QC
P0  SimilarityProvider 接口
P0  疾病–模型 BMA 和最大匹配评分
P0  模型到基因聚合
P0  基础排名与 benchmark metrics
P1  HPO–CPO 投影
P1  marker/function IDF 与相容性评分
P1  CellPathScore 和可解释路径
P1  rank fusion 与 A0–A3 消融
P1  自动 Markdown/HTML 报告
P2  known gene masking
P2  HPO–CPO 和 gene set 随机对照
P2  分层评估和统计检验
P2  单细胞标签映射、pseudobulk 和模块验证
P3  新矩阵 provider
P3  混合矩阵和 A4–A6
P3  网络扩展和时间留出验证
```

开发过程中，每完成一个 P0/P1 模块，必须同时补充：

- schema；
- 单元测试；
- 合成示例；
- CLI；
- 输出字段说明；
- 最小运行文档。
