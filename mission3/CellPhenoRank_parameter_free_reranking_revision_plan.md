# CellPhenoRank 无参数证据链重排修改计划

> 文档用途：交给 Codex 直接修改当前 `CellPhenoRank` 工程。  
> 基础版本：`CellPhenoRank_project_description_v2.md`。  
> 修改重点：保留现有项目内 HPO–MPO 表型初排，删除依赖人工权重的 `CellPathScore + percentile fusion`，改为无训练集、无拟合参数的“CPO 证据链富集 + 表型稳定性约束重排”。  
> 文档状态：P0 主流程与 P1 Bridge-LOO 已实施，当前正式重排规范。  
> 当前配置：`configs/real_frozen_parameter_free.yaml`。  
> 当前结果：`outputs/parameter_free/`。

---

## 0. 2026-09-03 实施状态与首轮结果

已经实现：

- HPO duplicate/ancestor reduction；
- Strict 与 Extended 二元 HPO-CPO 图；
- Disease-HPO-CPO-Function-Gene reachability；
- 独立 HPO 支持计数、原始路径计数和 marker concordance；
- 全量冻结 HPOA 背景超几何检验与疾病内 BH-FDR；
- full + leave-one-HPO-out 表型排名；
- 稳定表型顺序 DAG；
- 确定性约束拓扑排序；
- A0-A5、Strict/Extended 与 Bridge-LOO；
- migration manifest、输入哈希、图版本和零约束违反检查；
- `run-parameter-free` 完整 CLI。

当前真实运行规模：

```text
diseases                     50
single-gene genotypes        1,068
excluded multi-gene models   354
atomic candidate genes       49
original disease HPO rows    835
nonredundant HPO rows        799
frozen HPOA background       11,546
Strict enrichment universe   1,585
Extended enrichment universe 6,741
Strict binary edges          14,126
Extended binary edges        156,267
stable phenotype constraints 24,268
constraint violations        0
```

候选归因审计：冻结模型中有 354 个多基因 genotype。旧 loader 会把 `BLNK;BTK`
这类组合字符串当成一个候选“基因”，不符合基因优先级任务。当前主分析强制
`single_gene_only`：使用 1,068 个可归因 genotype 和 49 个原子人类基因；所有
50 个疾病均保留至少一个单基因 gold model。多基因 genotype 只允许进入单独的
Extended 模型敏感性分析，不能把联合表型任意归因给某一个组成基因。

首轮冻结 benchmark：

| Arm | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| A0 surface | 0.36 | 0.68 | 0.92 | 0.5279 | 0.6182 |
| A1 raw path | 0.30 | 0.62 | 0.84 | 0.4497 | 0.5333 |
| A2 enrichment | 0.26 | 0.62 | 0.84 | 0.4364 | 0.5239 |
| A3 Strict constrained | 0.38 | 0.72 | 0.92 | 0.5461 | 0.6320 |
| A4 Extended constrained | 0.44 | 0.82 | 0.98 | 0.6185 | 0.7049 |
| A5 Bridge-LOO | 0.34 | 0.68 | 0.92 | 0.5123 | 0.6058 |

必须如实解释：

- A3 相对 A0 的 mean rank change 为 0.06，4 win / 38 tie / 8 loss，Wilcoxon
  `p=0.810`，当前没有显著总体提升；
- A5 Bridge-LOO 相对 A0 的 mean rank change 为 -0.16，2 win / 39 tie / 9 loss，
  `p=0.0995`，没有稳定增益；
- A4 Extended 的 mean rank change 为 1.78，24 win / 18 tie / 8 loss，`p=0.000322`，
  但不能替代 Strict 主结果；
- A4 Extended 提升明显，但依赖 `cell_general` 边，只能作为敏感性分析，不能替代
  Strict 主结果；
- 原始路径计数和无约束 enrichment 均使总体性能下降，支持引入 hub 校正和表型稳定约束。

两次独立完整运行的 HPO reduction、LOO、稳定约束、Strict 图、富集、A3、A5 和
benchmark 输出 SHA-256 完全一致。

### 0.1 官方 IMPC PhenoDigm 对照

为避免把当前项目内 A0 误称为 IMPC 原算法，已经新增独立的官方 PhenoDigm 对照：

```text
command  run-phenodigm-official
config   configs/phenodigm_official.yaml
output   outputs/phenodigm_official/
```

该对照直接读取 IMPC `phenodigm` Solr 中的 `disease_model_avg_norm` 和
`disease_model_max_norm`，按照原始 PhenoDigm 定义计算：

\[
S_{PhenoDigm}(d,m)
=
\frac{AvgNorm(d,m)+MaxNorm(d,m)}{2}
\]

然后按照原论文在基因层取最高模型分：

\[
S_{PhenoDigm}(d,g)=\max_{m\in M_g}S_{PhenoDigm}(d,m)
\]

候选集仍固定为相同的 50 个疾病、49 个基因和 1,068 个单基因 genotype。结果为：

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| 当前 A0 | 0.36 | 0.68 | 0.92 | 0.5279 | 0.6182 |
| 官方 PhenoDigm | 0.30 | 0.66 | 0.90 | 0.4821 | 0.5756 |

官方 PhenoDigm 相对当前 A0 为 20 win / 9 tie / 21 loss，mean rank change 为
0.38，双侧配对 Wilcoxon `p=0.597`，没有显著总体差异。50/50 个疾病的 gold gene
均有正分；182 个 gold genotype 中 179 个有官方记录、177 个为正分。全部 50 个
原始响应均已保存并哈希。

解释边界：该结果固定了本地候选集和 gold standard，但官方分数使用 IMPC 当前疾病
HPO、跨物种本体和 IC 快照，而不是本地冻结核心 HPO。因此它是“官方算法在本地评价
宇宙中的外部对照”，不是只替换聚合公式的 same-input 消融，也不能直接生成冻结 HPO
的 leave-one-out 稳定性约束。

### 0.2 原始 PhenoDigm HPO-MP 矩阵

已经从官方 Google Code 历史源码归档取得 `hp_mp_mappings.bz2`：

```text
rows     52,022,155
bytes    701,426,412
SHA-256  b69f76e06aaf5ba7850dad7987ded7969424289f3c7b4d47acb26b030db22ec5
fields   HP ID, HP name, MP ID, MP name, simJ, IC
```

冻结 benchmark 子矩阵保留 4,500,868 行。591 个唯一疾病 HPO 覆盖 517 个
（87.48%），2,699 个候选 MPO 在历史矩阵中存在 2,309 个（85.55%）。使用
`sqrt(simJ*IC)`、论文双向最佳匹配、理论最优模型归一化和每基因最高 genotype
后，结果为：

| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| 当前 A0 | 0.36 | 0.68 | 0.92 | 0.5279 | 0.6182 |
| 历史矩阵 PhenoDigm | 0.42 | 0.72 | 0.82 | 0.5538 | 0.6105 |

相对当前 A0 为 17 win / 13 tie / 20 loss，Wilcoxon `p=0.435`。Top 1/5 和 MRR
略好，但 Top 10 下降且总体无显著差异。完整来源、公式、越界分数 QC 和复现命令见
`PHENODIGM_LEGACY_MATRIX.md`。

### 0.3 Exomiser 2602 当前矩阵与评分器

已在远程取得并校验 Exomiser 2602 phenotype H2 数据库。完整
`HP_MP_MAPPINGS` 为 95,051,352 行；冻结 591 个 HPO 的本地子矩阵为 3,143,719 行。
当前版本覆盖 589/591 个疾病 HPO 和 2,685/2,699 个候选 MPO。

根据当前 Exomiser 源码实现 `exomiser_current` 半对称匹配、理论最佳均值归一化和
100 分截断后：

| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| 当前项目 A0 | 0.36 | 0.68 | 0.92 | 0.5279 | 0.6182 |
| Exomiser 2602 PhenoDigm | 0.42 | 0.78 | 0.92 | 0.5548 | 0.6382 |

相对当前 A0 为 22 win / 11 tie / 17 loss，mean rank change 为 0.26，Wilcoxon
`p=0.872`。所有主要汇总指标数值不低于 A0，但疾病级差异不显著。完整来源、公式、
数据库哈希和远程路径见 `EXOMISER_2602_PHENODIGM_MATRIX.md`。

### 0.4 Exomiser 2602 加 CPO 重排

已经使用 2602 当前 PhenoDigm full/LOO 排名重新构建 43,264 条稳定表型约束，并运行
相同的 A0-A5 CPO 流程：

| Arm | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| A0 Exomiser 2602 | 0.42 | 0.76 | 0.92 | 0.5509 | 0.6349 |
| A3 Strict | 0.48 | 0.74 | 0.92 | 0.5943 | 0.6675 |
| A4 Extended | 0.56 | 0.88 | 0.92 | 0.6908 | 0.7440 |
| A5 Bridge-LOO | 0.46 | 0.72 | 0.92 | 0.5755 | 0.6527 |

A3 相对 A0 为 6 win / 37 tie / 7 loss，`p=0.541`。A3 的两个最大改善 PCDH15
和 WAS 在 Bridge-LOO 后消失；A5 为 4 win / 38 tie / 8 loss，`p=0.564`。
A4 为 26 win / 22 tie / 2 loss，`p=9.88e-5`，但依赖 `cell_general` 边且缺少
Extended Bridge-LOO，只能作为敏感性分析。完整解释见
`EXOMISER_2602_CPO_RERANKING.md`。

尚未实现：

- 新 HPO-MPO provider 与 A6；
- indirect-only 网络支持；
- 单独的分阶段 CLI（当前完整 CLI 已可生成全部中间产物）；
- degree-preserving 图随机化和 RRA 敏感性分析。

---

## 1. 项目目标

给定一个罕见病名称或疾病 ID，系统完成以下任务：

1. 获取疾病对应的 HPO 表型集合；
2. 使用现有 HPO–MPO 相似性矩阵寻找与疾病表型最相似的小鼠 genotype；
3. 将小鼠 genotype 映射到相应的人类同源基因，生成候选致病基因初始排名；
4. 利用 `Disease–HPO–CPO–Gene` 二元证据链，分析每个候选基因能够解释多少个疾病表型；
5. 使用统计富集结果，在不推翻稳定表型证据的前提下对候选基因进行无参数重排；
6. 输出可审计的疾病—表型—细胞—功能—基因证据链；
7. 使用独立单细胞数据验证排名靠前基因在预测细胞中的表达或功能异常。

本次修改不训练分类器或排序模型，不划分训练集、验证集和测试集，也不拟合 marker、function、CPO 与表型分数之间的权重。

---

## 2. 当前实现存在的问题

当前实现中，候选基因与 CPO 的相容性为：

\[
A(g,c)
=
\beta_FF'(g,f)
+
\beta_MM'(g,t)
+
\beta_J\sqrt{F'(g,f)M'(g,t)}
\]

并预设：

```yaml
function_weight: 0.65
marker_weight: 0.25
interaction_weight: 0.10
```

随后计算 `CellPathScore`，再通过：

\[
S_{final}
=
B
+
\lambda\times Coverage\times Percentile(CellPathScore)
\]

进行最终融合，当前 `lambda=0.2`。

该方案主要存在以下问题：

1. marker、function 和 interaction 的预设比例缺少独立依据；
2. `lambda` 决定 CPO 可以在多大程度上改变基础排名，同样缺少客观来源；
3. HPO–CPO 边本身未必具有可靠连续权重，强行转换为分数会引入额外假设；
4. percentile 只表示疾病内部相对位置，绝对证据很弱的候选也可能获得最大 CPO 加分；
5. marker 是细胞身份证据，不等同于基因参与相应致病功能；
6. 原始路径数量容易偏向连接度高的 HPO、CPO、GO 和基因；
7. 当前重排结果难以解释为一个成熟、可复现的统计分析流程。

因此，本次修改不再寻找另一组人工权重，而是删除连续加权融合。

---

## 3. 核心设计原则

### 3.1 表型证据为主

HPO–MPO/IMPC 表型相似性负责生成候选基因和确定主体排序。

```text
Disease
→ HPO profile
→ HPO–MPO similarity
→ mouse genotype
→ mutated mouse gene
→ human ortholog
→ base gene ranking
```

### 3.2 CPO 是辅助机制证据

CPO 不重新替代表型初排，而是在表型证据不能稳定区分的候选之间提供额外信息。

```text
Disease
→ HPO
→ CPO(cell + function)
→ function gene / cell-context gene
```

### 3.3 二元关系代替人工连续权重

正式重排中，以下关系只表示存在或不存在：

```text
Disease–HPO
HPO–CPO
CPO–Function
Function–Gene
CPO–Cell
Cell–Marker Gene
```

原始置信度、scope、source gene 和文献来源继续保留用于：

- 构建 Strict/Extended 两套图；
- 审计；
- 解释；
- 敏感性分析。

但不再机械转换成 `1.0 / 0.8 / 0.5` 后进入主公式。

### 3.4 不直接使用原始路径总数排名

原始链路数保留用于解释，但主排序使用：

1. 独立疾病 HPO 支持数；
2. 基于超几何分布的 CPO 富集显著性；
3. Leave-one-HPO-out 表型稳定性约束。

### 3.5 不将 marker 作为主要致病证据

主链路为：

```text
Disease → HPO → CPO → Function → Gene
```

marker 只用于：

- 判断候选基因是否具有相应细胞背景；
- 对完全相同的 function 证据进行次级排序；
- 选择单细胞案例中的待验证细胞；
- 输出解释信息。

marker 缺失不扣分，不删除候选，也不能单独将低表型相似度基因提升到前列。

---

## 4. 修改后的总体流程

```text
Disease name / ID
        |
        v
Curated disease HPO profile
        |
        v
HPO redundancy reduction
        |
        +----------------------------------+
        |                                  |
        v                                  v
HPO–MPO similarity                  Binary HPO–CPO graph
        |                                  |
        v                                  v
Mouse genotype similarity          HPO–CPO–Function–Gene paths
        |                                  |
        v                                  v
Gene-level base ranking             CPO enrichment statistics
        |                                  |
        +----------------+-----------------+
                         |
                         v
              Leave-one-HPO-out analysis
                         |
                         v
          Stable phenotype-order constraints
                         |
                         v
       CPO-prioritized constrained topological sort
                         |
                         v
                 Final candidate ranking
                         |
                         v
       Evidence-chain report + single-cell validation
```

---

## 5. 本次修改的范围

## 5.1 保留的模块

以下现有模块继续使用：

- 疾病 HPO 输入和规范化；
- HPO–MPO `SimilarityProvider`；
- 疾病—小鼠 genotype 表型相似度计算；
- genotype 聚合到人类同源基因；
- 冻结 benchmark；
- gold gene/model 评价；
- HPO–MPO 匹配证据输出；
- 单细胞 donor-level pseudobulk 流程；
- E0–E3 单细胞验证框架；
- 运行 manifest、配置解析、日志和报告框架。

表型初排在本次任务中视为冻结模块。除修复明确 bug 外，不在重排修改阶段重新调整其公式或权重。

## 5.2 停用的正式排序模块

以下逻辑不再进入正式排名：

```text
HPO–CPO continuous score multiplication
scope factor multiplication
confidence factor multiplication
marker/function continuous weighted sum
marker–function interaction term
candidate CellPathScore
positive-only CellPath percentile
lambda-based final score fusion
```

对应配置项应删除、弃用或仅保留为 legacy：

```yaml
function_weight
marker_weight
interaction_weight
lambda
coverage_mode
```

## 5.3 新增模块

```text
hpo_redundancy.py
binary_cpo_graph.py
gene_hpo_reachability.py
evidence_chain_counter.py
cpo_enrichment.py
leave_one_hpo_out.py
stable_order.py
constrained_reranker.py
bridge_loo.py
reranking_report.py
```

---

## 6. 输入数据规范

## 6.1 疾病 HPO

建议字段：

```text
disease_id
disease_name
hpo_id
hpo_name
is_negated
frequency
onset
source
```

正式主分析：

- 排除 negated HPO；
- 默认所有 HPO 等权；
- frequency 和 onset 只保留在结果中，不进入本次无参数主排序。

## 6.2 HPO–MPO 相似性矩阵

继续使用当前 provider 接口：

```python
get_score(hpo_id: str, mpo_id: str) -> float
get_evidence(hpo_id: str, mpo_id: str) -> dict
```

最低字段：

```text
hpo_id
mpo_id
score
source
version
evidence
```

## 6.3 HPO–CPO

最低字段：

```text
hpo_id
cpo_id
cl_id
go_id
relation_type
scope
confidence
anatomy_status
review_status
source_genes
source_records
```

正式算法只读取是否建立二元边。其余字段用于 Strict/Extended 图构建和证据溯源。

## 6.4 Function gene set

支持两种形式。

### GO 全局 function gene set

```text
go_id
gene_symbol
evidence_code
source
```

此时：

```text
CPO(CL, GO) → Gene
```

由 CPO 的 GO 与 GO–gene membership 建立。

### CPO-specific function gene set

```text
cpo_id
cl_id
go_id
gene_symbol
source
```

如果当前 function gene set 已经同时要求基因在相应细胞中表达，则直接使用 CPO-specific membership，避免再次把 marker 作为重复细胞证据。

程序必须在 manifest 中记录：

```text
function_set_scope = go_global | cpo_cell_specific
```

## 6.5 Marker gene set

```text
cl_id
gene_symbol
marker_source
specificity
reproducibility
```

正式重排只使用二元 membership：

```text
gene is marker of cell: true / false
```

连续 specificity 和 reproducibility 保留在解释报告，不进入主排序。

---

## 7. HPO 冗余处理

疾病 HPO 可能同时包含祖先和后代，例如：

```text
Seizure
Epileptic seizure
Myoclonic seizure
```

直接把这些术语分别计数会重复计算相近临床证据。

## 7.1 主分析规则

对每个疾病：

1. 删除完全重复的 HPO；
2. 如果输入集合同时包含某 HPO 及其后代，主分析保留更具体的后代；
3. 如果多个 HPO 只是兄弟节点，不进行带阈值的语义聚类；
4. 原始 HPO 全部保留在审计输出中；
5. 生成 `hpo_reduction.tsv`，记录保留和删除原因。

定义处理后的集合：

\[
H_d^{NR}
\]

其中 `NR` 表示 non-redundant。

## 7.2 输出字段

```text
disease_id
original_hpo_id
retained_hpo_id
status
reason
ancestor_distance
```

其中 `status`：

```text
retained
duplicate_removed
ancestor_removed
invalid_hpo
negated_removed
```

---

## 8. 表型基础排名

本次修改保留当前项目内 HPO–MPO 初排流程。其双向 BMA 结构与 PhenoDigm 的
`avgScore` 相近，但 HPO–MPO 单对分数、理想模型归一化和最终聚合并不相同，不能称为
官方 IMPC PhenoDigm 的原样复现。官方方法的独立对照见第 0.1 节。

对于疾病 \(d\) 和 genotype \(m\)：

\[
S_{surface}(d,m)
\]

由当前实现计算。

一个基因可能对应多个 genotype，继续使用：

\[
S_{base}(d,g)
=
\max_{m\in M_g}S_{surface}(d,m)
\]

得到：

```text
base_gene_rankings.tsv
```

至少包含：

```text
disease_id
gene_symbol
base_score
base_rank
best_model_id
best_model_score
hpo_mpo_coverage
best_hpo_id
best_mpo_id
```

后续 CPO 证据不与 `base_score` 直接相加。

---

## 9. 构建二元 CPO 图

## 9.1 节点

```text
Disease
HPO
CPO
CL
GO
Gene
```

## 9.2 边

```text
Disease --has_phenotype--> HPO
HPO --linked_to--> CPO
CPO --has_cell--> CL
CPO --has_function--> GO
GO --has_function_gene--> Gene
CL --has_marker_gene--> Gene
```

如果 function set 已经是 CPO-specific：

```text
CPO --has_function_gene--> Gene
```

## 9.3 Strict 图

正式主结果使用 Strict graph。建议纳入：

- HPO、CL 和 GO ID 均有效；
- anatomy status 为 allowed、compatible 或项目认可的同义状态；
- relation type 属于项目认定的直接、因果、构成或等价关系；
- 通过当前 review/QC 规则；
- 去除重复 HPO–CPO 边；
- 不对每个 HPO 进行任意 Top-N 截断。

具体允许的分类值由当前数据字段决定，写入配置，但不得转换为连续权重。

## 9.4 Extended 图

Extended graph 在 Strict graph 基础上增加：

- general cell；
- 自动构建但尚未完全审核的边；
- 较宽泛关系；
- 项目允许的 provisional links。

Extended 仅用于敏感性分析，不替代 Strict 主结果。

## 9.5 图输出

```text
strict_hpo_cpo_edges.tsv
extended_hpo_cpo_edges.tsv
cpo_function_gene_edges.tsv
cpo_cell_marker_edges.tsv
graph_qc_summary.tsv
```

---

## 10. Disease–HPO–CPO–Gene 证据链

## 10.1 Function 主证据链

对候选基因 \(g\)，定义其能够通过 CPO function 链路连接到的 HPO 集合：

\[
H_g^F
=
\left\{
h:\exists c,
(h\rightarrow c)
\land
(g\in FunctionGenes(c))
\right\}
\]

对疾病 \(d\)：

\[
k_{d,g}
=
|H_d^{NR}\cap H_g^F|
\]

`k_dg` 表示：

> 该候选基因能够通过至少一个 CPO function 链解释多少个不同的疾病 HPO。

同一个 HPO 即使通过多个 CPO 到达同一基因，在 `supported_hpo_count` 中只计算一次。

## 10.2 原始路径数量

同时保留：

\[
N_{raw}(d,g)
=
\#\{d\rightarrow h\rightarrow c\rightarrow g\}
\]

原始路径数只用于解释和次级排序，不作为主统计量。

## 10.3 输出指标

对每个 disease–gene：

```text
supported_hpo_count
supported_hpo_fraction
supported_cpo_count
supported_cell_count
supported_function_count
raw_function_path_count
raw_marker_path_count
concordant_cpo_count
```

其中：

```text
concordant_cpo_count
```

表示该基因在同一 CPO 中同时具备：

- function membership；
- 对应 cell 的 marker/context membership。

它只作为较低优先级的细胞背景证据。

---

## 11. CPO 富集检验

仅比较 `supported_hpo_count` 会偏向可以连接大量 HPO 的高连接度基因。因此，对每个疾病—候选基因进行单侧超几何检验。

## 11.1 HPO 背景宇宙

正式背景建议定义为：

> 冻结 HPOA/疾病注释版本中，所有非 negated、ID 有效、并且在当前 Strict HPO–CPO 图中至少存在一个可用映射的 HPO。

记为：

\[
U_H
\]

以及：

\[
N=|U_H|
\]

如果当前工程暂时没有全量 HPOA 背景，可在 pilot 中使用冻结 benchmark 的全部可映射 HPO 并明确标记：

```text
background_scope = benchmark_only
```

正式分析优先使用：

```text
background_scope = frozen_hpoa
```

## 11.2 超几何参数

对候选基因 \(g\)：

\[
K_g=|H_g^F\cap U_H|
\]

对疾病 \(d\)：

\[
n_d=|H_d^{NR}\cap U_H|
\]

观察重叠：

\[
k_{d,g}=|H_d^{NR}\cap H_g^F|
\]

假设：

\[
X\sim Hypergeometric(N,K_g,n_d)
\]

单侧富集 P 值：

\[
p_{CPO}(d,g)
=P(X\ge k_{d,g})
\]

实现时使用数值稳定形式：

```python
scipy.stats.hypergeom.logsf(k - 1, N, K, n)
```

同时输出：

```text
cpo_enrichment_log_p
cpo_enrichment_pvalue
cpo_enrichment_fdr
```

BH-FDR 在每个疾病的候选基因集合内计算。FDR 只用于结果标记，不作为是否进入排序的硬阈值。

## 11.3 边界条件

```text
n_d == 0        → p = 1，CPO 不改变基础顺序
K_g == 0        → p = 1
k_dg == 0       → p = 1
无 function set → 所有候选 p = 1
```

## 11.4 该检验的解释

该检验回答：

> 考虑到基因本身可以通过 CPO 连接多少个 HPO，它在当前疾病 HPO 集合中获得这么多独立支持，是否高于随机预期？

它能够降低以下因素造成的偏差：

- 某个基因出现在大量 function gene sets 中；
- 某个基因通过大量宽泛 CPO 连接到许多 HPO；
- 大 gene set 或高连接度功能造成的原始路径膨胀。

它不证明因果关系，只表示当前知识图中的疾病表型富集。

---

## 12. Marker 的正式定位

Marker 不进入超几何主检验。

对每个 disease–gene 仅输出：

```text
marker_supported_cell_count
marker_supported_cpo_count
concordant_cpo_count
marker_cells
```

Marker 的用途优先级：

1. `p_CPO` 完全相同且独立 HPO 支持数相同后的 tie-breaker；
2. 为 Top gene 指定可能受影响的细胞；
3. 单细胞验证中的 cell label 映射；
4. 证据链解释。

不得使用如下逻辑：

```text
not marker → gene penalty
```

因为致病基因可以在相应细胞中发挥功能，但不是该细胞的经典 marker。

---

## 13. Leave-one-HPO-out 表型稳定性

直接按照 CPO 富集重排全部候选，可能导致表型相似度较低的基因大幅跃升。因此，CPO 只允许改变表型证据不稳定的相对顺序。

## 13.1 生成表型排名集合

对疾病：

\[
H_d^{NR}=\{h_1,\ldots,h_n\}
\]

运行：

```text
R0：使用全部 HPO
R1：删除 h1
R2：删除 h2
...
Rn：删除 hn
```

只使用纯 HPO–MPO 表型流程，不加载 CPO。

无效 LOO 运行应跳过，例如：

- 删除后没有剩余 HPO；
- 删除后没有任何可映射 HPO；
- 输入 profile 无法计算表型分数。

每次运行中：

- 所有候选基因必须保留；
- 没有有效模型分数的候选赋予并列最低 rank；
- 完全相同 base score 使用同一个 dense rank；
- gene symbol 只用于最终输出稳定性，不用于定义表型优势。

## 13.2 稳定表型顺序

对任意两个基因 \(g_a,g_b\)：

如果在全部有效表型排名中均满足：

\[
Rank_r(g_a)<Rank_r(g_b)
\]

则定义：

\[
g_a\succ_{stable}g_b
\]

表示：

> 无论删除疾病中的哪一个 HPO，表型证据都稳定支持 \(g_a\) 排在 \(g_b\) 前面。

CPO 重排不得推翻这一顺序。

如果两者在部分运行中交换、并列或缺乏一致关系，则视为表型不可稳定区分，允许 CPO 决定其相对次序。

## 13.3 特殊情况

### 只有一个疾病 HPO

无法执行有效 LOO。此时：

- 保留完整表型排名的严格顺序；
- CPO 只打破完全相同的 base rank；
- 报告 `stability_mode=single_hpo_fallback`。

### 只有一个有效表型运行

同样使用完整排名约束，CPO 只处理并列候选。

---

## 14. 稳定顺序图与无参数重排

## 14.1 构建 DAG

候选基因作为节点。

若：

\[
g_a\succ_{stable}g_b
\]

添加有向边：

```text
gene_a → gene_b
```

由于该关系来自所有排名中的一致严格顺序，理论上不应形成环。如果检测到环，程序必须停止并输出诊断，不得静默删除边。

## 14.2 CPO 优先级键

拓扑排序过程中，如果同时有多个入度为 0 的候选基因，使用以下确定性顺序选择：

```text
1. cpo_enrichment_pvalue       ascending
2. supported_hpo_count         descending
3. supported_cpo_count         descending
4. concordant_cpo_count        descending
5. raw_function_path_count     descending
6. base_rank                   ascending
7. gene_symbol                 ascending
```

这里没有加权求和，也没有把 P 值转换成需要人工融合的总分。

## 14.3 拓扑排序伪代码

```python
from heapq import heappop, heappush


def priority_key(gene):
    return (
        gene.cpo_enrichment_pvalue,
        -gene.supported_hpo_count,
        -gene.supported_cpo_count,
        -gene.concordant_cpo_count,
        -gene.raw_function_path_count,
        gene.base_rank,
        gene.gene_symbol,
    )


def constrained_rerank(genes, stable_edges):
    graph = build_dag(genes, stable_edges)
    indegree = calculate_indegree(graph)

    queue = []
    for gene in genes:
        if indegree[gene] == 0:
            heappush(queue, (priority_key(gene), gene))

    final_order = []

    while queue:
        _, gene = heappop(queue)
        final_order.append(gene)

        for child in graph[gene]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(queue, (priority_key(child), child))

    if len(final_order) != len(genes):
        raise RuntimeError("Stable-order graph contains a cycle")

    return final_order
```

## 14.4 方法含义

最终排名满足：

- 稳定表型优势不可被 CPO 推翻；
- 表型不稳定或并列候选可由 CPO function 富集决定；
- CPO 证据很强但基础表型明显较弱的候选，不能跨越所有稳定表型约束直接升至首位；
- 没有 CPO 证据的候选仍保留，并按照基础表型顺序退化运行。

---

## 15. 最终分数的处理

主算法只需要最终 rank，不需要连续 `final_score`。

如果现有接口必须输出分数，定义为最终排名的显示性归一化：

\[
S_{final}(d,g)
=
1-
\frac{Rank_{final}(d,g)-1}
{N_d-1}
\]

该值：

- 仅用于前端、绘图和导出；
- 不参与任何候选排序计算；
- 不解释为概率；
- 不与 base score 比较绝对大小。

字段名建议使用：

```text
final_rank_score
```

避免误写为 disease probability。

---

## 16. Practical、Strict 与泄漏控制

## 16.1 Practical 模式

使用当前完整 HPO–CPO 图和 gene sets。

回答：

> 在现有知识库条件下，CPO 证据链是否能够改善已知疾病基因的实际重排？

Practical 提升不能直接解释为未知致病基因泛化。

## 16.2 Bridge-LOO 模式

这是建议的主要泄漏控制。

对每个测试 disease–gold gene 对 \((d,g^*)\)：

1. 找到 HPO–CPO 边构建时记录的 source-gene contributions；
2. 删除 \(g^*\) 对这些 HPO–CPO 边的贡献；
3. 如果删除后仍有其他基因、文献或人工证据支持，则保留该二元边；
4. 如果该边完全依赖 \(g^*\)，删除该边；
5. 重新构建该疾病的 Strict HPO–CPO 图；
6. 重新计算 evidence chains、hypergeometric enrichment 和最终重排；
7. 独立来源的 GO function membership 和 cell marker membership可以保留。

Bridge-LOO 回答：

> 在未知该致病基因与 HPO 的直接关联时，剩余 HPO–CPO 桥接是否仍能支持该基因？

## 16.3 Indirect-only 压力测试

作为 P1 可选任务：

- 在 Bridge-LOO 基础上，再删除目标基因自身的 function 和 marker membership；
- 仅允许未来 PPI、共表达、调控网络等间接证据支持。

当前尚无间接网络时，这一模式主要用于压力测试，不应作为主产品模式。

---

## 17. Strict 与 Extended 敏感性分析

分别运行：

```text
Strict graph reranking
Extended graph reranking
```

对每个基因输出：

```text
strict_rank
extended_rank
strict_pvalue
extended_pvalue
rank_difference
graph_stability_status
```

状态：

```text
stable_high
strict_only
extended_only
unstable
no_cpo_support
```

解释规则：

- Strict 和 Extended 均靠前：证据相对稳定；
- 仅 Extended 明显提升：标记为 provisional；
- 两套图方向相反：不得作为强结论；
- 无 Strict 证据但有表型证据：保留基础排名。

---

## 18. 消融与比较实验

建议正式运行以下实验：

| Arm | 表型初排 | CPO 证据 | 重排规则 | 用途 |
|---|---|---|---|---|
| A0 | 当前 IMPC/HPO–MPO | 无 | 无 | 基础表型基线 |
| A1 | 当前初排 | raw path count | 直接路径排序 | 证明原始计数偏差 |
| A2 | 当前初排 | hypergeometric function enrichment | 不加稳定约束 | 检查纯 CPO 重排幅度 |
| A3 | 当前初排 | Strict CPO enrichment | LOO 稳定约束 | 正式主方法 |
| A4 | 当前初排 | Extended CPO enrichment | LOO 稳定约束 | 敏感性分析 |
| A5 | 当前初排 | Strict Bridge-LOO | LOO 稳定约束 | 泄漏控制主结果 |
| A6 | 新 HPO–MPO 矩阵 | Strict CPO enrichment | 同一规则 | 新矩阵完成后运行 |

A1、A2 是诊断性实验，不作为正式推荐模型。

最重要比较：

```text
A3 vs A0：无参数 CPO 重排是否改善初排
A5 vs A0：去除 bridge 循环证据后是否仍有增益
A3 vs A4：结果是否依赖较低可信 HPO–CPO 边
```

---

## 19. 评价指标

保留当前 frozen benchmark，gold gene 只在排序完成后用于评价。

主要指标：

```text
Recall@1
Recall@5
Recall@10
Recall@50
MRR
NDCG@10
median rank
median rank percentile
win / tie / loss
mean and median rank change
```

定义：

\[
\Delta Rank_d
=
Rank_{A0,d}-Rank_{arm,d}
\]

正值表示已知基因排名改善。

统计分析：

- paired Wilcoxon signed-rank test；
- disease-level bootstrap 95% CI；
- Strict/Extended rank agreement；
- CPO coverage 分层；
- HPO 数量分层；
- MPO 数量和 genotype 数量分层；
- gene global HPO degree 分层。

不需要训练/验证划分，因为算法没有学习参数。冻结 benchmark 只用于最终评价，不能根据结果事后调整排序键顺序。

---

## 20. 单细胞案例验证

现有 donor-level pseudobulk 和 E0–E3 框架保留。

## 20.1 案例冻结

在读取 expression matrix 前冻结：

```text
disease
top candidate gene
gold/known gene status
predicted cell(s)
predicted function(s)
primary endpoint
reference cells
statistical unit
```

## 20.2 预测细胞确定

对最终 Top gene，汇总其 Strict function evidence paths。

细胞排序使用确定性规则：

```text
1. 支持该 cell 的不同疾病 HPO 数 descending
2. 对应 CPO 数 descending
3. function path 数 descending
4. CL ID ascending
```

如果存在完全并列，全部保留为 co-primary predicted cells，不通过查看表达结果后再选择。

## 20.3 验证层次

### E0：候选基因是否在预测细胞中可检测

检查：

- donor pseudobulk 中是否有稳定非零表达；
- 是否存在等位基因、剪接或蛋白层面信息；
- 避免把 dropout 误判为基因缺失。

### E1：候选基因是否在预测细胞中表现出疾病相关异常

统计单位：

```text
donor × cell type
```

优先使用 pseudobulk 差异分析。

对于预期造成 NMD、缺失或转录下降的变异，可将候选基因表达差异作为主终点。

对于 missense、gain-of-function、dominant-negative 等不一定改变 mRNA 的变异，不能要求候选基因必须差异表达。

### E2：预测 cell–function module 是否异常

- 在预测细胞中计算 function gene-set module；
- 从 module 中删除候选基因自身；
- 删除直接用于构建相应 HPO–CPO 边的 source genes，形成 strict downstream module；
- 使用表达匹配随机 gene sets 作为 null。

### E3：Top-ranked genes 是否集中于异常细胞背景

在存在多个预测候选时，可比较：

- Top-ranked genes 在预测细胞中的表达可检测比例；
- Top-ranked genes 的 disease effect 排名；
- Top genes 与低排名候选的差异；
- 预测 cell/function 与真实单细胞异常的一致性。

单细胞结果是机制一致性证据，不是新的致病因果证明。

---

## 21. 输出文件

## 21.1 HPO 处理

```text
hpo_reduction.tsv
nonredundant_disease_hpo.tsv
```

## 21.2 表型稳定性

```text
leave_one_hpo_out_rankings.tsv
phenotype_rank_matrix.tsv
stable_phenotype_constraints.tsv
stable_order_graph_summary.tsv
```

## 21.3 CPO 图和证据链

```text
strict_hpo_cpo_edges.tsv
extended_hpo_cpo_edges.tsv
gene_hpo_reachability.tsv
raw_evidence_paths.tsv
gene_cpo_enrichment.tsv
```

## 21.4 最终排名

```text
final_constrained_rankings.tsv
rank_changes.tsv
strict_vs_extended_stability.tsv
bridge_loo_rankings.tsv
```

## 21.5 评价

```text
benchmark_metrics.tsv
disease_level_metrics.tsv
ablation_metrics.tsv
stratified_metrics.tsv
statistical_tests.tsv
```

---

## 22. 关键表结构

## 22.1 `gene_cpo_enrichment.tsv`

```text
disease_id
disease_name
gene_symbol
base_rank
base_score
cpo_graph_version
disease_hpo_count
cpo_mappable_hpo_count
cpo_coverage
supported_hpo_count
supported_hpo_fraction
global_reachable_hpo_count
supported_cpo_count
supported_cell_count
supported_function_count
raw_function_path_count
raw_marker_path_count
concordant_cpo_count
cpo_enrichment_log_p
cpo_enrichment_pvalue
cpo_enrichment_fdr
has_cpo_evidence
```

## 22.2 `stable_phenotype_constraints.tsv`

```text
disease_id
higher_gene
lower_gene
valid_ranking_count
full_rank_higher
full_rank_lower
stable_in_all_runs
```

## 22.3 `final_constrained_rankings.tsv`

```text
disease_id
disease_name
gene_symbol
base_rank
base_score
final_rank
final_rank_score
rank_change
cpo_enrichment_pvalue
cpo_enrichment_fdr
supported_hpo_count
supported_cpo_count
concordant_cpo_count
raw_function_path_count
best_model_id
best_hpo_mpo_pair
top_cpo_paths
predicted_cells
predicted_functions
graph_version
reranking_mode
```

## 22.4 `raw_evidence_paths.tsv`

每一行表示一条可审计链：

```text
disease_id
gene_symbol
hpo_id
hpo_name
cpo_id
cl_id
cell_name
go_id
function_name
path_type
function_membership
marker_membership
hpo_cpo_relation
hpo_cpo_scope
hpo_cpo_confidence
source_genes
source_records
graph_version
```

---

## 23. 推荐代码结构

```text
src/cellphenorank/
├── baseline/
│   ├── similarity_provider.py
│   ├── genotype_scoring.py
│   └── gene_aggregation.py
├── cpo_binary/
│   ├── hpo_redundancy.py
│   ├── graph_builder.py
│   ├── reachability.py
│   ├── evidence_paths.py
│   └── enrichment.py
├── stability/
│   ├── leave_one_hpo_out.py
│   ├── rank_matrix.py
│   └── stable_order.py
├── reranking/
│   ├── constrained_toposort.py
│   ├── bridge_loo.py
│   └── outputs.py
├── benchmark/
├── single_cell/
└── report/
```

现有 legacy 模块可暂时保留：

```text
legacy_cellpath_score.py
legacy_percentile_fusion.py
```

但不得被新的正式配置调用。

---

## 24. 配置文件建议

新增：

```yaml
reranking:
  method: cpo_enrichment_constrained
  primary_graph: strict
  run_extended_sensitivity: true
  use_continuous_hpo_cpo_weights: false
  use_marker_as_primary_evidence: false
  final_score_is_rank_display_only: true

hpo_reduction:
  remove_duplicates: true
  remove_ancestor_when_descendant_present: true
  cluster_semantic_siblings: false

cpo_graph:
  strict:
    require_valid_hpo: true
    require_valid_cl: true
    require_valid_go: true
    require_anatomy_compatible: true
    allowed_relation_types:
      - causal
      - constituent
      - equivalent
      - direct
    require_reviewed_or_accepted: true
  extended:
    include_general_cell: true
    include_provisional: true

cpo_enrichment:
  background_scope: frozen_hpoa
  test: hypergeometric_greater
  fdr_method: fdr_bh
  fdr_used_for_ranking: false

phenotype_stability:
  method: leave_one_hpo_out
  dominance_rule: higher_rank_in_every_valid_run
  ties_create_constraint: false
  invalid_run_policy: skip
  single_hpo_policy: preserve_full_base_order

priority_order:
  - cpo_enrichment_pvalue_ascending
  - supported_hpo_count_descending
  - supported_cpo_count_descending
  - concordant_cpo_count_descending
  - raw_function_path_count_descending
  - base_rank_ascending
  - gene_symbol_ascending

leakage_control:
  run_practical: true
  run_bridge_loo: true
  run_indirect_only: false
```

旧配置项若仍出现，应产生 warning：

```text
function_weight is ignored by cpo_enrichment_constrained
marker_weight is ignored by cpo_enrichment_constrained
interaction_weight is ignored by cpo_enrichment_constrained
lambda is ignored by cpo_enrichment_constrained
```

---

## 25. CLI 建议

```bash
# 1. 构建二元 CPO 图
PYTHONPATH=src python3 -B -m cellphenorank build-cpo-graph \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free/cpo_graph

# 2. 运行 HPO 冗余处理与基础排名
PYTHONPATH=src python3 -B -m cellphenorank run-baseline \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free/baseline

# 3. 运行 leave-one-HPO-out
PYTHONPATH=src python3 -B -m cellphenorank run-hpo-stability \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free/stability

# 4. 计算 CPO enrichment
PYTHONPATH=src python3 -B -m cellphenorank score-cpo-enrichment \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free/cpo_enrichment

# 5. 运行约束重排
PYTHONPATH=src python3 -B -m cellphenorank rerank-constrained \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free/reranking

# 6. 完整流程
PYTHONPATH=src python3 -B -m cellphenorank run-parameter-free \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free

# 7. Bridge-LOO
PYTHONPATH=src python3 -B -m cellphenorank run-bridge-loo \
  --config configs/real_frozen_parameter_free.yaml \
  --output-dir outputs/parameter_free/bridge_loo
```

---

## 26. 单元测试与回归测试

必须新增以下测试。

## 26.1 HPO 冗余

```text
祖先与后代同时存在时只保留后代
兄弟 HPO 不被自动合并
重复 HPO 被删除
negated HPO 被排除
```

## 26.2 二元图

```text
Strict 不读取连续权重
同一 HPO–CPO 重复边去重
Extended 是 Strict 的超集
无效 CL/GO 边被过滤
```

## 26.3 Reachability 与路径计数

```text
同一 HPO 经多个 CPO 到达同一 gene，supported_hpo_count 只加 1
raw_path_count 保留所有不同路径
marker-only gene 不进入 function supported_hpo_count
concordant CPO 正确计数
```

## 26.4 超几何检验

```text
与 scipy.stats.hypergeom.sf 的已知结果一致
K=0 时 p=1
k=0 时 p=1
高连接度 gene 在相同 k 下显著性更弱
FDR 在疾病内计算
```

## 26.5 LOO 稳定性

```text
在所有运行中 A 均高于 B，则生成 A→B
有一次交换或并列，则不生成稳定约束
单 HPO 疾病回退到 full-order
无效 LOO 被跳过并记录
```

## 26.6 拓扑重排

```text
任何稳定表型边不得被最终排名违反
无稳定边时按 CPO priority 排序
无 CPO 时退化为 base rank
完全相同证据时按 gene symbol 稳定排序
检测到环时明确报错
```

## 26.7 Bridge-LOO

```text
目标 gene 为唯一 source 时删除 HPO–CPO 边
仍有其他 source 时保留二元边
独立 GO membership 不被误删
Practical 与 Bridge-LOO 输出分离
```

## 26.8 端到端回归

合成数据至少覆盖：

1. CPO 将两个表型不稳定候选交换；
2. CPO 不能越过稳定表型约束；
3. 高连接度 hub gene 被超几何背景校正；
4. 无 CPO 的疾病完全退化到基础排名；
5. Bridge-LOO 后 practical 阳性链路消失。

---

## 27. 实施顺序

## P0：必须完成

### 阶段 1：冻结旧结果

- 保存当前 `real_frozen` 全部输入 hash；
- 保存 A0、A1、A2、A3 旧结果；
- 生成 migration manifest；
- 禁止覆盖旧输出目录。

### 阶段 2：二元 CPO 图

- 实现 Strict/Extended graph；
- 移除连续边权在主分析中的使用；
- 输出图 QC 和边审计表。

### 阶段 3：证据链与富集

- 实现 HPO 冗余处理；
- 实现 function reachability；
- 实现 raw path count；
- 实现 hypergeometric P 值和 disease-wise FDR。

### 阶段 4：表型稳定性

- 实现 full + leave-one-HPO-out 基础排名；
- 实现 rank matrix；
- 实现稳定顺序 DAG。

### 阶段 5：约束重排

- 实现 priority-queue topological sort；
- 验证稳定顺序零违反；
- 输出最终排名和证据链。

### 阶段 6：Benchmark

- 运行 A0–A5；
- 报告 overall 与 CPO-evaluable 子集；
- 保留所有失败和下降疾病；
- 不依据 benchmark 结果修改 priority order。

## P1：建议完成

- Bridge-LOO；
- Strict/Extended 稳定性；
- 全量 HPOA 背景宇宙；
- 计算优化，避免大候选池下显式存储全部传递边；
- 可选 RRA 敏感性分析；
- 新的 blind single-cell case。

## P2：新矩阵完成后

- 接入新的 HPO–MPO provider；
- 保持同一 CPO 富集与稳定约束算法；
- 运行 A6；
- 分开报告矩阵贡献和重排贡献。

---

## 28. 验收标准

## 28.1 软件验收

- 新正式配置中不存在有效的 marker/function/lambda 加权；
- 相同输入和 seed-independent 算法重复运行结果完全一致；
- 无 CPO 数据时最终排名与基础排名一致；
- 所有稳定表型约束在最终排名中均被满足；
- 所有排名变化都能回溯到具体 HPO–CPO–Function–Gene 路径；
- Practical、Bridge-LOO、Strict、Extended 输出明确分离；
- 所有输入 hash、图版本和背景宇宙写入 manifest；
- 原有单细胞验证命令可继续运行。

## 28.2 方法验收

至少回答以下问题：

1. A3 是否优于 A0；
2. A3 中多少疾病改善、持平和下降；
3. A5 Bridge-LOO 是否仍有改善；
4. 改善是否集中在 CPO coverage 较高的疾病；
5. hub gene 校正前后结果有何变化；
6. Strict 与 Extended 是否给出一致 Top candidates；
7. CPO 重排是否违反任何稳定表型顺序；
8. 单细胞案例是否支持预测细胞和预测功能。

不设置“必须达到某个 Recall/MRR 才算成功”的事后阈值。即使无提升，也应完整报告，以判断 HPO–CPO 和 gene set 的真实适用边界。

---

## 29. 结论边界

本方案能够支持：

> 在固定跨物种表型初排的条件下，使用二元 Disease–HPO–CPO–Function–Gene 证据链、独立 HPO 富集检验以及 leave-one-HPO-out 稳定性约束，对表型不能稳定区分的候选致病基因进行无参数重排。

本方案不能单独支持：

- CPO 富集 P 值是致病概率；
- 单细胞差异表达证明候选基因具有因果致病性；
- practical 排名提升等同于未知基因泛化；
- 当前有限候选池结果等同于全基因组诊断性能；
- 新 HPO–MPO 矩阵优于旧矩阵，除非完成独立冻结比较。

---

## 30. 最终方法摘要

```text
输入疾病名称或 ID
    ↓
获得并简化疾病 HPO 集合
    ↓
使用当前项目内 HPO–MPO 方法生成候选基因基础排名
    ↓
构建无权 Disease–HPO–CPO–Function–Gene 网络
    ↓
统计每个候选基因支持的独立疾病 HPO 和原始证据链
    ↓
使用超几何检验校正高连接度基因和大 gene set 偏差
    ↓
逐一删除疾病 HPO，计算基础排名稳定性
    ↓
将所有稳定表型顺序写成有向约束
    ↓
在不违反约束的候选间，按 CPO 富集证据进行拓扑重排
    ↓
输出最终排名及可审计证据链
    ↓
在独立单细胞数据中验证预测基因、细胞和功能异常
```

核心原则：

\[
\boxed{\text{表型初排决定稳定主体顺序}}
\]

\[
\boxed{\text{CPO 富集只解决表型不稳定或并列候选}}
\]

\[
\boxed{\text{独立 HPO 支持与统计富集代替人工加权总分}}
\]

---

## 31. 方法学参考

1. Smedley D, et al. **PhenoDigm: analyzing curated annotations to associate animal models with human diseases.** Database (Oxford). 2013;2013:bat025. DOI: `10.1093/database/bat025`.
2. Himmelstein DS, Baranzini SE. **Heterogeneous Network Edge Prediction: A Data Integration Approach to Prioritize Disease-Associated Genes.** PLoS Computational Biology. 2015;11(7):e1004259. DOI: `10.1371/journal.pcbi.1004259`.
3. Zhao M, et al. **Phen2Gene: rapid phenotype-driven gene prioritization for rare diseases.** NAR Genomics and Bioinformatics. 2020;2(2):lqaa032. DOI: `10.1093/nargab/lqaa032`.
4. Kolde R, et al. **Robust rank aggregation for gene list integration and meta-analysis.** Bioinformatics. 2012;28(4):573–580. DOI: `10.1093/bioinformatics/btr709`.
5. Squair JW, et al. **Confronting false discoveries in single-cell differential expression.** Nature Communications. 2021;12:5692. DOI: `10.1038/s41467-021-25960-2`.

其中 RRA 仅建议作为独立敏感性分析，不作为正式主重排方法。
