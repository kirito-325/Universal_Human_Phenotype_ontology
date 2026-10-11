# CellPhenoRank 预测结果驱动的单细胞 Case 探索与验证计划

> 文档用途：交由 Codex 基于 CellPhenoRank 当前预测结果，自动探索适合的疾病—候选基因—细胞—功能单细胞验证病例。  
> 核心原则：**不预先指定疾病、基因、细胞类型、功能或单细胞数据集；先由预测结果形成候选池，再进行数据可用性预检，并在读取表达值之前冻结验证终点。**  
> 案例性质：完整知识库条件下的已知生物学恢复与机制一致性验证，不直接等同于未知致病基因发现。

---

## 1. 项目背景

CellPhenoRank 当前流程包括：

```text
疾病 HPO
  ↓
HPO–MPO 表型相似性
  ↓
疾病—小鼠 genotype 相似度
  ↓
小鼠模型对应的人类同源基因
  ↓
A0 表型基础排名
  ↓
HPO–CPO–Function–Gene 二元证据链
  ↓
Leave-one-HPO-out 稳定表型约束
  ↓
A4 完整 CPO 知识库重排
```

下一步希望从完整预测结果中自动寻找一个或多个适合的病例，验证以下完整链条：

```text
某疾病的已知致病基因
  ↓
在表型基础算法中排名不够靠前
  ↓
在完整 CPO 知识库重排后明显提升
  ↓
知识库给出可审计的 HPO–CPO–Cell–Function–Gene 路径
  ↓
找到相应疾病和组织的患者单细胞数据
  ↓
候选基因在预测细胞中出现分子异常
  ↓
CPO 预测的细胞功能在该细胞中发生异常
```

本计划不把某个预先知道存在阳性单细胞结果的疾病作为固定案例，而是要求 Codex 根据预测结果和公开数据条件，系统完成候选发现、预检、冻结、分析与报告。

---

## 2. 核心研究问题

### 2.1 排名层问题

> 哪些疾病—已知致病基因组合在 A0 表型基础排名中被相对低估，但在 A4 完整 CPO 知识库重排后得到明显提升？

### 2.2 机制层问题

> 对于这些排名改善的基因，是否存在可追溯的 Disease–HPO–CPO–Cell–Function–Gene 证据链？

### 2.3 单细胞层问题

> 在独立患者单细胞数据中，候选基因是否在知识库预测的细胞中出现表达、剪接、等位基因或下游功能异常？

### 2.4 功能层问题

> 在删除候选基因本身以及用于构建 HPO–CPO 关系的桥接源基因后，CPO 预测的下游功能模块是否仍然在预测细胞中异常？

---

## 3. 结论边界

### 3.1 可以支持的结论

如果候选筛选和单细胞验证成功，可以说明：

1. 完整 CPO 知识库能够在表型初排基础上恢复部分被低估的已知致病基因；
2. 排名提升可以解释为多个疾病 HPO 经 CPO 指向相应细胞和细胞功能的机制证据；
3. 独立患者单细胞数据与预测的候选基因、细胞类型和功能状态具有一致性；
4. CPO 不只是改变排名，还能产生可实验或可转录组验证的细胞机制假设。

### 3.2 不能支持的结论

不得声称：

- CellPhenoRank 新发现了该已知致病基因；
- 单细胞差异表达证明了新的致病因果关系；
- A4 的实际知识库提升等同于对未知基因的泛化能力；
- 在封闭候选基因池上的结果等同于全基因组临床诊断性能；
- 从多个病例中事后挑出的阳性病例属于独立确认性验证；
- 候选基因没有差异表达就意味着预测错误；
- 单细胞中的细胞比例变化等同于细胞内功能变化。

---

## 4. 本版相对于固定病例方案的修改

删除所有硬编码的：

```text
固定疾病名称
固定 disease ID
固定候选基因
固定单细胞 accession
固定组织
固定预测细胞
固定预测功能
固定比较组
```

改为以下自动流程：

```text
读取完整预测结果
  ↓
生成全部排名改善 disease–gene 候选
  ↓
提取每个候选的 HPO–CPO–Cell–Function 证据
  ↓
形成候选 Pareto 集合和确定性优先顺序
  ↓
检索患者单细胞数据
  ↓
只基于 metadata、标签和 feature names 完成预检
  ↓
冻结病例、基因、细胞、功能和终点
  ↓
最后读取表达矩阵并运行验证
```

---

## 5. 总体工作流

```text
A0 base_gene_rankings
A4 full/extended constrained rankings
A5 Bridge-LOO rankings
CPO enrichment
raw evidence paths
        |
        v
Candidate discovery
        |
        v
Ranking-improved disease–gene pairs
        |
        v
Evidence-path audit
        |
        v
Parameter-free candidate Pareto front
        |
        v
Public single-cell dataset discovery
        |
        v
Metadata-only dataset preflight
        |
        v
Eligible case list
        |
        +----------------------------+
        |                            |
        v                            v
Exploratory scan              Frozen primary case
all eligible cases            selected without expression
        |                            |
        +-------------+--------------+
                      v
             Load expression data
                      |
                      v
        donor × cell-type pseudobulk
                      |
          +-----------+------------+
          |           |            |
          v           v            v
 candidate gene   predicted cell   predicted function
 abnormality      specificity      module abnormality
          |           |            |
          +-----------+------------+
                      v
      supported / partial / failed report
```

---

## 6. 输入数据

Codex 不应依赖固定文件名，而应优先从当前正式配置、`run_manifest.json` 和结果目录解析实际路径。

最低需要以下逻辑输入。

### 6.1 表型基础排名 A0

至少包含：

```text
disease_id
disease_name
gene_symbol
base_rank
base_score
best_mouse_model
best_genotype
hpo_mpo_coverage
candidate_count
```

### 6.2 完整知识库重排 A4

至少包含：

```text
disease_id
gene_symbol
final_rank
rank_change
supported_hpo_count
supported_hpo_fraction
supported_cpo_count
cpo_enrichment_pvalue
cpo_enrichment_fdr
```

其中：

```text
rank_change = base_rank - final_rank
```

正值表示排名提高。

### 6.3 Bridge-LOO A5

至少包含：

```text
disease_id
gene_symbol
bridge_loo_rank
bridge_edges_removed
remaining_supported_hpo_count
remaining_supported_cpo_count
```

A5 只用于审计，不作为是否允许进入 practical 单细胞展示的硬否决条件。

### 6.4 原始证据路径

每一行表示一条可审计路径：

```text
disease_id
gene_symbol
hpo_id
hpo_name
cpo_id
cl_id
cl_name
go_id
go_name
edge_scope
relation_type
source_genes
source_records
function_gene_membership
marker_membership
```

### 6.5 Function gene sets

最低字段：

```text
go_id or cpo_id
gene_symbol
source
evidence_code
```

### 6.6 单细胞数据元信息

预检阶段只读取：

```text
study_accession
disease_name
disease_id
tissue
sample_id
donor_id
condition
case_or_control
sex
age
batch
platform
assay_type
cell_label
cell_count
feature_names
raw_count_available
processed_matrix_available
publication
```

预检阶段不得读取表达数值。

---

## 7. 候选 disease–gene 池生成

### 7.1 基础候选条件

从全部 A0/A4 结果中保留：

```text
final_rank < base_rank
```

即 A4 确实改善了该 disease–gene pair 的排名。

同时要求：

1. 该基因是冻结 gold/known disease gene；
2. 该基因存在于 A0 候选池中；
3. 至少有一条完整的 HPO–CPO–Function–Gene 路径；
4. 至少有一个可解释的 CL 和 GO/CPO function；
5. 结果不是由于文件缺失、候选池变化或排序错误产生；
6. 对应疾病 HPO 和小鼠模型输入已被冻结。

### 7.2 不硬编码排名阈值

主候选生成阶段不强制写死：

```text
base_rank > 5
final_rank <= 3
rank_change >= 5
```

这些字段全部保留，但只作为候选特征和后续优先顺序，不作为唯一纳入条件。

原因是不同疾病的候选数、表型覆盖和初始排序难度不同，固定阈值可能排除具有很好机制和单细胞数据条件的病例。

### 7.3 候选特征表

生成：

```text
case_candidate_table.tsv
```

至少包含：

```text
candidate_case_id
disease_id
disease_name
gene_symbol
base_rank
final_rank
rank_change
candidate_count
base_rank_percentile
final_rank_percentile
supported_hpo_count
supported_hpo_fraction
supported_cpo_count
raw_function_path_count
cell_type_specific_path_count
cell_general_path_count
cpo_enrichment_pvalue
cpo_enrichment_fdr
bridge_loo_rank
bridge_rank_change
bridge_dependency_status
predicted_cell_count
predicted_function_count
top_predicted_cells
top_predicted_functions
```

---

## 8. 无训练、无混合权重的候选优先级

本阶段不训练病例选择模型，也不构造：

```text
w1 × rank_gain
+ w2 × final_rank
+ w3 × CPO coverage
+ w4 × dataset quality
```

### 8.1 Pareto 前沿

对每个候选使用以下方向：

```text
final_rank              越小越好
rank_change             越大越好
supported_hpo_count     越大越好
cell_type_specific_paths 越多越好
cell_general_fraction   越低越好
data availability       越完整越好
```

如果候选 A 在所有维度上均不差于候选 B，且至少一个维度更好，则 A 支配 B。

保留第一层 Pareto front 作为优先检索集合；如果没有可用单细胞数据，再依次考察第二层和后续 Pareto front。

### 8.2 确定性并列顺序

在同一 Pareto front 内，仅为执行顺序使用以下字典序，不解释为生物学总分：

```text
1. final_rank ascending
2. rank_change descending
3. supported_hpo_count descending
4. cell_type_specific_path_count descending
5. cell_general_fraction ascending
6. disease_id ascending
7. gene_symbol ascending
```

### 8.3 Bridge-LOO 的处理

将候选标注为：

```text
bridge_stable
bridge_partially_dependent
bridge_fully_dependent
bridge_not_evaluable
```

解释规则：

- `bridge_stable`：更适合讨论潜在泛化机制；
- `bridge_fully_dependent`：仍可用于完整知识库的已知生物学恢复展示；
- Bridge-LOO 状态不应在查看单细胞表达后再修改；
- 主报告必须同时展示 A0、A4 和 A5。

---

## 9. 为每个候选自动提取预测细胞与功能

### 9.1 候选级证据聚合

对每个 disease–gene pair，按 HPO 折叠重复路径，输出：

```text
supported_hpo_ids
supported_hpo_count
cpo_ids
predicted_cl_ids
predicted_go_ids
cell_type_specific_path_count
cell_general_path_count
bridge_source_genes
source_records
```

### 9.2 主细胞的确定规则

在不读取表达值的前提下：

1. 统计每个 CL 被多少个独立疾病 HPO 支持；
2. 优先支持 HPO 数量更多的 CL；
3. 并列时优先 `cell_type_specific`；
4. 并列时优先可映射到目标组织的数据集标签；
5. 仍并列时按 CL ID 字典序；
6. 允许保留多个并列主细胞，但必须在 freeze manifest 中全部声明。

不允许根据后续哪个细胞 DE 更显著来更换预测细胞。

### 9.3 主功能的确定规则

1. 统计每个 GO/CPO function 被多少个独立疾病 HPO 支持；
2. 优先支持 HPO 数量更多的功能；
3. 并列时优先 `cell_type_specific` 路径；
4. 并列时优先具有可用 function gene set 的功能；
5. 并列时优先删除候选基因和桥接源基因后仍保留更多成员的功能；
6. 仍并列时按 GO/CPO ID 字典序。

主分析最多冻结：

```text
1–2 个主细胞
1 个主功能
最多 2 个次级功能
```

---

## 10. 单细胞数据自动检索

### 10.1 检索对象

针对每个候选疾病，Codex 搜索：

- GEO；
- SRA；
- ArrayExpress/BioStudies；
- Human Cell Atlas；
- CELLxGENE Discover；
- Single Cell Portal；
- NGDC/GSA-Human；
- 疾病相关数据门户；
- 论文补充数据和作者公开仓库。

### 10.2 查询模板

对每个疾病自动构造：

```text
"<disease name>" single-cell RNA-seq
"<disease synonym>" scRNA-seq patient control
"<gene symbol>" single-cell disease
"<disease name>" single-nucleus RNA-seq
"<disease name>" PBMC single cell
"<disease name>" tissue single cell transcriptome
```

基因查询仅用于发现疾病相关数据，不能因为论文已经报告该基因差异而优先选择病例。

### 10.3 数据发现输出

生成：

```text
single_cell_dataset_catalog.tsv
```

字段：

```text
candidate_case_id
study_accession
publication_id
disease_name
disease_synonym
tissue
assay_type
species
patient_donor_count
control_donor_count
other_group_count
longitudinal_or_cross_sectional
raw_count_available
processed_matrix_available
donor_metadata_available
cell_labels_available
variant_metadata_available
access_restriction
possible_tissue_match
possible_cell_match
possible_independence
retrieval_status
notes
```

---

## 11. Metadata-only 数据预检

### 11.1 强制信息边界

预检程序可以读取：

- 数据集说明；
- 样本和 donor metadata；
- 细胞标签；
- 每个 donor×cell label 的细胞数量；
- feature axis 中有哪些基因；
- 文件格式和可下载性；
- 研究是否病例—对照或纵向设计。

预检程序不得读取：

- 候选基因表达值；
- 候选 function module score；
- 疾病与对照差异表达结果；
- 原论文中以候选基因为重点的差异结果，除非仅用于数据定位并在审计表中标记；
- 为了寻找阳性结果而预先运行任意 DE。

### 11.2 必要 gate

一个数据集进入表达分析，至少需要：

1. 人类患者样本；
2. 存在健康对照、疾病阶段对照或可解释的同一患者纵向对照；
3. 组织与疾病和 CPO 预测具有合理关系；
4. 有明确 sample/donor ID；
5. 有原始 count 或可构建 pseudobulk 的表达矩阵；
6. 预测细胞在数据标签中可识别；
7. 候选基因存在于 feature axis；
8. 主 function gene set 在 feature axis 中具有足够覆盖；
9. 数据没有被用于构建当前待验证的 marker/function gene set，或可以明确标记证据重叠；
10. 数据获取权限允许在当前项目中完成分析。

### 11.3 供体数量等级

不直接用一个阈值删除所有小样本罕见病数据，而是分层：

```text
Tier A:
  patient donors >= 4
  control donors >= 4

Tier B:
  patient donors >= 2
  control donors >= 2

Tier C:
  单患者纵向或单患者 + 多健康参考

Tier D:
  无独立 donor 信息，或只有细胞级标签
```

使用规则：

- Tier A：可作为优先主病例；
- Tier B：可作为探索性病例；
- Tier C：只能做描述性/纵向机制展示；
- Tier D：不进行病例组统计推断，可用于数据可视化压力测试。

### 11.4 细胞数量 gate

建议每个 donor 的预测细胞：

```text
>= 20 cells：可构建正式 pseudobulk
10–19 cells：低覆盖，仅敏感性分析
< 10 cells：不作为主终点
```

### 11.5 Function gene-set gate

在不读取表达值时先检查 feature presence：

```text
practical module feature coverage >= 0.80
strict module 删除 candidate gene 和 bridge source genes 后 >= 5 genes
推荐 strict module >= 15 genes
```

实际非零表达覆盖在冻结后读取表达矩阵时再检查，不能用于事后更换主功能。

---

## 12. 候选病例的确定方式

### 12.1 推荐模式：分析所有通过预检的病例

最稳妥的实现是：

> 对所有通过 metadata-only preflight 的 disease–gene–dataset 组合运行相同单细胞验证流程，并完整报告支持、部分支持、未支持和数据不足结果。

这样可以避免只展示一个阳性病例造成的选择偏差。

### 12.2 只需要一个展示 Case 时

如果资源限制要求只运行一个主病例，按以下规则确定：

1. 从第一层 Pareto front 开始；
2. 选择通过 metadata-only preflight 的最高数据质量层级；
3. 同层级按第 8.2 节的确定性顺序；
4. 在读取表达值前生成并冻结 manifest；
5. 即使结果不支持，也不得替换为另一个病例作为同一轮“主病例”；
6. 后续病例只能标记为新的 exploratory follow-up。

### 12.3 Discovery 与 Confirmatory 分离

建议建立两种模式：

```text
exploratory_scan:
  对全部合格病例运行
  允许产生病例发现和方法调试

confirmatory_case:
  在新数据上预先冻结 disease/gene/cell/function/endpoints
  不根据 exploratory 结果调整
```

当前任务优先实现 `exploratory_scan`，但所有选择过程必须可审计。

---

## 13. Case Freeze

### 13.1 冻结顺序

```text
读取 A0/A4/A5 和证据链
  ↓
完成公开数据发现
  ↓
完成 metadata-only preflight
  ↓
确定 disease–gene–dataset
  ↓
确定 predicted cell 和 predicted function
  ↓
确定 reference cells 和阴性对照
  ↓
生成 freeze manifest 与 SHA-256
  ↓
最后读取表达矩阵
```

### 13.2 Freeze manifest

每个病例生成：

```yaml
case:
  case_id: auto_generated
  disease_id: frozen_value
  disease_name: frozen_value
  candidate_gene: frozen_value
  case_mode: exploratory | confirmatory

ranking:
  baseline_arm: A0
  reranked_arm: A4
  bridge_arm: A5
  base_rank: frozen_value
  final_rank: frozen_value
  bridge_loo_rank: frozen_value
  rank_change: frozen_value

knowledge_prediction:
  primary_hpo_ids: frozen_list
  primary_cpo_ids: frozen_list
  predicted_cl_ids: frozen_list
  predicted_go_or_function_ids: frozen_list
  bridge_source_genes: frozen_list
  edge_scope_summary: frozen_value

single_cell:
  study_accession: frozen_value
  tissue: frozen_value
  assay_type: frozen_value
  patient_donors: frozen_list
  control_donors: frozen_list
  longitudinal_groups: frozen_list
  dataset_cell_labels: frozen_list

endpoints:
  primary:
    - candidate_gene_abnormality_in_predicted_cell
    - predicted_cell_specificity
    - strict_downstream_function_module
  secondary:
    - practical_function_module
    - predicted_cell_composition
    - cell_state_change

controls:
  reference_cells: frozen_list
  unrelated_functions: frozen_list
  comparator_genes: frozen_list
  expression_matched_null_seed: frozen_value
```

### 13.3 禁止事项

冻结后禁止：

- 更换候选基因；
- 更换主要预测细胞；
- 从多个 GO 中挑选最显著者作为主功能；
- 根据 DE 结果更换病例；
- 删除未支持的主终点；
- 将次级分析重新命名为预声明主终点。

如确需修改，必须生成新版本 manifest，并将结果标记为 post hoc。

---

## 14. 单细胞数据处理

### 14.1 数据质控

至少执行：

- 低质量细胞过滤；
- 低表达基因过滤；
- 线粒体比例检查；
- doublet 检测；
- ambient RNA 检查；
- donor、condition、batch 核验；
- 样本与条件完全混杂检查；
- 原始标签与 marker 一致性检查；
- 病例与对照是否使用相同实验体系检查。

所有阈值写入：

```text
resolved_config.yaml
run_manifest.json
qc_summary.tsv
```

### 14.2 细胞标签标准化

```text
dataset cell label
  ↓
exact label / synonym
  ↓
Cell Ontology ID
  ↓
parent-child compatibility
  ↓
marker-supported review
  ↓
final standardized CL mapping
```

输出：

```text
cell_label_mapping.tsv
```

字段：

```text
dataset_label
cl_id
cl_name
mapping_type
ontology_distance
marker_support
manual_review
usable_for_primary_endpoint
```

不得把较远的祖先细胞静默视为 exact match。

### 14.3 Pseudobulk

统计单位固定为：

```text
donor × cell type
```

对同一 donor、同一 cell type 的原始 counts 求和：

```text
PseudobulkCount(donor, cell, gene)
  = sum CellCount(cell_i, gene)
```

同时输出：

- 原始 pseudobulk counts；
- library size；
- CPM；
- log2(CPM + 1)；
- donor×cell profile 的细胞数；
- donor×cell profile 的检测基因数。

禁止把每个 cell 当作独立生物学重复。

---

## 15. 候选基因验证

### 15.1 不预设只有差异表达一种结果

根据变异和疾病机制，候选基因异常可以表现为：

- mRNA 表达下降；
- 代偿性表达升高；
- 无显著转录量变化；
- 异常剪接或转录本；
- 等位基因不平衡；
- 蛋白功能缺陷导致的下游 module 变化；
- 特定细胞状态中的异常定位。

### 15.2 E0：预测细胞中的候选基因状态

检查：

```text
feature presence
nonzero detection per donor
pseudobulk expression
case-control effect
longitudinal effect
splice/allele evidence if available
```

基础表达效应：

```text
GeneEffect(predicted_cell)
  = mean(Expression_case)
    - mean(Expression_control)
```

小样本病例主要报告：

- 效应方向；
- 效应量；
- 健康参考分布；
- donor-level 原始点；
- 稳健性分析；
- 不夸大细胞级 P 值。

### 15.3 E1：细胞特异性

对每个 donor：

```text
CellContrast(donor)
  = Expression(predicted_cell)
    - mean(Expression(reference_cells))
```

进一步计算 difference-in-differences：

```text
DiD
  = mean(CellContrast_case)
    - mean(CellContrast_control)
```

该分析回答：

> 候选基因的异常是否在预测细胞中比参考细胞更明显？

---

## 16. CPO 功能验证

### 16.1 Practical module

使用完整预测 function gene set：

```text
G_practical = FunctionGenes(predicted_function)
```

回答：

> 当前完整知识库定义的功能模块在预测细胞中是否异常？

### 16.2 Strict downstream module

删除：

```text
candidate_gene
bridge_source_genes
unmapped_genes
feature-axis absent genes
```

得到：

```text
G_strict
  = G_practical
    - candidate_gene
    - bridge_source_genes
    - unavailable_genes
```

读取表达后再记录：

- 每个 donor 中可检测的 module gene 数；
- nonzero detection fraction；
- 是否满足冻结时声明的最低可分析标准。

不得因 strict set 表现不显著而换用另一个功能作为主结果。

### 16.3 Module score

在同一 dataset 和 cell type 内，对每个 function gene 跨 donor 标准化：

```text
Z(donor, gene)
  = [X(donor, gene) - mean(X(*, gene))]
    / sd(X(*, gene))
```

样本级功能分数：

```text
ModuleScore(donor, cell, function)
  = mean[Z(donor, gene)]
```

也可以补充：

- pseudobulk GSEA；
- camera/roast 等考虑基因相关性的集合检验；
- leading-edge genes；
- function gene coherence；
- disease effect 方向。

主结论不能只依赖一种 module 算法。

---

## 17. 细胞组成与细胞状态

### 17.1 组成变化

对每个 donor：

```text
CellFraction(donor, cell)
  = N_cells(donor, cell) / N_total_cells(donor)
```

### 17.2 状态变化

细胞内功能 module、候选基因表达和 DE 属于状态变化。

必须明确区分：

```text
细胞比例变化
!= 候选基因表达变化
!= 细胞内部功能变化
```

组成和状态可以同时发生，但必须分别报告。

---

## 18. 阴性对照

### 18.1 正确功能 + 错误细胞

在冻结的参考细胞中计算相同 function module，检验预测功能是否具有细胞特异性。

### 18.2 正确细胞 + 表达匹配随机 gene set

1. 按预测细胞中的平均表达将基因划分为表达 bins；
2. 对目标 function set 中每个基因，从同 bin 抽样；
3. 保持集合大小；
4. 固定随机种子；
5. 默认运行 1,000 次；
6. 计算真实 module effect 的经验 percentile 和经验 P 值。

### 18.3 正确细胞 + 无关功能

预检阶段按以下规则选择，而不是看完表达后选择：

- 同一细胞但不被该疾病 HPO 支持；
- gene-set 大小接近；
- feature coverage 接近；
- 与主功能不是祖先/后代重复关系。

### 18.4 相近基础排名但无 CPO 支持的比较基因

从同一疾病候选池中冻结：

- A0 排名接近；
- A4 未明显提升；
- 缺少主 CPO 功能证据；
- 在 feature axis 中存在。

用于检验是否所有表型候选都能在单细胞中找到变化。

---

## 19. 多数据集与 Meta-analysis

如果同一疾病存在多个独立数据集：

1. 每个数据集独立处理；
2. 不直接合并原始细胞；
3. 在每个 cohort 内构建 donor pseudobulk；
4. 分别计算候选基因和 module effect；
5. 方向一致时做 fixed-effect 或 random-effect meta-analysis；
6. 报告异质性；
7. 不因一个 cohort 失败而删除它。

对 cohort effect \(\hat\theta_i\) 和标准误 \(SE_i\)：

\[
w_i=\frac{1}{SE_i^2}
\]

\[
\hat\theta_{meta}
=\frac{\sum_iw_i\hat\theta_i}{\sum_iw_i}
\]

同时报告：

- 95% CI；
- Cochran Q；
- heterogeneity P；
- I-squared；
- leave-one-cohort-out 稳定性。

---

## 20. 结果状态分类

每个病例使用统一状态，不允许只保留阳性结果。

### 20.1 `fully_supported`

满足：

```text
排名改善
+ 预测细胞可验证
+ 候选基因分子异常得到支持
+ strict function module 得到支持
```

### 20.2 `partially_supported`

例如：

- 候选基因和预测细胞得到支持，但 strict function 未支持；
- function module 得到支持，但基因本身无明显转录变化；
- 多个 cohort 方向一致但统计能力不足。

### 20.3 `mechanism_only_supported`

候选基因不表现为明显 DE，但预测细胞中的下游功能模块异常。

### 20.4 `gene_only_supported`

候选基因在预测细胞中异常，但预声明 function module 未得到支持。

### 20.5 `not_supported`

数据充足，但预声明的基因、细胞和功能终点均未形成一致证据。

### 20.6 `insufficient_data`

表达覆盖、供体数、细胞数或 function gene-set 大小不足。

### 20.7 `preflight_failed`

未读取表达值前即因组织、标签、权限或数据结构不符合条件而终止。

---

## 21. 探索结果的展示规则

### 21.1 最优策略

主报告展示：

1. 全部候选病例数量；
2. 找到公开单细胞数据的数量；
3. 通过 metadata-only preflight 的数量；
4. 实际完成分析的数量；
5. 各状态病例数量；
6. 一个代表性完整支持病例；
7. 至少一个未支持或数据不足病例。

### 21.2 代表病例选择

如需在正文重点展示一个病例：

- 优先使用冻结前确定的 primary case；
- 若 exploratory scan 分析了所有病例，可选择信息最完整的病例作为 illustration；
- 但必须同时给出完整病例表，明确该代表病例是探索性选择；
- 不得把探索性代表病例描述为 blind confirmatory validation。

---

## 22. 输出目录

```text
outputs/single_cell_case_exploration/
├── run_manifest.json
├── resolved_config.yaml
├── discovery/
│   ├── case_candidate_table.tsv
│   ├── pareto_fronts.tsv
│   ├── candidate_priority_order.tsv
│   ├── candidate_evidence_summary.tsv
│   └── single_cell_dataset_catalog.tsv
├── preflight/
│   ├── dataset_metadata_preflight.tsv
│   ├── cell_mapping_preflight.tsv
│   ├── function_feature_coverage.tsv
│   ├── case_dataset_pair_status.tsv
│   └── preflight_failures.tsv
├── frozen_cases/
│   ├── frozen_case_registry.tsv
│   └── <case_id>/
│       ├── case_freeze_manifest.yaml
│       ├── manifest.sha256
│       └── frozen_endpoints.tsv
├── cases/
│   └── <case_id>/
│       ├── run_manifest.json
│       ├── qc/
│       ├── ranking/
│       ├── pseudobulk/
│       ├── gene/
│       ├── function/
│       ├── composition/
│       ├── controls/
│       ├── figures/
│       └── report/
├── summary/
│   ├── all_case_results.tsv
│   ├── case_status_counts.tsv
│   ├── endpoint_support_matrix.tsv
│   ├── supported_and_failed_cases.md
│   └── representative_case_report.md
└── figures/
```

---

## 23. 关键结果表

### 23.1 `all_case_results.tsv`

```text
case_id
disease_id
disease_name
gene_symbol
study_accession
tissue
data_tier
base_rank
final_rank
rank_change
bridge_loo_rank
predicted_cell
predicted_function
cell_mapping_type
patient_donors
control_donors
candidate_gene_effect
candidate_gene_effect_ci
candidate_gene_status
cell_specificity_effect
cell_specificity_status
practical_module_effect
practical_module_status
strict_module_effect
strict_module_empirical_p
strict_module_status
wrong_cell_status
composition_status
overall_case_status
limitations
```

### 23.2 `candidate_evidence_summary.tsv`

```text
case_id
hpo_id
hpo_name
cpo_id
cl_id
cl_name
go_id
go_name
edge_scope
relation_type
source_genes
source_records
candidate_is_bridge_source
candidate_is_function_member
candidate_is_marker
```

### 23.3 `endpoint_support_matrix.tsv`

行：病例。  
列：

```text
rank_rescue
cell_match
gene_detection
gene_effect
cell_specificity
practical_module
strict_module
wrong_cell_control
random_set_control
composition_change
```

值：

```text
supported
not_supported
insufficient_data
not_applicable
```

---

## 24. 核心图形

### Figure 1：Case discovery 漏斗

```text
All improved disease–gene pairs
  ↓
Evidence-auditable pairs
  ↓
Pairs with public single-cell data
  ↓
Metadata-preflight-passing pairs
  ↓
Frozen analyzed cases
  ↓
Supported / partial / failed
```

### Figure 2：候选 Pareto 图

例如：

```text
x-axis: base_rank - final_rank
y-axis: final_rank
point size: supported_hpo_count
shape: data quality tier
label: frozen cases
```

### Figure 3：代表病例完整知识链

```text
Disease
  → HPO
  → CPO
  → predicted cell
  → predicted function
  → candidate gene
  → matched mouse genotype/MPO
```

### Figure 4：候选基因细胞特异性

展示预测细胞与冻结参考细胞中的 donor-level pseudobulk 效应。

### Figure 5：Strict function module

展示：

- practical module；
- 删除候选基因后的 module；
- 删除候选基因和 bridge source genes 后的 strict module；
- wrong-cell 对照；
- expression-matched random null。

### Figure 6：病例结果矩阵

使用热图或 tile plot 展示所有病例的各终点状态，必须保留未支持病例。

---

## 25. Codex 模块设计

```text
src/cellphenorank/single_cell_discovery/
├── candidate_builder.py
├── pareto_selection.py
├── evidence_path_audit.py
├── dataset_search.py
├── dataset_catalog.py
├── metadata_preflight.py
├── tissue_match.py
├── cell_ontology_match.py
├── function_feature_preflight.py
├── case_freeze.py
├── pseudobulk.py
├── candidate_gene_endpoint.py
├── cell_specificity.py
├── function_module.py
├── expression_matched_null.py
├── composition.py
├── cohort_meta.py
├── case_status.py
├── case_report.py
└── exploration_summary.py
```

---

## 26. 建议命令行

### 26.1 生成候选池

```bash
PYTHONPATH=src python3 -B -m cellphenorank build-single-cell-case-candidates \
  --config configs/single_cell_case_discovery.yaml \
  --output-dir outputs/single_cell_case_exploration/discovery
```

### 26.2 检索数据集

```bash
PYTHONPATH=src python3 -B -m cellphenorank discover-single-cell-datasets \
  --candidates outputs/single_cell_case_exploration/discovery/case_candidate_table.tsv \
  --output-dir outputs/single_cell_case_exploration/discovery
```

### 26.3 Metadata-only preflight

```bash
PYTHONPATH=src python3 -B -m cellphenorank preflight-single-cell-cases \
  --config configs/single_cell_case_preflight.yaml \
  --output-dir outputs/single_cell_case_exploration/preflight
```

### 26.4 冻结病例

```bash
PYTHONPATH=src python3 -B -m cellphenorank freeze-single-cell-cases \
  --config configs/single_cell_case_freeze.yaml \
  --preflight outputs/single_cell_case_exploration/preflight/case_dataset_pair_status.tsv \
  --output-dir outputs/single_cell_case_exploration/frozen_cases
```

### 26.5 运行单病例

```bash
PYTHONPATH=src python3 -B -m cellphenorank validate-single-cell-case \
  --manifest outputs/single_cell_case_exploration/frozen_cases/<case_id>/case_freeze_manifest.yaml \
  --output-dir outputs/single_cell_case_exploration/cases/<case_id>
```

### 26.6 批量运行所有冻结病例

```bash
PYTHONPATH=src python3 -B -m cellphenorank validate-frozen-single-cell-cases \
  --registry outputs/single_cell_case_exploration/frozen_cases/frozen_case_registry.tsv \
  --output-dir outputs/single_cell_case_exploration/cases
```

### 26.7 汇总

```bash
PYTHONPATH=src python3 -B -m cellphenorank summarize-single-cell-case-exploration \
  --cases-dir outputs/single_cell_case_exploration/cases \
  --output-dir outputs/single_cell_case_exploration/summary
```

---

## 27. 伪代码

### 27.1 候选生成

```python
def build_candidates(base_rankings, full_rankings, bridge_rankings, paths):
    merged = merge_rankings(base_rankings, full_rankings, bridge_rankings)
    improved = merged[merged.final_rank < merged.base_rank]

    candidates = []
    for row in improved.itertuples():
        evidence = paths[
            (paths.disease_id == row.disease_id)
            & (paths.gene_symbol == row.gene_symbol)
        ]
        if not has_complete_hpo_cpo_function_gene_path(evidence):
            continue
        candidates.append(summarize_candidate(row, evidence))

    return assign_pareto_fronts(candidates)
```

### 27.2 确定性主病例选择

```python
def select_primary_case(preflight_table):
    eligible = preflight_table[preflight_table.status == "pass"]
    if eligible.empty:
        return None

    quality_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    eligible["quality_order"] = eligible.data_tier.map(quality_order)

    eligible = eligible.sort_values(
        by=[
            "pareto_front",
            "quality_order",
            "final_rank",
            "rank_change",
            "supported_hpo_count",
            "cell_type_specific_path_count",
            "disease_id",
            "gene_symbol",
            "study_accession",
        ],
        ascending=[True, True, True, False, False, False, True, True, True],
    )
    return eligible.iloc[0]
```

该顺序只决定执行对象，不生成生物学概率，也不使用任何表达结果。

### 27.3 Freeze 安全检查

```python
def validate_freeze_boundary(manifest, expression_access_log):
    if expression_access_log.any_access_before(manifest.created_at):
        raise RuntimeError("Expression values were accessed before case freeze")

    if sha256(manifest) != manifest.recorded_sha256:
        raise RuntimeError("Frozen case manifest has changed")
```

---

## 28. 必须实现的测试

### 28.1 候选生成

```text
只保留 final_rank < base_rank
无完整 function 路径的候选被过滤
候选数由输入结果动态决定
不硬编码任何 disease 或 gene
同一输入重复运行结果一致
```

### 28.2 Pareto front

```text
被完全支配的候选不进入第一前沿
非支配候选均进入同一前沿
并列顺序可复现
不计算人工混合分数
```

### 28.3 数据发现

```text
疾病同义词可生成查询
重复 accession 被去重
人和小鼠数据不混淆
病例、对照和纵向设计正确识别
受限数据被明确标记
```

### 28.4 Metadata-only 边界

```text
preflight 不能读取 expression matrix values
允许读取 feature names 和 cell labels
表达访问日志完整
freeze 前访问表达值会报错
```

### 28.5 Cell Ontology 映射

```text
exact 与 parent/child 映射分开
较远祖先不被静默接受
预测细胞缺失时 preflight 失败
同义词映射可追溯
```

### 28.6 Freeze

```text
manifest 包含 disease/gene/cell/function/dataset/endpoints
manifest hash 可复现
任何字段修改都会改变 hash
冻结后主终点不可静默更换
```

### 28.7 Pseudobulk

```text
donor×cell counts 正确求和
不能把 cell 当独立重复
低细胞数 profile 正确标记
library size 与 counts 一致
```

### 28.8 Strict module

```text
candidate gene 被删除
bridge source genes 被删除
重复 gene 被去重
不可用 gene 被记录
set 过小返回 insufficient_data
```

### 28.9 阴性对照

```text
随机集合大小保持一致
表达 bin 匹配正确
固定 seed 可复现
wrong-cell 使用冻结参考细胞
经验 P 值计算正确
```

### 28.10 批量汇总

合成数据至少覆盖：

1. 排名改善且全部终点支持；
2. 基因支持但功能不支持；
3. 功能支持但基因不 DE；
4. wrong-cell 强于 predicted cell；
5. preflight 失败；
6. 数据不足；
7. 完全未支持；
8. 多病例中不能只输出阳性病例。

---

## 29. 实施顺序

### P0：候选发现

1. 从当前 A0/A4/A5 和证据路径生成动态候选池；
2. 输出候选特征表；
3. 实现 Pareto front 和确定性执行顺序；
4. 确认代码中不存在固定 disease/gene 名称。

### P1：数据发现与预检

1. 为优先候选搜索公开单细胞数据；
2. 构建 dataset catalog；
3. 完成组织、供体、细胞标签和 feature-axis 预检；
4. 输出所有失败原因；
5. 不访问表达值。

### P2：冻结

1. 选择全部合格病例或确定性主病例；
2. 冻结 disease、gene、cell、function、dataset 和 endpoints；
3. 写入 hash；
4. 建立表达访问日志。

### P3：表达分析

1. 下载并标准化数据；
2. 构建 donor×cell pseudobulk；
3. 计算候选基因 E0/E1；
4. 计算 practical/strict function module；
5. 分析组成变化；
6. 运行 wrong-cell、随机集合和比较基因对照。

### P4：病例汇总

1. 汇总全部病例；
2. 按统一状态分类；
3. 保留未支持病例；
4. 输出一个代表性机制链图；
5. 明确 exploratory 与 confirmatory 边界。

---

## 30. 最终结果解释模板

### 30.1 完全支持的探索病例

> 在 CellPhenoRank 的表型基础排名中，疾病 D 的已知致病基因 G 排名为第 \(r_0\)；加入完整 HPO–CPO–Function–Gene 知识库和稳定约束重排后，G 上升至第 \(r_1\)。知识库预先将多个疾病 HPO 连接到细胞 C 和功能 F。对独立患者单细胞数据的 donor-level 分析显示，G 在预测细胞 C 中出现分子状态异常，且该异常相对冻结的参考细胞更明显。在删除 G 及 HPO–CPO 桥接源基因后，功能 F 的 strict downstream module 仍表现出异常，并强于表达匹配随机集合和错误细胞对照。该探索病例支持完整 CPO 知识库同时具有候选重排、细胞定位和功能解释价值。

### 30.2 基因不 DE，但功能支持

> 候选基因 G 在预测细胞中可被检测，但未表现出显著转录量差异。与此同时，预先冻结的下游功能 F 在删除 G 和桥接源基因后仍表现出协调性异常。考虑到致病变异可能主要影响剪接、蛋白活性、稳定性或亚细胞定位，该结果支持预测的细胞功能机制，但不以 mRNA 差异表达作为候选基因功能缺陷的唯一标准。

### 30.3 结果未支持

> 尽管完整 CPO 知识库改善了候选基因 G 的排名，但在预先冻结的单细胞数据、预测细胞和预测功能终点中未观察到一致支持。该病例被保留为 `not_supported`，不通过更换细胞、功能、数据集或候选基因制造阳性结果，用于界定当前知识库重排的适用边界。

### 30.4 数据不足

> 该 disease–gene pair 具有可审计的排名改善和 CPO 机制路径，但公开单细胞数据在供体数、预测细胞覆盖、function gene-set 覆盖或原始 counts 可用性方面不足，因此被标记为 `insufficient_data`，不据此判断生物学预测真伪。

---

## 31. 最终原则

Codex 的目标不是：

```text
在公开单细胞数据中找到一个看起来最显著的基因
```

而是：

```text
从 CellPhenoRank 已经产生的全部排名改善结果出发
  ↓
按统一规则形成候选病例
  ↓
仅根据知识路径与数据元信息完成筛选
  ↓
冻结疾病、基因、细胞和功能
  ↓
用独立单细胞表达数据进行验证
  ↓
完整保留阳性、部分支持、阴性与数据不足病例
```

必须始终保持：

```text
候选排名结果
!= 病例选择过程
!= 单细胞表达验证
```

最有说服力的成果不是一个事后挑选的阳性基因，而是一个可审计的探索框架：

1. **CellPhenoRank 自动提出值得验证的 disease–gene pair；**
2. **CPO 自动提出预测细胞与预测功能；**
3. **公开单细胞数据在冻结后独立检验这些预测；**
4. **所有通过预检的结果都被保留，不以结果好坏更换病例。**

---

## 32. 2026-09-04 执行状态

本计划的 P0-P4 已完成首轮实现与运行。当前实际结果为：

- 26 个 A4 改善且有完整 function 路径的候选；
- 12 个候选找到相关单细胞或转录组资源；
- 6 个候选具有疾病-基因精确匹配的人体单细胞资源；
- 2 个组合通过 metadata-only preflight 并在表达读取前冻结；
- `BTK-XLA / E-MTAB-11845`：`insufficient_data`；
- `FOXG1 deletion / GSE273180`：`gene_only_supported`，仅为单细胞系描述性证据；
- `fully_supported`：0。

完整结果、失败原因、解释边界和输出索引见：

```text
CellPhenoRank_single_cell_validation_results.md
outputs/single_cell_case_exploration/summary/supported_and_failed_cases.md
```
