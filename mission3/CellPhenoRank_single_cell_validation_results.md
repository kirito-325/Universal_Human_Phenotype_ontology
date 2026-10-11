# CellPhenoRank 预测驱动单细胞验证结果

> 执行日期：2026-09-04  
> 分析模式：prediction-driven exploratory scan  
> 当前结论：本轮得到 1 个 `gene_only_supported` 描述性病例和 1 个 `insufficient_data` 主病例，没有 `fully_supported` 病例。

> 后续已按“文献先行、原始数据复算、再看排名”的新方向完成 SAVI-STING1 案例。该结果见 `CellPhenoRank_literature_first_SAVI_case.md`；本文其余内容保留为此前 prediction-driven scan 的完整审计记录。

## 1. 本轮回答了什么

本轮不是从已知阳性单细胞论文反向挑病例，而是从当前 A4 重排真正改善的已知致病基因出发：

1. 生成全部排名改善且有完整 `HPO -> CPO -> Cell -> Function -> Gene` 路径的病例；
2. 按 Pareto front 和固定顺序检索数据；
3. 只用元数据、cell label、feature axis 和文件可用性做 preflight；
4. 在表达读取前冻结 disease、gene、cell、function 和 endpoints；
5. 冻结后运行供体或技术重复级 pseudobulk、候选基因终点、strict module 和表达匹配随机集对照；
6. 保留失败和不支持结果。

## 2. 队列流转

| 阶段 | 数量 |
|---|---:|
| A4 排名改善且有完整 function 路径 | 26 |
| 找到相关单细胞或转录组资源 | 12 |
| 疾病和致病基因精确匹配的人体单细胞资源 | 6 |
| 通过 metadata-only preflight | 2 |
| 完成表达分析 | 2 |
| `gene_only_supported` | 1 |
| `insufficient_data` | 1 |
| `preflight_failed` | 24 |
| `fully_supported` | 0 |

## 3. 两个冻结病例

| 病例 | A0 -> A4 | A5 | 数据等级 | E0 候选基因 | E1 细胞特异性 | E2 strict module | 总体状态 |
|---|---:|---:|---|---|---|---|---|
| BTK-XLA / E-MTAB-11845 | 3 -> 1 | 1 | Tier B, 3 vs 3 donors | 不可检验 | 不可检验 | 未支持 | `insufficient_data` |
| FOXG1 deletion / GSE273180 | 9 -> 7 | 10 | Tier C, 1 vs 1 cell line | 描述性支持 | 未支持 | 未支持 | `gene_only_supported` |

## 4. 主病例为什么失败：BTK-XLA

第一 Pareto front 中，`BTK-XLA / E-MTAB-11845` 是唯一通过冻结前必要 gate 的组合，因此按规则成为主病例。数据来自 3 名 XLA 和 3 名健康供体，包含 day 35 和 6 months 两个时间点。day 35 由“最大化两组最小供体数，再按时间点字典序”这一纯元数据规则确定为主时间点。

冻结预测：

- 候选基因：`BTK`
- 预测细胞：T cell，`CL:0000084`
- 主功能：T cell receptor signaling pathway，`GO:0050852`
- strict module：删除 `BTK` 和全部候选级桥接源基因后 87 个基因

实际结果：

- `BTK` 存在于原始 10x feature 定义，但在论文发布的过滤后 RDS RNA 轴中不存在，所有 donor pseudobulk 中不可检测。因此 E0 是 `insufficient_data`，不能把补零后的 0 差异当作阴性结果。
- 实验只测量 spike peptide 刺激后分选的 T 细胞，没有合法的非 T 参考细胞，因此 E1 是 `insufficient_data`。
- strict TCR module 在 day 35 的 XLA-control 效应为 `0.0211`，表达匹配随机集经验 `P=0.9291`；6 months 效应为 `-0.1471`，方向不一致。因此 E2 为 `not_supported`。
- 该论文已进入 marker 文献库，存在来源重叠；即使 strict module 删除桥接源，也不能把该病例表述为完全独立验证。

结论：XLA 数据在供体数上尚可，但实验情境、候选基因检测和参考细胞结构都不适合当前目标。

## 5. 可用于展示的病例：FOXG1 缺失

`FOXG1` 是第三 Pareto front 的追加探索病例。公开数据 GSE273180 是 day 105 患者脑类器官，包含一个杂合 FOXG1 全基因缺失细胞系和一个健康细胞系，每个细胞系各 2 个技术重复。公开的 4 个过滤后计数矩阵共包含 29,924 个细胞。

冻结预测：

- 候选基因：`FOXG1`
- 并列主预测中的可映射细胞：`CL:2000028` cerebellum glutamatergic neuron
- 数据集映射：DL-EN/UL-EN，分析时合并为 `excitatory_neuron`
- 主功能：negative regulation of DNA-templated transcription，`GO:0045892`
- strict module：删除 `FOXG1` 和全部桥接源后 557 个基因，feature axis 覆盖 544 个

单细胞处理未使用候选基因或功能模块做细胞注释。固定 marker 将细胞分为兴奋性神经元、抑制性神经元、神经祖细胞、星形胶质祖细胞和增殖祖细胞。

### 5.1 E0：致病基因缺失是否在预测细胞中体现

在冻结的兴奋性神经元中：

```text
mean control FOXG1        = 8.8399 log2(CPM + 1)
mean FOXG1 deletion       = 7.5306 log2(CPM + 1)
deletion - control        = -1.3093
technical replicate rule = 2/2 deletion replicates below 2/2 controls
```

E0 状态为 `descriptively_supported`。这直接展示了当前预测体系改善排名的致病基因，其缺失在模型预测的神经细胞中可见。

### 5.2 E1：是否特异于预测细胞

`FOXG1` 下降的绝对效应：

| 细胞类 | deletion - control |
|---|---:|
| neural progenitor | -1.4619 |
| excitatory neuron, frozen prediction | -1.3093 |
| interneuron | -1.1481 |
| cycling progenitor | -1.0016 |
| astrocyte progenitor | +0.2209 |

冻结预测细胞的绝对效应排第 2，不是最强。因此 E1 为 `not_supported`。这说明单细胞数据支持“基因缺失在预测细胞中出现”，但不支持“异常只在预测细胞中最强”。

### 5.3 E2：去泄漏 strict 功能是否异常

兴奋性神经元中：

```text
strict theoretical genes = 557
feature-present genes     = 544
cross-replicate variable  = 494
module effect             = +0.1531
expression-matched null P = 1.0000
```

E2 为 `not_supported`。真实 strict module 的效应甚至不比表达匹配随机基因集极端，因此不能用该结果证明冻结的 `GO:0045892` 功能解释成立。

### 5.4 FOXG1 病例的准确表述

可以表述：

> 在 prediction-driven 候选中，FOXG1 从第 9 位上升到第 7 位。独立来源的患者脑类器官单细胞数据中，FOXG1 在预先冻结的兴奋性神经元内下降 1.309 log2(CPM+1)，且两个技术重复方向一致，说明致病基因缺失能够在模型预测的细胞类中得到实际体现。

不能表述：

- FOXG1 是盲法确认性验证；
- 两个技术重复等于两个独立患者；
- 异常对预测细胞具有唯一特异性；
- 当前 strict CPO 功能模块已经被验证；
- 一个病例证明了整个 CPO 关系库有效。

## 6. 其他关键失败病例

| 候选 | 数据 | 主要原因 |
|---|---|---|
| STING1-SAVI | GSE226598, 5 untreated SAVI vs 7 controls | Tier A，但冻结主细胞 DN2a thymocyte 不在 PBMC |
| RPGR-XLRP | SRP535874, retinal organoid | 细胞和功能匹配好，但只有 1 vs 1 cell line；8 个 FASTQ run 合计约 300 GB，未提供可直接复用的 count matrix |
| CYBB-X-CGD | EGAS00001005463 | neutrophil/monocyte 和功能匹配好，但 EGA 受限访问，精确 CYBB donor 数未解析 |
| ADAR-AGS6 | GSE220764, 2 vs 2 donors | 数据公开且基因匹配，但冻结的 A-to-I editing strict module 删除候选后少于 5 个基因 |
| FOXJ1-PCD43 | GSE272189 | 组织和细胞匹配，但患者是 DNAH5 PCD，不是 FOXJ1 PCD43 |

## 7. 当前判断

当前单细胞验证任务的主要限制来自病例数据适配：

1. 罕见病公开单细胞数据本身少；
2. 疾病匹配不等于预测主细胞匹配；
3. feature axis 存在不等于候选基因在目标细胞可检测；
4. 很多数据只有单患者、单细胞系或技术重复；
5. 已进入 marker 文献库的数据不能作为完全独立关系验证；
6. 宽泛 GO function 即使覆盖很好，也可能在表达匹配 null 下没有特异信号。

因此本轮最诚实的结果是：预测体系已经产生一个能展示“致病基因缺失在预测细胞中体现”的 FOXG1 描述性案例，但尚未产生同时支持候选基因、预测细胞特异性和去泄漏 strict 功能模块的完整病例。

## 8. 主要输出

- 候选表：`outputs/single_cell_case_exploration/discovery/case_candidate_table.tsv`
- 数据集目录：`outputs/single_cell_case_exploration/discovery/single_cell_dataset_catalog.tsv`
- Preflight：`outputs/single_cell_case_exploration/preflight/case_dataset_pair_status.tsv`
- 冻结登记：`outputs/single_cell_case_exploration/frozen_cases/frozen_case_registry.tsv`
- 全病例结果：`outputs/single_cell_case_exploration/summary/all_case_results.tsv`
- 终点矩阵：`outputs/single_cell_case_exploration/summary/endpoint_support_matrix.tsv`
- XLA 报告：`outputs/single_cell_case_exploration/cases/OMIM_300755__BTK__E_MTAB_11845/report/case_report.md`
- FOXG1 报告：`outputs/single_cell_case_exploration/cases/OMIM_613454__FOXG1__GSE273180/report/case_report.md`

## 9. 外部数据来源

- XLA processed scRNA-seq: https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11845
- XLA publication: https://pmc.ncbi.nlm.nih.gov/articles/PMC9293955/
- FOXG1 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273180
- FOXG1 publication: https://pmc.ncbi.nlm.nih.gov/articles/PMC12205996/
- SAVI GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226598
- RPGR dataset description: https://pmc.ncbi.nlm.nih.gov/articles/PMC11599861/
- CGD dataset description: https://pmc.ncbi.nlm.nih.gov/articles/PMC12556223/
