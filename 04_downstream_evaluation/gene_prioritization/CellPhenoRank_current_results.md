# CellPhenoRank 当前结果

> 结果日期：2026-09-04  
> 结果状态：冻结 benchmark 上的当前完整分析  
> 表型基线：Exomiser 2602 current PhenoDigm  
> CPO 方法：无人工连续权重的二元证据链富集与稳定约束重排  
> 正式配置：`configs/real_frozen_parameter_free_2602.yaml`  
> 结果目录：`outputs/parameter_free_2602/`

## 1. 当前研究问题

当前分析的主要目标是评价：

> 我们构建的 HPO-CPO 关系、CPO function gene sets 和细胞上下文信息，能否在
> Exomiser 2602 表型初排基础上改善已知罕见病致病基因的排序。

这是知识库辅助的已知生物学恢复评价。正式分析保留正确答案基因及其真实功能关系，
不从候选池或功能基因集中删除 gold gene。

Bridge-LOO 仅作为补充审计：它不删除 gold gene，而是删除由该疾病 gold gene
独自支撑的 HPO-CPO 桥接边，用于判断结果对直接答案来源的依赖程度。

## 2. 冻结评价数据

疾病、HPO 和 gold mouse models 均在载入 CPO 资源前冻结：

```text
diseases                         50
single-gene genotypes         1,068
excluded multi-gene models      354
candidate genes                  49
original disease HPO rows       835
nonredundant HPO rows            799
```

当前 Exomiser 2602 `HP_MP_MAPPINGS` 完整表有 95,051,352 行。本地 benchmark
子矩阵有 3,143,719 行，覆盖 589/591 个唯一疾病 HPO 和 2,685/2,699 个候选 MPO。

## 3. 当前流程

```text
冻结疾病 HPO
    -> duplicate/ancestor reduction
    -> Exomiser 2602 HP-MP semantic similarity
    -> current semi-symmetric PhenoDigm genotype score
    -> best genotype per gene
    -> A0 candidate-gene ranking
    -> full + leave-one-HPO-out ranks
    -> stable phenotype constraints
    -> HPO-CPO-Function-Gene enrichment
    -> constrained CPO reranking
```

正式重排不使用 marker/function 人工混合权重、连续 HPO-CPO 置信度或 lambda 融合。
Marker 仅作为细胞身份一致性和并列辅助信息，不能单独构成主要功能支持。

## 4. 分析组定义

| Arm | 定义 | 回答的问题 |
|---|---|---|
| A0 | Exomiser 2602 表型初排，不使用 CPO | 基础表型预测效果 |
| A1 | 直接按原始 CPO 路径数量排序 | 未校正路径计数是否有效 |
| A2 | 按 CPO 富集排序，但无表型稳定约束 | 单独富集排序是否有效 |
| A3 | 仅 `cell_type_specific` Strict CPO 边，并满足 LOO 稳定顺序 | 高特异性关系子集效果 |
| A4 | Strict 加 `cell_general` 完整 CPO 图，并满足 LOO 稳定顺序 | 完整知识库效果 |
| A5 | Strict 图删除 gold-gene-only HPO-CPO 桥接边 | 直接答案来源依赖审计 |

按照当前明确的项目目标，A4 最接近“完整构建资源是否有效”；A3 是高置信关系子集，
A5 是循环性和未知基因泛化风险的补充审计。

## 5. 总体结果

| Arm | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | 中位名次 |
|---|---:|---:|---:|---:|---:|---:|
| A0 Exomiser 2602 | 0.42 | 0.76 | 0.92 | 0.5509 | 0.6349 | 2.5 |
| A1 raw path | 0.36 | 0.66 | 0.82 | 0.4842 | 0.5541 | 3.0 |
| A2 enrichment | 0.32 | 0.66 | 0.82 | 0.4709 | 0.5447 | 3.0 |
| A3 Strict CPO | 0.48 | 0.74 | 0.92 | 0.5943 | 0.6675 | 2.0 |
| A4 Full/Extended CPO | **0.56** | **0.88** | **0.92** | **0.6908** | **0.7440** | **1.0** |
| A5 Bridge-LOO | 0.46 | 0.72 | 0.92 | 0.5755 | 0.6527 | 2.0 |

## 6. 疾病级配对结果

| Arm | 平均名次变化 | 95% CI | Win/Tie/Loss | Wilcoxon p |
|---|---:|---:|---:|---:|
| A1 raw path | -1.28 | [-2.06, -0.56] | 3/26/21 | 0.000918 |
| A2 enrichment | -1.26 | [-2.02, -0.56] | 5/22/23 | 0.000921 |
| A3 Strict CPO | +0.12 | [-0.14, 0.44] | 6/37/7 | 0.541 |
| A4 Full/Extended CPO | **+1.34** | **[0.60, 2.02]** | **26/22/2** | **9.88e-5** |
| A5 Bridge-LOO | -0.06 | [-0.24, 0.12] | 4/38/8 | 0.564 |

正值表示相对 A0 的 gold gene 名次提高。

## 7. 完整 CPO 知识库结果

A4 将完整构建的 `cell_type_specific + cell_general` CPO 关系用于稳定约束重排：

```text
Recall@1   0.42 -> 0.56
Recall@5   0.76 -> 0.88
Recall@10  0.92 -> 0.92
MRR         0.5509 -> 0.6908
NDCG@10     0.6349 -> 0.7440
median rank 2.5 -> 1.0
```

50 个疾病中有 26 个改善、22 个不变、2 个下降，疾病级配对检验
`p=9.88e-5`。这说明在当前封闭候选集和知识库辅助设置下，完整 CPO 资源包含能够
显著改善已知致病基因排序的信息。

改善较明显的例子：

```text
EDNRA   9 -> 1
SALL1   9 -> 2
NPHP3   7 -> 1
BUB1B   7 -> 2
DCLRE1C 8 -> 3
GNAS    5 -> 1
CYBB   16 -> 12
IDS     7 -> 3
PCDH15  9 -> 5
```

两个下降病例：

```text
LZTFL1  1 -> 2
GATA3   1 -> 10
```

GATA3 的明显下降需要单独检查其 `cell_general` 路径、竞争候选和图连接度，不能仅用
总体指标掩盖。

## 8. Strict 高特异性关系子集

A3 只使用 `cell_type_specific` 且通过关系状态与解剖兼容筛选的 HPO-CPO 边。

它将 Recall@1 从 0.42 提高到 0.48，MRR 从 0.5509 提高到 0.5943，NDCG@10
从 0.6349 提高到 0.6675；但 Recall@5 从 0.76 降到 0.74。疾病级结果为
6 win / 37 tie / 7 loss，`p=0.541`，没有显著总体提升。

因此当前显著信号主要来自包含 `cell_general` 的完整关系图，而不是 Strict 子集。

## 9. Bridge-LOO 审计

A5 不删除正确答案，只删除由该 gold gene 独自支撑的 HPO-CPO 桥接边。

A3 中两个最大的 Strict 改善在 Bridge-LOO 后消失：

```text
PCDH15  A0:9 -> A3:4 -> A5:9
WAS     A0:4 -> A3:1 -> A5:5
```

A5 最终为 4 win / 38 tie / 8 loss，`p=0.564`。这说明部分 Strict 改善依赖
gold gene 参与构建的直接桥接证据，但不否定这些关系在已知生物学恢复任务中的真实性。

## 10. CPO 可评价子集

Strict 图中有 30/50 个疾病至少包含两个可比较的 CPO-mappable HPO。在该子集中：

| Arm | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | 中位名次 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.333 | 0.667 | 0.867 | 0.4670 | 0.5559 | 4.0 |
| A3 Strict | 0.433 | 0.600 | 0.867 | 0.5346 | 0.6060 | 2.5 |
| A4 Full/Extended | 0.500 | 0.833 | 0.867 | 0.6403 | 0.6915 | 1.5 |
| A5 Bridge-LOO | 0.400 | 0.600 | 0.867 | 0.5079 | 0.5855 | 3.5 |

## 11. 稳定性与实现检查

```text
valid leave-one-HPO-out runs  799
stable phenotype constraints  43,264
constraint checks             129,792
constraint violations         0
strict HPO-CPO edges           14,126
full/extended HPO-CPO edges   156,267
```

两次完整运行的 HPO reduction、LOO、稳定约束、Strict enrichment、A3、A5 和
benchmark 关键输出 SHA-256 完全一致。当前测试共 19 项，全部通过。

## 12. 当前可以支持的结论

当前结果支持：

> 在冻结的 50 个罕见病、49 个候选基因评价中，将完整 CPO 关系和功能基因集加入
> Exomiser 2602 表型初排后，已知致病基因的 Top 1、Top 5、MRR 和 NDCG@10 均在
> 数值上提高，且疾病级配对名次变化达到统计显著，说明构建的完整 CPO 知识库包含
> 疾病相关排序信息。

其中“显著”指 A4 相对 A0 的疾病级配对检验，不表示每一种 CPO 边类型或每一个 gene
set 都已经分别通过独立验证。

## 13. 当前不能支持的结论

当前结果尚不能证明：

- Strict 高特异性关系子集能够稳定提高总体预测性能；
- CPO 能够在构建时未知的致病基因上保持相同提升；
- A4 的全部提升分别来自真实 gene sets 和真实 HPO-CPO 关系，而不是更高图覆盖；
- 当前 49 基因封闭候选池结果等同于全基因组诊断性能；
- CPO enrichment P 值是基因致病概率。

此外，A4 是在查看已有结果后根据当前研究目标重新定位为完整知识库结果，因此本轮应
视为探索性证据。正式发表时应在新冻结验证集上预先声明主要比较。

## 14. 下一步验证

为了分别证明 gene sets 和关系本身有效，需要增加：

1. Gene-set degree-matched null：保持 gene set 大小和基因连接度，随机置换基因；
2. HPO-CPO degree-preserving null：保持 HPO/CPO 节点度数，随机重连关系；
3. 在新冻结疾病队列上预声明 A4 对 A0 为主要比较；
4. 使用独立单细胞病例验证预测基因、细胞类型和功能状态。

## 15. 主要结果文件

- 完整结果：`outputs/parameter_free_2602/`（以下产物均未上传）
- 总体指标：`benchmark/benchmark_metrics.tsv`
- 疾病级指标：`benchmark/disease_level_metrics.tsv`
- 配对统计：`benchmark/statistical_tests.tsv`
- A3 Strict 排名：`reranking/final_constrained_rankings.tsv`
- A4 完整图排名：`reranking/extended_constrained_rankings.tsv`
- A5 Bridge-LOO：`bridge_loo/bridge_loo_rankings.tsv`
- 2602 矩阵说明：[`EXOMISER_2602_PHENODIGM_MATRIX.md`](EXOMISER_2602_PHENODIGM_MATRIX.md)
- 完整方法说明：[`CellPhenoRank_parameter_free_reranking_revision_plan.md`](CellPhenoRank_parameter_free_reranking_revision_plan.md)
