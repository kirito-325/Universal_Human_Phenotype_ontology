# Exomiser 2602 初排加 CPO 无参数重排

## 1. 比较目标

本实验将表型初排替换为 Exomiser 2602 当前 PhenoDigm 方法，然后原样运行
CellPhenoRank 的二元 CPO 富集、leave-one-HPO-out 稳定约束和 Bridge-LOO。

固定不变：

- benchmark v4 的 50 个疾病及 gold standard；
- 49 个候选基因和 1,068 个单基因 genotype；
- HPO 去重与祖先项删除规则；
- Strict/Extended HPO-CPO 图；
- GO function gene sets 和 marker 审计数据；
- CPO 超几何检验、优先顺序和拓扑重排规则。

唯一方法性改变是 A0 及其 leave-one-HPO-out 分数来自 Exomiser 2602。

## 2. 运行流程

```text
冻结疾病 HPO
  -> duplicate/ancestor reduction
  -> Exomiser 2602 HP-MP score
  -> current semi-symmetric PhenoDigm genotype score
  -> best genotype per gene
  -> full + leave-one-HPO-out gene ranks
  -> stable phenotype constraints
  -> binary CPO enrichment
  -> constrained reranking
```

运行命令：

```bash
PYTHONPATH=src python3 -B -m cellphenorank run-parameter-free \
  --config configs/real_frozen_parameter_free_2602.yaml \
  --output-dir outputs/parameter_free_2602
```

## 3. 结果

| Arm | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | 中位名次 |
|---|---:|---:|---:|---:|---:|---:|
| A0 Exomiser 2602 | 0.42 | 0.76 | 0.92 | 0.5509 | 0.6349 | 2.5 |
| A1 raw path | 0.36 | 0.66 | 0.82 | 0.4842 | 0.5541 | 3.0 |
| A2 enrichment | 0.32 | 0.66 | 0.82 | 0.4709 | 0.5447 | 3.0 |
| A3 Strict constrained | 0.48 | 0.74 | 0.92 | 0.5943 | 0.6675 | 2.0 |
| A4 Extended constrained | 0.56 | 0.88 | 0.92 | 0.6908 | 0.7440 | 1.0 |
| A5 Bridge-LOO | 0.46 | 0.72 | 0.92 | 0.5755 | 0.6527 | 2.0 |

相对 A0：

| Arm | 平均名次变化 | Win/Tie/Loss | Wilcoxon p |
|---|---:|---:|---:|
| A1 raw path | -1.28 | 3/26/21 | 0.000918 |
| A2 enrichment | -1.26 | 5/22/23 | 0.000921 |
| A3 Strict | +0.12 | 6/37/7 | 0.541 |
| A4 Extended | +1.34 | 26/22/2 | 0.0000988 |
| A5 Bridge-LOO | -0.06 | 4/38/8 | 0.564 |

## 4. Strict 主结果

A3 将 Recall@1 从 0.42 提高到 0.48，MRR 从 0.5509 提高到 0.5943，NDCG@10
从 0.6349 提高到 0.6675，中位名次从 2.5 改善到 2.0。Recall@10 保持 0.92，
Recall@5 从 0.76 降到 0.74。

发生 gold gene 名次变化的疾病只有 13 个：6 个改善、7 个下降。改善包括：

```text
PCDH15  9 -> 4
WAS     4 -> 1
BTK     3 -> 1
PLP1    3 -> 1
TNFSF11 2 -> 1
NR2F1   4 -> 3
```

疾病级配对差异不显著，不能声称 Strict CPO 已产生稳定总体提升。

## 5. Bridge-LOO

删除由该疾病 gold gene 独自支撑的 HPO-CPO 边后，A3 中两个最大改善消失：

- `PCDH15 9 -> 4` 回到 9；
- `WAS 4 -> 1` 变为 5。

Bridge-LOO 只剩 4 win / 38 tie / 8 loss，平均名次变化 -0.06，`p=0.564`。
虽然 Recall@1 和 MRR 仍略高于 A0，但没有稳定的疾病级总体收益。

## 6. Extended 敏感性分析

A4 的数值改善最大，并达到配对显著：26 win / 22 tie / 2 loss，`p=9.88e-5`。
但 A4 允许 `cell_general` 边，关系范围明显宽于 Strict，且当前没有对应的
Extended Bridge-LOO。因此 A4 只能作为“扩大 CPO 边覆盖可能带来收益”的敏感性结果，
不能替代 A3 主结果。

## 7. 稳定性与软件检查

```text
nonredundant HPO rows       799
valid LOO runs              799
stable constraints          43,264
constraint checks           129,792
constraint violations       0
strict CPO-evaluable cases  30/50
```

所有 CPO 约束排序均满足表型稳定顺序。A1/A2 明显下降，继续支持不直接按原始路径数或
无约束富集结果排序。

## 8. 当前结论

当前结果支持：

> 在 Exomiser 2602 表型初排上，Strict CPO 约束重排能够提高部分头部指标，但该变化
> 在疾病级配对检验中不显著，并且两个最大改善依赖 gold-gene HPO-CPO 桥接证据。

当前结果不支持：

- Strict CPO 已稳定提高总体致病基因预测性能；
- Extended 结果可以直接作为正式主结果；
- CPO enrichment P 值可以解释为致病概率。

正式结果目录：`outputs/parameter_free_2602/`。
