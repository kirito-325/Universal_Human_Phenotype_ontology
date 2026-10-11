# CellPhenoRank 单细胞验证协议

## 结论先行

Huntington disease 不适合作为第一主案例，应保留为失败压力测试：

- 系统预测的是 fine neuronal CL，现有主分析只能映射到 generic `neuron`，本体距离为 5-7；
- top function gene sets 只有 3-17 个基因；
- HTT 从基础第 17 降至重排第 197；
- 两个 donor-level 研究均未验证预测功能，预测细胞效应仅排第 6-8；
- 现有 HPO-CPO top cell 还偏向 large basket/SST interneuron，而不是预期的 medium spiny neuron。

当前主案例改为 IPEX (`OMIM:304790`, `FOXP3`) 的 GSE167976。该数据包含 11 名
IPEX 和 11 名健康供者、55,656 个有公开注释的 CD4 T 细胞。为保证两个分析层
donor-disjoint，cohort 2 排除在 cohort 1 重复出现的 B4、B5、P4，最终比较为：

- cohort 1: 6 IPEX vs 4 healthy donors；
- cohort 2 donor-disjoint: 5 IPEX vs 7 healthy donors。

数据来源：[GSE167976](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167976)，
研究报告为 11 IPEX 与 11 healthy donors，且初始/复现队列结果一致
([Nature Immunology paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8173714/))。

## 强制顺序

```text
metadata-only case audit
-> freeze predicted gene, target cell and downstream functions
-> hash frozen case manifest
-> load expression matrix
-> donor x cell pseudobulk
-> preflight gates
-> expression-matched null analysis
-> report supported / not_supported without replacement
```

不得查看 causal-gene effect、module effect、DE、composition 或 enrichment 后再更换 endpoint 或疾病。
若主 endpoint 不支持，保留结果并报告失败。

## 案例选择门槛

主案例必须同时满足：

1. patient/control 来自相同组织和实验类型；
2. 每组至少 4 个独立 donor；重复 donor 只能保留一个队列或显式建模；
3. 每个 donor 的预测细胞至少 20 cells；
4. 预测 CL 到数据标签的本体距离为 0；距离 1-2 只能作为次级分析；
5. 致病基因在目标细胞中的 donor-level 检测率至少 80%；
6. 已知疾病基因遮蔽后 function gene set 至少 5 genes，推荐至少 15；
7. function gene coverage 按实际表达检测计算，至少 80%，不能只检查矩阵行名；
8. 单细胞数据未用于 HPO-CPO 或 marker/function gene set 构建；
9. selection 只使用 metadata、预测覆盖和可检验性，不能使用表达结果。

## 预声明 endpoint

每个案例冻结四层 endpoint，主次关系不得颠倒：

- `E0 causal_gene`: 致病基因在预测细胞中的表达、等位基因或蛋白功能缺陷；
- `E1 cell_specific_gene_defect`: 预测细胞中的基因缺陷是否强于错误细胞；
- `E2 strict_downstream`: 删除已知疾病基因后选择的下游 cell/function module；
- `E3 practical_downstream`: 允许当前知识库关系的生物学解释层。

不同变异不能统一要求 mRNA 完全缺失：无义/NMD、外显子删除可预期转录下降；
missense、部分剪接和显性负效应变异需要等位基因或蛋白功能证据。mRNA 保留不代表
致病基因功能正常。

下游两层定义为：

- `strict_primary`: 从 HPO-CPO 来源基因和待测 gene set 中删除已知疾病基因后选择；
- `practical_positive_control`: 允许当前知识库关系，仅用于生物学合理性检查，不能支持新基因泛化。

IPEX 当前冻结结果：

| Role | Cell | Function | Masked genes | Genes after mask | Strict retained |
|---|---|---|---|---:|---:|
| strict primary | T cell | positive regulation of T cell migration | FOXP3 | 20 | 100% |
| practical control | regulatory T cell | regulatory T cell differentiation | FOXP3 | 6 | 0.61% |

## 分析单位与分数

- 统计单位固定为 `donor x cell type`，禁止把 cell 当生物学重复；
- raw counts 聚合后转换为 log2 CPM；
- E0 比较预测细胞中致病基因的 patient-control log2 CPM；
- E1 比较 `(target cell - mean reference cells)` 的 patient-control 差分效应；
- 每个 function gene 在同一 dataset/cell type 的 donor 间做 z-score；
- donor module score 为可用 function genes 的平均 z-score；
- 每个 cohort 单独报告 Welch effect，再进行 fixed-effect meta-analysis；
- 报告 95% CI、Cochran Q、heterogeneity P 和 I-squared；
- composition 与 function-state 分开，聚合的 `T_cell_all` 不进入 composition 分母。

## Null 与细胞特异性

随机 gene set 必须保持目标 set 大小，并为每个目标基因从相同 pseudobulk
平均表达十分位中抽取背景基因。默认 1000 次。

需要细胞特异性的 endpoint 还必须通过：

- 预测细胞 marker identity control；
- marker set 中删除疾病基因；
- 预测细胞相对预声明错误细胞在两个队列中均排名第一。

## 支持判定

E0/E1 致病基因 endpoint 只有同时满足下列条件才标记 `supported`：

- 所有 cohort 中致病基因效应方向一致；
- meta FDR < 0.05；
- 致病基因 donor-level 检测率通过；
- E1 的预测细胞相对错误细胞差分效应方向与预声明一致。

下游 endpoint 只有同时满足下列条件才标记 `supported`：

- 所有 dataset preflight 通过；
- meta FDR < 0.05；
- 若预声明方向，meta effect 与方向一致；
- donor-disjoint cohorts 效应方向一致；
- 每个 cohort 均达到 expression-matched random gene-set 第 95 百分位；
- 对要求细胞特异性的 endpoint，identity control 和错误细胞 rank 均通过。

未通过时使用 `not_supported`，不得改阈值、替换 GO term 或删除不利队列。

## 当前 IPEX 结果

- FOXP3 在预测的 Treg 中下降：meta effect -1.148 log2 CPM，95% CI
  [-1.992, -0.304]，FDR 0.0077；两个 donor-disjoint cohort 同方向，I-squared=0。
- Treg 相对 Tconv 的 FOXP3 差分缺陷更强：meta effect -1.184，95% CI
  [-1.775, -0.593]，FDR 0.00017；两个 cohort 同方向。
- FOXP3 在所有 donor-level Treg profile 中均可检测，因此下降不是单细胞 dropout 造成的。
- strict T-cell migration endpoint: `supported`；meta effect -0.612，95% CI
  [-0.941, -0.283]，FDR 0.00054；两个 donor-disjoint cohort 同方向；匹配随机
  percentile 为 0.993 和 0.986。
- practical Treg differentiation endpoint: `not_supported`；FDR 0.805，两个队列
  方向不一致，预测 Treg 的 function effect 均不是 cell-type rank 1。
- Treg marker identity control 在两个队列均通过，说明不支持结果不是标签失效造成的。

案例展示可以支持：系统优先级较高的 FOXP3 在预测 Treg 中呈现细胞特异表达缺陷，
并伴随一个 leakage-controlled T-cell migration module 异常。它不证明 FOXP3 是由
这批单细胞数据新发现的致病基因，也不代表所有患者发生 mRNA 完全缺失。

本轮 E0/E1 是在用户澄清案例目标后补充的开发性展示，不属于最初 blind freeze 的
confirmatory endpoint。正式论文级验证需要把 E0/E1 写入新 case manifest，并在尚未
查看的独立患者单细胞数据中复验。
