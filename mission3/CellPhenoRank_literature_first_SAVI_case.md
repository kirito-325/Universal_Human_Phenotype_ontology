# CellPhenoRank literature-first 单细胞案例：SAVI-STING1

> 案例状态：`case_supported`  
> 目的：先选择已有患者单细胞证据且原始数据可复算的罕见病，再检查当前 CPO 重排是否改善其致病基因排名。  
> 数据：GSE226598，5 名未治疗 SAVI 患者，7 名健康对照。

## 1. 为什么选择这个案例

该案例不是因为已经看到当前排名提升才定义单细胞终点，而是因为它满足以下数据条件：

1. 疾病和致病基因明确：`STING1` gain-of-function 导致 SAVI；
2. 原论文在患者 PBMC 单细胞数据中报告了清楚的细胞异常；
3. GEO 公开每名 donor 的原始 10x H5 count 文件；
4. 有 5 名未治疗患者和 7 名健康对照；
5. 可以从 raw counts 自行完成 QC、细胞注释、pseudobulk 和统计，而不是只引用论文结论。

主要来源：

- 单细胞论文：[de Cevins et al., Cell Reports Medicine, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10772457/)
- 原始数据：[GSE226598](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226598)
- 原始致病机制研究：[Activated STING in a vascular and pulmonary syndrome](https://pubmed.ncbi.nlm.nih.gov/25029335/)

## 2. 文献给出的生物学链

```text
STING1 gain-of-function
  -> STING 持续激活
  -> disease-associated monocyte
       high UPR / integrated stress response
       CCL3 / CCL4 / IL6 inflammation
  -> T cell early activation / senescence / apoptosis
  -> effector lymphocyte loss
  -> lymphopenia和系统性炎症表型
  -> OMIM:615934 SAVI
```

对应的冻结人类 HPO 包括：

| HPO | 表型 | 与单细胞结果的关系 |
|---|---|---|
| HP:0001888 | Decreased total lymphocyte count | 与效应 T 细胞减少直接对应 |
| HP:0001954 | Recurrent fever | 与炎症单核细胞相关，属于机制联系 |
| HP:0011227 | Elevated circulating CRP | 与炎症细胞状态相关，属于机制联系 |
| HP:0003565 | Elevated ESR | 与系统性炎症相关，属于机制联系 |
| HP:0002206 | Pulmonary fibrosis | SAVI 核心临床表现；PBMC 数据不能直接验证肺组织因果 |

这里不能把“异常单核细胞导致全部临床表型”写成人体中已经证明的单向因果关系。单细胞数据直接支持细胞状态和组成异常，连接到发热、炎症指标和肺病属于结合遗传机制与临床资料后的解释。

## 3. 我们怎样从原始数据复算

### 3.1 输入与冻结

- 纳入患者：P1、P2、P4、P5、P6；
- 纳入对照：C1、C2、C7、C8、C9、C10、C11；
- 排除所有 JAK inhibitor 治疗样本；
- 冻结 manifest SHA-256：`9dbce27726f82d5eb822f3713c238302fa55c0f00b6b0ee41304a386d4512331`；
- 统计单位始终为 donor，不把单个细胞作为独立重复。

### 3.2 处理流程

```text
12 个 raw 10x H5
  -> 每个样本独立 barcode QC
  -> 12,517,792 raw barcodes 中保留 93,066 cells
  -> 2,500 HVG + PCA + Leiden + UMAP
  -> 固定经典 marker 注释主要 PBMC cell class
  -> donor x cell-type pseudobulk
  -> 冻结的 UPR、炎症和细胞组成终点
  -> outcome ~ SAVI + sequencing batch 敏感性分析
```

细胞注释 marker 在配置文件中预先声明，没有使用 `STING1`、UPR signature 或病例-对照差异结果定义细胞类别。分析也没有读取论文提供的 SAVI-vs-control DEG 表。

## 4. 原始数据复算结果

| 终点 | 未校正效应 | 未校正 P | 批次校正效应 | 批次校正 P | 结论 |
|---|---:|---:|---:|---:|---|
| 全部 monocyte UPR pseudobulk score | -0.0123 | 0.9404 | +0.0020 | 0.9864 | 未支持整体平移 |
| high-UPR monocyte donor fraction | +0.4620 | 0.0529 | +0.4546 | 0.00486 | 支持异常亚群增加 |
| monocyte CCL3/CCL4/IL6 module | +1.4239 | 0.0274 | +1.4119 | 0.01334 | 支持炎症状态 |
| monocyte PPP1R15A | +1.5357 | 0.2045 | +1.4521 | 0.1488 | 方向一致但未支持 |
| effector T fraction of PBMC | -0.2416 | 0.00372 | -0.2453 | 0.00890 | 支持效应 T 细胞减少 |

### 4.1 对 UPR 结果的准确解释

不能说“所有 SAVI 单核细胞的 UPR 都升高”。全部 monocyte donor pseudobulk 的平均 UPR 没有变化。

可以说“患者中出现了 high-UPR 单核细胞亚群”：使用健康单核细胞 pooled 95th percentile 作为预声明阈值，SAVI donor 的 high-UPR monocyte 比例平均增加 45.5 个百分点，批次校正 P=0.00486。患者异质性很明显，P1 接近 0，P4/P5/P6 的 high-UPR 比例约为 0.70-0.84。这种亚群型而非整体平移的结果与原论文描述一致。

### 4.2 临床表型连接

复算同时观察到：

- 单核细胞 `CCL3/CCL4/IL6` 炎症模块显著升高；
- 效应 T 细胞占 PBMC 的比例显著降低；
- 两项结果在控制三个建库批次后仍保留。

其中效应 T 细胞减少与 `HP:0001888 Decreased total lymphocyte count` 的联系最直接。炎症单核细胞与 CRP、ESR、反复发热之间是合理但不完全直接的机制联系。

## 5. 再看 CellPhenoRank 排名

病例和复算终点确定后，从当前结果读取同一疾病：

| 方法 | STING1 排名 | 变化 |
|---|---:|---:|
| A0 current Exomiser 2602 PhenoDigm | 2 | 基础值 |
| A4 Extended CPO | 1 | 上升 1 位 |
| A5 Bridge-LOO | 2 | 回到基础值 |

A4 的 CPO enrichment：

```text
P                      = 5.0098e-07
FDR                    = 2.4548e-05
supported HPO          = 16
supported HPO fraction = 0.5926
supported cells        = 25
supported functions    = 20
```

因此，这个案例满足：

```text
已有患者单细胞异常证据
  + 原始数据能够自行复算
  + 对应罕见病的致病基因在 A4 中由第 2 位升至第 1 位
```

## 6. 单细胞结果与当前 CPO 图的重合程度

| 单细胞发现 | 当前 CPO 图 | 判断 |
|---|---|---|
| inflammatory monocyte | `CL:4072015 monocyte-derived Kupffer cell` | 单核细胞谱系相符，但组织亚型不精确 |
| CCL3/CCL4/IL6 inflammation | `GO:0001819 positive regulation of cytokine production` | 功能方向一致 |
| type I IFN / NF-kappaB | `GO:0032481`, `GO:0032728`, `GO:0043123` | 功能方向一致 |
| cellular senescence | `GO:0090398 cellular senescence` | 功能方向一致 |
| UPR / ISR | 当前 A4 主功能集合未直接包含 UPR | 未完全覆盖 |
| effector T-cell loss / lymphopenia | 疾病 HPO 中有 HP:0001888，但不是 STING1 当前支持路径之一 | 临床链成立，当前 CPO 路径未直接命中 |

所以这不是“每个单细胞发现都被当前 CPO 图精确复现”。更准确的说法是：当前完整图捕获了 STING1 相关的单核细胞谱系、细胞因子、IFN、NF-kappaB 和 senescence 机制，从而把 STING1 从第 2 位推到第 1 位；但细胞亚型和 UPR/淋巴细胞减少路径仍有改进空间。

## 7. Bridge-LOO 和来源重叠

A5 回到第 2 位，说明 A4 的提升完全依赖包含 `STING1` 本身的桥接证据。这个结果符合当前展示目的，因为这里验证的是完整知识库能否利用已知 gene-cell-function 关系，而不是要求删除正确答案后仍然恢复它。

同时必须披露：该单细胞论文已经存在于 marker 文献库。因此本案例可以证明：

- 完整知识库和公开单细胞结果在生物学方向上具有一致性；
- 当前 CPO 重排确实能利用这类关系改善致病基因排名；
- raw counts 复算不是单纯引用论文结论。

它不能证明：

- 对完全未见过的新论文或新疾病具有独立外部泛化能力；
- 去除 STING1 桥接信息后仍能提升该基因；
- 当前细胞映射已经精确到外周血 disease-associated monocyte 亚群。

## 8. 最终推荐表述

> SAVI 是由 STING1 gain-of-function 引起的罕见自炎症病。已有患者 PBMC 单细胞研究报告高 UPR/ISR 的 disease-associated monocyte、炎症因子升高以及效应淋巴细胞减少。我们从 GSE226598 的 12 个未治疗病例/对照 raw 10x H5 文件重新分析 93,066 个 QC 合格细胞，在 donor 层面复现了 high-UPR monocyte 亚群增加、单核细胞 CCL3/CCL4/IL6 炎症模块升高及效应 T 细胞比例降低。随后在同一疾病的预测任务中，STING1 由当前 Exomiser 2602 基础排名第 2 位提升至 A4 Extended CPO 的第 1 位。该案例说明完整 CPO 知识库能够利用与已知单细胞病理一致的细胞和功能关系改善致病基因排序，但其提升依赖 STING1 桥接证据，且不构成独立外部泛化验证。

## 9. 可复现入口

```bash
NUMBA_CACHE_DIR=/tmp/cellphenorank-numba \
MPLCONFIGDIR=/tmp/mpl-cache \
XDG_CACHE_HOME=/tmp/xdg-cache \
PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/cellphenorank-pycache \
python3 -B -m cellphenorank validate-literature-first-savi-case \
  --config configs/savi_literature_first_validation.yaml \
  --output-dir outputs/literature_first_savi_case
```

关键输出：

- `outputs/literature_first_savi_case/case_summary.tsv`
- `outputs/literature_first_savi_case/endpoint_results.tsv`
- `outputs/literature_first_savi_case/cellular/high_upr_monocyte_fraction.tsv`
- `outputs/literature_first_savi_case/cellular/cell_type_effects.tsv`
- `outputs/literature_first_savi_case/ranking/sting1_rank_change.tsv`
- `outputs/literature_first_savi_case/figures/cellular_validation.png`
- `outputs/literature_first_savi_case/figures/rank_change.png`
- `outputs/literature_first_savi_case/run_manifest.json`
