# CellPhenoRank 项目说明与算法规范 v2

> 文档状态：旧版连续加权 CellPath 方案，保留为 legacy 对照，不再是正式主重排方法。  
> 当前正式方案：`CellPhenoRank_parameter_free_reranking_revision_plan.md`。  
> 原始设计稿：`CellPhenoRank_project_plan.md`  
> 当前实现包：`src/cellphenorank/`  
> Legacy 配置：`configs/real_frozen.yaml`

---

## 1. 项目定位

CellPhenoRank 是一个可审计的罕见病候选致病基因优先级系统。系统首先利用疾病
HPO 与具体小鼠 genotype 的 MPO 相似度生成基础排名，再利用 HPO-CPO、细胞 marker
和细胞功能 gene sets 对候选基因进行细胞机制重排。

当前系统解决三个彼此分离的问题：

1. **候选生成**：哪些小鼠模型在整体表型上与疾病最相似？
2. **候选重排**：哪些候选基因同时符合预测的细胞类型与细胞功能背景？
3. **案例验证**：预测基因的表达或功能缺陷，是否能在患者单细胞的预测细胞中体现？

核心逻辑边界为：

```text
候选基因生成 != CPO 机制重排 != 单细胞案例验证
```

单细胞数据不参与候选基因训练或 benchmark 选择，只用于独立案例展示和机制验证。

---

## 2. 当前可以与不可以支持的结论

### 2.1 当前可以支持

- 在固定当前 HPO-MPO 矩阵时，CPO、marker 和 function gene sets 会改变已知基因排名；
- 每个排名变化可以追溯到具体 HPO、MPO、CL、GO 和基因证据；
- practical 模式下，完整重排在当前冻结候选池中优于表型基线；
- IPEX 案例中，系统排名靠前的 FOXP3 在预测 Treg 中表现出细胞特异表达缺陷；
- 新 MPO-HPO 矩阵完成后可以通过替换 provider 输入接入。

### 2.2 当前不可以支持

- 不能声称当前结果证明了未知致病基因发现能力；
- 不能声称新 MPO-HPO 矩阵优于旧矩阵，因为新矩阵尚未接入正式比较；
- 不能把 practical 排名提升全部解释为泛化能力；
- 不能把单细胞表达变化解释为新的因果证明；
- 不能把当前 283 基因候选池结果解释为全基因组诊断性能。

主要原因是 strict masking 后性能明显下降，说明 practical 提升包含较强的已知基因
直接证据贡献。

---

## 3. 当前实现状态

| 模块 | 状态 | 说明 |
|---|---|---|
| MatrixSimilarityProvider | 已实现 | 支持标准矩阵和当前 MP-HPO 表 |
| CellPhenotypeSimilarityProvider | 未实现 | 等待新矩阵 |
| HybridSimilarityProvider | 未实现 | 等待新矩阵后实现 A6 |
| genotype 级 BMA | 已实现 | 只使用 abnormal MPO |
| genotype 聚合到基因 | 已实现 | 当前主聚合为 max |
| HPO-CPO 投影 | 已实现 | 疾病级和候选特异级 |
| marker/function IDF | 已实现 | 组内归一化后做频率校正 |
| CellPathScore | 已实现 | 路径加权平均 |
| A0-A3 消融 | 已实现 | surface、marker、function、完整模型 |
| A4-A6 消融 | 未实现 | 等待新 MPO-HPO 矩阵 |
| Strict gene masking | 已实现 | 遮蔽映射来源和直接 gene-set 成员 |
| Gene-set null | 已实现 | 保持每个 set 在候选池中的大小 |
| 网络间接支持 | 未实现 | 当前输出固定为 0 |
| 时间留出验证 | 未实现 | 后续任务 |
| 单细胞 E0-E3 | 已实现 | IPEX 为当前案例，HD 为压力测试 |

---

## 4. 总体数据流

```text
Rare disease HPO profile
        |
        v
HPO-MPO SimilarityProvider
        |
        v
Disease x concrete mouse genotype score
        |
        v
Best genotype per human ortholog gene
        |
        +-----------------------------+
        |                             |
        v                             v
Candidate-specific HPO-MPO       HPO-CPO mapping
evidence                          + CL markers
                                  + GO function genes
        |                             |
        +--------------+--------------+
                       v
              Candidate CellPathScore
                       |
                       v
               Percentile rank fusion
                       |
                       v
                 Final gene ranking
                       |
                       v
        Frozen patient single-cell case validation
```

---

## 5. 当前真实输入

正式配置见 `configs/real_frozen.yaml`。

| 输入 | 当前来源 |
|---|---|
| 疾病 HPO | benchmark v4 frozen `diseases.tsv` |
| 小鼠模型 | benchmark v4 frozen `models.tsv` |
| genotype-MPO | benchmark v4 frozen `model_phenotypes.tsv` |
| 当前 MP-HPO | `MPO_HPO_mapping_v3/.../mpo_hpo_mapping_v31.tsv` |
| HPO-CPO | `HPO_link_v5/results/links/hpo_cpo_links.tsv.gz` |
| Cell marker | `marker_literuture/marker_statistics/significant_marker_scores.tsv` |
| GO function genes | `GO_function/results/go_function_genes.tsv` |
| Gold models/genes | benchmark v4 frozen `gold_models.tsv` |

冻结 benchmark 包含：

- 50 个疾病；
- 1,422 个具体小鼠 genotype；
- 283 个候选人类基因；
- 182 个 gold mouse models；
- 49 个疾病满足至少两个可比较 HPO 的 CPO-evaluable 条件。

冻结 manifest 同时满足：

```text
frozen = true
middle_layer_independent = true
```

因此疾病、核心 HPO 和 gold models 在加载 CPO 前已经固定。

---

## 6. 模块一：HPO-MPO 相似度接口

统一接口：

```python
get_score(hpo_id, mpo_id) -> float in [0, 1]
get_evidence(hpo_id, mpo_id) -> provenance dict
```

当前 `MatrixSimilarityProvider`：

- 只加载当前疾病 HPO 与候选模型 MPO 的交集；
- 缺失 pair 返回 0，并标记 `not_mapped`；
- 重复 pair 保留最高有效分；
- 分数大于 1 时使用 `q/(1+q)` 压缩到 0-1；
- relation type 可以乘配置化系数。

当前关系系数：

| Relation | Factor |
|---|---:|
| exact | 1.0 |
| hp_broader | 0.9 |
| hp_narrower | 0.8 |
| uncertain | 0.7 |

因此 provider 实际返回：

\[
q_{effective}(h,p)
=
clip(q_{raw}(h,p)\times relation\_factor, 0, 1)
\]

未来接入新矩阵时，只需要保证 provider 输出相同接口和 0-1 分数。

---

## 7. 模块二：疾病与小鼠 genotype 表型评分

疾病 \(d\) 的 HPO 集合：

\[
H_d=\{h_1,\ldots,h_n\}
\]

具体小鼠 genotype \(m\) 的异常 MPO 集合：

\[
P_m=\{p_1,\ldots,p_k\}
\]

### 7.1 疾病到模型的正向覆盖

\[
S_{H\rightarrow P}(d,m)
=
\frac{\sum_{h\in H_d}w_h\max_{p\in P_m}q(h,p)}
{\sum_{h\in H_d}w_h}
\]

它回答：疾病表型有多少能够被该 genotype 解释？

### 7.2 模型到疾病的反向一致性

\[
S_{P\rightarrow H}(d,m)
=
\frac{1}{|P_m|}\sum_{p\in P_m}\max_{h\in H_d}q(h,p)
\]

它回答：该 genotype 的额外异常是否与疾病一致？

### 7.3 双向最佳匹配平均 BMA

\[
S_{BMA}(d,m)
=
\frac{
\sum_h w_h\max_p q(h,p)
+\alpha\sum_p\max_h q(h,p)
}{
\sum_h w_h+\alpha|P_m|
}
\]

当前：

```yaml
reverse_match_weight: 1.0
```

### 7.4 最强单对表型

\[
S_{max}(d,m)=\max_{h,p}q(h,p)
\]

### 7.5 genotype 总分

\[
S_{model}(d,m)
=
0.8S_{BMA}(d,m)+0.2S_{max}(d,m)
\]

当前只使用 `status=abnormal` 的 MPO。`tested_normal` 和 `untested` 可以被读取，
但尚未进入当前基因优先级总分。

每个 genotype 同时输出每个 HPO 的最佳 MPO：

\[
p^*(h,m)=\arg\max_{p\in P_m}q(h,p)
\]

以及候选特异的解释强度：

\[
E_{d,m}(h)=w_hq(h,p^*)
\]

---

## 8. 模块三：小鼠模型聚合到候选基因

一个人类基因可能对应多个小鼠 genotype。当前主聚合为：

\[
S_{base}(d,g)=\max_{m\in M_g}S_{model}(d,m)
\]

即选择与疾病最相似的具体 genotype 作为该基因的代表模型。

同时保留：

- 最佳 genotype；
- Top 3 genotype 平均分；
- genotype 数量；
- 最佳 genotype 的 MPO 数量；
- 模型分数标准差；
- 最佳 HPO-MPO pair；
- HPO-MPO 覆盖率。

基础基因分数随后在每个疾病内部转为 percentile：

\[
B_{d,g}=PercentileRank(S_{base}(d,g))
\]

最高分为 1，最低分为 0；同分使用平均 percentile，最终 ordinal rank 再按
gene symbol 稳定打破完全相同分数。

---

## 9. 模块四：HPO-CPO 边与置信度

每个 CPO 表示一个不可拆分的细胞机制：

```text
CPO = (CL cell type, GO biological process)
```

当前 HPO-CPO 原始置信度优先读取 `geometric_consensus`。实际进入重排的置信度为：

\[
R(h,c)
=
clip(R_{raw},0,1)
\times ScopeFactor
\times ConfidenceFactor
\]

当前 scope factor：

| Scope | Factor |
|---|---:|
| cell_type_specific / specific / direct_hpo | 1.0 |
| cell_general / 其他 | 0.5 |

文本置信度 factor：

| Confidence | Factor |
|---|---:|
| high | 1.0 |
| medium | 0.8 |
| low | 0.5 |

正式配置还要求 anatomy status 为 `allowed` 或兼容状态，并且每个 HPO 最多保留
置信度最高的 25 个唯一 CPO。

### 9.1 疾病级 CPO 画像

\[
C_d(c)=\sum_{h\in H_d}w_hR(h,c)
\]

### 9.2 疾病 CPO 覆盖率

\[
Coverage_d
=
\frac{
\sum_h w_h\mathbf{1}[h\text{ has usable CPO}]
}{
\sum_h w_h
}
\]

---

## 10. 模块五：marker 分数

对于细胞 \(t\) 和候选基因 \(g\)，marker 原始证据为：

\[
M_{raw}(g,t)
=
\log(1+marker\_weight)
\times specificity
\times reproducibility
\]

随后在每个细胞内除以该细胞最大 marker 分数：

\[
M_{norm}(g,t)
=
\frac{M_{raw}(g,t)}{\max_u M_{raw}(u,t)}
\]

再计算归一化 IDF：

\[
IDF_M(g)
=
\frac{
\log\frac{N_T+1}{n_g^M+1}
}{
\log(N_T+1)
}
\]

最终：

\[
M'(g,t)=clip(M_{norm}(g,t)IDF_M(g),0,1)
\]

在很多细胞中都是 marker 的泛化基因会被降低。marker 缺失返回 0，但不会删除候选。

---

## 11. 模块六：function 分数

当前 GO function 资源使用基因的 annotation count。原始分数为：

\[
F_{raw}(g,f)=\log(1+annotation\_count)
\]

没有 experimental evidence 的 annotation 再乘 0.7。随后执行每个 GO 内最大值归一化：

\[
F_{norm}(g,f)=\frac{F_{raw}(g,f)}{\max_uF_{raw}(u,f)}
\]

归一化 function IDF：

\[
IDF_F(g)
=
\frac{
\log\frac{N_F+1}{n_g^F+1}
}{
\log(N_F+1)
}
\]

最终：

\[
F'(g,f)=clip(F_{norm}(g,f)IDF_F(g),0,1)
\]

当前 function 分数只表示候选基因是否是该 GO gene set 的直接成员。PPI、共表达、
调控网络或语义邻近尚未实现，因此 `indirect_functional_support_score=0`。

---

## 12. 模块七：候选基因与 CPO 相容性

对 CPO \(c=(t,f)\)：

\[
A(g,c)
=
\beta_FF'(g,f)
+\beta_MM'(g,t)
+\beta_J\sqrt{F'(g,f)M'(g,t)}
\]

当前默认：

```yaml
function_weight: 0.65
marker_weight: 0.25
interaction_weight: 0.10
```

因此：

\[
A(g,c)
=
0.65F'+0.25M'+0.10\sqrt{F'M'}
\]

function 是主要机制证据，marker 是细胞定位先验，interaction 奖励同时具有两类证据
的基因。

---

## 13. 模块八：候选特异 CPO 路径

对候选基因 \(g\)，只使用其最佳 genotype \(m_g^*\) 实际解释到的疾病 HPO。

每条 HPO-MPO-CPO 路径权重：

\[
W_{path}(d,g,h,c)
=
w_h
\times q(h,p^*(h,m_g^*))
\times R(h,c)
\]

路径贡献：

\[
Contribution_{path}
=
W_{path}A(g,c)
\]

候选特异 CPO 画像：

\[
C_{d,g}(c)
=
\sum_hW_{path}(d,g,h,c)
\]

候选可比较覆盖率：

\[
Coverage_{d,g}
=
\frac{
\sum_h w_h\mathbf{1}[h\text{ has MPO match and CPO}]
}{
\sum_h w_h
}
\]

---

## 14. 模块九：CellPathScore

候选基因的细胞机制分数为路径相容性的加权平均：

\[
S_{cellpath}(d,g)
=
\frac{
\sum_{h,c}W_{path}(d,g,h,c)A(g,c)
}{
\sum_{h,c}W_{path}(d,g,h,c)
}
\]

同时输出加权 marker、function 和 interaction 分量，以及贡献最高的：

- HPO；
- matched MPO；
- CPO；
- cell；
- function；
- source genes。

`S_cellpath=0` 表示当前资源没有为该候选提供正向直接证据，不代表生物学负证据。

---

## 15. 模块十：最终排名融合

CellPathScore 在疾病内部只对正分候选转为 percentile：

\[
P_{d,g}=PercentileRank(S_{cellpath}(d,g))
\]

当前正式分数：

\[
S_{final}(d,g)
=
B_{d,g}
+\lambda Coverage_dP_{d,g}
\]

其中：

```yaml
lambda: 0.2
coverage_mode: disease
```

即：

\[
S_{final}=B+0.2\times Coverage_d\times P
\]

当前使用疾病级 coverage，而不是候选级 coverage。候选级 coverage 会输出，但不参与
正式配置的最终分数。

### 15.1 缺失证据策略

- 没有 CPO：CPO 加分为 0；
- 没有 marker：marker 分量为 0；
- 没有 function：function 分量为 0；
- 不因缺失 CPO 或 gene-set 证据删除候选；
- 当前融合只加分，不直接对缺失机制证据扣分。

### 15.2 Percentile 融合的含义

最终使用的是 CellPathScore 在当前疾病内的相对次序，而不是其绝对值。因此即使
绝对 CellPathScore 较小，只要它在该疾病候选中最高，仍可获得 `P=1` 的完整 CPO
加分。这是当前方法的重要特征，也是后续需要做绝对分校准的限制之一。

---

## 16. FOXP3 重排示例

在 IPEX (`OMIM:304790`) 中：

```text
gene                         FOXP3
base rank                    2
base score                   0.457039
base percentile              0.996454
disease CPO coverage         0.861793
CellPathScore                0.065588
CellPath percentile          1.0
lambda                       0.2
```

最终：

\[
S_{final}
=
0.996454+0.2\times0.861793\times1
=1.168812
\]

FOXP3 因此从第 2 升到第 1。

主要 practical evidence path：

```text
HP:0003212 Increased circulating IgE concentration
-> MP:0008626
-> CL:0000815 regulatory T cell
-> GO:0045066 regulatory T cell differentiation
-> FOXP3 marker/function membership
```

该路径的 HPO-CPO source genes 包含 `FOXP3;TGFB1`，因此必须结合 strict masking
解释，不能把 practical 提升直接当成未知基因泛化。

---

## 17. Strict gene masking

对每个测试疾病的已知基因集合 \(G_d^*\)，strict 模式执行两类遮蔽。

### 17.1 HPO-CPO 来源遮蔽

如果 HPO-CPO 边保存了 source-gene contributions：

\[
R_{masked}(h,c)
=
R(h,c)
\times
\frac{
\sum_{g\notin G_d^*}Contribution_g
}{
\sum_gContribution_g
}
\]

没有连续 contribution 时，退化为剩余 source-gene 数量比例。

### 17.2 直接 gene-set 遮蔽

当候选基因本身属于已知测试基因时：

```text
marker(g, cell) = 0
function(g, GO) = 0
```

当前没有网络间接支持，因此 strict 下该基因只能保留基础 HPO-MPO 排名，或者通过
仍然存在的非自身 HPO-CPO 来源间接影响整体疾病画像。

### 17.3 Strict 的解释

Strict 是泄漏压力测试，不是当前 practical 产品模式。当前实现对同一疾病的所有
evaluation-positive genes 同时遮蔽，比逐 disease-gene pair 遮蔽更保守。

---

## 18. 消融实验

当前已实现：

| Arm | 表型矩阵 | Marker | Function | Interaction |
|---|---|---:|---:|---:|
| A0_surface | 当前矩阵 | 0 | 0 | 0 |
| A1_marker | 当前矩阵 | 1 | 0 | 0 |
| A2_function | 当前矩阵 | 0 | 1 | 0 |
| A3_marker_function | 当前矩阵 | 0.25 | 0.65 | 0.10 |
| A3_strict_masked | 当前矩阵 | strict | strict | strict |

待新矩阵完成后增加：

- A4：新矩阵，无 CPO 重排；
- A5：新矩阵 + marker/function；
- A6：旧矩阵与新矩阵混合 + marker/function。

---

## 19. Gene-set 随机对照

正式运行执行 500 次 gene-set membership 随机化。

每次随机化：

1. 对每个 cell marker set 保持候选池内成员数量；
2. 对每个 function set 保持候选池内成员数量；
3. 保留该 set 的分数权重多重集合；
4. 从候选基因池中重新抽取成员；
5. 重新运行完整重排和评价。

经验置换 P 值：

\[
p_{perm}
=
\frac{1+\#(Metric_{null}\ge Metric_{observed})}
{N_{perm}+1}
\]

当前 null 保持每个 set 的大小，但不保持每个基因的全局 degree、表达水平或本体分支。
因此它是第一版结构对照，不是完整 degree-preserving null。

---

## 20. 总体评价

### 20.1 Gold standard

Gold genes 来自冻结 `gold_models.tsv` 中的 `human_gene_symbol`。评价不将 gold gene
作为输入排序特征；它只用于运行结束后的指标计算和 strict masking。

### 20.2 指标

对每个疾病记录最佳已知基因排名 \(r_d\)。

Recall@K：

\[
Recall@K=\frac{1}{|D|}\sum_d\mathbf{1}[r_d\le K]
\]

MRR：

\[
MRR=\frac{1}{|D|}\sum_d\frac{1}{r_d}
\]

NDCG@10 使用所有已知基因在前 10 名的位置计算。另报告：

- median rank；
- median rank percentile；
- mean/median rank change；
- win/tie/loss；
- 所有已知基因是否均被召回。

若疾病的已知基因不在候选池，Recall 和 MRR 记为失败，排名摘要使用
`candidate_count + 1`。

### 20.3 成对统计

定义：

\[
\Delta Rank_d=Rank_{A0,d}-Rank_{arm,d}
\]

正值表示改善。当前报告：

- paired Wilcoxon signed-rank test；
- 1,000 次 bootstrap mean-rank-change 95% CI；
- win/tie/loss 和 improved fraction。

### 20.4 分层

当前至少报告：

- 全部 50 个冻结疾病；
- 至少 2 个可比较 HPO 的 CPO-evaluable 子集；
- `CPO coverage >= 0.2` 子集。

CPO coverage 是冻结后的分析分层，不用于回头筛选疾病。

---

## 21. 当前 benchmark 结果

### 21.1 全部 50 个疾病

| Experiment | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR | NDCG@10 | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 surface | 0.30 | 0.72 | 0.82 | 0.96 | 0.4767 | 0.5520 | 3 |
| A1 marker | 0.28 | 0.68 | 0.84 | 0.96 | 0.4618 | 0.5465 | 3 |
| A2 function | 0.52 | 0.84 | 0.94 | 0.96 | 0.6565 | 0.7229 | 1 |
| A3 full | 0.48 | 0.86 | 0.94 | 0.96 | 0.6376 | 0.7096 | 2 |
| A3 strict | 0.04 | 0.46 | 0.72 | 0.96 | 0.2470 | 0.3461 | 6 |

### 21.2 A3 与 A0 的成对变化

```text
wins   23
ties   13
losses 14
mean rank improvement 2.22
Wilcoxon p = 0.0180
```

### 21.3 正式 500 次 null

```text
observed A3 MRR = 0.6376
null mean MRR   = 0.3715
permutation p  = 0.001996
```

### 21.4 必须同时报告的限制

Strict MRR 为 0.2470，并且相对 A0 为 0 wins、13 ties、37 losses。这说明 practical
提升没有通过当前直接基因遮蔽测试。因此：

> 当前结果证明知识库直接证据可以改善已知基因排序，但尚未证明对未知致病基因的
> 泛化能力。

---

## 22. 单细胞案例验证的新目标

单细胞案例现在优先验证：

> 系统排名靠前的候选致病基因，是否在系统预测的细胞中表现出表达、等位基因或
> 蛋白功能缺陷，并产生可解释的下游细胞功能变化。

单细胞验证分为四层：

| Level | Endpoint | 地位 |
|---|---|---|
| E0 | 致病基因在预测细胞中的表达/等位基因/蛋白缺陷 | 主终点 |
| E1 | 该基因缺陷是否强于参考细胞 | 主终点 |
| E2 | 遮蔽致病基因后的 strict downstream module | 次级终点 |
| E3 | practical cell/function mechanism | 解释性终点 |

不同变异类型不能统一要求 mRNA 消失：

- 无义、移码、NMD、删除：优先检验转录下降；
- 剪接变异：优先检验异常剪接和转录本变化；
- missense：需要蛋白功能或下游 module，mRNA 可正常存在；
- gain-of-function 或 dominant-negative：重点检验异常状态而非表达缺失。

---

## 23. 单细胞案例冻结与 preflight

强制顺序：

```text
metadata-only audit
-> freeze disease, gene, cell and function endpoints
-> hash case manifest
-> load expression
-> donor pseudobulk
-> analysis preflight
-> calculate endpoints
-> retain supported or failed result without replacement
```

当前 `freeze-single-cell-case` 只读取：

- 冻结疾病 HPO；
- HPO-CPO；
- gold genes；
- function gene sets；
- CL label map；
- 不含表达值的 cell metadata。

不读取 expression matrix。

主要 preflight gate：

- patient/control 同组织和实验类型；
- 每组至少 4 个独立 donor；
- 每个 donor 的预测细胞至少 20 cells；
- 预测 CL 与数据标签要求 exact match；
- 遮蔽致病基因后 function set 至少 5 genes，推荐至少 15；
- function genes 在表达矩阵中覆盖至少 80%；
- 数据未用于 marker/function gene-set 构建。

当前实现中，function coverage gate 检查的是 gene 是否存在于矩阵 feature axis，尚未
把“实际非零表达 donor 比例”作为硬 gate。该项已经在协议中要求，但仍是需要补齐的
代码差距。

---

## 24. 单细胞数据处理

### 24.1 分析单位

统计单位固定为：

```text
donor x cell type
```

原始 cell counts 按 donor 和 cell label 求和。禁止把每个 cell 当作独立生物学重复。

### 24.2 标准化

每个 pseudobulk profile 转换为：

\[
log2CPM_{d,c,g}
=
\log_2\left(
1+\frac{count_{d,c,g}}{\sum_u count_{d,c,u}}10^6
\right)
\]

### 24.3 E0：预测细胞中的候选基因效应

\[
GeneEffect_c
=
mean(X_{patient,c,g})-mean(X_{control,c,g})
\]

每个 cohort 使用 Welch test，并用 inverse-variance fixed-effect meta-analysis 合并。

### 24.4 E1：预测细胞特异的基因缺陷

先在每个 donor 内定义：

\[
CellContrast_d
=
X_{d,predicted,g}
-mean_{c\in ReferenceCells}X_{d,c,g}
\]

再计算 difference-in-differences：

\[
DiD
=
mean(CellContrast_{patient})
-mean(CellContrast_{control})
\]

负值表示候选基因在预测细胞中的下降强于参考细胞。

### 24.5 E2/E3：function module

对同一 dataset 和 cell type 内的每个 function gene 做 donor 间 z-score：

\[
Z_{d,g}=\frac{X_{d,g}-\mu_g}{\sigma_g}
\]

donor module score：

\[
ModuleScore_d=\frac{1}{|G_f|}\sum_{g\in G_f}Z_{d,g}
\]

候选疾病基因从下游 function gene set 中删除，避免直接把待验证基因计入 module。

### 24.6 表达匹配随机集

每个目标基因按照该 cell type donor-pseudobulk 的平均表达划入 10 个分箱，从同分箱
抽取背景基因，保持 gene-set 大小，默认执行 1,000 次。

### 24.7 跨队列 meta

对 cohort effect \(\hat\theta_i\) 和标准误 \(SE_i\)：

\[
w_i=\frac{1}{SE_i^2}
\]

\[
\hat\theta_{meta}=\frac{\sum_iw_i\hat\theta_i}{\sum_iw_i}
\]

\[
SE_{meta}=\frac{1}{\sqrt{\sum_iw_i}}
\]

同时报告 95% CI、Cochran Q、heterogeneity P 和 I-squared。

---

## 25. 当前 IPEX 案例

### 25.1 预测结果

疾病：IPEX，`OMIM:304790`  
已知致病基因：`FOXP3`

```text
FOXP3 base rank  = 2
FOXP3 final rank = 1
predicted cell   = regulatory T cell
```

### 25.2 单细胞数据

GSE167976：55,656 个 CD4 T cells，原始研究包括 11 IPEX 与 11 healthy donors。

原始两个 cohort 中有重复 IPEX donors。为避免伪重复：

- cohort 1：6 IPEX vs 4 healthy；
- cohort 2 donor-disjoint：排除重复 B4、B5、P4，保留 5 IPEX vs 7 healthy。

所有 donor 都有 Treg、resting Tconv 和 activated Tconv。最小 donor-level Treg profile
仍包含 39 cells。

### 25.3 E0/E1 结果

| Endpoint | Meta effect | 95% CI | FDR | I-squared | Status |
|---|---:|---:|---:|---:|---|
| FOXP3 in Treg | -1.148 | [-1.992, -0.304] | 0.0077 | 0 | supported |
| Treg-vs-Tconv FOXP3 DiD | -1.184 | [-1.775, -0.593] | 0.00017 | 0.027 | supported |

FOXP3 在所有 donor-level Treg profiles 中均可检测，因此结果不是单细胞 dropout
导致的假下降。

患者包含 missense、剪接、移码和起始位点异常等不同变异，所以结论是 FOXP3
表达/功能缺陷，而不是所有患者发生 mRNA 完全缺失。

### 25.4 下游结果

| Endpoint | Result |
|---|---|
| Strict T-cell migration module | supported |
| Practical Treg differentiation module | not_supported |
| Treg marker identity | 两个 cohort 均通过 |

因此当前案例可以展示：

> 系统把 FOXP3 从第 2 重排到第 1，并定位到 Treg；患者单细胞进一步显示 FOXP3
> 在 Treg 中发生比 Tconv 更明显的表达缺陷，同时存在部分下游 T-cell function 异常。

### 25.5 确认性边界

E0/E1 是在项目目标澄清后加入的开发性 endpoint，不属于最初 blind freeze 的主终点。
当前结果可作为实际案例展示，但论文级 confirmatory claim 仍需要在新的 case manifest
中预注册 E0/E1，并在未查看的独立患者数据中复验。

---

## 26. HD 压力测试

Huntington disease 保留为失败压力测试，不作为主案例：

- 使用全 MGI 候选池时，HTT 基础第 17，重排后第 197；
- HPO-CPO top cell 偏向 large basket/SST interneuron；
- 数据只能把 fine CL 映射到距离 5-7 的 generic neuron；
- top function sets 只有 3-17 genes；
- 5 个预声明功能均未通过细胞特异验证。

保留失败案例用于说明系统不会通过事后换疾病或换 GO term 制造阳性结果。

---

## 27. 输出目录

正式基准：

```text
outputs/real_frozen/
  run_manifest.json
  resolved_config.yaml
  qc/
  baseline/
    disease_model_scores.tsv
    base_gene_rankings.tsv
    hpo_mpo_evidence.tsv
  cpo/
    disease_cpo_profiles.tsv
    candidate_cpo_profiles.tsv
  gene_sets/
  reranking/
    candidate_cellpath_scores.tsv
    evidence_paths.tsv
    final_gene_rankings.tsv
    rank_changes.tsv
  benchmark/
    benchmark_metrics.tsv
    disease_level_metrics.tsv
    ablation_metrics.tsv
    stratified_metrics.tsv
    statistical_tests.tsv
  leakage_control/
  null_models/
  figures/
  report/
```

IPEX 单细胞案例：

```text
outputs/ipex_single_cell/
  analysis_preflight.tsv
  causal_gene_expression.tsv
  causal_gene_summary.tsv
  cell_identity_control.tsv
  function_enrichment.tsv
  cell_specificity.tsv
  cell_composition.tsv
  pseudobulk_de.tsv
  case_summary.tsv
  case_report.md
  run_manifest.json
  figures/
```

---

## 28. 命令行

### 28.1 合成端到端测试

```bash
PYTHONPATH=src python3 -B -m cellphenorank run \
  --config configs/example.yaml \
  --output-dir outputs/example
```

### 28.2 正式 benchmark

```bash
PYTHONPATH=src python3 -B -m cellphenorank validate-inputs \
  --config configs/real_frozen.yaml

PYTHONPATH=src python3 -B -m cellphenorank run \
  --config configs/real_frozen.yaml \
  --output-dir outputs/real_frozen
```

### 28.3 冻结单细胞案例

```bash
PYTHONPATH=src python3 -B -m cellphenorank freeze-single-cell-case \
  --config configs/ipex_case_freeze.yaml \
  --output-dir data/cases/ipex_case_frozen
```

### 28.4 构建 IPEX donor pseudobulk

```bash
python3 -B scripts/prepare_gse167976.py
```

### 28.5 运行单细胞案例

```bash
PYTHONPATH=src python3 -B -m cellphenorank validate-single-cell \
  --config configs/ipex_single_cell.yaml \
  --output-dir outputs/ipex_single_cell
```

### 28.6 测试

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

当前有 6 个回归测试，覆盖矩阵缺失 pair、BMA、无 CPO 退化、细胞本体映射、
gene-zscore module、表达匹配 null 和合成端到端排名变化。

---

## 29. 新 MPO-HPO 矩阵接入方式

新矩阵完成后，最低接入要求为：

```text
hpo_id
mpo_id
score in [0, 1]
relation/type
source
version
evidence
```

修改配置：

```yaml
inputs:
  similarity: path/to/new_hpo_mpo_similarity.tsv

similarity:
  score_columns: [score]
```

下游 BMA、模型聚合、CPO 重排、评价和单细胞流程不需要修改。

接入后运行：

1. A4：新矩阵基础排名；
2. A5：新矩阵 + CPO 重排；
3. A6：旧矩阵与新矩阵混合；
4. 与 A0/A3 做相同冻结疾病的成对比较；
5. 单独报告矩阵贡献与重排贡献。

混合 provider 计划：

\[
q_{hybrid}(h,p)
=
\eta q_{new}(h,p)+(1-\eta)q_{current}(h,p)
\]

新矩阵缺失 pair 时回退当前矩阵，但必须在 evidence 中标记来源。

---

## 30. 已知限制与下一步

### P0：进入正式方法比较前必须处理

1. 构建全基因组或明确定义的独立候选模型池；
2. 将 rank-fusion 权重在开发集选择，在独立测试集固定；
3. 把单细胞 function coverage 从 feature presence 改为实际 donor-level detection；
4. 为 E0/E1 建立新的 blind case manifest 和独立复现数据；
5. 分开报告 practical 与 leakage-controlled strict 结论。

### P1：提高重排泛化能力

1. 增加不依赖疾病已知基因的 HPO-CPO 文献证据；
2. 实现 PPI、共表达或调控网络的 indirect function support；
3. 比较疾病级与候选级 coverage 融合；
4. 校准绝对 CellPathScore，降低纯 percentile 极值效应；
5. 实现保持 gene degree 的二部图随机化。

### P2：新矩阵完成后

1. 实现 CellPhenotypeSimilarityProvider；
2. 实现 HybridSimilarityProvider；
3. 完成 A4-A6；
4. 分析新矩阵在哪些疾病和表型系统中获益；
5. 保持 benchmark 疾病、HPO 和 gold models 不变。

---

## 31. 文档优先级

发生冲突时按以下顺序解释：

1. 当前运行代码与 `resolved_config.yaml`；
2. 本文档 `CellPhenoRank_project_description_v2.md`；
3. `SINGLE_CELL_VALIDATION_PROTOCOL.md`；
4. 原始设计稿 `CellPhenoRank_project_plan.md`。

原始设计稿保留用于追踪设计演变，不再作为当前实现的唯一说明。
