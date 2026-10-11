# PhenoDigm 历史 HPO-MP 矩阵与冻结基准结果

## 1. 已取得的数据

已从 PhenoDigm 的 Google Code 官方历史源码归档取得：

```text
archive entry
phenodigm/trunk/files/mappings/hp_mp_mappings.bz2

local file
data/external/phenodigm_legacy/hp_mp_mappings.bz2
```

原始文件：

```text
bytes    701,426,412
CRC32    02e9598a
SHA-256  b69f76e06aaf5ba7850dad7987ded7969424289f3c7b4d47acb26b030db22ec5
rows     52,022,155
```

来源清单为
`data/external/phenodigm_legacy/source_manifest.json`（本地文件，未上传）。
下载没有获取 2.3 GB 的完整源码归档，而是根据 ZIP central directory 只读取该条目的
DEFLATE 字节范围，并验证 ZIP 声明的文件长度与 CRC32。

## 2. 原始字段

历史文件没有表头，每行 6 列：

```text
HP ID
HP name
MP ID
MP name
simJ
IC
```

单对表型分数按原始 PhenoDigm 定义计算：

\[
s(h,m)=\sqrt{simJ(h,m)\times IC(h,m)}
\]

`simJ` 是本体推断属性的 Jaccard 相似度，`IC` 是最小公共上位概念的信息量。

## 3. 冻结基准子矩阵

为了保留原始文件并减少重复读取，从 52,022,155 行中提取了冻结 benchmark 所有
HPO 对应的完整 MP 配对，而不是只提取候选模型 MP：

```text
data/external/phenodigm_legacy/benchmark_v4/hp_mp_mappings.tsv.gz
```

子矩阵：

```text
rows     4,500,868
bytes    122,165,031
SHA-256  fa70b14d9b870888970befd6a7e94a05c7c9fb433a9d31f85b84d162c2144205
```

保留全部相关 MP 是为了正确计算原算法的理论最优模型归一化分母。子矩阵新增
`score=sqrt(simJ*IC)`，没有修改原始 `simJ` 和 `IC`。

覆盖情况：

| 项目 | 输入 | 历史矩阵覆盖 | 覆盖率 |
|---|---:|---:|---:|
| 唯一疾病 HPO | 591 | 517 | 87.48% |
| 唯一候选 MPO | 2,699 | 2,309 | 85.55% |

74 个 HPO 和 390 个候选 MPO 在历史矩阵中不存在，主要是矩阵生成后新增的本体术语。
逐 ID 审计在
`data/external/phenodigm_legacy/benchmark_v4/subset_manifest.json`（本地文件）
和
`data/external/phenodigm_legacy/benchmark_v4/candidate_mpo_presence.tsv`（本地文件，未上传）。

## 4. 本地 PhenoDigm 计算

对疾病 HPO 集合 \(H\) 和 genotype MPO 集合 \(M\)，使用论文中的双向最佳匹配：

\[
Avg(d,m)=
\frac{
\sum_{h\in H}\max_{p\in M}s(h,p)+
\sum_{p\in M}\max_{h\in H}s(h,p)
}{|H|+|M|}
\]

`Avg` 和全局最大单对分数分别除以该疾病的理论最优模型值，最终取二者平均。
每个基因取其所有单基因 genotype 的最高分。疾病 HPO、genotype、MPO 和 gold
全部来自已冻结 benchmark。

运行命令：

```bash
PYTHONPATH=src python3 -B -m cellphenorank run-phenodigm-legacy \
  --config configs/phenodigm_legacy.yaml \
  --output-dir outputs/phenodigm_legacy
```

## 5. 当前结果

| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | 中位名次 |
|---|---:|---:|---:|---:|---:|---:|
| 当前 A0 | 0.36 | 0.68 | 0.92 | 0.5279 | 0.6182 | 2.5 |
| 历史矩阵 PhenoDigm | 0.42 | 0.72 | 0.82 | 0.5538 | 0.6105 | 2.0 |
| IMPC 在线官方分数 | 0.30 | 0.66 | 0.90 | 0.4821 | 0.5756 | 2.5 |

历史矩阵 PhenoDigm 相对当前 A0：

```text
win / tie / loss  17 / 13 / 20
mean rank change  -0.76
Wilcoxon p         0.4353
```

因此历史方法改善了最前端命中，但 Top 10 覆盖下降，且总体配对差异不显著。

## 6. 解释边界

- 这是原始 OWLSim/PhenoDigm 矩阵，不是 2026 年当前 IMPC 本体快照。
- 按原方法排除历史矩阵中不存在的 HPO 和 MPO，不把缺失项当作零相似度惩罚。
- 29 个模型的归一化总分超过 100，最高 109.36；这是旧矩阵与新病例本体组合后，
  逐 HPO 构造的理论最优模型不再是全局最优造成的，已保留而未截断。
- 本地实现遵循论文的双向公式，没有复现历史 Perl 文件反向循环中的明显数组索引笔误。
- 该结果适合做历史基线和方法敏感性分析，不应代替使用同版本本体重新生成的当前矩阵。

## 7. 复现文件

```text
scripts/fetch_phenodigm_legacy_matrix.py
scripts/extract_phenodigm_legacy_subset.py
src/cellphenorank/phenodigm_legacy.py
configs/phenodigm_legacy.yaml
outputs/phenodigm_legacy/
```

方法来源：PhenoDigm 原始论文
<https://doi.org/10.1093/database/bat025>；历史项目归档
<https://code.google.com/archive/p/phenodigm/>。
