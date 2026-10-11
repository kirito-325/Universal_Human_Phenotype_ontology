# Exomiser 2602 当前 PhenoDigm 矩阵

## 1. 数据取得情况

Exomiser 2602 phenotype 数据库已经在远程服务器下载、解压并校验：

```text
remote path  /path/to/remote/exomiser2602/2602_phenotype.mv.db
bytes        22,215,954,432
CRC32        24f6767a
SHA-256      8d758ed0cb4ddd73d6144ec3612dc356e3590b0477f97be53a86916d5f4e16d2
```

下载时只读取 2602 ZIP 内 H2 数据库条目的 11.10 GiB DEFLATE 区间，没有下载
其余无关条目。远程数据库保留，压缩中间文件已在 CRC/SHA 验证后删除。

数据版本：

| 资源 | 版本 |
|---|---|
| Exomiser phenotype | 2602 |
| HPO | 2026-01-08 |
| HPO annotations | 2026-01-08 |
| MP | 2026-01-27 |
| IMPC | DR23.0 |
| H2 | 2.2.224 |

版本来源为 [Exomiser 2602 release notes](https://github.com/exomiser/Exomiser/discussions/640)。

## 2. 完整矩阵

H2 数据库中的 `EXOMISER.HP_MP_MAPPINGS` 已检查：

```text
rows  95,051,352
```

字段为：

```text
MAPPING_ID
HP_ID
HP_TERM
MP_ID
MP_TERM
SIMJ
IC
SCORE
LCS_ID
LCS_TERM
```

其中：

\[
SCORE(h,m)=\sqrt{SIMJ(h,m)\times IC(h,m)}
\]

## 3. 冻结基准子矩阵

远程使用 `HP_ID` 索引导出了冻结 benchmark v4 的 591 个唯一 HPO 对应的全部
HP-MP 行，并直接压缩：

```text
data/external/exomiser_2602/current_matrix/hp_mp_mappings_2602.tsv.gz
rows     3,143,719
bytes    99,094,046
SHA-256  6af7fdb6f329a3c84e2a07b2edf09a5c2c7b2b1c5992b5407890b269b03edace
```

覆盖情况：

| 项目 | 输入 | 2602 覆盖 | 覆盖率 |
|---|---:|---:|---:|
| 唯一疾病 HPO | 591 | 589 | 99.66% |
| 唯一候选 MPO | 2,699 | 2,685 | 99.48% |

缺失 HPO 为 `HP:0001609`、`HP:0002829`。缺失的 14 个 MPO 及全部文件哈希见
`data/external/exomiser_2602/current_matrix/export_manifest.json`（本地生成，未上传）。

## 4. 当前 Exomiser 评分

当前 Exomiser 与 2013 论文最初公式存在一个重要区别。对疾病中可映射的 HPO 集合
\(H\)，仅保留与这些 HPO 至少有一条矩阵关系的模型 MPO 集合 \(M'\)：

\[
Avg_{current}(d,m)=
\frac{
\sum_{h\in H}\max_{p\in M'}s(h,p)+
\sum_{p\in M'}\max_{h\in H}s(h,p)
}{|H|+|M'|}
\]

理论平均分为每个 HPO 在整个 MP 中的最佳匹配均值：

\[
Avg_{ideal}(d)=
\frac{1}{|H|}\sum_{h\in H}\max_p s(h,p)
\]

最终分数为标准化平均分与标准化最大单对分的均值，并截断在 100：

\[
S_{current}=100\times\min\left(
1,
\frac{Avg_{current}/Avg_{ideal}+Max_{current}/Max_{ideal}}{2}
\right)
\]

这一实现依据当前 Exomiser `CrossSpeciesPhenotypeMatcher`、`QueryPhenotypeMatch` 和
`PhenodigmModelScorer` 源码，而不是根据接口结果反推。

## 5. 冻结 benchmark 结果

相同的 50 个疾病、49 个候选基因、1,068 个单基因 genotype 和 gold standard：

| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | 中位名次 |
|---|---:|---:|---:|---:|---:|---:|
| 当前项目 A0 | 0.36 | 0.68 | 0.92 | 0.5279 | 0.6182 | 2.5 |
| Exomiser 2602 PhenoDigm | 0.42 | 0.78 | 0.92 | 0.5548 | 0.6382 | 2.5 |
| IMPC 在线聚合分数 | 0.30 | 0.66 | 0.90 | 0.4821 | 0.5756 | 2.5 |

2602 相对当前 A0：

```text
win / tie / loss  22 / 11 / 17
mean rank change   0.26
Wilcoxon p         0.8716
```

2602 在 Top 1、Top 5、MRR 和 NDCG@10 上数值更高，Top 10 持平；但疾病级配对
差异不显著，不能声称稳定总体优于当前 A0。没有任何模型分数超过 100。

## 6. 解释边界

- 这是相同冻结 HPO 和小鼠 genotype 输入下，对当前 Exomiser 表型评分的受控复现。
- 在线 IMPC 分数使用其自身疾病 HPO profile，因此与本地 2602 结果不同是预期现象。
- 2602 HPO 版本早于 benchmark 冻结时间，仍有 2 个 HPO 和 14 个 MPO 缺失，但覆盖
  已显著高于历史矩阵。
- 当前结果只替换表型初排；尚未用该评分重新构建 CPO 重排所需的 leave-one-HPO-out
  稳定约束。

## 7. 复现文件

```text
scripts/fetch_exomiser_2602_phenotype_db.py
scripts/Exomiser2602HpMpExporter.java
configs/phenodigm_2602.yaml
src/cellphenorank/phenodigm_legacy.py
outputs/phenodigm_2602/
```
