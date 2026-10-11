# IMPC DR20.1/2024 测评复现

## 结论

可以使用作者发布的测试集。官方数据已从 Zenodo
10.5281/zenodo.15489406 下载，并通过发布方 MD5 校验。数据许可为 CC BY 4.0。

该数据不是传统 train/test split，而是固定版本的 gene-level disease-model
recapitulation benchmark。评价单元是具有一对一人鼠同源关系、IMPC abnormal MP
和关联疾病 HPO 的 gene knockout。

## 论文主结果复现

从 gene_summary_dr20.1.txt 按论文条件筛选：

    IMPC_pipeline=yes
    IMPC_phenotypes=yes
    HPO_phenotypes=yes
    disorder_id is present

得到：

| Metric | Paper | Reproduced |
|---|---:|---:|
| Eligible gene knockouts | 2,400 | 2,400 |
| PhenoDigm matched gene knockouts | 1,311 | 1,311 |
| Match rate | 55% after rounding | 54.625% |

论文中的 8,707 是所有进入 IMPC pipeline 的 mouse genes。Zenodo gene summary
限定在一对一人鼠同源集合，其中 IMPC_pipeline=yes 为 8,286，不能把这两个分母
当成同一统计量。

## 官方文件角色

| File | Role | Limitation |
|---|---|---|
| gene_summary_dr20.1.txt | 完整 gene-level test set 和官方 outcome | 含答案，只能在最终评价阶段读取 |
| phenodigm_matches_dr20.1.txt | 已知 disease-model 正匹配与匹配 HPO/MP | 只有正例，不能单独构造完整测试集 |
| phenodigm_other_dr20.1.txt.gz | 非已知疾病基因的开放候选 | 仅保留 score > 40，不能当作完整负例或完整排名 |

正匹配文件包含 3,085 个 model-disease records、1,311 个 gene、2,104 个
disease 和 2,566 个 disease-gene pairs。开放候选文件包含 1,787,288 条记录、
4,217 个 gene 和 7,411 个 disease。

## 防止答案泄漏

派生 benchmark 将数据分为：

- evaluation_inputs/eligible_gene_cohort.tsv：2,400 个输入单元，不含官方 match 和 score。
- withheld_truth/gene_level_truth.tsv：官方 PhenoDigm_match 和 max_score。

候选生成、embedding、阈值选择和 CPO 重排禁止读取 withheld_truth。

冻结入口：

    data/benchmark/impc_dr20_1_2024_reproduction_v1/freeze_manifest.json

## 原始输入重算覆盖

DR20.1 genotype-phenotype-assertions-ALL.csv.gz 对 2,400 个 gene 全覆盖。
论文访问日前最近的公开 HPO release 为 2023-10-09，它覆盖 2,384/2,400 个
gene test units，原始输入重算覆盖率为 99.33%。

缺失的 16 个单元主要是 2023 年后期新增的 OMIM 620xxx 疾病。它们全部保留在
官方 2,400 单元测试集中，并在 raw_replay_coverage.tsv 中标为
missing_human_hpo。后续报告必须同时给出：

1. 官方输出复现全集 2,400。
2. 公共原始输入可重算层 2,384。
3. 16 个 profile coverage failures。

目前已经复现的是官方冻结输出和测试集计数。要从零重新得到同样的 PhenoDigm
分数，还需要恢复作者 2023-11-10 当天的 HPO/Orphadata 快照以及当时的
PhenoDigm HPO-MP 跨本体相似度资源。

## SapBERT 测试

使用 DR20.1 全部 2,400 个官方 gene-level 单元测试两种 SapBERT 表示和三种
profile 聚合。16 个缺少公共历史 HPO profile 的单元仅在全体层按 score=0
保留；2,384 个 public raw replay-ready 单元单独报告。

全体 2,400 单元结果：

| Representation | Formula | AUROC (95% CI) | AUPRC (95% CI) | Positive-score Spearman |
|---|---|---:|---:|---:|
| label | symmetric BMA | 0.8143 [0.7969, 0.8304] | 0.8538 [0.8395, 0.8678] | 0.6415 |
| label | PhenoDigm 50/50 | 0.8646 [0.8500, 0.8780] | 0.8939 [0.8818, 0.9056] | 0.6494 |
| label | project 80/20 | 0.8466 [0.8312, 0.8605] | 0.8814 [0.8690, 0.8940] | 0.6705 |
| synonym centroid | symmetric BMA | 0.8153 [0.7973, 0.8307] | 0.8542 [0.8383, 0.8678] | 0.6342 |
| synonym centroid | PhenoDigm 50/50 | 0.8640 [0.8500, 0.8775] | 0.8946 [0.8827, 0.9061] | 0.6482 |
| synonym centroid | project 80/20 | 0.8462 [0.8310, 0.8601] | 0.8807 [0.8676, 0.8929] | 0.6649 |

AUROC/AUPRC 的标签是官方 PhenoDigm match/non-match，因此这些数值衡量 embedding
对 PhenoDigm 的复现一致性，不是独立生物学真值。该 endpoint 已知目标 gene，
评价其 mouse KO 是否复现疾病，所以没有加入 CPO candidate-gene reranking。

测试使用 4,399 个有公开 HPO profile 的疾病、11,442 个
allele/zygosity/life-stage 模型和 40,340 个 MP annotations。非 MP 的 1,618 条
MPATH assertions 被显式排除。

结果：

    outputs/impc_dr20_1_embedding_test/

运行：

    PYTHONPATH=src python3 -B -m cellphenorank run-impc-dr20-embedding \
      --config configs/impc_dr20_1_embedding_test.yaml \
      --output-dir outputs/impc_dr20_1_embedding_test

## 复现命令

    PYTHONPATH=src python3 -B scripts/prepare_impc_dr20_1_reproduction.py \
      --source-dir data/external/impc_dr20_1_2024 \
      --output-dir data/benchmark/impc_dr20_1_2024_reproduction_v1

    PYTHONPATH=src python3 -B scripts/audit_impc_dr20_1_raw_replay.py \
      --eligible-cohort data/benchmark/impc_dr20_1_2024_reproduction_v1/evaluation_inputs/eligible_gene_cohort.tsv \
      --assertions data/external/impc_dr20_1_2024/raw/genotype-phenotype-assertions-ALL.csv.gz \
      --hpoa data/external/impc_dr20_1_2024/raw/hpo_v2023-10-09/phenotype.hpoa \
      --output-dir data/benchmark/impc_dr20_1_2024_reproduction_v1/raw_replay

## 来源

- Paper: <https://doi.org/10.1242/dmm.050604>
- Frozen portal dataset: <https://doi.org/10.5281/zenodo.15489406>
- IMPC DR20.1 exports: <https://ftp.ebi.ac.uk/pub/databases/impc/all-data-releases/release-20.1/results/>
- HPO 2023-10-09: <https://github.com/obophenotype/human-phenotype-ontology/releases/tag/v2023-10-09>
