# SapBERT HPO-MP 开放候选实验

## 实验问题

将第一步 HPO-MP 相似度替换为 SapBERT cosine similarity 后，能否在不根据
gold gene 缩小候选集的情况下找回致病基因，以及 CPO 是否能在同一候选宇宙中进一步
改善排名。

## 固定评价边界

候选全集在载入 CPO 前冻结：

    diseases                              50
    phenotype-bearing single-gene models  53,199
    candidate genes                       13,194
    CPO reranking                         embedding baseline Top-200 only

候选规则是当前 MGI 快照中每一个至少有一条 abnormal MP annotation 的单基因
genotype。候选选择不读取 disease-gene gold、CPO、gene set 或任何排名结果。
gold 未获得正 embedding score 时按未召回处理，reciprocal rank 为 0；gold 不在
Top-200 时不加入 CPO shortlist，尾部保持基础名次。

权威清单：

- data/benchmark/open_mgi_50_v1/open_candidate_manifest.json
- data/embeddings/*.npz.manifest.json

## Embedding 定义

模型为 cambridgeltl/SapBERT-from-PubMedBERT-fulltext，使用官方推荐的
pre-pooler CLS 向量和 cosine similarity。权重 SHA-256 为
a4696930afef9aab296196d3d2142216c44cba24f21b4f285ceca7af21025614。

预声明三个敏感性版本：

1. label-K64：首选名向量，每个 HPO 保留最近 64 个 MP。
2. label-K256：首选名向量，每个 HPO 保留最近 256 个 MP。
3. synonym-K256：首选名和 EXACT 同义词分别编码后取质心，保留最近 256 个 MP。

HPO 覆盖 591/591；MP 覆盖 11,539/11,625。缺失的 86 个 MP 是 MGI 快照与
2026-03-24 MP ontology 之间的版本差，完整 ID 已记录在 embedding manifest。

## 结果

| Version | Arm | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR | Median rank |
|---|---|---:|---:|---:|---:|---:|---:|
| label-K64 | Surface | 0.04 | 0.20 | 0.34 | 0.68 | 0.1255 | 28.0 |
| label-K64 | Strict CPO | 0.10 | 0.22 | 0.34 | 0.68 | 0.1764 | 23.0 |
| label-K64 | Extended CPO | 0.10 | 0.36 | 0.46 | 0.74 | 0.2196 | 16.0 |
| label-K64 | Bridge-LOO | 0.08 | 0.20 | 0.32 | 0.68 | 0.1571 | 27.0 |
| label-K256 | Surface | 0.02 | 0.14 | 0.34 | 0.62 | 0.1029 | 31.5 |
| label-K256 | Strict CPO | 0.10 | 0.18 | 0.32 | 0.60 | 0.1664 | 30.5 |
| label-K256 | Extended CPO | 0.14 | 0.22 | 0.34 | 0.68 | 0.2102 | 18.5 |
| label-K256 | Bridge-LOO | 0.08 | 0.16 | 0.30 | 0.60 | 0.1474 | 30.5 |
| synonym-K256 | Surface | 0.04 | 0.18 | 0.26 | 0.64 | 0.1212 | 32.5 |
| synonym-K256 | Strict CPO | 0.06 | 0.22 | 0.28 | 0.62 | 0.1403 | 32.0 |
| synonym-K256 | Extended CPO | 0.10 | 0.18 | 0.34 | 0.64 | 0.1719 | 22.5 |
| synonym-K256 | Bridge-LOO | 0.04 | 0.18 | 0.26 | 0.62 | 0.1226 | 36.5 |

Extended CPO 相对各自 embedding surface：

| Version | Mean rank change | Win/Tie/Loss | Wilcoxon p |
|---|---:|---:|---:|
| label-K64 | +10.42 | 33/10/7 | 2.95e-5 |
| label-K256 | +8.88 | 30/9/11 | 5.34e-4 |
| synonym-K256 | +5.18 | 29/12/9 | 0.00230 |

## 解释

三版中 Extended CPO 的改善方向一致，但 Strict 和 Bridge-LOO 没有稳定总体改善。
同义词质心没有一致优于首选名，K256 也没有一致优于 K64。这说明 embedding
邻域定义是敏感参数，不能在本队列挑选最佳版本后将其作为确认性结论。

当前实验使用完整 MGI 候选快照，而不是 IMPC-only release；它符合开放候选排序
原则，但不是 IMPC DR5/DR20.1 的逐版本复现。SapBERT 原本为 biomedical entity
alignment 训练，cosine score 不是经过校准的 HPO-MP 生物学概率。

## 复现

    PYTHONPATH=src python3 -B scripts/prepare_open_candidate_manifest.py ...
    PYTHONPATH=src python3 -B scripts/build_ontology_embeddings.py ...
    PYTHONPATH=src python3 -B -m cellphenorank run-embedding-open \
      --config configs/embedding_open_sapbert_label_k64.yaml \
      --output-dir outputs/embedding_open_sapbert_label_k64

SapBERT 方法：<https://aclanthology.org/2021.naacl-main.334/>。

