# Expanded cell marker scores

本目录按 expanded catalog 的标准细胞 ID 汇总文献中的 marker 基因。

## 计分规则

计分单位是：`文章 × 表类型 × 标准细胞 ID × 基因`。

- 同一篇文章的 `marker_statistics_cleaned.csv` 中，只要某个人类标准基因作为该标准细胞的 marker 出现一次或多次，贡献 1 分。
- 同一篇文章的 `marker_signatures_cleaned.tsv` 中，只要某个人类标准基因作为该标准细胞的 marker 出现一次或多次，贡献 1 分。
- 因此，同一篇文章对同一个“标准细胞 ID × 基因”最多贡献 2 分。
- 原始细胞名称通过映射表归并到 `expanded/expanded_cell_catalog.tsv` 的标准 ID，再做文章内去重。SapBERT Top-1 仅适合候选召回；正式结果应使用带 `decision` 字段的语义判定表。
- statistics 先经过严格统计阈值，再读取 `marker_statistics_cleaned.csv`，不读取原始 `marker_statistics.csv`。
- 基因统一为 HGNC 批准的人类符号；明确的小鼠基因通过 MGI 唯一人类目标同源类转换。原始基因、物种、转换方法和参考编号保留在清洗表及网页证据中。

## 输出

- `expanded_marker_scores.tsv`：全部标准细胞-marker 组合及得分。
- `expanded_marker_scores_top100.tsv`：每个标准细胞得分最高的 100 个 marker，便于检查。
- `expanded_marker_scores_review.html`：标准细胞、原始名称映射和 Top100 marker 的离线可视化检查页。
- `expanded_cell_evidence_opportunities.tsv`：每个标准细胞实际拥有的文章-表类型证据机会数。
- `significant_marker_scores.tsv`：仅保留通过重复证据、富集检验和 FDR 阈值的 marker。
- `significant_marker_cell_summary.tsv`：每个标准细胞的检验数量及显著 marker 数量。
- `significant_marker_scores_review.html`：仅展示显著 marker 的离线可视化检查页。
- `significant_marker_summary.json`：显著性方法、阈值和结果规模。
- `unmapped_cell_names.tsv`：未进入过滤后细胞名称映射、因而没有参与计分的名称。
- `summary.json`：输入规模、去重数量、得分总量和错误信息。
- `evidence_explorer/marker_evidence.sqlite3`：标准细胞树、已接受名称、全部 Marker 和论文级来源的查询索引。
- `evidence_explorer/index.html`：通过本地查询服务使用的 Marker 证据浏览器。

`marker_score = statistics_score + signatures_score`。`marker_rank` 在每个标准细胞内按总分、两类子分和基因名称稳定排序。

## 显著性筛选

每个“文章 × 表类型 × 标准细胞 ID”是一次独立证据机会。对每个细胞-基因组合做单侧超几何富集检验，背景是其他 expanded 标准细胞的全部证据机会。多重检验使用每个细胞内的 Benjamini-Hochberg FDR，并将全局基因全集计入检验数量。

默认仅保留同时满足以下条件的基因：

- `marker_score >= 3`
- `q_value <= 0.05`
- `odds_ratio > 1`

`significance_score = -log10(q_value)`，数值越大表示统计显著性越强。显著性表示在当前文献数据中相对其他细胞富集，不等同于生物学因果关系。

## 重新生成

在项目根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/build_marker_scores.py
PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/build_marker_review.py
PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/build_significant_markers.py
PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/build_significant_marker_review.py
```

正式重算时要求映射已经完成语义判定，只接收 `match` 和 `exact_auto`：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/build_marker_scores.py \
  --mapping database/cell_name_catalog_matches.tsv \
  --require-adjudicated-mapping
```

当前 `name_mapping/sapbert_top1.tsv` 没有 `decision` 字段，不能通过该正式模式。

## Marker 证据浏览器

证据浏览器仅索引正式映射中的 `exact_auto` 和 `match` 名称。页面可以沿标准细胞树查看：

- 非标准名称映射到了哪个标准细胞；
- 每个名称贡献了哪些 Marker；
- 每个标准细胞的全部 Marker、Statistics 分和 Signatures 分；
- 每条 Marker 证据来自哪个非标准名称、哪篇论文和哪类表。
- 标准 Marker 在原始表中的基因写法、论文物种、HGNC/MGI 映射方法和参考编号。

Marker 分数仍按“论文 × 表类型 × 标准细胞 ID × 基因”封顶；名称证据页保留同一论文中不同非标准名称的贡献，用于追溯映射来源。

在项目根目录构建索引并启动网页：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/build_marker_evidence_explorer.py
PYTHONDONTWRITEBYTECODE=1 python3 -B marker_statistics/serve_marker_evidence_explorer.py --port 8766
```

浏览地址为 `http://127.0.0.1:8766`。映射、目录、Marker 分数或论文标准化表更新后，需要重新运行索引构建命令。
