# Cell marker 数据集

本目录独立维护 CL–marker 数据准备流程，不与 GO function gene 求交集，也不使用 GO→CL 约束。

## 生成方法

```bash
python3 -B -m cell_marker.export_cell_markers
```

默认读取：

- 细胞目录：`/path/to/workspace/marker_literuture/expanded/expanded_cell_catalog.tsv`
- Marker 评分：`/path/to/workspace/marker_literuture/marker_statistics/expanded_marker_scores.tsv`
- 人类基因符号：`/path/to/workspace/UHPO（6-8）/marker-agent/data/hgnc/hgnc_complete_set.txt`
- 小鼠到人类同源基因：`/path/to/workspace/HPO_expand/UHPO（6-8）/data/orthologs/mouse_human_orthologs.tsv`
- 小鼠当前及历史符号：`/path/to/workspace/HPO_expand/MPO-HPO/data/MGI/script/MRK_List2.rpt`

原始候选仍按输入表的 `marker_rank` 排序，但在截取 Top 50 前依次执行：

1. HGNC 当前官方符号直接保留并统一为标准大小写。
2. 无歧义的 HGNC 历史符号和别名转换为当前官方符号。
3. MGI 小鼠当前符号及无歧义历史同义词通过人鼠同源表转换为当前人类 HGNC 符号；一对多关系展开为多个候选。
4. 无法解析为人类基因或带人类同源基因的小鼠基因的值予以剔除。
5. 按转换后的人类符号去重，再保留前 50 个并重新生成连续的 `marker_rank`。

原始名次和符号分别保存在 `source_marker_rank` 和 `source_gene`；映射方法与
同源关系类型保存在 `gene_mapping_method` 和 `ortholog_mapping_type`。

## 结果

结果统一位于 `cell_marker/results/`：

- `cell_marker_genes.tsv.gz`：每个有数据的细胞最多 50 个清洗后人类 marker，并保留原始符号、原始排名、映射方法和评分字段。
- `cell_marker_summary.tsv`：全部目录 CL ID 及各细胞实际保留的 marker 数量。
- `cell_marker_rejected_genes.tsv.gz`：未通过基因规范化的原始值、拒绝原因和出现次数，供审计清洗规则。
- `cell_marker_manifest.json`：输入、规则和审计计数。
