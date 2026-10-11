# GO function gene 数据集

本目录独立维护 GO Biological Process–智人功能基因数据，不与 cell marker 求交集。

## 生成方法

```bash
python3 -B -m GO_function.export_go_functions
```

默认读取：

- 智人 GOA：`/path/to/workspace/UHPO（6-8）/data/go/goa_human.gaf`
- GO 本体：`/path/to/workspace/HPO_expand/CPO/data/go/go-basic.obo`

仅保留直接标注到有效 Biological Process 的智人正向 GOA 注释；去除 `NOT`，不把子 GO 的基因向祖先传播。

## 结果

结果统一位于 `GO_function/results/`：

- `go_function_genes.tsv`：每个直接 GO–gene 组合及其证据代码、关系、UniProt 对象和文献。
- `go_function_summary.tsv`：全部有效 GO Biological Process 及直接功能基因统计。
- `go_function_manifest.json`：输入、规则和审计计数。
