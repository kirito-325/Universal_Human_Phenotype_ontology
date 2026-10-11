# CellPhenoRank 候选基因排序

本任务将疾病 HPO、小鼠基因型 MPO 与表型相似度转为候选基因排名，再进行二元 CPO 富集和 leave-one-HPO-out 约束重排。保留 PhenoDigm、Exomiser、SapBERT 等对照实现。

`src/cellphenorank/` 是排序和基准代码；`configs/` 保存排序配置；`examples/` 是小型合成输入；`tests/` 是排序测试。单细胞实现已移到相邻 [single_cell_validation](../single_cell_validation/README.md)，统一 CLI 仍可调度该任务。

## 合成示例

安装根目录 `.[scientific,test]` 后，在仓库根目录运行：

```bash
python -m cellphenorank run \
  --config 04_downstream_evaluation/gene_prioritization/configs/example.yaml \
  --output-dir 04_downstream_evaluation/gene_prioritization/outputs/example
```

此示例检查既有排序工作流，不是生物学结果。当前正式重排的入口为：

```bash
python -m cellphenorank run-parameter-free \
  --config 04_downstream_evaluation/gene_prioritization/configs/real_frozen_parameter_free.yaml \
  --output-dir 04_downstream_evaluation/gene_prioritization/outputs/parameter_free
```

正式运行需另备冻结队列、基线矩阵、CPO、Marker 与功能基因证据。完整方法见 [当前重排规范](CellPhenoRank_parameter_free_reranking_revision_plan.md)，源实验结果见 [结果摘要](CellPhenoRank_current_results.md)。摘要中的数值未在本次重构中重新计算。
