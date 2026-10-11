# 单细胞验证

本任务集中保存原先散落在 CellPhenoRank 中的单细胞分析与验证代码。

| 代码 | 职责 |
| --- | --- |
| `src/cellphenorank_validation/analysis.py` | 预测细胞到数据标签映射、供体层分析、模块分数与匹配空模型 |
| `case_freeze.py` | 表达分析前冻结预测终点与预检条件 |
| `discovery/` | 病例发现、数据集目录、metadata 预检与 SAVI/XLA/FOXG1 分析 |
| `plotting.py` | 单细胞验证图表 |
| `configs/`、`scripts/`、`tests/` | 任务配置、数据准备与离线验证 |
| `docs/` | 源实验协议、结果与案例说明 |

通过根目录安装的 `cellphenorank` CLI 调用；实现仅在本目录维护，依赖基因排序任务的公共输入/输出契约。

```bash
python -m cellphenorank validate-single-cell \
  --config 04_downstream_evaluation/single_cell_validation/configs/ipex_single_cell.yaml \
  --output-dir 04_downstream_evaluation/single_cell_validation/outputs/ipex
```

正式运行需要真实数据、预检记录、冻结病例及基因排序结果。分析单位为供体/样本，不能把细胞数当作独立重复数。源实验摘要见 [单细胞验证结果](docs/CellPhenoRank_single_cell_validation_results.md)，当前代码重构未重做该实验。
