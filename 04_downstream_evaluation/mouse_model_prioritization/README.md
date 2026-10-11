# 具体小鼠模型优先级与评估

分析单位是 **MGI 基因型模型**，不是基因与 MP 的简单关联。`within_gene` 在已知疾病基因的模型内排序，`all_models` 遍历全部候选且移除遗传兼容评分。表型覆盖、CPO 机制、遗传机制与缺口报告分别保留。

| 目录 | 职责 |
| --- | --- |
| `src/mission3/` | 模型、表型和机制输入；候选筛选、评分、消融与报告 |
| `examples/` | 不依赖真实数据的小型合成输入 |
| `config/` | 严格、语义基线及 provisional 试验配置 |
| `skills/build-rare-disease-mouse-benchmark/` | 冻结队列的选择、证据核验与 schema 工具 |
| `tests/` | 排序、冻结与输入完整性测试 |

安装根目录项目后，先看 `python -m mission3 --help`。在本目录可运行合成示例：

```bash
python -m mission3 run \
  --diseases examples/diseases.tsv --models examples/models.tsv \
  --model-phenotypes examples/model_phenotypes.tsv --mp-hpo-mapping examples/mp_hpo_mapping.tsv \
  --hpo-mechanisms examples/hpo_mechanisms.tsv --mp-mechanisms examples/mp_mechanisms.tsv \
  --cell-catalog examples/cell_catalog.tsv --go-classification examples/go_classification.tsv \
  --cl-obo examples/cl.obo --go-obo examples/go.obo --cre-drivers examples/cre_drivers.tsv \
  --config examples/config.json --output-dir outputs/example
```

没有 CPO 证据时保持 unknown；`tested_normal`、`untested`、`opposite_direction` 与 `not_mappable` 分开记录。既有输入来自历史 TSV，尚未把新 agent 丰富输出转换接入。原实验状态见 [REAL_DATA_STATUS.md](REAL_DATA_STATUS.md)。
