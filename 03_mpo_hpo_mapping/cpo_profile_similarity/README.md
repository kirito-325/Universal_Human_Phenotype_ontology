# CPO Profile 相似度与术语映射

`cpo_mapping/` 保留较早四通道比较；`cpo_pipeline/` 实现统一 profile 的当前探索版本，包括来源归一、方向签名、CL/GO 相似度、矩阵与术语候选。

| 位置 | 用途 |
| --- | --- |
| `cpo_pipeline/prepare.py` | profile 与结构成员归一 |
| `cpo_pipeline/term_matrix.py` | 双向 BMA、self/inherited 和 CL/GO 辅助矩阵 |
| `cpo_pipeline/mapping.py` | 双向术语候选 |
| `config/` | 当前 v2 / 比较 v1 的配置与方向规则 |
| `scripts/` | 术语配对导出和 MHMI 案例审核 |
| `tests/` | 相似度、方向、矩阵及数值契约 |

安装根目录项目后，从任意目录运行：

```bash
python -m cpo_pipeline --help
python -m cpo_pipeline audit
python -m cpo_pipeline prepare
python -m cpo_pipeline term-matrix --resume
python -m cpo_pipeline mapping
```

后四条需要完整本体及 HPO/MPO-CPO 关系输入。配置中的 `project_root` 相对配置所属任务解析，不依赖终端当前目录。

原 `mission4` 的疾病模型、基因排序、总体评估与报告实现已拆到第四模块 [cpo_disease_gene_prediction](../../04_downstream_evaluation/cpo_disease_gene_prediction/README.md)。既有 `run-all` 入口仍可按顺序调用两模块；`query`、`explain`、`validate` 保留联合结果的审查功能。

真实数据、矩阵、数据库和运行结果未发布；最新 agent 的 JSONL 还没有转换接入此历史输入契约。
