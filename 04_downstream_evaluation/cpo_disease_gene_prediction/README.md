# CPO 矩阵驱动的疾病模型与基因预测

本任务从原 `mission4` 中拆出，消费第三模块生成的 HPO–MPO 相似度矩阵，不重复维护术语相似度算法。

| 实现 | 职责 |
| --- | --- |
| `uhpo_prediction/disease_model.py` | 疾病与具体小鼠模型的覆盖校正评分 |
| `gene_rank.py` | 小鼠基因、严格一对一人类基因及扩展同源排名 |
| `evaluate.py` | 术语映射与基因预测评估 |
| `report.py` | 联合流程的汇总报告与图表 |

安装根目录项目并准备第三模块的矩阵后运行：

```bash
python -m uhpo_prediction disease-model --resume
python -m uhpo_prediction gene-rank
python -m uhpo_prediction evaluate
python -m uhpo_prediction report
```

默认复用 `03_mpo_hpo_mapping/cpo_profile_similarity/config/cpo_v2.yaml` 的同一数据与结果契约，也可通过 `--config` 指定其他运行。原联合 `python -m cpo_pipeline run-all` 保留调度能力。

真实矩阵、疾病 HPO、MGI 模型、orthology 与独立 gold 均未上传；仅有离线契约测试。本次未接入最新 agent，也未重算结果。
