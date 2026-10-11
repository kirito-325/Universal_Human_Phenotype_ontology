# 既有 UHPO 四阶段整合实现

本目录收录原 `CPO_generize/UHPO` 的当前整合代码。它将既有文献 Marker、CPO 生成、HPO-CPO、MPO-CPO 流程组织为四个阶段，不代表最新 agent 的替代实现。

| 阶段 | 包 | 脚本 |
| --- | --- | --- |
| 文献 Marker | `uhpo.stages.stage1_markers` | `scripts/01_markers/run_stage.py` |
| CPO 构建 | `uhpo.stages.stage2_cpo` | `scripts/02_cpo/run_stage.py` |
| 既有 HPO 连接 | `uhpo.stages.stage3_hpo` | `scripts/03_hpo_cpo/run_stage.py` |
| 既有 MPO 连接 | `uhpo.stages.stage4_mpo` | `scripts/04_mpo_cpo/run_stage.py` |

统一入口为 `scripts/run_pipeline.py`；输入与输出契约在 `src/uhpo/pipeline.py` 的步骤定义及各阶段 `paths.py` 中。`scripts/export_hpo_cpo_snapshot.py` 和 `verify_snapshot.py` 用于既有结果快照导出与核验。

```bash
python3 -m pip install -e .
python3 scripts/run_pipeline.py --help
```

本次不发布任何 `data/`、文献数据库或快照。完整运行需要原始本体、基因注释、目录及各阶段输入；部分步骤访问网络或模型 API。最新术语调研主线见 [CPO_mapping](../../CPO_mapping/README.md)，构建导航见 [CPO_construction](../../CPO_construction/README.md)。
