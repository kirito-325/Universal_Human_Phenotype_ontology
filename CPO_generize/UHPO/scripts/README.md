# UHPO 脚本入口

`scripts/` 只保存可执行入口；可复用实现统一位于 `src/uhpo/`。脚本编号与论文当前完成的方法顺序一致：

1. `01_markers/run_stage.py`：文献 Marker 候选、扩展 CL 目录与统计表
2. `02_cpo/run_stage.py`：GO 功能基因、CPO 候选召回与精确 CL--GO 关系
3. `03_hpo_cpo/run_stage.py`：HPO 解剖物化与 HPO--CPO links
4. `04_mpo_cpo/run_stage.py`：MPO 清洗、同源转换、解剖物化与 MPO--CPO links

MPO--HPO 链接仍在开发，仅保存在 `work_in_progress/05_mpo_hpo_linking/`，不属于本目录的可复现主线。

## 环境

在仓库根目录运行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

需要模型调用的步骤从根目录 `.env` 读取 `ECNU_API_KEY`。密钥不得写入脚本、配置或 Git。标为 `network` 的步骤需要访问外部本体或文献服务。

## 完整运行

先检查步骤与现有快照：

```bash
.venv/bin/python scripts/run_pipeline.py --list
.venv/bin/python scripts/run_pipeline.py --validate-only
.venv/bin/python scripts/verify_snapshot.py
```

执行全部四个阶段：

```bash
.venv/bin/python scripts/run_pipeline.py
```

统一入口对 `2.05_llm_adjudication` 和 `4.02_classify_terms` 使用已保存批次的严格离线聚合模式；批次不完整或与源本体摘要不匹配时会直接失败，不会静默重请求 API 或改用回退规则。

Stage 4 的结构化批次由已审计的决策表恢复。仅在批次缓存缺失、且已确认决策表为权威版本时执行：

```bash
.venv/bin/python scripts/04_mpo_cpo/rebuild_classification_batches.py
```

只运行一个阶段：

```bash
.venv/bin/python scripts/run_pipeline.py --stage 3
```

从某一步续跑，或在某一步停止：

```bash
.venv/bin/python scripts/run_pipeline.py \
  --stage 4 \
  --start-at 4.03_prepare_terms \
  --stop-after 4.05_anatomy
```

使用 `--dry-run` 可在检查输入后仅打印命令。每个编号目录的 `run_stage.py` 接受相同的阶段内参数，例如：

```bash
.venv/bin/python scripts/02_cpo/run_stage.py --list
.venv/bin/python scripts/02_cpo/run_stage.py --validate-only
```

## 输出与验收

- 外部或冻结输入：`data/raw/01_*` 至 `data/raw/04_*`
- 可重建中间数据：`data/interim/01_*` 至 `data/interim/04_*`
- 论文权威输出：`data/processed/01_*` 至 `data/processed/04_*`
- 当前论文快照：`config/snapshot_manifest.json`

重跑完成后必须再次执行：

```bash
.venv/bin/python scripts/run_pipeline.py --validate-only
.venv/bin/python scripts/verify_snapshot.py
```

若确认要更新论文权威快照，先核对论文数字与所有产物，再显式运行：

```bash
.venv/bin/python scripts/verify_snapshot.py --write
```

仓库排布、数据版本与迁移规则见 `docs/repository/REPOSITORY_LAYOUT.md`。
