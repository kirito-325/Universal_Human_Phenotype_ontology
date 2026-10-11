# 单细胞文献 Marker 与细胞特征分析

从单细胞文献的补充表格中抽取细胞类型和 Marker，将名称映射到审核的 CL 目录，计算特征性、显著性及证据覆盖。本目录归第一模块的文献 Cell Marker 任务。

| 组件 | 入口 | 作用 |
| --- | --- | --- |
| 文献发现 | `scripts/run_literature_pipeline.py discover` | PubMed 查询与论文元数据归档 |
| 补充资料与表格 | 同一 CLI 的 `probe-assets`、`download-assets`、`extract-tables` | 附件发现、下载及表格提取 |
| Marker 标准化 | `standardize-marker-tables`、`agent-standardize-marker-tables` | 规则解析或结合论文上下文的模型映射；不补造统计值 |
| CL 扩展目录 | `scripts/build_expanded_cell_catalog.py` | 复用仓库内骨架与功能性 T 细胞粒度政策 |
| 细胞名称归一 | `scripts/build_sapbert_name_mapping.py`、`match_cell_names_to_catalog.py` | 语义候选与审核映射 |
| 特征分析 | `marker_statistics/` | Marker 分数、显著 Marker、覆盖审计与证据浏览 |

## 开始

先在仓库根目录安装 `.[scientific,test]`，再从本任务目录运行。实际语义编码另装 `.[embedding]`。

```bash
python3 scripts/run_literature_pipeline.py --help
python3 scripts/run_literature_pipeline.py discover --dry-run
python3 scripts/run_literature_pipeline.py --config config/literature_pipeline_all_cells.json discover --dry-run
```

默认配置限定期刊及单细胞检索范围；`literature_pipeline_all_cells.json` 扩展到所有细胞类型。先检查查询，再显式执行联网发现和附件下载。发现、下载、表格解析和模型标准化是不同步骤。

```bash
python3 scripts/run_literature_pipeline.py discover --max-results 10000
python3 scripts/run_literature_pipeline.py extract-tables --limit 0
python3 scripts/run_literature_pipeline.py standardize-marker-tables
```

模型标准化读取 `ECNU_API_KEY`，导出的环境变量优先，可通过 CLI 指定本地 `.env`。模型调用可能产生费用。不要把密钥写入配置或脚本。

## 产物与分析

`database/` 保存论文元数据、补充表格与逐论文标准化结果；`expanded/` 保存审核细胞目录；`name_mapping/` 保存名称映射；`marker_statistics/` 下的 TSV/数据库/HTML 在本地生成，均不上传。

统计格式包括 `cell_cluster,gene,p_val,avg_logFC,pct.1,pct.2,p_val_adj`；宽格式 signature 保留细胞/特征名称下的原基因列表。缺失 p 值或统计量不自动填充。显著性过滤、证据评分和目录覆盖是不同分析，不能互相替代。

方法和浏览入口见 [marker_statistics/README.md](marker_statistics/README.md)。CL 骨架代码见 [cl_skeleton](../cl_skeleton/README.md)，外部资源与占位路径见 [数据指南](../../docs/data.md)。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试使用临时目录与合成表格；正式数据与联网服务可用性需独立验证。系统级 LaunchAgent、运行日志和一次性论文恢复脚本不属于本次代码发布。
