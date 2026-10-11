# HPO–MPO 先导对接

本目录保留双端先导测试代码，原 `aligment_test` 已按任务重命名为 `agent_alignment`。

| 文件 | 作用 |
| --- | --- |
| `prepare_pairs.py` | 按固定选择和版本化本体生成 10 对对象、运行输入及校验清单；支持 `--check` |
| `run_research.py` | 两端独立调研、逐术语隔离输出、记录耗时与覆盖状态；支持 `--prepare-only` |
| `report_research.py` | 从指定运行目录汇总完整性、耗时和关系数量 |

原先导队列为 5 对人工精确映射、3 对人工层次映射、2 对未证实机制候选。它用于检查工作流和机制解释，不足以单独证明映射准确率提升。机制候选不是 gold positive。

截至源快照记录，20 个术语的调研已暂停：7 项结束（2 项完整、5 项不完整），2 项中断，11 项未开始；中间层对接尚未运行。此次整理未继续调研。

## 未附带的输入

真实先导对象、来源快照及运行记录此次均不上传。恢复原实验需另外提供：

- `selection_spec.json` 与 `reference/literature.json`；
- 固定人工映射源文件（路径及 SHA-256 由 selection spec 指定）；
- 两个 agent 的版本化本体/背景索引；
- `prepare_pairs.py` 生成的 `pairs.jsonl`、`inputs/` 和校验清单。

上游人工映射来自 [MHMI 固定提交](https://github.com/mapping-commons/mh_mapping_initiative/blob/fd29f919bf314b8dfc08f8b5130ccf019ff13f35/mappings/mp_hp_mgi_all.sssom.tsv)，原映射方向是 MP 为 subject、HP 为 object。应独立审查其许可、版本及方向。

安装两个 agent 后，在仓库根目录执行：

```bash
python 03_mpo_hpo_mapping/agent_alignment/prepare_pairs.py --check
python 03_mpo_hpo_mapping/agent_alignment/run_research.py --output 03_mpo_hpo_mapping/agent_alignment/runs/new_batch --prepare-only
python 03_mpo_hpo_mapping/agent_alignment/run_research.py --output 03_mpo_hpo_mapping/agent_alignment/runs/new_batch
python 03_mpo_hpo_mapping/agent_alignment/report_research.py --output 03_mpo_hpo_mapping/agent_alignment/runs/new_batch
```

前两条也需要上述本地输入。正式运行会调用模型；`--prepare-only` 只准备输入。脚本读取仓库根目录 `.env`，已导出的环境变量优先。
