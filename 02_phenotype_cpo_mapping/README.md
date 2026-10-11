# HPO / MPO 与 CPO 映射

| 任务 | 代码 | 研究单位 |
| --- | --- | --- |
| [HPO–CPO](hpo_cpo/README.md) | `hpo_cpo/src/hpo_agent/` | 人类 HP 术语；动物实验保留真实物种与外推条件 |
| [MPO–CPO](mpo_cpo/README.md) | `mpo_cpo/src/mpo_agent/` | MP 术语的 MGI 精确注释基因型；逐模型研究 |
| [联合审核](review_browser/README.md) | `review_browser/` | 两端本体树、自身关系与继承关系 |

两个 agent 独立安装，保留来源、方向、科学解释、条件和实验支持。相同 CL/GO 组合不表示所有模型都有相同机制。

各任务的 `dataset_tools/` 收纳原先混在下游目录中的旧关系表合并与继承工具。它们处理历史 TSV 契约，不是新 agent JSONL 的转换器；不应直接把新输出当作旧数据输入。

新 agent 的双端先导运行移到第三模块 [agent_alignment](../03_mpo_hpo_mapping/agent_alignment/README.md)，避免与单端术语挖掘混在一起。
