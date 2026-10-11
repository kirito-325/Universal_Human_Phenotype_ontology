# HPO Agent

从文献与合理科学推理中构建 **人类 HPO → 细胞异常（CL + GO + 方向）** 的关系。
仅接受 HP: 输入，结论针对人类细胞；动物实验保留真实研究物种，并明确适用性假设。

## 安装与运行

~~~sh
uv sync
uv run hpo-agent hpo-context --index data/hpo_context.sqlite --phenotype-id HP:0009511
uv run hpo-agent run --input examples/terms.jsonl --index data/ontology.sqlite \
  --hpo-context-index data/hpo_context.sqlite --settings examples/settings.json \
  --env-file ../../.env --output runs/example
~~~

也可使用 python -m hpo_agent。cell-agent 命令作为现有通用入口别名保留。
模型配置继续读取既有环境变量，凭据不复制到另一个项目。

新克隆需要准备本地版本化索引：

~~~sh
uv run hpo-agent build-index --cl /path/cl.obo --go /path/go.obo --output data/ontology.sqlite
uv run hpo-agent build-hpo-context --hpo /path/hp.obo \
  --annotations /path/phenotype.hpoa --output data/hpo_context.sqlite
uv run hpo-agent schemas --output schemas
~~~

已有索引不会被静默覆盖。本次公开代码快照不附带索引，需要先在本地建立。

## 调研流程

HPO 背景 → 多方向文献召回 → 阅读文献并提出假说 → 每条假说的支持检索 →
评估及必要修订 → 仅标准化保留项 → 审核 → 最终研究结论。

Root 继续通过 eval 的 camelCase 工具和 task 专家调度研究，无 literature-scout 循环。
输入不是疾病诊断；祖先术语的疾病关联只作为背景线索，否定或零频率注释不能当作正向关联。

模型只负责细胞、异常过程、表型连接的科学解释与适用假设，以及实验结果的解释、
本体选择和科学审核。程序生成引用定位、来源元数据、物种标签、连接字符串、规范标签和计数。
缺少实验证明可以保留合理推论为 conditional；不要求专门检索反证或排除其他机制。
证据标签不直接否决；已读且适用的结果否定关键前提、来源误读或无法辩护的外推仍会阻断接受。

## 提交与恢复

存储 Schema 为 2.0；科学模型接口保留 compact_v2。
字段错位和相同重复值由本地程序无损归一化，冲突不会猜测覆盖。
任务内 E1/E2 证据编号对应不可变文献、哈希与位置；失败后的定向补充恢复原映射。
验证来源在付费评估前检查并交付正文，不能把元数据当作读过的证据。
缺失科学解释或无效引用可通过原评估器 complete_reasoning 补充一次指定字段；
其他科学内容冻结，科学变化使用显式 reevaluate。网络或空响应保留补充机会。

正式模型 Schema 与丰富研究记录分别见 schemas/contracts.json。
历史丰富记录组件接口保留供隔离测试，不能通过 CLI 启动 component_test 模式。

## 输出

- terms.jsonl：研究状态、完整结论和覆盖情况。
- hypotheses.jsonl：完整假说池、评估决定和历史。
- relations.jsonl：审核接受的关系，包括条件性推论。
- conditional_edges.jsonl：条件性关系；edges.jsonl：直接主关系。
- cell_abnormalities.jsonl：规范化细胞异常节点。
- 每次尝试的 audit.jsonl、events.jsonl、session.json：来源和运行轨迹。

实际实验或观察单独保存于 experimental_validation，包含物种、范围、结果、
支持或挑战方向、原文及来源。组件证据不自动成为整条结论的实验证明。
退休结论保留历史，但从当前计数和导出中排除。

## 项目拆分

原 hpo-mpo-cell-agnet 目录已改名为 hpo-agent，Python 包改为 hpo_agent。
MP 调研移至相邻的 [mpo-agent](../mpo_cpo/README.md)，本项目不再接受 MP 输入或执行 MGI 模型桥接。
历史 HP 记录中的空模型字段可只读迁移；非空模型上下文会被拒绝。

本仓库只收录当前代码，不附带原工作区的 runs、archives、数据和研究结果。

[模块说明](../README.md) · [研究架构](ARCHITECTURE.md)

`dataset_tools/` 为旧 TSV 关系表的合并与继承工具，不自动转换本 agent 的丰富 JSONL 输出。
