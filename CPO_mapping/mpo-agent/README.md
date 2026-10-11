# MPO Agent

**MP → MGI 注释小鼠模型 → 逐模型细胞机制 → MP 级机制汇总。**
每个模型都先调研文献，再提出假说，再通过支持检索和合理性评审验证、修订。
允许读过文献后没有可保留机制，研究结果为 inconclusive。

## 安装与数据库

~~~sh
uv sync
uv run mpo-agent mp-context --phenotype-id MP:0000003 --limit 5
~~~

也可使用 python -m mpo_agent。公开代码快照不附带 CL/GO 和 MP/MGI 索引，需在本地建立。
数据来自 MGI 的 MP 本体及完整基因型注释报告，包含条件性突变。
下载地址和字段说明：[MGI 官方数据报告](https://www.informatics.jax.org/downloads/reports/index.html)。

新克隆可下载并建立索引：

~~~sh
mkdir -p data
curl -L --fail -o data/MPheno_OBO.ontology \
  https://www.informatics.jax.org/downloads/reports/MPheno_OBO.ontology
curl -L --fail -o data/MGI_PhenoGenoMP.rpt \
  https://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt
uv run mpo-agent build-mp-context --mp data/MPheno_OBO.ontology \
  --annotations data/MGI_PhenoGenoMP.rpt --output data/mp_context.sqlite
uv run mpo-agent build-index --cl /path/cl.obo --go /path/go.obo --output data/ontology.sqlite
~~~

已有数据库不会被覆盖，来源路径与 SHA-256 随索引和结果保存。
2026-10-10 建立的背景快照含 15,357 个 MP 术语、76,719 个基因型模型、396,060 条注释。
数据目录不加入 Git。

## 输入与运行

输入 JSONL 每行仅需 MP 标识符：

~~~json
{"phenotype_id":"MP:0000003","include_descendants":false}
~~~

默认只研究精确注释模型。include_descendants=true 显式包含子术语注释，并保留匹配范围。
模型定义、基因型、遗传背景和注释论文由数据库获取，不能让模型填写或按基因名合并。

~~~sh
uv run mpo-agent run --input examples/terms.jsonl --settings examples/settings.json \
  --index data/ontology.sqlite --mp-context-index data/mp_context.sqlite \
  --env-file ../.env --output runs/example
~~~

max_models 默认为 null，登记全部匹配模型。显式限制 max_models 或预算不足时，
未调研模型保留 deferred，整体状态为 partial，不宣称完整覆盖。
max_searches/max_papers 等为每模型预算；max_total_searches/max_total_subagent_calls 是全局预算。
模型环境变量、提供商、并发限制沿用 HPO Agent 的既有配置。

offline 只禁用文献网络，仍会调用配置的模型。离线文献必须显式绑定模型：

~~~json
{"model_ids":["MGI:123"],"document":{"document_id":"local:1","title":"Example","text":"Article body","source_url":"urn:example","source_kind":"local_text","phenotype_ids":["MP:0000003"]}}
~~~

通过 --documents corpus.jsonl 传入。测试使用可控模型替身，不调用付费模型。

## 调研与证据

一个 Root 通过 eval/task 调度所有模型，没有额外的自主调研循环。
所有文献操作和专家任务显式携带 model_id；每模型的查询、假说、引用和判断相互隔离。

1. 获取 MP 和 MGI 模型背景，分页读取进度。
2. fetchModelReferences 使用程序保存的注释 PMID 获取原始来源；补充搜索并获取其他论文。
3. 注册多方向计划，实际阅读来源后生成零到多个合理候选。可获取的注释来源须优先阅读。
4. 每条候选注册后检索支持信息，获取来源，再评估或定向补充；不强制反证检索。
5. 保留项使用 CL/GO 标准化和科学审核。没有实验不自动否定合理推论。
6. 每模型处理未使用来源并结束。来源无法获取、未调研、科研上不确定分开记录。
7. 汇总已接受的模型机制，保留全部模型来源、条件、假设、实验或观察及反证。

注释论文证明模型与 MP 的关联，不自动证明细胞机制。同基因、不同等位基因、
遗传背景、条件敲除、组织、年龄和处理条件不能被假定为同一模型。
其他模型或物种的论文可支持合理推论，但保留真实物种及外推条件。
关键模型身份或实验细节不在摘要中时，优先获取可用全文；获取失败如实记录。
不能将其他模型的研究自动升级为本模型的 whole_claim 实验证明。
以关键环节有支持、推理连贯和适用假设合理为保留标准，不要求排除其他解释。
whole_claim/contradicts 标签不直接否决；已读且适用的结果确实否定关键前提时，
评审才给出具体科学阻断。实验发现的汇总状态与整条推论是否保留分别表达。

## 输出

- models.jsonl：每模型的背景、状态、结论、假说和来源覆盖。
- relations.jsonl：接受的丰富细胞关系，程序绑定 model_id。
- mechanisms.jsonl：按 CL/GO/方向组织的 MP 机制组，保留每个模型的成员结论。
- hypotheses.jsonl：带 model_id 的全部候选及其评估。
- terms.jsonl、manifest.json：总体状态、模型覆盖、数据库与代码版本。
- coordinator.json、模型级 session.json 及 audit/events：运行与来源轨迹。

同一组本体 ID 不代表所有成员科学机制完全相同，也不代表该 MP 的所有模型均有此机制。
不同方向分组保存，不用投票消除差异。未获得机制的模型仍出现在结果中。
最终存储版本 1.0、模型契约 mpo_compact_v1，见 schemas/contracts.json。

## 代码复用与验证

文献客户端、Root/QuickJS 调度、专家引擎、证据短编号、确定性归一化、定向补充、
CL/GO 索引、科学审核和丰富输出从拆分前的 HPO 项目复用。
MPO 背景索引、模型调度和汇总属于本仓库；运行不依赖相邻 HPO 项目的 Python 包。

测试覆盖真实编译的 Root → eval → task 调用、来源合并、无机制结果、
不同模型与引用隔离、定向科学补充、方向不同的汇总、覆盖缺口及严格结果重载。
本轮没有调用付费模型执行实际 MP 科研案例。

[模块说明](../README.md)
