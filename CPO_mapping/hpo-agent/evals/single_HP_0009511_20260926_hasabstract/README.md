# HP:0009511 — 仅召回有摘要论文的文献优先流程实测

当前代码在 LiteratureClient.search 中对每次 PubMed 查询强制添加 `(原查询) AND hasabstract`，并向 root 和 literature-scout 说明筛选契约。查询日志保留 query、effective_query 和 filters。本轮不修改运行中的代码。

输入仅为官方 HPO 名称、定义及同义词，不提供细胞、机制、CL/GO、疾病答案或 PMID 种子。HPO 注释背景来自 data/hpo_context.sqlite；CL/GO 索引为 data/ontology.sqlite。

输出与全新文献缓存：runs/single_HP_0009511_20260926_hasabstract。旧轮论文、候选和缓存不导入本轮。

模型沿用 ecnu-max，术语并发 1，模型共享并行上限 3；沿用 100 次查询、100 篇论文及 4 小时总上限。smoke_test 仍要求至少 5 个方向、5 篇初检论文、5 个不同假说、逐个支持/挑战检索和评估，再对保留项进行标准化与审核。

监督重点：有效查询是否带 hasabstract，是否再次重复空取回；文献子任务能否交回 root；实际假说及淘汰理由；是否只标准化经验证的保留项；原文、推理链、术语合法性和运行错误。

实际结果：用户要求停止后进程已退出。完成 42 次过滤检索、14 篇摘要及其中 3 篇全文获取，空取回为 0；原生 scout 尚未返回，因此未生成假说。离线检查确认阶段前提通过、结构化交接接口可用，同时确认原生 scout 未接入统一的内部消息记录和无进展保护。详见输出目录 REPORT.md；具体内部触发原因因缺失逐轮日志尚不能确定。
