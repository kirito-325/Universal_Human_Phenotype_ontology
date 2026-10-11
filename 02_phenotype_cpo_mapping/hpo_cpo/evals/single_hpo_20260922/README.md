# 单术语外部调研流程测试

用户指定术语：HP:0002930 — Impaired sensitivity to thyroid hormone（甲状腺激素敏感性降低）。

名称、定义和同义词读取自官方 HPO 快照 `../external_hpo10_20260921/hp.obo`。
不传入该定义的参考 PMID，不提供已有文献、疾病背景、基因或 CL/GO 答案候选。
此次使用新的 `runs/single_HP_0002930_20260922` 目录和文献缓存。

目标是先使至少一条有依据的结论走完检索、提取、CL/GO 标准化、独立审核和结构化输出。
这是工作流测试，不代表对该 HPO 的完整知识覆盖。没有充分支持时必须如实保留未解决状态。
原来充足的单术语资源上限保留，模型请求总并发不超过 3。

前一批 10 术语测试结果已从 runs/ 移入 `.trash/hpo10_20260922_1436/`，可恢复。
中性粒细胞减少的单术语任务没有实际启动；当前输入已替换为用户指定的 HP:0002930。

运行命令：

```bash
PYTHONPATH=src LANGSMITH_TRACING=false python -m hpo_mpo_cell_agent run \
  --input evals/single_hpo_20260922/terms.jsonl \
  --index data/ontology.sqlite \
  --env-file ../deep-life-sci/.env \
  --settings evals/single_hpo_20260922/settings.json \
  --output runs/single_HP_0002930_20260922
```

运行期间每个图步骤会追加到对应 attempt 目录的 `events.jsonl`，完整审计保存为 `audit.jsonl`。
