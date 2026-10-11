# HP:0009511 — 直接/推理分流后的外部调研测试

使用 schema 1.2 和独立 evidence-reasoner 的新版框架。输入仅含官方 HPO 术语名称、定义和同义词，
与 2026-09-22 测试一致。不提供已有文献、PMID、疾病背景、预期细胞或 CL/GO 答案。

沿用充足的单术语资源上限；模型总并发 3、术语并发 1。smoke_test 要求先走通一条完整的
取证/推理—标准化—审核链，再根据证据缺口继续探索，不代表穷尽检索或科学质量基准。

新输出与独立文献缓存：`runs/single_HP_0009511_20260923_inference`。
旧的暂停运行及 `archives/HP_0009511_20260922_162934` 保留原样。

重点核验：是否实际调度 evidence-reasoner；能否以原文事实支撑推理链；解剖范围扩展是否记录；
CL/GO 是否支持原文细胞与过程；最终 direct/inferred 分流和审核范围是否正确。

```bash
PYTHONPATH=src LANGSMITH_TRACING=false ../deep-life-sci/.venv/bin/python -u -m hpo_mpo_cell_agent run \
  --input evals/single_HP_0009511_20260923_inference/terms.jsonl \
  --index data/ontology.sqlite \
  --env-file ../deep-life-sci/.env \
  --settings evals/single_HP_0009511_20260923_inference/settings.json \
  --output runs/single_HP_0009511_20260923_inference
```
