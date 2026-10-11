# HP:0009511 单术语外部调研测试

术语：Stippling of the epiphysis of the distal phalanx of the 2nd finger
（第 2 指末节指骨的骨骺点状钙化）。

名称、定义和同义词来自官方 HPO 快照 `../external_hpo10_20260921/hp.obo`。
疾病背景、PMID 种子、基因与预期 CL/GO 答案均未提供，不载入已有文献。

沿用 HP:0002930 单术语测试的 `../single_hpo_20260922/settings.json`：
smoke_test 模式、术语并发 1、所有模型请求共享最多 3 个并发槽位，保留充足资源上限。
使用全新的输出目录与文献缓存：`runs/single_HP_0009511_20260922`。

本术语的解剖定位很具体。结果复核应区分对该部位的直接证据与一般骨骺点状钙化的条件性机制证据，
不能根据词义相近或本体匹配就把后者升级为前者。

```bash
PYTHONPATH=src LANGSMITH_TRACING=false python -m hpo_mpo_cell_agent run \
  --input evals/single_HP_0009511_20260922/terms.jsonl \
  --index data/ontology.sqlite \
  --env-file ../deep-life-sci/.env \
  --settings evals/single_hpo_20260922/settings.json \
  --output runs/single_HP_0009511_20260922
```
