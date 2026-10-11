# deep-life-sci 代码来源与改动

本项目以相邻的 `../deep-life-sci` 工作树代码为基础裁剪。
读取时仓库 HEAD 为 `83009f8ecae0a4f1003eb032a77a0b19dc20079b`；
该工作树有用户已有改动，因此下表的实际源文件 SHA-256 比 HEAD 更精确。
原工作树未被本次改造修改。

| 上游文件 | 本项目位置与处理 |
| --- | --- |
| `deep_life_sci/agent.py` | `src/hpo_agent/agent.py`，保留 `build_agent`、`create_deep_agent`、QuickJS middleware、PTC allowlist、subagent 与错误包装的组装结构；替换任务工具和提示词，增加结构化终止契约 |
| `sources/pubmed.py` | 复制到 `upstream/sources/pubmed.py`，沿用查询、批量摘要、NCBI 限流/重试、标识符校验、解析与缓存 |
| `sources/pmc.py` | 复制到 `upstream/sources/pmc.py`，沿用 PMC S3 版本发现、JATS/表格/图注解析与全文获取；删去 sandbox 图像/附件上传工厂 |
| `sources/_http.py`、`_errors.py`、`cache_io.py` | 复制到 `upstream/sources/`，保留 HTTP 基础设施与缓存实现 |
| `middleware/tool_errors.py` | 复制到 `upstream/middleware/`，继续在 PTC 桥接处把可预期来源错误返回为 `{error}` |
| `paths.py` | 任务化简化为 `upstream/paths.py`：缓存目录由 ContextVar 按批次隔离，不引入 sandbox 路径 |
| `prompts/system.py`、`prompts/subagents.py` | 沿用 root 编排、上下文隔离、先取证后派发的设计，替换为 `prompts.py` 的细胞表型任务契约 |
| `tests/test_pmc.py` | 保留为 `tests/test_upstream_pmc.py`，只修改包导入路径 |

复制代码中的包导入改为 `hpo_agent.upstream`。
PubMed/PMC 中固定缓存常量改为上下文缓存路径函数，以便多个批次不串缓存目录。
PubMed 的 fatal esearch 错误增加显式检查，防止错误响应被当成零命中；
普通 query warnings 继续保持上游行为。

`literature.py` 现在仅是上述原工具到 `Document` 的数据契约适配层。
上一版自行实现的 NCBI/Europe PMC HTTP 客户端已被替换。
不依赖原项目的聊天 UI、临床试验、科学计算执行、实验设计或 LangSmith sandbox。

新增内容集中于：`ontology.py` 本体工具、专业子智能体、`session.py` 证据状态/版本校验，
以及 `pipeline.py` 的批次管理与导出。程序不替 root 固定检索/提取/标准化顺序。

源文件 SHA-256：

```text
03cff2f213181f8a60824a72d4b47c79cdd122f7dfa34937f5cf782d8f220e52  deep_life_sci/agent.py
abe3b8e1c6dcc14e82d9a2e03a08d9ea444e48bb8e541bf609b38daf06fbd92d  deep_life_sci/sources/pubmed.py
561a5a31e813a8ef9466eb883a9f04a83f874baff456a2ad1f2580158657ce71  deep_life_sci/sources/pmc.py
eee6785958729d9e7b5708784ecbc7e29cf1d597d13d7baa459f01fe8909f640  deep_life_sci/sources/_http.py
87dfb47efe5ce9953cd0476f60291d66fcfc7465ea547e0e029be9463ea496c3  deep_life_sci/sources/_errors.py
dcc15c9b7369acee2d5b0401d75ce4fedff840309e3741286333b5ac6ce2de85  deep_life_sci/sources/cache_io.py
f7869e5065e27eb6a3dc8a8c6bd24ea48af5ea97f9740e2222a5854ffea0e299  deep_life_sci/middleware/tool_errors.py
587a37a863f372bead34881340ac15fc3241c4e099bfe95397eaddc4112b38ae  tests/test_pmc.py
```

上游 MIT 许可保存在 `src/hpo_agent/upstream/LICENSE`，随包分发；
完整来源说明见 `THIRD_PARTY_NOTICES.md`。
