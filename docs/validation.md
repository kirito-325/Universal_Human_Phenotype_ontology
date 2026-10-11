# 验证记录

检查日期：2026-10-11。以下检查针对整理后的代码，使用模型替身、临时数据库及合成输入，没有发起真实文献调研或付费模型调用。

| 检查 | 结果 | 环境与范围 |
| --- | --- | --- |
| HPO Agent | 279 passed | Python 3.14.7；全部既有测试，含真实 Root/eval/task 调度与模型替身 |
| MPO Agent | 97 passed | Python 3.14.7；全部既有测试，含模型隔离与聚合 |
| 文献 Marker | 68 passed | Python 3.11.15；表格抽取、名称过滤、评分与证据浏览 |
| CellPhenoRank | 39 passed | Python 3.11.15；包含合成端到端排序 |
| 具体小鼠模型排序 | 19 passed | Python 3.11.15；含基准准备与 schema 验证 |
| Mission4 CPO 比较 | 15 passed | Python 3.11.15；矩阵、方向、数据构建及浏览器 |
| GO/CL 构建组件 | 51 passed | Python 3.11.15；所收录的相关测试，未执行 SapBERT 模型编码 |
| 无训练基准 | 6 passed | Python 3.11.15；数学契约及无需真实数据的 provider 测试 |
| 小鼠模型排序合成 CLI | 通过 | 写出 2 条模型排名，输出放在仓库之外 |
| 文献发现 dry-run | 通过 | 生成 PubMed 查询，不发起网络请求 |
| 发布内容检查 | 通过 | Python 编译、JSON 解析、数据/二进制排除、凭据及非回环 IP 检查 |
| Markdown 本地链接 | 通过 | 不将未公开结果文件伪装为可点击文件 |
| Git whitespace 检查 | 通过 | 保留 Markdown 换行和 TSV 空末字段的合法空白 |

合计 **574 项测试通过**。Agent 验证复用了本机已有的匹配依赖；科学计算测试使用隔离的临时环境。没有把测试依赖缓存或环境提交到 Git。

典型复现命令（先按各模块安装依赖）：

```bash
# 在相应 agent 目录分别执行
python -m pytest
# 在 marker_literuture、mission3、CPO_generize/mission3、CPO_generize/mission4 分别执行
python -m pytest
# 在 CPO_generize 执行
python -m pytest tests
# 在 CPO_generize/MPO_HPO_mapping_v4 执行
PYTHONPATH=src python -m pytest tests/test_math_contracts.py \
  tests/test_real_execution_contracts.py::RealExecutionContractTests::test_provider_accepts_term_ids_only
# 在仓库根目录执行
python scripts/check_repository.py
```

模型服务或本体数据版本变化后的结果不由这些离线检查保证。GitHub Actions 仅运行标准库发布检查，不自动触发付费调研或全量科研实验。

不在本次验证范围：付费模型调研、真实数据全量运行、完整矩阵生成、历史嵌入训练、需要原结果的执行契约测试，以及新 agent 与旧下游输入之间的转换。
