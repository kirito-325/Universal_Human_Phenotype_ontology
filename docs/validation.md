# 重构验证

检查日期：2026-10-11。所有测试按重构后的实际任务目录执行，未调用付费模型或运行真实数据实验。

| 测试组 | 结果 |
| --- | --- |
| HPO–CPO agent | 279 passed |
| MPO–CPO agent | 97 passed |
| GO–CL 关系与功能基因组件 | 51 passed |
| 文献 Marker 挖掘与分析 | 68 passed |
| HPO/MPO 旧数据工具及审核浏览器 | 5 passed |
| CPO profile 相似度 | 9 passed |
| 无训练语义/CPO 数学与 provider 契约 | 6 passed |
| CellPhenoRank 基因排序 | 29 passed |
| 具体小鼠模型排序与冻结工具 | 19 passed |
| 单细胞验证 | 10 passed |
| CPO 疾病/基因预测 | 1 passed |
| 仓库归属、跨目录导入、CLI 与配置 | 4 passed，另有 16 项 subtest |

共 **578 项离线测试通过**。科学计算模块用 Python 3.11.15 的隔离环境执行统一 `scripts/run_tests.py --group scientific`，12 组全部通过；两个 agent 在 Python 3.14.7 中使用匹配的已有依赖验证。所有测试环境和缓存均未提交。

另完成以下检查：根目录 editable 安装成功；从临时目录调用五个 CLI 的 `--help`；配置路径不随工作目录改变；CellPhenoRank 合成 CLI 运行成功；50 个 Markdown 本地链接均可解析；Python 编译、JSON、敏感内容、旧目录归属与 Git whitespace 检查通过。

不覆盖付费文献调研、真实矩阵生成、历史嵌入训练或新 agent 到旧下游格式的转换。
