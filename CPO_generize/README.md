# CPO 构建组件与下游实验

本目录保留原工作区内部结构，便于复用已有导入、脚本和配置。只收录相关的当前代码，不附带 `data/`、`work/`、`results/` 或模型文件。

| 模块 | 用途 | 阅读顺序 |
| --- | --- | --- |
| [GO_link](GO_link/README.md) | 当前 GO-BP 范围判定与精确 GO–CL 链接 | CPO 构建 |
| [GO_function](GO_function/README.md)、[cell_marker](cell_marker/README.md) | 功能基因与 Marker 导出 | CPO 构建 |
| `celltype_configure`、`llm_api` | 语义/SCEA 证据及模型 API 客户端 | 共享支持 |
| [UHPO](UHPO/README.md) | 既有四阶段整合实现 | 构建流程参考 |
| [mission3](mission3/README.md) | 具体小鼠基因型优先级排序 | 下游探索 |
| [mission4](mission4/README.md) | CPO 相似度矩阵、疾病模型评分与基因预测 | 下游探索 |
| [MPO_HPO_mapping_v4](MPO_HPO_mapping_v4/README.md) | 无训练语义/CPO 对照基准 | 下游验证 |

最新版术语调研见根目录 [CPO_mapping](../CPO_mapping/README.md)。较新的 CellPhenoRank 位于根目录 [mission3](../mission3/README.md)。历史结构嵌入版本已移至 [legacy/MPO_HPO_mapping](../legacy/MPO_HPO_mapping/README.md)，明确标为弃用。

原配置中对 `HPO_link*`、`03_MPO_link` 或旧结果文件的引用是外部数据接口，相关旧代码和结果没有整体复制。完整复现需准备相应输入，或显式配置新导出文件；本次没有将新 agent 的输出静默转换为旧接口。
