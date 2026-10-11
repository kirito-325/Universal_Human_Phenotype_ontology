# CPO 构建

| 任务 | 实际代码位置 | 产物与边界 |
| --- | --- | --- |
| [CL 骨架生成](cl_skeleton/README.md) | `cl_skeleton/scripts/` | 骨架、扩展目录、层次与目录政策 |
| [CL–GO 关系判定](cl_go_linking/README.md) | `GO_link/`、`GO_function/`、`celltype_configure/` | GO-BP 范围、精确 GO–CL 链接、功能基因与审核证据 |
| [文献 Marker 挖掘](cell_marker_mining/README.md) | `literature_pipeline/`、`scripts/`、`marker_statistics/`、`cell_marker/` | 文献发现、表格提取、名称映射、显著 Marker 与特征分析 |

推荐顺序为：准备 CL 本体 → 生成骨架和审核目录 → 文献 Marker 挖掘与名称标准化 → GO 语义/数据库证据 → 范围判定和精确关系审核。不同任务通过目录、Marker 和 GO 证据表交换数据，不相互复制整套实现。

`cl_go_linking/GO_link` 区分细胞特异、细胞通用和非直接人类细胞过程；精确细胞不在审核目录时保留未映射。没有检测到特异性信号不能自动证明过程通用。
