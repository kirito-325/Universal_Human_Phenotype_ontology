# CL 骨架与细胞目录

本目录补齐文献 Marker 流程原先从外部工作区加载的 CL 目录构建代码。

`scripts/build_cell_catalog_skeleton.py` 构建紧凑骨架，`build_expanded_cell_catalog.py` 构建扩展目录，`build_cl_lineage_catalog.py` 保留本体层次上下文。其余脚本用于 slim、简化分类、树形展示及 Marker 核验。

默认原始 CL 文件为 `cell_ontology/cl-basic.obo`；生成物写入本目录下的 `cell_ontology/`。该目录是外部本体与派生产物，不提交 Git。文献流程的功能性 T 细胞粒度策略由相邻 `cell_marker_mining/scripts/build_expanded_cell_catalog.py` 注入，使用该入口生成下游采用的目录。

所有人工骨架与过滤策略都位于源码中；它们是项目的目录政策，不声称等同于 CL 官方原生层次。
