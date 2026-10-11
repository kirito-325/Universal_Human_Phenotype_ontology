# CPO 构建

构建流程将细胞目录、单细胞文献 Marker 和 GO 生物过程证据组织为可追溯的细胞表型层。

| 步骤 | 代码 | 主要产物，均在本地生成 |
| --- | --- | --- |
| CL 骨架与扩展目录 | [cell_lineage](cell_lineage/README.md) | 骨架、层次目录、保留/排除策略 |
| 文献发现与 Marker 提取 | [marker_literuture](../marker_literuture/README.md) | 论文元信息、补充表格、标准化 Marker |
| 细胞名称归一与特征分析 | `marker_literuture/scripts/` 与 `marker_statistics/` | CL 名称映射、显著 Marker、证据分数、覆盖审计 |
| GO–CL 连接 | [GO_link](../CPO_generize/GO_link/README.md) | 全量 GO-BP 范围判定、精确细胞链接、通用过程 |
| 功能基因与 Marker 导出 | [GO_function](../CPO_generize/GO_function/README.md)、[cell_marker](../CPO_generize/cell_marker/README.md) | GO 功能基因及细胞 Marker 证据 |
| 四阶段整合 | [UHPO](../CPO_generize/UHPO/README.md) | 既有流程封装与审计入口 |

GO 链接区分 `cell_type_specific`、`cell_general`、`not_direct_human_cell_process`。精确细胞不在目录时保留未映射状态，不用邻近细胞代替。没有找到细胞特异性信号不构成“通用过程”的证据。

`celltype_configure/` 中的语义召回和 SCEA 富集保留为证据组件；旧的宽泛 CL 子树约束不是当前 GO 精确链接主输出。`CPO_generize/UHPO` 是既有整合代码，其历史 HPO/MPO 挖掘不替代最新 agent。

## 目录重建

准备 `cell_lineage/cell_ontology/cl-basic.obo` 后，从仓库根目录运行：

```bash
python3 CPO_construction/cell_lineage/scripts/build_cell_catalog_skeleton.py
python3 marker_literuture/scripts/build_expanded_cell_catalog.py
```

文献目录构建器默认复用本仓库的骨架代码，可用 `--source-root` 或 `CELL_LINEAGE_SOURCE_ROOT` 指定其他实现。真实 CL/GO 数据、审核目录和文献数据库不随代码上传；详见 [数据指南](../docs/data.md)。
