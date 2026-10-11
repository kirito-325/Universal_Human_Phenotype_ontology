# CL–GO 关系链接与判定

| 目录 | 职责 |
| --- | --- |
| [GO_link](GO_link/README.md) | 当前 GO-BP 范围汇总、精确细胞链接、LLM 审核和完整性检查 |
| [GO_function](GO_function/README.md) | 人类 GO 功能基因导出 |
| `celltype_configure/` | CL 目录读取、SapBERT 语义召回、SCEA 富集与旧约束比较 |
| `web/` | 从结果生成审核页面 |
| `tests/` | 此任务的规则、证据、链接与数据契约测试 |

从仓库根目录安装 `.[scientific,test]`，实际 SapBERT 编码另装 `.[embedding]`。下列命令可从任意工作目录调用：

```bash
python -m GO_link.build_complete_scope --help
python -m GO_link.llm_adjudicate_direct_phenotypes --help
python -m GO_function.export_go_functions --help
```

完整范围汇总读取此前准备的语义、直接链接和审核证据表，并非从零下载数据的入口。全部前序证据准备后运行 `python -m GO_link.build_complete_scope --require-complete`。具体来源与范围定义见 [GO_link 方法说明](GO_link/README.md)。

细胞目录来自相邻 `cl_skeleton` / `cell_marker_mining`；Marker 导出代码位于 `../cell_marker_mining/cell_marker`。跨任务路径由 `shared/uhpo_common/paths.py` 及 CLI 参数明确指定。
