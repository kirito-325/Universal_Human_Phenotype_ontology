# 术语与 CPO 映射

当前主线从独立文献调研中生成 `(CL, GO, direction)` 关系，并保存来源、科学解释、适用条件及实验支持。

| 模块 | 输入与研究单位 | 输出与边界 |
| --- | --- | --- |
| [hpo-agent](hpo-agent/README.md) | `HP:` 人类表型术语；疾病关联作为背景 | 人类细胞异常关系；动物证据保留真实物种和外推条件 |
| [mpo-agent](mpo-agent/README.md) | `MP:` 术语与 MGI 精确注释基因型；逐模型研究 | 模型关系及 MP 级机制组；不按基因名合并不同模型 |
| [aligment_test](aligment_test/README.md) | 5 对人工精确映射、3 对层次映射、2 对探索候选 | 独立调研的先导输入准备、执行与状态报告；尚非成熟映射评测 |

两个 agent 是独立 Python 包，共享依赖版本但不相互导入。各自的 `src/`、`tests/`、`schemas/`、`examples/`、`pyproject.toml` 是当前实现，未收录旧版本归档。

## 本地准备

从仓库根目录将 `.env.example` 复制为 `CPO_mapping/.env` 并在本地配置。单包运行时使用 `--env-file ../.env`。索引放在各自的 `data/`，运行结果放在各自的 `runs/`，都不提交到 Git。

依次准备 CL/GO 索引、HPO 注释背景和 MP/MGI 基因型背景，命令见两个 agent 的 README。不要将先导配对关系、参考答案或候选机制放入 agent 的调研输入。

本次整理去除了先导脚本对某台机器的 uv 缓存和旧运行目录的依赖。安装依赖后从正常 Python 环境执行；HPO 配置默认使用 `hpo-agent/examples/settings.json`。这意味着新运行不会自动复现原本未公开的历史预算设置。
