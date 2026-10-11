# 共享基础设施

`llm_api/` 是复用的 ECNU 模型客户端，`uhpo_common/paths.py` 是四模块的仓库路径定义。这里不存放科研任务实现、真实数据、密钥或运行结果。

根目录 `pyproject.toml` 注册所有跨任务 Python 包，使用 editable 安装保持实现位于各自任务目录。两个研究 agent 仍由各自 `pyproject.toml` 独立安装。
