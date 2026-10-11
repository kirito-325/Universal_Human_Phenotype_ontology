# 同步来源与取舍

本次从原工作区导出当前代码到独立 Git 仓库，未改写原项目。GitHub 目标仓库同步前为空；未导入其他本地仓库的历史，以免带入旧凭据、数据或多余版本。

详细来源映射见 [source_inventory.json](source_inventory.json)。该文件记录初始筛选来源，不是整理后逐文件哈希清单。

保留当前 `src/`、相关脚本、测试、schema、配置、上游许可、小型合成示例和结果摘要。补齐了原文献目录构建器引用的 CL 骨架源码，以及无训练基准所依赖的较新独立 CellPhenoRank 代码。

不收录原项目的 `.env`、远程机器配置、LaunchAgent plist、数据库、运行轨迹、文献正文、数据快照、权重、缓存、备份、旧 HPO_link v1–v6 代码和旧映射 v1/v3。后两类只作为既有数据接口在配置中出现。结构嵌入仅保留晚融合 v2 快照，移动到 `legacy/MPO_HPO_mapping`。

为减少对科研逻辑的影响，未统一改写算法或包名。可读性调整集中于模块导航、主线/历史状态、目录补齐、密钥与路径清理、先导运行入口和仓库检查。原绝对路径替换为明确占位符；主线目录构建器使用仓库内路径；先导脚本使用正常安装环境和公开示例配置。语义召回中的 Torch/Transformers 改为在实际编码时导入，纯规则检查无需加载模型依赖。

基准准备脚本及 schema 保留在 `CPO_generize/mission3/skills/build-rare-disease-mouse-benchmark/` 的 scripts、assets 和 references 子目录，供既有测试与工具调用；没有执行其中的文献调研。较新 CellPhenoRank 的合成输入从原 `data/example` 移到公开的 `mission3/examples`，配置和测试引用同步更新。

第三方 agent 组件保留 `UPSTREAM.md`、`THIRD_PARTY_NOTICES.md` 及包内 `upstream/LICENSE`。上游数据许可不因代码整理而改变。
