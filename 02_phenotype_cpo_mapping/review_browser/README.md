# HPO / MPO–CPO 联合审核

`build_database.py` 将两个任务的旧关系表及继承表装入审核索引；`server.py` 提供 `/hpo` 和 `/mpo` 页面，分别展示自身证据、继承证据及来源。

从仓库根目录运行：

```bash
python 02_phenotype_cpo_mapping/review_browser/build_database.py --help
python 02_phenotype_cpo_mapping/review_browser/server.py --help
```

默认输入分别位于 `hpo_cpo/data/hpo_cpo/` 与 `mpo_cpo/data/mpo_cpo/`；索引保存在本任务的 `data/`，不提交 Git。可用 CLI 参数覆盖本体及其他输入。

这是既有 TSV 数据的审核工具；新 agent 输出尚需显式验证转换。静态页面源码和相关测试均与工具放在本目录。
