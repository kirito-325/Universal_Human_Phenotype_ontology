# 下游任务与验证

| 任务 | 代码 | 评估对象 | 当前定位 |
| --- | --- | --- | --- |
| 罕见病候选基因重排 | [mission3 / CellPhenoRank](../mission3/README.md) | 冻结疾病队列、基因排名、单细胞供体层验证 | 较新下游主实现；含参数无关重排及多种基线 |
| 具体小鼠模型优先级 | [CPO_generize/mission3](../CPO_generize/mission3/README.md) | MGI 基因型，而非单个基因 | 支持 within-gene / all-models 及表型/CPO/遗传机制消融 |
| CPO 矩阵与预测 | [CPO_generize/mission4](../CPO_generize/mission4/README.md) | HPO–MPO 关系、疾病模型和基因 | 当前 v2 实现；旧 v1 保留为比较接口 |
| 无训练语义/CPO 对照 | [MPO_HPO_mapping_v4](../CPO_generize/MPO_HPO_mapping_v4/README.md) | 固定队列与预注册对照 | 探索验证；现有结果不支持总体 CPO 收益声明 |

`mission3` 与 `CPO_generize/mission3` 不应混为同一版本。前者代码包名为 `cellphenorank`，后者为 `mission3`。

## 首次运行

无真实数据时，使用 `CPO_generize/mission3/examples` 的小型合成输入检查流程。其 README 的 Quick start 给出完整命令；输出写到 `outputs/example`，不提交 Git。这些例子不是生物学发现。

真实评测需要冻结队列、疾病 HPO、MGI 模型与 MP、表型映射矩阵、HPO/MPO-CPO 关系、CL/GO、本体方向规则及独立 gold。先检查各模块配置，并将 `/path/to/workspace`、`/path/to/home` 改为本地资源位置。

最新 agent 的丰富结果与旧下游 TSV 契约不同。应先显式设计并验证转换器，保持模型来源、方向、直接/机制线索、条件性和覆盖状态，再重跑评测。本次整理不宣称已完成这一集成。
