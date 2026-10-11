# MPO-HPO Mapping v2

> **方法落后，已弃用，不采用，待更新。** 仅保留历史结构嵌入方法代码，详见 [方法状态](STATUS.md)。本次不附带数据、权重或结果；下文为原方法说明。

本目录实现语义、CL/GO 结构和机制重叠的可解释晚融合 MPO-HPO 检索。v2 与
`04_MPO_HPO_mapping` 独立，复用相同原始数据路径，但不读取或覆盖 v1 结果。

## 为什么做 v2

v1 在每个 prototype 内提前拼接 `[CL, GO, channel, semantic]`，随后统一经过
欧氏 MLP 和余弦相似度。该设计允许分类头忽略结构，而且没有直接使用
Poincare 距离。v2 改为：

1. 语义塔和结构塔完全分离；
2. CL/GO 始终保存为 Poincare 球坐标；
3. 直接计算 CL/GO Poincare 距离；
4. 使用非负且和为 1 的三路融合权重；
5. 增加结构专用对比损失和语义 modality dropout；
6. 对 Exact、expert broader、counterfactual、synthetic 来源做均衡采样和校准；
7. 输出每一路分数、融合权重和标准化关系分类系数。

## 模型

每个 HPO/MPO 最多保留 4 个高置信度 `(CL, GO, channel)` 机制 prototype：

```text
语义向量 -> 语义塔 ---------------------------- semantic score ----┐
                                                                  │
(CL,GO,channel) -> 结构塔 ---------------------- structure score ---+-> softmax 非负融合
                                                                  │
CL/GO Poincare 球坐标 -> 直接双曲距离 ----------- Poincare score ----┘
```

Exact 和 Ancestor 各有独立融合权重。训练目标为：

```text
L = L_fused + structure_aux_weight * L_structure + 0.5 * L_ordering
```

默认 `structure_aux_weight=0.35`，语义 dropout 为 `0.25`。分类与检索代码分别见：

- `src/mpo_hpo_mapping/model.py`
- `src/mpo_hpo_mapping/retrieval.py`
- `src/mpo_hpo_mapping/pipeline.py`

## 三种模式

- `late_fusion`：语义、学习结构、直接 Poincare 三路融合；
- `semantic_only`：屏蔽结构索引、Poincare、机制重叠和结构校准特征；
- `structure_only`：屏蔽语义索引、语义塔和别名特征。

单模态模式不仅改变神经网络权重，也会屏蔽关系校准、候选召回和最终排序中的
对应旁路，防止消融答案泄漏。

## 运行

快速连通性验证：

```bash
PYTHONPYCACHEPREFIX=/tmp/cpo_v2_pycache \
LOKY_MAX_CPU_COUNT=8 \
TOKENIZERS_PARALLELISM=false \
python3 scripts/run_pipeline.py \
  --smoke \
  --evaluation-only \
  --output-dir results/smoke_late_fusion_v2
```

正式晚融合模型：

```bash
PYTHONPYCACHEPREFIX=/tmp/cpo_v2_pycache \
LOKY_MAX_CPU_COUNT=8 \
TOKENIZERS_PARALLELISM=false \
python3 scripts/run_pipeline.py \
  --fusion-mode late_fusion \
  --evaluation-only \
  --output-dir results/full_late_fusion_v2
```

完整受控消融：

```bash
PYTHONPYCACHEPREFIX=/tmp/cpo_v2_pycache \
LOKY_MAX_CPU_COUNT=8 \
TOKENIZERS_PARALLELISM=false \
python3 scripts/run_late_fusion_experiments.py \
  --output-dir results/late_fusion_ablation
```

消融包含五组：

1. 纯语义；
2. 纯结构/机制；
3. 晚融合 + 常量结构；
4. 晚融合 + 真实 Poincare；
5. 晚融合 + 节点置换 Poincare。

各组固定相同的数据划分 seed、语义 seed、训练预算和测试队列。脚本会校验专家
样本、父类样本、语义 NPZ 和机制边 TSV 的 SHA-256，还会校验非实验超参数、
训练/测试查询隔离和单模态旁路屏蔽，并生成 `index.html`、`arm_metrics.tsv` 和
`experiment_report.json`。任一审计失败都会终止，不输出看似可比的结果。
正式消融中 SapBERT 只编码一次，后续实验组读取同一冻结 NPZ，避免重复计算和
浮点实现差异；引用文件的 SHA-256 会写入各组 manifest。

2026-08-03 正式全量结果位于
`results/late_fusion_ablation_full_20260803/`。`formal_summary.html` 按之前的
专家 Exact、专家 Broader、构建综合三块口径展示结果，`paired_comparisons.tsv`
提供同查询配对 bootstrap 95% CI 和 Recall 的精确 McNemar 检验。

## 输出

每个模型目录包含：

- `evaluation_summary.json`：四类金标准 Recall/MRR；
- `evaluation_cases.tsv`：逐查询目标和最佳名次；
- `training_manifest.json`：输入哈希、参数、融合权重和运行环境；
- `model/dual_head_encoder.pt`：晚融合双头网络；
- `model/training_loss.tsv`：融合损失、结构损失、采样量和每 epoch 权重；
- `model/fusion_attribution.json`：三路融合权重、关系校准特征系数及来源权重；
- `mpo_hpo_mapping_final.tsv`：非 `--evaluation-only` 时生成的 top-k 映射。

逐条检查映射及语义/结构/Poincare 分量：

```bash
python3 visualization/server.py --result-dir results/smoke_mapping_v2 --port 8765
```

候选结果新增以下可解释字段：

- `exact_structure_score` / `ancestor_structure_score`；
- `exact_semantic_tower_score` / `ancestor_semantic_tower_score`；
- `poincare_score`；
- `mechanism_jaccard`；
- `exact_embedding_score` / `ancestor_embedding_score`：非负三路融合分数。

## 测试

```bash
PYTHONPYCACHEPREFIX=/tmp/cpo_v2_pycache \
python3 -m unittest discover -s tests -v
```

注意：v2 的 `hierarchy_embeddings.npz` 包含
`coordinate_system=poincare_ball`。v1 的切空间存档没有该标记，不能作为 v2 的
`--hierarchy-reference`，流水线会主动拒绝。
