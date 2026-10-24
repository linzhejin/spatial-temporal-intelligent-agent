# T5 复合任务编题：选点后必须完成全程

## 发现与修复

在可见的 120 条编题草稿中，T5 的 `food_before_route`、`study_before_route`、`sport_then_food` 各 3 条原本只包含第一次地点推荐。虽然用户任务要求继续去终点，数据集却没有后续选择和路线，因此不能检验“按顺序走完全程”。根因在 `scripts/analysis/build_scenario_v2_authoring.py`：这三个分支只生成一个 `candidates` 轮次。

现在每题保留第一次候选，增加用户选定场所后的第二轮，并明确要求**从原起点经过已选场所到终点的全程步行路线**。`sport_then_food` 的第二轮同时明确指定吃饭地点，要求路线按起点→篮球场→吃饭地点排列。场所来自正式 POI 用途字段，首轮选择由独立图内距离参考的前四名产生；生成文件保存 POI、图、道路覆写与道路复核层 SHA-256。生成结果仍是可见开发编题，**未裁决为论文金标准**。

## 本轮可复核证据

| 文件 | SHA-256 | 观察 |
|---|---|---|
| `experiments/datasets/scenarios_test_v2_diverse_draft.json` | `0160f235dbd5246484f6849f713da50c8de87de9f4283868b7ebfe33889af2db` | 120 条草稿；第二次生成字节哈希完全相同 |
| `experiments/datasets/scenarios_test_v2_draft.json` | `fde90df4fb464527540b94f03514438b6fd1b29c6c04c97c73e9ba2607b109e7` | 保留原有空间锚点版本并同步完整轮次 |
| `output/experiments/T5_composite_reference_20261025.json` | `758cfd095b352cd416ef935d3b06f75d2bf7055b4e3354b641e527e26bd77e15` | 9 条选择均在首轮前四名；9 条两段路径图内连通；0 条端点接驳超过 30 米 |
| `output/experiments/scenario_split_T5_full_20261025.json` | `2426ef32e2b54df5bdc19048685112b86e1e2678c0cee3bba18ab8af09c9684a` | 60 开发题与 120 编题草稿指定字段无直接交叉；正式就绪仍为假，7 项阻碍 |
| `output/experiments/T1_diverse_reference_T5_full_20261025.json` | `459dcdf985762ed198a866996bfa41ffc8f722a6591eecf341ba3e9ffec93e5f` | 新草稿下 T1 仍为 24/24 图内连通 |
| `output/experiments/T2_diverse_reference_T5_full_20261025.json` | `cdcb563d102c1cda236de5fef2702da4b29fb01547390226d808745e9682e098` | 新草稿下 T2 仍标记 11 个接驳或排序边界 |
| `output/experiments/T2_uncertainty_T5_full_20261025.json` | `c9063b73045422cb78aae91751daff59b09fbb74c6dd7bea68d718db4453e2ff` | 15 条图内可判、9 条保持末端接驳未知 |
| `output/experiments/p1_t5_full_regressions_20261025.xml` | `076d463a4744560725385903efa4c0baefd786f4aa8d22db8189481de3744d4a` | 76 项相关回归通过 |

复现命令：

```powershell
python -m scripts.analysis.build_scenario_v2_authoring --diverse-anchors --output experiments/datasets/scenarios_test_v2_diverse_draft.json
python -m scripts.analysis.build_t5_composite_references_v2 --draft experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/T5_composite_reference_20261025.json
python -m pytest tests/test_build_scenario_v2_authoring.py tests/test_build_t5_composite_references_v2.py tests/test_scenario_split_audit.py tests/test_scenario_v2.py tests/test_scenario_protocol.py -q
```

报告输出默认拒绝覆盖既有文件；复跑参考报告时须换一个输出路径。图内可达、前四名归属与 30 米接驳检查仅说明编题不明显自相矛盾，不能证明路线现场可走、Agent 已正确回答或用户选点交互已在线上完成。旧版 T1/T2 报告仍保留，但其 `draft` 文件哈希不再对应最新编题草稿，不得混用。

## 接下来的闸门

用隔离服务对这些完整两轮任务做实际开发回放；同时核对 T3 澄清槽位和 T4 多轮修改的语义与来源。正式测试集仍需另行封存、逐题来源裁决和有效输入快照。
