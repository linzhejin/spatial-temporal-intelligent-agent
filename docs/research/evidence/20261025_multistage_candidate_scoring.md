# 2026-10-25 多阶段地点候选与全程路线开发验收

## 问题与改动

原 T5「先打篮球，再找附近吃饭处」编题只有一次球场候选，下一轮直接由用户同时报出球场和餐饮地点，跳过了以选中球场为起点检索餐饮的阶段。独立评分器和回放器也只支持一题一份 `candidate_spec` 与一轮候选参考。这样无法评测第二次推荐是否真正按球场位置排序。

现把这一场景的 3 条可见草稿改为三轮：篮球场候选 → 用户选球场后，按该球场附近检索正餐候选 → 用户选餐饮地点后，从原起点依次经过球场到餐饮地点。复合任务的其他 6 条草稿仍为「候选 → 全程路线」两轮。每轮候选使用独立 `candidate_specs`，按零基轮次标明活动用途与距离起点；旧的单候选轮次仍可用 `candidate_spec`。回放器从同一冻结图和 POI 逐轮重算，不复用首轮名单，并在清单记录重算轮次数。路线轮次继续使用独立的逐轮路线参考和有序途经节点审计。

## 可复核结果

| 产物 | SHA-256 或观察值 |
|---|---|
| `experiments/datasets/scenarios_test_v2_diverse_draft.json` | `bf6b670735fe3a0e03c3555037303a33d9beceb8579314dd30ddac0b61a294ce`；120 条仍全部为 `draft` |
| `output/experiments/T5_composite_reference_three_turn_20261025.json` | `7999639d56a9a20aefcf5b330466ecf7f5b4dc7ce3816de3c78775eaaf8b4aed`；9/9 首轮选点在前四名，3/3 第二轮餐饮选点在以球场为起点的前四名，9/9 图内两段路线可达，0 条端点接驳超过 30 米 |
| `output/experiments/multiturn_routes_three_turn_20261025.json` | `7f94b178d9a52358f7a7b44af1b9d3086817fcd572d342324990e6c078b5bc00`；T3–T5 共 72 个路线轮次，72 个图内可达 |
| `output/experiments/scenario_split_three_turn_20261025.json` | `ae617ac45166cdcf7da46c3128305340fff643cb33e2ff7f92989c836121497d`；开发 60、可见草稿 120、`formal_ready=false`，仍有 7 类阻碍 |

先运行聚焦反例，确认旧评分器拒绝 `candidate_references` 且旧编题只有两轮；修复后运行 `python -m pytest tests/test_scenario_v2.py tests/test_scenario_protocol.py tests/test_scenario_split_audit.py tests/test_scenario_live_adapter.py tests/test_scenario_draft_audit.py tests/test_build_scenario_v2_authoring.py tests/test_build_t5_composite_references_v2.py tests/test_build_scenario_v2_development.py tests/test_multiturn_route_reference_audit.py tests/test_candidate_reference.py tests/test_poi_activity_recommendations.py tests/test_route_reference.py -q`，**185/185 通过**。其中合成回放真实地从不同起点重算两轮候选，并核对选点后的有序路线与清单计数。修改文件的 Ruff 检查通过。同一生成命令第二次写入忽略目录，120 题文件与已提交草稿字节完全一致，SHA-256 仍为 `bf6b6707…a294ce`。

## 边界与下一步

这只验证编题流程、图内参考和离线评分器。篮球场及餐饮 POI 的末端接驳是图上估计，未做现场通行核查；没有把 3 条草稿交给三种 Agent 真实运行，也没有独立裁决全部 120 题。产品可能在第二轮同时给出到球场的部分路线和餐饮候选；真实回放时须检查观察投影能否公平记录这种混合输出，不能仅因附带部分路线就判为错，也不能用部分路线代替最终全程。旧 `20261025_t5_full_task_authoring.md` 和依赖旧草稿哈希的报告保留为历史，不能混用作当前版本的依据。下一步用已见开发题实跑三阶段对话，核对候选、上下文、全程路线及超时；正式盲测仍须另建来源隔离的未见集。
