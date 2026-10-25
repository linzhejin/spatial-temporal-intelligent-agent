# 2026-10-25 T3 已见草稿回放与起点解析修正

## 范围

依照桌面《珞珈智行_论文实验推进方案_最终执行版_20261023.docx》的 P0/P1 顺序，在独立本机 PostgreSQL 测试库中用实际 Flask API、LangGraph worker 和持久化任务状态回放 24 条 T3 多轮/追问编题草稿，共 27 轮。固定天气；禁止外部模型调用，检验不依赖模型时的确定性链路。原始输入是**已经用于开发的可见草稿**，不是留出测试集或论文成功率。

输入 `experiments/datasets/scenarios_test_v2_diverse_draft.json` 的 SHA-256：`b7e0971bba4c21591cbe3832d0cb70c2758819ec87cb5f8fcced8265a0d28bcd`。运行前研究分支提交为 `50330ee5768f06b4b60fdd0774ffb81c01107c9f`；本文件记录的是随后的修正及其验收，正式冻结仍需另做干净提交与来源裁决。

## 发现与处理

| 回放观察 | 定位与修正 |
|---|---|
| 以排球场、游泳池、食堂为起点时，起点名称可被误认成本轮运动或餐饮需求 | `resolve_activity_query` 先去掉明确起点子句，再识别真正的活动；同时覆盖“出发去打球，别走台阶”和无逗号写法。 |
| 已给起点、未给目的地的 3 条普通路线请求在**禁止模型的诊断条件下**失败；另 3 条平坦路线误走地点候选 | 在已有地点解析之后增加缺终点追问，保存起点、出行方式和硬约束；用户回答终点后保留原请求。这里不能推断生产模型可用时也必然失败。 |
| 参数层曾传入平坦权重，但实际工具仍按纯最短路线执行 | 工具为防止模型擅自制造权重，会按查询文本判断策略；第二轮仅有终点名称，故丢失原偏好。续接时显式传入已有策略 `flat`/`scenery`，以路由策略模块的既定权重为准，并把完整请求交给工具保留来源。 |
| “想打什么球”实际追问正确，任务内部槽位名为 `activity`，而实验草稿标为 `sport` | 仅在回放投影中依据实际展示的追问文本记为 `sport`；通用“想做哪种运动”仍记为 `activity`。产品任务状态和用户问题均不改写。 |

修正前的原始回放已保存为 `evidence/20261025_p1_t3_visible_replay_before.json`，SHA-256：`2fa9be45779de5328bc78756554f98966ec567838428342aad31f104968a368c`。最终修正后回放为 `evidence/20261025_p1_t3_visible_replay_after.json`，SHA-256：`717279a8831823cd1203c1e1ac022718adfa90f6b7511ecc3a375448c6533b8f`。运行脚本 `scripts/analysis/probe_visible_t3_replay.py` 会写入忽略的本机输出目录；运行前需设置 `TEST_DATABASE_URL` 指向隔离库。归档的 JSON 是删减后的可观察任务字段，不含模型提示或凭证。

## 检查结果与边界

- 修正后 24/24 个案例的 27/27 轮在**状态、输出类型、追问语义槽位**这三项上与可见草稿预期一致；其中三条天气后续请求为两轮，第二轮返回路线。本项没有把终点接驳或现场通行判作已验证。
- `python -m pytest tests/test_planner.py tests/test_task_planner.py tests/test_poi_activity_taxonomy.py tests/test_poi_activity_recommendations.py tests/test_scenario_live_adapter.py tests/test_multi_via_agent.py tests/test_multiturn_route_reference_audit.py tests/test_context_builder.py tests/test_route_strategy_persistence_unit.py -q`：194/194 通过；`python -m ruff check` 对本次 5 个改动 Python 文件通过；`git diff --check` 通过。新增回归分别先观察失败、再实施修正并观察通过。
- 两条“起点已知 → 追问终点 → 用户回答卓尔体育馆”的直接真实工具试算得到路线，`applied_weights` 分别为平坦 `{distance: 0.5, slope: 0.4, scenery: 0.1}`、风景 `{distance: 0.5, slope: 0.1, scenery: 0.4}`，与当前统一路由策略一致；此前仅检查传参的测试未能发现这项错误。
- 此次 24 条回放只比较上述三项离散字段，**不等于**全部任务质量或用户体验通过；两条策略路线的地图画线和现场通行未验收。模型被故意禁用，不能据此给真实模型的响应时间或费用下结论。
- T3 澄清语义仍缺逐题独立裁决；120 道草稿仍是已见开发材料，正式 `formal_ready=false`。下一步先对 T3 多轮追加起点、约束和最终边序列的独立观察，再补其他 T 类的独立参考与真正留出集。
