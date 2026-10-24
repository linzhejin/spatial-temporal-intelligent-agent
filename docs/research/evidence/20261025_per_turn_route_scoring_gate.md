# 2026-10-25 多轮路线判分与来源核对

## 发现与修复

复核 T3–T5 的逐轮评分入口时，发现 `scenario_v2.score_case` 原来对同一对话中的每次路线输出都读取 `case.route_reference`。T4 题第二轮修改终点后，如果观测结果仍给第一轮的边序列，却声称已到新终点，旧图审计仍可能按旧终点判为通过。聚焦反例在修复前确实出现 `task_success=true`。

现有合同调整为：单轮路线保留 `route_reference`；多轮对话中的图计算路线必须使用 `route_references`，按零基用户轮次标出每个路线轮次的起点、途经点和终点节点。缺轮、额外旧轮或不完整节点会被数据校验拒绝。评分只读取本轮参考；多轮仍仅有旧单份参考时，该轮图审计不给通过证据。

另发现实验回放器此前即使收到路网与 POI，也未核对题库预填的路线参考节点是否对应题目地点。新增来源核对：按本轮方式和硬约束，从当前图与 POI 独立计算起点、途经点、终点的吸附节点，与题库参考比较。预填参考指向另一地点时，回放在写结果目录前拒绝。正式图内判分需要冻结 POI 的 SHA-256，并核对输入和就绪快照；开发回放在同时提供路网与 POI 时也执行该核对。清单新增 `route_reference_turns_verified_against_sources`，用于区分核对过的路线轮次数。

## 验证范围

- 先运行三个红灯反例：正确的第二轮参考无法被使用；错误地重复第一轮路线却得到任务通过；缺逐轮参考的数据未被拒绝。另以错误 POI 节点重放，原回放器未拒绝。
- 修复后运行 `python -m pytest tests/test_scenario_v2.py tests/test_scenario_protocol.py tests/test_scenario_split_audit.py tests/test_scenario_live_adapter.py tests/test_scenario_draft_audit.py tests/test_scenario_adjudication_import.py tests/test_development_graph_reference.py tests/test_candidate_reference.py tests/test_multiturn_route_reference_audit.py tests/test_route_reference.py -q`：**143/143 通过**，包含补起点后第二轮路线的正例。
- `python -m ruff check experiments/scenario_v2.py experiments/scenario_runner.py tests/test_scenario_v2.py`：通过。
- 正确来源的合成回放清单报告 `route_reference_turns_verified_against_sources=1`；污染终点参考的反例在创建结果目录前报错。

## 尚未完成的条件

本次没有把可见 T3–T5 草稿直接升级为正式金标准。每轮用户意图、澄清条件、天气或路况输出、路线保留的客户端证据仍须按实际来源裁决；图内节点与 POI 一致也不能证明道路现场可通行。新的数据契约用于以后独立裁决后的正式实验，现有 120 条草稿仍为 `draft`，正式拆分的 7 类阻碍保持不变。
