# P2 产品 Agent 开发态真实链路试跑

日期：2026-10-25。隔离库：本机 `127.0.0.1:55439/whu_agent_experiment_test`。本记录只评估可见开发题；未使用未见测试集，未写生产库或发布路况。当前源代码以本记录所在提交为准，原始报告保存在本机忽略目录 `output/experiments/`。

## 小样本结果

| 已见题 | 链路和观察 | 模型请求 | 限制 |
|---|---|---:|---|
| `seed_mode_01`，T1 | 产品 Agent 输出路线，调用 `plan_route`；49 条边的图内核验通过 | 1 | 不证明现场可通行 |
| `seed_place_01`，T2 | 输出地点候选，调用一次地点工具 | 0 | 不能据此估算推荐准确率 |
| `v2_dev_clarify_01`，T3 | 输出澄清，无工具调用 | 0 | 仅一条已见表述 |
| `seed_compound_01`，T5 | 固定模拟天气下输出路线，调用两个工具 | 1 | 天气非实时观测 |
| `draft_multi_turn_revision_02`，T4 | 真实 Flask API、持久化任务、LangGraph worker 两轮均完成；同一起点 `poi_449`，终点依次为 `poi_015` 和 `poi_418`，避台阶要求保留；路线分别 13、33 条边 | 2 | 单条已见任务，不代表多轮成功率 |

T4 首次投影把路线状态中的“武汉大学图书馆总馆”当作未知名称，而实际结果提供了正式库中的“武汉大学图书馆(总馆)”。这是**评测投影**漏识别，不是产品路线终点错误。投影现在可用正式结果中的规范名称补全；若状态与结果解析为不同 POI，明确标记身份冲突，不能默默取其中一个。

## 发现并修复的真实任务缺陷

从可见编题草稿取 `v2h_T3_destination_weather_1`，显式降为开发题：首轮“带我去凌波门东湖观景点，起点等我补充”，次轮“我在武汉大学樱园，查天气后再带我过去”。修改前真实模型首轮只用普通聊天文字问起点，任务没有保存结构化目的地；工作流把首轮记为失败，次轮遗忘目的地并追问。原始报告的状态为 `failed/information → needs_input/clarify`，消耗 1 次模型请求。

规划器现对**明确、唯一解析的目的地且没有起点**生成结构化 `place_request`，保存目的地 ID、方式和已有硬约束，再询问起点。相同两轮任务复跑为 `needs_input/clarify → completed/route`；首轮保留 `poi_141`，次轮从 `poi_152` 到 `poi_141`，18 条路线边，`get_weather` 和 `plan_route` 均成功，用户可见输出包含路线与固定天气。第二次运行由确定性分支处理，模型请求 0 次。图内边验证通过不等于地点入口或道路现场核实。

## 复现与证据

```text
python -m scripts.analysis.probe_product_live_case --env-file <本机私有配置文件> --test-database-url postgresql://postgres@127.0.0.1:55439/whu_agent_experiment_test --dataset output/experiments/dev_t3_context_case_20261025.json --case-id v2h_T3_destination_weather_1 --weather-fixture experiments/fixtures/development_weather_clear_v1.json --output output/experiments/product_live_t3_context_weather_fixed_20261025.json --allow-model-call

$env:TEST_DATABASE_URL='postgresql://postgres@127.0.0.1:55439/whu_agent_experiment_test'; $env:DATABASE_URL=''; python -m pytest tests/test_planner.py tests/test_scenario_live_adapter.py tests/test_probe_product_live_case.py tests/test_scenario_live_integration.py tests/test_context_builder.py tests/test_task_planner.py tests/test_route_comparison_workflow_status.py -q --junitxml=output/experiments/p2_product_live_regression_20261025.xml
```

回归 121/121 通过，含 6 项隔离库集成检查；上述修改文件的 Ruff 检查通过。回放脚本要求显式允许模型调用、只收开发集、验证独立数据库，报告省略原始对话、密钥和完整路线几何。天气题必须指定固定夹具。此脚本是开发诊断入口，不能替代三条件正式实验。

| 原始文件（均为 `output/experiments/` 下） | SHA-256 |
|---|---|
| `product_agent_seed_mode_01_20261025.json` | `63674317619e472358d7a705325d8c81a181d6deb7863063c02c51c188a167c5` |
| `product_agent_seed_place_01_20261025.json` | `7fe573dcb542690bd81d47cacf994ec98e3d158da0fe3469b0028e92b89b340f` |
| `product_agent_v2_dev_clarify_01_20261025.json` | `171b129fed4b3fc6913891d3b332e3f5460f2f10913a0c4743eb0e5c290b100b` |
| `product_agent_seed_compound_01_20261025.json` | `d4c65a0e9f731d4e4f5d545285b60722460e0c194b7fa7f5b4f3d1f2df4418c7` |
| `product_live_draft_multi_turn_revision_02_projectionfix_20261025.json` | `a1c51ad96672f77fb54e2397b533d8cf4997117f90ec39abbae6139eb64e7081` |
| `product_live_t3_context_weather_20261025.json`（修复前） | `de30bc9c5888f4304c76193bbe4f52974fc4ca2637e0c50981775b3ea898550f` |
| `product_live_t3_context_weather_fixed_20261025.json` | `9b5d026260a548fc23e9119060986d771748edb95fae1e66553e9f8969de4023` |
| `p2_product_live_regression_20261025.xml` | `10f7005dc8f5ff7bd2549f4ca60a5efeea504c9311bcb118291bb5b5e919b09e` |

开发题临时副本哈希 `2a6a014f1595bcbc0efc4b02d80ac64c5201d09aee7933b95a7e8dda06c5947c`；固定天气夹具哈希 `fab71cf975d02313e09270500b50e236e6b5d4ce89daeed834bfad8eb7bc48b6`。该临时副本已被开发使用，其同源场景不能再称未见留出。

## 下一闸门

补同题普通工具 Agent 与固定 Harness 的多类型开发回放，按实际 HTTP、响应模型、工具、token、耗时和失败原因核对公平条件；按各题型样本估算正式费用上限。P1 的 9 条 T2 接驳边界仍未知，120 条草稿未独立裁决；因此 `formal_ready=false`，本轮结果不能写作论文效果。
