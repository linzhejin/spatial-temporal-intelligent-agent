# P1 多轮回放链路：开发态验收

日期：2026-10-25。范围：隔离 PostgreSQL、真实 Flask API、任务持久化、LangGraph worker 与回放投影。Agent 回答和天气由确定性夹具注入，**未调用付费模型或实时天气服务**；合成边 ID 不证明路线通行。

## 本轮修复

- 当目的地已被选中、系统只等待起点时，正式任务状态把目的地存在 `place_request.selected_poi_id`。回放投影先前只看路线状态，因而把已知目的地记为空；现仅在 `clarify` 且 `pending_slot=start` 时读取这个稳定 POI ID。未知 ID 仍不能投影为有效地点。
- 第一轮尚未规划路线时，T3 的编题预期不再填写 `shortest` 策略；这避免用未发生的算路结果当作金标准。
- 两份 120 题编题草稿按新规则重建，仍全部标记 `draft`。

## 运行与边界

隔离测试库为本机 `127.0.0.1:55439/whu_agent_experiment_test`，由 `scripts/dev/start_test_postgres.ps1` 启动；测试进程显式设置 `TEST_DATABASE_URL`，清空此专用库中的测试表。未连接生产数据库。

| 检查 | 结果 | 不能证明的事 |
|---|---|---|
| 真实 API／worker 多轮回放 | 6 项通过，其中新增“先给目的地、后补起点并查天气”和“已有路线后只问天气” | Agent 对自然语句的真实解析精度、天气真实性、浏览器是否保留画线 |
| P1 聚焦 Python 回归 | 159 项通过，含上述 6 项数据库集成 | 正式留出集的任务成功率 |
| 开发／草稿拆分审计 | 60／120；指定字段无直接重合；`formal_ready=false`，7 项阻碍 | 草稿是未见题或独立金标准 |
| 新草稿图内参考 | T1 24／24 连通；T2 11／24 保留接驳或名次疑问；T5 选点与全程图内审计 9／9 | 现场道路通行和地点入口均已核验 |

T2 进一步用独立审计脚本核对前四名：24 题中 15 题在当前图和容差内可判，9 题因末端接驳缺乏独立证据仍为 `unknown`。原先 11 个编题预警里有 2 题在此次图内规则下可判；这不会把其余 9 题自动改成通过，更不能证明地点实地入口。

运行命令：

```text
powershell -ExecutionPolicy Bypass -File scripts/dev/start_test_postgres.ps1 -Database whu_agent_experiment_test
$env:TEST_DATABASE_URL='postgresql://postgres@127.0.0.1:55439/whu_agent_experiment_test'; $env:DATABASE_URL=''; python -m pytest tests/test_scenario_protocol.py tests/test_scenario_v2.py tests/test_scenario_split_audit.py tests/test_scenario_draft_audit.py tests/test_scenario_adjudication_import.py tests/test_scenario_live_adapter.py tests/test_scenario_live_integration.py tests/test_build_scenario_v2_development.py tests/test_build_scenario_v2_authoring.py tests/test_route_reference.py tests/test_candidate_reference.py tests/test_task_planner.py tests/test_context_builder.py tests/test_route_comparison_workflow_status.py -q --junitxml=output/experiments/p1_live_chain_broad_20261025.xml
python -m experiments.scenario_split_audit --development experiments/datasets/scenarios_dev_v2.json --test experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/scenario_split_live_chain_20261025.json
python -m scripts.analysis.build_route_references_v2 --draft experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/T1_route_reference_live_chain_20261025.json
python -m scripts.analysis.build_candidate_references_v2 --draft experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/T2_candidate_reference_live_chain_20261025.json
python -m scripts.analysis.build_t5_composite_references_v2 --draft experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/T5_composite_reference_live_chain_20261025.json
python -m scripts.analysis.audit_t2_candidate_uncertainty --reference output/experiments/T2_candidate_reference_live_chain_20261025.json --output output/experiments/T2_uncertainty_live_chain_20261025.json
```

SHA-256：

```text
scenarios_test_v2_diverse_draft.json 9cfc444188fa01e04175ed58c7b9f291ff26a268e381b06b534a9d8a491d02a0
scenarios_test_v2_draft.json 25832a04eb926b0cc834ecdca5d22c613fc05544f91d4a9191f8dd8edc867286
scenario_split_live_chain_20261025.json a35fd1469ce914f4b7ad03f27a7c4537625f0c186b60f3c40e79bb70a8105e5e
T1_route_reference_live_chain_20261025.json af6654b8c31ed45db75c3571b7f18e29499ff48dc1ed5924bbfc9efab9dcd2f9
T2_candidate_reference_live_chain_20261025.json 0ad833d59143aee2cdeeafb5e9dbda3bf018a27a78a30f3968feeac0df5b03ec
T5_composite_reference_live_chain_20261025.json f0afe31e5f32e7aee32349c333de62ce67bac02dbdd2c7ee96d2b3227f7e2ebd
T2_uncertainty_live_chain_20261025.json d2ee119534e18e43019ad67f079bc0fe6b9963491cb41e8d347f9d39328ea214
p1_live_chain_broad_20261025.xml b4abccbf4a131ed464d3c754a0684f4ef47a907615e0ae8ed1b1d0234bf19792
```

下一步仍需真实模型的同输入开发回放、T2 不确定样本的独立裁决、未见测试集封存和浏览器路线保留证据。当前不满足正式对照或论文效果报告门槛。
