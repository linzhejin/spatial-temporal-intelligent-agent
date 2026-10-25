# 当前数据版本的多轮任务开发态回放

本记录只验证隔离研究工作区中的已见开发题。测试通过本机独立 PostgreSQL、真实 Flask API、持久化任务、LangGraph worker 与产品 Agent；没有向生产数据库或路况写入。使用当前 `data/pois.json`（SHA-256 `8322f12d0787e6aff7bceb489a7b8607bd748e3c4bbde543011475b7528d3307`）。三个任务均属于已经用于开发的题目，不是正式留出集，也不产生论文成功率。

| 题目与回合 | API/worker 观察 | 响应耗时 | 模型与网络请求 | 原始报告 SHA-256 |
|---|---|---:|---:|---|
| `v2h_T3_destination_weather_1`，先给终点、后补起点并查天气 | `needs_input/clarify → completed/route`；终点 `poi_141` 保留，补入起点 `poi_152`；第二轮同时返回路线、天气，18 条边 | 4.547 秒 | 逻辑模型 0 次，实际 HTTP 0 次 | `afb6983660b16aa7ae125727da7edbab14a2f55afc497f0b3dcf08fa452971f3` |
| `v2h_T5_food_before_route_1`，先选餐饮、后到原终点 | `needs_input/candidates → completed/route`；`poi_030 → poi_068 → poi_082`，42 条边；第一轮四个候选含所选餐饮点 | 6.555 秒 | 0 / 0 | `640459fd1a4177046f48e0f97d322ed37bd90d067a1fdd6fd2afea95ab8cf295` |
| `v2h_T5_sport_then_food_1`，先选球场、再选餐饮并走完全程 | `needs_input/candidates → needs_input/candidates → completed/route`；`poi_086 → poi_083 → poi_351`，33 条边；两轮候选均含实际选择点 | 8.089 秒 | 0 / 0 | `817a8ae9a523c3bb0553d3d730dc76c07c5d6a0e0b313f6f741d2efe1d02e24d` |
| `draft_multi_turn_revision_02`，修改终点并保留避台阶 | 两轮均返回路线；起点保持 `poi_449`，终点由 `poi_015` 改为 `poi_418`，`avoid_steps=true` 两轮均在；13、33 条边 | 11.965 秒 | 2 / 2，响应模型记录为 `deepseek-flash` | `387f3bb86431735697d63dc0874a70981a1a469c749d854d632006b2171a1b19` |

前三题的路线按对应的 `T3_T5_route_reference_rebased_20261025.json` 起终点和有序途经节点，使用 `experiments.scenario_v2.audit_route_edges` 对**实际边 ID**独立检查，结果均为 `route=pass`、无图内问题。第四题按当前 POI、路网与道路决策独立重新生成两轮端点参考，再审计边序列；两轮 `route=pass`，避台阶审计均为 `pass`。第四题第二轮终点距最近可导航节点超过 30 米，`access_uncertain=true`；图内通过不等于终点入口或现场道路已核实。

天气使用 `development_weather_clear_v1` 固定合成夹具，不能解释为实时天气。记录的耗时为单次本机开发回放，没有重复分布或端到端浏览器、手机体验数据。当前产品 T3 的实际 HTTP 尝试数已补齐为 0，但历史三架构试跑中的 T3 报告仍缺该字段；四次新回放没有同步重跑另两种架构，因此不能当作公平对照。控制台出现高德 Web key 未配置提示，以上指定任务仍完成；这不证明依赖高德的所有功能可用。

原始报告均保存在忽略目录 `output/experiments/`：`product_live_t3_current_poi_httpaudit_20261025.json`、`product_live_t5_current_poi_20261025.json`、`product_live_t5_sport_current_poi_20261025.json`、`product_live_t4_current_poi_20261025.json`。调用入口为 `python -m scripts.analysis.probe_product_live_case`，显式使用本机独立库与 `--allow-model-call`；私有配置、原始回答与几何不写入本记录。
