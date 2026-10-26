# 合成路况开发回放 v3

本目录只有两道**开发诊断题**：一题封路后可绕行，一题封路后无可行路。路况是模拟事件，不代表武大现场真实封路；图内可行也不等于现场可通行。`source_manifest_event_pilot_dev_v1.json` 固定列举的输入来源和代码哈希；运行时还会复核 v2 基础清单，并记录整个相关 Python 源码树的摘要。v1、v2 历史包不改写。

校验固定数据：

```powershell
python -m experiments.build_controlled_event_pilot_v3
```

将本地隔离服务的真实 Agent 运行结果投影为事件专用 JSONL 后，可离线评分：

```powershell
python -m experiments.controlled_event_replay_v3_run --replay <投影结果.jsonl> --run-id <本次唯一编号> --observation-origin unverified_projection
```

评分器核对路况查询、所用事件 ID、路线边序列和工具顺序。无可行路线时，须有算路器确认的 `graph_no_path` 证据，并由起终点节点、方式和避台阶条件的摘要匹配本题；参数错误或其他起终点的无路结果不能算作正确拒绝。结果写入 `output/experiments/controlled_event_pilot_v3/<本次唯一编号>/`，已有目录不会被覆盖。

输出中的 `projection_score_pass_count` 只表示所提供投影通过了规则，不代表 Agent 成功率。若输入由题目预期答案人工构造，须使用 `--observation-origin synthetic_projection`。当前入口不接受声称已核验的 Agent 来源。隔离服务须设置 `AMAP_POI_LOOKUP_ENABLED=false`，使实际起终点落点与冻结锚点一致；否则就绪检查拒绝运行。

真实服务采集需在本地测试库环境设置 `AGENT_REPLAY_ENABLE_LOCAL=1`、`AGENT_TOOL_TRACE_ENABLED=1`、`AMAP_POI_LOOKUP_ENABLED=false`，并按每题的冻结路况文件配置 `ROAD_CONDITIONS_FILE`、`AGENT_REPLAY_EVENT_EPOCH` 与 `AGENT_REPLAY_EVENT_SHA256`。就绪接口会检查本地测试数据库、路况文件和时间。当前目录未提供一键启动器，也未产出可核验的真实 Agent 成绩。

正式实验仍需逐题隔离的本地测试 PostgreSQL、真实 Agent 运行记录及其采集元数据、盲测数据与独立判据。本包的自动生成预期结果只用于检查评分器，不能充当 Agent 表现。
