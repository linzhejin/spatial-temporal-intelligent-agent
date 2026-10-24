# T3 已见题：同输入三架构开发诊断

日期：2026-10-25。只使用已见开发题 `v2h_T3_destination_weather_1`，**不是正式成绩**。该题首轮指定凌波门东湖观景点、待补起点；次轮明确“武汉大学樱园”，要求先查天气再按原目的地规划。题目原属于可见编题草稿，试跑前已复制并标为 `development`；本题及同源场景不能再充作未见测试。

三条件都读取相同的用户消息、POI／路网／覆写／审查文件、14 项工具模式和模拟天气夹具。程序核对 `case_messages_sha256`、`tool_schema_sha256`、四个数据组件 SHA-256 在三份报告中完全一致，天气夹具 SHA-256 均为 `fab71cf975d02313e09270500b50e236e6b5d4ce89daeed834bfad8eb7bc48b6`。Harness 官方上游固定提交 `687f282e8625c6b8d11ff58211a90524be2e4317` 且工作区干净。

| 条件 | 首轮可观察状态 | 第二轮可观察结果 | 模型／工具记录 |
|---|---|---|---|
| 珞珈智行 | `needs_input/clarify`；结构化保留终点 `poi_141`、待补起点 | `completed/route`；起点 `poi_152`，终点 `poi_141`，18 条边；用户可见输出含路线与固定天气 | 两轮均走确定性任务分支，0 次模型请求；第二轮 `get_weather`、`plan_route` 成功 |
| 普通工具 Agent | 文字问起点，但未调用 `ask_user`，投影为 `completed/information`，无结构化终点 | 调用天气和算路工具，路线起点为 `area_樱园`、终点 `poi_141`，26 条边；最终文本未实际复述天气 | 响应模型 `deepseek-flash`；传输审计复跑为 5 次逻辑请求、5 次实际 HTTP 200、6 次工具调用 |
| 固定 Harness | 调用 `resolve_poi`、`ask_user`，投影为 `needs_input/clarify`，澄清槽为 `start`；但未结构化保留终点 ID | 路线起点同样为 `area_樱园`、终点 `poi_141`，26 条边；最终文本包含路线和模拟天气 | 响应模型 `deepseek-flash`；7 次逻辑请求、7 次完整 HTTP 200、7 次工具调用 |

`poi_152` 是“武汉大学樱园”；`area_樱园` 是另一条名为“樱园”的区域 POI。用户原话明确含“武汉大学”，两种基线的第二轮算路起点因此与题目不符。路线边数不同也受此起点差异影响，不能单凭 18 与 26 条边判断路线优劣。Harness 第一轮保留了追问类型，但其工具产物没有终点 ID，不能把回复文字中提到的目的地算作持久化任务状态。普通 Agent 虽然能继续对话，首轮却没有可判的澄清工具记录。投影脚本显示两个基线的元数据缺口列表均为空；这**不等于任务通过**。

Harness 第二轮把固定开发天气称作“武汉实时”，并将没有生效事件表述成全程畅通；这两点超出夹具和路况记录所能证明的范围。普通 Agent 已成功调用天气工具，但最终文本没有给用户天气信息。后续评价须把**工具成功、最终交付和来源表述**分开记录。

## 原始证据与复核边界

所有原始报告均在本机忽略目录 `output/experiments/`，未包含凭证。命令入口分别为 `scripts.analysis.probe_product_live_case`、`scripts.analysis.probe_ordinary_tool_agent`、`scripts.analysis.probe_harness_agent`；基线投影使用 `scripts.analysis.audit_development_pilots`。报告中的用户文本和场景是已见开发输入；固定天气不是实时天气，图内路由不证明现场通行。

| 文件名 | SHA-256 |
|---|---|
| `product_live_t3_context_weather_fixed_20261025.json` | `9b5d026260a548fc23e9119060986d771748edb95fae1e66553e9f8969de4023` |
| `ordinary_t3_context_weather_20261025.json` | `05167fe670d9422238b3e046e1ababf5c690b706b2aea9c2b3802770a9444427` |
| `ordinary_t3_context_weather_20261025_run_evidence.json` | `4ef5a13e744a2931f32325627f8a726166f587457cf93283e81e1d496d088b24` |
| `ordinary_t3_context_weather_httpaudit_20261025.json`（传输审计复跑） | `990848963db57170ad0f80e0bce4c8b0ba0f5d1892e6c15a608a990dd34a5ac6` |
| `ordinary_t3_context_weather_httpaudit_20261025_run_evidence.json` | `dd16339af48d705d6e37f5e0780a0ea4a9523ff00bcb69b5b3d606b35c45bb24` |
| `harness_t3_context_weather_20261025.json` | `24a8d5e0bb5ce1ee60404112f43f3d3bd93edfd6ad68198c43c36a3dfc3104aa` |
| `harness_t3_context_weather_20261025_run_evidence.json` | `8710725673324bc11bb800e0b821cbed56762bcbd8ad00aaf814c15ddd54003e` |
| `t3_context_three_condition_baseline_projection_20261025.json` | `f0b820ca6254ebacdd8477c00567db44d522c2c9e40e640790edbfea1dec15c6` |
| `t3_context_three_condition_baseline_projection_httpaudit_20261025.json` | `78603c8e6430248d32803881cadbdc619da20efab42bd16f3dad404a7b5c310c` |
| `t3_context_three_condition_baseline_projection_canonical_20261025.json`（当前投影器） | `78603c8e6430248d32803881cadbdc619da20efab42bd16f3dad404a7b5c310c` |

普通工具 Agent 初跑为 6 次逻辑请求，尚未启用 HTTP 审计；补审计后的同输入复跑为 5 次逻辑请求、5 次实际 HTTP 200。两次均出现相同的未结构化追问、错误起点和漏答天气。不能把两次差异归因于审计代码；也不能用单次请求数估算稳定开销。审计只保存状态码，不保存请求内容、URL 或认证头；模拟 429→200 的单元用例证明 SDK 自动重试会被计为两次实际请求。

**尚缺的公平性证据：**产品条件本题没有模型调用，其他题型仍需真实 HTTP 请求审计。三条件还没有跨 T1–T5 的足量同题、重复次数、独立金标准或费用上限。下一步补分类型小样本，再决定正式批量实验预算；本条不能写成优势百分比。
