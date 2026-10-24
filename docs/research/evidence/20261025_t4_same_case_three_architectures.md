# T4 已见题：改终点并保留约束的三架构开发诊断

日期：2026-10-25。题目 `draft_multi_turn_revision_02` 来自可见开发集：先从玉兰2门步行到武汉大学图书馆总馆，避开已标记台阶；再把终点改成卓尔体育馆，保持原起点、步行和避台阶。此题**已见且无独立金标准**，以下只验证同题链路，不计算论文成功率。

三条件的用户消息哈希、14 项工具模式哈希及 POI／路网／覆写／审查四个数据组件哈希逐项相同。产品、普通工具 Agent 与固定 Harness 都观测到响应模型 `deepseek-flash`。Harness 上游固定提交 `687f282e8625c6b8d11ff58211a90524be2e4317`。

| 条件 | 第一轮 | 第二轮 | 记录的开销 |
|---|---|---|---|
| 珞珈智行 | `poi_449 → poi_015`，步行、避台阶，13 条边 | `poi_449 → poi_418`，步行、避台阶，33 条边 | 2 次模型请求、2 次实际 HTTP 200、2 次成功算路工具调用 |
| 普通工具 Agent | 同一组起终点、方式、约束和 13 条边 | 同一组起终点、方式、约束和 33 条边 | 6 次逻辑请求、6 次实际 HTTP 200、4 次成功工具调用 |
| 固定 Harness | 同一组起终点、方式、约束和 13 条边 | 同一组起终点、方式、约束和 33 条边 | 9 次逻辑请求、9 次实际 HTTP 200、10 次工具调用；首轮一次 `plan_route` 工具错误后重试成功 |

三份实际路线投影均无地点身份或边序列未知项。产品侧新增仅保存 HTTP 状态码的传输钩子，在隔离数据库按相同输入重跑；两次 HTTP 均为 200，传输记录完整，两轮边序列与新增钩子前的产品报告逐项相同。普通工具 Agent 初版投影曾把首轮“武汉大学图书馆总馆”记为未知，因为路线状态里的用户写法不是正式 POI 名；工具结果同时给出了“武汉大学图书馆(总馆)”的规范名称。投影已改为读取**实际路线工具产物**的规范起终点，且状态与结果解析成不同 POI 时仍标记冲突。修正后普通工具 Agent 首轮终点为 `poi_015`；这是评测投影修复，不是改动基线的实际选路。

本题的三条件输出一致，不能据此断言某种架构在任务正确率上更优。请求数、token 与时延也只有一次运行；普通 Agent 使用 OpenAI 兼容接口且有缓存命中拆分，Harness 使用官方 Messages 接口，未按共同价格和重复样本换算成本。图内路线属性不证明现场路段可通行。

## 本地原始证据

原始报告位于忽略目录 `output/experiments/`，不包含凭证或请求正文。基线报告保留工具调用结果和传输状态；产品报告保留 API／worker 逐轮投影。

| 文件名 | SHA-256 |
|---|---|
| `product_live_draft_multi_turn_revision_02_projectionfix_20261025.json` | `a1c51ad96672f77fb54e2397b533d8cf4997117f90ec39abbae6139eb64e7081` |
| `product_live_draft_multi_turn_revision_02_httpaudit_20261025.json` | `2b497aa9b294f1b79afe1633ead3f8a42e09f6753cce48ffeb387f37991a9080` |
| `ordinary_t4_revision_httpaudit_20261025.json` | `aada23c678d66f0d78cc71a2d7f1ac1bf97be334853df3844d4ee7cb93a7116d` |
| `ordinary_t4_revision_httpaudit_20261025_run_evidence.json` | `25b205c77760ea4fc804d0a2aa8c833cfb44d85605676103bcb8c3d25c957204` |
| `harness_t4_revision_20261025.json` | `7d06287c3767b373abba717ced806cf4bfaa4e66660c4def53b190c08f195ed3` |
| `harness_t4_revision_20261025_run_evidence.json` | `47548359383ed06c2ff46690d0e6b1ab49083ad7f970c522cf670c8ba1be3126` |
| `t4_revision_three_condition_baseline_projection_canonical_20261025.json` | `c5fd0410d3ab3ddcdc9ea2739a6b5b352ba3b8aaf6b14318c2da77807790187c` |

**下一步：**扩展 T1、T2、T5 同题小样本并验证可见交付、失败和来源表述。正式留出、独立金标准、重复运行及费用上限仍未具备。
