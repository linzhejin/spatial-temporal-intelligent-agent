# T1、T2、T5 同题开发诊断与复合任务修复（2026-10-25）

## 范围与输入

本记录只覆盖三条**已见开发题**，不是正式论文实验。三条件为产品 Agent、普通工具 Agent 和固定官方提交 `687f282e8625c6b8d11ff58211a90524be2e4317` 的 DeepSeek Harness。每题三条件记录的用户消息、14 项工具定义和四个数据组件哈希一致。T5 从可见的 120 题草稿复制后显式降级为开发题，文件 SHA-256 为 `ee7c522234d3cd081b19804a70c1a9fcd3c7bae9dce7c4c64e159d98910968ff`；不再用该题报告盲测结果。

隔离数据库为 `whu_agent_experiment_test`。原始报告在本机忽略目录 `output/experiments/`，没有提交个人凭证或原始模型请求内容。实际 HTTP 尝试另有审计；产品确定性路径没有调用模型时记 0 次，不把传输记录的 `false` 解释成模型调用失败。

## 逐题观察

| 已见题 | 产品 Agent | 普通工具 Agent | Harness | 独立图内核对 |
|---|---|---|---|---|
| T1 从玉兰2门到樱顶，步行避台阶 | 1 次实际 HTTP；调用 `plan_route`；返回 `poi_449 → poi_161`、49 条边 | 2 次实际 HTTP；同一条 49 边路线 | 5 次实际 HTTP、6 次工具调用；同一条 49 边路线 | 用冻结图、POI 和独立边审计核对，边序列一致，图内连续与避台阶通过；非现场通行证明 |
| T2 从玉兰2门附近找能吃饭的地点 | 0 次模型请求；先给 4 个候选，等待用户选点 | 4 次实际 HTTP；给 6 个候选 | 5 次实际 HTTP；给 6 个候选，搜索工具有一次错误后重试 | 全部合格候选先计算路网距离；三者前四名都是 `poi_308, 307, 306, 305`。独立距离约 105、214、358、409 米。产品 4 个候选符合展示上限；两种基线显示 6 个，超出上限。`needs_input/candidates` 与 `completed/candidates` 都表示候选已送达，评分器已统一此语义 |
| T5 从教3楼先找餐饮，再去梅园操场；次轮文字选桂园一食堂并要全程路线 | 修复前第二轮再次返回候选；修复后第一轮候选、第二轮完整路线 `poi_030 → poi_068 → poi_082`，42 条边，0 次模型请求 | 8 次实际 HTTP；首轮自行选食堂并直接给完整路线，第二轮再次给完整路线，42 条边 | 9 次实际 HTTP；首轮展示的路线从食堂出发，漏了原起点；次轮工具生成正确 42 边路线，但最终状态 `max-tokens`，未完成答复 | 产品修复后独立图内审计 `route=pass`、`issues=[]`，三个地点的节点接驳约 5.65、10.41、1.89 米；不证明实际步道通行或地点入口 |

T5 的产品缺陷有两个根因：只有一个模糊活动加一个明确终点时，原先的活动序列解析没有把终点写进任务状态；选定候选地点的文字回复未解析为当前候选 ID，因此第二轮重复推荐。修复后仅在当前未完成的候选列表中接受明确文字选择；否定、歧义、疑问句或列表外地点不自动选中。原有 ID 卡片选择路径继续保留。回归中复现了修复前的失败，再验证了改动后的两段顺序与终点。

## 本轮检查

- 先对五个相关测试模块运行：72 通过；6 项集成用例因未显式设置 `TEST_DATABASE_URL` 跳过。随后设置隔离库连接并扩大到九个相关模块，**193/193 通过**。扩大回归时发现一项旧断言仍期待“明确目的地但缺起点”返回普通聊天；当前产品会正确追问起点、保留目的地，故将该断言改为检查现行交互。
- `ruff check` 对本轮 Python 文件通过；`git diff --check` 无空白错误。
- T5 修复后用真实产品 API、持久化任务和 worker 回放同一两轮输入；输出顺序和图内边审计通过。完整 UI、线上部署与现场步道没有在本轮验证。

## 可追溯报告 SHA-256

| 报告文件（位于 `output/experiments/`） | SHA-256 |
|---|---|
| `product_live_seed_mode_01_httpaudit_20261025.json` | `e236aefd9e01a4277daba1cd5c52e55a03d71afc36a8994149868eed0136f48e` |
| `ordinary_seed_mode_01_httpaudit_20261025.json` | `bc99de6ca0a14ce860282b05b700dad95bd236387cce813af021101ce45a04b1` |
| `harness_seed_mode_01_httpaudit_20261025.json` | `c0c1f7cfdd77450e6a1447fb3e63873fd79bc41c27beb36692a3928dd12839bd` |
| `product_live_seed_place_01_httpaudit_20261025.json` | `0733bfaf75942e700223b4b5b40ee3f51379fa65e822a98dc9389437616baa23` |
| `ordinary_seed_place_01_httpaudit_20261025.json` | `7a1d4a4d54cb1bc848248c871e06fe1af747d3d1b66a5aae37fc723fd1a8d3dc` |
| `harness_seed_place_01_httpaudit_20261025.json` | `045ae5fb283d3531da4d478ff1e20a2c1b64c0da950769a98eb9c0ae6c54beb3` |
| `product_live_t5_full_route_20261025.json`（修复前） | `4a749436df627b5b0e2a9909633f6863258be96eb810f389100ebbf6fb8d0641` |
| `product_live_t5_full_route_fixed_20261025.json` | `fb2aa056fada6bcf4539dbea7dd3444cf66cca31a7fb2a726c786e82286bd7e0` |
| `ordinary_t5_full_route_20261025.json` | `4c05b622bd72620703df48dcdcd0de6633057dd23802b65e9396203bbf0a11cb` |
| `harness_t5_full_route_20261025.json` | `a19009fefce0da1d90df5eb8a603e4ed989f1c0952da06ba8dac7854935e05b1` |

## 下一步闸门

本轮修复与证据提交为 `87b79b3d761a72dc79cfd134db047276a537d9e1`。提交后在干净工作树捕获 `output/experiments/dev_input_inventory_87b79b3_20261025.json`，文件 SHA-256 为 `a31549d84f24d9ce9a1311f8cdf8ced261b9d38c908d66a758e9184f717ac989`；有效输入指纹为 `7cd3b0a1074dd447ab7f05b6fc09551b328fc822982986fff78c597aebd77dc6`。清单仍标注 `snapshot_kind=diagnostic`、`formal_eligible=false`，不是正式冻结版本。

继续审查 T3–T5 的答案与多轮状态，修复开发题暴露的功能问题；人工只审核少量歧义卡和影像道路关联。另建真正未见留出集，冻结代码、图、POI、事件时钟、模型参数与费用上限，之后才可运行三次重复的正式对照。当前正式结果仍为**未开始**，不能从三条已见题计算架构优越性、成功率或显著性。
