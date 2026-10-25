# 2026-10-25 P0 主实验门槛与 P2 开发态用量复核

本记录只对隔离研究分支的当前代码门槛和既有**已见开发题**负责。没有运行正式留出集，也没有把研究分支发布到线上。

## P0：当前提交的门槛

在干净提交 `68e18597653784302956e2e5b9a012cc771ad873` 捕获开发态输入清单 `output/experiments/dev_input_inventory_68e1859_20261025.json`，SHA-256 `ef7bd4341c00562106b523b6a1d03a9eef8ac2024e671b0790eb5b1b8e8c66fc`，有效输入指纹 `83ad771013cd1d857675760a60863541f688fc0fad5b2111e0f5f40d4c0dca03`。清单所列主实验输入均存在，`tracked_worktree_clean=true`，但 `snapshot_kind=diagnostic`、`formal_eligible=false`。

`python -m scripts.validate.check_agent_research_readiness` 的 11 组检查全部返回成功；`scripts/audit_output/agent_research_readiness.json` 的 SHA-256 为 `df47436b09854efec1b329cc6bff96fc76eb9f21d869ffb9c0471039b4538c15`。报告的 `agent_main_experiment.ready=true` **只表示代码与结构门槛**。独立的代表路线证据、坡度/风景属性和视觉类别范围仍为未通过；没有调用 `--write-snapshot`，因为 P1 来源裁决、真正留出集和正式费用门槛尚未满足。本次只读门槛检查会重写忽略目录中的诊断报告，旧报告已先备份为 `output/experiments/readiness_before_68e1859_20261025.json`。

## P2：五类任务各一条同题开发试跑

将已有报告按 T1–T5、每类三架构组成 15 条用量记录，另保留三条历史 T5 天气试跑供旧费用估算兼容。跟踪目录 `experiments/fixtures/development_cost_inputs_20261025.json` 的 SHA-256 为 `b75f64619269e4a3f5fb299a60c7abde7757120428a22e02a5e1e248d7d001d4`；每份原报告按哈希读取。五组三架构的题目 ID、用户消息、工具模式和数据组件哈希在**各组内部**一致。费用审计报告 `output/experiments/development_cost_audit_matched_20261025_v3.json` 的 SHA-256 为 `507065c364b06cc6f3ef2248ef0d0f879c989b08958ec39c15f79108620261da`。

| 已见任务 | 珞珈智行 | 普通工具 Agent | DeepSeek Harness |
|---|---:|---:|---:|
| T1 路线 | 1 次；输入/输出 12914/140 token | 2 次；20484/440 | 5 次；3268/1026 |
| T2 推荐 | 0 次；0/0 | 4 次；41958/516 | 5 次；2286/991 |
| T3 补起点并查天气 | 0 次；0/0 | 5 次；51726/606 | 7 次；2983/1244 |
| T4 多轮改终点 | 2 次；25719/249 | 6 次；70055/918 | 9 次；5057/1832 |
| T5 先选餐饮再到终点 | 0 次；0/0 | 8 次；101679/1393 | 9 次；10011/3318 |

表中的“次”是报告中记录的逻辑模型请求。除产品 T3 外，报告另有对应的实际 HTTP 尝试计数；产品 T3 缺这项传输审计，因此“0/0”仅是已记录模型轨迹为零，**不能当作账单已证实为零**。两种基线的响应模型在有调用的报告中为 `deepseek-flash`；产品的零调用题没有响应模型身份，不能据此宣称每道题都实际调用了同一模型。

这批报告的 POI 原始文件哈希为 `db5f4177b596a1b1ae614dacafa39ce78ae04d5a7f8ef41ef7fa166a62958b0c`，当前 POI 哈希已是 `8322f12d0787e6aff7bceb489a7b8607bd748e3c4bbde543011475b7528d3307`。故这张表**只用于历史开发用量分层**，不能直接代表当前数据版本的任务效果或正式调用量分布。费用审计保持 `formal_budget_ready=false`：每类只有一题、没有重复分布；Harness 缺缓存命中拆分；T3 产品缺 HTTP 尝试审计；正式付费前还须核对实时价格及用户费用上限。所附价格夹具标为 2026-10-24 开发材料，不作为当前报价。

费用脚本已能读取产品真实 API/worker 报告的 `run_evidence`，包括确定性零模型请求；无 HTTP 审计的零请求不输出“已观测账单为零”。`python -m pytest tests/test_development_cost_audit.py tests/test_deployment_readiness.py -q` 的 35 项检查通过，JUnit `output/experiments/p0_p2_gate_usage_regression_20261025.xml`，SHA-256 `736d8eeb9abd552733c3208f6a59d74d538da0209c5cff78a554c34f981f4005`。正式实验仍未启动。
