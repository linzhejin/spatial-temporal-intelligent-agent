# T3/T4 多轮语义与天气保路：开发态证据

日期：2026-10-25。范围：研究分支的产品分派修复、可见编题草稿与独立评分规则。**不是正式论文成功率，也未部署到生产。**

## 问题与改动

- 原 T3 `destination_weather` 只有“带我过去”一轮，因没有前文目的地而只能追问，漏测用户裁决的“同一未完成任务已经说过目的地”。现有 3 条同类草稿改为：先给目的地但缺起点，系统追问起点；再给起点并要求天气与全程路线。其他无目的地的 T3 草稿仍要求追问。
- 原 T4 `weather_after` 把“再告诉我天气，不要改路线”误标为再次输出路线；产品任务分派也会因句中“路线”二字重新调用算路。现有 3 条草稿改为第二轮 `information`＋`weather`，并标记 `preserve_route_from_turn=0`。产品分派会剔除“不要改路线”等否定子句后识别正向算路请求；如果同句确有新的“从甲去乙”，仍要求算路。
- 独立评分对保路增加单独判定：没有独立客户端证据时记 `route_preservation_unverified`，观察到路线变化记 `route_changed`。图内路段合法性不能证明浏览器仍保留原线，也不能证明第二轮未重算。`graph_computed` 路线审计可与天气和保路的人工独立审查并用。

## 运行与结果

| 检查 | 实际结果 |
|---|---|
| 可见题库重建 | 两份草稿各 120 题；T3 上述正例 3/3 为两轮，T4 上述天气续问 3/3 为信息回答 |
| 重建确定性 | 同参数重建到忽略目录后，120 题多样化草稿 SHA-256 与追踪文件完全一致 |
| 拆分闸门 | 60 开发题、120 草稿；指定字段无直接重合，`formal_ready=false`，7 项阻碍；120/120 仍为 `draft` |
| 独立参考重算 | T1 图内连通 24/24；T2 24 题中 11 条生成器标记待核；T5 选点前四名 9/9、两段图内连通 9/9 |
| Python 聚焦回归 | 118 passed；含路线保留缺证据、错误与人工审查的反例 |
| 前端交互回归 | `node --test tests/js/conversation-lifecycle.test.js`：81 passed；天气信息回答不清除已展示路线 |
| API/数据库集成 | `tests/test_scenario_live_integration.py`：4 skipped；未设置显式 `TEST_DATABASE_URL`，**未取得本次实际服务链路证据** |

运行命令：

```text
python -m pytest tests/test_task_planner.py tests/test_build_scenario_v2_authoring.py tests/test_scenario_protocol.py tests/test_scenario_v2.py tests/test_scenario_split_audit.py tests/test_scenario_live_adapter.py tests/test_context_builder.py tests/test_build_t5_composite_references_v2.py -q --junitxml=output/experiments/p1_t3_t4_regressions_20261025.xml
node --test tests/js/conversation-lifecycle.test.js
python -m experiments.scenario_split_audit --development experiments/datasets/scenarios_dev_v2.json --test experiments/datasets/scenarios_test_v2_diverse_draft.json --output output/experiments/scenario_split_T3T4_20261025.json
```

哈希（SHA-256）：

```text
scenarios_test_v2_diverse_draft.json 1bacd1e3c0728184c5375696e258f1154053ec4be612283d63cdec09725c835a
scenario_split_T3T4_20261025.json 77d2233d6ffd06a677c362136bfb8f4f3edd75c358c68236ab9816bc518fd7df
T5_composite_reference_T3T4_20261025.json 550b6e07d5f532129a5a99937ffa7fcde8730bdf871625cee54c06d5d98e9602
p1_t3_t4_regressions_20261025.xml 7726ec4ef1e98a88c57d2ce63d8b022973a2dbd77243b6c8e251c60b567db9c9
```

旧版草稿和相关参考报告因题库字节变化不再匹配当前输入哈希；新版报告在本地 `output/experiments/`，且该目录按项目规则不入库。路线保留的正式验收仍需在隔离服务和真实浏览器中核对前后路线 ID／边序列及工具调用，再由独立审查提供证据。未见测试集、逐题金标准和三架构批量对照仍待完成。
