# P0：任务超时轮询与浏览器提示回归

本次检查使用隔离研究工作区和本机独立 PostgreSQL 测试库；没有写入生产数据库或线上路况。代码基线为提交 `3db01e1`。测试将任务截止时间注入为过去，目的是核实排队中或运行中的任务即使没有 worker 继续处理，用户轮询仍能得到终态。

| 场景 | 检查结果 |
|---|---|
| 排队中任务逾期 | `/api/runs/<id>` 返回 `failed/deadline_exceeded` 和一次 `run_completed` 事件；重复轮询不会重复终结；同一会话的新请求可建立新任务。 |
| 运行中任务逾期 | 同上；本项通过测试库领取任务后再把截止时间置为过去。 |
| 浏览器收到逾期结果 | 聊天气泡显示“任务超过执行时限。”，不继续显示等待；其后可发起新请求。既有提交失败重试、取消及刷新会话流程仍通过。 |

运行命令与原始输出：

```powershell
$env:TEST_DATABASE_URL='postgresql://postgres@127.0.0.1:55439/whu_agent_experiment_test'
$env:DATABASE_URL=''
python -m pytest tests/test_run_api.py tests/test_run_worker.py -q --junitxml=output/experiments/p0_deadline_api_worker_20261025.xml
python -m pytest tests/browser/test_user_agent_web_flow.py -q --junitxml=output/experiments/p0_deadline_browser_suite_20261025.xml
python -m ruff check tests/test_run_api.py tests/browser/test_user_agent_web_flow.py
```

- 后端 API/worker：16/16 通过；JUnit SHA-256 `ee466cdac2f6658de9377938435a5acaa2a28f142905523380f4267b6a9b0310`。
- 浏览器：4/4 通过；JUnit SHA-256 `34bf10c83b2725fec738de005338745454eb5acf68f72cd8eec2f76cd6b61e0e`。
- Ruff：通过。

边界：浏览器接口由 Playwright 模拟，逾期由测试数据库人为注入；这些检查证明终态传递和用户继续提交的代码路径，**不证明**生产断网、持续高负载、所有模型超时或现场用户体验。正式实验仍需冻结输入和独立留出样本。
