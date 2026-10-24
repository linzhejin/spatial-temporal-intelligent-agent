# 2026-10-25 P0 浏览器有序途经与失败恢复回归

研究分支 `codex/isolation-cleanup`、起始提交 `4eb3e22`。本轮只检查本机浏览器和独立测试库，不调用收费模型、实时天气或线上 API。任务 API 在静态浏览器测试中由确定性结果替代；真实 Flask/worker 测试也使用替身 Agent。因此这是一组**交互链路验收**，不是自然语言解析或实际道路正确率。

新增 `tests/browser/test_user_agent_web_flow.py::test_ordered_multi_stop_route_remains_visible_after_weather_followup`：输入“从玉兰2门先到桂园操场，再到工学部集贸市场，最后卓尔体育馆”，模拟后端返回四个有序坐标和两个途经点。浏览器检查结果区、途经点数量和四点画线；随后只问天气，检查原画线及结果区保留，并确认第二条消息确实送往同一会话。浏览器地图使用本地页面和模拟任务结果，外部瓦片请求被拦截。

同一浏览器文件原有场景还覆盖 PC 与移动网页的球类澄清、候选点选择、出行方式切换、触屏面板拖动，以及请求失败后的重试、取消旧任务和新会话。另运行真实 Flask/worker 的独立 PostgreSQL 浏览器持久化场景、有序途经工具和组合路线＋天气 worker 回归。

| 命令范围 | 结果 | 本地原始报告 SHA-256 |
|---|---|---|
| `python -m pytest tests/browser/test_user_agent_web_flow.py -q --junitxml=output/experiments/p0_user_browser_flow_20261025_v2.xml` | 4/4 通过 | `4fe9c3c88d0d66ce9a96134e406ebf950af81ff0e4a41f36df8f554e784d7ae7` |
| 设置专用 `TEST_DATABASE_URL` 后运行 `tests/browser/test_user_agent_backend_e2e.py tests/test_ordered_via_execution.py tests/test_run_worker.py` | 38/38 通过 | `dfc289aa9d0382efe5fce5e426df55518ee1d45da2e7732c47ffbc493f873cb4` |

测试数据库由 `tests/db_safety.py` 核对名称和实际连接目标；本轮使用的是本机 `whu_agent_experiment_test`，未写生产数据库。测试只证明被模拟的完整路线能在浏览器中展示和保留；仍需真实模型在三学部逐题生成、工具输出几何与独立路线参考比对，以及失败时的真实服务超时回放，才能关闭 P0/P1 的相关验收项。
