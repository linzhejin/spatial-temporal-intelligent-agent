# 项目工作指南（2026-10-01）

本文件帮助 AI 编码助手快速进入项目。**代码、数据和当前评测报告优先于历史说明**：[当前进度与基础评测](docs/development/18_项目进度与基础评测_20261001.md)。旧设计和决策保存在 `docs/development/`，不要把其历史数量写成当前事实。

## 当前基线

- 项目：珞珈智行（WHU-Walker），武汉大学校园空间智能体，线上 `https://whuspati.online`。

<!-- FACTS:BEGIN -->
（以下为自动生成区，勿手改；更新：python scripts/docs/gen_facts.py）
- 版本：codex/aerial-vision-release @ 4162144 ｜ origin/main 同步 ｜ gitee/main 同步 ｜ 生成于 2026-10-15 21:54
- Agent 工具 14 ｜ POI 443（含校门 17） ｜ 路网 5,087 节点 / 14,622 有向边 ｜ 人工边覆盖 208
- 前端 SW whu-walker-v91 ｜ Android 1.4.1 (versionCode 6) ｜ 数据库迁移 13（最新 013）
<!-- FACTS:END -->

- 后端 Flask；Agent 工作流位于 `agents/`，LangGraph 编排，工具注册表 `agents/tools.py`（数量见上方自动事实块）。持久会话与任务在 `storage/`，后台执行由 Agent worker 承担。
- 空间算法在 `spatial/`。正式 POI 为 `data/pois.json`，正式融合路网为 `data/whu_road_network.graphml`，另有校门与人工边覆盖，数量见上方自动事实块。`data/edge_attribute_master.json` 与当前边 ID 全量绑定。
- 前端在 `static/`，Android WebView 壳在 `android-app/`；管理端入口 `/manager`。视觉分析/审核管线存在，线上视觉 worker 尚未启用。

## 关键行为约束

- 日常通勤与最短路径策略的距离/坡度/景观权重为 **`1/0/0`**；事实源见 `agents/routing_policy.py`。推荐、风景优先、平坦优先有独立预设。`spatial/routing.py` 的某些旧回退权重不是通勤策略。
- 出行方式切换只根据已保存的路线任务重算；不得交换起终点、遗失途经点或把“我的位置”当 POI 名检索。
- POI/高德底图使用 GCJ-02，自有路网使用 WGS-84；坐标系只在边界处转换，不能用转换掩盖缺路、错误线位和不可通行路段。
- 路线计算、硬约束、距离和前端几何必须引用相同的具体边 ID。校方道路融合不能破坏原有可达性；不把建筑轮廓相交自动判为封路。
- 服务端会话与任务修订有版本和并发保护；复合任务应有可追踪的部分成功/失败反馈。修改 Agent 状态时同步检查前端迟到响应保护。
- `static/sw.js` 缓存名以自动事实块为准（事实源为该文件的 `CACHE_NAME`）。本次盘点未发现前端 service worker 注册代码；修改离线/PWA 行为前先核实实际注册状态与缓存策略。

## 数据质量边界

`validate_all_data.py` 通过只表明结构与基本范围合格。当前发布就绪门槛仍因候选 POI、占位属性、风险边、POI 吸附、路线差异与现场证据不足而失败；详见[评测报告](docs/development/18_项目进度与基础评测_20261001.md)。正式空间数据的来源与核实状态应在每次改动后保持可追溯。高德 API 结果用于线上服务和对照，勿作为可自由再发布的自有路网。

## 常用检查

```powershell
python scripts/validate/validate_all_data.py
python scripts/validate/audit_campus_master.py
python scripts/validate/check_deployment_readiness.py
python scripts/validate/check_agent_research_readiness.py
python scripts/docs/gen_facts.py --check
python -m pytest tests/ -q
```

按改动范围选取测试。不要仅凭旧文档里写的通过数声称新代码通过。`.env` 放服务凭据，不提交。生产部署流程应以现有部署脚本与当前服务配置为准；部署后需核验版本和健康检查。Android APK 更新与网页部署是不同交付物，应分别核验版本。

## 文档维护纪律

- 版本与数据数量只出现在自动事实块与 [docs/当前事实.md](docs/当前事实.md) 中；描述“当前”的正文不写死数字，引用时指向这两处。
- `docs/development/`（带日期）与 `docs/archive/` 是历史快照，只读不改；新状态写入 `HANDOFF.md` 与事实文件。
- 改动代码/数据/前端版本后运行 `python scripts/docs/gen_facts.py` 更新文档；提交前由 `tests/test_docs_facts.py` 兜底检查。

## 本机资料

论文、竞赛材料与旧交接快照的整理位置见 [research/README.md](research/README.md) 和 [HANDOFF.md](HANDOFF.md)。`research/local/`、`output/local_test_runs/`、`scripts/audit_output/` 不入 Git。
