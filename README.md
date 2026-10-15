# 珞珈智行（WHU-Walker）

> **版本统一（2026-10-15）：** 当前维护基线包含最新无人机与管理台实现，并整合人工地点修正。版本取舍、工作区保留和清理结果见 [版本整合记录](docs/development/32_版本整合与同步记录_20261015.md)；[目录报告](docs/development/31_目录整理与当前状态_20261015.md)保留整合前的状态快照。

武汉大学校园空间智能体。用户可用自然语言提出地点、出行方式、途经点、时间和偏好需求；Agent 调用地点检索、路线规划、天气和路况等工具，向网页和 Android App 返回路线与解释。在线地址：[whuspati.online](https://whuspati.online)。

> 当前代码进度、分支/部署状态和接手事项以[项目交接说明](HANDOFF.md)为准。详细目录说明与测试背景见[2026-10-04 状态快照](docs/development/21_项目当前状态与整理记录_20261004.md)；全库数据审计见[项目进度与基础评测](docs/development/18_项目进度与基础评测_20261001.md)，其中数据数量是 2026-10-01 历史快照。

## 当前组成

| 部分 | 当前实现 |
|---|---|
| Agent | LangGraph 编排、工具注册表（数量见下方自动事实块）、版本化会话与任务状态；PostgreSQL 持久化，后台任务 worker |
| 空间计算 | NetworkX/OSMnx 路网、校方课程数据融合、分方式通行过滤、距离/坡度/景观权重 |
| 地图与数据 | 高德底图与线上 API；截至 2026-10-01 的快照为 440 个 POI、17 个校门、5,087 个节点和 14,622 条有向边 |
| 用户端 | 手机与电脑网页、Android WebView App、语音输入/播报、路线切换和步行导航 |
| 管理端 | 独立 `/manager` 界面，人工维护路况；视觉候选须经管理员复核、重新选路和填写核实依据后才能转成道路事件 |

<!-- FACTS:BEGIN -->
（以下为自动生成区，勿手改；更新：python scripts/docs/gen_facts.py）
- 版本：codex/aerial-vision-release @ 4162144 ｜ origin/main 同步 ｜ gitee/main 同步 ｜ 生成于 2026-10-15 21:54
- Agent 工具 14 ｜ POI 443（含校门 17） ｜ 路网 5,087 节点 / 14,622 有向边 ｜ 人工边覆盖 208
- 前端 SW whu-walker-v91 ｜ Android 1.4.1 (versionCode 6) ｜ 数据库迁移 13（最新 013）
<!-- FACTS:END -->

日常通勤与“最短路径”策略的距离/坡度/景观权重为 `1/0/0`；推荐、风景优先和平坦优先使用其他预设。2026-10-05 核验生产服务器检出 `3beea8d`，健康接口、管理页面及其 JS/CSS 均正常；路线可计算或画出，不等于每条边都已实地证明可通行。当前开放候选、风险路段与路线样本仍需核查，详见评测报告。

## 本地运行与检查

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

按 `.env.example` 配置自己的服务凭据。项目数据和路网缓存按仓库及开发文档的说明准备。

```powershell
python scripts/validate/validate_all_data.py
python scripts/validate/audit_campus_master.py
python scripts/validate/check_deployment_readiness.py
python -m pytest tests/ -q
```

前两项检查数据结构和来源绑定；发布就绪检查还要求候选裁决、人工通行证据和路线对照，当前会返回未通过。请勿把结构校验通过写成实地质量验收通过。

## 文档

- [当前交接说明](HANDOFF.md)
- [当前事实](docs/当前事实.md)
- [项目进度与基础评测](docs/development/18_项目进度与基础评测_20261001.md)
- [项目当前状态与整理记录](docs/development/21_项目当前状态与整理记录_20261004.md)
- [文档索引](docs/README.md)
- [本地研究资料索引](research/README.md)

## 许可

项目代码的许可见仓库声明。OSM 派生数据、校方资料和地图商 API 各有不同的使用边界；引用、再分发或公开数据前应逐项核对来源与授权。
