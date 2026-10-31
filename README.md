# 珞珈智行（WHU-Walker）

> 当前进度见 [HANDOFF.md](HANDOFF.md)，文件夹用途见 [项目目录说明](docs/项目目录说明.md)。

武汉大学校园空间智能体。用户可用自然语言提出地点、出行方式、途经点、时间和偏好需求；Agent 调用地点检索、路线规划、天气和路况等工具，向网页和 Android App 返回路线与解释。在线地址：[whuspati.online](https://whuspati.online)。


## 当前组成

| 部分 | 当前实现 |
|---|---|
| Agent | LangGraph 编排、工具注册表（数量见下方自动事实块）、版本化会话与任务状态；PostgreSQL 持久化，后台任务 worker |
| 空间计算 | NetworkX/OSMnx 路网、校方课程数据融合、分方式通行过滤、距离/坡度/景观权重 |
| 地图与数据 | 高德底图与线上 API；地点和路网规模见下方由当前数据生成的事实块 |
| 用户端 | 手机与电脑网页、Android WebView App、语音输入/播报、路线切换和步行导航 |
| 管理端 | 独立 `/manager` 界面，人工维护路况；视觉候选须经管理员复核、重新选路和填写核实依据后才能转成道路事件 |

<!-- FACTS:BEGIN -->
（以下为自动生成区，勿手改；更新：python scripts/docs/gen_facts.py）
- 版本：HEAD @ 2be9491 ｜ origin/main 本地领先 20 ｜ gitee/main 本地领先 20 ｜ 生成于 2026-10-31 11:42
- Agent 工具 14 ｜ POI 443（含校门 17） ｜ 路网 5,087 节点 / 14,622 有向边 ｜ 人工边覆盖 208
- 前端 SW whu-walker-v98 ｜ Android 1.4.1 (versionCode 6) ｜ 数据库迁移 15（最新 015）
<!-- FACTS:END -->

日常通勤与“最短路径”策略的距离/坡度/景观权重为 `1/0/0`；推荐、风景优先和平坦优先使用其他预设。路线可计算或画出不等于每条边均已实地核实；发布记录见文档目录，当前进度见交接说明。

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
- [文档索引](docs/README.md)
- [本地研究资料索引](research/README.md)

## 许可

项目代码的许可见仓库声明。OSM 派生数据、校方资料和地图商 API 各有不同的使用边界；引用、再分发或公开数据前应逐项核对来源与授权。
