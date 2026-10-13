# 04 · API 参考

- 业务端点前缀 `/api`（Blueprint `url_prefix="/api"`）；健康检查 `GET /health` 无前缀（另有 `/api/health` 别名）。
- 端点清单以 [api/routes.py](../api/routes.py) 与 [app.py](../app.py) 为准；本文档说明主要对外端点。
- 统一响应：成功 `{"data": {...}}`，失败 `{"error": "code", "message": "..."}`。
- 主入口是 **POST /api/chat**（全 Agent）；`/parse` 与 `/route` 是拆分式旧管道，Agent 故障时服务端内部也用它们兜底。

## 健康检查

### GET /health

负载均衡/监控探活，无需鉴权。

```json
{"project": "珞珈智行", "status": "ok", "timestamp": 1789540308}
```

## 对话与规划（主链路）

### POST /api/chat ⭐

一站式入口：NL → Agent 循环（工具调用）→ 路径 / 候选 / 澄清 / 闲聊。

**请求体**

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | string | 必填，≤500 字 |
| `context` | object | 多轮上下文（history、previous_route_state；未补全时含 previous_intent） |
| `travel_mode` | string | walk / bike / drive，前端方式切换传入 |
| `coord_start` / `coord_end` | object | GPS 端点 `{lng, lat}`，**WGS-84**；"我这/到我这"用 |
| `coord_waypoints` | array | 地图途经点 `[{lng, lat}, ...]`，WGS-84 |
| `whu_uid` / `uid` | string | 前端生成的持久用户 ID（画像与埋点用） |

**响应 data：按 response_kind 分四种形态**

1) `task_type: "path_planning"`（Agent 调了 plan_route / plan_via_route / plan_tour）

```json
{
  "task_type": "path_planning",
  "response_kind": "route",
  "route_kind": "direct",
  "explanation": "已为你规划好从珞珈门到樱顶的步行路线，约 830 米、12 分钟。",
  "start": {"name": "珞珈门", "type": "poi"},
  "end": {"name": "樱顶", "type": "poi"},
  "recommended": [{"lng": 114.358, "lat": 30.533}],
  "shortest": [{"lng": 114.358, "lat": 30.533}],
  "steps": [{"seq": 0, "type": "depart", "text": "...", "cumulative_m": 0, "point": {"lng": 0, "lat": 0}}],
  "pois": [{"name": "万林博物馆"}],
  "recommended_length_m": 830.0,
  "shortest_length_m": 760.0,
  "duration_min": 12.0,
  "applied_weights": {"distance": 0.8, "slope": 0.05, "scenery": 0.15},
  "mode": "walk",
  "filter_status": "no_filter",
  "suggestions": [{"label": "风景更好的", "query": "..."}],
  "via": {"...": "plan_via_route 时出现"},
  "tour": {"...": "plan_tour 时出现"},
  "legs": ["两段/多段路径包，via/tour 时出现"]
}
```

坐标字段（recommended/shortest/steps.point/POI）一律 **GCJ-02**，可直接上图。

2) `task_type: "candidates"`：`{"message", "candidates": [...]}`，前端渲染候选地点卡片。

3) `task_type: "unknown"`（clarify 反问）：`{"message", "clarify": {"question", "options": [...]}}`。

4) `task_type: "chat"`：`{"reply": "..."}`。

LLM API 故障时自动落旧管道，响应结构与上述保持兼容（旧管道另有 help / unknown 等分支文案）。

### POST /api/parse（旧管道·拆分接口）

NL 或快捷按钮 → TaskIntent。

```json
// 请求：NL 模式
{"query": "从珞珈门到樱顶，避开陡坡", "input_method": "nl"}
// 请求：快捷模式
{"start": {"name": "珞珈门"}, "end": {"name": "樱顶"}, "input_method": "shortcut", "mode": "distance_first"}
```

```json
{
  "task_type": "path_planning",
  "start": {"name": "珞珈门", "type": "poi"},
  "end": {"name": "樱顶", "type": "poi"},
  "constraints": {"distance": "medium", "slope": "avoid", "scenery": "normal"},
  "weights": {"distance": 0.2, "slope": 0.6, "scenery": 0.2},
  "mode": "walk",
  "weight_source": "explicit_nl",
  "ambiguity": null
}
```

`weight_source ∈ explicit_nl | shortcut | default`；task_type 另含 poi_query / help / chat / unknown。

### POST /api/route（旧管道·拆分接口）

结构化意图 → 路径。请求体含 start/end/constraints/weights/mode，可附 `coord_start/coord_end`。响应为路径包（字段同 /chat 的 path_planning 形态）+ `explanation`。

### POST /api/route/replan

已保存路线任务的**方式/策略切换重算**，不经过 LLM。请求体含 `route_state`（带 `data_version`）与 `change`；数据版本不一致时返回 409 `route_state_version_conflict`，路况不可用时拒绝重算。前端切换出行方式、策略按钮走此接口。

### POST /api/candidates

"只有起点 + 想去的类别"场景的候选 POI（路网距离排序，不经过 LLM）。

```json
// 请求
{"start": {"name": "珞珈门"}, "poi_type": "scenery", "keyword": "樱花"}
// 响应 data
{"candidates": [{"name": "樱花大道", "type": "scenery", "distance_m": 320}], "count": 5}
```

## POI

### GET /api/pois

Query 参数：`type`（类型）、`season`（spring/summer/autumn/winter）、`keyword`（模糊）。返回 `{"pois": [...]}`。

### GET /api/pois/&lt;name&gt;

名称/别名模糊查询，返回 `{"poi": {...}}`，未命中 404 `poi_not_found`。

## 天气

### GET /api/weather

```json
{
  "weather": "晴", "temperature": 23.5, "humidity": 61,
  "winddirection": "北", "windpower": "≤3", "reporttime": "...",
  "slippery": false, "hot": false, "low_visibility": false,
  "label": "适宜步行", "advice": "天气晴朗，适合步行赏景"
}
```

上游不可用时 503 `weather_unavailable`（不影响路径规划主流程）。

## 路况事件（5 个端点）

### GET /api/road-conditions

默认只返回**当前生效**事件（普通用户视角）：`{"conditions": [...], "count": n}`。
管理员（session 或 Token）加 `?all=1` 返回含 scheduled/expired 的全部事件并附 `status`。

### POST /api/road-conditions ⚠️ 管理员

事件由**服务端吸附到最近路段**（30m 容差），不接受手填 geometry/半径：

```json
{
  "type": "construction",
  "name": "樱花大道施工",
  "lng": 114.3651,
  "lat": 30.5362,
  "description": "施工期间禁止通行",
  "start_time": "2026-09-10T08:00",
  "end_time": "2026-09-12T18:00"
}
```

- `type`：`closure` 封闭 / `construction` 施工 / `flooding` 积水 / `accident` 事故 / `event` 活动
- 时间均可省略：省略 start=立即生效，省略 end=长期有效；支持 ISO 字符串或时间戳
- 201 返回 `{"condition": {...}, "snap": {...}}`

### GET /api/road-conditions/snap ⚠️ 管理员

`?lng=&lat=` 选点预览：把点击点吸附到最近路段，返回边标识、吸附点（GCJ-02）、道路名、几何，供前端画预览线。

### PATCH /api/road-conditions/&lt;id&gt; ⚠️ 管理员

支持 `{"action": "end"}`（立即结束，保留记录）、改名/描述、调整 start_time/end_time。

影像来源事件可单独更正绑定道路；服务端会重新吸附到当前路网，要求管理员填写核实依据，并保留旧路段、复核人和更正记录：

```json
{
  "road_point": {"lng": 114.3651, "lat": 30.5362},
  "field_confirmation": "重新核对原视频和地图后，确认该候选对应樱花大道北段。"
}
```

只允许用于尚未撤销的影像来源事件。离路网超过 30 米、缺少核实依据或事件不是影像来源时会拒绝更正；事件类型、影响方式及有效期保持不变。智能体路况查询只返回审核来源、观察时间、道路关联历史和有限证据摘要，不返回原始媒体路径。

### DELETE /api/road-conditions/&lt;id&gt; ⚠️ 管理员

彻底删除事件（前端退出管理模式时同步清除所有本地标记）。

## 管理员鉴权（双通道）

| 通道 | 方式 | 配置 |
|---|---|---|
| 网页 session | `POST /api/admin/login {"password": "..."}` 写 session cookie | 环境变量 `ROAD_CONDITION_ADMIN_PASSWORD`（未配置则登录入口关闭） |
| 系统对接 Token | 请求头 `X-Admin-Token: <token>` 或 `Authorization: Bearer <token>`，hmac 常量时间比较 | 环境变量 `ROAD_CONDITION_ADMIN_TOKEN` |

- `POST /api/admin/login` / `POST /api/admin/logout` / `GET /api/admin/status`（返回 `{is_admin, login_enabled}`）
- 写操作（POST/PATCH/DELETE/snap）任一通道通过即可；普通用户只读且只见到生效中事件。

### 管理员影像上传

图片继续通过 `POST /api/manager/vision-jobs` 上传，默认上限 24 MiB。视频使用分块续传，默认上限 1 GiB、每块 8 MiB；续传会话 24 小时未更新后过期。以下写接口需要管理员登录和 CSRF token。

- `POST /api/manager/vision-uploads`：JSON 提交 `filename`、`size`、必填 `captured_at`（Unix 秒或带时区 ISO 时间）、可选 `lng`、`lat`、`camera_stabilized` 和归一化 `observation_regions`，返回 `upload_id`、`offset`、`chunk_size`。经纬度可缺省，但发布路况前仍须在地图上确认具体道路。
- `GET /api/manager/vision-uploads/<upload_id>`：读取服务器确认的续传偏移 `offset`。
- `PUT /api/manager/vision-uploads/<upload_id>/chunks`：请求体为当前视频二进制分块，使用 `Upload-Offset` 请求头声明起始字节。服务端只接受与已确认偏移连续的分块。
- `POST /api/manager/vision-uploads/<upload_id>/complete`：校验文件头和完整长度后，将视频放入影像任务队列。
- `POST /api/manager/vision-jobs/<job_id>/cancel` 与 `/retry`：取消任务，或从最后完成的视频分段重新排队。
- `POST /api/manager/vision-jobs/<job_id>/candidates/<index>/review`：逐项确认或排除候选并保存审核依据。
- 影像来源路况事件必须关联已确认候选、拍摄时间、管理员选定的道路、现场核实依据和未来结束时间；超出配置时效的影像不能发布为当前路况。

## 行为埋点

### POST /api/telemetry

前端行为回流，**任何失败都返回 ok，绝不影响主流程**；事件追加写入 `data/telemetry.jsonl`。

```json
{"uid":"whu_uid","event":"navigation_started","route_id":"route-...","strategy":"scenery","strategy_source":"button","applied_weights":{"distance":0.5,"slope":0.1,"scenery":0.4}}
```

- 路线类事件必须同时带 `route_id`、`strategy`、`strategy_source` 和 `applied_weights`。
- `route_shown`、`strategy_selected`、`strategy_abandoned` 只计行为，不学习权重。
- `navigation_started` / `navigation_completed` 仅对用户明确选择的非最短休闲策略学习；同一 `route_id` 最多更新一次。

## 路网

### POST /api/network/init

触发路网加载（生产启动时已后台预热）。`{"force": true}` 强制重新下载。返回 `{status, nodes, edges, cached}`。

## 出行方式约定

所有规划调用支持 `travel_mode` / `mode = walk | bike | drive`（默认 walk）。服务端最终取值优先级：**NL 显式关键词（骑车/开车/步行，正则后处理纠偏） > body 参数 > 上下文继承 > 默认 walk**。

## 其他接口

- `GET /api/course-spatial-reference`：校方课程数据参考层（对照与研究中用）。
- `GET/POST /api/manager/vision-*`：管理端视觉任务（状态、任务列表/上传、详情/删除、媒体、复核），详见[视觉识别验收](development/19_视觉图像识别功能实施与验收_20261003.md)与[管理端验收](development/20_管理端双流程实施与验收_20261003.md)。
