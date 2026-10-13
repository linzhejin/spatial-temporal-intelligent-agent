"""
珞珈智行 (WHU-Walker) — RESTful API 路由

端点:
  POST /api/parse   — 自然语言 → 结构化任务意图
  POST /api/route   — 任务意图 → 多因素路径规划
  POST /api/chat    — 一站式 NL → 解析 + 路径 + 解释
  GET  /api/pois   — POI 列表（支持 type/season 筛选）
  GET  /api/pois/<name> — 单 POI 查询
  POST /api/network/init — 触发路网加载
  GET  /api/road-conditions — 路况事件列表
  POST /api/road-conditions — 新增路况事件
  DELETE /api/road-conditions/<id> — 删除路况事件
"""
import json
import hmac
import hashlib
import logging
import math
import mimetypes
import os
import re
import secrets
import shutil
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
from flask import Blueprint, request, jsonify, session, current_app, send_file
from werkzeug.utils import secure_filename

import config
from agents.parser import parse_query, detect_travel_mode
from agents.planner import PlannerError
from agents.routing_policy import select_route_strategy
from agents.route_state import (
    RouteStateVersionConflict,
    apply_change,
    build_route_state,
    current_data_version,
    current_road_condition_version,
    route_state_to_context,
    validate_route_state,
)
from agents.timings import normalize_timings
from agents.explainer import generate_explanation, generate_chat_response, generate_suggestions, generate_poi_guidance
from spatial.poi import get_poi, search_pois, list_all_pois, load_pois, find_poi_ambiguous, importance_score
from spatial.network import get_network, load_or_download_network, get_nearest_node, get_node_coords
from spatial.routing import (
    compute_route,
    _path_length,
    TRAVEL_MODES,
    normalize_mode,
    filter_graph_for_mode,
    build_turn_by_turn,
)
from spatial.coord_transform import gcj02_to_wgs84, wgs84_to_gcj02
from spatial.amap_poi import navigation_wgs
from spatial.road_conditions import (
    list_conditions, add_condition, remove_condition, update_condition,
    purge_revoked_conditions, snap_to_edge, CONDITION_LABELS, CONDITION_EFFECTS,
    SNAP_MAX_DIST_M, TRAVEL_MODE_KEYS, CONDITION_ACTIONS,
    RoadConditionsUnavailableError, VisionConditionAlreadyPublishedError,
)
from spatial import weather as weather_mod
from storage import vision_scene_repository

logger = logging.getLogger(__name__)

_ADMIN_IDLE_SECONDS = 30 * 60
_ADMIN_MAX_SECONDS = 8 * 60 * 60
_ADMIN_POLL_PATHS = {"/api/manager/vision-status", "/api/manager/vision-jobs"}
_ADMIN_PASSIVE_ENDPOINTS = {"api.manager_vision_media"}
_VISION_REVIEW_CANDIDATE_KINDS = {
    "vehicle_cluster_review", "possible_congestion", "possible_crowding", "possible_accident",
    "possible_flooding",
}
_VISION_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_VISION_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
_VISION_UPLOAD_TTL_SECONDS = 24 * 60 * 60
_VISION_SCENE_MAX_SIGNATURE_DISTANCE = 8
_VISION_SCENE_LINKED_REGION_KINDS = {"vehicle_lane", "pedestrian", "road_surface"}


class _VisionSceneError(ValueError):
    def __init__(self, code: str, message: str, status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


def _valid_scene_anchor(value) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("请先标注影像所在的校园区域。")
    try:
        lng, lat = float(value.get("lng")), float(value.get("lat"))
    except (TypeError, ValueError):
        raise ValueError("影像区域坐标无效。") from None
    bbox = config.WHU_BBOX
    if not (math.isfinite(lng) and math.isfinite(lat)
            and bbox["west"] - .01 <= lng <= bbox["east"] + .01
            and bbox["south"] - .01 <= lat <= bbox["north"] + .01):
        raise ValueError("观察点需要落在武汉大学校园范围附近。")
    return {"lng": lng, "lat": lat, "crs": "GCJ02"}


def _scene_public(scene: dict) -> dict:
    return {key: value for key, value in scene.items() if key != "created_by"}


def _parse_scene_regions(value) -> list[dict]:
    regions = _parse_vision_observation_regions(value)
    try:
        source_regions = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        source_regions = []
    source_by_id = {
        str(item.get("id")): item for item in source_regions
        if isinstance(item, dict)
    } if isinstance(source_regions, list) else {}
    for region in regions:
        source = source_by_id.get(region["id"], {})
        if isinstance(source.get("road_link"), dict):
            region["road_link"] = source["road_link"]
        if region["kind"] not in _VISION_SCENE_LINKED_REGION_KINDS:
            continue
        point = source.get("road_point_gcj")
        if not isinstance(point, dict):
            raise _VisionSceneError(
                "vision_scene_road_link_required",
                "机动车道、人行区域和路面分析区都要绑定对应道路。", 400,
            )
        try:
            lng, lat = float(point.get("lng")), float(point.get("lat"))
        except (TypeError, ValueError):
            raise _VisionSceneError("vision_scene_road_link_required", "观察区域道路点位无效。", 400) from None
        if not (math.isfinite(lng) and math.isfinite(lat)):
            raise _VisionSceneError("vision_scene_road_link_required", "观察区域道路点位无效。", 400)
        region["road_point_gcj"] = {"lng": lng, "lat": lat}
    return regions


def _resolve_vision_scene(scene_id: str, media_path: Path, media_kind: str, *,
                          anchor_override=None, camera_stabilized_override=None,
                          observation_regions_override=None) -> dict:
    """Load a reusable scene, verify its view signature and re-check each road link."""
    try:
        from storage import database
        from vision.scene_signature import media_signature, signature_hamming_distance

        database_url = current_app.config.get("DATABASE_URL")
        database.initialize(database_url)
        scene = vision_scene_repository.get_scene(database_url, scene_id)
        if not scene:
            raise _VisionSceneError("vision_scene_not_found", "观察场景不存在，请重新选择场景。", 404)
        actual_signature = media_signature(media_path, media_kind)
        distance = signature_hamming_distance(scene["frame_signature"], actual_signature)
        if distance > _VISION_SCENE_MAX_SIGNATURE_DISTANCE:
            raise _VisionSceneError(
                "vision_scene_view_mismatch",
                f"当前画面与“{scene['name']}”差异较大（指纹差异 {distance}/64），请重新圈选区域或换用匹配机位。",
            )
        anchor_gcj = _valid_scene_anchor(anchor_override if anchor_override is not None else scene.get("anchor_gcj"))
        camera_stabilized = (
            bool(camera_stabilized_override) if camera_stabilized_override is not None
            else bool(scene.get("camera_stabilized"))
        )
        regions = _parse_scene_regions(
            observation_regions_override if observation_regions_override is not None
            else scene.get("observation_regions"),
        )
        if not regions:
            raise _VisionSceneError("vision_scene_regions_required", "观察场景没有可用区域，请重新圈选。", 400)
        needs_graph = any(region["kind"] in _VISION_SCENE_LINKED_REGION_KINDS for region in regions)
        graph = _ensure_network()[0] if needs_graph else None
        for region in regions:
            if region["kind"] not in _VISION_SCENE_LINKED_REGION_KINDS:
                continue
            expected = region.get("road_link") or {}
            point = region["road_point_gcj"]
            bbox = config.WHU_BBOX
            if not (bbox["west"] - .01 <= point["lng"] <= bbox["east"] + .01
                    and bbox["south"] - .01 <= point["lat"] <= bbox["north"] + .01):
                raise _VisionSceneError("vision_scene_road_link_required", "观察区域道路点位不在校园范围附近。", 400)
            current = snap_to_edge(graph, point["lng"], point["lat"])
            if not current:
                raise _VisionSceneError("vision_scene_road_stale", "场景绑定的道路已无法匹配，请重新绑定后再用。")
            expected_edges = expected.get("edges")
            if not expected_edges or json.dumps(expected_edges, sort_keys=True) != json.dumps(current.get("edges"), sort_keys=True):
                raise _VisionSceneError("vision_scene_road_stale", "路网已变化，场景中的道路绑定已过期，请重新绑定。")
            region["road_link"] = {
                "edges": current.get("edges"), "road_name": current.get("road_name"),
                "geometry_gcj": current.get("geometry_gcj"),
                "snap_lng_gcj": current.get("snap_lng_gcj"),
                "snap_lat_gcj": current.get("snap_lat_gcj"),
                "dist_m": current.get("dist_m"),
                "chain_length_m": current.get("chain_length_m"),
            }
        return {
            "scene_id": str(scene["scene_id"]), "anchor_gcj": anchor_gcj,
            "camera_stabilized": camera_stabilized,
            "observation_regions": regions,
            "signature_distance": distance,
        }
    except _VisionSceneError:
        raise
    except (ValueError, RuntimeError):
        raise
    except Exception as error:
        logger.exception("读取或核对无人机观察场景失败")
        raise _VisionSceneError("vision_scene_unavailable", "观察场景暂时无法核对，请稍后重试。", 503) from error


def _valid_review_only_candidate(candidate) -> bool:
    return (
        isinstance(candidate, dict)
        and candidate.get("kind") in _VISION_REVIEW_CANDIDATE_KINDS
        and candidate.get("review_required") is True
        and candidate.get("auto_publish") is False
    )


def _parse_vision_observation_regions(value) -> list[dict]:
    """Validate normalized polygons supplied by the admin's media annotation UI."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("观察区域数据格式无效。") from error
    if not isinstance(value, list) or len(value) > 24:
        raise ValueError("最多可以标注 24 个观察区域。")
    allowed_kinds = {"vehicle_lane", "pedestrian", "road_surface", "parking", "exclude"}
    normalized = []
    seen_ids = set()
    for index, region in enumerate(value):
        if not isinstance(region, dict) or region.get("kind") not in allowed_kinds:
            raise ValueError("观察区域类型无效。")
        polygon = region.get("polygon")
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 128:
            raise ValueError("每个观察区域需要 3 到 128 个顶点。")
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("观察区域坐标无效。")
            x, y = point
            if (isinstance(x, bool) or isinstance(y, bool)
                    or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                    or not math.isfinite(x) or not math.isfinite(y)
                    or not 0 <= x <= 1 or not 0 <= y <= 1):
                raise ValueError("观察区域坐标必须是 0 到 1 之间的画面比例。")
            points.append([float(x), float(y)])
        region_id = str(region.get("id") or f"region-{index + 1}")[:64]
        if not region_id or region_id in seen_ids:
            raise ValueError("观察区域标识必须非空且唯一。")
        seen_ids.add(region_id)
        normalized.append({
            "id": region_id,
            "kind": region["kind"], "polygon": points,
        })
    if normalized and not any(region["kind"] != "exclude" for region in normalized):
        raise ValueError("排除区域不能单独作为识别范围，请再圈选道路、人行或停车区域。")
    return normalized


def _parse_vision_capture_time(value, *, required: bool) -> datetime:
    """Accept an epoch or timezone-aware ISO timestamp; persist every observation in UTC."""
    if value is None or value == "":
        if required:
            raise ValueError("请填写这段影像的实际拍摄时间。")
        return datetime.now(timezone.utc)
    timestamp = _parse_time_input(value)
    if timestamp is None or not math.isfinite(float(timestamp)):
        raise ValueError("影像拍摄时间无效。")
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)


_VISION_CAPTURE_TIME_SOURCES = {"unknown", "manual", "dji_srt"}
_VISION_TIMEZONE_NAME = re.compile(r"^[A-Za-z0-9_+./:-]{1,64}$")


def _parse_vision_capture_provenance(source, timezone_name) -> tuple[str, str | None]:
    """Validate audit metadata without using the timezone to reinterpret the UTC instant."""
    source = str(source or "unknown").strip().lower()
    if source not in _VISION_CAPTURE_TIME_SOURCES:
        raise ValueError("影像拍摄时间来源无效。")
    timezone_name = str(timezone_name or "").strip() or None
    if timezone_name and not _VISION_TIMEZONE_NAME.fullmatch(timezone_name):
        raise ValueError("影像拍摄时间时区格式无效。")
    if source in {"manual", "dji_srt"} and not timezone_name:
        raise ValueError("请记录拍摄时间对应的设备时区。")
    return source, timezone_name


def _clear_admin_session():
    for key in ("is_admin", "admin_csrf_token", "admin_issued_at", "admin_last_active_at"):
        session.pop(key, None)

# ===== 管理员鉴权 =====
def _is_admin() -> bool:
    return _admin_identity() is not None


def _admin_identity():
    """返回管理员来源标识：'web'（网页 session）/ 'token'（保卫部系统 Token）/ None。"""
    if session.get("is_admin"):
        now = time.time()
        issued = session.get("admin_issued_at")
        last_active = session.get("admin_last_active_at")
        if (not isinstance(issued, (int, float)) or not isinstance(last_active, (int, float))
                or now < issued or now < last_active
                or now - issued > _ADMIN_MAX_SECONDS
                or now - last_active > _ADMIN_IDLE_SECONDS):
            _clear_admin_session()
        else:
            if (request.path not in _ADMIN_POLL_PATHS
                    and request.endpoint not in _ADMIN_PASSIVE_ENDPOINTS):
                session["admin_last_active_at"] = now
            return "web"
    token = request.headers.get("X-Admin-Token", "").strip()
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
    if token and config.ROAD_CONDITION_ADMIN_TOKEN:
        import hmac as _hmac
        if _hmac.compare_digest(token, config.ROAD_CONDITION_ADMIN_TOKEN):
            return "token"
    return None


def _admin_login_enabled() -> bool:
    return bool(config.ROAD_CONDITION_ADMIN_PASSWORD)


def _require_admin():
    """写操作鉴权（函数式），未登录返回 (err_response, None)；成功返回 (None, identity)。"""
    identity = _admin_identity()
    if identity is None:
        return _err("unauthorized", "需要管理员权限，请先登录", 401)
    if identity == "web" and request.method in {"POST", "PATCH", "PUT", "DELETE"}:
        expected = session.get("admin_csrf_token", "")
        provided = request.headers.get("X-CSRF-Token", "")
        if not expected or not provided or not hmac.compare_digest(expected, provided):
            return _err("csrf_failed", "管理操作校验已过期，请刷新管理页面后重试。", 403)
    return None

api_bp = Blueprint("api", __name__, url_prefix="/api")

_network_initialized = False

# 快捷模式预设（与前端 quick-chip 的 data-mode 一一对应）：
#   distance_first 最短路径 / scenery_first 风景优先 / slope_avoid 平坦优先（避开陡坡）
SHORTCUT_MODE_PRESETS = {
    "distance_first": {
        "strategy": "shortest",
        "weights": {"distance": 1.0, "slope": 0.0, "scenery": 0.0},
        "constraints": {"distance": "short", "slope": "normal", "scenery": "normal"},
    },
    "scenery_first": {
        "strategy": "scenery",
        "weights": {"distance": 0.5, "slope": 0.1, "scenery": 0.4},
        "constraints": {"distance": "medium", "slope": "normal", "scenery": "high"},
    },
    "slope_avoid": {
        "strategy": "flat",
        "weights": {"distance": 0.5, "slope": 0.4, "scenery": 0.1},
        "constraints": {"distance": "medium", "slope": "avoid", "scenery": "normal"},
    },
}


def _ok(data, status=200):
    resp = {"data": data}
    return jsonify(resp), status


def _err(code, message, status):
    return jsonify({"error": code, "message": message}), status


def _current_data_version(G=None):
    return current_data_version(G)


def _current_road_condition_version(conditions=None):
    if conditions is None:
        conditions = list_conditions(strict=True)
    return current_road_condition_version(conditions)


# ====== WGS-84 → GCJ-02 转换（前端高德底图用 GCJ-02）======
# 后端路网/路径坐标来自 OSM(WGS-84)，直接画在高德(GCJ-02)上会偏 50-100m
def _coords_wgs_to_gcj(coords_list):
    """路径坐标点列表 [{lng, lat}, ...] 整体转换"""
    if not coords_list:
        return coords_list
    return [{"lng": round(gcj_lng, 6), "lat": round(gcj_lat, 6)}
            for c in coords_list
            for gcj_lng, gcj_lat in [wgs84_to_gcj02(c["lng"], c["lat"])]]


def _build_steps_gcj(G, route_nodes, mode, end_name="", route_edges=None):
    """生成逐步转向指令，并把动作点坐标从 WGS-84 转成 GCJ-02（前端高德底图）。

    任何异常都不应阻断路径规划主流程，失败时返回空列表（前端降级为无指令导航）。
    """
    if not route_nodes or len(route_nodes) < 2:
        return []
    try:
        steps = build_turn_by_turn(
            G, route_nodes, mode=mode, end_name=end_name or "", route_edges=route_edges,
        )
    except Exception:
        logger.warning("转向指令生成失败 mode=%s", mode, exc_info=True)
        return []
    for s in steps:
        p = s.get("point")
        if p:
            g_lng, g_lat = wgs84_to_gcj02(p["lng"], p["lat"])
            s["point"] = {"lng": round(g_lng, 6), "lat": round(g_lat, 6)}
    return steps


def _pois_wgs_to_gcj(pois_list):
    """沿途 POI 列表 [{"lng", "lat", ...}, ...] 整体转换"""
    if not pois_list:
        return pois_list
    result = []
    for p in pois_list:
        gcj_lng, gcj_lat = wgs84_to_gcj02(p.get("lng", 0), p.get("lat", 0))
        new_p = dict(p)
        new_p["lng"] = round(gcj_lng, 6)
        new_p["lat"] = round(gcj_lat, 6)
        result.append(new_p)
    return result


def _resolve_travel_mode(query=None, body_mode=None, intent_mode=None):
    """确定最终出行方式（walk/bike/drive）。

    优先级：
      1. NL 显式关键词（detect_travel_mode explicit=True）
      2. body.travel_mode（经 normalize_mode，非法值兜底为 walk）
      3. intent.mode（LLM 输出 / 多轮上下文继承）
      4. 默认 "walk"
    """
    if query:
        nl_mode, explicit = detect_travel_mode(query)
        if explicit:
            return normalize_mode(nl_mode)
    if body_mode is not None:
        return normalize_mode(body_mode)
    if intent_mode is not None:
        return normalize_mode(intent_mode)
    return "walk"


def _weights_for_destination(end_poi, mode, explicit_weights=None):
    """显式偏好优先；通勤统一默认值，不根据目的地是否为景点改变权重。"""
    if explicit_weights is not None:
        return explicit_weights
    mode = normalize_mode(mode)
    return {"distance": 1.0, "slope": 0.0, "scenery": 0.0}


def _mode_filtered_graph(G, mode):
    """按出行方式过滤路网，返回 (G_mode, status, penalty_map)。

    起终点 snap 用 G_mode（驾车时吸附到最近车行节点）；路径坐标展开/POI 沿途
    检索仍用原图 G（副本节点 id 与 geometry 与原图一致）。
    """
    return filter_graph_for_mode(G, mode)


def _apply_coord_override(intent, coord_start, coord_end):
    """把前端 GPS 定位坐标（WGS-84）覆盖到解析意图的起/终点。

    触发场景：用户说「从我这到樱顶」「到我这来」等，浏览器定位在前端完成，
    以 coord_start/coord_end 随请求上送，坐标即 WGS-84，与路网同源、无需转换。
    仅接受数值合法的坐标；静默忽略非法值（不影响其余解析结果）。
    """
    from agents.parser import PoiRef

    for role, coord in (("start", coord_start), ("end", coord_end)):
        if not isinstance(coord, dict):
            continue
        try:
            lng = float(coord.get("lng"))
            lat = float(coord.get("lat"))
        except (TypeError, ValueError):
            continue
        if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
            continue
        label = str(coord.get("name") or "我的位置")[:20]
        setattr(intent, role, PoiRef(
            type="coord",
            name=label,
            coordinates={"lng": round(lng, 6), "lat": round(lat, 6)},
        ))


def _endpoint_to_wgs(ref, fallback_poi, mode='walk'):
    """把 PoiRef 解析为 (lon_wgs, lat_wgs) 供 nearest_node 使用。

    - type="coord"：坐标本身即 WGS-84（GPS 定位），直接使用；
    - type="poi"（默认）：POI 存 GCJ-02，需转 WGS-84。
    """
    if isinstance(ref, dict) and ref.get("type") == "coord" and ref.get("coordinates"):
        c = ref["coordinates"]
        return float(c["lng"]), float(c["lat"])
    return navigation_wgs(fallback_poi, mode)


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


_MAX_ROUTE_ENDPOINT_SNAP_M = 80.0


def _route_endpoint_access(G_mode, node, lng_wgs, lat_wgs, label):
    node_lng, node_lat = get_node_coords(G_mode, node)
    distance = _haversine(lat_wgs, lng_wgs, node_lat, node_lng)
    if not math.isfinite(distance) or distance > _MAX_ROUTE_ENDPOINT_SNAP_M:
        return None, _err(
            "endpoint_too_far_from_network",
            f"{label}距可规划道路约 {round(distance) if math.isfinite(distance) else '未知'} 米，末端接驳无法可靠确认。请在地图上选择附近道路或校门。",
            422,
        )
    return {
        "snap_distance_m": round(distance, 1),
        "status": "unverified_long_connector" if distance > 60.0
                  else "unverified_nearby_network_node",
        "access_link_verified": False,
        "access_link_basis": "straight_line_to_nearest_routable_network_node",
        "note": "末端直线接驳未核实是否存在实际通行连接。",
    }, None


def _node_to_coord(G, node_id):
    lng, lat = get_node_coords(G, node_id)
    return {"node_id": int(node_id), "lng": round(lng, 6), "lat": round(lat, 6)}


def _parse_linestring(wkt):
    """解析 WKT LINESTRING 字符串为 [(lng, lat), ...] 坐标点列表。

    路网边的 geometry 以 WKT 字符串形式存储（如 "LINESTRING (114.36 30.53, ...)"），
    坐标顺序为 lng lat（经度 纬度）。解析失败返回 None。
    """
    if not wkt or not isinstance(wkt, str):
        return None
    m = re.match(r"LINESTRING\s*\((.*)\)", wkt.strip(), re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    pts = []
    for pair in m.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pts if pts else None


def _path_to_coords(G, route_nodes, route_edges=None):
    """把路径节点序列展开为密集坐标序列（含边的 geometry 中间点）。

    路网下载时用了 simplify=True，节点只保留交叉路口，弯曲道路的中间点
    保存在边的 geometry 属性里。若只返回节点坐标，前端用直线连接会丢失弯道
    细节（"曲线变直线"）。这里沿路径逐边展开 geometry 中间点，让渲染贴合道路。

    Args:
        G: 路网图
        route_nodes: [node_id, ...] 路径节点序列

    Returns:
        [{"lng": float, "lat": float}, ...] 密集坐标序列（不含 node_id，前端只画线）
    """
    if not route_nodes:
        return []

    coords = []
    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue

        data = (edge_data[route_edges[i][2]] if route_edges is not None
                else min(edge_data.values(), key=lambda d: d.get("length", float("inf"))))

        # 添加当前边起点（首段才加；后续段的起点已由上一段终点覆盖）
        if not coords:
            ulng, ulat = get_node_coords(G, u)
            coords.append({"lng": round(ulng, 6), "lat": round(ulat, 6)})

        geom = data.get("geometry")
        pts = _parse_linestring(geom) if geom else None
        if pts and len(pts) >= 2:
            ulng, ulat = get_node_coords(G, u)
            if (pts[0][0] - ulng) ** 2 + (pts[0][1] - ulat) ** 2 > (pts[-1][0] - ulng) ** 2 + (pts[-1][1] - ulat) ** 2:
                pts.reverse()
            # geometry 首尾点即 u/v，跳过首点，追加中间点与终点
            for lng, lat in pts[1:]:
                coords.append({"lng": round(lng, 6), "lat": round(lat, 6)})
        else:
            # 无 geometry（直线边）→ 直接用终点 v
            vlng, vlat = get_node_coords(G, v)
            coords.append({"lng": round(vlng, 6), "lat": round(vlat, 6)})

    return coords


def _compute_route_costs(G, route_nodes, weights, max_len=0.0, route_edges=None):
    if len(route_nodes) < 2:
        return {"distance": 0.0, "slope": 0.0, "scenery": 0.0}

    if max_len == 0:
        for u, v, data in G.edges(data=True):
            length = data.get("length", 0)
            if length > max_len:
                max_len = length

    if max_len == 0:
        max_len = 1.0

    d_cost = 0.0
    s_cost = 0.0
    v_cost = 0.0

    for i in range(len(route_nodes) - 1):
        u, v = route_nodes[i], route_nodes[i + 1]
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue

        data = (edge_data[route_edges[i][2]] if route_edges is not None
                else min(edge_data.values(), key=lambda d: d.get("length", float("inf"))))
        length = data.get("length", 0)
        norm_length = length / max_len

        slope = float(data.get("slope_level", 3)) / 5.0
        scenery = float(data.get("scenery_level", 3)) / 5.0

        d_cost += weights["distance"] * norm_length
        s_cost += weights["slope"] * slope
        v_cost += weights["scenery"] * (1.0 - scenery)

    return {
        "distance": round(d_cost, 4),
        "slope": round(s_cost, 4),
        "scenery": round(v_cost, 4),
    }


def _find_pois_along_route(G, route_nodes, threshold_m=100.0, limit=8, min_importance=8.0):
    all_pois = load_pois()
    route_coords = []
    for nid in route_nodes:
        try:
            lng, lat = get_node_coords(G, nid)
            route_coords.append((lng, lat))
        except Exception:
            continue

    if not route_coords:
        return []

    along = []
    for poi in all_pois:
        coords = poi.get("coordinates", {})
        poi_lat = coords.get("lat", 0)
        poi_lng = coords.get("lng", 0)

        # GCJ-02 → WGS-84：POI 坐标转成 WGS-84 后与路网节点（WGS-84）比较距离
        poi_lng_wgs, poi_lat_wgs = gcj02_to_wgs84(poi_lng, poi_lat)

        min_dist = float("inf")
        for rlng, rlat in route_coords:
            dist = _haversine(poi_lat_wgs, poi_lng_wgs, rlat, rlng)
            if dist < min_dist:
                min_dist = dist

        if min_dist <= threshold_m:
            from spatial.poi import _flatten_poi
            imp = importance_score(poi)
            if imp < min_importance:
                continue  # 不重要的点（如普通宿舍）不标
            flat = _flatten_poi(poi)
            flat["distance_to_route_m"] = round(min_dist, 1)
            flat["importance"] = round(imp, 2)
            along.append(flat)

    # 按重要度降序（搜索频率 + 景观分 + 类型加权），只返回最重要的前 limit 个
    along.sort(key=lambda p: (-p["importance"], p["distance_to_route_m"]))
    return along[:limit]


def _ensure_network():
    global _network_initialized
    G = get_network()
    if G is not None:
        return G, None

    try:
        G = load_or_download_network()
        _network_initialized = True
        return G, None
    except RuntimeError as e:
        return None, _err("network_load_failed", f"路网加载失败: {e}", 500)


def _weather_snapshot():
    """获取实时天气 + 影响标志（带 30 分钟缓存，失败返回 None 不影响主流程）。"""
    try:
        live = weather_mod.fetch_weather_live()
        if not live:
            return None
        impact = weather_mod.classify_weather(live)
        return {"live": live, "impact": impact}
    except Exception as e:
        logger.warning("天气快照获取失败: %s", e)
        return None


def _weather_public(snap):
    """将天气快照转为前端可用的精简结构（无数据返回 None）。"""
    if not snap:
        return None
    live, impact = snap["live"], snap["impact"]
    return {
        "weather": live.get("weather", ""),
        "temperature": live.get("temperature"),
        "windpower": live.get("windpower", ""),
        "slippery": impact["slippery"],
        "hot": impact["hot"],
        "low_visibility": impact["low_visibility"],
        "label": impact["label"],
        "advice": impact["advice"],
    }


def _agent_response_to_legacy(resp: dict, coord_start=None, coord_end=None) -> dict:
    """Agent 循环响应 → 前端兼容结构（P4 前端再按 response_kind 细分渲染）。

    response_kind:
      route      → task_type=path_planning，透传路径包（含 via/tour 扩展字段）
      candidates → task_type=candidates，携带候选 POI 列表（P4 渲染卡片）
      clarify    → task_type=unknown，message 为澄清问题，附 clarify.options
      chat       → task_type=chat，reply 为答复
    """
    kind = resp.get("response_kind", "chat")
    route = resp.get("route")

    if kind == "route" and route:
        out = {
            "task_type": "path_planning",
            "response_kind": kind,
            "route_kind": resp.get("route_kind", "direct"),
            "explanation": resp.get("message", ""),
            "start": {"name": route.get("start_name", ""), "type": "coord" if coord_start else "poi"},
            "end": {"name": route.get("end_name", ""), "type": "coord" if coord_end else "poi"},
        }
        # 坐标端点：透传原始 GPS 坐标（前端地图聚焦用）
        if coord_start:
            out["start"]["coordinates"] = {"lng": coord_start.get("lng"), "lat": coord_start.get("lat")}
        if coord_end:
            out["end"]["coordinates"] = {"lng": coord_end.get("lng"), "lat": coord_end.get("lat")}
        for k in ("recommended", "shortest", "steps", "pois", "filter_status", "overlap_rate",
                  "recommended_length_m", "shortest_length_m", "length_capped", "degraded",
                  "distance_m", "shortest_distance_m", "applied_weights", "mode",
                  "duration_min", "shortest_duration_min", "speed_kmh",
                  "legs", "via", "tour", "detour_ratio", "strategy", "route_state",
                  "endpoint_access", "timings_ms"):
            if k in route:
                out[k] = route[k]
        out.setdefault("shortest", out.get("recommended", []))
        out.setdefault("costs", {})
        if resp.get("suggestions"):
            out["suggestions"] = resp["suggestions"]
        else:
            out.setdefault("suggestions", [])
        return out

    if kind == "candidates":
        return {
            "task_type": "candidates",
            "response_kind": kind,
            "message": resp.get("message", ""),
            "candidates": resp.get("candidates") or [],
            "place_request": resp.get("place_request"),
        }

    if kind == "clarify":
        clarify = resp.get("clarify") or {}
        return {
            "task_type": "unknown",
            "response_kind": kind,
            "message": clarify.get("question") or resp.get("message", ""),
            "clarify": clarify,
            "place_request": resp.get("place_request"),
        }

    return {
        "task_type": "chat",
        "response_kind": kind,
        "query": "",
        "reply": resp.get("message", ""),
    }


@api_bp.route("/weather", methods=["GET"])
def weather():
    """GET /api/weather — 当前武汉实时天气及对步行/骑行的路况影响"""
    snap = _weather_snapshot()
    if not snap:
        return _err("weather_unavailable", "天气暂时获取不到", 503)
    return _ok({
        "weather": snap["live"]["weather"],
        "temperature": snap["live"]["temperature"],
        "humidity": snap["live"]["humidity"],
        "winddirection": snap["live"]["winddirection"],
        "windpower": snap["live"]["windpower"],
        "reporttime": snap["live"]["reporttime"],
        "slippery": snap["impact"]["slippery"],
        "hot": snap["impact"]["hot"],
        "low_visibility": snap["impact"]["low_visibility"],
        "label": snap["impact"]["label"],
        "advice": snap["impact"]["advice"],
    })


# ===== 管理员鉴权接口 =====
@api_bp.route("/admin/status", methods=["GET"])
def admin_status():
    """GET /api/admin/status — 是否已登录管理员"""
    identity = _admin_identity()
    return _ok({
        "is_admin": identity is not None,
        "login_enabled": _admin_login_enabled(),
        "csrf_token": session.get("admin_csrf_token") if identity == "web" else None,
    })


@api_bp.route("/admin/login", methods=["POST"])
def admin_login():
    """POST /api/admin/login — 管理员登录 {password}"""
    if not _admin_login_enabled():
        return _err("admin_disabled", "管理员功能未配置", 403)
    body = request.get_json(silent=True) or {}
    password = str(body.get("password", "")).strip()
    if hmac.compare_digest(password, config.ROAD_CONDITION_ADMIN_PASSWORD):
        _clear_admin_session()
        session["is_admin"] = True
        session["admin_csrf_token"] = secrets.token_urlsafe(32)
        session["admin_issued_at"] = time.time()
        session["admin_last_active_at"] = session["admin_issued_at"]
        session.permanent = True
        return _ok({"is_admin": True, "csrf_token": session["admin_csrf_token"]})
    return _err("invalid_password", "密码错误", 401)


@api_bp.route("/admin/logout", methods=["POST"])
def admin_logout():
    """POST /api/admin/logout — 退出管理员登录"""
    _clear_admin_session()
    return _ok({"is_admin": False})


@api_bp.route("/parse", methods=["POST"])
def parse():
    """POST /api/parse — 自然语言 → 任务意图

    入参 (二选一):
      1) NL 模式:  {"query": "...", "input_method": "nl"}
      2) 快捷按钮: {"start": {...}, "end": {...}, "mode": "...", "input_method": "shortcut"}

    出参:
      {task_type, start, end, constraints, weights, input_method, ambiguity}
    """
    body = request.get_json(silent=True)
    if body is None:
        return _err("invalid_json", "请求体必须为合法 JSON", 400)

    query = body.get("query")
    input_method = body.get("input_method", "nl")

    if input_method == "nl":
        if not query or not isinstance(query, str):
            return _err("missing_query", "NL 模式下 query 字段必填且为字符串", 400)
        if len(query) > 500:
            query = query[:500]
    elif input_method == "shortcut":
        start = body.get("start")
        end = body.get("end")
        mode = body.get("mode", "")
        if not start or not end:
            return _err("missing_endpoints", "快捷模式下 start 和 end 字段必填", 400)

        # 快捷按钮模式 → 直接构建结构化意图，不调 LLM（DEC-011 权重映射）
        preset = SHORTCUT_MODE_PRESETS.get(mode, SHORTCUT_MODE_PRESETS["distance_first"])

        return _ok({
            "task_type": "path_planning",
            "start": start,
            "end": end,
            "constraints": preset["constraints"],
            "weights": preset["weights"],
            "strategy_hint": preset["strategy"],
            # 出行方式：快捷按钮链路携带 travel_mode（步行/骑行/驾车），非法值兜底 walk
            "mode": normalize_mode(body.get("travel_mode")),
            "input_method": "shortcut",
            "ambiguity": None,
            "weight_source": "shortcut",
        })
    else:
        return _err("invalid_input_method", "input_method 必须为 'nl' 或 'shortcut'", 400)

    try:
        intent = parse_query(query)
        intent_data = intent.model_dump() if hasattr(intent, "model_dump") else intent.dict()
        return _ok(intent_data)
    except ValueError as e:
        return _err("parse_validation_error", str(e), 400)
    except Exception as e:
        logger.exception("NL 解析失败")
        return _err("parse_failed", f"NL 解析失败: {e}", 500)


@api_bp.route("/route", methods=["POST"])
def route():
    """POST /api/route — 任务意图 → 多因素路径规划

    入参:
      {
        "task_type": "path_planning",
        "start": {"name": "...", "type": "poi"},
        "end": {"name": "...", "type": "poi"},
        "constraints": {"distance": "...", "slope": "...", "scenery": "..."},
        "weights": {"distance": 0.5, "slope": 0.2, "scenery": 0.3} | null
      }

    出参:
      {
        recommended: [{node_id, lng, lat}, ...],
        shortest: [{node_id, lng, lat}, ...],
        costs: {distance, slope, scenery},
        pois: [...],
        filter_status: str,
        overlap_rate: float,
        recommended_length_m: float,
        shortest_length_m: float,
        length_capped: bool,
        degraded: bool,
        explanation: str
      }
    """
    body = request.get_json(silent=True)
    if body is None:
        return _err("invalid_json", "请求体必须为合法 JSON", 400)

    start = body.get("start")
    end = body.get("end")
    if not start or not end:
        return _err("missing_endpoints", "start 和 end 字段必填", 400)

    request_started = time.perf_counter()
    G, err = _ensure_network()
    if err:
        return err

    # 出行方式：优先 body.travel_mode；兼容 /api/parse 返回体里的 mode 字段
    # （注意与快捷预设 distance_first 等区分：只有值在 TRAVEL_MODES 内才采纳）
    body_mode = body.get("travel_mode")
    if body_mode is None and body.get("mode") in TRAVEL_MODES:
        body_mode = body.get("mode")
    final_mode = _resolve_travel_mode(query=None, body_mode=body_mode, intent_mode=None)

    # 按出行方式过滤路网：snap 用过滤后的图（驾车吸附到最近车行节点）；
    # 坐标展开/沿途 POI 仍用原图 G（副本节点 id 与 geometry 一致）
    G_mode, _mode_status, _mode_penalty = _mode_filtered_graph(G, final_mode)
    api_graph_prepare_ms = (time.perf_counter() - request_started) * 1000
    endpoint_access = {}

    def _record_endpoint_access(label, display_name, node, lng_wgs, lat_wgs):
        evidence, error_response = _route_endpoint_access(
            G_mode, node, lng_wgs, lat_wgs, f"{label}「{display_name}」",
        )
        if error_response:
            return error_response
        evidence["name"] = display_name
        endpoint_access[label] = evidence
        return None

    def _resolve_endpoint(ep, label):
        """端点 → (node, display_name, poi_or_None, error_response_or_None)。

        支持两种端点：
          - POI：{"name": "樱顶"} → 查 pois.json，坐标 GCJ-02 转 WGS-84
          - 坐标：{"type": "coord", "coordinates": {"lng","lat"}} → 直接 WGS-84 snap
        """
        coords = ep.get("coordinates")
        is_coord = ep.get("type") == "coord" or (coords and coords.get("lng") is not None)
        if is_coord:
            if not coords:
                coords = ep
            try:
                lng = float(coords.get("lng", ep.get("lng")))
                lat = float(coords.get("lat", ep.get("lat")))
            except (TypeError, ValueError):
                return None, None, None, _err("invalid_coordinates", f"{label}坐标无效", 400)
            if not (math.isfinite(lng) and math.isfinite(lat)
                    and -180 <= lng <= 180 and -90 <= lat <= 90):
                return None, None, None, _err("invalid_coordinates", f"{label}坐标超出有效范围", 400)
            try:
                node = get_nearest_node(G_mode, lng, lat)
            except RuntimeError as e:
                return None, None, None, _err("nearest_node_failed", f"{label}最近节点查找失败: {e}", 500)
            evidence_error = _record_endpoint_access(label, ep.get("name") or label, node, lng, lat)
            if evidence_error:
                return None, None, None, evidence_error
            return node, ep.get("name") or label, None, None

        name = ep.get("name")
        if not name:
            return None, None, None, _err("missing_poi_names", "端点需提供 name 或 coordinates", 400)
        poi = get_poi(name)
        if poi is None:
            return None, None, None, _err("poi_not_found", f"{label} '{name}' 未找到", 404)
        lon_wgs, lat_wgs = navigation_wgs(poi, final_mode)
        try:
            node = get_nearest_node(G_mode, lon_wgs, lat_wgs)
        except RuntimeError as e:
            return None, None, None, _err("nearest_node_failed", f"{label}最近节点查找失败: {e}", 500)
        evidence_error = _record_endpoint_access(label, poi["name"], node, lon_wgs, lat_wgs)
        if evidence_error:
            return None, None, None, evidence_error
        return node, poi["name"], poi, None

    poi_started = time.perf_counter()
    start_node, start_name, start_poi, start_err = _resolve_endpoint(start, "起点")
    if start_err:
        return start_err
    end_node, end_name, end_poi, end_err = _resolve_endpoint(end, "终点")
    if end_err:
        return end_err
    poi_resolution_ms = (time.perf_counter() - poi_started) * 1000

    # 起终点同名 POI 才拦截（坐标端点可能恰好重合，交给 compute_route 处理）
    if start_poi and end_poi and start_poi["name"] == end_poi["name"]:
        return _err("same_poi", "起点和终点相同，请选择不同的地点", 400)

    constraints = body.get("constraints", {})
    # 路线策略由统一策略中心决定，解析层和终点类型不再自行发明数值权重。
    raw_weights = body.get("weights")
    explicit_strategy = body.get("strategy") or body.get("strategy_hint")
    if explicit_strategy is None and raw_weights is not None:
        explicit_strategy = "custom"
    decision = select_route_strategy(
        query=body.get("query", ""),
        end_poi=end_poi,
        explicit_strategy=explicit_strategy,
        strategy_source=body.get("strategy_source", "explicit_nl"),
        custom_weights=raw_weights,
    )
    weights = decision.weights

    try:
        route_conditions = list_conditions(strict=True)
    except RoadConditionsUnavailableError:
        logger.exception("路况快照不可用，拒绝生成无法核验封路状态的路线")
        return _err("road_conditions_unavailable", "当前路况无法确认，暂时不能可靠规划路线", 503)

    # 实时天气：雨雪天自动避陡坡、高温倾向树荫（失败不影响规划）
    weather_snap = _weather_snapshot()
    weather_info = weather_snap["live"] if weather_snap else None

    try:
        route_result = compute_route(
            G=G,
            start_node=start_node,
            end_node=end_node,
            constraints=constraints,
            weights=weights,
            mode=final_mode,
            weather_info=weather_info,
            road_conditions=route_conditions,
            strategy_name=decision.name,
            detour_cap=decision.detour_cap,
        )
    except ValueError as e:
        # 如驾车不可达："驾车无法到达…建议切换骑行或步行"，消息原样透传给前端
        return _err("route_not_found", str(e), 404)
    except RoadConditionsUnavailableError:
        logger.exception("路况事件无法匹配当前路网，拒绝生成路线")
        return _err("road_conditions_unavailable", "当前管制信息无法与路网对应，暂时不能可靠规划路线", 503)
    except Exception as e:
        logger.exception("路径计算异常")
        return _err("route_computation_failed", f"路径计算失败: {e}", 500)

    response_started = time.perf_counter()
    recommended_nodes = route_result["recommended"]
    shortest_nodes = route_result["shortest"]
    resolved_weights = route_result["applied_weights"]

    recommended_edges = route_result["recommended_edges"]
    shortest_edges = route_result["shortest_edges"]
    recommended_coords = _path_to_coords(G, recommended_nodes, recommended_edges)
    shortest_coords = _path_to_coords(G, shortest_nodes, shortest_edges)

    costs = _compute_route_costs(
        G, recommended_nodes, resolved_weights, route_result.get("max_len", 0.0),
        recommended_edges,
    )

    pois_along = _find_pois_along_route(G, recommended_nodes)

    # WGS-84 → GCJ-02：路网路径坐标转成高德坐标系再返回前端
    # 注意：pois_along 里的 POI 本身来自 pois.json(GCJ-02)，不需要再转！
    recommended_coords = _coords_wgs_to_gcj(recommended_coords)
    shortest_coords = _coords_wgs_to_gcj(shortest_coords)

    # 逐步转向指令（动作点同步转 GCJ-02），供前端实时导航与语音播报
    steps = _build_steps_gcj(
        G, recommended_nodes, final_mode, end_name=end_name,
        route_edges=recommended_edges,
    )

    response = {
        "recommended": recommended_coords,
        "shortest": shortest_coords,
        "recommended_edge_ids": recommended_edges,
        "shortest_edge_ids": shortest_edges,
        "steps": steps,
        "costs": costs,
        "pois": pois_along,
        "filter_status": route_result["filter_status"],
        "overlap_rate": route_result["overlap_rate"],
        "recommended_length_m": route_result["recommended_length_m"],
        "shortest_length_m": route_result["shortest_length_m"],
        "length_capped": route_result["length_capped"],
        "degraded": route_result["degraded"],
        "distance_m": route_result["recommended_length_m"],
        "shortest_distance_m": route_result["shortest_length_m"],
        "applied_weights": route_result.get("applied_weights", resolved_weights),
        "degraded_count": route_result.get("degraded_count", 0),
        # 出行方式与预计用时
        "mode": route_result["mode"],
        "duration_min": route_result["duration_min"],
        "shortest_duration_min": route_result["shortest_duration_min"],
        "speed_kmh": route_result["speed_kmh"],
        "road_conditions_applied": route_result.get("road_conditions_applied", 0),
        "weather_applied": route_result.get("weather_applied", False),
        "weather": _weather_public(weather_snap),
        "road_condition_version": _current_road_condition_version(route_conditions),
        "endpoint_access": {**endpoint_access, "max_snap_m": _MAX_ROUTE_ENDPOINT_SNAP_M},
        "strategy": decision.as_dict(),
    }

    response["route_state"] = build_route_state(
        route_kind="direct",
        original_query=body.get("query", ""),
        start=start,
        end=end,
        travel_mode=final_mode,
        hard_constraints=constraints,
        strategy=decision.as_dict(),
        data_version=_current_data_version(G),
        road_condition_version=_current_road_condition_version(route_conditions),
    )

    timings = normalize_timings(route_result.get("timings_ms"))
    timings["graph_prepare"] += api_graph_prepare_ms
    timings["poi_resolution"] += poi_resolution_ms
    timings["response_build"] += (time.perf_counter() - response_started) * 1000
    response["timings_ms"] = normalize_timings(timings, agent=0.0)

    return _ok(response)


def _replan_tool_request(state):
    """Convert validated route semantics into one deterministic tool request."""
    common = {
        "start": state["start"],
        "constraints": state["hard_constraints"],
        "strategy": state["strategy"]["name"],
        "strategy_source": state["strategy"].get("source", "button"),
    }
    if state["strategy"]["name"] == "custom":
        common["weights"] = state["strategy"].get("weights")

    kind = state["route_kind"]
    if kind == "direct":
        return "plan_route", {
            **common,
            "end": state["end"],
            "mode": state["travel_mode"],
        }
    if kind == "via":
        via = state["via"]
        args = {
            **common,
            "end": state["end"],
            "mode": state["travel_mode"],
        }
        if via.get("type") == "multi":
            args["via_points"] = via.get("points", [])
            return "plan_via_route", args
        coords = via.get("coordinates") if isinstance(via, dict) else None
        if coords:
            args["via_coord"] = coords
        else:
            args["via_name"] = via.get("name", "")
        return "plan_via_route", args
    if kind == "itinerary":
        via = state["via"]
        stops = via.get("points", []) if via.get("type") == "multi" else [via]
        itinerary = state["itinerary"]
        return "plan_itinerary", {
            **common,
            "end": state["end"],
            "mode": state["travel_mode"],
            "stops": stops,
            "time_budget_min": itinerary["time_budget_min"],
            "stop_duration_min": itinerary["stop_duration_min"],
        }
    if kind == "tour":
        tour = state["tour"]
        return "plan_tour", {
            **common,
            "mode": state["travel_mode"],
            "theme": tour.get("theme", "scenery"),
            "loop": bool(tour.get("loop", True)),
            "poi_names": [p.get("name") for p in tour.get("pois", []) if p.get("name")],
            "max_pois": max(2, len(tour.get("pois", []))),
        }

    legs = []
    for leg in state["legs"]:
        legs.append({
            "end": leg.get("end"),
            "mode": leg.get("travel_mode") or leg.get("mode") or state["travel_mode"],
            **({"via_name": leg["via"]["name"]}
               if isinstance(leg.get("via"), dict) and leg["via"].get("name") else {}),
        })
    return "plan_multimodal_route", {**common, "legs": legs}


@api_bp.route("/route/replan", methods=["POST"])
def replan_route():
    """Recompute a route-state change without invoking the LLM."""
    body = request.get_json(silent=True) or {}
    try:
        prior = validate_route_state(body.get("route_state"))
        current_data_version = _current_data_version()
        if prior.get("data_version") != current_data_version:
            raise RouteStateVersionConflict(
                "路线数据已更新，请按当前地点重新规划"
            )
        requested = apply_change(prior, body.get("change"))
        prior_strategy = requested["strategy"]
        decision = select_route_strategy(
            query=requested["original_query"],
            explicit_strategy=prior_strategy["name"],
            strategy_source=prior_strategy.get("source", "button"),
            custom_weights=prior_strategy.get("weights"),
        )
        requested["strategy"] = decision.as_dict()
        route_conditions = list_conditions(strict=True)
        requested["road_condition_version"] = _current_road_condition_version(route_conditions)
    except RouteStateVersionConflict as exc:
        return _err("route_state_version_conflict", str(exc), 409)
    except RoadConditionsUnavailableError:
        logger.exception("路况快照不可用，拒绝重新规划")
        return _err("road_conditions_unavailable", "当前路况无法确认，暂时不能可靠重新规划路线", 503)
    except ValueError as exc:
        return _err("invalid_route_state", str(exc), 400)

    from agents.tools import execute_tool

    tool_name, args = _replan_tool_request(requested)
    result, artifact = execute_tool(
        tool_name,
        args,
        {
            "query": requested["original_query"], "replan": True,
            "_route_conditions_snapshot": route_conditions,
        },
    )
    if result.get("error"):
        status = 503 if result["error"] == "road_conditions_unavailable" else 404
        return _err(result["error"], result.get("message", "路线重新规划失败"), status)
    if requested["route_kind"] == "itinerary" and result.get("status") != "feasible":
        if result.get("status") == "over_budget":
            return _err(
                "itinerary_over_budget",
                f"当前路线预计超出行程预算 {result.get('over_by_min', 0)} 分钟，请增加时间或减少途经点。",
                422,
            )
        return _err(
            "itinerary_validation_failed",
            "重算后的路线缺少有效的时间预算校验，未返回路线。",
            503,
        )
    payload = (artifact or {}).get("route") or result
    payload["route_state"] = validate_route_state(requested)
    if requested["route_kind"] == "itinerary":
        payload["route_kind"] = "itinerary"
    payload["timings_ms"] = normalize_timings(payload.get("timings_ms"), agent=0.0)
    return _ok(payload)


def _normalize_chat_context(context):
    """Use route_state as the source of truth for completed-route continuations."""
    if not isinstance(context, dict):
        return context
    raw_state = context.get("previous_route_state")
    if not raw_state:
        return context
    try:
        projected = route_state_to_context(raw_state)
    except (TypeError, ValueError):
        logger.warning("忽略无效 previous_route_state")
        clean = dict(context)
        clean.pop("previous_route_state", None)
        return clean

    merged = dict(context)
    merged["previous_route_state"] = projected["previous_route_state"]
    # A pending ambiguity is newer than the last completed route. Otherwise the
    # canonical state supplies every legacy parser field from one snapshot.
    if not isinstance(merged.get("previous_intent"), dict):
        merged.update(projected)
    return merged


@api_bp.route("/chat", methods=["POST"])
def chat():
    """POST /api/chat — 一站式 NL → 完整流程（解析 + 路径 + 解释）

    入参:
      {
        "query": "我第一次来武大，想看樱花...",
        "context": {...}  // 可选，多轮对话上下文
      }

    出参:
      {
        "task_type": "...",
        "start": {...},
        "end": {...},
        "constraints": {...},
        "weights": {...},
        "recommended": [...],
        "shortest": [...],
        "costs": {...},
        "pois": [...],
        "explanation": "...",
        "filter_status": "...",
        "overlap_rate": 0.35,
        ...
      }
    """
    body = request.get_json(silent=True)
    if body is None:
        return _err("invalid_json", "请求体必须为合法 JSON", 400)

    query = body.get("query")
    if not query or not isinstance(query, str):
        return _err("missing_query", "query 字段必填且为字符串", 400)
    if len(query) > 500:
        query = query[:500]

    context = _normalize_chat_context(body.get("context"))
    agent_started = time.perf_counter()
    fallback_agent_ms = 0.0

    # v2 全 Agent 架构：所有输入优先进入 Agent 循环（LLM 决策 + 工具执行）。
    # LLM 本身故障（断网/鉴权/超时）时落回旧管道——停电保险，不是备用通道。
    try:
        from agents.planner import run_agent
        agent_resp = run_agent(
            query,
            context=context,
            coord_start=body.get("coord_start"),
            coord_end=body.get("coord_end"),
            uid=body.get("whu_uid") or body.get("uid"),
            travel_mode=body.get("travel_mode"),
            coord_waypoints=body.get("coord_waypoints"),
        )
        return _ok(_agent_response_to_legacy(agent_resp,
                    coord_start=body.get("coord_start"),
                    coord_end=body.get("coord_end")))
    except PlannerError as e:
        fallback_agent_ms = (time.perf_counter() - agent_started) * 1000
        logger.warning("Agent 规划器不可用（%s: %s），落回旧管道", type(e).__name__, e)
    except Exception:
        logger.exception("Agent 规划处理发生内部错误")
        return _err("internal_error", "路线规划服务暂时异常，请稍后重试。", 500)

    try:
        intent = parse_query(query, context)
        # GPS 坐标覆盖（WGS-84；前端「从我这/到我这」触发），必须在 model_dump 前完成
        _apply_coord_override(intent, body.get("coord_start"), body.get("coord_end"))
        intent_data = intent.model_dump() if hasattr(intent, "model_dump") else intent.dict()
    except ValueError as e:
        return _err("parse_validation_error", str(e), 400)
    except Exception as e:
        logger.exception("NL 解析失败")
        return _err("parse_failed", f"NL 解析失败: {e}", 500)

    task_type = intent_data.get("task_type")

    # poi_query → 返回景点详情
    if task_type == "poi_query":
        start = intent_data.get("start")
        poi_name = start.get("name") if start else None
        if poi_name:
            poi, alts = find_poi_ambiguous(poi_name)
            if poi:
                desc = (poi.get("description") or "").strip()
                message = desc if desc else f"这是 {poi['name']} 的信息～"
                return _ok({
                    "task_type": "poi_query",
                    "poi": poi,
                    "message": message,
                    # 供前端多轮上下文保存：问「X在哪」后，下一轮「从A怎么去」可把 X 当作终点承接
                    "start": start,
                })
            guidance = generate_poi_guidance(query, poi_name, alternatives=alts)
        else:
            guidance = generate_poi_guidance(query, poi_name)
        return _ok({
            "task_type": "unknown",
            "message": guidance,
            "example_queries": ["樱顶在哪", "从珞珈门到樱顶"],
        })

    # help → 返回功能介绍
    if task_type == "help":
        return _ok({
            "task_type": "help",
            "message": "我可以帮你规划武大校园路线、查询景点、回答校园问题～",
            "features": ["路径规划（支持避开陡坡/风景优先/最短路径）", "景点查询与介绍", "校园生活问答"],
            "example_queries": ["从珞珈门到樱顶，避开陡坡", "樱花开了吗", "哪个食堂好吃"],
        })

    # unknown → 返回引导
    if task_type == "unknown":
        return _ok({
            "task_type": "unknown",
            "message": intent_data.get("ambiguity", "抱歉，我只能回答武大校园相关的问题哦～"),
            "example_queries": ["从珞珈门到樱顶", "樱顶在哪", "樱花开了吗"],
        })

    # chat → 调用 LLM 闲聊回复
    if task_type == "chat":
        chat_reply = generate_chat_response(query)
        return _ok({
            "task_type": "chat",
            "query": query,
            "reply": chat_reply,
        })

    # path_planning → 继续现有逻辑
    if task_type != "path_planning":
        return _err("unsupported_task", f"暂不支持的任务类型: {task_type}", 400)

    start = intent_data.get("start")
    end = intent_data.get("end")
    if not start or not end:
        # 缺起终点不是"错误"，而是信息不完整——用对话式引导，而非报错弹窗
        if not start and not end:
            guide = "想从哪走到哪呢？告诉我起点和终点，我就能帮你规划啦～比如「从珞珈门到樱顶」😊"
            ambiguity = "请指定起点和终点"
        elif not start:
            guide = "从哪出发呢？告诉我起点就好啦～比如「从珞珈门出发」"
            ambiguity = "请指定起点"
        else:
            guide = "要去哪儿呢？告诉我目的地，我帮你规划路线～比如「到樱顶」"
            ambiguity = "请指定终点"
        return _ok({
            "task_type": "unknown",
            "message": guide,
            "start": start,
            "end": end,
            "constraints": intent_data.get("constraints", {}),
            "weights": intent_data.get("weights"),
            "ambiguity": ambiguity,
            "example_queries": ["从珞珈门到樱顶", "从教五到总图书馆", "去樱顶"],
        })

    G, err = _ensure_network()
    if err:
        return err

    # type="coord"（GPS「我的位置」）直接使用坐标，不走 POI 模糊匹配；
    # type="poi" 保持名称解析 + 多候选消歧引导
    if start.get("type") == "coord" and start.get("coordinates"):
        start_poi, start_alts = {"name": start.get("name") or "我的位置"}, []
    else:
        start_poi, start_alts = find_poi_ambiguous(start.get("name"))
        if start_poi is None:
            guidance = generate_poi_guidance(query, start.get("name"), alternatives=start_alts)
            return _ok({
                "task_type": "unknown",
                "message": guidance,
                "start": start,
                "end": end,
                "ambiguity": "请指定起点",
                "example_queries": ["从牌坊出发"],
            })

    if end.get("type") == "coord" and end.get("coordinates"):
        end_poi, end_alts = {"name": end.get("name") or "我的位置"}, []
    else:
        end_poi, end_alts = find_poi_ambiguous(end.get("name"))
        if end_poi is None:
            guidance = generate_poi_guidance(query, end.get("name"), alternatives=end_alts)
            return _ok({
                "task_type": "unknown",
                "message": guidance,
                "start": start,
                "end": end,
                "ambiguity": "请指定终点",
                "example_queries": ["到樱顶"],
            })

    # 出行方式优先级：NL 显式关键词（骑车/开车/步行…）> body.travel_mode > intent.mode（含上下文继承）> walk
    final_mode = _resolve_travel_mode(
        query=query,
        body_mode=body.get("travel_mode"),
        intent_mode=intent_data.get("mode"),
    )

    # 按出行方式过滤路网：snap 用过滤后的图（驾车吸附到最近车行节点）；
    # 坐标展开/沿途 POI 仍用原图 G（副本节点 id 与 geometry 一致）
    G_mode, _mode_status, _mode_penalty = _mode_filtered_graph(G, final_mode)

    # 坐标统一到 WGS-84：GPS coord 本身即 WGS-84；POI 为 GCJ-02 需转换
    start_lon_wgs, start_lat_wgs = _endpoint_to_wgs(start, start_poi, final_mode)
    end_lon_wgs, end_lat_wgs = _endpoint_to_wgs(end, end_poi, final_mode)
    if (not all(math.isfinite(value) for value in (
            start_lon_wgs, start_lat_wgs, end_lon_wgs, end_lat_wgs))
            or not (-180 <= start_lon_wgs <= 180 and -90 <= start_lat_wgs <= 90
                    and -180 <= end_lon_wgs <= 180 and -90 <= end_lat_wgs <= 90)):
        return _err("invalid_coordinates", "起点或终点坐标无效", 400)

    # 起终点重合判定：含坐标时看球面距离（<10m 视为同点）；POI-POI 看规范名
    if start.get("type") == "coord" or end.get("type") == "coord":
        if _haversine(start_lat_wgs, start_lon_wgs, end_lat_wgs, end_lon_wgs) < 10.0:
            return _err("same_poi", "起点和终点距离太近，换一个目的地试试", 400)
    elif start_poi.get("name") == end_poi.get("name"):
        return _err("same_poi", "起点和终点相同，请选择不同的地点", 400)

    try:
        start_node = get_nearest_node(G_mode, start_lon_wgs, start_lat_wgs)
    except RuntimeError as e:
        return _err("nearest_node_failed", f"起点最近节点查找失败: {e}", 500)

    try:
        end_node = get_nearest_node(G_mode, end_lon_wgs, end_lat_wgs)
    except RuntimeError as e:
        return _err("nearest_node_failed", f"终点最近节点查找失败: {e}", 500)

    start_access, start_access_error = _route_endpoint_access(
        G_mode, start_node, start_lon_wgs, start_lat_wgs, "起点",
    )
    if start_access_error:
        return start_access_error
    start_access["name"] = start.get("name") or "起点"
    end_access, end_access_error = _route_endpoint_access(
        G_mode, end_node, end_lon_wgs, end_lat_wgs, "终点",
    )
    if end_access_error:
        return end_access_error
    end_access["name"] = end.get("name") or "终点"

    constraints = intent_data.get("constraints", {})
    # 旧管道仅作为 Agent 不可用时的保险，也必须使用同一策略中心。
    raw_weights = intent_data.get("weights")
    explicit_strategy = intent_data.get("strategy_hint")
    if explicit_strategy is None and raw_weights is not None:
        explicit_strategy = "custom"
    decision = select_route_strategy(
        query=query,
        end_poi=end_poi,
        explicit_strategy=explicit_strategy,
        strategy_source=("explicit_nl" if explicit_strategy else "commute_default"),
        custom_weights=raw_weights,
    )
    weights = decision.weights

    try:
        route_conditions = list_conditions(strict=True)
    except RoadConditionsUnavailableError:
        logger.exception("路况快照不可用，拒绝生成无法核验封路状态的路线")
        return _err("road_conditions_unavailable", "当前路况无法确认，暂时不能可靠规划路线", 503)

    # 实时天气：雨雪天自动避陡坡、高温倾向树荫（失败不影响规划）
    weather_snap = _weather_snapshot()
    weather_info = weather_snap["live"] if weather_snap else None

    try:
        route_result = compute_route(
            G=G,
            start_node=start_node,
            end_node=end_node,
            constraints=constraints,
            weights=weights,
            mode=final_mode,
            weather_info=weather_info,
            road_conditions=route_conditions,
            strategy_name=decision.name,
            detour_cap=decision.detour_cap,
        )
    except ValueError as e:
        # 如驾车不可达："驾车无法到达…建议切换骑行或步行"，消息原样透传给前端
        return _err("route_not_found", str(e), 404)
    except RoadConditionsUnavailableError:
        logger.exception("回退规划无法将路况事件匹配到当前路网")
        return _err("road_conditions_unavailable", "当前管制信息无法与路网对应，暂时不能可靠规划路线", 503)
    except Exception as e:
        logger.exception("路径计算异常")
        return _err("route_computation_failed", f"路径计算失败: {e}", 500)

    response_started = time.perf_counter()
    recommended_nodes = route_result["recommended"]
    shortest_nodes = route_result["shortest"]
    resolved_weights = route_result["applied_weights"]

    recommended_edges = route_result["recommended_edges"]
    shortest_edges = route_result["shortest_edges"]
    recommended_coords = _path_to_coords(G, recommended_nodes, recommended_edges)
    shortest_coords = _path_to_coords(G, shortest_nodes, shortest_edges)

    costs = _compute_route_costs(
        G, recommended_nodes, resolved_weights, route_result.get("max_len", 0.0),
        recommended_edges,
    )

    pois_along = _find_pois_along_route(G, recommended_nodes)

    route_data_for_explainer = {
        "distance_m": route_result["recommended_length_m"],
        "shortest_distance_m": route_result["shortest_length_m"],
        "costs": costs,
        "pois": pois_along,
        "filter_status": route_result["filter_status"],
        "mode": final_mode,
        "duration_min": route_result["duration_min"],
    }

    explanation = ""
    try:
        weight_source = intent_data.get("weight_source")
        explanation = generate_explanation(route_data_for_explainer, constraints, weights, weight_source)
    except Exception:
        explanation = "已为您规划好路线。"

    # 天气影响提示：把真实天气对路线的调整说清楚（湿滑避坡/高温走树荫）
    weather_pub = _weather_public(weather_snap)
    if weather_pub and weather_pub.get("advice"):
        explanation = (explanation + " " + weather_pub["advice"]).strip()
    if any(item.get("status") == "unverified_long_connector"
           for item in (start_access, end_access)):
        explanation = (explanation + " 起点或终点离可规划道路较远，末端接驳尚未核实；建议在附近道路或校门重新选点。").strip()

    # 跟进建议（"可能想问"）：LLM 根据对话上下文动态生成，失败则空列表（前端兜底）
    suggestions = []
    try:
        suggestions = generate_suggestions(query, route_data_for_explainer, constraints, weights)
    except Exception as e:
        logger.warning("跟进建议生成异常: %s", e)

    result = {
        "task_type": intent_data.get("task_type"),
        "start": start,
        "end": end,
        "constraints": constraints,
        "weights": weights,
        # WGS-84 → GCJ-02：路网路径坐标转成高德坐标系再返回前端
        # 注意：pois_along 里的 POI 本身来自 pois.json(GCJ-02)，不需要再转！
        "recommended": _coords_wgs_to_gcj(recommended_coords),
        "shortest": _coords_wgs_to_gcj(shortest_coords),
        "recommended_edge_ids": recommended_edges,
        "shortest_edge_ids": shortest_edges,
        "steps": _build_steps_gcj(
            G, recommended_nodes, final_mode,
            end_name=(end_poi.get("name") if end_poi else end.get("name")) or "",
            route_edges=recommended_edges,
        ),
        "costs": costs,
        "pois": pois_along,
        "filter_status": route_result["filter_status"],
        "overlap_rate": route_result["overlap_rate"],
        "recommended_length_m": route_result["recommended_length_m"],
        "shortest_length_m": route_result["shortest_length_m"],
        "length_capped": route_result["length_capped"],
        "degraded": route_result["degraded"],
        "distance_m": route_result["recommended_length_m"],
        "shortest_distance_m": route_result["shortest_length_m"],
        "applied_weights": route_result.get("applied_weights", resolved_weights),
        "degraded_count": route_result.get("degraded_count", 0),
        # 出行方式与预计用时
        "mode": final_mode,
        "duration_min": route_result["duration_min"],
        "shortest_duration_min": route_result["shortest_duration_min"],
        "speed_kmh": route_result["speed_kmh"],
        "road_conditions_applied": route_result.get("road_conditions_applied", 0),
        "weather_applied": route_result.get("weather_applied", False),
        "weather": weather_pub,
        "explanation": explanation,
        "suggestions": suggestions,
        "endpoint_access": {
            "start": start_access,
            "end": end_access,
            "max_snap_m": _MAX_ROUTE_ENDPOINT_SNAP_M,
        },
        "strategy": decision.as_dict(),
    }
    result["route_state"] = build_route_state(
        route_kind="direct",
        original_query=query,
        start=start,
        end=end,
        travel_mode=final_mode,
        hard_constraints=constraints,
        strategy=decision.as_dict(),
        data_version=_current_data_version(G),
        road_condition_version=_current_road_condition_version(route_conditions),
    )
    fallback_timings = normalize_timings(
        route_result.get("timings_ms"), agent=fallback_agent_ms
    )
    fallback_timings["response_build"] += (
        time.perf_counter() - response_started
    ) * 1000
    result["timings_ms"] = normalize_timings(fallback_timings)
    return _ok(result)


@api_bp.route("/pois", methods=["GET"])
def list_pois():
    """GET /api/pois — POI 列表

    Query 参数:
      type     — 按类型筛选 (landmark/scenery/study)
      season   — 按季节标签筛选 (spring/summer/autumn/winter)
      keyword  — 关键词模糊搜索

    出参:
      {pois: [...]}
    """
    poi_type = request.args.get("type")
    season = request.args.get("season")
    keyword = request.args.get("keyword", "").strip()

    try:
        if keyword:
            pois = search_pois(keyword, poi_type=poi_type, season=season)
        else:
            pois = list_all_pois(poi_type=poi_type, season=season)
        return _ok({"pois": pois})
    except Exception as e:
        logger.exception("POI 列表获取失败")
        return _err("poi_list_failed", f"POI 列表获取失败: {e}", 500)


@api_bp.route("/course-spatial-reference", methods=["GET"])
def course_spatial_reference():
    """Return the complete school-supplied road/spot reference layer.

    Source geometries remain WGS-84 in the API. The map converts them once to
    GCJ-02 at the display boundary; this endpoint never changes route behavior.
    """
    path = Path(__file__).resolve().parents[1] / "data" / "course_spatial_reference.geojson"
    try:
        with path.open("r", encoding="utf-8") as source_file:
            document = json.load(source_file)
        if document.get("type") != "FeatureCollection":
            raise ValueError("school source layer is not a GeoJSON FeatureCollection")
        response, status = _ok(document)
        response.headers["Cache-Control"] = "no-store"
        return response, status
    except FileNotFoundError:
        return _err("course_spatial_reference_unavailable",
                    "校方空间数据图层尚未发布", 503)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.exception("校方空间数据图层读取失败")
        return _err("course_spatial_reference_invalid", str(exc), 500)


@api_bp.route("/pois/<name>", methods=["GET"])
def get_single_poi(name):
    """GET /api/pois/<name> — 单 POI 查询（支持模糊匹配）

    出参:
      {poi: {...}}
    """
    if not name or len(name) > 100:
        return _err("invalid_poi_name", "POI 名称参数不合法", 400)

    try:
        poi = get_poi(name, fuzzy=True)
        if poi is None:
            return _err("poi_not_found", f"未找到匹配 '{name}' 的 POI", 404)
        return _ok({"poi": poi})
    except Exception as e:
        logger.exception("POI 查询失败")
        return _err("poi_query_failed", f"POI 查询失败: {e}", 500)


@api_bp.route("/network/init", methods=["POST"])
def network_init():
    """POST /api/network/init — 触发路网加载

    首次调用可能需要下载 OSM 数据（约 30 秒），后续从缓存加载（< 1 秒）。

    出参:
      {status, nodes, edges, cached}
    """
    global _network_initialized

    body = request.get_json(silent=True) or {}
    force = body.get("force", False)

    try:
        if force:
            from spatial.network import reload_network
            G = reload_network()
            _network_initialized = True
            return _ok({
                "status": "loaded",
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "cached": False,
            })

        G = get_network()
        if G is not None:
            _network_initialized = True
            return _ok({
                "status": "already_loaded",
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "cached": True,
            })

        from config import ROAD_NETWORK_CACHE
        cache_existed = os.path.exists(ROAD_NETWORK_CACHE)

        G = load_or_download_network()
        _network_initialized = True
        return _ok({
            "status": "loaded",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "cached": cache_existed,
        })
    except RuntimeError as e:
        logger.exception("路网加载失败")
        return _err("network_load_failed", str(e), 500)
    except Exception as e:
        logger.exception("路网初始化异常")
        return _err("network_init_failed", f"路网初始化失败: {e}", 500)


# ===========================================================================
# T-022: 场景 3 候选 POI 交互 — POST /api/candidates
# ===========================================================================
@api_bp.route("/candidates", methods=["POST"])
def candidates():
    """POST /api/candidates — scenario 3: find candidate POIs by type near a start point.

    Input: {"start": {"name": "珞珈门"}, "poi_type": "scenery", "keyword": "樱花"}
    Output: {"candidates": [{"name": "...", "type": "...", "scenery_score": N, "distance_m": N}, ...]}
    """
    body = request.get_json(silent=True)
    if body is None:
        return _err("invalid_json", "请求体必须为合法 JSON", 400)

    start = body.get("start")
    if not start or not start.get("name"):
        return _err("missing_start", "start.name 必填", 400)

    start_poi = get_poi(start.get("name"))
    if start_poi is None:
        return _err("poi_not_found", f"起点 '{start.get('name')}' 未找到", 404)

    poi_type = body.get("poi_type")
    keyword = body.get("keyword", "").strip()

    # Get all POIs matching type
    all_pois = list_all_pois(poi_type=poi_type if poi_type else None)

    # Filter by keyword if provided, exclude start POI
    candidates = []
    for poi in all_pois:
        if poi["name"] == start.get("name"):
            continue
        if keyword and keyword not in poi.get("name", "") and keyword not in poi.get("description", ""):
            continue
        candidates.append(poi)

    if not candidates:
        return _err(
            "no_candidates",
            f"未找到匹配的候选POI（类型={poi_type}, 关键词={keyword}）",
            404,
        )

    # Calculate road-network distance from start to each candidate
    G, err = _ensure_network()
    if err:
        # Fallback: use haversine distance if network not loaded
        start_lat = start_poi["lat"]
        start_lon = start_poi["lon"]
        for c in candidates:
            c_lat = c.get("lat", 0)
            c_lon = c.get("lon", 0)
            dist = _haversine(start_lat, start_lon, c_lat, c_lon)
            c["distance_m"] = round(dist, 1)
            c["scenery_score"] = c.get("scenery_score", 3)
    else:
        start_lon_wgs, start_lat_wgs = gcj02_to_wgs84(start_poi["lon"], start_poi["lat"])
        try:
            start_node = get_nearest_node(G, start_lon_wgs, start_lat_wgs)
        except Exception:
            start_node = None

        for c in candidates:
            c_lat = c.get("lat", 0)
            c_lon = c.get("lon", 0)
            if start_node is not None:
                try:
                    c_lon_wgs, c_lat_wgs = gcj02_to_wgs84(c_lon, c_lat)
                    end_node = get_nearest_node(G, c_lon_wgs, c_lat_wgs)
                    try:
                        path = nx.dijkstra_path(G, start_node, end_node, weight="length")
                        dist = _path_length(G, path)
                        c["distance_m"] = round(dist, 1)
                    except Exception:
                        dist = _haversine(start_poi["lat"], start_poi["lon"], c_lat, c_lon)
                        c["distance_m"] = round(dist, 1)
                except Exception:
                    dist = _haversine(start_poi["lat"], start_poi["lon"], c_lat, c_lon)
                    c["distance_m"] = round(dist, 1)
            else:
                dist = _haversine(start_poi["lat"], start_poi["lon"], c_lat, c_lon)
                c["distance_m"] = round(dist, 1)
            c["scenery_score"] = c.get("scenery_score", 3)

    # Sort by scenery_score desc, then distance asc
    candidates.sort(key=lambda p: (-p.get("scenery_score", 3), p.get("distance_m", 9999)))

    # Return top 5
    result = []
    for c in candidates[:5]:
        result.append({
            "name": c["name"],
            "type": c.get("type", "landmark"),
            "scenery_score": c.get("scenery_score", 3),
            "distance_m": c.get("distance_m", 0),
            "description": c.get("description", "")[:80],
        })

    return _ok({"candidates": result, "start": start})


# ===================== 路况管理 =====================

def _parse_time_input(value):
    """把时间入参解析为 Unix 时间戳（秒）。

    支持：None/空串/0 → None；数字 → 秒级时间戳；
    ISO 字符串（如 "2026-09-10T08:00"）→ 本地时间时间戳。
    """
    if value is None or value == "" or value == 0:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        from datetime import datetime
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            raise ValueError(f"时间格式无法识别: {value}（请用 YYYY-MM-DDTHH:MM 或时间戳）")
    raise ValueError(f"时间参数类型无效: {type(value)}")


@api_bp.route("/road-conditions", methods=["GET"])
def get_road_conditions():
    """GET /api/road-conditions — 返回当前生效的路况事件列表

    管理员（session 或 Token）带 ?all=1 时返回全部事件（含未开始/已过期），并附加 status。
    """
    include_all = request.args.get("all") in ("1", "true", "yes")
    is_admin = _is_admin() if include_all else False
    try:
        if include_all and is_admin:
            purge_revoked_conditions()
        conditions = list_conditions(
            include_inactive=include_all and is_admin,
            strict=True,
        )
        active_conditions = (
            list_conditions(strict=True) if include_all and is_admin else conditions
        )
    except RoadConditionsUnavailableError:
        logger.exception("路况查询无法读取有效数据")
        return _err("road_conditions_unavailable", "路况数据暂时不可用", 503)
    # Hash the same untouched event snapshot used by route construction. Public
    # response decoration/removal of admin-only fields must not trigger a false
    # route-change notification.
    active_event_version = _current_road_condition_version(active_conditions)
    now = time.time()
    for c in conditions:
        c["type_label"] = CONDITION_LABELS.get(c["type"], c["type"])
        if include_all and is_admin:
            start = c.get("start_time", 0) or 0
            end = c.get("end_time", 0) or 0
            if c.get("revoked_at"):
                c["status"] = "revoked"
            elif end and now > end:
                c["status"] = "expired"
            elif start and now < start:
                c["status"] = "scheduled"
            else:
                c["status"] = "active"
        else:
            c.pop("audit", None)
            c.pop("source", None)
    if request.args.get("version_only") in ("1", "true", "yes"):
        return _ok({"active_event_version": active_event_version})
    return _ok({
        "conditions": conditions,
        "count": len(conditions),
        "active_event_version": active_event_version,
    })


@api_bp.route("/road-conditions/snap", methods=["GET"])
def snap_road_condition():
    """GET /api/road-conditions/snap?lng=&lat= — 管理员选点预览：把点击点吸附到最近路段。

    返回边标识、吸附点(GCJ-02)、道路名、距离与边几何，供前端把标记移到路上并画线。
    """
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    cond_type = (request.args.get("type") or "").strip()
    if cond_type and cond_type not in CONDITION_EFFECTS:
        return _err("invalid_type", f"未知路况类型: {cond_type}", 400)
    action = (request.args.get("action") or "").strip() or None
    affected_modes = [item for item in request.args.get("modes", "").split(",") if item]
    try:
        cost_multiplier = float(request.args.get("cost_multiplier", "1.5"))
    except ValueError:
        return _err("invalid_cost_multiplier", "缓行成本倍数必须是数字", 400)
    if action is not None:
        if action not in CONDITION_ACTIONS:
            return _err("invalid_action", "路况影响动作只能是提示、缓行或禁止通行。", 400)
        if (not affected_modes or len(affected_modes) != len(set(affected_modes))
                or any(mode not in TRAVEL_MODES for mode in affected_modes)):
            return _err("invalid_modes", "请选择有效且不重复的受影响出行方式。", 400)
        if action == "slowdown" and not 1 <= cost_multiplier <= 5:
            return _err("invalid_cost_multiplier", "缓行成本倍数必须在 1 到 5 之间", 400)
    try:
        lng = float(request.args.get("lng"))
        lat = float(request.args.get("lat"))
    except (TypeError, ValueError):
        return _err("invalid_lnglat", "lng/lat 必须是数字", 400)

    G, err = _ensure_network()
    if err:
        return err
    snap = snap_to_edge(G, lng, lat)
    if snap is None:
        return _err(
            "too_far_from_road",
            f"点击位置离最近道路超过 {int(SNAP_MAX_DIST_M)} 米，请放大地图点在道路上",
            400,
        )
    response = {"snap": snap, "max_dist_m": SNAP_MAX_DIST_M}
    if cond_type:
        requested_modes = request.args.getlist("blocked_modes")
        blocked_modes = requested_modes if requested_modes else list(TRAVEL_MODE_KEYS)
        try:
            response["impact_preview"] = _build_road_condition_impact_preview(
                G, snap, cond_type, action=action, affected_modes=affected_modes,
                cost_multiplier=cost_multiplier, blocked_modes=blocked_modes,
            )
        except RoadConditionsUnavailableError:
            logger.exception("路况预览无法读取有效事件快照")
            return _err("road_conditions_unavailable", "当前路况数据不可用，无法生成可靠的发布影响预览", 503)
        except ValueError as error:
            return _err("invalid_impact_preview", str(error), 400)
    return _ok(response)


def _build_road_condition_impact_preview(
    G, snap: dict, cond_type: str, *, action=None, affected_modes=None,
    cost_multiplier=1.5, blocked_modes=None,
) -> dict:
    """Preview effects and a local before/after route sample; never publishes an event."""
    active_conditions = list_conditions(strict=True)
    effects = {}
    if action:
        if action not in CONDITION_ACTIONS:
            raise ValueError("路况影响动作无效")
        selected_modes = set(affected_modes or TRAVEL_MODES)
        if not selected_modes or selected_modes - TRAVEL_MODES:
            raise ValueError("受影响的出行方式无效")
        if action == "slowdown" and not 1 <= float(cost_multiplier) <= 5:
            raise ValueError("缓行成本倍数必须在 1 到 5 之间")
    else:
        blocked_modes = list(TRAVEL_MODE_KEYS) if blocked_modes is None else blocked_modes
        if (not isinstance(blocked_modes, list) or not blocked_modes
                or len(blocked_modes) != len(set(blocked_modes))
                or any(mode not in TRAVEL_MODE_KEYS for mode in blocked_modes)):
            raise ValueError("至少选择一种禁行方式")
        selected_modes = set(blocked_modes)
    for mode in CONDITION_EFFECTS[cond_type]:
        if action:
            if mode not in selected_modes:
                effects[mode] = {"status": "unaffected", "label": "不受该事件影响"}
                continue
            effect = "block" if action == "closure" else float(cost_multiplier) if action == "slowdown" else None
        else:
            effect = "block" if mode in selected_modes else None
        if effect == "block":
            effects[mode] = {"status": "blocked", "label": "该方式在所选路段禁行"}
        elif effect is not None:
            effects[mode] = {
                "status": "cost_increased",
                "cost_multiplier": float(effect),
                "label": f"该路段规划成本 ×{float(effect):g}",
            }
        else:
            effects[mode] = {"status": "notice_only", "label": "提示信息，不改变路线成本"}

    edge_record = {
        "u": int(snap["u"]),
        "v": int(snap["v"]),
        "key": int(snap.get("key", 0)),
        "edges": snap.get("edges") or [[int(snap["u"]), int(snap["v"]), int(snap.get("key", 0))]],
        "chain_length_m": snap.get("chain_length_m", 0),
        "road_name": snap.get("road_name") or "",
        "snap": {"lng": float(snap["snap_lng_gcj"]), "lat": float(snap["snap_lat_gcj"])},
        "geometry_gcj": snap.get("geometry_gcj") or [],
    }
    condition = {"id": "preview-only", "type": cond_type, "edge": edge_record}
    if action:
        condition.update({
            "action": action,
            "affected_modes": list(selected_modes),
            "cost_multiplier": float(cost_multiplier),
        })
    else:
        condition["blocked_modes"] = list(selected_modes)
    sample_routes = {}
    for mode in TRAVEL_MODES:
        common = {
            "mode": mode,
            "weights": {"distance": 0.90, "slope": 0.05, "scenery": 0.05},
            "strategy_name": "recommended",
            "road_conditions": active_conditions,
        }
        try:
            before = compute_route(G, edge_record["u"], edge_record["v"], **common)
            after = compute_route(
                G, edge_record["u"], edge_record["v"],
                **{**common, "road_conditions": [*active_conditions, condition]},
            )
            before_m = float(before["recommended_length_m"])
            after_m = float(after["recommended_length_m"])
            sample_routes[mode] = {
                "status": "available",
                "before_distance_m": round(before_m),
                "after_distance_m": round(after_m),
                "detour_m": round(after_m - before_m),
                "before_duration_min": round(float(before.get("duration_min", 0)), 1),
                "after_duration_min": round(float(after.get("duration_min", 0)), 1),
            }
        except RoadConditionsUnavailableError:
            raise
        except (ValueError, KeyError, TypeError, nx.NetworkXException):
            sample_routes[mode] = {"status": "unreachable", "message": "该方式下所选路段两端无可用的绕行路线"}
        except Exception:
            logger.warning("路况影响示例计算失败，mode=%s", mode, exc_info=True)
            sample_routes[mode] = {"status": "unknown", "message": "示例路线暂时无法计算"}

    return {
        "type": cond_type,
        "road_name": edge_record["road_name"],
        "affected_road_segments": len(edge_record["edges"]),
        "affected_length_m": round(float(edge_record["chain_length_m"] or 0)),
        "effects_by_mode": effects,
        "sample_routes": sample_routes,
        "active_event_count": len(active_conditions),
        "active_event_version": current_road_condition_version(active_conditions),
        "route_sample_scope": "所选路段两端之间的示例路线，不代表全校总影响",
        "event_scope_note": "示例基于当前有效事件快照；发布前若路况变化需重新查看预览",
        "published": False,
    }


@api_bp.route("/road-conditions", methods=["POST"])
def create_road_condition():
    """POST /api/road-conditions — 新增路况事件（需管理员 session 或 Token）

    事件由服务端吸附到最近路段，不接受"半径"参数：
    Input: {
        "type": "closure|construction|event|flooding|accident",
        "name": "樱花大道施工",
        "lng": 114.365,          # 管理员点击点（GCJ-02）
        "lat": 30.536,
        "description": "施工期间禁止通行",
        "start_time": "2026-09-10T08:00",  # 可选，缺省=立即生效
        "end_time": "2026-09-12T18:00"     # 可选，缺省=长期有效
    }
    """
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    identity = _admin_identity()

    body = request.get_json(silent=True)
    if body is None:
        return _err("invalid_json", "请求体必须为合法 JSON", 400)

    cond_type = body.get("type")
    name = (body.get("name") or "").strip()
    lng = body.get("lng")
    lat = body.get("lat")
    action = body.get("action")
    affected_modes = body.get("affected_modes")
    cost_multiplier = body.get("cost_multiplier")
    blocked_modes = body.get("blocked_modes", list(TRAVEL_MODE_KEYS))

    if not cond_type or not name or lng is None or lat is None:
        return _err("missing_fields", "type, name, lng, lat 必填", 400)
    if cond_type not in CONDITION_EFFECTS:
        return _err("invalid_type", f"未知路况类型: {cond_type}", 400)
    if action is not None and action not in CONDITION_ACTIONS:
        return _err("invalid_action", "路况影响动作只能是提示、缓行或禁止通行。", 400)

    source = None
    source_job_id = body.get("source_vision_job_id")
    if source_job_id:
        try:
            source_job_id = str(uuid.UUID(str(source_job_id)))
        except (ValueError, TypeError, AttributeError):
            return _err("invalid_vision_source", "影像任务标识无效。", 400)
        field_confirmation = str(body.get("field_confirmation") or "").strip()
        if len(field_confirmation) < 8 or len(field_confirmation) > 300:
            return _err("field_confirmation_required", "请填写具体道路的现场核实依据（8 至 300 字）。", 400)
        try:
            from storage import database, vision_repository
            database.initialize(current_app.config.get("DATABASE_URL"))
            source_job = vision_repository.get_job(current_app.config.get("DATABASE_URL"), source_job_id)
        except Exception:
            logger.exception("影像来源验证暂时不可用")
            return _err("vision_storage_unavailable", "影像任务暂时无法核对，请稍后重试。", 503)
        candidates = (source_job.get("result") or {}).get("candidates") if source_job else None
        if not source_job or not isinstance(candidates, list):
            return _err("vision_source_not_confirmed", "该影像任务没有可用的待复核候选。", 409)
        candidate_index = body.get("source_vision_candidate_index")
        manual_registration = not candidates and source_job.get("status") == "completed"
        candidate = None
        observation_region = None
        if manual_registration:
            if candidate_index is not None:
                return _err("vision_candidate_invalid", "没有模型候选的影像不能指定候选编号。", 409)
        else:
            if source_job.get("status") != "needs_review" or not candidates:
                return _err("vision_source_not_confirmed", "该影像任务没有可用的待复核候选。", 409)
            if candidate_index is None and source_job.get("review_status") == "confirmed":
                candidate_index = source_job.get("review_candidate_index")
                if candidate_index is None and len(candidates) == 1:
                    candidate_index = 0  # 兼容已完成的旧版单候选审核
            if (isinstance(candidate_index, bool) or not isinstance(candidate_index, int)
                    or not 0 <= candidate_index < len(candidates)):
                return _err("vision_candidate_ambiguous", "请明确选择已单独确认的影像候选。", 409)
            candidate = candidates[candidate_index]
            if not _valid_review_only_candidate(candidate):
                return _err("vision_candidate_invalid", "影像候选记录无效。", 409)
            candidate_review = (source_job.get("candidate_reviews") or {}).get(str(candidate_index))
            legacy_confirmed = (
                source_job.get("review_status") == "confirmed"
                and source_job.get("review_candidate_index") == candidate_index
            )
            if not (candidate_review and candidate_review.get("status") == "confirmed") and not legacy_confirmed:
                return _err("vision_source_not_confirmed", "该影像候选尚未单独确认，不能作为事件来源。", 409)
            evidence = candidate.get("evidence") if isinstance(candidate.get("evidence"), dict) else {}
            candidate_summary = evidence.get("summary") if isinstance(evidence.get("summary"), dict) else {}
            region_id = candidate_summary.get("region_id") or evidence.get("region_id")
            if region_id is not None:
                if not isinstance(region_id, str) or not region_id or len(region_id) > 64:
                    return _err("vision_region_mismatch", "影像候选的观察区域标识无效，请重新分析。", 409)
                regions = source_job.get("observation_regions")
                matches = [region for region in regions if isinstance(region, dict)
                           and region.get("id") == region_id] if isinstance(regions, list) else []
                if len(matches) != 1:
                    return _err("vision_region_mismatch", "影像候选对应的观察区域已不存在或存在歧义，请重新分析。", 409)
                observation_region = matches[0]
                region_kind = observation_region.get("kind")
                allowed_region_kinds = {
                    "possible_flooding": {"road_surface"},
                    "possible_accident": {"vehicle_lane", "road_surface"},
                    "possible_congestion": {"vehicle_lane"},
                    "possible_crowding": {"pedestrian"},
                    "vehicle_cluster_review": {"vehicle_lane"},
                }.get(candidate.get("kind"), set())
                polygon = observation_region.get("polygon")
                if (region_kind not in allowed_region_kinds or not isinstance(polygon, list)
                        or not 3 <= len(polygon) <= 128):
                    return _err("vision_region_mismatch", "影像候选与观察区域类型不匹配，请重新分析。", 409)
                for point in polygon:
                    if (not isinstance(point, (list, tuple)) or len(point) != 2
                            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                   or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
                                   for value in point)):
                        return _err("vision_region_mismatch", "影像观察区域坐标无效，请重新分析。", 409)
        captured_at = source_job.get("captured_at")
        if not captured_at:
            return _err("vision_capture_time_missing", "影像缺少实际拍摄时间，不能作为当前路况来源。", 409)
        captured_ts = captured_at.timestamp() if isinstance(captured_at, datetime) else float(captured_at)
        age_seconds = time.time() - captured_ts
        if age_seconds < -300 or age_seconds > config.VISION_MAX_EVENT_AGE_SECONDS:
            return _err("vision_source_stale", "影像拍摄时间已超出当前路况有效窗口，请重新获取现场信息。", 409)
        source = {"kind": "vision_job", "job_id": source_job_id,
                  "candidate_index": candidate_index,
                  "candidate_kind": (
                      "manual_visual_event" if manual_registration
                      else str(candidate.get("kind") or "unknown")
                  ),
                  "captured_at": captured_ts,
                  "field_confirmation": field_confirmation,
                  "road_association": "manager_selected_edge"}
        if manual_registration:
            source["manual_registration"] = True
        if observation_region is not None:
            source.update({
                "region_id": region_id,
                "region_kind": observation_region["kind"],
                "region_polygon": observation_region["polygon"],
            })

    if source_job_id and (
            action not in CONDITION_ACTIONS
            or not isinstance(affected_modes, list)
            or not affected_modes
            or len(affected_modes) > len(TRAVEL_MODE_KEYS)
            or any(not isinstance(mode, str) or mode not in TRAVEL_MODE_KEYS
                   for mode in affected_modes)
            or len(set(affected_modes)) != len(affected_modes)):
        return _err(
            "vision_effect_required",
            "发布影像路况时必须明确选择提示、缓行或封路，并指定受影响的出行方式。",
            400,
        )

    try:
        lng_f, lat_f = float(lng), float(lat)
        start_ts = _parse_time_input(body.get("start_time"))
        end_ts = _parse_time_input(body.get("end_time"))
        if start_ts and end_ts and end_ts <= start_ts:
            return _err("invalid_time", "结束时间必须晚于开始时间", 400)
        if source_job_id and (end_ts is None or end_ts <= time.time()):
            return _err("vision_event_expiry_required", "影像来源事件必须设置一个未来的结束时间。", 400)

        G, err = _ensure_network()
        if err:
            return err
        # 服务端重新吸附，不信任前端坐标
        snap = snap_to_edge(G, lng_f, lat_f)
        if snap is None:
            return _err(
                "too_far_from_road",
                f"点击位置离最近道路超过 {int(SNAP_MAX_DIST_M)} 米，请放大地图点在道路上",
                400,
            )

        if source is not None:
            source["associated_edge"] = {
                "u": int(snap["u"]), "v": int(snap["v"]),
                "key": int(snap.get("key", 0)),
                "road_name": str(snap.get("road_name") or ""),
            }

        condition = add_condition(
            cond_type=cond_type,
            name=name[:30],
            edge=snap,
            click_point={"lng": lng_f, "lat": lat_f},
            description=(body.get("description") or "").strip()[:200],
            start_time=start_ts,
            end_time=end_ts,
            created_by=identity if identity == "web" else "token",
            source=source,
            action=action,
            affected_modes=affected_modes,
            cost_multiplier=cost_multiplier,
            blocked_modes=blocked_modes,
        )
        condition["type_label"] = CONDITION_LABELS.get(cond_type, cond_type)
        return _ok({"condition": condition, "snap": snap}, status=201)
    except VisionConditionAlreadyPublishedError as e:
        return _ok({
            "condition": e.condition,
            "snap": snap,
            "already_published": True,
        }, status=200)
    except RoadConditionsUnavailableError:
        logger.exception("路况存储无效，拒绝新增以免覆盖既有事件")
        return _err("road_conditions_unavailable", "现有路况数据无效，已拒绝修改；请先修复数据文件", 503)
    except ValueError as e:
        return _err("invalid_field", str(e), 400)
    except Exception as e:
        logger.exception("新增路况失败")
        return _err("create_failed", f"新增失败: {e}", 500)


@api_bp.route("/road-conditions/<cond_id>", methods=["PATCH"])
def patch_road_condition(cond_id):
    """PATCH /api/road-conditions/<id> — 更新路况（需管理员）

    支持：
      {"action": "end"}                         立即结束（end_time=now，保留记录可审计）
      {"name": "..."} / {"description": "..."}  改名/补充描述
      {"start_time": ..., "end_time": ...}      调整生效时段（ISO 或时间戳）
    """
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    body = request.get_json(silent=True) or {}

    changes = {}
    if body.get("action") == "end":
        changes["end_time"] = time.time()
    else:
        for field in ("name", "description"):
            if body.get(field) is not None:
                changes[field] = str(body[field])[:200]
        start_ts = _parse_time_input(body.get("start_time")) if body.get("start_time") is not None else None
        end_ts = _parse_time_input(body.get("end_time")) if body.get("end_time") is not None else None
        if start_ts is not None:
            changes["start_time"] = start_ts
        if end_ts is not None:
            changes["end_time"] = end_ts

    if not changes:
        return _err("empty_changes", "没有可更新的字段", 400)

    try:
        updated = update_condition(cond_id, changes, actor=_admin_identity() or "unknown",
                                   action="ended" if body.get("action") == "end" else "updated")
    except RoadConditionsUnavailableError:
        logger.exception("路况存储无效，拒绝更新")
        return _err("road_conditions_unavailable", "现有路况数据无效，已拒绝修改；请先修复数据文件", 503)
    except ValueError as e:
        return _err("invalid_field", str(e), 400)
    if updated is None:
        return _err("not_found", f"路况事件 {cond_id} 不存在", 404)
    updated["type_label"] = CONDITION_LABELS.get(updated.get("type"), updated.get("type"))
    return _ok({"condition": updated})


@api_bp.route("/road-conditions/<cond_id>", methods=["DELETE"])
def delete_road_condition(cond_id):
    """DELETE /api/road-conditions/<id> — permanently remove the road restriction."""
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    try:
        success = remove_condition(cond_id, actor=_admin_identity() or "unknown")
    except RoadConditionsUnavailableError:
        logger.exception("路况存储无效，拒绝删除")
        return _err("road_conditions_unavailable", "现有路况数据无效，已拒绝修改；请先修复数据文件", 503)
    if not success:
        return _err("not_found", f"路况事件 {cond_id} 不存在", 404)
    return _ok({"message": "已永久删除路段管制", "id": cond_id})


def _vision_upload_root() -> Path:
    return Path(config.VISION_UPLOAD_DIR).resolve() / ".resumable"


def _vision_upload_session_dir(upload_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id or ""):
        return None
    root = _vision_upload_root()
    candidate = root / upload_id
    return candidate if candidate.parent == root else None


@contextmanager
def _lock_vision_upload(session_dir: Path):
    """Serialize status, chunk and completion operations across app workers."""
    lock_path = session_dir / ".lock"
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _read_vision_upload(session_dir: Path) -> dict | None:
    try:
        with (session_dir / "metadata.json").open("r", encoding="utf-8") as source:
            metadata = json.load(source)
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def _write_vision_upload(session_dir: Path, metadata: dict) -> None:
    temporary = session_dir / "metadata.json.tmp"
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, session_dir / "metadata.json")


def _vision_video_header_valid(path: Path, extension: str) -> bool:
    with path.open("rb") as source:
        header = source.read(16)
    return (
        (extension in {".mp4", ".mov"} and header[4:8] == b"ftyp")
        or (extension == ".avi" and header.startswith(b"RIFF") and header[8:12] == b"AVI ")
        or (extension == ".webm" and header.startswith(b"\x1aE\xdf\xa3"))
    )


def _cleanup_expired_vision_uploads(root: Path) -> None:
    if not root.is_dir():
        return
    now = time.time()
    for session_dir in root.iterdir():
        if not session_dir.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", session_dir.name):
            continue
        try:
            with _lock_vision_upload(session_dir):
                metadata = _read_vision_upload(session_dir)
                updated_at = float((metadata or {}).get("updated_at", 0))
                expired = not updated_at or now - updated_at > _VISION_UPLOAD_TTL_SECONDS
            if expired:
                shutil.rmtree(session_dir, ignore_errors=True)
        except OSError:
            logger.info("跳过正在使用的影像续传会话清理: %s", session_dir.name)


def _public_vision_job(job: dict) -> dict:
    visible = {key: value for key, value in job.items()
               if key not in {"media_path", "sha256", "checkpoint"}}
    if visible.get("job_id"):
        visible["media_url"] = f"/api/manager/vision-jobs/{visible['job_id']}/media"
        stored_result = visible.get("result")
        if isinstance(stored_result, dict):
            # Frame-level overlays are fetched only when an administrator opens
            # a video preview. Avoid copying or returning them in status responses.
            result = {key: value for key, value in stored_result.items()
                      if key not in {"preview_frames", "preview_frame_interval_seconds"}}
            candidates = result.get("candidates")
            if isinstance(candidates, list):
                result["candidates"] = json.loads(json.dumps(candidates))
            visible["result"] = result
            candidates = result.get("candidates")
            if isinstance(candidates, list):
                for candidate_index, candidate in enumerate(candidates):
                    evidence = candidate.get("evidence") if isinstance(candidate, dict) else None
                    clips = evidence.get("clips") if isinstance(evidence, dict) else None
                    if not isinstance(clips, list):
                        continue
                    for clip_index, clip in enumerate(clips):
                        expected = f"{visible['job_id']}-evidence-c{candidate_index}-s{clip_index}.webm"
                        if isinstance(clip, dict) and clip.get("filename") == expected:
                            clip["media_url"] = (
                                f"/api/manager/vision-jobs/{visible['job_id']}/media"
                                f"?clip={candidate_index}:{clip_index}"
                            )
                            clip.pop("filename", None)
    return visible


def _purge_tombstoned_vision_jobs(database_url, jobs: list[dict]) -> int:
    """Retry physical cleanup without ever making an explicitly deleted job visible."""
    from storage import vision_repository

    upload_dir = Path(config.VISION_UPLOAD_DIR).resolve()
    removed_media_ids = []
    retry_ids = []
    pending_count = 0
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        stored_path = str(job.get("media_path") or "")
        media_name = Path(stored_path)
        if (not job_id or not stored_path or media_name.is_absolute()
                or media_name.name != stored_path or stored_path in {".", ".."}):
            logger.error("拒绝清理不安全的影像存储路径: job_id=%s", job_id or "unknown")
            pending_count += 1
            if job_id:
                retry_ids.append(job_id)
            continue

        media_path = upload_dir / media_name
        if media_path.parent != upload_dir or media_path.is_symlink():
            logger.error("拒绝清理越界的影像存储路径: job_id=%s", job_id)
            pending_count += 1
            retry_ids.append(job_id)
            continue
        try:
            media_path.unlink(missing_ok=True)
            for clip_path in upload_dir.glob(f"{job_id}-evidence-c*-s*.webm"):
                if clip_path.parent == upload_dir and not clip_path.is_symlink():
                    clip_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("删除影像文件失败，记录保持隐藏并等待重试: job_id=%s", job_id)
            pending_count += 1
            retry_ids.append(job_id)
            continue
        removed_media_ids.append(job_id)

    if removed_media_ids:
        finalized = vision_repository.finalize_delete_jobs(database_url, removed_media_ids)
        pending_count += max(0, len(removed_media_ids) - finalized)
        if finalized < len(removed_media_ids):
            retry_ids.extend(removed_media_ids)
    if retry_ids:
        vision_repository.mark_delete_retry_attempts(database_url, retry_ids)
    return pending_count


def _vision_inference_readiness(database_url=None) -> tuple[bool, bool, bool, bool]:
    import importlib.util

    weights_ready = bool(config.VISION_MODEL_PATH and Path(config.VISION_MODEL_PATH).is_file())
    dependencies_ready = all(
        importlib.util.find_spec(name)
        for name in ("onnxruntime", "numpy", "cv2", "PIL")
    )
    worker_ready = False
    if weights_ready and dependencies_ready:
        try:
            from storage import database, vision_repository
            database.initialize(database_url)
            worker_ready = vision_repository.has_ready_worker(database_url)
        except Exception:
            logger.info("Vision worker is not ready or its heartbeat is unavailable", exc_info=True)
    return weights_ready and dependencies_ready and worker_ready, weights_ready, dependencies_ready, worker_ready


def _vision_worker_capabilities(database_url=None) -> dict:
    from storage import database, vision_repository

    try:
        database.initialize(database_url)
        worker = vision_repository.ready_worker_status(database_url)
        detail = json.loads(worker["status_detail"]) if worker and worker.get("status_detail") else {}
        return detail if isinstance(detail, dict) else {}
    except Exception:
        logger.info("Optional vision capability status is unavailable", exc_info=True)
        return {}


def _vision_optional_model_ready(path_key: str, enabled_key: str, capability_key: str,
                                 inference_ready: bool, worker_capabilities: dict) -> bool:
    model_path = getattr(config, path_key, "")
    return bool(
        getattr(config, enabled_key, False) and inference_ready
        and model_path and Path(model_path).is_file()
        and worker_capabilities.get(capability_key) is True
    )


@api_bp.route("/manager/vision-scenes", methods=["GET", "POST"])
def manager_vision_scenes():
    if request.method == "GET":
        auth_error = _require_admin()
        if auth_error:
            return auth_error
        try:
            from storage import database
            database_url = current_app.config.get("DATABASE_URL")
            database.initialize(database_url)
            scenes = vision_scene_repository.list_scenes(database_url)
            return _ok({"scenes": [_scene_public(scene) for scene in scenes]})
        except Exception:
            logger.exception("读取无人机观察场景失败")
            return _err("vision_scene_unavailable", "观察场景暂时无法读取。", 503)

    auth_error = _require_admin()
    if auth_error:
        return auth_error
    body = request.get_json(silent=True) or {}
    name = str(body.get("name") or "").strip()
    if not 1 <= len(name) <= 100:
        return _err("invalid_scene_name", "场景名称需要为 1 到 100 个字符。", 400)
    update_scene_id = str(body.get("scene_id") or "").strip() or None
    if update_scene_id:
        try:
            uuid.UUID(update_scene_id)
        except (ValueError, TypeError):
            return _err("vision_scene_not_found", "观察场景不存在，请刷新场景列表。", 404)
    try:
        from storage import database
        from vision.scene_signature import normalize_signature
        anchor_gcj = _valid_scene_anchor(body.get("anchor_gcj"))
        signature = normalize_signature(body.get("frame_signature"))
        regions = _parse_scene_regions(body.get("observation_regions"))
    except _VisionSceneError as error:
        return _err(error.code, str(error), error.status)
    except (TypeError, ValueError) as error:
        return _err("invalid_observation_metadata", str(error), 400)
    linked_regions = [region for region in regions
                      if region["kind"] in _VISION_SCENE_LINKED_REGION_KINDS]
    if not regions:
        return _err("vision_scene_regions_required", "请先圈选至少一个观察区域。", 400)
    database_url = current_app.config.get("DATABASE_URL")
    try:
        database.initialize(database_url)
        existing = vision_scene_repository.list_scenes(database_url)
        if update_scene_id and not any(str(item.get("scene_id")) == update_scene_id for item in existing):
            return _err("vision_scene_not_found", "观察场景已删除，请刷新列表后重新保存。", 404)
        if any(str(item.get("scene_id")) != update_scene_id
               and str(item.get("name", "")).casefold() == name.casefold() for item in existing):
            return _err("vision_scene_name_conflict", "已有同名观察场景，请换一个名称或先删除旧场景。", 409)
        if linked_regions:
            graph, _ = _ensure_network()
            for region in linked_regions:
                point = region["road_point_gcj"]
                bbox = config.WHU_BBOX
                if not (bbox["west"] - .01 <= point["lng"] <= bbox["east"] + .01
                        and bbox["south"] - .01 <= point["lat"] <= bbox["north"] + .01):
                    return _err("vision_scene_road_link_required", "道路绑定点需要落在校园道路范围附近。", 400)
                snap = snap_to_edge(graph, point["lng"], point["lat"])
                if not snap or float(snap.get("dist_m", float("inf"))) > SNAP_MAX_DIST_M:
                    return _err("vision_scene_road_link_required", "观察区域绑定点没有匹配到足够近的校园道路，请在地图上重新选路。", 400)
                region["road_link"] = {
                    "edges": snap.get("edges"), "road_name": snap.get("road_name"),
                    "geometry_gcj": snap.get("geometry_gcj"),
                    "snap_lng_gcj": snap.get("snap_lng_gcj"),
                    "snap_lat_gcj": snap.get("snap_lat_gcj"),
                    "dist_m": snap.get("dist_m"),
                    "chain_length_m": snap.get("chain_length_m"),
                }
        if update_scene_id:
            scene = vision_scene_repository.update_scene(
                database_url, update_scene_id, name=name, anchor_gcj=anchor_gcj,
                camera_stabilized=body.get("camera_stabilized") is True,
                frame_signature=signature, observation_regions=regions,
            )
            if not scene:
                return _err("vision_scene_not_found", "观察场景已删除，请刷新列表后重新保存。", 404)
        else:
            scene = vision_scene_repository.create_scene(
                database_url, name=name, created_by=_admin_identity() or "unknown",
                anchor_gcj=anchor_gcj, camera_stabilized=body.get("camera_stabilized") is True,
                frame_signature=signature, observation_regions=regions,
            )
        return _ok({"scene": _scene_public(scene)}, status=200 if update_scene_id else 201)
    except Exception as error:
        from psycopg.errors import UniqueViolation
        if isinstance(error, UniqueViolation):
            return _err("vision_scene_name_conflict", "已有同名观察场景，请刷新列表或换一个名称。", 409)
        logger.exception("保存无人机观察场景失败")
        return _err("vision_scene_unavailable", "观察场景保存失败，请稍后重试。", 503)


@api_bp.route("/manager/vision-scenes/<scene_id>", methods=["DELETE"])
def delete_manager_vision_scene(scene_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    try:
        uuid.UUID(scene_id)
    except (ValueError, TypeError, AttributeError):
        return _err("vision_scene_not_found", "观察场景不存在。", 404)
    try:
        from storage import database
        database_url = current_app.config.get("DATABASE_URL")
        database.initialize(database_url)
        if not vision_scene_repository.delete_scene(database_url, scene_id):
            return _err("vision_scene_not_found", "观察场景不存在。", 404)
        return _ok({"deleted": True})
    except Exception:
        logger.exception("删除无人机观察场景失败")
        return _err("vision_scene_unavailable", "观察场景删除失败，请稍后重试。", 503)


def _vision_optional_model_status(path_key: str, enabled_key: str, capability_key: str,
                                  inference_ready: bool, worker_capabilities: dict) -> str:
    model_path = getattr(config, path_key, "")
    if not model_path:
        return "not_configured"
    if not getattr(config, enabled_key, False):
        return "disabled"
    if not Path(model_path).is_file():
        return "file_missing"
    if not inference_ready:
        return "worker_not_ready"
    if worker_capabilities.get(capability_key) is True:
        return "ready"
    return "load_failed"


@api_bp.route("/manager/vision-status", methods=["GET"])
def manager_vision_status():
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    inference_ready, weights_ready, dependencies_ready, worker_ready = _vision_inference_readiness(
        current_app.config.get("DATABASE_URL"),
    )
    worker_capabilities = _vision_worker_capabilities(current_app.config.get("DATABASE_URL")) if worker_ready else {}
    return _ok({
        "upload_enabled": inference_ready,
        "inference_ready": inference_ready,
        "weights_ready": weights_ready,
        "dependencies_ready": dependencies_ready,
        "worker_ready": worker_ready,
        "max_media_bytes": config.VISION_MAX_MEDIA_BYTES,
        "max_image_bytes": config.VISION_MAX_MEDIA_BYTES,
        "max_video_bytes": config.VISION_MAX_VIDEO_BYTES,
        "upload_chunk_bytes": config.VISION_CHUNK_BYTES,
        "capabilities": {
            "aerial_vehicle_detection": inference_ready,
            "congestion_candidate_review": inference_ready,
            "pedestrian_region_counts": inference_ready,
            "accident_recognition_supported": _vision_optional_model_ready(
                "VISION_ACCIDENT_MODEL_PATH", "VISION_ACCIDENT_ENABLED",
                "accident_recognition_supported",
                inference_ready, worker_capabilities,
            ),
            "flood_segmentation_supported": _vision_optional_model_ready(
                "VISION_FLOOD_MODEL_PATH", "VISION_FLOOD_ENABLED",
                "flood_segmentation_supported",
                inference_ready, worker_capabilities,
            ),
            "automatic_routing_updates": False,
        },
        "optional_model_status": {
            "accident": _vision_optional_model_status(
                "VISION_ACCIDENT_MODEL_PATH", "VISION_ACCIDENT_ENABLED",
                "accident_recognition_supported",
                inference_ready, worker_capabilities,
            ),
            "flood": _vision_optional_model_status(
                "VISION_FLOOD_MODEL_PATH", "VISION_FLOOD_ENABLED",
                "flood_segmentation_supported",
                inference_ready, worker_capabilities,
            ),
        },
        "notice": (
            "车辆与行人模型已就绪；圈选机动车道、人行区域或路面可限定分析范围，结果只作为待复核线索。"
            if inference_ready else
            "管理界面已就绪；基础视觉模型、依赖或后台工作进程尚未就绪，当前不会接收影像任务。"
        ),
    })


@api_bp.route("/manager/vision-uploads", methods=["POST"])
def start_manager_vision_upload():
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    inference_ready, _, _, _ = _vision_inference_readiness(current_app.config.get("DATABASE_URL"))
    if not inference_ready:
        return _err("vision_not_ready", "视觉模型、依赖或后台工作进程尚未就绪，暂不能提交任务。", 503)

    body = request.get_json(silent=True) or {}
    safe_name = secure_filename(str(body.get("filename", "")))
    extension = Path(safe_name).suffix.lower()
    size = body.get("size")
    if extension not in _VISION_VIDEO_EXTENSIONS:
        return _err("unsupported_media", "分块续传仅支持 MP4、MOV、AVI、WebM 视频。", 415)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        return _err("invalid_media_size", "视频文件大小无效。", 400)
    if size > config.VISION_MAX_VIDEO_BYTES:
        return _err("media_too_large", f"视频不能超过 {round(config.VISION_MAX_VIDEO_BYTES / (1024 * 1024))} MB。", 413)
    if config.VISION_CHUNK_BYTES < 1:
        return _err("upload_unavailable", "视频分块大小配置无效。", 503)
    try:
        captured_at = _parse_vision_capture_time(body.get("captured_at"), required=True)
        capture_time_source, capture_timezone = _parse_vision_capture_provenance(
            body.get("capture_time_source"), body.get("capture_timezone"),
        )
        scene_id = str(body.get("scene_id") or "").strip() or None
        if scene_id:
            uuid.UUID(scene_id)
            from storage import database
            database.initialize(current_app.config.get("DATABASE_URL"))
            if not vision_scene_repository.get_scene(current_app.config.get("DATABASE_URL"), scene_id):
                return _err("vision_scene_not_found", "观察场景不存在，请刷新场景列表。", 404)
            observation_regions = (
                _parse_scene_regions(body.get("observation_regions"))
                if "observation_regions" in body else None
            )
        else:
            observation_regions = _parse_vision_observation_regions(body.get("observation_regions"))
    except (TypeError, ValueError) as error:
        return _err("invalid_observation_metadata", str(error), 400)
    anchor_gcj = None
    raw_lng, raw_lat = body.get("lng"), body.get("lat")
    if raw_lng is not None or raw_lat is not None:
        if raw_lng is None or raw_lat is None:
            return _err("invalid_anchor", "经纬度必须同时提供。", 400)
        try:
            lng, lat = float(raw_lng), float(raw_lat)
        except (TypeError, ValueError):
            return _err("invalid_anchor", "影像区域坐标无效。", 400)
        bbox = config.WHU_BBOX
        if not (bbox["west"] - .01 <= lng <= bbox["east"] + .01
                and bbox["south"] - .01 <= lat <= bbox["north"] + .01):
            return _err("outside_campus", "观察点需要落在武汉大学校园范围附近。", 400)
        anchor_gcj = {"lng": lng, "lat": lat, "crs": "GCJ02"}

    upload_dir = Path(config.VISION_UPLOAD_DIR).resolve()
    sessions_root = upload_dir / ".resumable"
    try:
        sessions_root.mkdir(parents=True, exist_ok=True)
        _cleanup_expired_vision_uploads(sessions_root)
        upload_id = secrets.token_hex(16)
        session_dir = sessions_root / upload_id
        session_dir.mkdir(mode=0o700)
        (session_dir / "upload.part").touch()
        now = time.time()
        metadata = {
            "upload_id": upload_id, "filename": safe_name[:180], "extension": extension,
            "size": size, "offset": 0, "chunk_size": config.VISION_CHUNK_BYTES,
            "anchor_gcj": anchor_gcj,
            "camera_stabilized": body.get("camera_stabilized") is True,
            "captured_at": captured_at.timestamp(),
            "capture_time_source": capture_time_source,
            "capture_timezone": capture_timezone,
            "observation_regions": observation_regions,
            "scene_id": scene_id,
            "created_by": _admin_identity() or "unknown", "created_at": now, "updated_at": now,
        }
        _write_vision_upload(session_dir, metadata)
        return _ok({"upload_id": upload_id, "offset": 0, "size": size,
                    "chunk_size": config.VISION_CHUNK_BYTES}, status=201)
    except OSError:
        logger.exception("影像续传会话初始化失败")
        return _err("vision_upload_unavailable", "无法创建视频上传任务，请稍后重试。", 503)


@api_bp.route("/manager/vision-uploads/<upload_id>", methods=["GET"])
def manager_vision_upload_status(upload_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    session_dir = _vision_upload_session_dir(upload_id)
    if session_dir is None or not session_dir.is_dir():
        return _err("upload_not_found", "视频续传任务不存在或已过期，请重新选择视频。", 404)
    expired = False
    try:
        with _lock_vision_upload(session_dir):
            metadata = _read_vision_upload(session_dir)
            if not metadata:
                return _err("upload_not_found", "视频续传任务不存在或已过期，请重新选择视频。", 404)
            if time.time() - float(metadata.get("updated_at", 0)) > _VISION_UPLOAD_TTL_SECONDS:
                expired = True
            else:
                if metadata.get("completed_job"):
                    return _ok({"upload_id": upload_id, "size": metadata["size"],
                                "offset": metadata["size"], "completed": True,
                                "job": metadata["completed_job"]})
                part_path = session_dir / "upload.part"
                offset = int(metadata.get("offset", 0))
                if not part_path.is_file() or part_path.stat().st_size < offset:
                    return _err("upload_data_unavailable", "已上传的视频数据不完整，请重新选择视频。", 409)
                if part_path.stat().st_size > offset:
                    with part_path.open("r+b") as output:
                        output.truncate(offset)
                return _ok({"upload_id": upload_id, "filename": metadata["filename"],
                            "size": metadata["size"], "offset": offset,
                            "chunk_size": metadata["chunk_size"]})
        if expired:
            shutil.rmtree(session_dir, ignore_errors=True)
            return _err("upload_expired", "视频续传任务已过期，请重新选择视频。", 410)
    except OSError:
        logger.exception("读取影像续传状态失败")
        return _err("vision_upload_unavailable", "暂时无法读取视频上传进度，请稍后重试。", 503)


@api_bp.route("/manager/vision-uploads/<upload_id>/chunks", methods=["PUT"])
def append_manager_vision_upload_chunk(upload_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    session_dir = _vision_upload_session_dir(upload_id)
    if session_dir is None or not session_dir.is_dir():
        return _err("upload_not_found", "视频续传任务不存在或已过期，请重新选择视频。", 404)
    try:
        requested_offset = int(request.headers.get("Upload-Offset", "-1"))
    except (TypeError, ValueError):
        return _err("invalid_upload_offset", "视频分块位置无效。", 400)
    try:
        with _lock_vision_upload(session_dir):
            metadata = _read_vision_upload(session_dir)
            if not metadata:
                return _err("upload_not_found", "视频续传任务不存在或已过期，请重新选择视频。", 404)
            now = time.time()
            if now - float(metadata.get("updated_at", 0)) > _VISION_UPLOAD_TTL_SECONDS:
                return _err("upload_expired", "视频续传任务已过期，请重新选择视频。", 410)
            offset = int(metadata.get("offset", 0))
            if requested_offset != offset:
                return _err("upload_offset_conflict", "上传进度已变化，请读取最新进度后继续。", 409)
            total_size = int(metadata["size"])
            chunk_size = int(metadata["chunk_size"])
            if offset < 0 or offset >= total_size:
                return _err("upload_already_complete", "视频数据已全部上传。", 409)
            expected_size = min(chunk_size, total_size - offset)
            if request.content_length != expected_size:
                return _err("invalid_chunk_size", "视频分块大小不正确，请重新上传当前分块。", 400)
            part_path = session_dir / "upload.part"
            if not part_path.is_file() or part_path.stat().st_size < offset:
                return _err("upload_data_unavailable", "已上传的视频数据不完整，请重新选择视频。", 409)
            if part_path.stat().st_size > offset:
                with part_path.open("r+b") as output:
                    output.truncate(offset)

            staging_path = session_dir / "chunk.tmp"
            received = 0
            with staging_path.open("wb") as staging:
                while received < expected_size:
                    data = request.stream.read(min(1024 * 1024, expected_size - received))
                    if not data:
                        break
                    received += len(data)
                    staging.write(data)
            if received != expected_size:
                staging_path.unlink(missing_ok=True)
                return _err("invalid_chunk_size", "视频分块未完整到达，请重试。", 400)
            with part_path.open("r+b") as output, staging_path.open("rb") as staging:
                output.seek(offset)
                shutil.copyfileobj(staging, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
                output.truncate(offset + expected_size)
            staging_path.unlink(missing_ok=True)
            metadata["offset"] = offset + expected_size
            metadata["updated_at"] = now
            _write_vision_upload(session_dir, metadata)
            return _ok({"upload_id": upload_id, "offset": metadata["offset"], "size": total_size})
    except OSError:
        logger.exception("写入视频分块失败")
        return _err("vision_upload_unavailable", "保存视频分块失败，请重试当前上传。", 503)


@api_bp.route("/manager/vision-uploads/<upload_id>/complete", methods=["POST"])
def complete_manager_vision_upload(upload_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    session_dir = _vision_upload_session_dir(upload_id)
    if session_dir is None or not session_dir.is_dir():
        return _err("upload_not_found", "视频续传任务不存在或已过期，请重新选择视频。", 404)
    completed_job = None
    target = None
    try:
        with _lock_vision_upload(session_dir):
            metadata = _read_vision_upload(session_dir)
            if not metadata:
                return _err("upload_not_found", "视频续传任务不存在或已过期，请重新选择视频。", 404)
            if time.time() - float(metadata.get("updated_at", 0)) > _VISION_UPLOAD_TTL_SECONDS:
                return _err("upload_expired", "视频续传任务已过期，请重新选择视频。", 410)
            if metadata.get("completed_job"):
                return _ok({"job": metadata["completed_job"]}, status=202)
            if int(metadata.get("offset", 0)) != int(metadata.get("size", -1)):
                return _err("upload_incomplete", "视频尚未上传完整，请继续上传后再提交分析。", 409)
            part_path = session_dir / "upload.part"
            if not part_path.is_file() or part_path.stat().st_size != int(metadata["size"]):
                return _err("upload_data_unavailable", "视频文件不完整，请重新上传。", 409)
            if not _vision_video_header_valid(part_path, metadata["extension"]):
                return _err("invalid_media", "文件内容与扩展名不匹配，或视频文件已损坏。", 400)
            digest = hashlib.sha256()
            with part_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            upload_dir = Path(config.VISION_UPLOAD_DIR).resolve()
            job_id = str(uuid.uuid4())
            stored_name = job_id + metadata["extension"]
            target = upload_dir / stored_name
            os.replace(part_path, target)
            try:
                from storage import database, vision_repository
                database.initialize(current_app.config.get("DATABASE_URL"))
                scene_values = {}
                if metadata.get("scene_id"):
                    scene_values = _resolve_vision_scene(
                        metadata["scene_id"], target, "video",
                        anchor_override=metadata.get("anchor_gcj"),
                        camera_stabilized_override=metadata.get("camera_stabilized"),
                        observation_regions_override=metadata.get("observation_regions"),
                    )
                completed_job = vision_repository.create_job(
                    current_app.config.get("DATABASE_URL"),
                    created_by=metadata["created_by"], original_name=metadata["filename"],
                    media_kind="video", media_path=stored_name, sha256=digest.hexdigest(),
                    anchor_gcj=scene_values.get("anchor_gcj", metadata["anchor_gcj"]),
                    camera_stabilized=scene_values.get("camera_stabilized", metadata["camera_stabilized"]),
                    captured_at=datetime.fromtimestamp(float(metadata["captured_at"]), tz=timezone.utc),
                    capture_time_source=metadata.get("capture_time_source", "unknown"),
                    capture_timezone=metadata.get("capture_timezone"),
                    observation_regions=scene_values.get(
                        "observation_regions", metadata.get("observation_regions") or [],
                    ),
                    observation_scene_id=scene_values.get("scene_id"),
                )
            except Exception:
                os.replace(target, part_path)
                target = None
                raise
            public_job = json.loads(current_app.json.dumps(_public_vision_job(completed_job)))
            metadata["completed_job"] = public_job
            metadata["updated_at"] = time.time()
            try:
                _write_vision_upload(session_dir, metadata)
            except OSError:
                # The queued job and media are already durable. Keep the successful
                # response even if the small resumable-session marker cannot be updated.
                logger.exception("影像任务已入队，但无法保存上传完成标记")
        return _ok({"job": _public_vision_job(completed_job)}, status=202)
    except _VisionSceneError as error:
        logger.info("视频未通过观察场景复用校验: %s", error.code)
        return _err(error.code, str(error), error.status)
    except ValueError as error:
        return _err("invalid_media", str(error), 400)
    except RuntimeError:
        logger.exception("影像任务存储不可用")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    except OSError:
        logger.exception("完成视频上传失败")
        return _err("vision_upload_unavailable", "完成视频上传失败，请重试。", 503)
    except Exception:
        logger.exception("影像任务入队失败")
        return _err("vision_enqueue_failed", "影像任务暂时无法入队，请稍后重试。", 503)
    finally:
        if completed_job is None and target is not None:
            target.unlink(missing_ok=True)


@api_bp.route("/manager/vision-jobs", methods=["GET", "POST"])
def manager_vision_jobs():
    if request.method == "GET":
        auth_error = _require_admin()
        if auth_error:
            return auth_error
        try:
            from storage import database, vision_repository
            database.initialize(current_app.config.get("DATABASE_URL"))
            database_url = current_app.config.get("DATABASE_URL")
            pending_deletions = vision_repository.list_pending_deletions(database_url, limit=200)
            if pending_deletions:
                _purge_tombstoned_vision_jobs(database_url, pending_deletions)
            try:
                limit = int(request.args.get("limit", "50"))
            except (TypeError, ValueError):
                return _err("invalid_limit", "limit 必须是 1 到 200 之间的整数。", 400)
            if not 1 <= limit <= 200:
                return _err("invalid_limit", "limit 必须是 1 到 200 之间的整数。", 400)
            items = vision_repository.list_jobs(current_app.config.get("DATABASE_URL"), limit=limit)
            return _ok({"jobs": [_public_vision_job(item) for item in items]})
        except RuntimeError as error:
            logger.exception("影像任务列表存储不可用")
            return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)

    auth_error = _require_admin()
    if auth_error:
        return auth_error
    inference_ready, _, _, _ = _vision_inference_readiness(current_app.config.get("DATABASE_URL"))
    if not inference_ready:
        return _err(
            "vision_not_ready",
            "视觉模型、依赖或后台工作进程尚未就绪，暂不能提交任务。",
            503,
        )
    uploaded = request.files.get("media")
    if uploaded is None or not uploaded.filename:
        return _err("missing_media", "请选择一张图片或一个视频文件。", 400)
    safe_name = secure_filename(uploaded.filename)
    extension = Path(safe_name).suffix.lower()
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    video_extensions = {".mp4", ".mov", ".avi", ".webm"}
    if extension not in image_extensions | video_extensions:
        return _err("unsupported_media", "只支持 JPG、PNG、WebP 图片和 MP4、MOV、AVI、WebM 视频。", 415)
    is_video = extension in video_extensions
    try:
        captured_at = _parse_vision_capture_time(request.form.get("captured_at"), required=True)
        capture_time_source, capture_timezone = _parse_vision_capture_provenance(
            request.form.get("capture_time_source"), request.form.get("capture_timezone"),
        )
        scene_id = request.form.get("scene_id", "").strip() or None
        if scene_id:
            uuid.UUID(scene_id)
            from storage import database
            database.initialize(current_app.config.get("DATABASE_URL"))
            if not vision_scene_repository.get_scene(current_app.config.get("DATABASE_URL"), scene_id):
                return _err("vision_scene_not_found", "观察场景不存在，请刷新场景列表。", 404)
            observation_regions = (
                _parse_scene_regions(request.form.get("observation_regions"))
                if "observation_regions" in request.form else None
            )
        else:
            observation_regions = _parse_vision_observation_regions(
                request.form.get("observation_regions"),
            )
    except (TypeError, ValueError) as error:
        return _err("invalid_observation_metadata", str(error), 400)
    raw_lng = request.form.get("lng", "").strip()
    raw_lat = request.form.get("lat", "").strip()
    anchor_gcj = None
    if raw_lng or raw_lat:
        if not raw_lng or not raw_lat:
            return _err("invalid_anchor", "经纬度必须同时提供。", 400)
        try:
            lng, lat = float(raw_lng), float(raw_lat)
        except (TypeError, ValueError):
            return _err("invalid_anchor", "影像区域坐标无效。", 400)
        bbox = config.WHU_BBOX
        if not (bbox["west"] - .01 <= lng <= bbox["east"] + .01
                and bbox["south"] - .01 <= lat <= bbox["north"] + .01):
            return _err("outside_campus", "观察点需要落在武汉大学校园范围附近。", 400)
        anchor_gcj = {"lng": lng, "lat": lat, "crs": "GCJ02"}
    upload_dir = Path(config.VISION_UPLOAD_DIR).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = secrets.token_hex(16)
    stored_name = job_id + extension
    target = upload_dir / stored_name
    digest = __import__("hashlib").sha256()
    size = 0
    try:
        with target.open("wb") as output:
            while True:
                chunk = uploaded.stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > config.VISION_MAX_MEDIA_BYTES:
                    raise OverflowError
                digest.update(chunk)
                output.write(chunk)
        if not size:
            raise ValueError("上传文件为空。")
        with target.open("rb") as source:
            header = source.read(16)
        if extension in image_extensions:
            valid_header = (
                header.startswith(b"\xff\xd8\xff")
                or header.startswith(b"\x89PNG\r\n\x1a\n")
                or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")
            )
        else:
            valid_header = (
                (header[4:8] == b"ftyp")
                or (header.startswith(b"RIFF") and header[8:12] == b"AVI ")
                or header.startswith(b"\x1aE\xdf\xa3")
            )
        if not valid_header:
            raise ValueError("文件内容与扩展名不匹配，或影像文件已损坏。")
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        scene_values = {}
        if scene_id:
            media_kind = "video" if is_video else "image"
            scene_values = _resolve_vision_scene(
                scene_id, target, media_kind,
                anchor_override=anchor_gcj,
                camera_stabilized_override=(
                    extension in video_extensions
                    and request.form.get("camera_stabilized", "false").strip().lower() == "true"
                ),
                observation_regions_override=observation_regions,
            )
        job = vision_repository.create_job(
            current_app.config.get("DATABASE_URL"),
            created_by=_admin_identity() or "unknown",
            original_name=safe_name[:180],
            media_kind="image" if extension in image_extensions else "video",
            media_path=stored_name,
            sha256=digest.hexdigest(),
            anchor_gcj=scene_values.get("anchor_gcj", anchor_gcj),
            camera_stabilized=scene_values.get("camera_stabilized", (
                extension in video_extensions
                and request.form.get("camera_stabilized", "false").strip().lower() == "true"
            )),
            captured_at=captured_at,
            capture_time_source=capture_time_source,
            capture_timezone=capture_timezone,
            observation_regions=scene_values.get("observation_regions", observation_regions),
            observation_scene_id=scene_values.get("scene_id"),
        )
        return _ok({"job": _public_vision_job(job)}, status=202)
    except _VisionSceneError as error:
        target.unlink(missing_ok=True)
        return _err(error.code, str(error), error.status)
    except OverflowError:
        target.unlink(missing_ok=True)
        return _err("media_too_large", f"文件超过 {round(config.VISION_MAX_MEDIA_BYTES / (1024 * 1024))} MB。", 413)
    except ValueError as error:
        target.unlink(missing_ok=True)
        return _err("invalid_media", str(error), 400)
    except RuntimeError as error:
        target.unlink(missing_ok=True)
        logger.exception("影像任务存储不可用")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    except Exception as error:
        target.unlink(missing_ok=True)
        logger.exception("影像任务入队失败")
        return _err("vision_enqueue_failed", "影像任务暂时无法入队，请稍后重试。", 503)


@api_bp.route("/manager/vision-jobs/<job_id>", methods=["GET", "DELETE"])
def manager_vision_job(job_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    try:
        uuid.UUID(job_id)
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        if request.method == "DELETE":
            database_url = current_app.config.get("DATABASE_URL")
            jobs = vision_repository.begin_delete_jobs(database_url, job_id)
            if not jobs:
                return _ok({"deleted": True, "already_deleted": True})
            pending_count = _purge_tombstoned_vision_jobs(database_url, jobs)
            if pending_count:
                return _ok(
                    {"deleted": True, "job_id": job_id, "deleted_count": len(jobs),
                     "cleanup_pending": True, "cleanup_pending_count": pending_count},
                    status=202,
                )
            return _ok({"deleted": True, "job_id": job_id, "deleted_count": len(jobs)})
        job = vision_repository.get_job(current_app.config.get("DATABASE_URL"), job_id)
    except (ValueError, RuntimeError) as error:
        if isinstance(error, ValueError):
            return _err("invalid_job_id", "影像任务标识无效。", 400)
        logger.exception("影像任务读取失败")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    if not job:
        return _err("job_not_found", "影像任务不存在。", 404)
    return _ok({"job": _public_vision_job(job)})


@api_bp.route("/manager/vision-jobs/<job_id>/preview-frames", methods=["GET"])
def manager_vision_preview_frames(job_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    try:
        uuid.UUID(job_id)
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        job = vision_repository.get_job(current_app.config.get("DATABASE_URL"), job_id)
    except ValueError:
        return _err("invalid_job_id", "影像任务标识无效。", 400)
    except Exception:
        logger.exception("读取视频抽样检测框失败")
        return _err("vision_storage_unavailable", "视频检测框暂时无法读取，请稍后重试。", 503)
    if not job:
        return _err("job_not_found", "影像任务不存在。", 404)
    if job.get("media_kind") != "video":
        return _err("preview_frames_unavailable", "只有视频任务包含按时间抽样的检测框。", 409)
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    media = result.get("media") if isinstance(result.get("media"), dict) else {}
    frames = result.get("preview_frames")
    previews = []
    for frame in frames[:180] if isinstance(frames, list) else []:
        if not isinstance(frame, dict):
            continue
        when = frame.get("time_seconds")
        detections = frame.get("detections")
        if (isinstance(when, bool) or not isinstance(when, (int, float))
                or not math.isfinite(float(when)) or not isinstance(detections, list)):
            continue
        previews.append({
            "time_seconds": round(float(when), 3),
            "detections": [
                {
                    "label": item["label"],
                    "box": item["box"],
                    "confidence": item["confidence"],
                    **({"track_id": item["track_id"]} if item.get("track_id") is not None else {}),
                }
                for item in detections[:80]
                if isinstance(item, dict)
                and isinstance(item.get("label"), str)
                and isinstance(item.get("box"), list) and len(item["box"]) == 4
                and isinstance(item.get("confidence"), (int, float))
                and not isinstance(item.get("confidence"), bool)
                and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(float(value)) and 0 <= value <= 1
                        for value in item["box"])
                and math.isfinite(float(item["confidence"]))
                and 0 <= item["confidence"] <= 1
            ],
        })
    interval = result.get("preview_frame_interval_seconds", 1.0)
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not math.isfinite(float(interval)) or interval <= 0):
        interval = 1.0
    return _ok({
        "media": {"width": media.get("width"), "height": media.get("height")},
        "preview_frame_interval_seconds": round(float(interval), 3),
        "preview_frames": previews,
    })


@api_bp.route("/manager/vision-jobs/<job_id>/cancel", methods=["POST"])
def cancel_manager_vision_job(job_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    try:
        uuid.UUID(job_id)
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        result = vision_repository.request_cancel_job(current_app.config.get("DATABASE_URL"), job_id)
    except ValueError:
        return _err("invalid_job_id", "影像任务标识无效。", 400)
    except Exception:
        logger.exception("取消影像任务失败")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    if not result:
        return _err("job_not_cancellable", "任务已结束或不存在，无法取消。", 409)
    return _ok({"job": result})


@api_bp.route("/manager/vision-jobs/<job_id>/retry", methods=["POST"])
def retry_manager_vision_job(job_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    try:
        uuid.UUID(job_id)
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        result = vision_repository.retry_job(current_app.config.get("DATABASE_URL"), job_id)
    except ValueError:
        return _err("invalid_job_id", "影像任务标识无效。", 400)
    except Exception:
        logger.exception("重新排队影像任务失败")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    if not result:
        return _err("job_not_retryable", "只有失败或已取消的任务可以重新排队。", 409)
    return _ok({"job": result})


@api_bp.route("/manager/vision-jobs/<job_id>/media", methods=["GET"])
def manager_vision_media(job_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    try:
        uuid.UUID(job_id)
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        job = vision_repository.get_job(current_app.config.get("DATABASE_URL"), job_id)
    except (ValueError, RuntimeError) as error:
        if isinstance(error, ValueError):
            return _err("invalid_job_id", "影像任务标识无效。", 400)
        logger.exception("影像媒体读取失败")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    if not job:
        return _err("job_not_found", "影像任务不存在。", 404)
    root = Path(config.VISION_UPLOAD_DIR).resolve()
    clip_query = request.args.get("clip")
    if clip_query is not None:
        match = re.fullmatch(r"(\d{1,4}):(\d{1,2})", clip_query)
        if not match:
            return _err("media_not_found", "影像证据片段不存在。", 404)
        candidate_index, clip_index = map(int, match.groups())
        candidates = (job.get("result") or {}).get("candidates")
        try:
            clips = candidates[candidate_index]["evidence"]["clips"]
            clip = clips[clip_index]
        except (IndexError, KeyError, TypeError):
            return _err("media_not_found", "影像证据片段不存在。", 404)
        expected_name = (
            f"{str(uuid.UUID(job_id))}-evidence-c{candidate_index}-s{clip_index}.webm"
        )
        if not isinstance(clip, dict) or clip.get("filename") != expected_name:
            return _err("media_not_found", "影像证据片段不存在。", 404)
        path = (root / expected_name).resolve()
        if path.parent != root or path.is_symlink() or not path.is_file():
            return _err("media_not_found", "影像证据片段不存在。", 404)
        return send_file(path, mimetype="video/webm", as_attachment=False,
                         download_name=f"vision-evidence-{candidate_index}-{clip_index}.webm")

    stored_name = str(job.get("media_path") or "")
    if not stored_name or Path(stored_name).name != stored_name:
        return _err("media_not_found", "影像文件不存在。", 404)
    path = root / stored_name
    if path.parent != root or path.is_symlink() or not path.is_file():
        return _err("media_not_found", "影像文件不存在。", 404)
    return send_file(path, mimetype=mimetypes.guess_type(job["original_name"])[0],
                     as_attachment=False, download_name=job["original_name"])


@api_bp.route("/manager/vision-jobs/<job_id>/review", methods=["POST"])
def review_manager_vision_job(job_id):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    body = request.get_json(silent=True) or {}
    review_status = body.get("status")
    if review_status not in {"confirmed", "dismissed"}:
        return _err("invalid_review", "审核状态只能为 confirmed 或 dismissed。", 400)
    try:
        uuid.UUID(job_id)
    except ValueError:
        return _err("invalid_job_id", "影像任务标识无效。", 400)
    try:
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        existing_job = vision_repository.get_job(current_app.config.get("DATABASE_URL"), job_id)
    except RuntimeError:
        logger.exception("影像审核存储不可用")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    if (not existing_job or existing_job.get("status") != "needs_review"
            or existing_job.get("review_status") is not None):
        return _err("job_not_reviewable", "任务不存在，或当前状态不可审核。", 409)

    review_note = str(body.get("note") or "").strip()
    if len(review_note) < 8 or len(review_note) > 1000:
        return _err("review_note_required", "请填写影像复核依据（8 至 1000 字）。", 400)
    candidate_index = body.get("candidate_index")
    if review_status == "confirmed" and (
        isinstance(candidate_index, bool) or not isinstance(candidate_index, int)
        or candidate_index < 0
    ):
        return _err("invalid_candidate", "请选择要确认的具体影像候选。", 400)
    if review_status == "dismissed":
        candidate_index = None
    try:
        if review_status == "confirmed":
            candidates = (existing_job.get("result") or {}).get("candidates")
            if (not isinstance(candidates, list)
                    or candidate_index >= len(candidates)
                    or not _valid_review_only_candidate(candidates[candidate_index])):
                return _err("vision_candidate_invalid", "影像候选记录无效。", 409)
        result = vision_repository.review_job(
            current_app.config.get("DATABASE_URL"), job_id,
            review_status=review_status,
            review_note=review_note,
            reviewed_by=_admin_identity() or "unknown",
            candidate_index=candidate_index,
        )
    except (ValueError, RuntimeError) as error:
        if isinstance(error, ValueError):
            return _err("invalid_job_id", "影像任务标识无效。", 400)
        logger.exception("影像审核存储不可用")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    if not result:
        return _err("job_not_reviewable", "任务不存在，或当前状态不可审核。", 409)
    return _ok({"job": result})


@api_bp.route("/manager/vision-jobs/<job_id>/candidates/<int:candidate_index>/review", methods=["POST"])
def review_manager_vision_candidate(job_id, candidate_index):
    auth_error = _require_admin()
    if auth_error:
        return auth_error
    body = request.get_json(silent=True) or {}
    review_status = body.get("status")
    if review_status not in {"confirmed", "dismissed"}:
        return _err("invalid_review", "审核状态只能是确认或排除。", 400)
    review_note = str(body.get("note") or "").strip()
    if len(review_note) < 8 or len(review_note) > 1000:
        return _err("review_note_required", "请填写影像复核依据（8 至 1000 字）。", 400)
    try:
        uuid.UUID(job_id)
        from storage import database, vision_repository
        database.initialize(current_app.config.get("DATABASE_URL"))
        job = vision_repository.get_job(current_app.config.get("DATABASE_URL"), job_id)
        candidates = (job.get("result") or {}).get("candidates") if job else None
        if (not isinstance(candidates, list) or candidate_index >= len(candidates)
                or not _valid_review_only_candidate(candidates[candidate_index])):
            return _err("vision_candidate_invalid", "影像候选记录无效。", 409)
        result = vision_repository.review_candidate(
            current_app.config.get("DATABASE_URL"), job_id,
            candidate_index=candidate_index, review_status=review_status,
            review_note=review_note, reviewed_by=_admin_identity() or "unknown",
        )
    except ValueError:
        return _err("invalid_job_id", "影像任务标识无效。", 400)
    except Exception:
        logger.exception("独立复核影像候选失败")
        return _err("vision_storage_unavailable", "影像任务暂时不可用，请稍后重试。", 503)
    if not result:
        return _err("candidate_already_reviewed", "候选已复核、任务已结束或索引无效，请刷新任务列表。", 409)
    return _ok({"job": result})


# ===== 行为埋点（P4：用户画像学习 + 产品观测）=====

_TELEMETRY_PATH = Path(__file__).parent.parent / "data" / "telemetry.jsonl"
_ROUTE_SIGNAL_EVENTS = {
    "route_shown", "strategy_selected", "strategy_abandoned",
    "navigation_started", "navigation_completed",
}
_TELEMETRY_EVENTS = {
    *_ROUTE_SIGNAL_EVENTS, "candidate_click",
    "clarify_answer", "chat", "error",
}


@api_bp.route("/telemetry", methods=["POST"])
def telemetry():
    """POST /api/telemetry — 前端行为埋点。

    路线信号必须同时携带 route_id、strategy、strategy_source、applied_weights。
    曝光和策略点选只作统计；只有显式休闲策略开始/完成导航才更新画像。
    所有事件追加写入 data/telemetry.jsonl 供离线分析。
    埋点是锦上添花：任何失败都返回 ok，绝不影响前端主流程。
    """
    body = request.get_json(silent=True) or {}
    uid = body.get("uid") or body.get("whu_uid")
    event = body.get("event")

    if not uid or event not in _TELEMETRY_EVENTS:
        return _ok({"recorded": False})

    if event in _ROUTE_SIGNAL_EVENTS:
        route_id = body.get("route_id")
        strategy = body.get("strategy")
        strategy_source = body.get("strategy_source")
        weights = body.get("applied_weights")
        if not all(isinstance(value, str) and value for value in (
            route_id, strategy, strategy_source,
        )) or not isinstance(weights, dict):
            return _ok({"recorded": False})

    record = {
        "ts": time.time(),
        "uid": uid,
        "event": event,
        "payload": {k: v for k, v in body.items() if k not in ("uid", "whu_uid", "event")},
    }
    try:
        with open(_TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("埋点写入失败: %s", e)

    # 画像学习：只由已验证的路线信号入口决定是否更新。
    try:
        from agents import profile
        if event in _ROUTE_SIGNAL_EVENTS:
            profile.record_strategy_signal(
                uid=uid,
                route_id=body["route_id"],
                strategy=body["strategy"],
                strategy_source=body["strategy_source"],
                applied_weights=body["applied_weights"],
                signal=event,
            )
    except Exception as e:
        logger.warning("画像更新失败: %s", e)

    return _ok({"recorded": True})
