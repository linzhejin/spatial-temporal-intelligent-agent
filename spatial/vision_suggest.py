"""疑似受影响路段的三级降级打分编排（纯 shapely，供无 numpy 的生产 api 进程用）。

管理端「疑似受影响路段」建议的三级降级：
  - L2：配准掩膜投影（vision.registration.project_mask_to_edges，被动注入，需 numpy）
  - L1：姿态拍摄范围 footprint（spatial.vision_footprint）
  - L0：半径圈（spatial.road_conditions.nearest_edges）

此模块仅供隔离验证；生产接口只提供邻近参考。registration 由调用方注入。
camera_pose 与路网内部为 WGS-84，anchor 与输出展示字段为 GCJ-02。
"""

import logging
import math

from shapely.geometry import Polygon

from spatial.coord_transform import gcj02_to_wgs84
from spatial.road_conditions import nearest_edges
from spatial.vision_footprint import pose_to_footprint, project_footprint_to_edges

logger = logging.getLogger(__name__)

_M_PER_DEG = 111320.0
_MAX_SUGGESTED = 5
_MAX_OUTLINES = 8
_DEFAULT_ALT_M = 50.0
_MAX_POSE_TIME_DIFFERENCE_S = 0.5


def _finite_number(value) -> bool:
    """数值（排除 bool）且有限。"""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive_number(value) -> bool:
    """严格正数且有限。"""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _valid_point(point: dict) -> bool:
    """轨迹点是否为可用定位点（有限、合法经纬、非 (0,0)）。"""
    if not isinstance(point, dict):
        return False
    lng = point.get("lng")
    lat = point.get("lat")
    if not _finite_number(lng) or not _finite_number(lat):
        return False
    lng = float(lng)
    lat = float(lat)
    if abs(lng) > 180.0 or abs(lat) > 90.0:
        return False
    if lng == 0.0 and lat == 0.0:
        return False
    return True


def _dist_sq_m(point: dict, lng0: float, lat0: float) -> float:
    """轨迹点到 WGS 原点的近似距离平方（米）。"""
    lng = float(point["lng"])
    lat = float(point["lat"])
    dx = (lng - lng0) * _M_PER_DEG * math.cos(math.radians(lat0))
    dy = (lat - lat0) * _M_PER_DEG
    return dx * dx + dy * dy


def _effective_alt(point: dict):
    """只有经确认的离地高度可用于几何投影。"""
    value = point.get("height_agl_m")
    return float(value) if _positive_number(value) else None


def _alt_near_anchor(camera_pose, anchor_lng_wgs: float, anchor_lat_wgs: float) -> float:
    """取距 anchor 最近轨迹点的有效高度（全无 → 默认 50.0 米）。"""
    trajectory = camera_pose.get("trajectory") if isinstance(camera_pose, dict) else None
    points = [pt for pt in trajectory if _valid_point(pt)] if isinstance(trajectory, list) else []
    if not points:
        return _DEFAULT_ALT_M
    closest = min(points, key=lambda pt: _dist_sq_m(pt, anchor_lng_wgs, anchor_lat_wgs))
    alt = _effective_alt(closest)
    if alt is not None:
        return alt
    for pt in points:
        alt = _effective_alt(pt)
        if alt is not None:
            return alt
    return _DEFAULT_ALT_M


def _pick_pose(camera_pose, anchor, time_seconds=None, *, require_time=False) -> dict:
    """从轨迹挑出与 mask 时间 / anchor 距离最近的姿态点；无可用点抛 ValueError。"""
    anchor_lng_wgs, anchor_lat_wgs = gcj02_to_wgs84(
        float(anchor["lng"]), float(anchor["lat"]))
    trajectory = camera_pose.get("trajectory") if isinstance(camera_pose, dict) else None
    if not isinstance(trajectory, list):
        raise ValueError("camera_pose.trajectory missing or not a list")
    valid = [pt for pt in trajectory if _valid_point(pt)]
    if not valid:
        raise ValueError("camera_pose.trajectory has no usable points")

    if require_time and not _finite_number(time_seconds):
        raise ValueError("a valid frame timestamp is required")
    if time_seconds is not None:
        if not _finite_number(time_seconds):
            raise ValueError("invalid frame timestamp")
        target = float(time_seconds)
        timed = [pt for pt in valid if _finite_number(pt.get("seconds"))]
        if not timed:
            raise ValueError("trajectory has no timestamps")
        t_min = min(float(pt["seconds"]) for pt in timed)
        t_max = max(float(pt["seconds"]) for pt in timed)
        if not t_min <= target <= t_max:
            raise ValueError("frame timestamp outside trajectory")
        closest = min(timed, key=lambda pt: abs(float(pt["seconds"]) - target))
        if abs(float(closest["seconds"]) - target) > _MAX_POSE_TIME_DIFFERENCE_S:
            raise ValueError("no sufficiently close camera pose")
        return closest

    return min(valid, key=lambda pt: _dist_sq_m(pt, anchor_lng_wgs, anchor_lat_wgs))


def _media_size(media):
    """media 的像素尺寸（宽、高均为正数）→ (w, h)；否则 None。"""
    if not isinstance(media, dict):
        return None
    w = media.get("width")
    h = media.get("height")
    if _positive_number(w) and _positive_number(h):
        return float(w), float(h)
    return None


def _flood_outlines(candidates):
    """取第一条可能积水候选的有效 outline 列表与时间；无 → (None, None)。"""
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        if cand.get("kind") != "possible_flooding" and cand.get("type") != "possible_flooding":
            continue
        evidence = cand.get("evidence")
        if not isinstance(evidence, dict):
            continue
        summary = evidence.get("summary")
        if not isinstance(summary, dict):
            continue
        outlines = summary.get("outline_polygons")
        if isinstance(outlines, list) and outlines:
            return outlines, summary.get("outline_time_seconds")
    return None, None


def _outline_polygon(outline, width, height):
    """归一化 outline → 像素坐标 Polygon；<3 点 / 非法 / 面积≤0 → None。"""
    if not isinstance(outline, (list, tuple)):
        return None
    coords = []
    for point in outline:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
                return None
            px = x * (width - 1.0)
            py = y * (height - 1.0)
        except (TypeError, ValueError, OverflowError):
            return None
        coords.append((px, py))
    if len(coords) < 3:
        return None
    try:
        polygon = Polygon(coords)
    except Exception:
        return None
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        return None
    return polygon


def _pool_index(pool):
    """pool 项按 (u, v, key) → item 建索引。"""
    return {(item["u"], item["v"], item["key"]): item for item in pool}


def _relay(score_map, pool_index, basis) -> dict:
    """命中边按 (u,v,key) 转挂 pool 展示字段，加 score/basis，按 score 降序截断。"""
    items = []
    for key, overlap in score_map.items():
        item = pool_index.get(key)
        if item is None:
            continue
        out = dict(item)
        out["score"] = round(float(overlap), 4)
        out["basis"] = basis
        items.append(out)
    items.sort(key=lambda it: it["score"], reverse=True)
    return {"suggested_edges": items[:_MAX_SUGGESTED], "basis": basis}


def _radius_only(G, lng_gcj, lat_gcj) -> dict:
    """L0：半径圈，沿用原有 nearest_edges 的顺序与字段。"""
    items = []
    for item in nearest_edges(G, lng_gcj, lat_gcj):
        out = dict(item)
        out["score"] = 0.0
        out["basis"] = "radius_only"
        items.append(out)
    return {"suggested_edges": items[:_MAX_SUGGESTED], "basis": "radius_only"}


def _try_mask_projection(G, anchor, pool, pool_index, camera_pose, media,
                         candidates, registration):
    """L2：配准掩膜投影。前置条件不满足或任何异常 → None（降级 L1）。"""
    if registration is None:
        return None
    size = _media_size(media)
    if size is None:
        return None
    width, height = size
    outlines, outline_time = _flood_outlines(candidates)
    if outlines is None:
        return None

    try:
        merged = {}
        for outline in outlines[:_MAX_OUTLINES]:
            polygon = _outline_polygon(outline, width, height)
            if polygon is None:
                continue
            pose = _pick_pose(camera_pose, anchor, time_seconds=outline_time, require_time=True)
            transform = registration.pose_to_homography_init(pose, int(width), int(height))
            for hit in registration.project_mask_to_edges(
                    polygon, transform, G, pool,
                    image_w_px=int(width), image_h_px=int(height)):
                ratio = float(hit["overlap_ratio"])
                if ratio <= 0:
                    continue
                key = (hit["u"], hit["v"], hit["k"])
                merged[key] = max(merged.get(key, 0.0), ratio)
    except Exception as error:
        logger.warning("suggested-edges L2 mask scoring skipped: %s", error)
        return None

    if not merged:
        return None
    return _relay(merged, pool_index, basis="mask_projection")


def _try_footprint(G, anchor, pool, pool_index, camera_pose, media, time_seconds=None):
    """L1：姿态拍摄范围 footprint。无轨迹/无高度/异常 → None（降级 L0）。"""
    try:
        pose = _pick_pose(camera_pose, anchor, time_seconds=time_seconds,
                          require_time=time_seconds is not None)
        size = _media_size(media)
        if size is None:
            raise ValueError("image dimensions unavailable")
        image_w, image_h = int(size[0]), int(size[1])
        foot = pose_to_footprint(pose, image_w, image_h)
        hits = project_footprint_to_edges(foot["polygon"], foot["origin"], G, pool)
        if not hits:
            return None
        score_map = {
            (hit["u"], hit["v"], hit["k"]): float(hit["overlap_ratio"]) for hit in hits
        }
        return _relay(score_map, pool_index, basis="pose_footprint")
    except Exception as error:
        logger.warning("suggested-edges L1 footprint scoring skipped: %s", error)
        return None


def score_suggested_edges(G, anchor: dict, *, camera_pose=None, media=None,
                          candidates=None, registration=None) -> dict:
    """给「疑似受影响路段」建议三级降级打分，返回 ≤5 条 + 实际执行的层级。

    编排顺序：L2-mask（配准掩膜投影）→ L1-footprint（姿态拍摄范围）→ L0-radius
    （半径圈）。任何一级未满足条件或异常都降级到下一级，绝不抛出。
    """
    anchor_lng_gcj = float(anchor["lng"])
    anchor_lat_gcj = float(anchor["lat"])

    # Step 1：取池（anchor 半径 R 内边，复用 nearest_edges 的距离/去重语义）
    anchor_lng_wgs, anchor_lat_wgs = gcj02_to_wgs84(anchor_lng_gcj, anchor_lat_gcj)
    alt = _alt_near_anchor(camera_pose, anchor_lng_wgs, anchor_lat_wgs)
    radius_m = min(500.0, max(50.0, 3.0 * alt))
    pool = nearest_edges(G, anchor_lng_gcj, anchor_lat_gcj, max_dist_m=radius_m, limit=40)
    if not pool:
        return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj)
    pool_index = _pool_index(pool)

    outlines, outline_time = _flood_outlines(candidates)
    if outlines is not None:
        try:
            _pick_pose(camera_pose, anchor, time_seconds=outline_time, require_time=True)
        except ValueError as error:
            logger.info("suggested-edges frame time unavailable; using proximity: %s", error)
            return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj)
    else:
        trajectory = camera_pose.get("trajectory") if isinstance(camera_pose, dict) else None
        valid = [pt for pt in trajectory if _valid_point(pt)] if isinstance(trajectory, list) else []
        if not isinstance(media, dict) or media.get("kind") != "image" or len(valid) != 1:
            logger.info("suggested-edges image time or pose ambiguous; using proximity")
            return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj)

    # Step 2：L2 配准掩膜投影
    result = _try_mask_projection(
        G, anchor, pool, pool_index, camera_pose, media, candidates, registration)
    if result is not None:
        return result

    # Step 3：L1 姿态 footprint
    result = _try_footprint(G, anchor, pool, pool_index, camera_pose, media,
                            time_seconds=outline_time if outlines is not None else None)
    if result is not None:
        return result

    # Step 4：L0 半径圈
    return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj)
