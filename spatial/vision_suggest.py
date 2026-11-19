"""疑似受影响路段的三级降级打分编排（纯 shapely，供无 numpy 的生产 api 进程用）。

管理端「疑似受影响路段」建议的三级降级：
  - L2：配准掩膜投影（vision.registration.project_mask_to_edges，被动注入，需 numpy）
  - L1：姿态拍摄范围 footprint（spatial.vision_footprint）
  - L0：半径圈（spatial.road_conditions.nearest_edges）

生产在相机内参标定通过后使用 footprint；未标定或缺少姿态时降级为邻近参考。
registration 由调用方注入。camera_pose 与路网内部为 WGS-84，anchor 与输出展示字段为 GCJ-02。
"""

import logging
import math

from shapely.geometry import LineString, Polygon

from spatial.coord_transform import gcj02_to_wgs84, wgs84_to_gcj02
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
    """从轨迹挑出与 mask 时间 / anchor 距离最近的姿态点；无可用点抛 ValueError。

    anchor 为 None 时只按时间或第一个有效点选取，用于“影像自带定位”路径。
    """
    anchor_lng_wgs = anchor_lat_wgs = None
    if anchor is not None:
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

    if anchor_lng_wgs is None:
        return valid[0]
    return min(valid, key=lambda pt: _dist_sq_m(pt, anchor_lng_wgs, anchor_lat_wgs))


def _pose_anchor_gcj(camera_pose, time_seconds=None) -> dict | None:
    """由姿态点生成 GCJ 锚点；无可用姿态返回 None。"""
    try:
        pose = _pick_pose(camera_pose, None, time_seconds=time_seconds)
    except ValueError:
        return None
    try:
        lng, lat = wgs84_to_gcj02(float(pose["lng"]), float(pose["lat"]))
    except (TypeError, ValueError, KeyError):
        return None
    return {"lng": lng, "lat": lat, "crs": "GCJ02"}


def _media_size(media):
    """media 的像素尺寸（宽、高均为正数）→ (w, h)；否则 None。"""
    if not isinstance(media, dict):
        return None
    w = media.get("width")
    h = media.get("height")
    if _positive_number(w) and _positive_number(h):
        return float(w), float(h)
    return None


def _intrinsics_from_profile(profile, width, height):
    fx = profile.get("fx_px") if isinstance(profile, dict) else None
    fy = profile.get("fy_px") if isinstance(profile, dict) else None
    if not (_positive_number(fx) and _positive_number(fy)):
        focal = profile.get("focal_35mm") if isinstance(profile, dict) else None
        focal = float(focal) if _positive_number(focal) else 24.0
        fx = width * focal / 36.0
        fy = fx
    return {
        "fx_px": float(fx),
        "fy_px": float(fy),
        "cx_px": width / 2.0,
        "cy_px": height / 2.0,
    }


def _with_intrinsics(pose, profile, media):
    if isinstance(pose.get("intrinsics"), dict):
        return pose
    if not isinstance(profile, dict):
        raise ValueError("camera profile required for projection")
    size = _media_size(media)
    if size is None:
        raise ValueError("image dimensions unavailable")
    width, height = size
    return {**pose, "intrinsics": _intrinsics_from_profile(profile, width, height)}


def _footprint_to_gcj(polygon: Polygon, origin: dict) -> list[list[float]]:
    """局部米制 footprint → GCJ-02 多边形坐标。"""
    lng0 = float(origin["lng_wgs"])
    lat0 = float(origin["lat_wgs"])
    east_scale = _M_PER_DEG * math.cos(math.radians(lat0))
    coords = []
    for x, y in polygon.exterior.coords:
        lng = lng0 + float(x) / east_scale
        lat = lat0 + float(y) / _M_PER_DEG
        gcj_lng, gcj_lat = wgs84_to_gcj02(lng, lat)
        coords.append([round(gcj_lng, 7), round(gcj_lat, 7)])
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords.pop()
    return coords


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


def _candidate_region(candidate, observation_regions):
    """Return candidate-specific normalized polygons, region id, and evidence time."""
    if not isinstance(candidate, dict):
        return None, None, None
    evidence = candidate.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    summary = evidence.get("summary")
    summary = summary if isinstance(summary, dict) else evidence
    region_id = summary.get("region_id") or evidence.get("region_id")

    polygons = summary.get("outline_polygons")
    if isinstance(polygons, list) and polygons:
        polygons = polygons[:_MAX_OUTLINES]
    else:
        polygon = summary.get("region_polygon")
        if not isinstance(polygon, (list, tuple)) and isinstance(region_id, str):
            region = next((item for item in (observation_regions or [])
                           if isinstance(item, dict) and item.get("id") == region_id), None)
            polygon = region.get("polygon") if isinstance(region, dict) else None
        polygons = [polygon] if isinstance(polygon, (list, tuple)) else None

    time_seconds = summary.get("outline_time_seconds")
    if not _finite_number(time_seconds):
        time_seconds = summary.get("event_time_seconds")
    if not _finite_number(time_seconds):
        segments = evidence.get("segments")
        segments = [item for item in segments if isinstance(item, dict)] if isinstance(segments, list) else []
        timed = []
        for segment in segments:
            when = segment.get("time_seconds")
            if not _finite_number(when):
                start, end = segment.get("start_seconds"), segment.get("end_seconds")
                if _finite_number(start) and _finite_number(end) and float(end) >= float(start):
                    # Detector evidence timestamps the beginning of each sampled
                    # interval; using its midpoint can exceed a sparse pose track.
                    when = float(start)
            if _finite_number(when):
                timed.append((float(segment.get("confidence", 0.0))
                              if _finite_number(segment.get("confidence", 0.0)) else 0.0,
                              float(when)))
        time_seconds = max(timed, default=(0.0, None))[1]
    return polygons, region_id, time_seconds


def score_candidate_road_association(candidate, G, anchor=None, *, observation_regions=None,
                                     camera_pose=None, media=None, registration=None,
                                     camera_profile=None, calibration=None) -> dict:
    """Associate one candidate ROI with roads; only a clear geometric winner is selected.

    ``score`` is a buffered-road overlap ratio, not a calibrated probability. A
    candidate-specific projection is required; job-wide camera footprints and
    radius suggestions are deliberately not promoted to a candidate match.
    """
    calibration = calibration if isinstance(calibration, dict) else {"status": "nominal"}
    calibration_status = calibration.get("status", "nominal")
    polygons, region_id, time_seconds = _candidate_region(candidate, observation_regions)
    base = {
        "status": "context_only",
        "basis": "candidate_region_projection",
        "region_id": region_id,
        "calibration_status": calibration_status,
        "ranked_edges": [],
        "selected_edge": None,
        "score_semantics": "road_buffer_overlap_ratio_not_probability",
    }
    if G is None or not hasattr(G, "edges"):
        return {**base, "status": "unavailable", "reason_code": "road_network_unavailable"}
    if registration is None:
        return {**base, "status": "unavailable", "reason_code": "projection_runtime_unavailable"}
    if not polygons:
        return {**base, "reason_code": "candidate_region_geometry_missing"}
    size = _media_size(media)
    if size is None:
        return {**base, "status": "insufficient_pose", "reason_code": "media_dimensions_missing"}
    width, height = size
    media_kind = media.get("kind") if isinstance(media, dict) else None
    if media_kind == "image":
        evidence = candidate.get("evidence") if isinstance(candidate, dict) else None
        summary = evidence.get("summary") if isinstance(evidence, dict) else None
        summary = summary if isinstance(summary, dict) else {}
        explicit_time = summary.get("outline_time_seconds", summary.get("event_time_seconds"))
        time_seconds = float(explicit_time) if _finite_number(explicit_time) else None
    if media_kind == "video" and not _finite_number(time_seconds):
        return {**base, "status": "insufficient_pose", "reason_code": "candidate_time_missing"}

    try:
        if anchor is None:
            anchor = _pose_anchor_gcj(camera_pose, time_seconds=time_seconds)
            if anchor is None:
                return {**base, "status": "insufficient_pose", "reason_code": "anchor_missing"}
        pose = _pick_pose(camera_pose, anchor, time_seconds=time_seconds,
                          require_time=media_kind == "video")
        if not _positive_number(pose.get("height_agl_m")):
            return {**base, "status": "insufficient_pose", "reason_code": "height_agl_missing"}
        for field, lower, upper in (
            ("gb_yaw_deg", -360.0, 360.0),
            ("gb_pitch_deg", -90.0, 0.0),
            ("gb_roll_deg", -180.0, 180.0),
        ):
            value = pose.get(field)
            in_range = (_finite_number(value) and lower <= float(value) and
                        (float(value) < upper if field == "gb_pitch_deg" else float(value) <= upper))
            if not in_range:
                return {**base, "status": "insufficient_pose",
                        "reason_code": f"{field}_missing_or_invalid"}
        pose = _with_intrinsics(pose, camera_profile, media)
        transform = registration.pose_to_homography_init(pose, int(width), int(height))
    except Exception as error:
        logger.info("candidate road association pose unavailable: %s", error)
        return {**base, "status": "insufficient_pose", "reason_code": "pose_projection_unavailable"}

    try:
        anchor_lng_wgs, anchor_lat_wgs = gcj02_to_wgs84(float(anchor["lng"]), float(anchor["lat"]))
        radius_m = min(500.0, max(50.0, 3.0 * float(pose["height_agl_m"])))
        pool = nearest_edges(G, float(anchor["lng"]), float(anchor["lat"]),
                             max_dist_m=radius_m, limit=200)
        pool_index = _pool_index(pool)
        if not pool:
            return {**base, "status": "no_match", "reason_code": "no_roads_near_candidate"}
        merged = {}
        for outline in polygons:
            polygon = _outline_polygon(outline, width, height)
            if polygon is None:
                continue
            for hit in registration.project_mask_to_edges(
                    polygon, transform, G, pool,
                    image_w_px=int(width), image_h_px=int(height)):
                ratio = float(hit.get("overlap_ratio", 0.0))
                if not math.isfinite(ratio) or ratio <= 0:
                    continue
                key = (int(hit["u"]), int(hit["v"]), int(hit.get("k", hit.get("key", 0))))
                merged[key] = max(merged.get(key, 0.0), ratio)
    except Exception as error:
        logger.warning("candidate road association projection skipped: %s", error)
        return {**base, "status": "unavailable", "reason_code": "projection_failed"}

    ranked = _relay(merged, pool_index, basis="candidate_region_projection")["suggested_edges"]
    if not ranked or ranked[0]["score"] < 0.05:
        return {**base, "status": "no_match", "ranked_edges": ranked,
                "reason_code": "candidate_region_has_no_clear_road_overlap"}
    second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
    margin = round(ranked[0]["score"] - second_score, 4)
    enriched = {**base, "ranked_edges": ranked, "score_margin": margin}
    if margin < 0.05:
        return {**enriched, "status": "ambiguous", "reason_code": "road_overlap_not_distinct"}
    reason_code = ("calibrated_candidate_region_unique_overlap"
                   if calibration_status == "calibrated"
                   else "nominal_candidate_region_unique_overlap")
    return {**enriched, "status": "matched", "selected_edge": ranked[0],
            "manual_confirmation_required": True, "reason_code": reason_code}


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


def _relay(score_map, pool_index, basis, *, extra=None) -> dict:
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
    result = {"suggested_edges": items[:_MAX_SUGGESTED], "basis": basis}
    if extra:
        result.update(extra)
    return result


def _radius_only(G, lng_gcj, lat_gcj, *, extra=None) -> dict:
    """L0：半径圈，沿用原有 nearest_edges 的顺序与字段。"""
    items = []
    for item in nearest_edges(G, lng_gcj, lat_gcj):
        out = dict(item)
        out["score"] = 0.0
        out["basis"] = "radius_only"
        items.append(out)
    result = {"suggested_edges": items[:_MAX_SUGGESTED], "basis": "radius_only"}
    if extra:
        result.update(extra)
    return result


def _try_mask_projection(G, anchor, pool, pool_index, camera_pose, media,
                         candidates, registration, camera_profile):
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
            pose = _with_intrinsics(pose, camera_profile, media)
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


def _try_footprint(G, anchor, pool, pool_index, camera_pose, media, camera_profile,
                   time_seconds=None):
    """L1：姿态拍摄范围 footprint。无轨迹/无高度/异常 → None（降级 L0）。"""
    try:
        pose = _pick_pose(camera_pose, anchor, time_seconds=time_seconds,
                          require_time=time_seconds is not None)
        pose = _with_intrinsics(pose, camera_profile, media)
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
        return _relay(score_map, pool_index, basis="pose_footprint", extra={
            "footprint": _footprint_to_gcj(foot["polygon"], foot["origin"]),
            "footprint_area_m2": round(float(foot["polygon"].area), 1),
        })
    except Exception as error:
        logger.warning("suggested-edges L1 footprint scoring skipped: %s", error)
        return None


def _context(camera_pose, anchor_source, calibration):
    pose_source = None
    if isinstance(camera_pose, dict):
        pose_source = camera_pose.get("source") or "srt"
    return {
        "anchor_source": anchor_source,
        "pose_source": pose_source,
        "calibration_status": calibration.get("status", "nominal"),
        "calibration_residual_m": calibration.get("residual_m"),
        "calibration_ground_offset_m": calibration.get("ground_offset_m", 0.0),
    }


def score_suggested_edges(G, anchor=None, *, camera_pose=None, media=None,
                          candidates=None, registration=None, camera_profile=None,
                          calibration=None) -> dict:
    """给「疑似受影响路段」建议三级降级打分，返回 ≤5 条 + 实际执行的层级。

    anchor 缺省时用姿态点生成临时锚点，不再要求人工标点；两者都没有时抛
    ValueError，由接口返回明确错误。编排顺序：L2-mask → L1-footprint → L0-radius。
    """
    calibration = calibration if isinstance(calibration, dict) else {
        "status": "nominal", "residual_m": None, "ground_offset_m": 0.0,
    }
    outlines, outline_time = _flood_outlines(candidates)
    anchor_source = "manual"
    if anchor is None:
        anchor = _pose_anchor_gcj(
            camera_pose,
            time_seconds=outline_time if outlines is not None else None,
        )
        if anchor is None:
            raise ValueError("anchor or camera pose required")
        anchor_source = "pose"
    context = _context(camera_pose, anchor_source, calibration)

    anchor_lng_gcj = float(anchor["lng"])
    anchor_lat_gcj = float(anchor["lat"])

    # Step 1：取池（锚点半径 R 内边，复用 nearest_edges 的距离/去重语义）
    anchor_lng_wgs, anchor_lat_wgs = gcj02_to_wgs84(anchor_lng_gcj, anchor_lat_gcj)
    alt = _alt_near_anchor(camera_pose, anchor_lng_wgs, anchor_lat_wgs)
    radius_m = min(500.0, max(50.0, 3.0 * alt))
    pool = nearest_edges(G, anchor_lng_gcj, anchor_lat_gcj, max_dist_m=radius_m, limit=200)
    if not pool:
        return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj, extra=context)
    pool_index = _pool_index(pool)

    if outlines is not None:
        try:
            _pick_pose(camera_pose, anchor, time_seconds=outline_time, require_time=True)
        except ValueError as error:
            logger.info("suggested-edges frame time unavailable; using proximity: %s", error)
            return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj, extra=context)
    else:
        trajectory = camera_pose.get("trajectory") if isinstance(camera_pose, dict) else None
        valid = [pt for pt in trajectory if _valid_point(pt)] if isinstance(trajectory, list) else []
        if not isinstance(media, dict) or media.get("kind") != "image" or len(valid) != 1:
            logger.info("suggested-edges image time or pose ambiguous; using proximity")
            return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj, extra=context)

    # Step 2：L2 配准掩膜投影
    result = _try_mask_projection(
        G, anchor, pool, pool_index, camera_pose, media, candidates,
        registration, camera_profile)
    if result is not None:
        return {**result, **context}

    # Step 3：L1 姿态 footprint
    result = _try_footprint(
        G, anchor, pool, pool_index, camera_pose, media, camera_profile,
        time_seconds=outline_time if outlines is not None else None)
    if result is not None:
        return {**result, **context}

    # Step 4：L0 半径圈
    return _radius_only(G, anchor_lng_gcj, anchor_lat_gcj, extra=context)
