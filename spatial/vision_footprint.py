"""L1 姿态拍摄范围打分（纯 shapely，供无 numpy 的生产 api 进程用）。

管理端「疑似受影响路段」建议的三级降级：L2-mask / L1-footprint / L0-radius。
本模块是 L1-footprint 的纯几何实现：

  - pose_to_footprint：把相机姿态（位置、高度、pitch、yaw、焦距、像幅）换算成
    「相机拍摄地面范围」的多边形（局部米制平面，以 pose 位置为原点）。
  - project_footprint_to_edges：把候选路网边折算到同一局部平面，与 footprint
    的 buffer 相交，按覆盖比例打分排序。

只依赖 shapely + 标准库 + spatial.coord_transform，顶层禁止 import numpy /
vision.registration / Flask / storage。坐标约定：pose 与路网为 WGS-84，
GCJ-02 仅在输出 origin 与 geometry_gcj 回退的边界处转换。
"""

import math

from shapely.geometry import LineString, Polygon

from spatial.coord_transform import gcj02_to_wgs84, wgs84_to_gcj02

_M_PER_DEG = 111320.0


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


def _wgs_line_from_geometry(geometry):
    """把边 geometry（LineString 或 WKT 字符串）转成 WGS-84 LineString；失败返回 None。"""
    if geometry is None:
        return None
    if hasattr(geometry, "coords"):
        try:
            coords = [[float(x), float(y)] for x, y in geometry.coords]
        except Exception:
            return None
        return LineString(coords) if len(coords) >= 2 else None
    if isinstance(geometry, str) and geometry.strip():
        try:
            from shapely import wkt
            line = wkt.loads(geometry)
            if line.geom_type == "LineString" and len(line.coords) >= 2:
                return line
        except Exception:
            return None
    return None


def _wgs_line_from_gcj(geometry_gcj):
    """把 GCJ-02 折线逐点转 WGS-84 LineString；失败返回 None。"""
    if not isinstance(geometry_gcj, (list, tuple)) or len(geometry_gcj) < 2:
        return None
    coords = []
    for point in geometry_gcj:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            lng, lat = gcj02_to_wgs84(float(point[0]), float(point[1]))
        except (TypeError, ValueError):
            continue
        coords.append((lng, lat))
    return LineString(coords) if len(coords) >= 2 else None


def _road_name(edge_data, candidate) -> str:
    """道路名：优先边属性 name，缺失时回退候选快照的 road_name。"""
    if isinstance(edge_data, dict):
        raw = edge_data.get("name")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(raw, list) and raw:
            return str(raw[0]).strip()
    return str(candidate.get("road_name") or "") if isinstance(candidate, dict) else ""


def pose_to_footprint(pose: dict, image_w_px: int, image_h_px: int, *,
                      focal_len_35mm: float | None = None,
                      sensor_w_mm: float = 36.0) -> dict:
    """把相机姿态换算成「拍摄地面范围」多边形（局部米制平面，原点 = pose 位置）。

    数学约定（DJI 惯例 gb_pitch=-90 为正下视）：
      - focal = 传入 focal_len_35mm（有效正数）else pose.focal_len_35mm else 24.0
      - f_px = focal / sensor_w_mm * image_w_px
      - hfov_half = atan((image_w/2)/f_px), vfov_half = atan((image_h/2)/f_px)
      - tilt = 90 - abs(gb_pitch); offset = alt * tan(tilt)
      - 局部平面（东、北）: right=(cos(yaw), -sin(yaw)), up=(sin(yaw), cos(yaw))
      - 中心 C = up * offset；四角 = C ± right*h_w ± up*h_h

    Returns:
        {"polygon": Polygon(局部米制四角), "origin": {lng_wgs, lat_wgs, lng_gcj, lat_gcj}}
    """
    if not _positive_number(image_w_px) or not _positive_number(image_h_px):
        raise ValueError("image dimensions must be positive numbers")
    image_w = float(image_w_px)
    image_h = float(image_h_px)

    focal_candidates = (focal_len_35mm, pose.get("focal_len_35mm"))
    focal = next((float(f) for f in focal_candidates if _positive_number(f)), 24.0)
    if not _positive_number(sensor_w_mm):
        raise ValueError("sensor width must be a positive number")
    sensor_w = float(sensor_w_mm)

    f_px = focal / sensor_w * image_w
    hfov_half = math.atan((image_w / 2.0) / f_px)
    vfov_half = math.atan((image_h / 2.0) / f_px)

    alt = None
    for field in ("rel_alt_m", "abs_alt_m"):
        value = pose.get(field)
        if _finite_number(value):
            alt = float(value)
            break
    if alt is None or alt <= 0:
        raise ValueError("pose altitude missing or invalid")

    gb_pitch_raw = pose.get("gb_pitch_deg")
    gb_pitch = float(gb_pitch_raw) if _finite_number(gb_pitch_raw) else -90.0
    tilt_deg = 90.0 - abs(gb_pitch)
    if tilt_deg > 85.0:
        raise ValueError("pose pitch too shallow for nadir footprint")
    tilt_rad = math.radians(tilt_deg)
    offset = alt * math.tan(tilt_rad)

    yaw = 0.0
    for field in ("gb_yaw_deg", "yaw_deg"):
        value = pose.get(field)
        if _finite_number(value):
            yaw = float(value)
            break
    yaw_rad = math.radians(yaw)

    if not _finite_number(pose.get("lng")) or not _finite_number(pose.get("lat")):
        raise ValueError("pose position missing or invalid")
    lng = float(pose["lng"])
    lat = float(pose["lat"])

    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    right = (c, -s)  # 右手（东、北分量）
    up = (s, c)      # 机头前向（东、北分量）
    center = (up[0] * offset, up[1] * offset)

    h_w = alt * math.tan(hfov_half)
    h_h = alt * math.tan(vfov_half)

    corners = [
        (center[0] + right[0] * h_w + up[0] * h_h,
         center[1] + right[1] * h_w + up[1] * h_h),
        (center[0] - right[0] * h_w + up[0] * h_h,
         center[1] - right[1] * h_w + up[1] * h_h),
        (center[0] - right[0] * h_w - up[0] * h_h,
         center[1] - right[1] * h_w - up[1] * h_h),
        (center[0] + right[0] * h_w - up[0] * h_h,
         center[1] + right[1] * h_w - up[1] * h_h),
    ]

    gcj_lng, gcj_lat = wgs84_to_gcj02(lng, lat)
    return {
        "polygon": Polygon(corners),
        "origin": {"lng_wgs": lng, "lat_wgs": lat, "lng_gcj": gcj_lng, "lat_gcj": gcj_lat},
    }


def project_footprint_to_edges(footprint_poly: Polygon, origin: dict, G, candidate_edges: list, *,
                               buffer_m: float = 2.0) -> list[dict]:
    """把候选路网边折算到局部平面，按与 footprint buffer 的覆盖比例降序打分。

    每条候选取其 (u,v,key) 在 G 中的 WGS-84 geometry；缺失时回退候选快照的
    geometry_gcj（GCJ 逐点转 WGS）；两者都无则跳过该候选。重叠比例 =
    buffered.intersection(footprint).area / footprint.area，<=0 过滤。

    Returns:
        [{"u": int, "v": int, "k": int, "overlap_ratio": float, "road_name": str}, ...]
        按 overlap_ratio 降序（同值保持候选出现顺序）。
    """
    try:
        fp_area = float(footprint_poly.area)
    except Exception:
        return []
    if fp_area <= 0:
        return []

    lng0 = float(origin["lng_wgs"])
    lat0 = float(origin["lat_wgs"])
    east_scale = _M_PER_DEG * math.cos(math.radians(lat0))

    results = []
    for cand in candidate_edges or []:
        if not isinstance(cand, dict):
            continue
        try:
            u = int(cand.get("u"))
            v = int(cand.get("v"))
            k = int(cand.get("key", 0))
        except (TypeError, ValueError):
            continue

        edge_data = G.get_edge_data(u, v, k)
        line_wgs = _wgs_line_from_geometry(edge_data.get("geometry")) if edge_data else None
        if line_wgs is None:
            line_wgs = _wgs_line_from_gcj(cand.get("geometry_gcj"))
            if line_wgs is None:
                continue
        road_name = _road_name(edge_data, cand)

        coords = []
        for x, y in line_wgs.coords:
            coords.append(((x - lng0) * east_scale, (y - lat0) * _M_PER_DEG))
        if len(coords) < 2:
            continue
        try:
            buffered = LineString(coords).buffer(buffer_m)
            overlap = buffered.intersection(footprint_poly).area / fp_area
        except Exception:
            continue
        if overlap <= 0:
            continue
        results.append({
            "u": u,
            "v": v,
            "k": k,
            "overlap_ratio": round(float(overlap), 6),
            "road_name": road_name,
        })

    results.sort(key=lambda item: item["overlap_ratio"], reverse=True)
    return results
