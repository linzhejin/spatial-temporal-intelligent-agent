"""基于显式标定参数的拍摄范围打分，仅用于隔离验证。

管理端「疑似受影响路段」建议的三级降级：L2-mask / L1-footprint / L0-radius。
本模块是 L1-footprint 的纯几何实现：

  - pose_to_footprint：把相机位置、离地高度、姿态及像素内参换算成
    「相机拍摄地面范围」的多边形（局部米制平面，以 pose 位置为原点）。
  - project_footprint_to_edges：把候选路网边折算到同一局部平面，与 footprint
    的 buffer 相交，按覆盖比例打分排序。

只依赖 shapely + 标准库 + spatial.coord_transform，顶层禁止 import numpy /
vision.registration / Flask / storage。坐标约定：pose 与路网为 WGS-84，
GCJ-02 仅在输出 origin 与 geometry_gcj 回退的边界处转换。
"""

import math

from shapely.geometry import LineString, Polygon

from spatial.coord_transform import gcj02_to_wgs84
from spatial.camera_projection import camera_ground_homography, project_pixel

_M_PER_DEG = 111320.0


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


def pose_to_footprint(pose: dict, image_w_px: int, image_h_px: int) -> dict:
    """Project the calibrated image corners onto the local ground plane."""
    transform = camera_ground_homography(pose, image_w_px, image_h_px)
    matrix = transform["homography"]
    corners = [(0, 0), (image_w_px, 0),
               (image_w_px, image_h_px), (0, image_h_px)]
    polygon = Polygon([project_pixel(matrix, x, y) for x, y in corners])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("camera footprint is invalid")
    return {"polygon": polygon, "origin": transform["origin"]}


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
