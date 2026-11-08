"""L2 影像配准原语：像素 ↔ 地面（GCJ-02）的单应配准与 mask→边重叠打分。

实现论文创新点「自有路网作为影像配准绝对基准」的离线管线底子：
  1. pose_to_homography_init   —— 相机姿态 → 近正下视单应初值；
  2. fit_pixel_to_gcj           —— 像素 ↔ GCJ 对应点的 RANSAC + DLT 单应配准；
  3. project_mask_to_edges      —— mask → 候选边重叠打分（返回边 ID 排序）。

仅用于论文实验 / 离线评测，不 import 进 worker/api，不自动落边、不接生产链路。
"""

from __future__ import annotations

import math

import numpy as np
from shapely import wkt
from shapely.geometry import LineString, Polygon

from spatial.coord_transform import gcj02_to_wgs84, wgs84_to_gcj02

_M_PER_DEG = 111320.0


def pose_to_homography_init(pose: dict, image_w_px: int, image_h_px: int, *,
                            focal_len_35mm: float | None = None,
                            sensor_w_mm: float = 36.0) -> dict:
    """由相机姿态构造近正下视单应初值（像素 → 局部平面）。

    Args:
        pose: 含 lng / lat（WGS-84）与可用高度（rel_alt_m 优先，其次 abs_alt_m）、
              可选 yaw（gb_yaw_deg 优先，其次 yaw_deg）与 focal_len_35mm。
        image_w_px / image_h_px: 图像像素宽高。
        focal_len_35mm: 35mm 等效焦距；优先于 pose["focal_len_35mm"]，两者都缺省用 24.0。
        sensor_w_mm: 35mm 等效传感器宽，默认 36.0。

    Returns:
        {"homography": (3,3) ndarray, "origin": {lng_wgs, lat_wgs, lng_gcj, lat_gcj}}
    """
    if image_w_px <= 0 or image_h_px <= 0:
        raise ValueError("image dimensions must be positive")
    if sensor_w_mm <= 0:
        raise ValueError("sensor width must be positive")

    lng = pose.get("lng")
    lat = pose.get("lat")
    if lng is None or lat is None:
        raise ValueError("pose must contain lng and lat (WGS-84)")

    focal = focal_len_35mm
    if focal is None:
        focal = pose.get("focal_len_35mm")
    if not focal:
        focal = 24.0

    alt_m = pose.get("rel_alt_m")
    if alt_m is None:
        alt_m = pose.get("abs_alt_m")
    if alt_m is None:
        raise ValueError("pose must contain a usable altitude (rel_alt_m or abs_alt_m)")
    alt_m = float(alt_m)
    if alt_m <= 0:
        raise ValueError("altitude must be positive")

    f_px = float(focal) / float(sensor_w_mm) * float(image_w_px)
    g = alt_m / f_px

    yaw = pose.get("gb_yaw_deg")
    if yaw is None:
        yaw = pose.get("yaw_deg")
    if yaw is None:
        yaw = 0.0
    theta = math.radians(float(yaw))

    cx = float(image_w_px) / 2.0
    cy = float(image_h_px) / 2.0
    c, s = math.cos(theta), math.sin(theta)

    homography = np.array([
        [g * c, -g * s, -g * (cx * c - cy * s)],
        [-g * s, -g * c, g * (cx * s + cy * c)],
        [0.0, 0.0, 1.0],
    ])

    lng_gcj, lat_gcj = wgs84_to_gcj02(float(lng), float(lat))
    return {
        "homography": homography,
        "origin": {
            "lng_wgs": float(lng),
            "lat_wgs": float(lat),
            "lng_gcj": lng_gcj,
            "lat_gcj": lat_gcj,
        },
    }


def _apply_h(H, pts) -> np.ndarray:
    """齐次坐标乘 H，返回非齐次 (N,2)。"""
    pts = np.asarray(pts, dtype=np.float64)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = hom @ np.asarray(H, dtype=np.float64).T
    # 全投影单应下某些点落在“无穷远线”上时 w≈0，产生 nan 被内点阈值自然剔除。
    with np.errstate(divide="ignore", invalid="ignore"):
        return proj[:, :2] / proj[:, 2:3]


def _normalization_matrix(pts) -> np.ndarray:
    """点集中心化 + 缩放的归一化变换（防病态）。"""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    mean_dist = np.linalg.norm(centered, axis=1).mean()
    scale = math.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])


def _solve_homography(src, dst) -> np.ndarray:
    """归一化 DLT 最小二乘：src（像素）→ dst（局部平面）。"""
    t_src = _normalization_matrix(src)
    t_dst = _normalization_matrix(dst)
    src_n = _apply_h(t_src, src)
    dst_n = _apply_h(t_dst, dst)

    a = np.zeros((2 * len(src), 9), dtype=np.float64)
    for i, ((x, y), (X, Y)) in enumerate(zip(src_n, dst_n)):
        a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, X * x, X * y, X]
        a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, Y * x, Y * y, Y]

    _, _, vt = np.linalg.svd(a)
    h_norm = vt[-1].reshape(3, 3)
    return np.linalg.inv(t_dst) @ h_norm @ t_src


def _gcj_to_local(gcj_points, origin_lng_wgs, origin_lat_wgs) -> np.ndarray:
    """世界 GCJ → WGS → 局部平面（以给定 WGS 原点为局部原点）。"""
    cos_lat = math.cos(math.radians(origin_lat_wgs))
    out = []
    for glng, glat in gcj_points:
        wlng, wlat = gcj02_to_wgs84(float(glng), float(glat))
        out.append((
            (wlng - origin_lng_wgs) * _M_PER_DEG * cos_lat,
            (wlat - origin_lat_wgs) * _M_PER_DEG,
        ))
    return np.asarray(out, dtype=np.float64)


def fit_pixel_to_gcj(pixel_pts, world_pts_gcj, *,
                     inlier_thresh_m: float = 3.0,
                     min_inliers: int = 4, iters: int = 200) -> dict:
    """由像素 ↔ 世界 GCJ 对应点估计单应（像素 → 局部平面），RANSAC 抗外点。

    Returns:
        {"homography", "residual_m", "inlier_count", "origin": {...}}
        origin 为第一组 GCJ 点对应的 WGS 原点。
    """
    pixel = np.asarray(pixel_pts, dtype=np.float64)
    world = np.asarray(world_pts_gcj, dtype=np.float64)
    if pixel.ndim != 2 or pixel.shape[1] != 2 or world.ndim != 2 or world.shape[1] != 2:
        raise ValueError("pixel_pts and world_pts_gcj must be (N, 2) arrays")
    if pixel.shape[0] != world.shape[0]:
        raise ValueError("pixel_pts and world_pts_gcj must have equal length")
    if pixel.shape[0] < 4:
        raise ValueError("at least 4 correspondence points are required")

    finite = np.isfinite(pixel).all(axis=1) & np.isfinite(world).all(axis=1)
    pixel = pixel[finite]
    world = world[finite]
    if pixel.shape[0] < 4:
        raise ValueError("at least 4 finite correspondence points are required")

    origin_lng_gcj, origin_lat_gcj = float(world[0][0]), float(world[0][1])
    origin_lng_wgs, origin_lat_wgs = gcj02_to_wgs84(origin_lng_gcj, origin_lat_gcj)
    local = _gcj_to_local(world, origin_lng_wgs, origin_lat_wgs)

    needed = max(min_inliers, 4)
    rng = np.random.default_rng(0)

    best_indices = None
    best_count = 0
    n = pixel.shape[0]
    if n > 4:
        for _ in range(int(iters)):
            sample = rng.choice(n, size=4, replace=False)
            h_candidate = _solve_homography(pixel[sample], local[sample])
            pred = _apply_h(h_candidate, pixel)
            dist = np.linalg.norm(pred - local, axis=1)
            inliers = np.where(dist <= inlier_thresh_m)[0]
            count = int(inliers.size)
            if count >= needed and count > best_count:
                best_count = count
                best_indices = inliers

    if best_indices is not None:
        homography = _solve_homography(pixel[best_indices], local[best_indices])
        pred = _apply_h(homography, pixel)
        dist = np.linalg.norm(pred - local, axis=1)
        inlier_mask = dist <= inlier_thresh_m
    else:
        homography = _solve_homography(pixel, local)
        pred = _apply_h(homography, pixel)
        dist = np.linalg.norm(pred - local, axis=1)
        inlier_mask = dist <= inlier_thresh_m

    inlier_count = int(inlier_mask.sum())
    residual_m = float(np.sqrt(np.mean(dist[inlier_mask] ** 2))) if inlier_count else 0.0

    return {
        "homography": homography,
        "residual_m": residual_m,
        "inlier_count": inlier_count,
        "origin": {
            "lng_wgs": origin_lng_wgs,
            "lat_wgs": origin_lat_wgs,
            "lng_gcj": origin_lng_gcj,
            "lat_gcj": origin_lat_gcj,
        },
    }


def _polygon_to_local(poly, H) -> Polygon:
    """像素 Polygon → 局部平面 Polygon（顶点齐次乘 H）。"""
    coords = []
    for x, y in poly.exterior.coords:
        projected = H @ np.array([x, y, 1.0])
        coords.append((projected[0] / projected[2], projected[1] / projected[2]))
    return Polygon(coords)


def _coerce_linestring(geom):
    """边数据里的 geometry → WGS LineString（支持 shapely 对象 / WKT 字符串）。"""
    if geom is None:
        return None
    if hasattr(geom, "coords"):
        coords = list(geom.coords)
        if len(coords) >= 2:
            return LineString(coords)
        return None
    if isinstance(geom, str) and geom.strip():
        try:
            candidate = wkt.loads(geom)
        except Exception:
            return None
        if isinstance(candidate, LineString) and len(candidate.coords) >= 2:
            return candidate
    return None


def _gcj_linestring_to_wgs(polyline):
    """GCJ 折线 [[lng,lat],...] → WGS LineString。"""
    if not isinstance(polyline, (list, tuple)) or len(polyline) < 2:
        return None
    coords = []
    for point in polyline:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        coords.append(gcj02_to_wgs84(float(point[0]), float(point[1])))
    try:
        return LineString(coords)
    except Exception:
        return None


def _wgs_linestring_to_local(line_wgs, origin_lng_wgs, origin_lat_wgs) -> LineString:
    """WGS LineString → 局部平面折线。"""
    cos_lat = math.cos(math.radians(origin_lat_wgs))
    coords = []
    for lng, lat in line_wgs.coords:
        coords.append((
            (lng - origin_lng_wgs) * _M_PER_DEG * cos_lat,
            (lat - origin_lat_wgs) * _M_PER_DEG,
        ))
    return LineString(coords)


def _road_name_of(value) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)) and value:
        return str(value[0]).strip()
    return ""


def project_mask_to_edges(mask_poly_px, transform: dict, G, candidate_edges: list[dict], *,
                          image_w_px: int, image_h_px: int,
                          buffer_m: float = 2.0) -> list[dict]:
    """把像素 mask 投影到局部平面，与候选边 buffered 几何重叠打分并降序返回。

    Args:
        mask_poly_px: shapely Polygon，像素坐标。
        transform: pose_to_homography_init / fit_pixel_to_gcj 的返回（homography + origin）。
        G: networkx MultiDiGraph（边 data 带 WGS geometry）。
        candidate_edges: [{"u","v","k"/"key","road_name","geometry_gcj"}, ...]。
        image_w_px / image_h_px: 图像尺寸（签名保留，本算法不裁剪）。
        buffer_m: 边两侧缓冲宽度（米）。

    Returns:
        [{"u","v","k","overlap_ratio","road_name"}, ...]，按 overlap_ratio 降序。
    """
    homography = np.asarray(transform.get("homography"), dtype=np.float64)
    origin = transform.get("origin") or {}
    origin_lng_wgs = float(origin.get("lng_wgs"))
    origin_lat_wgs = float(origin.get("lat_wgs"))

    try:
        mask_local = _polygon_to_local(mask_poly_px, homography)
    except Exception:
        return []
    if mask_local.is_empty or not mask_local.is_valid:
        return []
    try:
        mask_area = float(mask_local.area)
    except Exception:
        return []
    if not math.isfinite(mask_area):
        return []

    results = []
    for candidate in candidate_edges:
        if not isinstance(candidate, dict):
            continue
        u = candidate.get("u")
        v = candidate.get("v")
        k = candidate.get("k", candidate.get("key", 0))
        if u is None or v is None:
            continue

        edge_data = None
        try:
            edge_data = G.get_edge_data(u, v, k)
        except Exception:
            edge_data = None

        line_wgs = None
        road_name = ""
        if edge_data:
            line_wgs = _coerce_linestring(edge_data.get("geometry"))
            road_name = _road_name_of(edge_data.get("name"))
        if line_wgs is None:
            line_wgs = _gcj_linestring_to_wgs(candidate.get("geometry_gcj"))
            if not road_name:
                road_name = _road_name_of(candidate.get("road_name"))
        if line_wgs is None:
            continue

        try:
            line_local = _wgs_linestring_to_local(
                line_wgs, origin_lng_wgs, origin_lat_wgs,
            )
            overlap_area = line_local.buffer(buffer_m).intersection(mask_local).area
        except Exception:
            continue

        overlap_ratio = float(overlap_area) / mask_area if mask_area > 0 else 0.0
        results.append({
            "u": int(u),
            "v": int(v),
            "k": int(k),
            "overlap_ratio": overlap_ratio,
            "road_name": road_name,
        })

    results.sort(key=lambda item: item["overlap_ratio"], reverse=True)
    return results