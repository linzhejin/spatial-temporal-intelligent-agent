"""L2 影像配准器（vision/registration.py）合成真值测试。

只用 numpy / shapely / networkx 与 vision.registration 本身，不依赖 DB、
不依赖 Flask app。合成真值按下述约定手写 H_true，不调用被测函数求逆：

  - origin WGS (114.3623, 30.5363)；g = 0.05、θ = 15°、cx=cy=500（image 1000×1000）。
  - 局部平面：east = (lng_wgs - lng0) * 111320 * cos(lat0), north = (lat_wgs - lat0) * 111320。
  - H_true 把像素 → 局部平面（“近正下视初值”矩阵公式）。
"""

import math

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from spatial.coord_transform import gcj02_to_wgs84, wgs84_to_gcj02
from vision.registration import (
    fit_pixel_to_gcj,
    pose_to_homography_init,
    project_mask_to_edges,
)

ORIGIN_LNG_WGS = 114.3623
ORIGIN_LAT_WGS = 30.5363
G_M = 0.05
THETA_DEG = 15.0
IMAGE_W = 1000
IMAGE_H = 1000
_CX = IMAGE_W / 2.0
_CY = IMAGE_H / 2.0
_M_PER_DEG = 111320.0


def _homography_true(g=G_M, theta_deg=THETA_DEG, cx=_CX, cy=_CY):
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [g * c, -g * s, -g * (cx * c - cy * s)],
        [-g * s, -g * c, g * (cx * s + cy * c)],
        [0.0, 0.0, 1.0],
    ])


H_TRUE = _homography_true()
H_TRUE_INV = np.linalg.inv(H_TRUE)


def _apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _local_to_pixel(local_pts):
    return _apply_h(H_TRUE_INV, local_pts)


def _local_to_gcj(local_pts):
    """局部平面（全局原点）→ WGS → GCJ-02（构造阶段的世界点真值）。"""
    cos_lat = math.cos(math.radians(ORIGIN_LAT_WGS))
    out = []
    for east, north in local_pts:
        lng = ORIGIN_LNG_WGS + east / (_M_PER_DEG * cos_lat)
        lat = ORIGIN_LAT_WGS + north / _M_PER_DEG
        out.append(wgs84_to_gcj02(lng, lat))
    return np.array(out)


def _ground_truth_points():
    """3 条局部水平线（north ∈ {-20, 0, 20}，east ∈ [0, 80]），各 10 采样点。"""
    edges = []
    locals_list = []
    for north in (-20.0, 0.0, 20.0):
        edge = [(east, north) for east in np.linspace(0.0, 80.0, 10)]
        edges.append(edge)
        locals_list.extend(edge)
    locals_arr = np.array(locals_list)
    return locals_arr, _local_to_pixel(locals_arr), _local_to_gcj(locals_arr), edges


def _world_gcj_to_local(gcj):
    """把世界 GCJ → WGS → 局部平面（以第一点为原点，与 fit 内部一致）。"""
    origin_lng_wgs, origin_lat_wgs = gcj02_to_wgs84(*gcj[0])
    cos_lat = math.cos(math.radians(origin_lat_wgs))
    out = []
    for glng, glat in gcj:
        wlng, wlat = gcj02_to_wgs84(glng, glat)
        out.append((
            (wlng - origin_lng_wgs) * _M_PER_DEG * cos_lat,
            (wlat - origin_lat_wgs) * _M_PER_DEG,
        ))
    return np.array(out)


# ---- fit_pixel_to_gcj ----

def test_fit_pixel_to_gcj_noiseless_recovers_homography():
    _, pixel, gcj, _ = _ground_truth_points()
    result = fit_pixel_to_gcj(pixel, gcj)
    local_truth = _world_gcj_to_local(gcj)
    pred = _apply_h(result["homography"], pixel)
    assert np.max(np.abs(pred - local_truth)) < 1e-6
    assert result["residual_m"] < 1e-6
    assert result["inlier_count"] == len(pixel)


def test_fit_pixel_to_gcj_small_gaussian_noise():
    local, pixel, _, _ = _ground_truth_points()
    rng = np.random.default_rng(0)
    noisy_local = local + rng.normal(0.0, 0.3, size=local.shape)
    gcj_noisy = _local_to_gcj(noisy_local)
    result = fit_pixel_to_gcj(pixel, gcj_noisy)
    assert result["residual_m"] < 0.5
    assert result["inlier_count"] == len(pixel)


def test_fit_pixel_to_gcj_robust_to_wild_outliers():
    east = np.linspace(0.0, 80.0, 5)
    north = np.linspace(-20.0, 20.0, 4)
    local = np.array([(e, n) for n in north for e in east])
    rng = np.random.default_rng(0)
    noisy_local = local.copy()
    for i, idx in enumerate([2, 7, 13, 18]):
        sign = 1.0 if i % 2 == 0 else -1.0
        noisy_local[idx] = (local[idx][0] + sign * 20.0, local[idx][1])
    pixel = _local_to_pixel(local)
    gcj = _local_to_gcj(noisy_local)
    clean_gcj = _local_to_gcj(local)
    result = fit_pixel_to_gcj(pixel, gcj)
    assert result["residual_m"] < 0.5
    assert result["inlier_count"] >= 15
    # origin 由第一组 GCJ（干净点）确定，不被 outlier 抬走
    assert result["origin"]["lng_gcj"] == pytest.approx(clean_gcj[0][0], abs=1e-12)
    assert result["origin"]["lat_gcj"] == pytest.approx(clean_gcj[0][1], abs=1e-12)


def test_fit_pixel_to_gcj_seed_parameter_converges():
    _, pixel, gcj, _ = _ground_truth_points()
    seeded = fit_pixel_to_gcj(pixel, gcj, seed=0)
    unseeded = fit_pixel_to_gcj(pixel, gcj, seed=None)
    for result in (seeded, unseeded):
        assert result["residual_m"] < 1e-6
        assert result["inlier_count"] == len(pixel)


def test_fit_pixel_to_gcj_all_outliers_residual_none():
    local, pixel, _, _ = _ground_truth_points()
    rng = np.random.default_rng(7)
    displaced = local + rng.normal(0.0, 40.0, size=local.shape)
    gcj = _local_to_gcj(displaced)
    result = fit_pixel_to_gcj(
        pixel, gcj, min_inliers=len(local) + 5, inlier_thresh_m=1e-6,
    )
    assert result["inlier_count"] == 0
    assert result["residual_m"] is None


def test_coordinate_roundtrip_stable():
    _, pixel, gcj, _ = _ground_truth_points()
    result = fit_pixel_to_gcj(pixel, gcj)
    origin = result["origin"]
    origin_lng_wgs = origin["lng_wgs"]
    origin_lat_wgs = origin["lat_wgs"]
    cos_lat = math.cos(math.radians(origin_lat_wgs))
    pred = _apply_h(result["homography"], pixel)
    recovered = []
    for east, north in pred:
        lng_wgs = origin_lng_wgs + east / (_M_PER_DEG * cos_lat)
        lat_wgs = origin_lat_wgs + north / _M_PER_DEG
        recovered.append(wgs84_to_gcj02(lng_wgs, lat_wgs))
    recovered = np.array(recovered)
    assert np.max(np.abs(recovered - gcj)) < 1e-9


def test_fit_pixel_to_gcj_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_pixel_to_gcj([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                         [[114.0, 30.0], [114.0, 30.0], [114.0, 30.0]])
    with pytest.raises(ValueError):
        fit_pixel_to_gcj(np.zeros((5, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        # 4 对中 3 对含 NaN → 清洗后 < 4
        fit_pixel_to_gcj(
            [[0.0, 0.0], [np.nan, 0.0], [2.0, 2.0], [3.0, 3.0]],
            [[114.0, 30.0], [114.0, 30.0], [114.0, 30.0], [114.0, 30.0]],
        )


# ---- pose_to_homography_init ----

def test_pose_to_homography_init_matches_true_homography():
    f_px = 24.0 / 36.0 * IMAGE_W
    alt_m = G_M * f_px
    pose = {
        "lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS,
        "height_agl_m": alt_m, "gb_yaw_deg": THETA_DEG,
        "gb_pitch_deg": -90, "gb_roll_deg": 0,
        "intrinsics": {"fx_px": f_px, "fy_px": f_px, "cx_px": _CX, "cy_px": _CY},
    }
    result = pose_to_homography_init(
        pose, IMAGE_W, IMAGE_H,
    )
    assert np.allclose(result["homography"], H_TRUE, atol=1e-9)
    assert result["origin"]["lng_wgs"] == pytest.approx(ORIGIN_LNG_WGS)
    assert result["origin"]["lat_wgs"] == pytest.approx(ORIGIN_LAT_WGS)
    glng, glat = wgs84_to_gcj02(ORIGIN_LNG_WGS, ORIGIN_LAT_WGS)
    assert result["origin"]["lng_gcj"] == pytest.approx(glng)
    assert result["origin"]["lat_gcj"] == pytest.approx(glat)


def test_pose_to_homography_init_yaw_zero_anchor_points():
    f_px = 24.0 / 36.0 * IMAGE_W
    alt_m = 10.0
    pose = {
        "lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS,
        "height_agl_m": alt_m, "gb_yaw_deg": 0.0,
        "gb_pitch_deg": -90, "gb_roll_deg": 0,
        "intrinsics": {"fx_px": f_px, "fy_px": f_px, "cx_px": _CX, "cy_px": _CY},
    }
    result = pose_to_homography_init(
        pose, IMAGE_W, IMAGE_H,
    )
    H = result["homography"]
    g = alt_m / f_px
    center = H @ np.array([_CX, _CY, 1.0])
    right = H @ np.array([_CX + 100.0, _CY, 1.0])
    assert np.allclose(center[:2] / center[2], [0.0, 0.0], atol=1e-9)
    assert np.allclose(right[:2] / right[2], [100.0 * g, 0.0], atol=1e-9)


def test_pose_to_homography_init_rejects_invalid_input():
    with pytest.raises(ValueError):
        pose_to_homography_init({"lat": ORIGIN_LAT_WGS, "rel_alt_m": 10.0}, IMAGE_W, IMAGE_H)
    with pytest.raises(ValueError):
        pose_to_homography_init({"lng": ORIGIN_LNG_WGS, "rel_alt_m": 10.0}, IMAGE_W, IMAGE_H)
    with pytest.raises(ValueError):
        pose_to_homography_init({"lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS}, IMAGE_W, IMAGE_H)
    with pytest.raises(ValueError):
        pose_to_homography_init(
            {"lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS, "abs_alt_m": 0.0},
            IMAGE_W, IMAGE_H,
        )
    with pytest.raises(ValueError):
        pose_to_homography_init(
            {"lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS, "rel_alt_m": 10.0},
            0, IMAGE_H,
        )


# ---- project_mask_to_edges ----

def _edge_graph():
    """3 条双向边（key=0），geometry 为 WGS LineString，north ∈ {-20, 0, 20}。"""
    G = nx.MultiDiGraph()
    cos_lat = math.cos(math.radians(ORIGIN_LAT_WGS))
    for idx, north in enumerate([-20.0, 0.0, 20.0]):
        u, v = 10 * idx + 1, 10 * idx + 2
        coords = []
        for east in (0.0, 80.0):
            lng = ORIGIN_LNG_WGS + east / (_M_PER_DEG * cos_lat)
            lat = ORIGIN_LAT_WGS + north / _M_PER_DEG
            coords.append((lng, lat))
        G.add_edge(u, v, key=0, geometry=LineString(coords), name=f"road{idx}")
        G.add_edge(v, u, key=0, geometry=LineString(list(reversed(coords))),
                   name=f"road{idx}")
    return G


def _transform():
    f_px = 24.0 / 36.0 * IMAGE_W
    return pose_to_homography_init(
        {
            "lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS,
            "height_agl_m": G_M * f_px, "gb_yaw_deg": THETA_DEG,
            "gb_pitch_deg": -90, "gb_roll_deg": 0,
            "intrinsics": {"fx_px": f_px, "fy_px": f_px, "cx_px": _CX, "cy_px": _CY},
        },
        IMAGE_W, IMAGE_H,
    )


def _center_chunk_mask_px():
    """覆盖边 1（north=0）center 区域 buffer(2.0) 的多边形（转像素）。"""
    chunk = LineString([(30.0, 0.0), (50.0, 0.0)]).buffer(2.0)
    local_coords = np.array(chunk.exterior.coords)
    return Polygon(_local_to_pixel(local_coords))


def test_project_mask_to_edges_ranks_hit_edge():
    G = _edge_graph()
    candidates = [
        {"u": 1, "v": 2, "k": 0},
        {"u": 11, "v": 12, "k": 0},
        {"u": 21, "v": 22, "k": 0},
    ]
    results = project_mask_to_edges(
        _center_chunk_mask_px(), _transform(), G, candidates,
        image_w_px=IMAGE_W, image_h_px=IMAGE_H, buffer_m=2.0,
    )
    assert len(results) == 3
    assert results[0]["u"] == 11 and results[0]["v"] == 12
    assert 0.0 < results[0]["overlap_ratio"] <= 1.0
    assert results[0]["road_name"] == "road1"
    ratios = [item["overlap_ratio"] for item in results]
    assert ratios == sorted(ratios, reverse=True)


def test_project_mask_to_edges_empty_candidates():
    G = _edge_graph()
    results = project_mask_to_edges(
        Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        _transform(), G, [],
        image_w_px=IMAGE_W, image_h_px=IMAGE_H,
    )
    assert results == []


def test_project_mask_to_edges_skips_candidates_without_geometry():
    G = _edge_graph()
    candidates = [
        {"u": 11, "v": 12, "k": 0},
        {"u": 999, "v": 998, "k": 0},  # G 无此边，且无 geometry_gcj → 跳过
    ]
    results = project_mask_to_edges(
        _center_chunk_mask_px(), _transform(), G, candidates,
        image_w_px=IMAGE_W, image_h_px=IMAGE_H, buffer_m=2.0,
    )
    assert len(results) == 1
    assert results[0]["u"] == 11 and results[0]["v"] == 12


def test_project_mask_to_edges_invalid_mask_returns_empty():
    G = _edge_graph()
    candidates = [{"u": 11, "v": 12, "k": 0}]
    # 非法（自交）多边形 → 返回空列表，不抛
    invalid = Polygon([(0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)])
    results = project_mask_to_edges(
        invalid, _transform(), G, candidates,
        image_w_px=IMAGE_W, image_h_px=IMAGE_H,
    )
    assert results == []


def _mask_px_from_local(shell_local, hole_local=None):
    shell_px = _local_to_pixel(np.asarray(shell_local, dtype=np.float64))
    if hole_local is None:
        return Polygon([tuple(p) for p in shell_px])
    hole_px = _local_to_pixel(np.asarray(hole_local, dtype=np.float64))
    return Polygon([tuple(p) for p in shell_px], [[tuple(p) for p in hole_px]])


def test_project_mask_to_edges_hole_reduces_center_edge_overlap():
    G = _edge_graph()
    candidates = [
        {"u": 1, "v": 2, "k": 0},
        {"u": 11, "v": 12, "k": 0},
        {"u": 21, "v": 22, "k": 0},
    ]
    shell_local = [(20.0, -8.0), (60.0, -8.0), (60.0, 8.0), (20.0, 8.0)]
    solid = _mask_px_from_local(shell_local)
    solid_ratios = {
        r["u"]: r["overlap_ratio"]
        for r in project_mask_to_edges(
            solid, _transform(), G, candidates,
            image_w_px=IMAGE_W, image_h_px=IMAGE_H, buffer_m=2.0,
        )
    }
    assert solid_ratios[11] > 0

    hole_local = [(28.0, -3.0), (52.0, -3.0), (52.0, 3.0), (28.0, 3.0)]
    holed = _mask_px_from_local(shell_local, hole_local)
    holed_ratios = {
        r["u"]: r["overlap_ratio"]
        for r in project_mask_to_edges(
            holed, _transform(), G, candidates,
            image_w_px=IMAGE_W, image_h_px=IMAGE_H, buffer_m=2.0,
        )
    }
    # 挖洞后中心边仍被报告，但 overlap 显著下降
    assert holed_ratios[11] > 0
    assert holed_ratios[11] < solid_ratios[11]
