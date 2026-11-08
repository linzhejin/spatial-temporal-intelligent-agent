"""L1 姿态拍摄范围打分器（spatial/vision_footprint.py）合成真值测试。

只用 pytest / shapely / networkx 与 spatial.vision_footprint 本身，不依赖 DB、不依赖 Flask、
不 import numpy。合成真值按下述约定手写：

  - origin WGS (114.3623, 30.5363)。
  - 局部平面：east = (lng_wgs - lng0) * 111320 * cos(lat0), north = (lat_wgs - lat0) * 111320。
  - nadir（yaw=0、缺 gb_pitch）image 1000x500 focal 24：f_px = 24/36*1000，
    h_w = 100*tan(atan(500/f_px)) = 75 m，h_h = 100*tan(atan(250/f_px)) = 37.5 m。
"""

import math

import networkx as nx
import pytest
from shapely.geometry import LineString

from spatial.coord_transform import wgs84_to_gcj02
from spatial.vision_footprint import pose_to_footprint, project_footprint_to_edges

ORIGIN_LNG_WGS = 114.3623
ORIGIN_LAT_WGS = 30.5363
_M_PER_DEG = 111320.0
_COS_LAT = math.cos(math.radians(ORIGIN_LAT_WGS))


def _wgs_point(east, north):
    lng = ORIGIN_LNG_WGS + east / (_M_PER_DEG * _COS_LAT)
    lat = ORIGIN_LAT_WGS + north / _M_PER_DEG
    return (lng, lat)


def _add_edge(G, u, v, north, name, east_lo=-40.0, east_hi=40.0):
    forward = LineString([_wgs_point(east_lo, north), _wgs_point(east_hi, north)])
    backward = LineString([_wgs_point(east_hi, north), _wgs_point(east_lo, north)])
    G.add_edge(u, v, key=0, geometry=forward, name=name)
    G.add_edge(v, u, key=0, geometry=backward, name=name)


def _candidate(u, v, name, geometry_gcj=None):
    return {"u": u, "v": v, "key": 0, "road_name": name, "geometry_gcj": geometry_gcj or []}


def _nadir_pose(**overrides):
    pose = {
        "lng": ORIGIN_LNG_WGS,
        "lat": ORIGIN_LAT_WGS,
        "rel_alt_m": 100.0,
        "yaw_deg": 0.0,
    }
    pose.update(overrides)
    return pose


def _nadir_half_sizes(alt=100.0, image_w=1000, image_h=500, focal=24.0, sensor_w=36.0):
    f_px = focal / sensor_w * image_w
    hfov_half = math.atan((image_w / 2.0) / f_px)
    vfov_half = math.atan((image_h / 2.0) / f_px)
    return alt * math.tan(hfov_half), alt * math.tan(vfov_half)


# ---- pose_to_footprint ----

def test_nadir_centers_on_origin_and_covers_center_edge():
    G = nx.MultiDiGraph()
    _add_edge(G, 1, 2, 0.0, "正下方")
    _add_edge(G, 3, 4, 50.0, "北远")
    _add_edge(G, 5, 6, -50.0, "南远")

    result = pose_to_footprint(_nadir_pose(), 1000, 500)
    poly = result["polygon"]

    # offset=0 → footprint 中心 = origin（局部平面原点）
    assert poly.centroid.x == pytest.approx(0.0, abs=1e-9)
    assert poly.centroid.y == pytest.approx(0.0, abs=1e-9)

    # origin 结构
    assert result["origin"]["lng_wgs"] == ORIGIN_LNG_WGS
    assert result["origin"]["lat_wgs"] == ORIGIN_LAT_WGS
    gcj = wgs84_to_gcj02(ORIGIN_LNG_WGS, ORIGIN_LAT_WGS)
    assert result["origin"]["lng_gcj"] == pytest.approx(gcj[0])
    assert result["origin"]["lat_gcj"] == pytest.approx(gcj[1])

    # 四角面积与 (2*h_w)*(2*h_h) 一致
    h_w, h_h = _nadir_half_sizes()
    assert poly.area == pytest.approx((2 * h_w) * (2 * h_h), rel=1e-6)

    candidates = [_candidate(1, 2, "正下方"), _candidate(3, 4, "北远"), _candidate(5, 6, "南远")]
    hits = project_footprint_to_edges(poly, result["origin"], G, candidates)
    names = [r["road_name"] for r in hits]
    assert "正下方" in names
    assert "北远" not in names
    assert "南远" not in names
    assert all(r["overlap_ratio"] > 0 for r in hits)


def test_yaw_90_rotates_corners():
    h_w, h_h = _nadir_half_sizes()
    result = pose_to_footprint(_nadir_pose(yaw_deg=90.0), 1000, 500)
    coords = list(result["polygon"].exterior.coords)[:-1]
    expected = [(h_h, -h_w), (h_h, h_w), (-h_h, h_w), (-h_h, -h_w)]
    assert len(coords) == 4
    for (ex, ey), (ax, ay) in zip(coords, expected):
        assert ex == pytest.approx(ax, abs=1e-6)
        assert ey == pytest.approx(ay, abs=1e-6)


def test_pitch_forward_offsets_center_and_changes_coverage():
    G = nx.MultiDiGraph()
    _add_edge(G, 1, 2, 100.0, "投影点附近")
    _add_edge(G, 3, 4, 0.0, "正下方")

    # gb_pitch = -45 → tilt = 45 → offset = alt * tan(45°) ≈ alt，机头朝北（yaw=0）
    result = pose_to_footprint(_nadir_pose(gb_pitch_deg=-45.0), 1000, 500)
    poly = result["polygon"]
    assert poly.centroid.x == pytest.approx(0.0, abs=1e-9)
    assert poly.centroid.y == pytest.approx(100.0, abs=1e-6)

    candidates = [_candidate(1, 2, "投影点附近"), _candidate(3, 4, "正下方")]
    hits = project_footprint_to_edges(poly, result["origin"], G, candidates)
    names = [r["road_name"] for r in hits]
    assert "投影点附近" in names
    assert "正下方" not in names


def test_altitude_missing_or_invalid_raises():
    with pytest.raises(ValueError, match="altitude"):
        pose_to_footprint({"lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS}, 1000, 500)
    with pytest.raises(ValueError, match="altitude"):
        pose_to_footprint({"lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS, "rel_alt_m": 0.0}, 1000, 500)
    with pytest.raises(ValueError, match="altitude"):
        pose_to_footprint({"lng": ORIGIN_LNG_WGS, "lat": ORIGIN_LAT_WGS, "rel_alt_m": -5.0}, 1000, 500)


def test_pitch_too_shallow_raises():
    with pytest.raises(ValueError, match="shallow"):
        pose_to_footprint(_nadir_pose(gb_pitch_deg=-3.0), 1000, 500)
    with pytest.raises(ValueError, match="shallow"):
        pose_to_footprint(_nadir_pose(gb_pitch_deg=3.0), 1000, 500)


def test_invalid_image_dimensions_raise():
    with pytest.raises(ValueError):
        pose_to_footprint(_nadir_pose(), 0, 500)
    with pytest.raises(ValueError):
        pose_to_footprint(_nadir_pose(), 1000, -1)


# ---- project_footprint_to_edges ----

def test_project_empty_candidates_returns_empty():
    G = nx.MultiDiGraph()
    result = pose_to_footprint(_nadir_pose(), 1000, 500)
    assert project_footprint_to_edges(result["polygon"], result["origin"], G, []) == []


def test_project_skips_candidates_without_geometry():
    G = nx.MultiDiGraph()
    _add_edge(G, 1, 2, 0.0, "中心")
    result = pose_to_footprint(_nadir_pose(), 1000, 500)

    candidates = [
        _candidate(1, 2, "中心"),
        _candidate(999, 998, "孤儿"),  # 不在 G 中，且无 geometry_gcj
    ]
    hits = project_footprint_to_edges(result["polygon"], result["origin"], G, candidates)
    names = [r["road_name"] for r in hits]
    assert names == ["中心"]
    assert all(r["overlap_ratio"] > 0 for r in hits)


def test_project_sorts_descending_and_buffer_covers_touching_edge():
    G = nx.MultiDiGraph()
    _add_edge(G, 1, 2, 0.0, "中心")
    _add_edge(G, 3, 4, 37.5, "相触")  # 恰好落在 footprint 北边界，靠 buffer 才 overlap>0
    _add_edge(G, 5, 6, 50.0, "远")
    result = pose_to_footprint(_nadir_pose(), 1000, 500)

    candidates = [_candidate(3, 4, "相触"), _candidate(1, 2, "中心"), _candidate(5, 6, "远")]
    hits = project_footprint_to_edges(result["polygon"], result["origin"], G, candidates)
    names = [r["road_name"] for r in hits]
    assert names == ["中心", "相触"]
    assert hits[0]["overlap_ratio"] > hits[1]["overlap_ratio"] > 0
    # 显式校验返回结构与键
    for r in hits:
        assert set(r) == {"u", "v", "k", "overlap_ratio", "road_name"}
        assert isinstance(r["u"], int) and isinstance(r["v"], int) and isinstance(r["k"], int)


def test_project_falls_back_to_gcj_geometry():
    G = nx.MultiDiGraph()
    G.add_edge(10, 11, key=0)
    G.add_edge(11, 10, key=0)
    result = pose_to_footprint(_nadir_pose(), 1000, 500)

    gcj = [list(wgs84_to_gcj02(*_wgs_point(east, 0.0))) for east in (-40.0, 40.0)]
    candidates = [{"u": 10, "v": 11, "key": 0, "road_name": "GCJ边", "geometry_gcj": gcj}]
    hits = project_footprint_to_edges(result["polygon"], result["origin"], G, candidates)
    names = [r["road_name"] for r in hits]
    assert names == ["GCJ边"]
    assert hits[0]["overlap_ratio"] > 0