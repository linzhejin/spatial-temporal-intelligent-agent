"""疑似受影响路段三级降级打分编排（spatial/vision_suggest.py）合成真值测试。

只用 pytest / networkx / shapely 与 spatial.vision_suggest 本身，不依赖 DB、不依赖
Flask；L2 用例用 pytest.importorskip("numpy") 门控（生产 api 进程可能无 numpy）。

坐标约定：anchor 为 GCJ-02；camera_pose 与合成路网为 WGS-84；合成真值按
coord_transform 真实函数换算。origin WGS = gcj02_to_wgs84(anchor)。
"""

import math

import networkx as nx
import pytest
from shapely.geometry import LineString

from spatial.coord_transform import gcj02_to_wgs84
from spatial.vision_suggest import score_suggested_edges

ANCHOR = {"lng": 114.363, "lat": 30.5365, "crs": "GCJ02"}
ORIGIN_LNG_WGS, ORIGIN_LAT_WGS = gcj02_to_wgs84(ANCHOR["lng"], ANCHOR["lat"])
_M_PER_DEG = 111320.0
_COS_LAT = math.cos(math.radians(ORIGIN_LAT_WGS))

BASE_FIELDS = {
    "u", "v", "key", "edges", "chain_length_m", "road_name",
    "snap_lng_gcj", "snap_lat_gcj", "dist_m", "geometry_gcj", "score", "basis",
}

_CENTER_MASK = [[[0.45, 0.45], [0.55, 0.45], [0.55, 0.55], [0.45, 0.55]]]


def _wgs_point(east, north):
    lng = ORIGIN_LNG_WGS + east / (_M_PER_DEG * _COS_LAT)
    lat = ORIGIN_LAT_WGS + north / _M_PER_DEG
    return (lng, lat)


def _h_edge(north, east_lo=-80.0, east_hi=80.0):
    return LineString([_wgs_point(east_lo, north), _wgs_point(east_hi, north)])


def _road_graph():
    """4 条独立双向边（key=0），north ∈ {0, 50, -50, 120}。"""
    G = nx.MultiDiGraph()

    def add(u, v, north, name):
        fwd = _h_edge(north)
        bwd = LineString(list(reversed(fwd.coords)))
        G.add_edge(u, v, key=0, geometry=fwd, name=name, length=160.0)
        G.add_edge(v, u, key=0, geometry=bwd, name=name, length=160.0)

    add(1, 2, 0.0, "正下方")
    add(3, 4, 50.0, "北中")
    add(5, 6, -50.0, "南中")
    add(7, 8, 120.0, "北远")
    return G


def _pose_point(seconds=None, east=0.0, north=0.0, alt=100.0, yaw=0.0, pitch=None):
    lng, lat = _wgs_point(east, north)
    point = {"lng": lng, "lat": lat, "rel_alt_m": alt, "gb_yaw_deg": yaw}
    if seconds is not None:
        point["seconds"] = seconds
    if pitch is not None:
        point["gb_pitch_deg"] = pitch
    return point


def _nadir_camera_pose():
    return {"trajectory": [_pose_point()]}


def _flood_candidate(outline_polygons, outline_time_seconds=0.0):
    return [{
        "kind": "possible_flooding",
        "confidence": 0.8,
        "reason": "test",
        "evidence": {
            "segments": [],
            "summary": {
                "outline_polygons": outline_polygons,
                "outline_time_seconds": outline_time_seconds,
            },
        },
    }]


# ---- 无姿态 → L0 半径圈 ----

def test_no_pose_falls_back_to_radius_only():
    G = _road_graph()
    result = score_suggested_edges(G, ANCHOR)
    assert result["basis"] == "radius_only"
    items = result["suggested_edges"]
    assert 0 < len(items) <= 5
    for item in items:
        assert item["score"] == 0.0
        assert item["basis"] == "radius_only"
        assert BASE_FIELDS <= set(item)
    # 默认 30m 半径内只有正下方边
    assert {item["road_name"] for item in items} == {"正下方"}


# ---- L1 footprint ----

def test_nadir_pose_without_mask_uses_footprint():
    G = _road_graph()
    result = score_suggested_edges(G, ANCHOR, camera_pose=_nadir_camera_pose())
    assert result["basis"] == "pose_footprint"
    items = result["suggested_edges"]
    assert 0 < len(items) <= 5
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True)
    assert all(item["score"] > 0 for item in items)
    names = {item["road_name"] for item in items}
    assert "正下方" in names
    assert "北远" not in names  # 120m 超出 footprint（h_h=56.25m）


def test_shallow_pitch_falls_back_to_radius_only():
    G = _road_graph()
    camera_pose = {"trajectory": [_pose_point(pitch=-3.0)]}
    result = score_suggested_edges(G, ANCHOR, camera_pose=camera_pose)
    assert result["basis"] == "radius_only"
    assert all(item["score"] == 0.0 for item in result["suggested_edges"])


# ---- L2 配准掩膜投影 ----

def test_mask_projection_branch():
    pytest.importorskip("numpy")
    import vision.registration as registration
    G = _road_graph()
    result = score_suggested_edges(
        G, ANCHOR, camera_pose=_nadir_camera_pose(),
        media={"width": 1000, "height": 1000},
        candidates=_flood_candidate(_CENTER_MASK),
        registration=registration,
    )
    assert result["basis"] == "mask_projection"
    items = result["suggested_edges"]
    assert 0 < len(items) <= 5
    names = {item["road_name"] for item in items}
    assert "正下方" in names
    assert "北中" not in names
    assert all(item["score"] > 0 for item in items)
    assert all(item["basis"] == "mask_projection" for item in items)


def test_mask_bad_outline_falls_back_to_footprint():
    pytest.importorskip("numpy")
    import vision.registration as registration
    G = _road_graph()
    # 只有 2 点 → 该条 outline 被跳过，L2 无命中 → 落到 L1
    candidates = _flood_candidate([[[0.45, 0.45], [0.55, 0.55]]])
    result = score_suggested_edges(
        G, ANCHOR, camera_pose=_nadir_camera_pose(),
        media={"width": 1000, "height": 1000},
        candidates=candidates, registration=registration,
    )
    assert result["basis"] == "pose_footprint"


def test_mask_non_list_outlines_falls_back_to_footprint():
    pytest.importorskip("numpy")
    import vision.registration as registration
    G = _road_graph()
    # outline_polygons 非 list → L2 前置条件不成立 → 直接 L1
    candidates = _flood_candidate("not-a-list")
    result = score_suggested_edges(
        G, ANCHOR, camera_pose=_nadir_camera_pose(),
        media={"width": 1000, "height": 1000},
        candidates=candidates, registration=registration,
    )
    assert result["basis"] == "pose_footprint"


def test_registration_none_skips_mask_directly_to_footprint():
    G = _road_graph()
    result = score_suggested_edges(
        G, ANCHOR, camera_pose=_nadir_camera_pose(),
        media={"width": 1000, "height": 1000},
        candidates=_flood_candidate(_CENTER_MASK),
        registration=None,
    )
    assert result["basis"] == "pose_footprint"
    assert all(item["basis"] == "pose_footprint" for item in result["suggested_edges"])


# ---- _pick_pose 时间匹配 ----

def test_mask_time_matching_selects_seconds_pose():
    pytest.importorskip("numpy")
    import vision.registration as registration
    G = _road_graph()
    pose_a = _pose_point(seconds=0.0, east=0.0, north=0.0)
    pose_b = _pose_point(seconds=10.0, east=0.0, north=50.0)
    result = score_suggested_edges(
        G, ANCHOR, camera_pose={"trajectory": [pose_a, pose_b]},
        media={"width": 1000, "height": 1000},
        candidates=_flood_candidate(_CENTER_MASK, outline_time_seconds=10.0),
        registration=registration,
    )
    assert result["basis"] == "mask_projection"
    names = {item["road_name"] for item in result["suggested_edges"]}
    assert "北中" in names
    assert "正下方" not in names


def test_mask_time_out_of_range_falls_back_to_nearest_pose():
    pytest.importorskip("numpy")
    import vision.registration as registration
    G = _road_graph()
    pose_a = _pose_point(seconds=0.0, east=0.0, north=0.0)
    pose_b = _pose_point(seconds=10.0, east=0.0, north=50.0)
    # 时间超区间 → 按 anchor 最近点（pose_a，origin）→ 命中正下方
    result = score_suggested_edges(
        G, ANCHOR, camera_pose={"trajectory": [pose_a, pose_b]},
        media={"width": 1000, "height": 1000},
        candidates=_flood_candidate(_CENTER_MASK, outline_time_seconds=100.0),
        registration=registration,
    )
    assert result["basis"] == "mask_projection"
    names = {item["road_name"] for item in result["suggested_edges"]}
    assert "正下方" in names
    assert "北中" not in names


# ---- 输出字段完整性 ----

def test_output_items_have_full_field_set():
    G = _road_graph()
    result = score_suggested_edges(G, ANCHOR, camera_pose=_nadir_camera_pose())
    assert result["basis"] == "pose_footprint"
    assert 0 < len(result["suggested_edges"]) <= 5
    for item in result["suggested_edges"]:
        assert BASE_FIELDS <= set(item)