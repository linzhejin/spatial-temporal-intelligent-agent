"""
特殊路况模块测试（边绑定模型）

覆盖：
  - snap_to_edge：GCJ-02 点击点吸附到最近边、距离阈值、WKT geometry
  - 边绑定双向生效 + 路网 id 失效几何回退 + 旧 radius 圆模型兼容
  - CONDITION_EFFECTS 影响矩阵：5 类事件 × walk/bike/drive 的硬封/软惩罚
  - 时间窗（未开始/已结束不生效）
  - HTTP：snap/POST/PATCH/DELETE、session 与 Token 双通道鉴权
"""

import math
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import pytest

from spatial import road_conditions as rc
from spatial.coord_transform import wgs84_to_gcj02, gcj02_to_wgs84


# ===================== 合成路网 =====================
#
#   4 ────── 5 ─────── 6 ────── 7     北绕路（40/80/40m）
#   │         │         │         │
#   0 ────── 1 ─────── 2 ────── 3     南主路（40/80/40m）
#         40m      80m      40m
#            ╱           ╲
#           8             9          死胡同（10m，让 1/2 成为度=3 路口，
#                                      边链扩展在此停止；不构成绕行捷径）
#
# 竖边 0-4、3-7 约 44m。主路总长 160m，绕行约 249m。
# 对中间 80m 边(1,2)：新软惩罚 ≤1.5×，主路成本 160+80×1.5=280 仍 < 绕行 249？
# 实测约 40+80×1.5+40=200 远小于绕行 248，所以软惩罚下仍走主路（虚线=可通行）。
# 只有硬封(closure)才导致绕行。

def _haversine_m(lng1, lat1, lng2, lat2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _make_graph():
    G = nx.MultiDiGraph()
    pts = {
        0: (114.36000, 30.53000),
        1: (114.36042, 30.53000),
        2: (114.36125, 30.53000),
        3: (114.36167, 30.53000),
        4: (114.36000, 30.53040),
        5: (114.36042, 30.53040),
        6: (114.36125, 30.53040),
        7: (114.36167, 30.53040),
        # 死胡同叶子：在节点 1、2 正南 10m
        8: (114.36042, 30.52991),
        9: (114.36125, 30.52991),
    }
    for nid, (x, y) in pts.items():
        G.add_node(nid, x=x, y=y)
    edges = [
        (0, 1, "自强大道"), (1, 2, "自强大道"), (2, 3, "自强大道"),
        (4, 5, "北环路"), (5, 6, "北环路"), (6, 7, "北环路"),
        (0, 4, ""), (3, 7, ""),
        (1, 8, ""), (2, 9, ""),
    ]
    for u, v, name in edges:
        length = _haversine_m(pts[u][0], pts[u][1], pts[v][0], pts[v][1])
        attrs = {"length": length, "highway": "residential"}
        if name:
            attrs["name"] = name
        G.add_edge(u, v, 0, **attrs)
        G.add_edge(v, u, 0, **attrs)
    return G


@pytest.fixture
def G():
    return _make_graph()


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    """每个测试独立的路况 JSON，不污染真实 data/road_conditions.json。"""
    path = tmp_path / "road_conditions.json"
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    yield


def test_routing_snapshot_rejects_corrupt_road_condition_store(tmp_path, monkeypatch):
    path = tmp_path / "road_conditions.json"
    path.write_text("{ invalid json", encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)

    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)


def test_lenient_corrupt_read_does_not_poison_strict_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "road_conditions.json"
    path.write_text("{ invalid json", encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)

    assert rc.list_conditions() == []
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)

    corrupt_stat = path.stat()
    path.write_text("[]", encoding="utf-8")
    os.utime(path, ns=(corrupt_stat.st_atime_ns, corrupt_stat.st_mtime_ns))
    assert rc.list_conditions(strict=True) == []


def test_configured_runtime_event_store_fails_closed_if_missing(tmp_path, monkeypatch):
    missing = tmp_path / "runtime" / "road_conditions.json"
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(missing))
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(missing))
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)


def test_reading_road_conditions_returns_detached_snapshot(G):
    _edge_condition(G, "closure")

    snapshot = rc.list_conditions(strict=True)
    snapshot[0]["type_label"] = "UI-only"
    snapshot[0]["edge"]["u"] = 999

    fresh = rc.list_conditions(strict=True)
    assert "type_label" not in fresh[0]
    assert fresh[0]["edge"]["u"] != 999


def test_strict_snapshot_rejects_event_without_a_supported_binding(tmp_path, monkeypatch):
    path = tmp_path / "road_conditions.json"
    path.write_text('[{"id":"bad","type":"closure","start_time":0,"end_time":0}]', encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)

    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)


def test_strict_snapshot_validates_events_loaded_into_non_strict_cache(tmp_path, monkeypatch):
    path = tmp_path / "road_conditions.json"
    path.write_text('[{"id":"bad","type":"closure","start_time":0,"end_time":0}]', encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)

    assert len(rc.list_conditions()) == 1
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)


def test_add_condition_preserves_corrupt_store_instead_of_overwriting(tmp_path, monkeypatch, G):
    path = tmp_path / "road_conditions.json"
    original = "{ invalid json"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    lng, lat = _mid_12_gcj(G)
    edge = rc.snap_to_edge(G, lng, lat)

    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.add_condition("closure", "测试封路", edge)

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("operation", ["remove", "update"])
def test_road_condition_mutations_reject_corrupt_store(tmp_path, monkeypatch, operation):
    path = tmp_path / "road_conditions.json"
    original = "{ invalid json"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)

    with pytest.raises(rc.RoadConditionsUnavailableError):
        if operation == "remove":
            rc.remove_condition("any")
        else:
            rc.update_condition("any", {"name": "改名"})

    assert path.read_text(encoding="utf-8") == original


def _edge_condition(G, cond_type, u=1, v=2, **overrides):
    """直接在边 (u,v) 中点吸附，构造一条绑定该边的事件。"""
    x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
    x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
    mid_lng, mid_lat = (x1 + x2) / 2, (y1 + y2) / 2
    click_gcj = wgs84_to_gcj02(mid_lng, mid_lat)
    snap = rc.snap_to_edge(G, click_gcj[0], click_gcj[1])
    assert snap is not None
    cond = rc.add_condition(
        cond_type=cond_type,
        name="测试-" + rc.CONDITION_LABELS[cond_type],
        edge=snap,
        click_point={"lng": click_gcj[0], "lat": click_gcj[1]},
        start_time=overrides.get("start_time"),
        end_time=overrides.get("end_time"),
    )
    return cond


def _path_with_penalty(G, penalties, start=0, end=3):
    # networkx 3.x MultiDiGraph 可调用 weight 第三参是 {edge_key: attr_dict}
    def weight(u, v, edges_by_key):
        k, d = next(iter(edges_by_key.items()))
        return d["length"] * penalties.get((u, v, k), 1.0)
    return nx.dijkstra_path(G, start, end, weight=weight)


# ===================== snap =====================

class TestSnap:
    def test_snap_midpoint_to_edge(self, G):
        mid_lng = (G.nodes[1]["x"] + G.nodes[2]["x"]) / 2
        mid_lat = G.nodes[1]["y"]
        gj_lng, gj_lat = wgs84_to_gcj02(mid_lng, mid_lat)
        snap = rc.snap_to_edge(G, gj_lng, gj_lat)
        assert snap is not None
        assert {snap["u"], snap["v"]} == {1, 2}
        assert snap["road_name"] == "自强大道"
        assert snap["dist_m"] < 5  # 转换往返后基本落在边上
        assert len(snap["geometry_gcj"]) >= 2
        # snap 点必须真的在路上：转回 WGS 后到原边距离≈0
        wlng, wlat = gcj02_to_wgs84(snap["snap_lng_gcj"], snap["snap_lat_gcj"])
        assert _haversine_m(wlng, wlat, mid_lng, mid_lat) < 5

    def test_snap_rejects_far_click(self, G):
        # 中点北偏约 110m，超出 30m 阈值
        gj_lng, gj_lat = wgs84_to_gcj02(G.nodes[1]["x"], G.nodes[1]["y"] + 0.001)
        assert rc.snap_to_edge(G, gj_lng, gj_lat) is None

    def test_snap_works_with_wkt_geometry(self, G):
        """graphml 加载后边 geometry 是 WKT 字符串，吸附必须照常工作。"""
        G.edges[1, 2, 0]["geometry"] = (
            "LINESTRING (%s %s, %s %s)" % (
                G.nodes[1]["x"], G.nodes[1]["y"],
                G.nodes[2]["x"], G.nodes[2]["y"],
            )
        )
        G.edges[2, 1, 0]["geometry"] = G.edges[1, 2, 0]["geometry"]
        mid_lng = (G.nodes[1]["x"] + G.nodes[2]["x"]) / 2
        gj = wgs84_to_gcj02(mid_lng, G.nodes[1]["y"])
        snap = rc.snap_to_edge(G, gj[0], gj[1])
        assert snap is not None and {snap["u"], snap["v"]} == {1, 2}


# ===================== 影响矩阵 =====================

class TestEffectMatrix:
    @pytest.mark.parametrize("mode", ["walk", "bike", "drive"])
    def test_closure_blocks_all_modes_both_directions(self, G, mode):
        _edge_condition(G, "closure")
        G2, penalties, closed, applied = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], mode)
        assert applied == 1
        assert not G2.has_edge(1, 2, 0)
        assert not G2.has_edge(2, 1, 0)  # 双向封闭
        # 主路断开 → 绕行北路
        path = nx.dijkstra_path(G2, 0, 3, weight="length")
        assert 2 not in path and 5 in path

    @pytest.mark.parametrize("mode", ["bike", "drive"])
    def test_construction_blocks_vehicles(self, G, mode):
        _edge_condition(G, "construction")
        G2, _, closed, _ = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], mode)
        assert (1, 2, 0) in closed
        path = nx.dijkstra_path(G2, 0, 3, weight="length")
        assert 2 not in path

    def test_construction_defaults_to_hard_block_for_walk(self, G):
        _edge_condition(G, "construction")
        G2, penalties, closed, applied = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], "walk")
        assert (1, 2, 0) in closed and applied == 1 and penalties == {}
        assert not G2.has_edge(1, 2, 0)
        assert 5 in nx.dijkstra_path(G2, 0, 3, weight="length")

    def test_explicit_slowdown_only_penalizes_selected_modes_without_closing_road(self, G):
        condition = _edge_condition(G, "event")
        condition.update({"action": "slowdown", "affected_modes": ["drive"], "cost_multiplier": 1.8})
        rc._save_conditions([condition])

        walk_graph, walk_penalties, walk_closed, walk_applied = rc.apply_conditions_to_graph(
            G, rc.list_conditions(), "walk", strict=True,
        )
        drive_graph, drive_penalties, drive_closed, drive_applied = rc.apply_conditions_to_graph(
            G, rc.list_conditions(), "drive", strict=True,
        )

        assert walk_graph is G and not walk_penalties and not walk_closed and walk_applied == 0
        assert drive_graph is G and not drive_closed and drive_applied == 1
        assert drive_penalties[(1, 2, 0)] == 1.8

    def test_notice_does_not_change_route_cost(self, G):
        condition = _edge_condition(G, "event")
        condition.update({"action": "notice", "affected_modes": ["walk", "bike", "drive"]})

        updated_graph, penalties, closed, applied = rc.apply_conditions_to_graph(
            G, [condition], "walk", strict=True,
        )

        assert updated_graph is G
        assert penalties == {} and closed == set()
        assert applied == 1

    @pytest.mark.parametrize("mode", ["walk", "bike"])
    def test_flooding_blocks_pedestrian(self, G, mode):
        _edge_condition(G, "flooding")
        G2, _, closed, _ = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], mode)
        assert (1, 2, 0) in closed

    def test_flooding_defaults_to_hard_block_for_drive(self, G):
        _edge_condition(G, "flooding")
        G2, penalties, closed, _ = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], "drive")
        assert (1, 2, 0) in closed and penalties == {}
        assert 5 in nx.dijkstra_path(G2, 0, 3, weight="length")

    @pytest.mark.parametrize("mode", ["walk", "bike"])
    def test_accident_defaults_to_hard_block_for_walk_and_bike(self, G, mode):
        _edge_condition(G, "accident")
        G2, penalties, closed, _ = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], mode)
        assert (1, 2, 0) in closed and penalties == {}
        assert 5 in nx.dijkstra_path(G2, 0, 3, weight="length")

    def test_accident_blocks_drive(self, G):
        _edge_condition(G, "accident")
        G2, _, closed, _ = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], "drive")
        assert (1, 2, 0) in closed

    @pytest.mark.parametrize("mode", ["walk", "bike"])
    def test_event_defaults_to_hard_block_for_walk_and_bike(self, G, mode):
        _edge_condition(G, "event")
        G2, penalties, closed, applied = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], mode)
        assert applied == 1 and penalties == {} and (1, 2, 0) in closed
        assert 5 in nx.dijkstra_path(G2, 0, 3, weight="length")

    def test_event_blocks_drive(self, G):
        _edge_condition(G, "event")
        G2, _, closed, _ = rc.apply_conditions_to_graph(G, [rc.list_conditions()[0]], "drive")
        assert (1, 2, 0) in closed

    def test_no_conditions_is_noop(self, G):
        G2, penalties, closed, applied = rc.apply_conditions_to_graph(G, [], "walk")
        assert G2 is G and penalties == {} and closed == set() and applied == 0


# ===================== 绑定解析 / 时间窗 / 兼容 =====================

class TestResolveAndLifecycle:
    def test_edge_id_invalid_falls_back_to_snap_point(self, G):
        """路网重建导致边链 id 全部失效时，用整链几何 12m 回退仍能命中。"""
        cond = _edge_condition(G, "closure")
        cond["edge"]["u"] = 999001
        cond["edge"]["v"] = 999002
        cond["edge"]["edges"] = []  # 链上 id 全部失效，触发几何回退
        keys = rc._resolve_edge_keys(G, cond)
        assert (1, 2, 0) in keys and (2, 1, 0) in keys

    def test_strict_application_rejects_unmatched_active_closure(self, G):
        cond = {
            "id": "stale-closure", "type": "closure", "start_time": 0, "end_time": 0,
            "edge": {"u": 999001, "v": 999002, "key": 0,
                     "snap": {"lng": 114.37, "lat": 30.54}},
        }

        with pytest.raises(rc.RoadConditionBindingError):
            rc.apply_conditions_to_graph(G, [cond], "walk", strict=True)

    def test_strict_application_allows_geometry_fallback_after_node_id_rebuild(self, G):
        cond = _edge_condition(G, "closure")
        cond["edge"]["u"] = 999001
        cond["edge"]["v"] = 999002
        cond["edge"]["edges"] = []

        _, _, closed, applied = rc.apply_conditions_to_graph(G, [cond], "walk", strict=True)

        assert applied == 1
        assert (1, 2, 0) in closed and (2, 1, 0) in closed

    def test_strict_application_rejects_partially_matched_edge_chain(self):
        old_graph, _ = _straight_road_graph(n_seg=4)
        _, cond = _chain_condition(old_graph, 12, 13)
        new_graph = old_graph.copy()
        # Remove one physical segment from the old graph; the other stored IDs still match.
        new_graph.remove_edge(12, 13, 0)
        new_graph.remove_edge(13, 12, 0)

        with pytest.raises(rc.RoadConditionBindingError):
            rc.apply_conditions_to_graph(new_graph, [cond], "walk", strict=True)

    def test_legacy_radius_model_still_works(self, G):
        mid_lng = (G.nodes[1]["x"] + G.nodes[2]["x"]) / 2
        gj_lng, gj_lat = wgs84_to_gcj02(mid_lng, G.nodes[1]["y"])
        legacy = {
            "id": "legacy01", "type": "closure", "name": "旧版半径事件",
            "coordinates": {"lng": gj_lng, "lat": gj_lat}, "radius_m": 20.0,
            "start_time": 0, "end_time": 0,
        }
        keys = rc._resolve_edge_keys(G, legacy)
        assert (1, 2, 0) in keys and (2, 1, 0) in keys

    def test_scheduled_and_expired_inactive(self, G):
        future = time.time() + 86400
        _edge_condition(G, "closure", start_time=future)
        assert rc.list_conditions() == []
        assert rc.apply_conditions_to_graph(G, None, "walk")[3] == 0

        _edge_condition(G, "closure", start_time=time.time() - 100, end_time=time.time() - 10)
        assert rc.list_conditions() == []

    def test_end_event_sets_end_time(self, G):
        cond = _edge_condition(G, "accident")
        updated = rc.update_condition(cond["id"], {"end_time": time.time()})
        assert updated["end_time"] > 0
        assert rc.list_conditions() == []
        # 管理端仍可看到（含已结束）
        assert rc.list_conditions(include_inactive=True)[0]["id"] == cond["id"]

    def test_remove_condition(self, G):
        cond = _edge_condition(G, "closure")
        assert rc.remove_condition(cond["id"], actor="web") is True
        assert rc.list_conditions() == []
        assert rc.list_conditions(include_inactive=True) == []
        assert rc.remove_condition("nope") is False

    def test_update_keeps_audit_history(self, G):
        cond = _edge_condition(G, "closure")
        rc.update_condition(cond["id"], {"description": "现场核实"}, actor="token")
        record = rc.list_conditions(include_inactive=True)[0]
        assert [entry["action"] for entry in record["audit"]] == ["created", "updated"]
        assert record["audit"][-1]["actor"] == "token"
        assert record["audit"][-1]["changes"] == {"description": "现场核实"}

    def test_interrupted_save_does_not_corrupt_existing_events(self, G, monkeypatch):
        _edge_condition(G, "closure")
        before = rc.list_conditions(include_inactive=True)
        original_dump = rc.json.dump

        def interrupted_dump(value, stream, *args, **kwargs):
            stream.write("[incomplete")
            raise OSError("disk write interrupted")

        monkeypatch.setattr(rc.json, "dump", interrupted_dump)
        with pytest.raises(OSError):
            _edge_condition(G, "construction")
        monkeypatch.setattr(rc.json, "dump", original_dump)
        assert rc.list_conditions(include_inactive=True) == before

    def test_concurrent_event_writes_keep_every_record(self, G):
        from concurrent.futures import ThreadPoolExecutor

        snap = rc.snap_to_edge(G, *_mid_12_gcj(G))
        with ThreadPoolExecutor(max_workers=6) as pool:
            created = list(pool.map(
                lambda index: rc.add_condition("event", f"并发事件 {index}", snap),
                range(18),
            ))
        records = rc.list_conditions(include_inactive=True, strict=True)
        assert len(records) == 18
        assert {item["id"] for item in records} == {item["id"] for item in created}

    def test_independent_processes_keep_every_event_record(self, tmp_path):
        target = tmp_path / "shared-road-conditions.json"
        target.write_text("[]\n", encoding="utf-8")
        worker = r'''
import os
from spatial.road_conditions import add_condition

edge = {
    "u": 1, "v": 2, "key": 0, "road_name": "并发测试路",
    "snap_lng_gcj": 114.36, "snap_lat_gcj": 30.53, "dist_m": 0,
    "geometry_gcj": [[114.36, 30.53], [114.361, 30.53]],
    "edges": [[1, 2, 0]],
}
worker_id = os.environ["ROAD_EVENT_WORKER_ID"]
for index in range(4):
    add_condition("event", f"{worker_id}-{index}", edge)
'''
        project_root = Path(__file__).resolve().parents[1]
        processes = []
        for worker_id in range(4):
            environment = os.environ.copy()
            environment["ROAD_CONDITIONS_FILE"] = str(target)
            environment["ROAD_EVENT_WORKER_ID"] = str(worker_id)
            processes.append(subprocess.Popen(
                [sys.executable, "-c", worker], cwd=project_root, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        failures = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            if process.returncode:
                failures.append((process.returncode, stdout, stderr))
        assert failures == []

        records = json.loads(target.read_text(encoding="utf-8"))
        assert len(records) == 16
        assert {item["name"] for item in records} == {
            f"{worker_id}-{index}" for worker_id in range(4) for index in range(4)
        }

    def test_add_rejects_unknown_type(self, G):
        snap = rc.snap_to_edge(G, *wgs84_to_gcj02(G.nodes[1]["x"], G.nodes[1]["y"]))
        with pytest.raises(ValueError):
            rc.add_condition("earthquake", "地震", edge=snap)


# ===================== 边链扩展（路口到路口整段） =====================

def _straight_road_graph(n_seg=4, names=None, start_id=10, extra_edges=None,
                         base=(114.37000, 30.54000), step_deg=0.00055):
    """水平直路 start_id … start_id+n_seg，每段约 53m（双向边）。

    extra_edges: [(a, b, name)] 在已有节点 a 上接一条向北的岔路到新节点 b。
    """
    G = nx.MultiDiGraph()
    ids = list(range(start_id, start_id + n_seg + 1))
    for nid in ids:
        i = nid - start_id
        G.add_node(nid, x=base[0] + step_deg * i, y=base[1])

    def _add_bidir(a, b, name):
        length = _haversine_m(G.nodes[a]["x"], G.nodes[a]["y"],
                              G.nodes[b]["x"], G.nodes[b]["y"])
        attrs = {"length": length, "highway": "residential"}
        if name:
            attrs["name"] = name
        G.add_edge(a, b, 0, **attrs)
        G.add_edge(b, a, 0, **attrs)

    if names is None:
        names = ["樱花大道"] * n_seg
    for i in range(n_seg):
        _add_bidir(ids[i], ids[i + 1], names[i])
    if extra_edges:
        for a, b, name in extra_edges:
            if b not in G:
                G.add_node(b, x=G.nodes[a]["x"], y=G.nodes[a]["y"] + step_deg)
            _add_bidir(a, b, name)
    return G, ids


def _undirected_edge_set(triples):
    return {frozenset((t[0], t[1])) for t in triples}


def _chain_condition(G, u, v, cond_type="closure"):
    x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
    x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
    click = wgs84_to_gcj02((x1 + x2) / 2, (y1 + y2) / 2)
    snap = rc.snap_to_edge(G, click[0], click[1])
    assert snap is not None
    return snap, rc.add_condition(
        cond_type=cond_type, name="链测试", edge=snap,
        click_point={"lng": click[0], "lat": click[1]},
    )


class TestEdgeChain:
    def test_chain_extends_through_degree2_to_both_ends(self):
        """直路中间边：链穿过所有 degree=2 节点，延伸到两端断头。"""
        G, ids = _straight_road_graph(n_seg=4)
        snap, _ = _chain_condition(G, 12, 13)
        assert _undirected_edge_set(snap["edges"]) == _undirected_edge_set(
            [(10, 11), (11, 12), (12, 13), (13, 14)]
        )
        assert len(snap["edges"]) == 4
        assert 200 < snap["chain_length_m"] < 225
        # 合并几何覆盖道路两端
        first = snap["geometry_gcj"][0]
        last = snap["geometry_gcj"][-1]
        g10 = wgs84_to_gcj02(G.nodes[10]["x"], G.nodes[10]["y"])
        g14 = wgs84_to_gcj02(G.nodes[14]["x"], G.nodes[14]["y"])
        assert abs(first[0] - g10[0]) < 1e-6 and abs(first[1] - g10[1]) < 1e-6
        assert abs(last[0] - g14[0]) < 1e-6 and abs(last[1] - g14[1]) < 1e-6

    def test_chain_stops_at_intersection(self):
        """端点是路口（无向度=3）时链停止，不串到岔路。"""
        G, ids = _straight_road_graph(n_seg=4, extra_edges=[(12, 20, "岔路")])
        snap, _ = _chain_condition(G, 11, 12)
        assert _undirected_edge_set(snap["edges"]) == _undirected_edge_set(
            [(10, 11), (11, 12)]
        )

    def test_chain_stops_on_road_name_change(self):
        """degree=2 直连点处道路名变化即停止（不串到另一条路）。"""
        G, ids = _straight_road_graph(
            n_seg=4, names=["樱花大道", "梅园路", "梅园路", "梅园路"]
        )
        snap, _ = _chain_condition(G, 10, 11)
        assert _undirected_edge_set(snap["edges"]) == _undirected_edge_set([(10, 11)])

    def test_chain_stops_on_sharp_turn(self):
        """degree=2 但相邻边夹角 >100°（急弯折返）时停止。"""
        G = nx.MultiDiGraph()
        G.add_node(30, x=114.37000, y=30.54000)
        G.add_node(31, x=114.37105, y=30.54000)   # 30→31 向东
        G.add_node(32, x=114.37032, y=30.53900)   # 31→32 向西南（夹角约 149°）
        for a, b in [(30, 31), (31, 32)]:
            length = _haversine_m(G.nodes[a]["x"], G.nodes[a]["y"],
                                  G.nodes[b]["x"], G.nodes[b]["y"])
            for u, v in ((a, b), (b, a)):
                G.add_edge(u, v, 0, length=length, highway="residential", name="环山道")
        snap, _ = _chain_condition(G, 30, 31)
        assert _undirected_edge_set(snap["edges"]) == _undirected_edge_set([(30, 31)])

    def test_chain_respects_600m_cap(self):
        """约 800m 直路：600m 预算下主边两侧各只延伸 2 段，整链 5 段。"""
        G, ids = _straight_road_graph(n_seg=8, step_deg=0.00105)  # 每段约 100m
        snap, _ = _chain_condition(G, 14, 15)
        assert _undirected_edge_set(snap["edges"]) == _undirected_edge_set(
            [(12, 13), (13, 14), (14, 15), (15, 16), (16, 17)]
        )
        assert snap["chain_length_m"] <= 601.0

    def test_resolve_edge_keys_covers_full_chain_both_directions(self):
        """规划期解析：链上每条边双向全部命中。"""
        G, ids = _straight_road_graph(n_seg=4)
        _, cond = _chain_condition(G, 12, 13)
        keys = rc._resolve_edge_keys(G, cond)
        assert len(keys) == 8
        for a, b in [(10, 11), (11, 12), (12, 13), (13, 14)]:
            assert (a, b, 0) in keys and (b, a, 0) in keys

    def test_persisted_condition_contains_chain(self):
        G, ids = _straight_road_graph(n_seg=4)
        _, cond = _chain_condition(G, 12, 13)
        edge = cond["edge"]
        assert [12, 13, 0] in edge["edges"]
        assert len(edge["edges"]) == 4
        assert edge["chain_length_m"] > 200
        assert len(edge["geometry_gcj"]) >= 5

    def test_chain_geometry_fallback_after_node_id_rebuild(self):
        """路网重建后 id 全变：靠整链几何回退仍能命中新图上整段（双向）。"""
        G1, _ = _straight_road_graph(n_seg=4, start_id=10)
        _, cond = _chain_condition(G1, 12, 13)
        G2, _ = _straight_road_graph(n_seg=4, start_id=1010)  # 坐标完全相同、id 全换
        keys = rc._resolve_edge_keys(G2, cond)
        assert len(keys) == 8
        for a, b in [(1010, 1011), (1011, 1012), (1012, 1013), (1013, 1014)]:
            assert (a, b, 0) in keys and (b, a, 0) in keys

    def test_legacy_condition_without_edges_list_still_resolves(self, G):
        """旧数据（只有 u/v、无 edges 链）继续按单条边双向解析。"""
        legacy = {
            "id": "old01", "type": "closure", "name": "旧单条边事件",
            "edge": {"u": 1, "v": 2, "key": 0, "snap": {"lng": 0, "lat": 0}},
        }
        keys = rc._resolve_edge_keys(G, legacy)
        assert keys == {(1, 2, 0), (2, 1, 0)}


# ===================== HTTP 层 =====================

@pytest.fixture
def client(G, monkeypatch):
    import config
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "test-pw-123")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "test-token-xyz")
    from app import create_app
    import api.routes as routes
    monkeypatch.setattr(routes, "_ensure_network", lambda: (G, None))
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _mid_12_gcj(G):
    """边 (1,2) 中点的 GCJ-02 坐标（端点会同时相邻两条边，用中点避免歧义）。"""
    mid_lng = (G.nodes[1]["x"] + G.nodes[2]["x"]) / 2
    return wgs84_to_gcj02(mid_lng, G.nodes[1]["y"])


def test_api_route_rejects_endpoint_far_from_routable_network(client, G, monkeypatch):
    import api.routes as routes
    monkeypatch.setattr(routes, "_mode_filtered_graph", lambda graph, mode: (graph, "ok", {}))

    response = client.post("/api/route", json={
        "start": {"type": "coord", "name": "地图起点",
                  "coordinates": {"lng": 114.37, "lat": 30.54}},
        "end": {"type": "coord", "name": "终点",
                "coordinates": {"lng": 114.36167, "lat": 30.53}},
    })

    assert response.status_code == 422
    assert response.get_json()["error"] == "endpoint_too_far_from_network"


def test_api_route_returns_503_when_active_closure_cannot_bind_to_graph(client, G, monkeypatch):
    import api.routes as routes
    monkeypatch.setattr(routes, "_mode_filtered_graph", lambda graph, mode: (graph, "ok", {}))
    monkeypatch.setattr(routes, "list_conditions", lambda **_kwargs: [{
        "id": "stale-closure", "type": "closure", "start_time": 0, "end_time": 0,
        "edge": {"u": 999001, "v": 999002, "key": 0,
                 "snap": {"lng": 114.37, "lat": 30.54}},
    }])

    response = client.post("/api/route", json={
        "start": {"type": "coord", "name": "起点",
                  "coordinates": {"lng": G.nodes[0]["x"], "lat": G.nodes[0]["y"]}},
        "end": {"type": "coord", "name": "终点",
                "coordinates": {"lng": G.nodes[3]["x"], "lat": G.nodes[3]["y"]}},
    })

    assert response.status_code == 503
    assert response.get_json()["error"] == "road_conditions_unavailable"


def test_new_manager_restrictions_default_to_hard_block_for_every_travel_mode(G):
    """A newly published restriction must never silently remain traversable by a mode."""
    for cond_type in rc.CONDITION_EFFECTS:
        condition = _edge_condition(G, cond_type)
        for mode in ("walk", "bike", "drive"):
            modified, _penalties, closed, applied = rc.apply_conditions_to_graph(
                G, [condition], mode, strict=True,
            )
            assert applied == 1
            assert (1, 2, 0) in closed
            assert not modified.has_edge(1, 2, 0)
            assert not modified.has_edge(2, 1, 0)


def test_explicit_passability_override_is_authoritative_for_each_mode(G):
    condition = _edge_condition(G, "accident")
    condition["blocked_modes"] = ["walk"]

    for mode in ("walk", "bike", "drive"):
        modified, penalties, closed, applied = rc.apply_conditions_to_graph(
            G, [condition], mode, strict=True,
        )
        if mode == "walk":
            assert applied == 1
            assert (1, 2, 0) in closed
            assert not modified.has_edge(1, 2, 0)
        else:
            assert applied == 0
            assert not closed
            assert not penalties
            assert modified is G


def test_restriction_blocks_overlapping_parallel_edge_keys(G):
    attrs = dict(G.get_edge_data(1, 2, 0))
    reverse_attrs = dict(G.get_edge_data(2, 1, 0))
    G.add_edge(1, 2, 7, **attrs)
    G.add_edge(2, 1, 7, **reverse_attrs)
    condition = _edge_condition(G, "closure")

    modified, _penalties, closed, _applied = rc.apply_conditions_to_graph(
        G, [condition], "walk", strict=True,
    )

    assert (1, 2, 7) in closed and (2, 1, 7) in closed
    assert not modified.has_edge(1, 2, 7)
    assert not modified.has_edge(2, 1, 7)


def test_deleted_road_condition_is_physically_removed_from_persistent_snapshot(G):
    condition = _edge_condition(G, "closure")

    assert rc.remove_condition(condition["id"], actor="web") is True

    assert all(item["id"] != condition["id"] for item in rc.list_conditions(include_inactive=True))
    assert all(item["id"] != condition["id"] for item in json.loads(Path(rc._CONDITIONS_FILE).read_text(encoding="utf-8")))


def test_randomly_selected_real_campus_road_cannot_be_reopened_by_parallel_geometry():
    """Select a real graph edge at random, duplicate its physical geometry, and verify closure."""
    import random

    graph_path = Path(__file__).resolve().parents[1] / "data" / "whu_road_network.graphml"
    graph = nx.read_graphml(graph_path, node_type=int)
    candidates = [
        (u, v, key) for u, v, key, data in graph.edges(keys=True, data=True)
        if graph.has_edge(v, u) and float(data.get("length", 0) or 0) > 4
    ]
    rng = random.Random(20261005)
    rng.shuffle(candidates)

    for u, v, key in candidates[:300]:
        data = graph.get_edge_data(u, v, key) or {}
        line = rc._edge_line(graph, u, v, data)
        midpoint = line.interpolate(0.5, normalized=True)
        click = wgs84_to_gcj02(midpoint.x, midpoint.y)
        snap = rc.snap_to_edge(graph, click[0], click[1])
        if snap is None:
            continue
        chain_segment = rng.choice(snap["edges"])
        a, b, selected_key = map(int, chain_segment)
        selected = graph.get_edge_data(a, b, selected_key)
        reverse = graph.get_edge_data(b, a, selected_key)
        if not selected or not reverse:
            continue
        duplicate_key = max(graph.get_edge_data(a, b).keys()) + 1000000
        graph.add_edge(a, b, duplicate_key, **dict(selected))
        graph.add_edge(b, a, duplicate_key, **dict(reverse))
        condition = rc.add_condition(
            "closure", "随机实路验收", snap,
            blocked_modes=["walk", "bike", "drive"],
        )

        modified, _penalties, closed, applied = rc.apply_conditions_to_graph(
            graph, [condition], "walk", strict=True,
        )

        assert applied == 1
        assert (a, b, duplicate_key) in closed
        assert (b, a, duplicate_key) in closed
        assert not modified.has_edge(a, b, duplicate_key)
        assert not modified.has_edge(b, a, duplicate_key)
        from spatial.routing import compute_route
        try:
            route = compute_route(
                graph, a, b, mode="walk", strategy_name="shortest",
                road_conditions=[condition],
            )
        except ValueError as error:
            assert "封闭" in str(error) or "暂时无法通行" in str(error)
        else:
            assert not any(tuple(edge) in closed for edge in route["recommended_edges"])
        return

    pytest.fail("could not select a snap-valid random edge from the real campus network")


def test_live_selfqiang_closure_blocks_the_parallel_path_that_bypassed_it():
    """Regression for a production route that escaped a closure on a nearby footway."""
    from spatial.routing import compute_route

    graph_path = Path(__file__).resolve().parents[1] / "data" / "whu_road_network.graphml"
    graph = nx.read_graphml(graph_path, node_type=int)
    u, v, key = 3110802248, 3110802296, 0
    assert graph.has_edge(u, v, key)

    line = rc._edge_line(graph, u, v, graph.get_edge_data(u, v, key))
    midpoint = line.interpolate(0.5, normalized=True)
    click_lng, click_lat = wgs84_to_gcj02(midpoint.x, midpoint.y)
    snap = rc.snap_to_edge(graph, click_lng, click_lat)
    assert snap is not None
    assert {snap["u"], snap["v"]} == {u, v}
    condition = {
        "id": "real-campus-selfqiang-regression",
        "type": "closure",
        "blocked_modes": ["walk", "bike", "drive"],
        "start_time": 0,
        "end_time": 0,
        "edge": {
            "u": snap["u"], "v": snap["v"], "key": snap["key"],
            "edges": snap["edges"], "chain_length_m": snap["chain_length_m"],
            "road_name": snap["road_name"],
            "snap": {"lng": snap["snap_lng_gcj"], "lat": snap["snap_lat_gcj"]},
            "geometry_gcj": snap["geometry_gcj"],
        },
    }

    resolved = rc._resolve_edge_keys(graph, condition, strict=True)
    closed_corridor = rc._expand_coincident_corridor_edges(graph, resolved, condition["edge"])
    restricted_graph, _penalties, applied_corridor, applied = rc.apply_conditions_to_graph(
        graph, [condition], "walk", strict=True,
    )
    assert applied == 1
    assert applied_corridor == closed_corridor
    # This is a real OSM footway edge in the production bypass. It follows the
    # managed corridor, but its middle segment sits 11.7 m from the centerline.
    bypass = (13239306781, 13239758406, 0)
    assert bypass in closed_corridor
    # The 90-degree connectors at either end only touch the selected road; they
    # must remain available while the parallel continuation is closed.
    crossings = [(u, 13239306781, 0), (13239306755, v, 0)]
    assert all(edge not in closed_corridor for edge in crossings)
    assert all(restricted_graph.has_edge(*edge) for edge in crossings)

    route = compute_route(
        graph, u, v, mode="walk", strategy_name="shortest",
        road_conditions=[condition],
    )
    route_edges = {tuple(edge) for edge in route["recommended_edges"]}
    assert bypass not in route_edges
    assert float(route["recommended_length_m"]) > float(condition["edge"]["chain_length_m"])


def test_random_real_campus_closures_remove_their_aligned_corridor_from_routes():
    """Exercise random, untouched production road edges instead of edited examples."""
    import random
    from spatial.routing import compute_route
    from spatial.routing_index import get_routing_index

    graph_path = Path(__file__).resolve().parents[1] / "data" / "whu_road_network.graphml"
    graph = nx.read_graphml(graph_path, node_type=int)
    candidates = [
        (u, v, key, data)
        for u, v, key, data in graph.edges(keys=True, data=True)
        if u < v and graph.has_edge(v, u, key)
        and 20 <= float(data.get("length", 0) or 0) <= 250
    ]
    rng = random.Random(20261005)
    samples = rng.sample(candidates, 12)
    prepared = get_routing_index(graph).for_mode("walk")
    corridors_with_real_parallel_edges = 0
    successful_routes = 0

    for sample_index, (u, v, key, data) in enumerate(samples):
        line = rc._edge_line(graph, u, v, data)
        midpoint = line.interpolate(rng.uniform(0.25, 0.75), normalized=True)
        click_lng, click_lat = wgs84_to_gcj02(midpoint.x, midpoint.y)
        geometry_gcj = [
            list(wgs84_to_gcj02(lng, lat))
            for lng, lat in rc._line_coords(line)
        ]
        condition = {
            "id": f"random-real-edge-{sample_index}",
            "type": "closure",
            "blocked_modes": ["walk", "bike", "drive"],
            "start_time": 0,
            "end_time": 0,
            "edge": {
                "u": u, "v": v, "key": key, "edges": [[u, v, key]],
                "chain_length_m": float(data.get("length", 0) or 0),
                "road_name": rc._road_name_of(data),
                "snap": {"lng": click_lng, "lat": click_lat},
                "geometry_gcj": geometry_gcj,
            },
        }
        exact_edges = rc._resolve_edge_keys(graph, condition, strict=True)
        restricted, _penalties, closed, applied = rc.apply_conditions_to_graph(
            graph, [condition], "walk", strict=True,
        )
        assert applied == 1
        assert exact_edges.issubset(closed)
        if closed - exact_edges:
            corridors_with_real_parallel_edges += 1
        assert all(not restricted.has_edge(*edge) for edge in closed)

        try:
            route = compute_route(
                graph, u, v, mode="walk", strategy_name="shortest",
                road_conditions=[condition], prepared=prepared,
            )
        except ValueError:
            # No legal detour is safe: the planner must report unreachable.
            continue
        successful_routes += 1
        assert all(tuple(edge) not in closed for edge in route["recommended_edges"])

    assert corridors_with_real_parallel_edges >= 3
    assert successful_routes >= 1


class TestRoadConditionAPI:
    def test_snap_requires_auth(self, client):
        r = client.get("/api/road-conditions/snap?lng=114.36&lat=30.53")
        assert r.status_code == 401

    def test_snap_with_token(self, client, G):
        gj = _mid_12_gcj(G)
        r = client.get(
            f"/api/road-conditions/snap?lng={gj[0]}&lat={gj[1]}",
            headers={"X-Admin-Token": "test-token-xyz"},
        )
        assert r.status_code == 200
        snap = r.get_json()["data"]["snap"]
        assert {snap["u"], snap["v"]} == {1, 2}

    def test_route_refuses_to_plan_when_road_condition_snapshot_is_unavailable(self, client, G, monkeypatch):
        import api.routes as routes
        monkeypatch.setattr(
            routes, "list_conditions",
            lambda **_kwargs: (_ for _ in ()).throw(rc.RoadConditionsUnavailableError("bad store")),
        )
        monkeypatch.setattr(
            routes, "compute_route",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not route without condition state")),
        )
        response = client.post("/api/route", json={
            "start": {"type": "coord", "coordinates": {"lng": G.nodes[0]["x"], "lat": G.nodes[0]["y"]}},
            "end": {"type": "coord", "coordinates": {"lng": G.nodes[3]["x"], "lat": G.nodes[3]["y"]}},
        })

        assert response.status_code == 503
        assert response.get_json()["error"] == "road_conditions_unavailable"

    def test_snap_type_includes_mode_and_sample_route_impact_preview(self, client, G, monkeypatch):
        import api.routes as routes
        calls = []
        active_condition = {"id": "active-1", "type": "accident", "edge": {"u": 4, "v": 5, "key": 0}}

        monkeypatch.setattr(routes, "list_conditions", lambda **_kwargs: [active_condition])

        def fake_compute_route(graph, start, end, **kwargs):
            calls.append((start, end, kwargs["mode"], kwargs["road_conditions"]))
            return {"recommended_length_m": 100 if len(kwargs["road_conditions"]) == 1 else 165,
                    "duration_min": 2 if len(kwargs["road_conditions"]) == 1 else 3}

        monkeypatch.setattr(routes, "compute_route", fake_compute_route)
        gj = _mid_12_gcj(G)
        response = client.get(
            f"/api/road-conditions/snap?lng={gj[0]}&lat={gj[1]}&type=closure",
            headers={"X-Admin-Token": "test-token-xyz"},
        )

        assert response.status_code == 200
        preview = response.get_json()["data"]["impact_preview"]
        assert preview["affected_road_segments"] == 1
        assert preview["effects_by_mode"]["walk"]["status"] == "blocked"
        assert preview["route_sample_scope"] == "所选路段两端之间的示例路线，不代表全校总影响"
        assert preview["sample_routes"]["walk"]["detour_m"] == 65
        assert preview["active_event_count"] == 1
        assert calls[0][3] == [active_condition]
        assert calls[1][3][0] == active_condition
        assert calls[1][3][1]["id"] == "preview-only"
        assert len(calls) == 6

    def test_snap_rejects_unknown_preview_event_type(self, client, G):
        gj = _mid_12_gcj(G)
        response = client.get(
            f"/api/road-conditions/snap?lng={gj[0]}&lat={gj[1]}&type=unknown",
            headers={"X-Admin-Token": "test-token-xyz"},
        )
        assert response.status_code == 400
        assert response.get_json()["error"] == "invalid_type"

    def test_snap_wrong_token(self, client):
        r = client.get(
            "/api/road-conditions/snap?lng=114.36&lat=30.53",
            headers={"X-Admin-Token": "wrong"},
        )
        assert r.status_code == 401

    def test_snap_bearer_auth(self, client, G):
        gj = _mid_12_gcj(G)
        r = client.get(
            f"/api/road-conditions/snap?lng={gj[0]}&lat={gj[1]}",
            headers={"Authorization": "Bearer test-token-xyz"},
        )
        assert r.status_code == 200

    def test_post_binds_edge_no_radius(self, client, G):
        gj = _mid_12_gcj(G)
        r = client.post("/api/road-conditions", json={
            "type": "closure", "name": "接口封闭测试",
            "lng": gj[0], "lat": gj[1], "blocked_modes": ["walk", "bike", "drive"],
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert r.status_code == 201
        cond = r.get_json()["data"]["condition"]
        assert cond["edge"]["road_name"] == "自强大道"
        assert "radius_m" not in cond
        assert cond["created_by"] == "token"
        assert set(cond["blocked_modes"]) == {"walk", "bike", "drive"}

    def test_post_requires_explicit_route_effect(self, client, G):
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "缺少路线影响设置",
            "lng": gj[0], "lat": gj[1],
        }, headers={"X-Admin-Token": "test-token-xyz"})

        assert response.status_code == 400
        assert response.get_json()["error"] == "route_effect_required"
        assert rc.list_conditions() == []

    def test_post_persists_explicit_modes_that_manager_confirms_are_blocked(self, client, G):
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "construction", "name": "仅步行封闭",
            "lng": gj[0], "lat": gj[1], "blocked_modes": ["walk"],
        }, headers={"X-Admin-Token": "test-token-xyz"})

        assert response.status_code == 201
        assert response.get_json()["data"]["condition"]["blocked_modes"] == ["walk"]

    def test_post_persists_action_and_only_selected_travel_modes(self, client, G):
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "仅步行缓行",
            "lng": gj[0], "lat": gj[1],
            "action": "slowdown", "affected_modes": ["walk"],
            "cost_multiplier": 1.7,
        }, headers={"X-Admin-Token": "test-token-xyz"})

        assert response.status_code == 201
        condition = response.get_json()["data"]["condition"]
        assert condition["action"] == "slowdown"
        assert condition["affected_modes"] == ["walk"]
        assert condition["cost_multiplier"] == 1.7
        assert "blocked_modes" not in condition
        public = client.get("/api/road-conditions").get_json()["data"]["conditions"][0]
        assert public["action"] == "slowdown"
        assert public["affected_modes"] == ["walk"]

    def test_confirmed_vision_requires_field_evidence_before_road_publication(self, client, G, monkeypatch):
        import api.routes as routes
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        job = {"job_id": job_id, "status": "needs_review", "review_status": "confirmed",
               "review_candidate_index": 0,
               "captured_at": time.time(),
               "result": {"candidates": [{
                   "kind": "vehicle_cluster_review",
                   "review_required": True,
                   "auto_publish": False,
               }]}}
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: job)
        gj = _mid_12_gcj(G)
        body = {"type": "event", "name": "现场复核的人流事件", "lng": gj[0], "lat": gj[1],
                "action": "notice", "affected_modes": ["walk"],
                "source_vision_job_id": job_id}
        headers = {"X-Admin-Token": "test-token-xyz"}

        missing = client.post("/api/road-conditions", json=body, headers=headers)
        assert missing.status_code == 400
        assert rc.list_conditions() == []

        body["field_confirmation"] = "现场人员反馈该路段正在拥堵，已核实作用范围。"
        body["end_time"] = time.time() + 1800
        published = client.post("/api/road-conditions", json=body, headers=headers)
        assert published.status_code == 201
        condition = published.get_json()["data"]["condition"]
        assert condition["source"]["job_id"] == job_id
        assert condition["source"]["candidate_index"] == 0
        assert condition["source"]["candidate_kind"] == "vehicle_cluster_review"
        assert condition["source"]["field_confirmation"] == body["field_confirmation"]
        public_records = client.get("/api/road-conditions").get_json()["data"]["conditions"]
        assert len(public_records) == 1
        assert "source" not in public_records[0] and "audit" not in public_records[0]
        version = client.get("/api/road-conditions?version_only=1").get_json()["data"]
        assert version["active_event_version"] == routes._current_road_condition_version(
            rc.list_conditions(strict=True),
        )

    def test_vision_publication_requires_explicit_action_and_affected_modes(
        self, client, G, monkeypatch,
    ):
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        job = {
            "job_id": job_id, "status": "needs_review", "captured_at": time.time(),
            "candidate_reviews": {"0": {"status": "confirmed"}},
            "result": {"candidates": [{
                "kind": "possible_congestion", "review_required": True,
                "auto_publish": False,
            }]},
        }
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: job)
        lng, lat = _mid_12_gcj(G)
        body = {
            "type": "event", "name": "影像确认的路况", "lng": lng, "lat": lat,
            "source_vision_job_id": job_id, "source_vision_candidate_index": 0,
            "field_confirmation": "现场已核实该路段需要临时管制，请按实际情况发布。",
            "end_time": time.time() + 1800,
        }

        response = client.post(
            "/api/road-conditions", json=body,
            headers={"X-Admin-Token": "test-token-xyz"},
        )

        assert response.status_code == 400
        assert response.get_json()["error"] == "vision_effect_required"
        assert rc.list_conditions() == []

    def test_agent_condition_lookup_reports_unavailable_store_instead_of_no_conditions(self):
        from agents import tools as agent_tools

        Path(rc._CONDITIONS_FILE).write_text("{ invalid json", encoding="utf-8")
        result, _ = agent_tools.execute_tool("list_road_conditions", {})

        assert result["error"] == "road_conditions_unavailable"
        assert "不能确认是否存在管制信息" in result["message"]
        assert "当前没有生效的路况信息" not in result["message"]

    @pytest.mark.parametrize("action,cost_multiplier", [
        ("closure", None),
        ("slowdown", 1.5),
        ("notice", None),
    ])
    def test_confirmed_vision_candidate_reaches_agent_and_changes_route(
        self, client, G, monkeypatch, action, cost_multiplier,
    ):
        """A manager-confirmed image candidate must survive publication, lookup, and routing."""
        from agents import tools as agent_tools
        from spatial.routing import compute_route
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        candidate = {
            "kind": "vehicle_cluster_review",
            "review_required": True,
            "auto_publish": False,
        }
        job = {
            "job_id": job_id,
            "status": "needs_review",
            "captured_at": time.time(),
            "candidate_reviews": {},
            "result": {"candidates": [candidate]},
        }
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: job)

        def confirm_candidate(*_args, **kwargs):
            job["candidate_reviews"][str(kwargs["candidate_index"])] = {
                "status": kwargs["review_status"],
                "note": kwargs["review_note"],
                "reviewed_by": kwargs["reviewed_by"],
            }
            return job

        monkeypatch.setattr(vision_repository, "review_candidate", confirm_candidate)
        headers = {"X-Admin-Token": "test-token-xyz"}
        reviewed = client.post(
            f"/api/manager/vision-jobs/{job_id}/candidates/0/review",
            json={"status": "confirmed", "note": "已核对视频，确认该路段需要临时管制。"},
            headers=headers,
        )
        assert reviewed.status_code == 200

        lng, lat = _mid_12_gcj(G)
        published = client.post("/api/road-conditions", json={
            "type": "event",
            "action": action,
            "affected_modes": ["walk"],
            "cost_multiplier": cost_multiplier,
            "name": "航拍复核道路临时封闭",
            "lng": lng,
            "lat": lat,
            "source_vision_job_id": job_id,
            "source_vision_candidate_index": 0,
            "field_confirmation": "管理员核对现场信息，确认自强大道该路段暂时不能步行通行。",
            "end_time": time.time() + 1800,
        }, headers=headers)
        assert published.status_code == 201

        conditions = rc.list_conditions(strict=True)
        assert len(conditions) == 1
        condition = conditions[0]
        assert condition["source"]["candidate_index"] == 0
        assert condition["edge"]["road_name"] == "自强大道"

        tool_result, _ = agent_tools.execute_tool("list_road_conditions", {})
        agent_condition = tool_result["conditions"][0]
        assert agent_condition["source_kind"] == "vision_job"
        assert agent_condition["road_name"] == "自强大道"
        assert agent_condition["route_effect"] == {
            "action": action,
            "affected_modes": ["walk"],
            "cost_multiplier": cost_multiplier,
        }

        baseline = compute_route(
            G, 0, 3, mode="walk", strategy_name="shortest", road_conditions=[],
        )
        route = compute_route(
            G, 0, 3, mode="walk", strategy_name="shortest",
            road_conditions=conditions,
        )
        assert route["road_conditions_applied"] == 1
        if action == "closure":
            assert route["recommended"] == [0, 4, 5, 6, 7, 3]
        else:
            assert route["recommended"] == [0, 1, 2, 3]
            if action == "slowdown":
                assert route["duration_min"] > baseline["duration_min"]
            else:
                assert route["duration_min"] == baseline["duration_min"]

        duplicate = client.post("/api/road-conditions", json={
            "type": "event", "action": action, "affected_modes": ["walk"],
            "cost_multiplier": cost_multiplier, "name": "航拍复核道路临时封闭",
            "lng": lng, "lat": lat, "source_vision_job_id": job_id,
            "source_vision_candidate_index": 0,
            "field_confirmation": "管理员核对现场信息，确认自强大道该路段暂时不能步行通行。",
            "end_time": time.time() + 1800,
        }, headers=headers)
        assert duplicate.status_code == 200
        assert duplicate.get_json()["data"]["already_published"] is True
        assert duplicate.get_json()["data"]["condition"]["id"] == condition["id"]
        assert len(rc.list_conditions(strict=True)) == 1

    def test_completed_vision_job_without_candidates_can_be_registered_manually(self, client, G, monkeypatch):
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        job = {
            "job_id": job_id, "status": "completed", "captured_at": time.time(),
            "result": {"candidates": []}, "observation_regions": [],
        }
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: job)
        gj = _mid_12_gcj(G)
        body = {
            "type": "event", "action": "notice", "affected_modes": ["walk"],
            "name": "人工登记的影像路况", "lng": gj[0], "lat": gj[1],
            "source_vision_job_id": job_id, "source_vision_candidate_index": None,
            "field_confirmation": "管理员查看完整影像后现场核实该人行路段通行受影响。",
            "end_time": time.time() + 1800,
        }

        response = client.post(
            "/api/road-conditions", json=body,
            headers={"X-Admin-Token": "test-token-xyz"},
        )

        assert response.status_code == 201
        source = response.get_json()["data"]["condition"]["source"]
        assert source["job_id"] == job_id
        assert source["candidate_index"] is None
        assert source["candidate_kind"] == "manual_visual_event"
        assert source["manual_registration"] is True
        assert source["field_confirmation"] == body["field_confirmation"]
        assert source["road_association"] == "manager_selected_edge"
        assert source["associated_edge"]["road_name"] == "自强大道"

    @pytest.mark.parametrize("candidate_kind", ["possible_congestion", "possible_flooding"])
    def test_independently_confirmed_recent_candidate_can_publish_with_expiry(
        self, client, G, monkeypatch, candidate_kind,
    ):
        import time
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        candidate = {"kind": candidate_kind, "review_required": True, "auto_publish": False}
        job = {
            "job_id": job_id, "status": "needs_review", "review_status": None,
            "captured_at": time.time(),
            "candidate_reviews": {"0": {"status": "dismissed"}, "1": {"status": "confirmed"}},
            "result": {"candidates": [candidate, candidate]},
        }
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: job)
        gj = _mid_12_gcj(G)
        body = {
            "type": "event", "name": "现场复核的人流事件", "lng": gj[0], "lat": gj[1],
            "action": "notice", "affected_modes": ["walk"],
            "source_vision_job_id": job_id, "source_vision_candidate_index": 1,
            "field_confirmation": "现场人员已核实该路段当前拥堵情况。",
            "end_time": time.time() + 1800,
        }
        response = client.post("/api/road-conditions", json=body, headers={"X-Admin-Token": "test-token-xyz"})
        assert response.status_code == 201
        assert response.get_json()["data"]["condition"]["source"]["candidate_index"] == 1
        assert response.get_json()["data"]["condition"]["source"]["candidate_kind"] == candidate_kind

    @pytest.mark.parametrize(("candidate_kind", "region_id", "region_kind", "evidence"), [
        ("possible_flooding", "surface-east", "road_surface",
         {"summary": {"region_id": "surface-east", "water_depth_estimated": False}}),
        ("possible_crowding", "walkway-east", "pedestrian",
         {"region_id": "walkway-east", "peak_person_count": 18}),
    ])
    def test_vision_road_event_keeps_observation_region_provenance(
        self, client, G, monkeypatch, candidate_kind, region_id, region_kind, evidence,
    ):
        """The manually selected road must remain traceable to the analyzed ROI."""
        import time
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        polygon = [[0.1, 0.2], [0.8, 0.2], [0.8, 0.6], [0.1, 0.6]]
        candidate = {
            "kind": candidate_kind, "review_required": True, "auto_publish": False,
            "evidence": evidence,
        }
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: {
            "job_id": job_id, "status": "needs_review", "captured_at": time.time(),
            "candidate_reviews": {"0": {"status": "confirmed"}},
            "observation_regions": [{"id": region_id, "kind": region_kind, "polygon": polygon}],
            "result": {"candidates": [candidate]},
        })
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "复核的道路积水", "lng": gj[0], "lat": gj[1],
            "action": "notice", "affected_modes": ["walk"],
            "source_vision_job_id": job_id, "source_vision_candidate_index": 0,
            "field_confirmation": "现场人员已核实该道路有积水并确认影响范围。",
            "end_time": time.time() + 1800,
        }, headers={"X-Admin-Token": "test-token-xyz"})

        assert response.status_code == 201
        source = response.get_json()["data"]["condition"]["source"]
        edge = response.get_json()["data"]["condition"]["edge"]
        assert source["region_id"] == region_id
        assert source["region_kind"] == region_kind
        assert source["region_polygon"] == polygon
        assert source["road_association"] == "manager_selected_edge"
        assert source["associated_edge"] == {
            "u": edge["u"], "v": edge["v"], "key": edge["key"],
            "road_name": edge["road_name"],
        }

    def test_vision_candidate_with_unknown_region_cannot_publish(self, client, G, monkeypatch):
        import time
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: {
            "job_id": job_id, "status": "needs_review", "captured_at": time.time(),
            "candidate_reviews": {"0": {"status": "confirmed"}},
            "observation_regions": [{"id": "surface-east", "kind": "road_surface",
                                     "polygon": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.6]]}],
            "result": {"candidates": [{
                "kind": "possible_flooding", "review_required": True, "auto_publish": False,
                "evidence": {"summary": {"region_id": "not-in-job"}},
            }]},
        })
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "来源区域不匹配", "lng": gj[0], "lat": gj[1],
            "source_vision_job_id": job_id, "source_vision_candidate_index": 0,
            "field_confirmation": "现场人员已核实该道路有积水并确认影响范围。",
            "end_time": time.time() + 1800,
        }, headers={"X-Admin-Token": "test-token-xyz"})

        assert response.status_code == 409
        assert response.get_json()["error"] == "vision_region_mismatch"

    def test_stale_drone_footage_cannot_be_published_as_current_road_condition(self, client, G, monkeypatch):
        import time
        from storage import database, vision_repository

        job_id = "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
        candidate = {"kind": "possible_congestion", "review_required": True, "auto_publish": False}
        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: {
            "job_id": job_id, "status": "needs_review", "captured_at": time.time() - 3600,
            "candidate_reviews": {"0": {"status": "confirmed"}},
            "result": {"candidates": [candidate]},
        })
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "过期影像事件", "lng": gj[0], "lat": gj[1],
            "source_vision_job_id": job_id, "source_vision_candidate_index": 0,
            "field_confirmation": "现场核实该路段情况并确认影响范围。",
            "end_time": time.time() + 1800,
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert response.status_code == 409
        assert response.get_json()["error"] == "vision_source_stale"
        assert rc.list_conditions() == []

    def test_dismissed_vision_cannot_publish_road_event(self, client, G, monkeypatch):
        from storage import database, vision_repository

        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: {
            "status": "needs_review", "review_status": "dismissed",
            "result": {"candidates": [{"kind": "vehicle_cluster_review"}]},
        })
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "伪造来源", "lng": gj[0], "lat": gj[1],
            "source_vision_job_id": "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e",
            "field_confirmation": "已现场核实此路段需要管制",
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert response.status_code == 409
        assert rc.list_conditions() == []

    def test_legacy_multi_candidate_review_cannot_be_attributed_to_one_candidate(self, client, G, monkeypatch):
        from storage import database, vision_repository

        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: {
            "status": "needs_review", "review_status": "confirmed",
            "review_candidate_index": None,
            "result": {"candidates": [{"kind": "possible_congestion"},
                                      {"kind": "possible_accident"}]},
        })
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "含糊来源", "lng": gj[0], "lat": gj[1],
            "source_vision_job_id": "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e",
            "field_confirmation": "经现场人员确认路段的通行情况。",
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert response.status_code == 409
        assert rc.list_conditions() == []

    @pytest.mark.parametrize("candidate", [
        {},
        {"kind": "unknown_candidate", "review_required": True, "auto_publish": False},
        {"kind": "possible_congestion", "review_required": False, "auto_publish": False},
        {"kind": "possible_congestion", "review_required": True, "auto_publish": True},
    ])
    def test_malformed_vision_candidate_cannot_become_road_event(self, client, G, monkeypatch, candidate):
        from storage import database, vision_repository

        monkeypatch.setattr(database, "initialize", lambda *_args: None)
        monkeypatch.setattr(vision_repository, "get_job", lambda *_args: {
            "status": "needs_review", "review_status": "confirmed",
            "review_candidate_index": 0,
            "result": {"candidates": [candidate]},
        })
        gj = _mid_12_gcj(G)
        response = client.post("/api/road-conditions", json={
            "type": "event", "name": "无效视觉来源", "lng": gj[0], "lat": gj[1],
            "source_vision_job_id": "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e",
            "field_confirmation": "现场人员已核实该道路当前通行情况。",
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert response.status_code == 409
        assert response.get_json()["error"] == "vision_candidate_invalid"
        assert rc.list_conditions() == []

    def test_post_too_far(self, client):
        r = client.post("/api/road-conditions", json={
            "type": "closure", "name": "湖里",
            "lng": 114.380, "lat": 30.545, "blocked_modes": ["walk"],  # 东湖里，远离路网
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert r.status_code == 400
        assert r.get_json()["error"] == "too_far_from_road"

    def test_post_requires_auth(self, client):
        r = client.post("/api/road-conditions", json={
            "type": "closure", "name": "x", "lng": 114.36, "lat": 30.53})
        assert r.status_code == 401

    def test_session_login_then_patch_end_and_delete(self, client, G):
        # 网页 session 登录
        login = client.post("/api/admin/login", json={"password": "test-pw-123"})
        assert login.status_code == 200
        csrf = login.get_json()["data"]["csrf_token"]
        headers = {"X-CSRF-Token": csrf}
        gj = _mid_12_gcj(G)
        cid = client.post("/api/road-conditions", json={
            "type": "accident", "name": "session事故",
            "lng": gj[0], "lat": gj[1], "blocked_modes": ["walk", "bike", "drive"],
        }, headers=headers).get_json()["data"]["condition"]["id"]
        # 立即结束
        r = client.patch(f"/api/road-conditions/{cid}", json={"action": "end"}, headers=headers)
        assert r.status_code == 200 and r.get_json()["data"]["condition"]["end_time"] > 0
        # 普通视图看不到，管理 all 视图能看到
        assert client.get("/api/road-conditions").get_json()["data"]["count"] == 0
        assert client.get("/api/road-conditions?all=1").get_json()["data"]["count"] == 1
        # 管理员删除后，列表和持久化状态都不再返回该限制
        assert client.delete(f"/api/road-conditions/{cid}", headers=headers).status_code == 200
        records = client.get("/api/road-conditions?all=1").get_json()["data"]["conditions"]
        assert len(records) == 0

    def test_patch_invalid_time_window_returns_400_and_keeps_record(self, client, G):
        gj = _mid_12_gcj(G)
        created = client.post("/api/road-conditions", json={
            "type": "closure", "name": "时间窗校验",
            "lng": gj[0], "lat": gj[1],
            "blocked_modes": ["walk", "bike", "drive"],
            "start_time": 2000000000, "end_time": 2000003600,
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert created.status_code == 201
        condition_id = created.get_json()["data"]["condition"]["id"]

        response = client.patch(
            f"/api/road-conditions/{condition_id}",
            json={"end_time": 1999999999},
            headers={"X-Admin-Token": "test-token-xyz"},
        )

        assert response.status_code == 400
        assert response.get_json()["error"] == "invalid_field"
        saved = client.get(
            "/api/road-conditions?all=1",
            headers={"X-Admin-Token": "test-token-xyz"},
        ).get_json()["data"]["conditions"][0]
        assert saved["end_time"] == 2000003600

    def test_public_list_shape(self, client, G):
        gj = _mid_12_gcj(G)
        client.post("/api/road-conditions", json={
            "type": "flooding", "name": "公开视图积水",
            "lng": gj[0], "lat": gj[1],
            "action": "notice", "affected_modes": ["walk"],
        }, headers={"X-Admin-Token": "test-token-xyz"})
        data = client.get("/api/road-conditions").get_json()["data"]
        assert data["count"] == 1
        cond = data["conditions"][0]
        assert cond["type_label"] == "积水"
        assert cond["edge"]["geometry_gcj"]  # 前端需要几何画线

    def test_public_road_condition_version_changes_when_active_event_is_published(self, client, G):
        before = client.get("/api/road-conditions?version_only=1").get_json()["data"]
        assert before["active_event_version"].startswith("roads-")

        lng, lat = _mid_12_gcj(G)
        published = client.post("/api/road-conditions", json={
            "type": "event", "action": "notice", "affected_modes": ["walk"],
            "name": "版本轮询验收事件", "lng": lng, "lat": lat,
        }, headers={"X-Admin-Token": "test-token-xyz"})
        assert published.status_code == 201

        after = client.get("/api/road-conditions?version_only=1").get_json()["data"]
        assert after["active_event_version"] != before["active_event_version"]
        assert "conditions" not in after
