"""HTTP integration for clue collection, review, adoption and event linkage."""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest

from spatial import road_clues
from spatial import road_conditions as rc
from spatial.coord_transform import wgs84_to_gcj02

HEADERS = {"X-Admin-Token": "test-token-xyz"}


def _haversine_m(lng1, lat1, lng2, lat2):
    radius = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _make_graph():
    graph = nx.MultiDiGraph()
    points = {
        0: (114.36000, 30.53000), 1: (114.36042, 30.53000),
        2: (114.36125, 30.53000), 3: (114.36167, 30.53000),
        4: (114.36000, 30.53040), 5: (114.36042, 30.53040),
        6: (114.36125, 30.53040), 7: (114.36167, 30.53040),
        8: (114.36042, 30.52991), 9: (114.36125, 30.52991),
    }
    for node, (x, y) in points.items():
        graph.add_node(node, x=x, y=y)
    edges = [
        (0, 1, "自强大道"), (1, 2, "自强大道"), (2, 3, "自强大道"),
        (4, 5, "北环路"), (5, 6, "北环路"), (6, 7, "北环路"),
        (0, 4, ""), (3, 7, ""), (1, 8, ""), (2, 9, ""),
    ]
    for u, v, name in edges:
        length = _haversine_m(points[u][0], points[u][1], points[v][0], points[v][1])
        attrs = {"length": length, "highway": "residential"}
        if name:
            attrs["name"] = name
        graph.add_edge(u, v, 0, **attrs)
        graph.add_edge(v, u, 0, **attrs)
    return graph


@pytest.fixture
def graph():
    return _make_graph()


@pytest.fixture
def client(graph, tmp_path, monkeypatch):
    import config
    import api.routes as routes
    from app import create_app

    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "test-pw-123")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "test-token-xyz")
    conditions = tmp_path / "road_conditions.json"
    conditions.write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    monkeypatch.setattr(road_clues, "_CLUES_FILE", str(tmp_path / "clues.json"))
    road_clues.reset_collect_attempts()
    monkeypatch.setattr(routes, "_ensure_network", lambda: (graph, None))
    import agents.tools as agent_tools
    from spatial import network
    monkeypatch.setattr(network, "_G", graph)
    monkeypatch.setattr(agent_tools, "_ensure_graph", lambda: graph)
    app = create_app()
    app.config["TESTING"] = True
    yield app.test_client()


def _point_gcj(graph, u, v):
    lng = (graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2
    lat = (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2
    return wgs84_to_gcj02(lng, lat)


def _snap(client, candidate, direction="both"):
    lng = candidate.get("snap_lng_gcj")
    lat = candidate.get("snap_lat_gcj")
    response = client.get("/api/road-conditions/snap", query_string={
        "lng": lng, "lat": lat, "u": candidate["u"], "v": candidate["v"],
        "key": candidate["key"], "direction": direction,
    }, headers=HEADERS)
    assert response.status_code == 200, response.get_json()
    return response.get_json()["data"]["snap"]


def _detail(client, clue_id):
    response = client.get(f"/api/manager/clues/{clue_id}", headers=HEADERS)
    assert response.status_code == 200, response.get_json()
    return response.get_json()["data"]["clue"]


def _adopt(client, clue_id, *, action="closure", modes=("walk",), note=None,
           clue_ids=None, start=None, end=None, candidate_index=0):
    clue = _detail(client, clue_id)
    if clue["status"] == "pending":
        reviewed = client.patch(f"/api/manager/clues/{clue_id}", json={
            "status": "reviewing",
        }, headers=HEADERS)
        assert reviewed.status_code == 200, reviewed.get_json()
        clue = _detail(client, clue_id)
    candidate = clue["edge_candidates"][candidate_index]
    snap = _snap(client, candidate)
    now = time.time()
    body = {
        "type": clue["draft"].get("event_type") or "construction",
        "name": (clue["summary"] or "线索采纳")[:30],
        "description": "由线索复核采纳",
        "direction": "both",
        "action": action,
        "affected_modes": list(modes),
        "start_time": now - 1 if start is None else start,
        "end_time": now + 1800 if end is None else end,
        "review_note": note or "核对原始通知与候选道路，确认时间与影响方式。",
        "lng": snap["snap_lng_gcj"],
        "lat": snap["snap_lat_gcj"],
        "edge_ref": {"u": snap["u"], "v": snap["v"], "key": snap["key"]},
        "binding_token": snap["binding_token"],
        "clue_ids": clue_ids or [],
    }
    return client.post(f"/api/manager/clues/{clue_id}/adopt", json=body, headers=HEADERS)


def test_clue_endpoints_require_admin(client):
    assert client.get("/api/manager/clues").status_code == 401
    assert client.post("/api/manager/clues", json={}).status_code == 401
    assert client.get("/api/manager/clues/does-not-exist").status_code == 401


def test_ugc_report_converts_wgs84_and_finds_candidates(client, graph):
    lng_wgs, lat_wgs = graph.nodes[1]["x"], graph.nodes[1]["y"]
    response = client.post("/api/manager/clues", json={
        "kind": "ugc_report", "place": "自强路路面施工",
        "coord": {"lng": lng_wgs, "lat": lat_wgs, "crs": "WGS84"},
        "crs": "WGS84", "description": "路面正在施工，行人和自行车难以通过。",
    }, headers=HEADERS)
    assert response.status_code == 201, response.get_json()
    clue = response.get_json()["data"]["clues"][0]
    assert clue["status"] == "pending"
    assert clue["candidate_count"] > 0
    expected_lng, expected_lat = wgs84_to_gcj02(lng_wgs, lat_wgs)
    assert abs(clue["location"]["lng"] - expected_lng) < 1e-7
    assert abs(clue["location"]["lat"] - expected_lat) < 1e-7
    assert abs(clue["location"]["lng"] - lng_wgs) > 0.001


def test_ugc_report_rejects_unknown_crs(client):
    response = client.post("/api/manager/clues", json={
        "kind": "ugc_report", "place": "自强路",
        "coord": {"lng": 114.36, "lat": 30.53},
        "description": "未说明坐标系",
    }, headers=HEADERS)
    assert response.status_code == 400
    assert response.get_json()["error"] == "coord_crs_required"


def test_ugc_report_far_from_road_gets_no_verified_candidate(client):
    response = client.post("/api/manager/clues", json={
        "kind": "ugc_report", "place": "东湖边（远离路网）",
        "coord": {"lng": 114.38, "lat": 30.545, "crs": "GCJ02"},
        "crs": "GCJ02", "description": "远处看到施工，但附近没有可绑定道路。",
    }, headers=HEADERS)
    assert response.status_code == 201, response.get_json()
    clue = response.get_json()["data"]["clues"][0]
    assert clue["candidate_count"] == 0
    patched = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "candidate_point": {"lng": 114.38, "lat": 30.545},
    }, headers=HEADERS)
    assert patched.status_code == 400
    assert patched.get_json()["error"] == "too_far_from_road"


def test_official_notice_extraction_failure_returns_502_without_fake_clue(client, monkeypatch):
    def fail(_text):
        raise road_clues.ClueExtractionError("model down")

    monkeypatch.setattr(road_clues, "extract_notice_draft", fail)
    response = client.post("/api/manager/clues", json={
        "kind": "official_notice",
        "raw_text": "通知：自强大道施工，禁止通行，请师生绕行。",
    }, headers=HEADERS)
    assert response.status_code == 502
    assert response.get_json()["error"] == "notice_extraction_failed"
    assert road_clues.list_clues() == []


def test_official_notice_low_confidence_then_adopt_publishes_event(client, graph, monkeypatch):
    now = datetime.now(timezone.utc)
    initial_version = client.get("/api/road-conditions?version_only=1").get_json()[
        "data"]["active_event_version"]
    monkeypatch.setattr(road_clues, "extract_notice_draft", lambda _text: {
        "summary": "自强大道施工封闭",
        "confidence": 0.42,
        "draft": {
            "event_type": "construction", "road_hint": "自强大道",
            "valid_from": (now - timedelta(minutes=5)).isoformat(),
            "valid_to": (now + timedelta(hours=2)).isoformat(),
        },
    })
    created = client.post("/api/manager/clues", json={
        "kind": "official_notice",
        "raw_text": "通知：自强大道自今日起进行路面施工，施工期间禁止通行，请师生绕行。",
        "source_url": "https://example.org/notice",
    }, headers=HEADERS)
    assert created.status_code == 201, created.get_json()
    clue = created.get_json()["data"]["clues"][0]
    assert clue["status"] == "reviewing"
    assert clue["need_human"] is True
    assert clue["trust_level"] == "A"
    assert clue["candidate_count"] >= 1
    assert any(item["road_name"] == "自强大道" for item in clue["edge_candidates"])
    assert clue["raw_text"].startswith("通知")
    assert clue["source_url"] == "https://example.org/notice"
    after_create = client.get("/api/road-conditions?version_only=1").get_json()[
        "data"]["active_event_version"]
    assert after_create == initial_version

    route_payload = {
        "start": {"type": "coord", "name": "起点", "coordinates": {
            "lng": graph.nodes[0]["x"], "lat": graph.nodes[0]["y"]}},
        "end": {"type": "coord", "name": "终点", "coordinates": {
            "lng": graph.nodes[3]["x"], "lat": graph.nodes[3]["y"]}},
        "travel_mode": "walk", "strategy": "shortest",
    }
    baseline = client.post("/api/route", json=route_payload)
    assert baseline.status_code == 200, baseline.get_json()
    baseline_data = baseline.get_json()["data"]
    baseline_edges = {tuple(edge) for edge in baseline_data["shortest_edge_ids"]}

    adopted = _adopt(client, clue["clue_id"], candidate_index=1)
    assert adopted.status_code == 201, adopted.get_json()
    data = adopted.get_json()["data"]
    condition = data["condition"]
    assert condition["source"]["kind"] == "clue"
    assert condition["source"]["primary_clue_id"] == clue["clue_id"]
    assert condition["source"]["clue_ids"] == [clue["clue_id"]]
    assert data["clues"][0]["status"] == "adopted"
    assert data["clues"][0]["condition_ids"] == [condition["id"]]

    public = client.get("/api/road-conditions")
    assert public.status_code == 200
    assert any(item["id"] == condition["id"] for item in public.get_json()["data"]["conditions"])

    closed = {tuple(edge) for edge in condition["edge"]["edges"]}
    assert closed & baseline_edges
    replanned = client.post("/api/route/replan", json={
        "route_state": baseline_data["route_state"],
        "change": {"refresh_road_conditions": True},
    })
    assert replanned.status_code == 200, replanned.get_json()
    replanned_data = replanned.get_json()["data"]
    assert (replanned_data["route_state"]["road_condition_version"]
            == public.get_json()["data"]["active_event_version"])
    replanned_edges = {tuple(edge) for edge in replanned_data["shortest_edge_ids"]}
    assert not (closed & replanned_edges)

    repeated = _adopt(client, clue["clue_id"], candidate_index=1)
    assert repeated.status_code == 200
    assert repeated.get_json()["data"]["already_adopted"] is True
    stored = rc.list_conditions(include_inactive=True, strict=True)
    assert len(stored) == 1

    route = client.post("/api/route", json=route_payload)
    assert route.status_code == 200, route.get_json()
    route_edges = {tuple(edge) for edge in route.get_json()["data"]["shortest_edge_ids"]}
    assert not (closed & route_edges)


def test_vision_clue_creation_and_adoption(client, graph, monkeypatch):
    from storage import database, vision_repository
    import api.routes as routes

    candidate = {
        "kind": "possible_flooding", "review_required": True, "auto_publish": False,
        "reason": "路面出现疑似积水区域，需要人工核对。",
        "confidence": 0.7,
    }
    job = {
        "job_id": "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e",
        "purpose": "operational", "media_kind": "image", "status": "needs_review",
        "captured_at": datetime.now(timezone.utc),
        "result": {"candidates": [candidate]},
        "anchor_gcj": {"lng": 114.36, "lat": 30.53},
    }
    monkeypatch.setattr(database, "initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vision_repository, "get_job", lambda *_args, **_kwargs: dict(job))
    lng, lat = _point_gcj(graph, 1, 2)
    monkeypatch.setattr(routes, "_vision_suggested_edges_for_job", lambda _job, _graph: {
        "suggested_edges": [{
            "u": 1, "v": 2, "key": 0, "road_name": "自强大道", "dist_m": 1.0,
            "snap_lng_gcj": lng, "snap_lat_gcj": lat,
            "geometry_gcj": [[lng, lat], [lng + 0.0001, lat]],
            "edges": [[1, 2, 0]], "chain_length_m": 80.0, "score": 0.6,
        }],
    })
    created = client.post("/api/manager/clues/from-vision", json={
        "job_id": job["job_id"], "candidate_index": 0,
    }, headers=HEADERS)
    assert created.status_code == 201, created.get_json()
    data = created.get_json()["data"]
    assert data["publishable"] is False
    assert "候选尚未单独确认" in data["publish_blockers"]
    clue = data["clues"][0]
    assert clue["kind"] == "vision"
    assert clue["trust_level"] == "B"
    assert clue["draft"]["event_type"] == "flooding"
    assert clue["candidate_count"] == 1
    blocked = _adopt(client, clue["clue_id"])
    assert blocked.status_code == 409
    assert blocked.get_json()["error"] == "clue_source_not_publishable"
    invalid = client.post("/api/manager/clues/from-vision", json={
        "job_id": job["job_id"], "candidate_index": 5,
    }, headers=HEADERS)
    assert invalid.status_code == 409
    assert invalid.get_json()["error"] == "vision_candidate_invalid"

    job["candidate_reviews"] = {
        "0": {"status": "confirmed", "reviewed_by": "tester",
              "reviewed_at": datetime.now(timezone.utc).isoformat()},
    }
    clue_detail = _detail(client, clue["clue_id"])
    candidate_snap = _snap(client, clue_detail["edge_candidates"][0])
    base = {
        "type": "flooding", "name": "影像线索",
        "lng": candidate_snap["snap_lng_gcj"], "lat": candidate_snap["snap_lat_gcj"],
        "edge_ref": {"u": candidate_snap["u"], "v": candidate_snap["v"],
                     "key": candidate_snap["key"]},
        "binding_token": candidate_snap["binding_token"],
        "review_note": "核对影像候选与道路绑定，确认积水影响范围。",
    }
    no_action = client.post(f"/api/manager/clues/{clue['clue_id']}/adopt", json={
        **base, "blocked_modes": ["walk"], "end_time": time.time() + 1800,
    }, headers=HEADERS)
    assert no_action.status_code == 400
    no_end = client.post(f"/api/manager/clues/{clue['clue_id']}/adopt", json={
        **base, "action": "closure", "affected_modes": ["walk"],
    }, headers=HEADERS)
    assert no_end.status_code == 400

    adopted = _adopt(client, clue["clue_id"])
    assert adopted.status_code == 201, adopted.get_json()
    condition = adopted.get_json()["data"]["condition"]
    assert condition["source"]["kind"] == "clue"
    assert 8 <= len(condition["source"]["field_confirmation"]) <= 300


def test_vision_trial_and_replay_cannot_be_adopted(client, graph, monkeypatch):
    from storage import database, vision_repository
    import api.routes as routes

    candidate = {
        "kind": "possible_flooding", "review_required": True, "auto_publish": False,
        "reason": "疑似积水", "confidence": 0.6,
    }
    base_job = {
        "job_id": "b7f6c6b1-d237-43fa-8ea2-b01ae48a3e8e",
        "purpose": "trial", "media_kind": "image",
        "captured_at": datetime.now(timezone.utc),
        "result": {"candidates": [candidate]},
    }
    jobs = {}

    def get_job(_url, job_id):
        return dict(jobs.get(job_id))

    monkeypatch.setattr(database, "initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vision_repository, "get_job", get_job)
    lng, lat = _point_gcj(graph, 1, 2)
    monkeypatch.setattr(routes, "_vision_suggested_edges_for_job", lambda _job, _graph: {
        "suggested_edges": [{
            "u": 1, "v": 2, "key": 0, "road_name": "自强大道", "dist_m": 1.0,
            "snap_lng_gcj": lng, "snap_lat_gcj": lat,
            "geometry_gcj": [[lng, lat], [lng + 0.0001, lat]],
            "edges": [[1, 2, 0]], "chain_length_m": 80.0,
        }],
    })
    jobs[base_job["job_id"]] = dict(base_job)
    created = client.post("/api/manager/clues/from-vision", json={
        "job_id": base_job["job_id"], "candidate_index": 0,
    }, headers=HEADERS)
    assert created.status_code == 201
    data = created.get_json()["data"]
    assert data["publishable"] is False
    assert "试用影像" in data["publish_blockers"]
    rejected = _adopt(client, data["clues"][0]["clue_id"])
    assert rejected.status_code == 409
    assert rejected.get_json()["error"] == "clue_source_not_publishable"

    replay_job = dict(base_job)
    replay_job["job_id"] = "c7f6c6b1-d237-43fa-8ea2-b01ae48a3e8e"
    replay_job["purpose"] = "operational"
    replay_job["result"] = {"candidates": [candidate], "research_replay": {"review_origin": "scripted"}}
    jobs[replay_job["job_id"]] = replay_job
    replay = client.post("/api/manager/clues/from-vision", json={
        "job_id": replay_job["job_id"], "candidate_index": 0,
    }, headers=HEADERS)
    assert replay.status_code == 201
    replay_data = replay.get_json()["data"]
    assert replay_data["publishable"] is False
    assert "历史回放" in replay_data["publish_blockers"]


def test_delete_event_restores_clue_to_reviewing(client, monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(road_clues, "extract_notice_draft", lambda _text: {
        "summary": "自强大道施工", "confidence": 0.9,
        "draft": {"event_type": "construction", "road_hint": "自强大道"},
    })
    created = client.post("/api/manager/clues", json={
        "kind": "official_notice", "raw_text": "自强大道因施工临时封闭，请绕行。",
    }, headers=HEADERS)
    clue = created.get_json()["data"]["clues"][0]
    adopted = _adopt(client, clue["clue_id"])
    condition_id = adopted.get_json()["data"]["condition"]["id"]
    deleted = client.delete(f"/api/road-conditions/{condition_id}", headers=HEADERS)
    assert deleted.status_code == 200
    restored = _detail(client, clue["clue_id"])
    assert restored["status"] == "reviewing"
    assert restored["condition_ids"] == []


def test_patch_rejects_privileged_fields_and_reject_requires_reason(client):
    created = client.post("/api/manager/clues", json={
        "kind": "ugc_report", "place": "桂园一路",
        "description": "路面有小范围积水。", "crs": "GCJ02",
    }, headers=HEADERS)
    clue = created.get_json()["data"]["clues"][0]
    forbidden = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "trust_level": "A",
    }, headers=HEADERS)
    assert forbidden.status_code == 400
    adopted = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "status": "adopted",
    }, headers=HEADERS)
    assert adopted.status_code == 400
    missing = client.post(f"/api/manager/clues/{clue['clue_id']}/reject", json={},
                          headers=HEADERS)
    assert missing.status_code == 400
    rejected = client.post(f"/api/manager/clues/{clue['clue_id']}/reject", json={
        "reason": "现场核对后未发现积水。",
    }, headers=HEADERS)
    assert rejected.status_code == 200
    assert rejected.get_json()["data"]["clue"]["status"] == "rejected"
    ended_edit = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "summary": "修改已驳回线索",
    }, headers=HEADERS)
    assert ended_edit.status_code == 400


def test_corrupt_clue_store_returns_503(client):
    with open(road_clues._CLUES_FILE, "w", encoding="utf-8") as stream:
        stream.write("{")
    response = client.get("/api/manager/clues", headers=HEADERS)
    assert response.status_code == 503
    assert response.get_json()["error"] == "clue_store_unavailable"


def test_adopt_api_recovers_event_written_before_clue_writback(client, graph):
    lng, lat = _point_gcj(graph, 1, 2)
    created = client.post("/api/manager/clues", json={
        "kind": "ugc_report", "place": "自强大道",
        "coord": {"lng": lng, "lat": lat, "crs": "GCJ02"},
        "description": "路面施工，行人难以通过。", "crs": "GCJ02",
    }, headers=HEADERS)
    clue = created.get_json()["data"]["clues"][0]
    reviewed = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "status": "reviewing",
    }, headers=HEADERS)
    assert reviewed.status_code == 200
    detail = _detail(client, clue["clue_id"])
    snap = _snap(client, detail["edge_candidates"][0])
    token = "api-crash-recovery-token"
    condition = rc.add_condition(
        cond_type="construction", name="已写入但未回写",
        edge=snap, start_time=time.time() - 1, end_time=time.time() + 1800,
        action="closure", affected_modes=["walk"],
        source={"kind": "clue", "clue_ids": [clue["clue_id"]],
                "primary_clue_id": clue["clue_id"], "adoption_token": token},
    )
    rows = road_clues._load()
    for row in rows:
        if row["clue_id"] == clue["clue_id"]:
            row["adoption"] = {"token": token, "started_at": time.time()}
    road_clues._save(rows)

    response = client.post(f"/api/manager/clues/{clue['clue_id']}/adopt", json={
        "type": "construction", "name": "重试采纳",
        "action": "closure", "affected_modes": ["walk"],
        "start_time": time.time() - 1, "end_time": time.time() + 1800,
        "review_note": "模拟事件已写入但线索尚未回写后的 API 重试。",
        "lng": snap["snap_lng_gcj"], "lat": snap["snap_lat_gcj"],
        "edge_ref": {"u": snap["u"], "v": snap["v"], "key": snap["key"]},
        "binding_token": snap["binding_token"],
    }, headers=HEADERS)
    assert response.status_code == 200, response.get_json()
    data = response.get_json()["data"]
    assert data["already_adopted"] is True
    assert data["condition"]["id"] == condition["id"]
    assert len(rc.list_conditions(include_inactive=True, strict=True)) == 1


def test_concurrent_adoption_creates_one_event(client, monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(road_clues, "extract_notice_draft", lambda _text: {
        "summary": "自强大道施工", "confidence": 0.9,
        "draft": {"event_type": "construction", "road_hint": "自强大道"},
    })
    created = client.post("/api/manager/clues", json={
        "kind": "official_notice", "raw_text": "自强大道施工，禁止通行。",
    }, headers=HEADERS)
    clue = created.get_json()["data"]["clues"][0]

    results = []

    def adopt_once():
        local = client.application.test_client()
        response = _adopt(local, clue["clue_id"])
        results.append(response.status_code)

    threads = [threading.Thread(target=adopt_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(results) == [200, 201]
    assert len(rc.list_conditions(include_inactive=True, strict=True)) == 1


def test_web_clue_can_get_a_verified_candidate_on_the_map(client, graph, monkeypatch):
    monkeypatch.setattr(road_clues.registry["web"], "fetch", lambda **_kwargs: [{
        "kind": "web", "summary": "网页提到自强大道施工", "raw_text": "网页正文",
        "source_url": "https://example.org/road", "source_label": "网页检索",
        "source_tags": ["web"], "search_query": "自强大道施工",
        "draft": {"event_type": "construction"}, "edge_candidates": [],
        "confidence": 0.3,
    }])
    created = client.post("/api/manager/clues", json={
        "kind": "web", "query": "自强大道施工",
    }, headers=HEADERS)
    assert created.status_code == 201, created.get_json()
    clue = created.get_json()["data"]["clues"][0]
    assert clue["kind"] == "web"
    assert clue["trust_level"] == "C"
    assert clue["candidate_count"] == 0

    lng, lat = _point_gcj(graph, 1, 2)
    patched = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "candidate_point": {"lng": lng, "lat": lat},
    }, headers=HEADERS)
    assert patched.status_code == 200, patched.get_json()
    updated = patched.get_json()["data"]["clue"]
    assert updated["candidate_count"] == 1
    assert updated["edge_candidates"][0]["road_name"] == "自强大道"
    adopted = _adopt(client, clue["clue_id"])
    assert adopted.status_code == 201, adopted.get_json()

    def unavailable(**_kwargs):
        raise road_clues.ClueSourceUnavailable("provider down")

    monkeypatch.setattr(road_clues.registry["web"], "fetch", unavailable)
    failed = client.post("/api/manager/clues", json={
        "kind": "web", "query": "自强大道施工",
    }, headers=HEADERS)
    assert failed.status_code == 503
    assert failed.get_json()["error"] == "clue_source_unavailable"


def test_official_notice_screenshots_use_the_configured_image_channel(client, monkeypatch):
    import config
    from spatial import road_clues as clue_module

    unavailable = client.post("/api/manager/clues", json={
        "kind": "official_notice", "image_base64": "AAAA",
    }, headers=HEADERS)
    assert unavailable.status_code == 503
    assert unavailable.get_json()["error"] == "image_extraction_unavailable"

    local_ref = client.post("/api/manager/clues", json={
        "kind": "official_notice", "media_ref": "/tmp/notice.png",
    }, headers=HEADERS)
    assert local_ref.status_code == 400
    assert local_ref.get_json()["error"] == "image_extraction_unavailable"

    invalid_url = client.post("/api/manager/clues", json={
        "kind": "official_notice", "raw_text": "通知：自强大道施工，禁止通行。",
        "source_url": "javascript:alert(1)",
    }, headers=HEADERS)
    assert invalid_url.status_code == 400
    assert invalid_url.get_json()["error"] == "invalid_source_url"

    monkeypatch.setattr(config, "NOTICE_VISION_ENABLED", True)
    monkeypatch.setattr(
        clue_module, "extract_notice_draft_from_image",
        lambda ref: {"summary": "截图识别：自强大道施工",
                     "confidence": 0.72,
                     "draft": {"event_type": "construction", "road_hint": "自强大道",
                               "action": "closure", "affected_modes": ["walk"]}},
    )
    extracted = client.post("/api/manager/clues", json={
        "kind": "official_notice", "image_url": "https://example.org/notice.png",
    }, headers=HEADERS)
    assert extracted.status_code == 201, extracted.get_json()
    clue = extracted.get_json()["data"]["clues"][0]
    assert clue["status"] == "reviewing"
    assert clue["raw_text"] == ""
    assert "screenshot_extracted" in clue["source_tags"]
    assert clue["draft"]["road_hint"] == "自强大道"
    assert clue["condition_ids"] == []


def test_official_notice_manual_entry_still_works_without_an_image_channel(client):
    screenshot = client.post("/api/manager/clues", json={
        "kind": "official_notice", "manual": True,
        "media_ref": "https://example.org/notice.png",
        "summary": "通知截图待人工核对",
        "draft": {"event_type": "construction", "road_hint": "自强大道"},
    }, headers=HEADERS)
    assert screenshot.status_code == 201, screenshot.get_json()
    screenshot_clue = screenshot.get_json()["data"]["clues"][0]
    assert screenshot_clue["status"] == "reviewing"
    assert screenshot_clue["raw_text"] == ""
    assert screenshot_clue["media_ref"] == "https://example.org/notice.png"
    assert "screenshot_reference" in screenshot_clue["source_tags"]

    manual = client.post("/api/manager/clues", json={
        "kind": "official_notice", "manual": True,
        "raw_text": "通知：自强大道施工，禁止通行，请师生绕行。",
        "summary": "自强大道施工（人工补录）",
        "draft": {"event_type": "construction", "road_hint": "自强大道"},
    }, headers=HEADERS)
    assert manual.status_code == 201, manual.get_json()
    clue = manual.get_json()["data"]["clues"][0]
    assert clue["status"] == "reviewing"
    assert clue["need_human"] is True
    assert clue["raw_text"].startswith("通知")
    assert clue["source_url"] is None


def test_session_auth_csrf_and_privileged_patch_fields(client, graph):
    login = client.post("/api/admin/login", json={"password": "test-pw-123"})
    assert login.status_code == 200
    csrf = login.get_json()["data"]["csrf_token"]
    assert client.get("/api/manager/clues").status_code == 200

    created = client.post("/api/manager/clues", json={
        "kind": "ugc_report", "place": "自强大道",
        "description": "现场看到路面施工。", "crs": "GCJ02",
        "trust_level": "A",
    }, headers={"X-CSRF-Token": csrf})
    assert created.status_code == 201, created.get_json()
    clue = created.get_json()["data"]["clues"][0]
    assert clue["trust_level"] == "B"

    no_csrf = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "summary": "改摘要",
    })
    assert no_csrf.status_code == 403

    for forbidden in ("condition_ids", "confidence", "trust_level", "kind"):
        response = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
            forbidden: [] if forbidden == "condition_ids" else 0.9,
        }, headers={"X-CSRF-Token": csrf})
        assert response.status_code == 400, (forbidden, response.get_json())

    too_long = client.patch(f"/api/manager/clues/{clue['clue_id']}", json={
        "summary": "长" * 61,
    }, headers={"X-CSRF-Token": csrf})
    assert too_long.status_code == 400


def test_expired_clue_is_not_adoptable(client, graph):
    clue = road_clues.create_clue({
        "kind": "ugc_report", "summary": "已过期施工",
        "valid_from": time.time() - 7200, "valid_to": time.time() - 3600,
        "draft": {"event_type": "construction"},
        "edge_candidates": [{
            "u": 1, "v": 2, "key": 0, "road_name": "自强大道", "distance_m": 1.0,
        }],
    }, actor="test")
    assert road_clues.get_clue(clue["clue_id"])["status"] == "expired"
    lng, lat = _point_gcj(graph, 1, 2)
    snap_response = client.get("/api/road-conditions/snap", query_string={
        "lng": lng, "lat": lat, "u": 1, "v": 2, "key": 0,
    }, headers=HEADERS)
    assert snap_response.status_code == 200
    snap = snap_response.get_json()["data"]["snap"]
    response = client.post(f"/api/manager/clues/{clue['clue_id']}/adopt", json={
        "type": "construction", "name": "过期线索",
        "action": "closure", "affected_modes": ["walk"],
        "start_time": time.time() - 1, "end_time": time.time() + 1800,
        "review_note": "核对后尝试采纳一条已经过期的线索。",
        "lng": snap["snap_lng_gcj"], "lat": snap["snap_lat_gcj"],
        "edge_ref": {"u": snap["u"], "v": snap["v"], "key": snap["key"]},
        "binding_token": snap["binding_token"],
    }, headers=HEADERS)
    assert response.status_code == 400
