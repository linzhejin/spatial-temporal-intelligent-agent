"""Preview and publication must agree on a single-direction road binding."""

import pytest

from spatial import road_conditions as rc
from spatial.coord_transform import wgs84_to_gcj02
from tests.test_road_conditions import _make_graph, _mid_12_gcj


@pytest.fixture
def road_api(tmp_path, monkeypatch):
    import config
    import api.routes as routes
    from app import create_app

    graph = _make_graph()
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "direction-test-token")
    monkeypatch.setattr(routes, "_ensure_network", lambda: (graph, None))
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(tmp_path / "events.json"))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    calls = []

    def route(_graph, start, end, **kwargs):
        calls.append((start, end, kwargs["mode"], kwargs["road_conditions"]))
        return {"recommended_length_m": 100, "duration_min": 2}

    monkeypatch.setattr(routes, "compute_route", route)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client(), graph, calls


def _snap(client, graph, direction="forward"):
    lng, lat = _mid_12_gcj(graph)
    response = client.get(
        f"/api/road-conditions/snap?lng={lng}&lat={lat}"
        f"&type=closure&action=closure&modes=walk&direction={direction}",
        headers={"X-Admin-Token": "direction-test-token"},
    )
    return response, lng, lat


def test_directional_preview_and_publication_use_the_same_binding(road_api):
    client, graph, calls = road_api
    response, lng, lat = _snap(client, graph)
    assert response.status_code == 200, response.get_json()
    data = response.get_json()["data"]
    token = data["snap"]["binding_token"]
    assert len(token) == 64
    assert data["impact_preview"]["direction"] == "forward"
    assert any(conditions and conditions[-1].get("direction") == "forward"
               for _, _, _, conditions in calls)
    created = client.post("/api/road-conditions", json={
        "type": "closure", "name": "单方向测试", "lng": lng, "lat": lat,
        "action": "closure", "affected_modes": ["walk"],
        "direction": "forward", "binding_token": token,
    }, headers={"X-Admin-Token": "direction-test-token"})
    assert created.status_code == 201, created.get_json()
    condition = created.get_json()["data"]["condition"]
    assert condition["direction"] == "forward"
    closed = rc.resolve_condition_effects(graph, [condition], "walk", strict=True)[1]
    assert bool((1, 2, 0) in closed) != bool((2, 1, 0) in closed)


def test_directional_publication_rejects_stale_binding_and_keeps_store_empty(road_api, monkeypatch):
    import api.routes as routes

    client, graph, _ = road_api
    response, lng, lat = _snap(client, graph)
    assert response.status_code == 200
    token = response.get_json()["data"]["snap"]["binding_token"]
    original = routes.snap_to_edge

    def changed_snap(*args, **kwargs):
        snap = original(*args, **kwargs)
        snap["geometry_gcj"] = list(reversed(snap["geometry_gcj"]))
        return snap

    monkeypatch.setattr(routes, "snap_to_edge", changed_snap)
    response = client.post("/api/road-conditions", json={
        "type": "closure", "name": "旧方向", "lng": lng, "lat": lat,
        "action": "closure", "affected_modes": ["walk"],
        "direction": "forward", "binding_token": token,
    }, headers={"X-Admin-Token": "direction-test-token"})
    assert response.status_code == 409
    assert response.get_json()["error"] == "road_binding_changed"
    assert rc.list_conditions(strict=True) == []


def test_direction_cannot_change_after_preview_without_a_new_binding_token(road_api):
    client, graph, _ = road_api
    response, lng, lat = _snap(client, graph, "forward")
    assert response.status_code == 200
    token = response.get_json()["data"]["snap"]["binding_token"]
    response = client.post("/api/road-conditions", json={
        "type": "closure", "name": "方向已变化", "lng": lng, "lat": lat,
        "action": "closure", "affected_modes": ["walk"],
        "direction": "reverse", "binding_token": token,
    }, headers={"X-Admin-Token": "direction-test-token"})
    assert response.status_code == 409
    assert response.get_json()["error"] == "road_binding_changed"
    assert rc.list_conditions(strict=True) == []


def test_publication_rejects_graph_change_during_direction_check(road_api, monkeypatch):
    import api.routes as routes

    client, graph, _ = road_api
    response, lng, lat = _snap(client, graph, "forward")
    assert response.status_code == 200
    token = response.get_json()["data"]["snap"]["binding_token"]
    original = routes._directional_corridor_edges

    def change_graph_after_check(*args, **kwargs):
        edges = original(*args, **kwargs)
        graph.nodes[1]["x"] += 0.0001
        return edges

    monkeypatch.setattr(routes, "_directional_corridor_edges", change_graph_after_check)
    response = client.post("/api/road-conditions", json={
        "type": "closure", "name": "核对期间图变化", "lng": lng, "lat": lat,
        "action": "closure", "affected_modes": ["walk"],
        "direction": "forward", "binding_token": token,
    }, headers={"X-Admin-Token": "direction-test-token"})
    assert response.status_code == 409
    assert response.get_json()["error"] == "road_binding_changed"
    assert rc.list_conditions(strict=True) == []


@pytest.mark.parametrize("token", [None, "测试令牌", "x" * 64, ["bad"]])
def test_directional_publication_rejects_invalid_binding_tokens(road_api, token):
    client, graph, _ = road_api
    lng, lat = _mid_12_gcj(graph)
    response = client.post("/api/road-conditions", json={
        "type": "closure", "name": "无效绑定", "lng": lng, "lat": lat,
        "action": "closure", "affected_modes": ["walk"],
        "direction": "forward", "binding_token": token,
    }, headers={"X-Admin-Token": "direction-test-token"})
    assert response.status_code == 409
    assert rc.list_conditions(strict=True) == []


def test_preview_uses_displayed_geometry_direction_for_its_sample_route(road_api):
    import api.routes as routes

    _, graph, calls = road_api
    geometry = [list(wgs84_to_gcj02(graph.nodes[node]["x"], graph.nodes[node]["y"]))
                for node in (2, 1)]
    middle = _mid_12_gcj(graph)
    snap = {"u": 1, "v": 2, "key": 0, "edges": [[1, 2, 0]],
            "geometry_gcj": geometry, "snap_lng_gcj": middle[0],
            "snap_lat_gcj": middle[1], "chain_length_m": 80}
    preview = routes._build_road_condition_impact_preview(
        graph, snap, "closure", action="closure", affected_modes=["walk"], direction="forward")
    assert preview["direction"] == "forward"
    assert calls and all((start, end) == (2, 1) for start, end, _, _ in calls)


@pytest.mark.parametrize("direction", ["sideways", ["forward"], None])
def test_invalid_direction_is_a_client_error(road_api, direction):
    client, graph, _ = road_api
    lng, lat = _mid_12_gcj(graph)
    response = client.post("/api/road-conditions", json={
        "type": "closure", "name": "非法方向", "lng": lng, "lat": lat,
        "action": "closure", "affected_modes": ["walk"], "direction": direction,
    }, headers={"X-Admin-Token": "direction-test-token"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_direction"
    assert rc.list_conditions(strict=True) == []


def test_legacy_publication_without_direction_keeps_bidirectional_behavior(road_api):
    client, graph, _ = road_api
    lng, lat = _mid_12_gcj(graph)
    response = client.post("/api/road-conditions", json={
        "type": "closure", "name": "双向兼容", "lng": lng, "lat": lat,
        "action": "closure", "affected_modes": ["walk"],
    }, headers={"X-Admin-Token": "direction-test-token"})
    assert response.status_code == 201
    condition = response.get_json()["data"]["condition"]
    closed = rc.resolve_condition_effects(graph, [condition], "walk", strict=True)[1]
    assert {(1, 2, 0), (2, 1, 0)} <= closed
