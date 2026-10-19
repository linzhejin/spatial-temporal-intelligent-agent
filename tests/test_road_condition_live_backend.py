"""Local Flask integration checks for persistent road-event publication."""

from __future__ import annotations

import json
import time

import pytest

from spatial import network, road_conditions


@pytest.fixture
def live_backend_client(tmp_path, monkeypatch):
    """Use the real cached campus graph and a per-test road-event store."""
    import config
    import api.routes as routes
    from app import create_app

    store = tmp_path / "road_conditions.json"
    store.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(store))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "local-integration-test-only")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    monkeypatch.setattr(routes, "_network_initialized", False)
    monkeypatch.setattr(network, "_G", None)

    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client(), store


def test_admin_event_persists_and_is_consumed_by_user_route(live_backend_client):
    client, store = live_backend_client

    login = client.post(
        "/api/admin/login",
        json={"password": "local-integration-test-only"},
    )
    assert login.status_code == 200
    csrf = login.get_json()["data"]["csrf_token"]

    click_lng, click_lat = 114.363, 30.5365
    snap_response = client.get(
        f"/api/road-conditions/snap?lng={click_lng}&lat={click_lat}",
    )
    assert snap_response.status_code == 200
    snapped_edge = snap_response.get_json()["data"]["snap"]
    graph = network.get_network()
    start_node, end_node = snapped_edge["u"], snapped_edge["v"]

    def endpoint(node_id):
        return {
            "type": "coord",
            "coordinates": {
                "lng": float(graph.nodes[node_id]["x"]),
                "lat": float(graph.nodes[node_id]["y"]),
            },
        }

    baseline = client.post(
        "/api/route",
        json={
            "start": endpoint(start_node),
            "end": endpoint(end_node),
            "travel_mode": "walk",
            "strategy": "shortest",
        },
    )
    assert baseline.status_code == 200, baseline.get_json()
    baseline_data = baseline.get_json()["data"]
    target_edges = {tuple(edge) for edge in snapped_edge["edges"]}
    assert target_edges.intersection(tuple(edge) for edge in baseline_data["shortest_edge_ids"])

    publish = client.post(
        "/api/road-conditions",
        json={
            "type": "event",
            "name": "本地联动验收封路",
            "description": "仅写入 pytest 临时目录",
            "action": "closure",
            "affected_modes": ["walk"],
            "lng": click_lng,
            "lat": click_lat,
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert publish.status_code == 201
    event = publish.get_json()["data"]["condition"]
    assert event["edge"]["u"] == snapped_edge["u"]
    assert event["edge"]["v"] == snapped_edge["v"]
    assert json.loads(store.read_text(encoding="utf-8"))[0]["id"] == event["id"]

    public_snapshot = client.get("/api/road-conditions")
    assert public_snapshot.status_code == 200
    snapshot_data = public_snapshot.get_json()["data"]
    assert snapshot_data["count"] == 1
    assert snapshot_data["conditions"][0]["id"] == event["id"]

    route = client.post(
        "/api/route/replan",
        json={
            "route_state": baseline_data["route_state"],
            "change": {"refresh_road_conditions": True},
        },
    )
    assert route.status_code == 200, route.get_json()
    route_data = route.get_json()["data"]
    assert route_data["route_state"]["road_condition_version"] == snapshot_data["active_event_version"]
    closed_ids = {tuple(edge) for edge in event["edge"]["edges"]}
    route_ids = {tuple(edge) for edge in route_data["shortest_edge_ids"]}
    assert not closed_ids.intersection(route_ids)


@pytest.mark.parametrize(
    ("action", "cost_multiplier"),
    [("slowdown", 1.8), ("notice", None)],
)
def test_live_route_applies_slowdown_and_notice_without_closing_the_road(
    live_backend_client, action, cost_multiplier,
):
    client, _store = live_backend_client
    login = client.post(
        "/api/admin/login",
        json={"password": "local-integration-test-only"},
    )
    assert login.status_code == 200
    csrf = login.get_json()["data"]["csrf_token"]

    click_lng, click_lat = 114.363, 30.5365
    snap_response = client.get(
        f"/api/road-conditions/snap?lng={click_lng}&lat={click_lat}",
    )
    assert snap_response.status_code == 200
    snapped_edge = snap_response.get_json()["data"]["snap"]
    graph = network.get_network()
    start_node, end_node = snapped_edge["u"], snapped_edge["v"]

    def endpoint(node_id):
        return {
            "type": "coord",
            "coordinates": {
                "lng": float(graph.nodes[node_id]["x"]),
                "lat": float(graph.nodes[node_id]["y"]),
            },
        }

    baseline = client.post(
        "/api/route",
        json={
            "start": endpoint(start_node),
            "end": endpoint(end_node),
            "travel_mode": "walk",
            "strategy": "shortest",
        },
    )
    assert baseline.status_code == 200, baseline.get_json()
    baseline_data = baseline.get_json()["data"]
    target_edges = {tuple(edge) for edge in snapped_edge["edges"]}
    assert target_edges.intersection(
        tuple(edge) for edge in baseline_data["shortest_edge_ids"]
    )

    publish = client.post(
        "/api/road-conditions",
        json={
            "type": "event",
            "name": f"本地联动验收{action}",
            "description": "仅写入 pytest 临时目录",
            "action": action,
            "affected_modes": ["walk"],
            "cost_multiplier": cost_multiplier,
            "lng": click_lng,
            "lat": click_lat,
            "end_time": time.time() + 3600,
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert publish.status_code == 201, publish.get_json()
    event = publish.get_json()["data"]["condition"]

    route = client.post(
        "/api/route/replan",
        json={
            "route_state": baseline_data["route_state"],
            "change": {"refresh_road_conditions": True},
        },
    )
    assert route.status_code == 200, route.get_json()
    route_data = route.get_json()["data"]
    route_edges = {tuple(edge) for edge in route_data["shortest_edge_ids"]}
    assert target_edges.intersection(route_edges)
    assert event["affected_modes"] == ["walk"]
    if action == "slowdown":
        assert route_data["duration_min"] > baseline_data["duration_min"]
    else:
        assert route_edges == {tuple(edge) for edge in baseline_data["shortest_edge_ids"]}
        assert route_data["duration_min"] == baseline_data["duration_min"]

