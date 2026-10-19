"""Direct-route API must audit exact edges before publishing geometry."""

import networkx as nx
from flask import Flask

import api.routes as routes
from agents.route_reuse import has_valid_reuse_certificate
import spatial.routing as routing_module


def _client(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5350)
    graph.add_node(2, x=114.3610, y=30.5350)
    graph.add_edge(1, 2, key=0, length=100.0, highway="footway",
                   slope_level=2, scenery_level=3)
    monkeypatch.setattr(routes, "_ensure_network", lambda: (graph, None))
    monkeypatch.setattr(routes, "get_nearest_node",
                        lambda _graph, lng, lat: 1 if lng < 114.3605 else 2)
    monkeypatch.setattr(routes, "list_conditions", lambda **_kw: [])
    monkeypatch.setattr(routes, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(routes, "_find_pois_along_route", lambda *_a, **_kw: [])
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "route-api-test-secret-with-enough-entropy"
    app.register_blueprint(routes.api_bp)
    return app.test_client(), graph


def _request(start_lng=114.3600, end_lng=114.3610):
    return {
        "start": {"type": "coord", "coordinates": {"lng": start_lng, "lat": 30.5350}},
        "end": {"type": "coord", "coordinates": {"lng": end_lng, "lat": 30.5350}},
        "strategy": "shortest",
        "travel_mode": "walk",
    }


def test_direct_route_response_contains_pass_summary(monkeypatch):
    client, _graph = _client(monkeypatch)
    response = client.post("/api/route", json=_request())
    assert response.status_code == 200, response.get_json()
    data = response.get_json()["data"]
    assert data["route_verification"]["status"] == "pass"
    assert data["route_verification"]["endpoint_status"] == "matched"
    assert data["route_verification"]["scope"] == "internal_graph_and_event_snapshot"
    assert data["route_verification"]["data_version_scope"] == "graph_and_local_file_content"
    assert data["route_verification"]["field_truth_verified"] is False
    assert data["route_verification"]["checked_paths"] == ["recommended", "shortest"]
    with client.application.app_context():
        assert not has_valid_reuse_certificate(data["route_state"])


def test_corrupt_edge_choice_cannot_be_published(monkeypatch):
    client, _ = _client(monkeypatch)
    original = routes.compute_route

    def wrong_edge(**kwargs):
        result = original(**kwargs)
        result["recommended_edges"] = [(1, 2, 999)]
        return result

    monkeypatch.setattr(routes, "compute_route", wrong_edge)
    response = client.post("/api/route", json=_request())
    assert response.status_code == 503
    body = response.get_json()
    assert body["error"] == "route_verification_failed"
    assert body["route_verification"]["status"] == "fail"
    assert "missing_edge" in body["route_verification"]["issues"]
    assert "data" not in body


def test_unknown_endpoint_evidence_does_not_publish_route(monkeypatch):
    client, _ = _client(monkeypatch)
    original = routes.verify_route_result

    def missing_endpoints(graph, result, *, road_conditions, **kwargs):
        return original(graph, result, road_conditions=road_conditions)

    monkeypatch.setattr(routes, "verify_route_result", missing_endpoints)
    response = client.post("/api/route", json=_request())
    assert response.status_code == 503
    body = response.get_json()
    assert body["error"] == "route_verification_unknown"
    assert body["route_verification"]["status"] == "unknown"
    assert "endpoint_unknown" in body["route_verification"]["issues"]


def test_same_snapped_node_zero_length_route_is_preserved(monkeypatch):
    client, _ = _client(monkeypatch)
    response = client.post("/api/route", json=_request(
        start_lng=114.3600, end_lng=114.3601,
    ))
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["recommended_edge_ids"] == []
    assert data["recommended_length_m"] == 0.0
    assert data["route_verification"]["status"] == "pass"


def test_corrupt_shortest_baseline_cannot_be_published(monkeypatch):
    client, _ = _client(monkeypatch)
    original = routes.compute_route

    def wrong_shortest(**kwargs):
        result = original(**kwargs)
        result["shortest_edges"] = [(1, 2, 999)]
        return result

    monkeypatch.setattr(routes, "compute_route", wrong_shortest)
    response = client.post("/api/route", json=_request())
    assert response.status_code == 503
    report = response.get_json()["route_verification"]
    assert report["status"] == "fail"
    assert report["path"] == "shortest"
    assert "missing_edge" in report["issues"]


def test_direct_route_rejects_line_shifted_away_from_selected_edge(monkeypatch):
    client, _ = _client(monkeypatch)
    original = routes._path_to_coords

    def shifted(*args, **kwargs):
        coordinates = original(*args, **kwargs)
        if coordinates:
            coordinates[0]["lng"] += 0.001
        return coordinates

    monkeypatch.setattr(routes, "_path_to_coords", shifted)
    response = client.post("/api/route", json=_request())
    assert response.status_code == 503
    report = response.get_json()["route_verification"]
    assert report["status"] == "fail"
    assert "display_geometry_mismatch" in report["issues"]


def test_direct_route_explains_hard_budget_exceeded(monkeypatch):
    client, _ = _client(monkeypatch)
    request = _request()
    request["constraints"] = {"max_distance_m": 50}

    response = client.post("/api/route", json=request)

    assert response.status_code == 404
    body = response.get_json()
    assert body["error"] == "route_not_found"
    assert "距离上限" in body["message"]


def test_direct_route_reports_search_limit_as_unknown_not_no_route(monkeypatch):
    client, _ = _client(monkeypatch)
    monkeypatch.setattr(routes, "compute_route", lambda **_kwargs: (
        (_ for _ in ()).throw(routing_module.BudgetSearchIncomplete("预算搜索超过计算上限"))
    ))
    response = client.post("/api/route", json=_request())
    assert response.status_code == 503
    body = response.get_json()
    assert body["error"] == "route_search_incomplete"


def test_direct_route_explains_unverified_accessibility(monkeypatch):
    client, _ = _client(monkeypatch)
    request = _request()
    request["constraints"] = {"require_verified_accessibility": True}

    response = client.post("/api/route", json=request)

    assert response.status_code == 503
    body = response.get_json()
    assert body["error"] == "route_verification_unknown"
    assert "accessibility_unverified" in body["route_verification"]["issues"]
    assert "无障碍证据" in body["message"]


def test_direct_route_uses_budget_compliant_shortest_instead_of_rejecting(monkeypatch):
    client, graph = _client(monkeypatch)
    graph[1][2][0].update(slope_level=5, scenery_level=1)
    graph.add_node(3, x=114.3605, y=30.5351)
    graph.add_edge(1, 3, key=0, length=70.0, highway="footway",
                   slope_level=2, scenery_level=5)
    graph.add_edge(3, 2, key=0, length=70.0, highway="footway",
                   slope_level=2, scenery_level=5)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    request = _request()
    request.update({"strategy": "scenery", "query": "想看风景，远一点没关系",
                    "constraints": {"max_distance_m": 120}})

    response = client.post("/api/route", json=request)

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["hard_budget_fallback"] is True
    assert data["recommended_length_m"] == 100.0
    assert data["recommended_edge_ids"] == [[1, 2, 0]]
    assert data["route_verification"]["status"] == "pass"


def test_direct_route_hides_shortest_reference_when_it_breaks_time_limit(monkeypatch):
    client, graph = _client(monkeypatch)
    graph[1][2][0].update(slope_level=5, scenery_level=1)
    graph.add_node(3, x=114.3605, y=30.5351)
    graph.add_edge(1, 3, key=0, length=70.0, highway="footway",
                   slope_level=2, scenery_level=5)
    graph.add_edge(3, 2, key=0, length=70.0, highway="footway",
                   slope_level=2, scenery_level=5)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    request = _request()
    request.update({"strategy": "scenery", "query": "想看风景，远一点没关系",
                    "constraints": {"max_duration_s": 120}})

    response = client.post("/api/route", json=request)

    assert response.status_code == 200, response.get_json()
    data = response.get_json()["data"]
    assert data["recommended_length_m"] == 140.0
    assert data["shortest_baseline_feasible"] is False
    assert data["route_verification"]["status"] == "pass"
