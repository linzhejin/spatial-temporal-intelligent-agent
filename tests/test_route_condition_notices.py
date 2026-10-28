"""Only current-mode events on the actual route may be presented as along it."""
import json
import time

import networkx as nx

from agents import tools
from api import routes
from api.routes import _agent_response_to_legacy
from spatial import road_conditions as rc


def graph():
    value = nx.MultiDiGraph()
    for node, x, y in [(1, 114.36, 30.53), (2, 114.361, 30.53),
                       (3, 114.362, 30.53), (4, 114.36, 30.531),
                       (5, 114.361, 30.531)]:
        value.add_node(node, x=x, y=y)
    for a, b in [(1, 2), (2, 3), (4, 5)]:
        value.add_edge(a, b, 0, length=100, highway="residential")
        value.add_edge(b, a, 0, length=100, highway="residential")
    return value


def event(identifier, edge, action="notice", mode="walk", *, start=None, end=None):
    now = time.time()
    return {"id": identifier, "name": identifier, "type": "event", "action": action,
            "affected_modes": [mode], "cost_multiplier": 1.8 if action == "slowdown" else None,
            "start_time": now - 100 if start is None else start,
            "end_time": now + 100 if end is None else end,
            "edge": {"u": edge[0], "v": edge[1], "key": 0, "road_name": identifier}}


def coord(node, value):
    return {"type": "coord", "name": str(node), "coordinates": {
        "lng": value.nodes[node]["x"], "lat": value.nodes[node]["y"]}}


def test_direct_route_returns_only_active_matching_on_path_notice_and_slowdown(tmp_path, monkeypatch):
    value = graph()
    now = time.time()
    conditions = [event("notice-reverse", (2, 1)), event("slow", (2, 3), "slowdown"),
                  event("neighbor", (4, 5)), event("bike-only", (1, 2), mode="bike"),
                  event("expired", (1, 2), start=now - 100, end=now - 1)]
    store = tmp_path / "events.json"
    store.write_text(json.dumps(conditions), encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(store))
    monkeypatch.setattr(rc, "_cache", None)
    snapshot = rc.list_conditions(strict=True)
    context = {"_route_graph_snapshot": value, "_route_conditions_snapshot": snapshot,
               "_route_weather_snapshot": None}
    result, artifact = tools._tool_plan_route({"start": coord(1, value), "end": coord(3, value),
        "mode": "walk", "strategy": "shortest"}, context)
    assert artifact is not None, result
    assert {item["id"] for item in result["route_events"]} == {"notice-reverse", "slow"}
    assert result["road_conditions_applied"] == 3  # Includes neighbor, excludes bike and expired.
    assert all(item["action"] in {"notice", "slowdown"} for item in result["route_events"])


def test_ordered_route_reports_events_on_all_legs_once():
    value = graph()
    snapshot = [event("first", (1, 2)), event("second", (2, 3), "slowdown")]
    snapshot[0]["source"] = {"kind": "vision_job", "captured_at": 123456,
                             "media_path": "private-media-path"}
    context = {"_route_graph_snapshot": value, "_route_conditions_snapshot": snapshot,
               "_route_weather_snapshot": None}
    result, artifact = tools._tool_plan_via_route({"start": coord(1, value), "end": coord(3, value),
        "via_points": [coord(2, value)], "mode": "walk", "strategy": "shortest"}, context)
    assert artifact is not None, result
    assert [item["id"] for item in result["route_events"]] == ["first", "second"]
    assert result["route_events"][0]["observation_time"] == 123456
    assert "private-media-path" not in json.dumps(result["route_events"])
    assert result["road_conditions_applied"] == 2


def test_crossing_road_does_not_become_an_along_route_event():
    value = graph()
    value.add_node(6, x=114.361, y=30.531)
    value.add_edge(2, 6, 0, length=100, highway="residential")
    value.add_edge(6, 2, 0, length=100, highway="residential")
    notices, count = rc.route_condition_notices(value, [event("crossing", (2, 6))], "walk",
                                                 [(1, 2, 0), (2, 3, 0)])
    assert count == 1
    assert notices == []


def test_legacy_route_conversion_preserves_route_event_field():
    route = {"recommended": [{"lat": 30.53, "lng": 114.36}],
             "route_events": [{"id": "a", "action": "notice"}]}
    result = _agent_response_to_legacy({"response_kind": "route", "route": route})
    assert result["route_events"] == route["route_events"]


def test_direct_route_api_returns_along_route_event(monkeypatch):
    from app import create_app

    value = graph()
    snapshot = [event("along", (1, 2)), event("neighbor", (4, 5))]
    monkeypatch.setattr(routes, "_ensure_network", lambda: (value, None))
    monkeypatch.setattr(routes, "list_conditions", lambda **_kwargs: snapshot)
    monkeypatch.setattr(routes, "_weather_snapshot", lambda: None)
    app = create_app()
    app.config["TESTING"] = True
    response = app.test_client().post("/api/route", json={
        "start": coord(1, value), "end": coord(3, value), "mode": "walk", "strategy": "shortest",
    })
    assert response.status_code == 200, response.get_json()
    assert [item["id"] for item in response.get_json()["data"]["route_events"]] == ["along"]


def test_tour_reports_route_events_without_repeating_a_return_path(monkeypatch):
    value = graph()
    pois = {"景点二": {"id": "p2", "name": "景点二", "type": "scenery", "subcategory": "landmark",
                       "lon": value.nodes[2]["x"], "lat": value.nodes[2]["y"]},
            "景点三": {"id": "p3", "name": "景点三", "type": "scenery", "subcategory": "landmark",
                       "lon": value.nodes[3]["x"], "lat": value.nodes[3]["y"]}}
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda name: (pois.get(name), []))
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda x, y: (x, y))
    snapshot = [event("along", (1, 2)), event("neighbor", (4, 5))]
    context = {"_route_graph_snapshot": value, "_route_conditions_snapshot": snapshot,
               "_route_weather_snapshot": None}
    result, artifact = tools._tool_plan_tour({"start": coord(1, value), "poi_names": list(pois),
        "mode": "walk", "strategy": "shortest", "loop": True}, context)
    assert artifact is not None, result
    assert [item["id"] for item in artifact["route"]["route_events"]] == ["along"]
    assert result["road_conditions_applied"] == 2


def test_multimodal_events_respect_the_mode_of_each_leg():
    value = graph()
    shared = event("shared", (1, 2))
    shared["affected_modes"] = ["walk", "bike"]
    snapshot = [shared, event("walk-only", (1, 2)), event("bike-only", (2, 3), mode="bike"),
                event("wrong-mode", (2, 3), mode="walk")]
    context = {"_route_graph_snapshot": value, "_route_conditions_snapshot": snapshot,
               "_route_weather_snapshot": None}
    result, artifact = tools._tool_plan_multimodal_route({"start": coord(1, value),
        "legs": [{"mode": "walk", "end": coord(2, value)},
                 {"mode": "bike", "end": coord(3, value)}],
        "strategy": "shortest"}, context)
    assert artifact is not None, result
    found = {item["id"]: item for item in result["route_events"]}
    assert set(found) == {"shared", "walk-only", "bike-only"}
    assert found["shared"]["travel_modes"] == ["walk"]
    assert found["bike-only"]["travel_modes"] == ["bike"]
    assert found["bike-only"]["leg_indices"] == [2]
    assert [item["id"] for item in result["legs"][0]["route_events"]] == ["shared", "walk-only"]
    assert [item["id"] for item in result["legs"][1]["route_events"]] == ["bike-only"]


def test_same_multimodal_event_merges_modes_and_segments():
    value = graph()
    shared = event("shared", (1, 2))
    shared["affected_modes"] = ["walk", "bike"]
    context = {"_route_graph_snapshot": value, "_route_conditions_snapshot": [shared],
               "_route_weather_snapshot": None}
    result, artifact = tools._tool_plan_multimodal_route({"start": coord(1, value),
        "legs": [{"mode": "walk", "end": coord(2, value)},
                 {"mode": "bike", "end": coord(1, value)}],
        "strategy": "shortest"}, context)
    assert artifact is not None, result
    assert len(result["route_events"]) == 1
    assert result["route_events"][0]["travel_modes"] == ["walk", "bike"]
    assert result["route_events"][0]["leg_indices"] == [1, 2]


def test_itinerary_retains_events_from_its_ordered_route():
    value = graph()
    context = {"_route_graph_snapshot": value,
               "_route_conditions_snapshot": [event("along", (1, 2))],
               "_route_weather_snapshot": None}
    result, artifact = tools._tool_plan_itinerary({"start": coord(1, value), "end": coord(3, value),
        "stops": [coord(2, value)], "mode": "walk", "strategy": "shortest",
        "time_budget_min": 60, "stop_duration_min": 0}, context)
    assert artifact is not None, result
    assert [item["id"] for item in artifact["route"]["route_events"]] == ["along"]
