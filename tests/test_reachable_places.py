from types import SimpleNamespace

import networkx as nx
import pytest

import agents.tools as tools


def test_find_reachable_places_uses_network_distance_and_respects_budget(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_node(3, x=114.3620, y=30.5300)
    graph.add_edge(1, 2, length=900.0)
    graph.add_edge(1, 3, length=100.0)
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    pois = [
        {"id": "near-straight-far-network", "name": "近但绕路食堂", "type": "dining",
         "subcategory": "canteen", "coordinates": {"lng": 114.3610, "lat": 30.5300},
         "lat": 30.5300, "lon": 114.3610},
        {"id": "far-straight-near-network", "name": "远但路近食堂", "type": "dining",
         "subcategory": "canteen", "coordinates": {"lng": 114.3620, "lat": 30.5300},
         "lat": 30.5300, "lon": 114.3620},
    ]
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "search_by_category", lambda **_: pois)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    result, artifact = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "当前位置", "lng": 114.36, "lat": 30.53},
        "subcategory": "canteen",
        "mode": "walk",
        "max_distance_m": 500,
    })

    assert [item["name"] for item in result["candidates"]] == ["远但路近食堂"]
    candidate = result["candidates"][0]
    assert candidate["network_distance_m"] == 100
    assert candidate["estimated_duration_min"] == 1.3
    assert result["reachability_basis"] == "network_path_plus_unverified_straight_line_access_estimate"
    assert artifact["candidates"] == result["candidates"]
    assert candidate["access_link_verified"] is False
    assert candidate["access_link_basis"] == "straight_line_to_nearest_network_node"


def test_find_reachable_places_includes_start_connector_in_ranked_distance(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_edge(1, 2, 0, length=100.0)
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    poi = {"id": "poi-2", "name": "附近食堂", "type": "dining", "subcategory": "canteen",
           "coordinates": {"lng": 114.3610, "lat": 30.5300},
           "lat": 30.5300, "lon": 114.3610}
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "search_by_category", lambda **_: [poi])
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    result, _ = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "地图起点", "lng": 114.3595, "lat": 30.5300},
        "subcategory": "canteen",
        "mode": "walk",
        "max_distance_m": 500,
    })

    candidate = result["candidates"][0]
    assert candidate["start_access_m"] == pytest.approx(48.2, abs=0.5)
    assert candidate["network_path_distance_m"] == 100
    assert candidate["access_snap_m"] == 0
    assert candidate["network_distance_m"] == round(
        candidate["start_access_m"] + candidate["network_path_distance_m"]
        + candidate["access_snap_m"]
    )


def test_find_reachable_places_rejects_conflicting_or_invalid_budgets():
    result, artifact = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "当前位置", "lng": 114.36, "lat": 30.53},
        "max_distance_m": 500,
        "max_time_min": 10,
    })

    assert result["error"] == "invalid_reachability_budget"
    assert artifact is None


def test_reachability_filters_tagged_steps_and_reports_unknown_access(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_edge(1, 2, 0, length=100.0, highway="steps")
    graph.add_edge(2, 1, 0, length=100.0, highway="steps")
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    poi = {"id": "poi-2", "name": "目的地", "type": "dining", "subcategory": "canteen",
           "coordinates": {"lng": 114.3610, "lat": 30.5300}, "lat": 30.5300, "lon": 114.3610}
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "search_by_category", lambda **_: [poi])
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    result, artifact = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "起点", "lng": 114.36, "lat": 30.53},
        "subcategory": "canteen",
        "max_distance_m": 500,
        "constraints": {"avoid_steps": True},
    })

    assert result["count"] == 0
    assert artifact["candidates"] == []


def test_reachability_discloses_unknown_step_status_for_remaining_edges(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    # The graph has no explicit highway/steps tag: it remains routable but is not verified stair-free.
    graph.add_edge(1, 2, 0, length=100.0)
    poi = {"id": "poi-2", "name": "候选地点", "type": "dining", "subcategory": "canteen",
           "coordinates": {"lng": 114.3610, "lat": 30.5300}, "lat": 30.5300, "lon": 114.3610}
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "search_by_category", lambda **_: [poi])
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    result, _ = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "起点", "lng": 114.36, "lat": 30.53},
        "subcategory": "canteen",
        "max_distance_m": 500,
        "constraints": {"avoid_steps": True},
    })

    assert result["count"] == 1
    assert result["avoid_steps_requested"] is True
    assert result["step_access_status"] == "known_tagged_steps_filtered_unknown_edges_unverified"
    assert result["remaining_step_access_verified"] is False
    assert result["known_tagged_steps_filtered"] == 0
    assert result["candidates"][0]["step_access_status"] == result["step_access_status"]


def test_reachability_schema_exposes_avoid_steps_constraint():
    schema = next(item for item in tools.TOOL_SCHEMAS
                  if item["function"]["name"] == "find_reachable_places")

    assert schema["function"]["parameters"]["properties"]["constraints"]["properties"]["avoid_steps"]["type"] == "boolean"


def test_reachability_keeps_route_supported_connector_and_excludes_longer_one(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_edge(1, 2, 0, length=100.0)
    near_poi = {
        "id": "poi-near", "name": "端点规则内地点", "type": "dining", "subcategory": "canteen",
        "coordinates": {"lng": 114.3610, "lat": 30.5306}, "lat": 30.5306, "lon": 114.3610,
    }
    far_poi = {
        "id": "poi-far", "name": "超出端点规则地点", "type": "dining", "subcategory": "canteen",
        "coordinates": {"lng": 114.3610, "lat": 30.5310}, "lat": 30.5310, "lon": 114.3610,
    }
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "search_by_category", lambda **_: [near_poi, far_poi])
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    result, artifact = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "起点", "lng": 114.36, "lat": 30.53},
        "subcategory": "canteen",
        "max_distance_m": 500,
    })

    assert result["count"] == 1
    assert result["excluded_long_access_count"] == 1
    assert result["max_access_snap_m"] == tools._MAX_ENDPOINT_ACCESS_SNAP_M
    assert artifact["candidates"][0]["poi_id"] == "poi-near"
    assert artifact["candidates"][0]["access_link_verified"] is False
    assert artifact["candidates"][0]["access_snap_m"] > 60
