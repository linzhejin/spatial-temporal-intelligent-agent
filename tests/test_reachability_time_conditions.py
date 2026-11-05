"""Reachable-place estimates must honor the selected road snapshot."""

import json
from types import SimpleNamespace

import networkx as nx
import pytest

import agents.tools as tools


@pytest.fixture
def reachability_network(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3630, y=30.5300)
    graph.add_node(3, x=114.3615, y=30.5310)
    graph.add_edge(1, 2, 0, length=300.0)
    graph.add_edge(1, 3, 0, length=200.0)
    graph.add_edge(3, 2, 0, length=200.0)
    pois = [
        {"id": "destination", "name": "终点食堂", "type": "dining", "subcategory": "canteen", "activities": ["canteen_meal"],
         "coordinates": {"lng": 114.3630, "lat": 30.5300}, "lat": 30.5300, "lon": 114.3630,
         "verification_status": "source_only", "source_refs": [{"source": "test fixture"}]},
        {"id": "intermediate", "name": "中途食堂", "type": "dining", "subcategory": "canteen", "activities": ["canteen_meal"],
         "coordinates": {"lng": 114.3615, "lat": 30.5310}, "lat": 30.5310, "lon": 114.3615,
         "verification_status": "source_only", "source_refs": [{"source": "test fixture"}]},
    ]
    conditions = [{"id": "slow-1", "type": "construction", "action": "slowdown",
                   "affected_modes": ["walk"], "cost_multiplier": 3.0,
                   "edge": {"u": 1, "v": 2, "key": 0}, "start_time": 0, "end_time": 0}]
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "search_by_category", lambda **_: pois)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_: list(conditions))
    return graph, conditions


def _search(**budget):
    result, artifact = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "subcategory": "canteen", "mode": "walk", **budget,
    })
    assert "error" not in result
    assert artifact["candidates"] == result["candidates"]
    return result


def test_distance_search_updates_estimated_time_when_slowdown_ends(reachability_network):
    _graph, conditions = reachability_network
    slow = _search(max_distance_m=1000)
    slow_places = {place["poi_id"]: place for place in slow["candidates"]}
    assert slow_places["destination"]["network_distance_m"] == 300
    assert slow_places["destination"]["estimated_duration_min"] == 12.0
    assert slow_places["intermediate"]["estimated_duration_min"] == 2.7

    conditions.clear()
    recovered = _search(max_distance_m=1000)
    recovered_places = {place["poi_id"]: place for place in recovered["candidates"]}
    assert recovered_places["destination"]["network_distance_m"] == 300
    assert recovered_places["destination"]["estimated_duration_min"] == 4.0
    assert recovered["road_condition_version"] != slow["road_condition_version"]


def test_time_budget_checks_fastest_route_instead_of_distance_only(reachability_network):
    _graph, conditions = reachability_network
    result = _search(max_time_min=6)
    places = {place["poi_id"]: place for place in result["candidates"]}
    assert places["destination"]["network_distance_m"] == 400
    assert places["destination"]["estimated_duration_min"] == 5.3
    assert result["path_selection_basis"] == "minimum_estimated_duration"

    too_short = _search(max_time_min=5)
    assert [place["poi_id"] for place in too_short["candidates"]] == ["intermediate"]

    conditions.clear()
    recovered = _search(max_time_min=5)
    places = {place["poi_id"]: place for place in recovered["candidates"]}
    assert places["destination"]["network_distance_m"] == 300
    assert places["destination"]["estimated_duration_min"] == 4.0


def test_time_budget_uses_real_parallel_edge_and_step_estimate(reachability_network):
    graph, conditions = reachability_network
    conditions.clear()
    graph.remove_edge(1, 2, 0)
    graph.remove_edge(1, 3, 0)
    graph.add_edge(1, 2, 0, length=100.0, highway="steps")
    graph.add_edge(1, 2, 1, length=130.0)
    result = _search(max_time_min=2)
    places = {place["poi_id"]: place for place in result["candidates"]}
    assert places["destination"]["network_distance_m"] == 130
    assert places["destination"]["estimated_duration_min"] == 1.7

    distance_result = _search(max_distance_m=120)
    places = {place["poi_id"]: place for place in distance_result["candidates"]}
    assert places["destination"]["network_distance_m"] == 100
    assert places["destination"]["estimated_duration_min"] == 3.3


def test_time_budget_includes_both_estimated_access_links(reachability_network, monkeypatch):
    graph, _conditions = reachability_network
    graph.remove_edge(1, 3, 0)
    poi = {"id": "destination", "name": "终点食堂", "type": "dining", "subcategory": "canteen",
           "coordinates": {"lng": 114.3630, "lat": 30.5304}, "lat": 30.5304, "lon": 114.3630,
           "verification_status": "source_only", "source_refs": [{"source": "test fixture"}]}
    monkeypatch.setattr(tools, "search_by_category", lambda **_: [poi])
    start = {"type": "coord", "name": "起点", "lng": 114.3595, "lat": 30.5300}

    assert not _search(start=start, max_time_min=13)["candidates"]
    candidate = _search(start=start, max_time_min=14)["candidates"][0]
    assert candidate["start_access_m"] > 40
    assert candidate["access_snap_m"] > 40
    assert candidate["estimated_duration_min"] == 13.2


def test_event_modes_and_natural_expiry_update_estimates(reachability_network, monkeypatch, tmp_path):
    from spatial import road_conditions

    _graph, conditions = reachability_network
    conditions[0].update(start_time=10, end_time=20)
    path = tmp_path / "road_conditions.json"
    path.write_text(json.dumps(conditions), encoding="utf-8")
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(path))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    clock = [15.0]
    monkeypatch.setattr(road_conditions.time, "time", lambda: clock[0])
    monkeypatch.setattr(tools, "list_conditions", road_conditions.list_conditions)

    active_walk = _search(max_distance_m=1000)
    active_bike = _search(mode="bike", max_distance_m=1000)
    walk = {place["poi_id"]: place for place in active_walk["candidates"]}
    bike = {place["poi_id"]: place for place in active_bike["candidates"]}
    assert walk["destination"]["estimated_duration_min"] == 12.0
    assert bike["destination"]["estimated_duration_min"] == 1.3

    clock[0] = 20.0
    expired_walk = _search(max_distance_m=1000)
    expired_bike = _search(mode="bike", max_distance_m=1000)
    walk = {place["poi_id"]: place for place in expired_walk["candidates"]}
    bike_after = {place["poi_id"]: place for place in expired_bike["candidates"]}
    assert walk["destination"]["estimated_duration_min"] == 4.0
    assert bike_after["destination"]["estimated_duration_min"] == bike["destination"]["estimated_duration_min"]
    assert expired_walk["road_condition_version"] != active_walk["road_condition_version"]


def test_selection_after_road_recovery_refreshes_the_real_candidate_estimates(reachability_network, monkeypatch):
    from agents import planner, route_state

    _graph, conditions = reachability_network
    monkeypatch.setattr(tools, "_data_version_for_graph", lambda _graph: "test-data")
    monkeypatch.setattr(route_state, "current_data_version", lambda *_: "test-data")
    old = _search(max_distance_m=10000)
    previous = {place["poi_id"]: place for place in old["candidates"]}
    assert previous["destination"]["estimated_duration_min"] == 12.0
    pending = {
        "request_id": "test-recovery", "root_query": "找附近食堂", "original_query": "找食堂",
        "activities": [], "subcategories": ["canteen"], "pending_slot": "selection",
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "candidate_poi_ids": list(previous), "selected_poi_id": None,
        "completed_legs": [], "follow_up_requests": [], "mode": "walk", "constraints": {},
        "result_dependencies": {"candidate_snapshot": {
            "data_version": "test-data", "road_condition_version": old["road_condition_version"],
        }},
    }
    conditions.clear()
    result = planner.run_agent("就去这里", context={
        "place_request": pending, "selected_poi_id": "destination",
    })
    assert result["response_kind"] == "candidates"
    assert result.get("route") is None
    refreshed = {place["poi_id"]: place for place in result["candidates"]}
    assert refreshed["destination"]["estimated_duration_min"] == 4.0
    assert result["place_request"]["result_dependencies"]["candidate_snapshot"]["road_condition_version"] != old["road_condition_version"]
    assert pending["result_dependencies"]["candidate_snapshot"]["road_condition_version"] == old["road_condition_version"]


def test_invalid_path_length_fails_closed_instead_of_reporting_partial_candidates(reachability_network):
    graph, _conditions = reachability_network
    graph.add_edge(1, 3, 1, length=float("nan"))
    result, artifact = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "subcategory": "canteen", "mode": "walk", "max_time_min": 10,
    })
    assert result["error"] == "reachability_failed"
    assert artifact is None


@pytest.mark.parametrize("constraints", [
    {"max_duration_s": 300},
    {"max_distance_m": 250},
    {"max_distance_m": 350, "max_duration_s": 360},
])
def test_nearby_candidates_obey_route_budget_constraints(reachability_network, constraints):
    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "subcategories": ["canteen"], "mode": "walk", "constraints": constraints,
    }, {"query": "附近食堂"})
    assert "error" not in result
    assert [place["poi_id"] for place in artifact["candidates"]] == ["intermediate"]


def test_nearby_budget_uses_a_longer_path_when_it_meets_both_limits(reachability_network):
    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "subcategories": ["canteen"], "mode": "walk",
        "constraints": {"max_distance_m": 500, "max_duration_s": 360},
    }, {"query": "附近食堂"})
    assert "error" not in result
    places = {place["poi_id"]: place for place in artifact["candidates"]}
    assert places["destination"]["network_distance_m"] == 400
    assert places["destination"]["estimated_duration_min"] == 5.3
    assert result["route_budget_scope"] == "current_leg_only"


def test_route_budget_excludes_unverified_access_links(reachability_network, monkeypatch):
    graph, conditions = reachability_network
    conditions.clear()
    graph.remove_edge(1, 3, 0)
    poi = {"id": "destination", "name": "终点食堂", "type": "dining", "subcategory": "canteen",
           "activities": ["canteen_meal"], "coordinates": {"lng": 114.3630, "lat": 30.5304},
           "lat": 30.5304, "lon": 114.3630,
           "verification_status": "source_only", "source_refs": [{"source": "test fixture"}]}
    monkeypatch.setattr(tools, "search_by_category", lambda **_: [poi])
    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3595, "lat": 30.5300},
        "subcategories": ["canteen"], "mode": "walk", "constraints": {"max_distance_m": 310},
    }, {"query": "附近食堂"})
    assert "error" not in result
    assert artifact["candidates"][0]["network_path_distance_m"] == 300
    assert artifact["candidates"][0]["network_distance_m"] > 310


def test_incomplete_budget_search_does_not_publish_a_partial_candidate_list(reachability_network, monkeypatch):
    from spatial import routing

    def incomplete(*_args, **_kwargs):
        raise routing.BudgetSearchIncomplete("test resource limit")

    monkeypatch.setattr(routing, "_budgeted_path", incomplete)
    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "subcategories": ["canteen"], "mode": "walk",
        "constraints": {"max_distance_m": 500, "max_duration_s": 360},
    }, {"query": "附近食堂"})
    assert result["error"] == "reachability_incomplete"
    assert artifact is None


def test_candidate_search_respects_exact_excluded_edge(reachability_network):
    _graph, conditions = reachability_network
    conditions.clear()
    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "subcategories": ["canteen"], "mode": "walk", "constraints": {"excluded_edge_ids": ["[1,2,0]"]},
    }, {"query": "附近食堂"})
    assert "error" not in result
    places = {place["poi_id"]: place for place in artifact["candidates"]}
    assert places["destination"]["network_distance_m"] == 400


def test_verified_accessibility_is_not_claimed_from_unverified_connectors(reachability_network):
    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "起点", "lng": 114.3600, "lat": 30.5300},
        "subcategories": ["canteen"], "mode": "walk",
        "constraints": {"require_verified_accessibility": True},
    }, {"query": "附近食堂"})
    assert result["error"] == "accessibility_unverified"
    assert artifact is None
