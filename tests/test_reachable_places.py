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
    monkeypatch.setattr(tools, "has_source_backed_record", lambda _poi: True)
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
    monkeypatch.setattr(tools, "has_source_backed_record", lambda _poi: True)
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


def test_model_dialogue_route_pool_keeps_valid_live_amap_places(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_edge(1, 2, length=100.0)
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    valid = {"id": "amap:A123", "name": "在线火锅店", "type": "dining",
             "subcategory": "restaurant", "source": "amap_online",
             "coordinate_system": "GCJ-02",
             "coordinates": {"lng": 114.361, "lat": 30.53},
             "lat": 30.53, "lon": 114.361}
    next_valid = {**valid, "id": "amap:A124", "poi_id": "amap:A124", "name": "下一批在线火锅店"}
    invalid = {**valid, "id": "amap:B123", "name": "坐标系不明的店", "coordinate_system": "unknown"}
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "信图", None))
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_: {"snap_distance_m": 0})
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    result, _artifact = tools._tool_find_reachable_places(
        {"start": {"type": "poi", "name": "信图"}, "subcategories": ["restaurant"],
         "mode": "walk", "max_distance_m": 500},
        {"query": "信图附近火锅", "_model_dialogue": True,
         "_candidate_pool": [valid, next_valid, invalid], "_excluded_candidate_ids": ["amap:A123"],
         "_route_graph_snapshot": graph,
         "_route_conditions_snapshot": []},
    )

    assert [row["poi_id"] for row in result["candidates"]] == ["amap:A124"]


def test_provider_sport_type_evidence_survives_candidate_delivery_to_model():
    import json
    from agents.dialogue import _model_tool_content

    venue = {"id": "amap:SPORT123", "name": "测试体育场所", "type": "sports", "subcategory": "court",
             "type_name": "体育休闲服务;运动场馆;羽毛球场", "type_code": "080118",
             "source": "amap_online", "verification_status": "provider_only"}
    public = tools._poi_public(venue, with_provenance=True)
    saved = json.loads(json.dumps(public, ensure_ascii=False))
    delivered = json.loads(_model_tool_content("search_poi_candidates", {"candidates": [saved]}))["candidates"][0]

    assert delivered["type_name"] == venue["type_name"]
    assert delivered["activities"] == ["badminton"]
    assert "地图" in delivered["activity_basis"]
    assert delivered["verification_status"] == "provider_only"


def test_generic_provider_court_does_not_invent_a_specific_sport():
    venue = {"id": "amap:SPORT456", "name": "测试体育场所", "type": "sports", "subcategory": "court",
             "type_name": "体育休闲服务;运动场馆", "source": "amap_online", "verification_status": "provider_only"}
    public = tools._poi_public(venue, with_provenance=True)
    assert public["activities"] == []


def test_model_sport_distance_search_includes_online_candidates_for_followup(monkeypatch):
    from agents import discovery

    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_node(3, x=114.3620, y=30.5300)
    graph.add_edge(1, 2, length=100.0)
    graph.add_edge(1, 3, length=150.0)
    local = {"id": "poi-local", "name": "校园篮球场", "type": "sports", "subcategory": "field",
             "activities": ["basketball"], "coordinates": {"lng": 114.361, "lat": 30.53},
             "lat": 30.53, "lon": 114.361}
    online = {"id": "amap:SPORT123", "name": "地图篮球场", "type": "sports", "subcategory": "court",
              "type_name": "体育休闲服务;运动场馆;篮球场", "type_code": "080104",
              "source": "amap_online", "verification_status": "provider_only", "coordinate_system": "GCJ-02",
              "coordinates": {"lng": 114.362, "lat": 30.53}, "lat": 30.53, "lon": 114.362}
    duplicate = {**online, "id": "amap:DUP", "name": local["name"],
                 "coordinates": local["coordinates"], "lat": local["lat"], "lon": local["lon"]}
    distant_same_name = {**online, "id": "amap:FAR", "name": local["name"]}
    provider_calls = []
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: SimpleNamespace(
        for_mode=lambda _mode: SimpleNamespace(graph=graph, mode_penalty={})))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "_endpoint_target_wgs", lambda *_: (114.36, 30.53, "起点", None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_: {"snap_distance_m": 0})
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(discovery, "wgs84_to_gcj02", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [local])
    monkeypatch.setattr(discovery, "has_source_backed_record", lambda _poi: True)
    monkeypatch.setattr(tools, "has_source_backed_record", lambda _poi: True)
    monkeypatch.setattr(discovery.amap_poi, "search_nearby", lambda _center, **kwargs: (
        provider_calls.append(kwargs) or {"status": "ok", "candidates": [duplicate, online, distant_same_name]}))

    result, _ = tools._tool_find_reachable_places(
        {"start": {"type": "poi", "name": "起点"}, "mode": "walk", "activities": ["basketball"],
         "max_distance_m": 500, "constraints": {"avoid_steps": True}},
        {"query": "", "_model_dialogue": True, "_semantic_request": {"operation": "place_search"},
         "_route_graph_snapshot": graph, "_route_conditions_snapshot": []},
    )

    assert {row["poi_id"] for row in result["candidates"]} == {"poi-local", "amap:SPORT123", "amap:FAR"}
    assert provider_calls and provider_calls[0]["poi_type"] == "sports"
    assert result["online_status"] == "ok"


def test_find_reachable_places_rejects_conflicting_or_invalid_budgets():
    result, artifact = tools.execute_tool("find_reachable_places", {
        "start": {"type": "coord", "name": "当前位置", "lng": 114.36, "lat": 30.53},
        "max_distance_m": 500,
        "max_time_min": 10,
    })

    assert result["error"] == "invalid_reachability_budget"
    assert artifact is None


def _dual_budget_graph(monkeypatch, *, short_minutes):
    import spatial.routing as routing
    graph = nx.MultiDiGraph()
    for node, lng in ((1, 114.360), (2, 114.362), (3, 114.361)):
        graph.add_node(node, x=lng, y=30.530)
    graph.add_edge(1, 2, key=0, length=700.0, minutes=4.0)
    graph.add_edge(1, 3, key=0, length=200.0, minutes=short_minutes / 2)
    graph.add_edge(3, 2, key=0, length=200.0, minutes=short_minutes / 2)
    poi = {"id": "poi-dual", "name": "候选地点", "type": "dining", "subcategory": "canteen",
           "coordinates": {"lng": 114.362, "lat": 30.530}, "lat": 30.530, "lon": 114.362}
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: SimpleNamespace(
        for_mode=lambda _mode: SimpleNamespace(graph=graph, mode_penalty={})))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (1, "起点", None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_: {"snap_distance_m": 0})
    monkeypatch.setattr(tools, "has_source_backed_record", lambda _poi: True)
    monkeypatch.setattr(tools, "search_by_category", lambda **_: [poi])
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])
    monkeypatch.setattr(routing, "_edge_duration_min", lambda data, _mode, _penalty=None: data["minutes"])
    return graph


def test_time_and_scope_radius_can_use_a_shorter_slower_feasible_path(monkeypatch):
    graph = _dual_budget_graph(monkeypatch, short_minutes=10)
    result, _ = tools._tool_find_reachable_places(
        {"start": {"type": "coord", "lng": 114.360, "lat": 30.530},
         "subcategory": "canteen", "mode": "walk", "max_time_min": 20},
        {"_model_dialogue": True, "_search_scope_distance_m": 500,
         "_route_graph_snapshot": graph, "_route_conditions_snapshot": []},
    )
    assert result["candidates"][0]["network_distance_m"] == 400
    assert result["candidates"][0]["estimated_duration_min"] == 10


def test_time_and_radius_reject_when_no_one_path_meets_both(monkeypatch):
    graph = _dual_budget_graph(monkeypatch, short_minutes=25)
    result, _ = tools._tool_find_reachable_places(
        {"start": {"type": "coord", "lng": 114.360, "lat": 30.530},
         "subcategory": "canteen", "mode": "walk", "max_time_min": 20},
        {"_model_dialogue": True, "_search_scope_distance_m": 500,
         "_route_graph_snapshot": graph, "_route_conditions_snapshot": []},
    )
    assert result["candidates"] == []


def test_time_and_radius_include_start_connector_in_scope(monkeypatch):
    graph = _dual_budget_graph(monkeypatch, short_minutes=10)
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_: {"snap_distance_m": 120})
    result, _ = tools._tool_find_reachable_places(
        {"start": {"type": "coord", "lng": 114.360, "lat": 30.530},
         "subcategory": "canteen", "mode": "walk", "max_time_min": 20},
        {"_model_dialogue": True, "_search_scope_distance_m": 500,
         "_route_graph_snapshot": graph, "_route_conditions_snapshot": []},
    )
    assert result["candidates"] == []


def test_distance_search_also_enforces_total_time_including_connector(monkeypatch):
    graph = _dual_budget_graph(monkeypatch, short_minutes=10)
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_: {"snap_distance_m": 120})
    result, _ = tools._tool_find_reachable_places(
        {"start": {"type": "coord", "lng": 114.360, "lat": 30.530},
         "subcategory": "canteen", "mode": "walk", "max_distance_m": 600},
        {"_model_dialogue": True, "_search_scope_duration_s": 600,
         "_route_graph_snapshot": graph, "_route_conditions_snapshot": []},
    )
    assert result["candidates"] == []


def test_time_only_search_keeps_a_farther_candidate(monkeypatch):
    graph = _dual_budget_graph(monkeypatch, short_minutes=25)
    result, _ = tools._tool_find_reachable_places(
        {"start": {"type": "coord", "lng": 114.360, "lat": 30.530},
         "subcategory": "canteen", "mode": "walk", "max_time_min": 20},
        {"_model_dialogue": True, "_route_graph_snapshot": graph, "_route_conditions_snapshot": []},
    )
    assert result["candidates"][0]["network_distance_m"] == 700


def test_search_radius_uses_center_but_time_limit_uses_actual_origin(monkeypatch):
    import agents.discovery as discovery
    import spatial.routing as routing
    graph = nx.MultiDiGraph()
    for node, lng in ((1, 114.360), (2, 114.361), (3, 114.362), (4, 114.363)):
        graph.add_node(node, x=lng, y=30.530)
    for start, end, length, minutes in ((2, 3, 100, 1), (2, 4, 200, 2),
                                         (1, 3, 900, 18), (1, 4, 300, 3)):
        graph.add_edge(start, end, key=0, length=float(length), minutes=float(minutes))
    pois = [{"id": f"poi-{node}", "name": f"候选{node}", "type": "sports", "subcategory": "court",
             "activities": ["basketball"], "coordinates": {"lng": lng, "lat": 30.530},
             "lat": 30.530, "lon": lng, "source_refs": [{"source": "test", "id": f"{node}"}]}
            for node, lng in ((3, 114.362), (4, 114.363))]
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: SimpleNamespace(
        for_mode=lambda _mode: SimpleNamespace(graph=graph, mode_penalty={})))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, *_: (
        {"出发点": 1, "搜索中心": 2}[ref["name"]], ref["name"], None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_: {"snap_distance_m": 0})
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: pois)
    monkeypatch.setattr(tools, "has_source_backed_record", lambda _poi: True)
    monkeypatch.setattr(discovery, "has_source_backed_record", lambda _poi: True)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(routing, "_edge_duration_min", lambda data, _mode, _penalty=None: data["minutes"])
    result, _ = discovery.search_places_for_dialogue(
        {"start": {"type": "poi", "name": "出发点"}, "search_anchor": {"type": "poi", "name": "搜索中心"},
         "activities": ["basketball"], "mode": "walk", "max_distance_m": 500,
         "constraints": {"max_duration_s": 600}},
        {"_model_dialogue": True, "_route_graph_snapshot": graph, "_route_conditions_snapshot": [],
         "_semantic_request": {"operation": "place_search"}},
    )
    assert [row["poi_id"] for row in result["candidates"]] == ["poi-4"]
    assert result["candidates"][0]["distance_m"] == 200
    assert result["candidates"][0]["estimated_duration_min"] == 3


def test_search_with_travel_time_requires_an_actual_origin(monkeypatch):
    import agents.discovery as discovery
    fetch = []
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: fetch.append(True) or [])
    result, _ = discovery.search_places_for_dialogue(
        {"activities": ["basketball"], "constraints": {"max_duration_s": 600}},
        {"_model_dialogue": True, "_semantic_request": {"operation": "place_search"}},
    )
    assert result["error"] == "origin_required_for_route_budget"
    assert fetch == []


def test_dialogue_search_honors_a_single_subcategory(monkeypatch):
    import agents.discovery as discovery
    seen = []
    monkeypatch.setattr(tools, "_search_poi_categories",
                        lambda categories, *_args, **_kwargs: seen.append(categories) or [])
    result, _ = discovery.search_places_for_dialogue(
        {"subcategory": "field"}, {"_model_dialogue": True, "_semantic_request": {"operation": "place_search"}})
    assert result["count"] == 0
    assert seen == [["field"]]


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
    monkeypatch.setattr(tools, "has_source_backed_record", lambda _poi: True)
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
    monkeypatch.setattr(tools, "has_source_backed_record", lambda _poi: True)
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
