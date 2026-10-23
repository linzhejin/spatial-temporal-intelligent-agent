from types import SimpleNamespace

import agents.tools as tools


def test_route_tools_expose_the_same_hard_budget_and_access_contract():
    names = {
        "plan_route", "plan_via_route", "plan_itinerary",
        "plan_tour", "plan_multimodal_route", "compare_routes",
    }
    expected = {
        "max_distance_m", "max_duration_s", "excluded_edge_ids",
        "require_verified_accessibility", "avoid_steps",
    }
    for schema in tools.TOOL_SCHEMAS:
        name = schema["function"]["name"]
        if name in names:
            properties = schema["function"]["parameters"]["properties"]
            assert expected <= set(properties["constraints"]["properties"]), name


def test_non_routing_tool_executors_return_schema_shaped_results(monkeypatch):
    poi = {"name": "卓尔体育馆", "type": "sports", "subcategory": "gym",
           "lat": 30.53, "lon": 114.36, "description": "体育场馆"}
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda name: (poi, []))
    result, artifact = tools.execute_tool("resolve_poi", {"name": "卓尔体育馆"})
    assert result["found"] is True
    assert result["poi"]["name"] == "卓尔体育馆"
    assert artifact is None

    monkeypatch.setattr(tools, "search_by_category", lambda **kwargs: [poi])
    result, artifact = tools.execute_tool("search_poi_candidates", {"subcategory": "gym"}, {
        "query": "武汉大学有哪些体育馆",
    })
    assert result["count"] == 1
    assert artifact["candidates"][0]["name"] == "卓尔体育馆"

    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {
        "live": {"weather": "晴", "temperature": 24, "windpower": "2级"},
        "impact": {"slippery": False, "hot": False, "low_visibility": False,
                   "label": "适宜出行", "advice": "天气适宜步行"},
    })
    result, artifact = tools.execute_tool("get_weather", {})
    assert result["weather"] == "晴"
    assert artifact is None

    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])
    result, artifact = tools.execute_tool("list_road_conditions", {})
    assert result["count"] == 0
    assert "没有生效" in result["message"]
    assert artifact is None

    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [{
        "id": "vision-1", "type": "event", "name": "入口缓行", "created_at": 200,
        "start_time": 100, "end_time": 300, "description": "已复核",
        "source": {"kind": "vision_job", "captured_at": 90, "field_confirmation": "现场确认"},
        "action": "slowdown", "affected_modes": ["walk"], "cost_multiplier": 1.5,
        "edge": {"road_name": "科技路"},
    }])
    result, _ = tools.execute_tool("list_road_conditions", {})
    condition = result["conditions"][0]
    assert condition["observation_time"] == 90
    assert condition["confirmed_time"] == 200
    assert condition["route_effect"] == {
        "action": "slowdown", "affected_modes": ["walk"], "cost_multiplier": 1.5,
    }
    assert condition["confirmation_note"] == "现场确认"

    ball_options = ["篮球", "羽毛球", "乒乓球", "足球", "其他球类", "其他"]
    ask_user_schema = next(
        schema for schema in tools.TOOL_SCHEMAS
        if schema["function"]["name"] == "ask_user"
    )
    assert ask_user_schema["function"]["parameters"]["properties"]["options"]["maxItems"] == 5
    result, artifact = tools.execute_tool("ask_user", {
        "question": "想打什么球？", "options": ball_options,
    })
    expected_options = ball_options[:5]
    assert result["options"] == expected_options
    assert artifact["clarify"]["question"] == "想打什么球？"
    assert artifact["clarify"]["options"] == expected_options
    result, artifact = tools.execute_tool("ask_user", {
        "question": "想打什么球？", "options": ["篮球", "羽毛球"], "slot": "sport",
    })
    assert result["slot"] == "sport"
    assert artifact["clarify"]["slot"] == "sport"
    invalid, invalid_artifact = tools.execute_tool("ask_user", {
        "question": "想打什么球？", "slot": "unsupported",
    })
    assert invalid["error"] == "invalid_clarification_slot"
    assert invalid_artifact is None
    non_string, _ = tools.execute_tool("ask_user", {
        "question": "想打什么球？", "slot": ["sport"],
    })
    assert non_string["error"] == "invalid_clarification_slot"

    result, artifact = tools.execute_tool("suggest_followup", {"suggestions": [
        {"label": "附近食堂", "query": "附近有什么食堂"},
    ]})
    assert result["suggestions"][0]["label"] == "附近食堂"
    assert artifact["suggestions"] == result["suggestions"]


def test_get_weather_returns_explicit_unavailable_result(monkeypatch):
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: None)

    result, artifact = tools.execute_tool("get_weather", {})

    assert result["error"] == "weather_unavailable"
    assert artifact is None


def test_get_weather_uses_the_same_snapshot_as_route_tools(monkeypatch):
    snapshot = {"live": {"weather": "晴", "temperature": 20, "windpower": "≤3"},
                "impact": {"slippery": False, "hot": False,
                           "low_visibility": False, "label": "", "advice": ""}}
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: snapshot)
    context = {}

    first, _ = tools.execute_tool("get_weather", {}, context)
    assert context["_route_weather_snapshot"] is snapshot
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: (_ for _ in ()).throw(
        AssertionError("weather fetched twice")))
    second, _ = tools.execute_tool("get_weather", {}, context)
    assert first == second
    assert first["temperature"] == 20


def test_direct_tour_and_multimodal_route_executors_accept_their_declared_shapes(monkeypatch):
    graph = object()
    mode_graph = {1, 2, 3, 4}
    index = SimpleNamespace(for_mode=lambda mode: SimpleNamespace(graph=mode_graph))
    decision = SimpleNamespace(
        name="shortest", weights={"distance": 1.0, "slope": 0.0, "scenery": 0.0},
        detour_cap=1.0, as_dict=lambda: {"name": "shortest", "source": "button"},
    )
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "_plan_common", lambda args, ctx: (graph, mode_graph, "walk", None))
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: index)
    names = {"起点": 1, "A": 2, "B": 3, "终点": 4}
    monkeypatch.setattr(tools, "_resolve_endpoint",
                        lambda ref, _graph: (names[ref["name"]], ref["name"], None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda ref, node, graph, mode: {
        "name": ref.get("name", "端点"), "snap_distance_m": 0,
        "status": "unverified_nearby_network_node", "access_link_verified": False,
    })
    monkeypatch.setattr(tools, "get_node_coords", lambda _graph, node: (114.36, 30.53))
    monkeypatch.setattr(tools, "_strategy_for_args", lambda args, ctx, **kwargs: decision)

    def route_result(*args, **kwargs):
        start, end = args[:2] if len(args) >= 2 else (kwargs["start_node"], kwargs["end_node"])
        return {
            "recommended": [start, end], "shortest": [start, end],
            "recommended_edges": [(start, end, 0)], "shortest_edges": [(start, end, 0)],
            "recommended_length_m": 100.0, "shortest_length_m": 100.0,
            "filter_status": "", "overlap_rate": 1.0, "duration_min": 2.0,
            "shortest_duration_min": 2.0, "applied_weights": kwargs.get("weights"),
            "degraded": False, "length_capped": False, "mode": kwargs.get("mode", "walk"),
            "timings_ms": {"path_search": 1.0},
        }

    monkeypatch.setattr(tools, "compute_route", route_result)
    monkeypatch.setattr(tools, "_attach_route_state", lambda *args, **kwargs: None)
    # This shape-only test uses an object instead of a graph; edge verification
    # is exercised separately with real MultiDiGraph fixtures.
    monkeypatch.setattr(tools, "_audit_tool_route", lambda *_args: None)
    monkeypatch.setattr(tools, "_audit_tool_tour", lambda *_args: None)
    monkeypatch.setattr(tools, "verify_segmented_route_result", lambda *_args, **_kwargs: {
        "status": "pass", "issues": [], "field_truth_verified": False,
    })
    monkeypatch.setattr(tools, "_audit_segment_geometry", lambda *_args: None)

    def payload(_graph, result, start_name, end_name, mode):
        a, b = result["recommended"]
        return {
            "start_name": start_name, "end_name": end_name, "mode": mode,
            "recommended": [{"lng": a, "lat": 0}, {"lng": b, "lat": 0}],
            "shortest": [{"lng": a, "lat": 0}, {"lng": b, "lat": 0}],
            "recommended_edge_ids": result["recommended_edges"],
            "shortest_edge_ids": result["shortest_edges"], "steps": [], "pois": [],
            "recommended_length_m": 100.0, "shortest_length_m": 100.0,
            "distance_m": 100.0, "duration_min": 2.0, "shortest_duration_min": 2.0,
            "filter_status": "", "overlap_rate": 1.0, "degraded": False,
            "length_capped": False, "timings_ms": {
                "path_search": 1.0, "graph_prepare": 0.0, "poi_resolution": 0.0,
                "response_build": 0.0,
            },
        }

    monkeypatch.setattr(tools, "_route_payload", payload)

    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
    })
    assert artifact["route"] is result
    assert result["distance_m"] == 100.0

    pois = [
        {"name": name, "type": "scenery", "subcategory": "landmark",
         "lat": 30.53, "lon": 114.36}
        for name in ("A", "B")
    ]
    poi_by_name = {poi["name"]: poi for poi in pois}
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda name: (poi_by_name.get(name), []))
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lon, lat: (lon, lat))
    monkeypatch.setattr(tools, "get_nearest_node", lambda _graph, lng, lat: 2 if lng < 114.365 else 3)
    seen_tour_names = {}

    def tour_result(_graph, poi_nodes, **kwargs):
        seen_tour_names["names"] = [poi["name"] for poi, _ in poi_nodes]
        return {
            "legs": [route_result(1, 2), route_result(2, 1)], "ordered_pois": pois,
            "total_length_m": 200.0, "duration_min": 4.0, "dropped": [], "loop": True,
        }

    monkeypatch.setattr(tools, "compute_tour_route", tour_result)
    result, artifact = tools.execute_tool("plan_tour", {"loop": True}, {
        "query": "游览A和B", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None,
    })
    assert artifact["route_kind"] == "tour"
    assert len(result["tour"]["ordered_pois"]) == 2
    assert seen_tour_names["names"] == ["A", "B"]

    result, artifact = tools.execute_tool("plan_multimodal_route", {
        "start": {"name": "起点"},
        "legs": [
            {"mode": "walk", "end": {"name": "A"}},
            {"mode": "bike", "end": {"name": "终点"}},
        ],
    })
    assert artifact["route_kind"] == "multimodal"
    assert [leg["mode"] for leg in result["legs"]] == ["walk", "bike"]
