import agents.tools as tools


def test_compare_routes_returns_three_comparable_summaries_without_committing_route(monkeypatch):
    calls = []
    metrics = {
        "shortest": (500.0, 7.0, 3.0, 2.0),
        "scenery": (575.0, 8.1, 2.8, 4.1),
        "flat": (540.0, 7.8, 1.7, 2.4),
    }

    def plan_route(args, ctx):
        strategy = args["strategy"]
        calls.append((strategy, args["start"], args["end"], args["mode"], args["constraints"]))
        distance, duration, slope, scenery = metrics[strategy]
        return {
            "start_name": "星湖园食堂",
            "end_name": "卓尔体育馆",
            "mode": "walk",
            "recommended_length_m": distance,
            "duration_min": duration,
            "slope_avg_recommended": slope,
            "scenery_avg_recommended": scenery,
            "applied_weights": {"distance": 1, "slope": 0, "scenery": 0},
            "route_state": {"should_not": "be_committed_by_comparison"},
            "recommended": [[114.36, 30.53]],
        }, {"route": {"ignored": True}}

    monkeypatch.setattr(tools, "_tool_plan_route", plan_route)
    monkeypatch.setattr(tools, "_ensure_graph", lambda: object())
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: None)

    result, artifact = tools.execute_tool("compare_routes", {
        "start": {"type": "poi", "name": "星湖园食堂"},
        "end": {"type": "poi", "name": "卓尔体育馆"},
        "mode": "walk",
        "constraints": {"slope": "normal"},
    }, {"query": "比较三种走法"})

    assert [call[0] for call in calls] == ["shortest", "scenery", "flat"]
    assert all(call[1] == calls[0][1] and call[2] == calls[0][2] for call in calls)
    assert all(call[3] == "walk" and call[4] == {"slope": "normal"} for call in calls)
    assert [item["strategy"] for item in result["alternatives"]] == ["shortest", "scenery", "flat"]
    assert [item["distance_m"] for item in result["alternatives"]] == [500, 575, 540]
    assert [item["distance_delta_from_shortest_m"] for item in result["alternatives"]] == [0, 75, 40]
    assert result["alternatives"][1]["duration_min"] == 8.1
    assert result["alternatives"][1]["slope_level_avg"] == 2.8
    assert result["alternatives"][1]["scenery_level_avg"] == 4.1
    assert artifact == {"route_comparison": result}


def test_compare_routes_reports_partial_success_and_each_unavailable_strategy(monkeypatch):
    def plan_route(args, _ctx):
        if args["strategy"] == "scenery":
            return {"error": "route_not_found", "message": "无可行路线"}, None
        distance = 400 if args["strategy"] == "shortest" else 430
        return {
            "start_name": "起点",
            "end_name": "终点",
            "mode": "walk",
            "recommended_length_m": distance,
            "duration_min": 6.0,
            "applied_weights": {},
        }, None

    monkeypatch.setattr(tools, "_tool_plan_route", plan_route)
    monkeypatch.setattr(tools, "_ensure_graph", lambda: object())
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: None)

    result, _artifact = tools.execute_tool("compare_routes", {
        "start": {"type": "coord", "name": "起点", "lng": 114.36, "lat": 30.53},
        "end": {"type": "coord", "name": "终点", "lng": 114.37, "lat": 30.54},
    }, {})

    assert [item["strategy"] for item in result["alternatives"]] == ["shortest", "flat"]
    assert result["unavailable"] == [{"strategy": "scenery", "label": "风景优先", "reason": "无可行路线"}]
    assert result["status"] == "partial"


def test_compare_routes_returns_structured_artifact_when_every_strategy_is_unavailable(monkeypatch):
    def plan_route(_args, _ctx):
        return {"error": "route_not_found", "message": "当前条件下无可行路线"}, None

    monkeypatch.setattr(tools, "_tool_plan_route", plan_route)
    monkeypatch.setattr(tools, "_ensure_graph", lambda: object())
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: None)

    result, artifact = tools.execute_tool("compare_routes", {
        "start": {"type": "coord", "name": "起点", "lng": 114.36, "lat": 30.53},
        "end": {"type": "coord", "name": "终点", "lng": 114.37, "lat": 30.54},
    }, {})

    assert result["status"] == "unavailable"
    assert result["alternatives"] == []
    assert len(result["unavailable"]) == 3
    assert artifact == {"route_comparison": result}


def test_compare_routes_reuses_supplied_snapshots(monkeypatch):
    graph = object()
    conditions = [{"id": "closure-1", "type": "closure"}]
    seen_contexts = []

    def plan_route(args, ctx):
        seen_contexts.append(ctx)
        return {
            "start_name": "起点", "end_name": "终点", "mode": "walk",
            "recommended_length_m": 100, "duration_min": 2,
            "applied_weights": {},
        }, None

    monkeypatch.setattr(tools, "_ensure_graph", lambda: (_ for _ in ()).throw(AssertionError("must reuse graph snapshot")))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must reuse condition snapshot")))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: (_ for _ in ()).throw(AssertionError("must reuse weather snapshot")))
    monkeypatch.setattr(tools, "_tool_plan_route", plan_route)

    result, artifact = tools.execute_tool("compare_routes", {
        "start": {"type": "coord", "name": "起点", "lng": 114.36, "lat": 30.53},
        "end": {"type": "coord", "name": "终点", "lng": 114.37, "lat": 30.54},
    }, {
        "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": conditions,
        "_route_weather_snapshot": {"live": {"weather": "晴"}},
    })

    assert result["status"] == "complete"
    assert artifact == {"route_comparison": result}
    assert len(seen_contexts) == 3
    assert all(ctx["_route_graph_snapshot"] is graph for ctx in seen_contexts)
    assert all(ctx["_route_conditions_snapshot"] is conditions for ctx in seen_contexts)


def test_compare_routes_schema_is_registered_for_function_calling():
    tool = next((item for item in tools.TOOL_SCHEMAS
                 if item["function"]["name"] == "compare_routes"), None)

    assert tool is not None
    assert tool["function"]["parameters"]["required"] == ["start", "end"]


def test_route_payload_preserves_spatial_quality_metrics_for_route_comparison(monkeypatch):
    monkeypatch.setattr(tools, "_path_to_coords", lambda *_args: [[114.36, 30.53], [114.37, 30.54]])
    monkeypatch.setattr(tools, "_coords_wgs_to_gcj", lambda coords: coords)
    monkeypatch.setattr(tools, "_build_steps_gcj", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "_pois_along_route", lambda *_args: [])
    route_result = {
        "recommended": [1, 2],
        "shortest": [1, 2],
        "recommended_edges": [(1, 2, 0)],
        "shortest_edges": [(1, 2, 0)],
        "filter_status": "no_filter",
        "overlap_rate": 1.0,
        "recommended_length_m": 100.0,
        "shortest_length_m": 100.0,
        "duration_min": 2.0,
        "shortest_duration_min": 2.0,
        "applied_weights": {"distance": 0.5, "slope": 0.2, "scenery": 0.3},
        "degraded": False,
        "length_capped": False,
        "mode": "walk",
        "speed_kmh": 4.5,
        "slope_avg_recommended": 1.7,
        "scenery_avg_recommended": 3.2,
        "timings_ms": {},
    }

    payload = tools._route_payload(None, route_result, "起点", "终点", "walk")

    assert payload["slope_avg_recommended"] == 1.7
    assert payload["scenery_avg_recommended"] == 3.2
