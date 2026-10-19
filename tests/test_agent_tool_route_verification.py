from types import SimpleNamespace
from tests.test_route_reuse import mock_proof_identity  # noqa: F401 - register shared pytest fixture

import networkx as nx

from agents import tools
from agents import route_reuse
from agents.route_state import apply_change, current_road_condition_version
import spatial.routing as routing_module


def _graph():
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3):
        graph.add_node(node, x=114.36 + node * 0.0001, y=30.53)
    graph.add_edge(1, 2, key=0, length=60.0, highway="footway")
    graph.add_edge(2, 3, key=0, length=40.0, highway="footway")
    return graph


def _patch_common(monkeypatch, graph):
    monkeypatch.setattr(tools, "_plan_common", lambda _args, _ctx: (graph, graph, "walk", None))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _graph: (
        {"起点": 1, "中点": 2, "终点": 3}[ref["name"]], ref["name"], None,
    ))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_args: {
        "status": "unverified_nearby_network_node", "snap_distance_m": 0,
    })
    decision = SimpleNamespace(
        name="shortest", weights={"distance": 1, "slope": 0, "scenery": 0},
        detour_cap=1.0, as_dict=lambda: {"name": "shortest"},
    )
    monkeypatch.setattr(tools, "_strategy_for_args", lambda *_args, **_kwargs: decision)
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda _name: (None, []))


def _route_result(start, end, mode="walk"):
    edges = [(node, node + 1, 0) for node in range(start, end)]
    length = sum(60.0 if edge[0] == 1 else 40.0 for edge in edges)
    duration = round(length / (routing_module.MODE_SPEEDS_KMH[mode] * 1000 / 60), 1)
    return {"recommended": list(range(start, end + 1)),
            "shortest": list(range(start, end + 1)),
            "recommended_edges": edges, "shortest_edges": edges,
            "recommended_length_m": length, "shortest_length_m": length,
            "filter_status": "", "overlap_rate": 1.0, "duration_min": duration,
            "shortest_duration_min": duration, "degraded": False,
            "length_capped": False, "mode": mode, "timings_ms": {}}


def _payload(_graph, result, start_name, end_name, mode):
    recommended = tools._coords_wgs_to_gcj(tools._path_to_coords(
        _graph, result["recommended"], result["recommended_edges"],
    ))
    shortest = tools._coords_wgs_to_gcj(tools._path_to_coords(
        _graph, result["shortest"], result["shortest_edges"],
    ))
    return {"start_name": start_name, "end_name": end_name, "mode": mode,
            "recommended": recommended,
            "shortest": shortest,
            "recommended_edge_ids": result["recommended_edges"],
            "shortest_edge_ids": result["shortest_edges"],
            "recommended_length_m": result["recommended_length_m"],
            "shortest_length_m": result["shortest_length_m"],
            "distance_m": result["recommended_length_m"],
            "duration_min": result["duration_min"],
            "shortest_duration_min": result["shortest_duration_min"],
            "steps": [], "pois": [], "filter_status": "", "overlap_rate": 1.0,
            "degraded": False, "length_capped": False, "timings_ms": {
                "graph_prepare": 0.0, "poi_resolution": 0.0,
            }}


def test_agent_direct_tool_rejects_mismatched_selected_edge_length(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "compute_route", lambda **_kwargs: _route_result(1, 3))

    def bad_payload(*args):
        payload = _payload(*args)
        payload["recommended_length_m"] = 99.0
        return payload

    monkeypatch.setattr(tools, "_route_payload", bad_payload)
    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
    }, {"query": "从起点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert result["error"] == "route_verification_failed"
    assert "length_mismatch" in result["route_verification"]["issues"]
    assert artifact is None


def test_agent_direct_tool_returns_verified_route(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "compute_route", lambda **_kwargs: _route_result(1, 3))
    monkeypatch.setattr(tools, "_route_payload", _payload)
    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
    }, {"query": "从起点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None,
        "task_revision": 4})
    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert result["route_verification"]["task_revision"] == 4
    assert result["route_verification"]["checks"]["display_geometry"]["status"] == "pass"
    assert result["route_verification"]["violations"] == []
    assert result["route_verification"]["unknowns"] == []
    assert result["route_verification"]["paths"]["recommended"]["task_revision"] == 4


def test_verified_shortest_route_reuses_exact_path_after_off_path_closure(monkeypatch, mock_proof_identity):  # noqa: F811
    import app as app_module

    graph = _graph()
    graph.add_node(4, x=114.361, y=30.531)
    graph.add_edge(1, 4, key=0, length=80.0, highway="footway")
    graph.add_edge(4, 3, key=0, length=80.0, highway="footway")
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    args = {"start": {"name": "起点"}, "end": {"name": "终点"}}
    context = {
        "query": "从起点到终点走最短路线",
        "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [],
        "_route_weather_snapshot": {"live": None},
    }

    context["_route_proof_identity"] = mock_proof_identity

    with app_module.app.app_context():
        first, first_artifact = tools.execute_tool("plan_route", args, context)
        assert first_artifact["route"] is first
        assert first["route_verification"]["status"] == "pass"
        assert route_reuse.has_valid_reuse_certificate(first["route_state"])

        condition = {
            "id": "closure-off-shortest", "type": "event", "action": "closure",
            "affected_modes": ["walk"], "start_time": 0, "end_time": 0,
            "edge": {"u": 1, "v": 4, "key": 0, "edges": [[1, 4, 0]]},
        }
        monkeypatch.setattr(
            "spatial.road_conditions._resolve_edge_keys",
            lambda _graph, _condition, strict=False: {(1, 4, 0)},
        )
        monkeypatch.setattr(
            "spatial.road_conditions._expand_coincident_corridor_edges",
            lambda _graph, keys, _edge: keys,
        )
        expected = apply_change(first["route_state"], {"refresh_road_conditions": True})
        expected["road_condition_version"] = current_road_condition_version([condition])
        refreshed_context = {
            **context,
            "replan": True,
            "_route_conditions_snapshot": [condition],
            "_reuse_route_state": first["route_state"],
            "_reuse_expected_state": expected,
        }

        refreshed, refreshed_artifact = tools.execute_tool(
            "plan_route", args, refreshed_context,
        )

    assert refreshed_artifact["route"] is refreshed
    assert refreshed["route_verification"]["status"] == "pass"
    assert refreshed["execution_optimization"]["path_reused"] is True
    assert refreshed["execution_optimization"]["dijkstra_runs"] == 0
    assert refreshed["recommended_edge_ids"] == [(1, 2, 0), (2, 3, 0)]
    with app_module.app.app_context():
        assert route_reuse.has_valid_reuse_certificate(refreshed["route_state"])


def test_agent_direct_tool_rechecks_duration_with_the_same_weather_snapshot(monkeypatch):
    graph = _graph()
    graph[1][2][0]["slope_level"] = 4
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "_plan_common", lambda _args, ctx: (
        graph, graph, "walk", ctx["_route_weather_snapshot"]["live"],
    ))
    monkeypatch.setattr(tools, "_route_payload", _payload)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    weather = {"weather": "小雨", "temperature": 10}

    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
    }, {"query": "从起点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": {"live": weather}})

    assert artifact["route"] is result
    assert result["duration_min"] > 1.3
    assert result["route_verification"]["paths"]["recommended"]["expected_duration_min"] == result["duration_min"]
    assert set(result["route_verification"]["paths"]) == {"recommended", "shortest"}


def test_agent_direct_tool_does_not_publish_over_budget_route(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "compute_route", lambda **_kwargs: _route_result(1, 3))
    monkeypatch.setattr(tools, "_route_payload", _payload)

    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "constraints": {"max_distance_m": 90},
    }, {"query": "90米以内从起点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert result["error"] == "route_verification_failed"
    assert "distance_budget_exceeded" in result["route_verification"]["issues"]
    assert "距离" in result["message"]
    assert artifact is None


def test_agent_prefers_feasible_shortest_when_scenic_candidate_exceeds_budget(monkeypatch):
    graph = _graph()
    graph[1][2][0].update(length=70.0, slope_level=2, scenery_level=5)
    graph[2][3][0].update(length=70.0, slope_level=2, scenery_level=5)
    graph.add_edge(1, 3, key=0, length=100.0, highway="footway",
                   slope_level=5, scenery_level=1)
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    decision = SimpleNamespace(
        name="scenery", weights={"distance": 0.3, "slope": 0.1, "scenery": 0.6},
        detour_cap=1.5, as_dict=lambda: {"name": "scenery"},
    )
    monkeypatch.setattr(tools, "_strategy_for_args", lambda *_args, **_kwargs: decision)

    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "constraints": {"max_distance_m": 120},
    }, {"query": "风景优先但不超过120米", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert result["hard_budget_fallback"] is True
    assert result["recommended_edge_ids"] == [(1, 3, 0)]
    assert result["recommended_length_m"] == 100.0


def test_agent_keeps_feasible_scenic_route_when_shortest_reference_is_too_slow(monkeypatch):
    graph = _graph()
    graph[1][2][0].update(length=70.0, slope_level=2, scenery_level=5)
    graph[2][3][0].update(length=70.0, slope_level=2, scenery_level=5)
    graph.add_edge(1, 3, key=0, length=100.0, highway="footway",
                   slope_level=5, scenery_level=1)
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    decision = SimpleNamespace(
        name="scenery", weights={"distance": 0.3, "slope": 0.1, "scenery": 0.6},
        detour_cap=1.5, as_dict=lambda: {"name": "scenery"},
    )
    monkeypatch.setattr(tools, "_strategy_for_args", lambda *_args, **_kwargs: decision)

    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "constraints": {"max_duration_s": 120},
    }, {"query": "风景优先且两分钟内到达", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert result["shortest_baseline_feasible"] is False
    assert result["recommended_edge_ids"] == [(1, 2, 0), (2, 3, 0)]


def test_agent_direct_tool_rejects_line_drawn_from_other_geometry(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "compute_route", lambda **_kwargs: _route_result(1, 3))

    def wrong_line(*args):
        payload = _payload(*args)
        payload["recommended"][1]["lng"] += 0.001
        return payload

    monkeypatch.setattr(tools, "_route_payload", wrong_line)
    result, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
    }, {"query": "从起点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert result["error"] == "route_verification_failed"
    assert "display_geometry_mismatch" in result["route_verification"]["issues"]
    assert result["route_verification"]["checks"]["display_geometry"]["status"] == "fail"
    assert "display_geometry_mismatch" in result["route_verification"]["violations"]
    assert artifact is None


def test_agent_via_tool_rejects_wrong_second_leg_before_answer(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "compute_route", lambda _graph, start, end, **_kwargs: _route_result(start, end))

    def bad_second_leg(*args):
        payload = _payload(*args)
        if payload["start_name"] == "中点":
            payload["recommended_edge_ids"] = [(1, 2, 0)]
        return payload

    monkeypatch.setattr(tools, "_route_payload", bad_second_leg)
    result, artifact = tools.execute_tool("plan_via_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "via_points": [{"name": "中点"}],
    }, {"query": "从起点经中点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert result["error"] == "route_verification_failed"
    assert "leg_2:endpoint_mismatch" in result["route_verification"]["issues"]
    assert artifact is None


def test_agent_via_tool_returns_ordered_verified_route(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "compute_route", lambda _graph, start, end, **_kwargs: _route_result(start, end))
    monkeypatch.setattr(tools, "_route_payload", _payload)
    result, artifact = tools.execute_tool("plan_via_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "via_points": [{"name": "中点"}],
    }, {"query": "从起点经中点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert len(result["route_verification"]["paths"]["recommended"]["leg_reports"]) == 2


def test_agent_via_keeps_fast_route_when_shortest_reference_breaks_total_time(monkeypatch):
    graph = _graph()
    graph[1][2][0].update(slope_level=5, scenery_level=1)
    graph[2][3][0].update(slope_level=5, scenery_level=1)
    graph.add_edge(1, 2, key=1, length=70.0, highway="footway",
                   slope_level=2, scenery_level=5)
    graph.add_edge(2, 3, key=1, length=50.0, highway="footway",
                   slope_level=2, scenery_level=5)
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)
    decision = SimpleNamespace(
        name="scenery", weights={"distance": 0.3, "slope": 0.1, "scenery": 0.6},
        detour_cap=2.0, as_dict=lambda: {"name": "scenery"},
    )
    monkeypatch.setattr(tools, "_strategy_for_args", lambda *_args, **_kwargs: decision)

    result, artifact = tools.execute_tool("plan_via_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "via_points": [{"name": "中点"}],
        "constraints": {"max_duration_s": 120},
    }, {"query": "经过中点，两分钟内到达", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert result["shortest_baseline_feasible"] is False
    assert result["recommended_edge_ids"] == [(1, 2, 1), (2, 3, 1)]


def test_agent_legacy_response_preserves_internal_verification_report():
    from api.routes import _agent_response_to_legacy

    report = {"status": "pass", "field_truth_verified": False}
    response = _agent_response_to_legacy({
        "response_kind": "route", "route_kind": "direct", "message": "路线已生成",
        "route": {"start_name": "起点", "end_name": "终点",
                  "recommended": [[114.36, 30.53]], "route_verification": report,
                  "shortest_baseline_feasible": False, "hard_budget_fallback": True},
    })
    assert response["route_verification"] == report
    assert response["shortest_baseline_feasible"] is False
    assert response["hard_budget_fallback"] is True


def test_agent_legacy_response_preserves_route_comparison_artifact():
    from api.routes import _agent_response_to_legacy

    comparison = {"status": "partial", "alternatives": [], "unavailable": []}
    response = _agent_response_to_legacy({
        "response_kind": "route_comparison",
        "message": "比较了可用路线。",
        "route_comparison": comparison,
    })

    assert response["task_type"] == "chat"
    assert response["response_kind"] == "route_comparison"
    assert response["route_comparison"] == comparison


def test_agent_tour_tool_rejects_corrupt_leg_before_publication(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    pois = [
        {"id": "middle", "name": "中点", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3602, "lat": 30.53},
        {"id": "end", "name": "终点", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3603, "lat": 30.53},
    ]
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: pois)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lon, lat: (lon, lat))
    monkeypatch.setattr(tools, "_snap_wgs_to_network", lambda lon, _lat, _graph, _name: (
        2 if lon < 114.36025 else 3, {}, None,
    ))
    monkeypatch.setattr(tools, "importance_score", lambda _poi: 10)
    monkeypatch.setattr(tools, "_route_payload", lambda graph, route, start, end, mode: (
        {**_payload(graph, route, start, end, mode),
         "recommended_edge_ids": [(1, 2, 0)] if start == "中点" else route["recommended_edges"]}
    ))
    monkeypatch.setattr(tools, "compute_tour_route", lambda _graph, _pois, **_kwargs: {
        "ordered_pois": pois, "ordered_nodes": [1, 2, 3], "loop": False,
        "legs": [_route_result(1, 2), _route_result(2, 3)],
        "total_length_m": 100.0, "dropped": [],
    })
    result, artifact = tools.execute_tool("plan_tour", {
        "start": {"name": "起点"}, "loop": False,
    }, {"query": "游览校园", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert result["error"] == "route_verification_failed"
    assert "leg_2:endpoint_mismatch" in result["route_verification"]["issues"]
    assert artifact is None


def test_agent_multimodal_tool_rejects_wrong_second_mode_edge(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    graph[2][3][0]["bicycle"] = "no"
    index = SimpleNamespace(for_mode=lambda _mode: SimpleNamespace(graph=graph))
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: index)
    monkeypatch.setattr(tools, "compute_route", lambda **kwargs: _route_result(
        kwargs["start_node"], kwargs["end_node"], kwargs["mode"],
    ))
    monkeypatch.setattr(tools, "_route_payload", _payload)
    result, artifact = tools.execute_tool("plan_multimodal_route", {
        "start": {"name": "起点"},
        "legs": [{"mode": "walk", "end": {"name": "中点"}},
                 {"mode": "bike", "end": {"name": "终点"}}],
    }, {"query": "步行到中点再骑行到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert result["error"] == "route_verification_failed"
    assert "leg_2:mode_forbidden" in result["route_verification"]["issues"]
    assert artifact is None


def test_multimodal_transfer_rejects_unmapped_straight_connector():
    graph = _graph()
    assert tools._unverified_transfer_gap_m(graph, 1, 2) > 3
    assert tools._unverified_transfer_gap_m(graph, 2, 2) == 0


def test_agent_tour_tool_returns_verified_route(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    pois = [
        {"id": "middle", "name": "中点", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3602, "lat": 30.53},
        {"id": "end", "name": "终点", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3603, "lat": 30.53},
    ]
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: pois)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lon, lat: (lon, lat))
    monkeypatch.setattr(tools, "_snap_wgs_to_network", lambda lon, _lat, _graph, _name: (
        2 if lon < 114.36025 else 3, {}, None,
    ))
    monkeypatch.setattr(tools, "importance_score", lambda _poi: 10)
    monkeypatch.setattr(tools, "_route_payload", _payload)
    monkeypatch.setattr(tools, "compute_tour_route", lambda _graph, _pois, **_kwargs: {
        "ordered_pois": pois, "ordered_nodes": [1, 2, 3], "loop": False,
        "legs": [_route_result(1, 2), _route_result(2, 3)],
        "total_length_m": 100.0, "dropped": [],
    })
    result, artifact = tools.execute_tool("plan_tour", {
        "start": {"name": "起点"}, "loop": False,
    }, {"query": "游览校园", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert len(result["route_verification"]["leg_reports"]) == 2
    assert result["tour"]["duration_min"] == 1.3


def test_tour_budget_search_limit_is_not_reported_as_tool_exception(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    pois = [
        {"id": "middle", "name": "中点", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3602, "lat": 30.53},
        {"id": "end", "name": "终点", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3603, "lat": 30.53},
    ]
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: pois)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lon, lat: (lon, lat))
    monkeypatch.setattr(tools, "_snap_wgs_to_network", lambda lon, _lat, _graph, _name: (
        2 if lon < 114.36025 else 3, {}, None,
    ))
    monkeypatch.setattr(tools, "importance_score", lambda _poi: 10)

    def incomplete(*_args, **_kwargs):
        raise routing_module.BudgetSearchIncomplete("预算搜索超出上限")

    monkeypatch.setattr(tools, "compute_tour_route", incomplete)
    result, artifact = tools.execute_tool("plan_tour", {
        "start": {"name": "起点"}, "loop": False,
    }, {"query": "游览校园", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert result["error"] == "route_search_incomplete"
    assert artifact is None


def test_agent_multimodal_tool_returns_verified_route(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    index = SimpleNamespace(for_mode=lambda _mode: SimpleNamespace(graph=graph))
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: index)
    monkeypatch.setattr(tools, "compute_route", lambda **kwargs: _route_result(
        kwargs["start_node"], kwargs["end_node"], kwargs["mode"],
    ))
    monkeypatch.setattr(tools, "_route_payload", _payload)
    result, artifact = tools.execute_tool("plan_multimodal_route", {
        "start": {"name": "起点"},
        "legs": [{"mode": "walk", "end": {"name": "中点"}},
                 {"mode": "bike", "end": {"name": "终点"}}],
    }, {"query": "步行到中点再骑行到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert len(result["route_verification"]["leg_reports"]) == 2


def test_multimodal_budget_search_limit_is_not_reported_as_no_route(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    index = SimpleNamespace(for_mode=lambda _mode: SimpleNamespace(graph=graph))
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: index)

    def incomplete(**_kwargs):
        raise routing_module.BudgetSearchIncomplete("预算搜索超出上限")

    monkeypatch.setattr(tools, "compute_route", incomplete)
    result, artifact = tools.execute_tool("plan_multimodal_route", {
        "start": {"name": "起点"},
        "legs": [{"mode": "walk", "end": {"name": "中点"}},
                 {"mode": "bike", "end": {"name": "终点"}}],
    }, {"query": "步行到中点再骑行到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert result["error"] == "route_search_incomplete"
    assert artifact is None


def test_multimodal_tool_uses_global_budget_fallback_and_verifies_exact_edges(monkeypatch):
    graph = nx.MultiDiGraph()
    coordinates = {
        0: (114.3600, 30.5350), 1: (114.3601, 30.5350),
        2: (114.3602, 30.5350), 3: (114.3603, 30.5350),
    }
    for node, (longitude, latitude) in coordinates.items():
        graph.add_node(node, x=longitude, y=latitude)
    graph.add_edge(0, 1, key=0, length=100.0, highway="steps", slope_level=1)
    graph.add_edge(0, 2, key=0, length=70.0, highway="path", slope_level=1)
    graph.add_edge(2, 1, key=0, length=70.0, highway="path", slope_level=1)
    graph.add_edge(1, 3, key=0, length=100.0, highway="residential", slope_level=5)
    names = {"start": 0, "transfer": 1, "end": 3}
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _graph: (
        names[ref["name"]], ref["name"], None,
    ))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_args: {
        "status": "unverified_nearby_network_node", "snap_distance_m": 0,
    })

    result, artifact = tools.execute_tool("plan_multimodal_route", {
        "start": {"name": "start"},
        "legs": [{"mode": "walk", "end": {"name": "transfer"}},
                 {"mode": "bike", "end": {"name": "end"}}],
        "constraints": {"max_distance_m": 250.0, "max_duration_s": 240.0},
    }, {"query": "walk then bike", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert result["budget_fallback"] is True
    assert result["recommended_edge_ids"] == [(0, 2, 0), (2, 1, 0), (1, 3, 0)]
    assert result["recommended_length_m"] == 240.0
    assert result["duration_min"] * 60 <= 243.0


def test_multimodal_tool_fails_closed_when_only_combined_route_exceeds_budget(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    index = SimpleNamespace(for_mode=lambda _mode: SimpleNamespace(graph=graph))
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: index)

    result, artifact = tools.execute_tool("plan_multimodal_route", {
        "start": {"name": "起点"},
        "legs": [{"mode": "walk", "end": {"name": "中点"}},
                 {"mode": "bike", "end": {"name": "终点"}}],
        "constraints": {"max_distance_m": 75.0},
    }, {"query": "walk then bike", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert result["error"] == "route_budget_infeasible"
    assert artifact is None


def test_multimodal_via_leg_counts_each_actual_subleg_time(monkeypatch):
    graph = _graph()
    graph.add_node(4, x=114.3604, y=30.53)
    graph.add_edge(3, 4, key=0, length=40.0, highway="footway")
    _patch_common(monkeypatch, graph)
    names = {"起点": 1, "中点": 2, "终点": 3, "远端": 4}
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _graph: (
        names[ref["name"]], ref["name"], None,
    ))
    index = SimpleNamespace(for_mode=lambda _mode: SimpleNamespace(graph=graph))
    monkeypatch.setattr(tools, "get_routing_index", lambda _graph: index)
    monkeypatch.setattr(tools, "compute_via_route", lambda *_args, **_kwargs: {
        "leg1": _route_result(1, 2), "leg2": _route_result(2, 3),
        "total_length_m": 100.0,
    })
    monkeypatch.setattr(tools, "compute_route", lambda **kwargs: _route_result(3, 4, kwargs["mode"]))
    monkeypatch.setattr(tools, "_route_payload", _payload)

    result, artifact = tools.execute_tool("plan_multimodal_route", {
        "start": {"name": "起点"},
        "legs": [{"mode": "walk", "via_name": "中点", "end": {"name": "终点"}},
                 {"mode": "bike", "end": {"name": "远端"}}],
    }, {"query": "步行经中点到终点，再骑行到远端", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert artifact["route"] is result
    assert result["legs"][0]["duration_min"] == 1.3
    assert result["duration_min"] == 1.5
    assert result["route_verification"]["status"] == "pass"


def test_tour_tool_uses_whole_trip_budget_fallback(monkeypatch):
    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _graph: (
        1, ref["name"], None,
    ))
    pois = [
        {"id": "p2", "name": "stop-two", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3602, "lat": 30.53},
        {"id": "p3", "name": "stop-three", "type": "scenery", "subcategory": "landmark",
         "lon": 114.3603, "lat": 30.53},
    ]
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: pois)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lon, lat: (lon, lat))
    monkeypatch.setattr(tools, "_snap_wgs_to_network", lambda lon, _lat, _graph, _name: (
        2 if lon < 114.36025 else 3, {}, None,
    ))
    monkeypatch.setattr(tools, "importance_score", lambda _poi: 10)
    base_legs = [_route_result(1, 2), _route_result(2, 3)]
    for leg in base_legs:
        leg["recommended_length_m"] = 120.0
    monkeypatch.setattr(tools, "compute_tour_route", lambda *_args, **_kwargs: {
        "ordered_pois": pois, "ordered_nodes": [1, 2, 3], "loop": False,
        "legs": base_legs, "total_length_m": 240.0, "dropped": [],
    })
    selected = [
        {"nodes": [1, 2], "edges": [(1, 2, 0)], "recommended_length_m": 60.0,
         "duration_min": 0.8, "slope_avg_recommended": 2.0,
         "scenery_avg_recommended": 3.0, "degraded": False},
        {"nodes": [2, 3], "edges": [(2, 3, 0)], "recommended_length_m": 40.0,
         "duration_min": 0.5, "slope_avg_recommended": 2.0,
         "scenery_avg_recommended": 3.0, "degraded": False},
    ]
    calls = []

    def budget_search(_graph, waypoints, **_kwargs):
        calls.append(list(waypoints))
        return {"legs": selected, "distance_m": 100.0, "duration_min": 1.3,
                "search_ms": 0.5}

    monkeypatch.setattr(tools, "compute_budgeted_ordered_route", budget_search)
    monkeypatch.setattr(tools, "_route_payload", _payload)
    monkeypatch.setattr(tools, "_audit_tool_tour", lambda *_args: None)

    result, artifact = tools.execute_tool("plan_tour", {
        "start": {"name": "start"}, "loop": False,
        "constraints": {"max_distance_m": 150},
    }, {"query": "tour", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})

    assert artifact["route"] is result
    assert calls == [[1, 2, 3]]
    assert result["recommended_length_m"] == 100.0
    assert result["budget_fallback"] is True
    assert [leg["hard_budget_fallback"] for leg in base_legs] == [True, True]
    assert [leg["recommended_edges"] for leg in base_legs] == [
        [(1, 2, 0)], [(2, 3, 0)],
    ]


def test_via_tool_finds_route_under_cumulative_trip_time_budget(monkeypatch):
    graph = nx.MultiDiGraph()
    coords = {1: (114.3600, 30.5350), 2: (114.3602, 30.5350),
              4: (114.3604, 30.5350), 5: (114.3601, 30.5351)}
    for node, (lng, lat) in coords.items():
        graph.add_node(node, x=lng, y=lat)
    graph.add_edge(1, 2, key=0, length=100.0, highway="steps", slope_level=1,
                   scenery_level=3)
    graph.add_edge(1, 5, key=0, length=120.0, highway="path", slope_level=1,
                   scenery_level=3)
    graph.add_edge(5, 2, key=0, length=120.0, highway="path", slope_level=1,
                   scenery_level=3)
    graph.add_edge(2, 4, key=0, length=100.0, highway="path", slope_level=1,
                   scenery_level=3)
    names = {"start": 1, "via": 2, "end": 4}
    monkeypatch.setattr(tools, "_plan_common", lambda *_args: (graph, graph, "walk", None))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _graph: (
        names[ref["name"]], ref["name"], None,
    ))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_args: {
        "status": "unverified_nearby_network_node", "snap_distance_m": 0,
    })
    monkeypatch.setattr(tools, "_strategy_for_args", lambda *_args, **_kwargs: SimpleNamespace(
        name="shortest", weights={"distance": 1.0, "slope": 0.0, "scenery": 0.0},
        detour_cap=1.0, as_dict=lambda: {"name": "shortest", "source": "test"},
    ))
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda _name: (None, []))
    monkeypatch.setattr(routing_module, "_should_degrade_annotations", lambda: None)

    context = {"query": "route via ordered waypoint", "_route_graph_snapshot": graph,
               "_route_conditions_snapshot": [], "_route_weather_snapshot": None}
    result, artifact = tools.execute_tool("plan_via_route", {
        "start": {"name": "start"}, "end": {"name": "end"},
        "via_points": [{"name": "via"}],
        "constraints": {"max_distance_m": 350, "max_duration_s": 270},
    }, context)

    assert artifact["route"] is result
    assert result["route_verification"]["status"] == "pass"
    assert result["duration_min"] * 60 <= 273
    assert result["recommended_length_m"] == 340.0
    assert result["recommended_edge_ids"][:2] == [(1, 5, 0), (5, 2, 0)]
    assert result["legs"][0]["hard_budget_fallback"] is True
