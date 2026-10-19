import networkx as nx
import pytest

import spatial.result_verifier as verifier_module
from agents.route_state import current_data_version, current_road_condition_version
from spatial.coord_transform import wgs84_to_gcj02
from spatial.result_verifier import (
    verify_display_geometry, verify_ordered_route_result, verify_route_result,
    verify_segmented_route_result,
)


def _verified(graph, result, *, road_conditions, expected_start_node=1, expected_end_node=3,
              task_revision=None):
    return verify_route_result(
        graph, result, road_conditions=road_conditions,
        expected_start_node=expected_start_node,
        expected_end_node=expected_end_node,
        task_revision=task_revision,
    )


def _graph():
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3):
        graph.add_node(node, x=114.36, y=30.535)
    graph.add_edge(1, 2, key=0, length=60.0, highway="footway")
    graph.add_edge(1, 2, key=1, length=80.0, highway="footway")
    graph.add_edge(2, 3, key=0, length=40.0, highway="footway")
    return graph


def _route(graph, events=(), *, edges=None, length=100.0, mode="walk"):
    return {
        "recommended_edge_ids": edges if edges is not None else [[1, 2, 0], [2, 3, 0]],
        "recommended_length_m": length,
        "mode": mode,
        "route_state": {
            "route_kind": "direct",
            "travel_mode": mode,
            "hard_constraints": {},
            "data_version": current_data_version(graph),
            "road_condition_version": current_road_condition_version(list(events)),
        },
    }


def test_exact_directed_edge_sequence_is_accepted():
    graph = _graph()
    route = _route(graph)
    report = _verified(graph, route, road_conditions=[], task_revision=7)
    assert report["valid"] is True
    assert report["status"] == "pass"
    assert report["issues"] == []
    assert report["measured_length_m"] == 100.0
    assert report["checks"]["route_structure"]["status"] == "pass"
    assert report["checks"]["exact_edges"]["status"] == "pass"
    assert report["checks"]["road_conditions"]["status"] == "pass"
    assert report["violations"] == []
    assert report["unknowns"] == []
    assert report["event_version"] == route["route_state"]["road_condition_version"]
    assert report["task_revision"] == 7


def test_missing_edge_ids_fail_closed_even_if_planner_reports_success():
    graph = _graph()
    result = _route(graph)
    del result["recommended_edge_ids"]
    result["success"] = True
    report = _verified(graph, result, road_conditions=[])
    assert "missing_edge_sequence" in report["issues"]
    assert report["status"] == "fail"
    assert report["valid"] is False


def test_disconnected_and_unknown_parallel_edge_are_rejected():
    graph = _graph()
    gap = _verified(graph, _route(graph, edges=[[1, 2, 0], [1, 2, 1]]), road_conditions=[])
    missing = _verified(graph, _route(graph, edges=[[1, 2, 9], [2, 3, 0]]), road_conditions=[])
    assert "discontinuous_edges" in gap["issues"]
    assert "missing_edge" in missing["issues"]


def test_distance_is_recomputed_using_selected_parallel_key():
    graph = _graph()
    report = _verified(graph, _route(graph, edges=[[1, 2, 1], [2, 3, 0]], length=100), road_conditions=[])
    assert report["measured_length_m"] == 120.0
    assert "length_mismatch" in report["issues"]


def test_reported_duration_cannot_be_faster_than_selected_edges_allow():
    graph = _graph()
    route = _route(graph)
    route["duration_min"] = 0.1
    report = _verified(graph, route, road_conditions=[])
    assert "duration_underreported" in report["issues"]


def test_reported_duration_is_recomputed_from_exact_edges_and_frozen_inputs():
    graph = _graph()
    route = _route(graph)
    route["duration_min"] = 2.0  # A deliberately inflated estimate.

    report = verify_route_result(
        graph, route, road_conditions=[], expected_start_node=1,
        expected_end_node=3, duration_context_available=True, weather_info=None,
    )

    assert report["status"] == "fail"
    assert "duration_overreported" in report["issues"]
    assert report["expected_duration_min"] == 1.3


def test_exact_duration_recheck_includes_weather_slope_and_route_penalties(monkeypatch):
    graph = _graph()
    graph[1][2][0]["slope_level"] = 4
    events = [{"id": "slowdown"}]
    route = _route(graph, events=events)
    route["duration_min"] = 0.8
    route["route_state"]["road_condition_version"] = current_road_condition_version(events)
    monkeypatch.setattr(
        "spatial.result_verifier.apply_conditions_to_graph",
        lambda _graph, _events, _mode, strict: (
            graph, {(1, 2, 0): 2.0}, set(), 1,
        ),
    )
    monkeypatch.setattr(
        "spatial.result_verifier.weather_slope_penalty",
        lambda _weather: {4: 1.5},
        raising=False,
    )

    report = verify_route_result(
        graph, route, road_conditions=events, expected_start_node=1,
        expected_end_node=3, duration_context_available=True,
        weather_info={"weather": "小雨"},
    )

    assert "duration_underreported" in report["issues"]
    assert report["expected_duration_min"] > 1.3


def test_reported_duration_includes_confirmed_slowdown(monkeypatch):
    graph = _graph()
    route = _route(graph)
    route["duration_min"] = 1.3
    events = [{"id": "slowdown"}]
    route["route_state"]["road_condition_version"] = current_road_condition_version(events)
    monkeypatch.setattr(
        "spatial.result_verifier.apply_conditions_to_graph",
        lambda _graph, _events, _mode, strict: (
            graph, {(1, 2, 0): 2.0}, set(), 1,
        ),
    )
    report = _verified(graph, route, road_conditions=events)
    assert "duration_underreported" in report["issues"]


def test_direct_route_checks_distance_and_time_budgets():
    graph = _graph()
    route = _route(graph)
    route["duration_min"] = 1.4
    route["route_state"]["hard_constraints"] = {
        "max_distance_m": 90,
        "max_duration_s": 60,
    }

    report = _verified(graph, route, road_conditions=[])

    assert "distance_budget_exceeded" in report["issues"]
    assert "duration_budget_exceeded" in report["issues"]


def test_network_distance_budget_excludes_unverified_endpoint_connectors():
    graph = _graph()
    route = _route(graph)
    route["endpoint_access"] = {
        "start": {"snap_distance_m": 500.0, "access_link_verified": False},
        "end": {"snap_distance_m": 700.0, "access_link_verified": False},
    }
    route["route_state"]["hard_constraints"] = {"max_distance_m": 101.0}

    report = _verified(graph, route, road_conditions=[])

    assert report["status"] == "pass"
    assert report["measured_length_m"] == 100.0
    assert "distance_budget_exceeded" not in report["issues"]


def test_reference_route_can_skip_budget_only_while_still_checking_edges():
    graph = _graph()
    route = _route(graph)
    route["duration_min"] = 1.4
    route["route_state"]["hard_constraints"] = {
        "max_distance_m": 90, "max_duration_s": 60,
    }

    report = verify_route_result(
        graph, route, road_conditions=[], expected_start_node=1,
        expected_end_node=3, enforce_budgets=False,
    )
    assert report["status"] == "pass"
    assert report["budgets_enforced"] is False

    route["recommended_edge_ids"] = [[1, 2, 9], [2, 3, 0]]
    invalid = verify_route_result(
        graph, route, road_conditions=[], expected_start_node=1,
        expected_end_node=3, enforce_budgets=False,
    )
    assert invalid["status"] == "fail"
    assert "missing_edge" in invalid["issues"]


def test_direct_route_requires_time_evidence_for_time_budget():
    graph = _graph()
    route = _route(graph)
    route["route_state"]["hard_constraints"] = {"max_duration_s": 120}

    report = _verified(graph, route, road_conditions=[])

    assert "missing_duration_for_budget" in report["issues"]


def test_excluded_exact_parallel_edge_is_not_accepted():
    graph = _graph()
    route = _route(graph)
    route["route_state"]["hard_constraints"] = {
        "excluded_edge_ids": ["[1,2,0]"],
    }

    report = _verified(graph, route, road_conditions=[])

    assert "excluded_edge_used" in report["issues"]


def test_verified_accessibility_request_is_unknown_without_access_evidence():
    graph = _graph()
    route = _route(graph)
    route["route_state"]["hard_constraints"] = {
        "require_verified_accessibility": True,
    }

    report = _verified(graph, route, road_conditions=[])

    assert report["status"] == "unknown"
    assert "accessibility_unverified" in report["issues"]


def test_ordered_route_rejects_malformed_constraints_without_crashing():
    graph = _graph()
    route = _via_route(graph)
    route["route_state"]["hard_constraints"] = ["bad"]

    report = verify_ordered_route_result(
        graph, route, road_conditions=[], expected_nodes=[1, 2, 3],
    )

    assert report["status"] == "fail"
    assert "invalid_constraints" in report["issues"]


def test_mode_and_hard_constraint_violations_are_rejected():
    graph = _graph()
    graph[1][2][0]["foot"] = "no"
    denied = _verified(graph, _route(graph), road_conditions=[])
    assert "mode_forbidden" in denied["issues"]
    del graph[1][2][0]["foot"]
    graph[1][2][0]["highway"] = "steps"
    constrained = _route(graph)
    constrained["route_state"]["hard_constraints"] = {"avoid_steps": True}
    report = _verified(graph, constrained, road_conditions=[])
    assert "constraint_violation" in report["issues"]


def test_event_closure_is_checked_against_frozen_snapshot():
    graph = _graph()
    events = [{
        "id": "closure-1", "type": "closure", "start_time": 0, "end_time": 0,
        "edge": {"u": 1, "v": 2, "key": 0, "snap": {"lng": 114.36, "lat": 30.535}},
    }]
    route = _route(graph, events)
    report = _verified(graph, route, road_conditions=events)
    assert "closed_edge" in report["issues"]
    assert report["valid"] is False


def test_version_and_event_mismatch_are_both_reported():
    graph = _graph()
    result = _route(graph)
    result["route_state"]["data_version"] = "data-stale"
    result["route_state"]["road_condition_version"] = "roads-stale"
    report = _verified(graph, result, road_conditions=[])
    assert "data_version_mismatch" in report["issues"]
    assert "event_version_mismatch" in report["issues"]


def test_unfrozen_event_snapshot_and_composite_route_are_not_accepted():
    graph = _graph()
    report = _verified(graph, _route(graph), road_conditions=None)
    assert "missing_event_snapshot" in report["issues"]
    assert report["checks"]["road_conditions"]["status"] == "unknown"
    assert report["unknowns"] == ["missing_event_snapshot"]
    assert report["violations"] == []
    assert report["event_version"] is None
    route = _route(graph)
    route["route_state"]["route_kind"] = "multimodal"
    report = _verified(graph, route, road_conditions=[])
    assert "unsupported_route_kind" in report["issues"]


def test_top_level_event_version_must_agree_with_state_and_snapshot():
    graph = _graph()
    result = _route(graph)
    result["road_condition_version"] = "roads-from-another-query"
    report = _verified(graph, result, road_conditions=[])
    assert "event_version_mismatch" in report["issues"]


def test_unknown_slope_is_reported_as_unknown_not_proven_violation():
    graph = _graph()
    result = _route(graph)
    result["route_state"]["hard_constraints"] = {"slope": "avoid"}
    report = _verified(graph, result, road_conditions=[])
    assert "slope_evidence_unknown" in report["issues"]
    assert report["status"] == "unknown"
    assert report["checks"]["mode_and_constraints"]["status"] == "unknown"
    assert report["violations"] == []
    assert report["unknowns"] == ["slope_evidence_unknown"]


def test_known_steep_slope_is_reported_as_a_constraint_violation():
    graph = _graph()
    graph[1][2][0]["slope_level"] = 5
    graph[2][3][0]["slope_level"] = 5
    result = _route(graph)
    result["route_state"]["hard_constraints"] = {"slope": "avoid"}
    report = _verified(graph, result, road_conditions=[])
    assert "constraint_violation" in report["issues"]
    assert report["status"] == "fail"
    assert "constraint_violation" in report["violations"]
    assert report["unknowns"] == []


def test_invalid_event_binding_fails_closed_even_with_matching_version():
    graph = _graph()
    events = [{
        "id": "stale", "type": "closure", "start_time": 0, "end_time": 0,
        "edge": {"u": 991, "v": 992, "key": 0,
                 "snap": {"lng": 114.38, "lat": 30.55}},
    }]
    report = _verified(graph, _route(graph, events), road_conditions=events)
    assert "event_snapshot_unusable" in report["issues"]


@pytest.mark.parametrize("bad_mode", [[], {"mode": "walk"}, None])
def test_malformed_mode_returns_invalid_report_instead_of_crashing(bad_mode):
    graph = _graph()
    result = _route(graph)
    result["mode"] = bad_mode
    report = _verified(graph, result, road_conditions=[])
    assert report["valid"] is False
    assert "mode_mismatch" in report["issues"]


def test_missing_independent_endpoint_nodes_is_unknown_and_not_accepted():
    graph = _graph()
    report = verify_route_result(graph, _route(graph), road_conditions=[])
    assert report["endpoint_status"] == "unknown"
    assert report["status"] == "unknown"
    assert "endpoint_unknown" in report["issues"]
    assert report["valid"] is False


def test_route_edge_endpoints_must_match_independent_endpoint_nodes():
    graph = _graph()
    report = _verified(graph, _route(graph), road_conditions=[], expected_start_node=2)
    assert report["endpoint_status"] == "mismatch"
    assert "endpoint_mismatch" in report["issues"]


def test_same_network_node_can_have_explicit_zero_edge_route():
    graph = _graph()
    result = _route(graph, edges=[], length=0.0)
    report = _verified(
        graph, result, road_conditions=[],
        expected_start_node=1, expected_end_node=1,
    )
    assert report["endpoint_status"] == "matched"
    assert report["measured_length_m"] == 0.0
    assert report["valid"] is True


def test_zero_edges_with_distinct_endpoints_is_rejected():
    graph = _graph()
    report = _verified(graph, _route(graph, edges=[], length=0), road_conditions=[])
    assert "empty_edge_sequence" in report["issues"]
    assert report["valid"] is False


def test_zero_edges_must_still_be_explicit_and_have_zero_length():
    graph = _graph()
    omitted = _route(graph, length=0)
    del omitted["recommended_edge_ids"]
    missing = _verified(graph, omitted, road_conditions=[], expected_start_node=1, expected_end_node=1)
    assert "missing_edge_sequence" in missing["issues"]
    mismatch = _verified(
        graph, _route(graph, edges=[], length=5), road_conditions=[],
        expected_start_node=1, expected_end_node=1,
    )
    assert "length_mismatch" in mismatch["issues"]


def test_actual_planner_same_node_shortest_result_has_zero_edges():
    from spatial.routing import compute_route

    graph = _graph()
    planned = compute_route(graph, 1, 1, strategy_name="shortest", road_conditions=[])
    result = _route(
        graph,
        edges=planned["recommended_edges"],
        length=planned["recommended_length_m"],
    )
    report = _verified(
        graph, result, road_conditions=[],
        expected_start_node=1, expected_end_node=1,
    )
    assert planned["recommended_edges"] == []
    assert report["valid"] is True


def _via_route(graph, *, second_edges=None, top_edges=None):
    state = _route(graph)["route_state"]
    state["route_kind"] = "via"
    first = {"recommended_edge_ids": [[1, 2, 0]], "recommended_length_m": 60.0,
             "recommended": [[1, 0], [2, 0]], "mode": "walk"}
    second = {"recommended_edge_ids": second_edges if second_edges is not None else [[2, 3, 0]],
              "recommended_length_m": 40.0, "recommended": [[2, 0], [3, 0]],
              "mode": "walk"}
    return {"route_state": state, "mode": "walk", "legs": [first, second],
            "recommended_edge_ids": top_edges if top_edges is not None else [[1, 2, 0], [2, 3, 0]],
            "recommended_length_m": 100.0,
            "recommended": [[1, 0], [2, 0], [3, 0]]}


def test_ordered_via_verifier_checks_each_leg_and_displayed_path():
    graph = _graph()
    report = verify_ordered_route_result(
        graph, _via_route(graph), road_conditions=[], expected_nodes=[1, 2, 3],
    )
    assert report["status"] == "pass"
    assert report["measured_length_m"] == 100.0


def test_ordered_via_hashes_frozen_graph_once_for_all_legs(monkeypatch):
    graph = _graph()
    route = _via_route(graph)
    original = verifier_module.current_data_version
    calls = []

    def counted(current_graph):
        calls.append(current_graph)
        return original(current_graph)

    monkeypatch.setattr(verifier_module, "current_data_version", counted)
    report = verify_ordered_route_result(
        graph, route, road_conditions=[], expected_nodes=[1, 2, 3],
    )

    assert report["status"] == "pass"
    assert calls == [graph]


def test_ordered_via_verifier_rejects_skipped_stop_and_wrong_display_edges():
    graph = _graph()
    skipped = _via_route(graph, second_edges=[[1, 2, 0]])
    skipped_report = verify_ordered_route_result(
        graph, skipped, road_conditions=[], expected_nodes=[1, 2, 3],
    )
    assert "leg_2:endpoint_mismatch" in skipped_report["issues"]
    wrong_display = _via_route(graph, top_edges=[[1, 2, 1], [2, 3, 0]])
    display_report = verify_ordered_route_result(
        graph, wrong_display, road_conditions=[], expected_nodes=[1, 2, 3],
    )
    assert "display_edge_mismatch" in display_report["issues"]


def test_ordered_via_verifier_rejects_missing_leg_or_fabricated_geometry():
    graph = _graph()
    missing = _via_route(graph)
    missing["legs"].pop()
    report = verify_ordered_route_result(
        graph, missing, road_conditions=[], expected_nodes=[1, 2, 3],
    )
    assert "leg_count_mismatch" in report["issues"]
    geometry = _via_route(graph)
    geometry["recommended"] = [[1, 0], [3, 0]]
    report = verify_ordered_route_result(
        graph, geometry, road_conditions=[], expected_nodes=[1, 2, 3],
    )
    assert "display_geometry_mismatch" in report["issues"]


def test_ordered_via_verifier_rejects_total_time_not_equal_to_leg_times():
    graph = _graph()
    route = _via_route(graph)
    route["legs"][0]["duration_min"] = 1.0
    route["legs"][1]["duration_min"] = 1.0
    route["duration_min"] = 0.5
    report = verify_ordered_route_result(
        graph, route, road_conditions=[], expected_nodes=[1, 2, 3],
    )
    assert "duration_sum_mismatch" in report["issues"]


def test_ordered_via_checks_whole_trip_budget_not_only_each_leg():
    graph = _graph()
    route = _via_route(graph)
    route["route_state"]["hard_constraints"] = {
        "max_distance_m": 90,
        "max_duration_s": 60,
    }
    route["legs"][0]["duration_min"] = 0.8
    route["legs"][1]["duration_min"] = 0.6
    route["duration_min"] = 1.4

    report = verify_ordered_route_result(
        graph, route, road_conditions=[], expected_nodes=[1, 2, 3],
    )

    assert "distance_budget_exceeded" in report["issues"]
    assert "duration_budget_exceeded" in report["issues"]


def test_segmented_route_verifies_each_mode_and_displayed_edge_chain():
    graph = _graph()
    first = {"recommended_edge_ids": [[1, 2, 0]], "recommended_length_m": 60,
             "recommended": [[1, 0], [2, 0]], "mode": "walk"}
    second = {"recommended_edge_ids": [[2, 3, 0]], "recommended_length_m": 40,
              "recommended": [[2, 0], [3, 0]], "mode": "bike"}
    state = _route(graph)["route_state"]
    state.update({"route_kind": "multimodal", "travel_mode": "walk"})
    result = {"route_state": state, "recommended_edge_ids": [[1, 2, 0], [2, 3, 0]],
              "recommended": [[1, 0], [2, 0], [2, 0], [3, 0]],
              "recommended_length_m": 100}
    segments = [
        {"route": first, "start_node": 1, "end_node": 2, "mode": "walk"},
        {"route": second, "start_node": 2, "end_node": 3, "mode": "bike"},
    ]
    report = verify_segmented_route_result(
        graph, result, road_conditions=[], expected_kind="multimodal", segments=segments,
        task_revision=3,
    )
    assert report["status"] == "pass"
    assert report["task_revision"] == 3
    assert report["event_version"] == current_road_condition_version([])
    assert report["violations"] == []
    assert report["unknowns"] == []
    result["recommended_edge_ids"] = [[1, 2, 1], [2, 3, 0]]
    changed = verify_segmented_route_result(
        graph, result, road_conditions=[], expected_kind="multimodal", segments=segments,
    )
    assert "display_edge_mismatch" in changed["issues"]


def test_segmented_route_verifier_rejects_total_time_not_equal_to_segments():
    graph = _graph()
    first = {"recommended_edge_ids": [[1, 2, 0]], "recommended_length_m": 60,
             "recommended": [[1, 0], [2, 0]], "mode": "walk", "duration_min": 1.0}
    second = {"recommended_edge_ids": [[2, 3, 0]], "recommended_length_m": 40,
              "recommended": [[2, 0], [3, 0]], "mode": "bike", "duration_min": 1.0}
    state = _route(graph)["route_state"]
    state.update({"route_kind": "multimodal", "travel_mode": "walk"})
    result = {"route_state": state, "recommended_edge_ids": [[1, 2, 0], [2, 3, 0]],
              "recommended": [[1, 0], [2, 0], [2, 0], [3, 0]],
              "recommended_length_m": 100, "duration_min": 0.5}
    report = verify_segmented_route_result(graph, result, road_conditions=[],
                                           expected_kind="multimodal", segments=[
        {"route": first, "start_node": 1, "end_node": 2, "mode": "walk"},
        {"route": second, "start_node": 2, "end_node": 3, "mode": "bike"},
    ])
    assert "duration_sum_mismatch" in report["issues"]


def test_segmented_route_rejects_wrong_leg_mode_and_distance():
    graph = _graph()
    graph[2][3][0]["bicycle"] = "no"
    first = {"recommended_edge_ids": [[1, 2, 0]], "recommended_length_m": 60,
             "recommended": [[1, 0], [2, 0]], "mode": "walk"}
    second = {"recommended_edge_ids": [[2, 3, 0]], "recommended_length_m": 40,
              "recommended": [[2, 0], [3, 0]], "mode": "bike"}
    state = _route(graph)["route_state"]
    state.update({"route_kind": "multimodal", "travel_mode": "walk"})
    result = {"route_state": state, "recommended_edge_ids": [[1, 2, 0], [2, 3, 0]],
              "recommended": [[1, 0], [2, 0], [2, 0], [3, 0]],
              "recommended_length_m": 90}
    report = verify_segmented_route_result(
        graph, result, road_conditions=[], expected_kind="multimodal", segments=[
            {"route": first, "start_node": 1, "end_node": 2, "mode": "walk"},
            {"route": second, "start_node": 2, "end_node": 3, "mode": "bike"},
        ],
    )
    assert "leg_2:mode_forbidden" in report["issues"]
    assert "length_mismatch" in report["issues"]


def test_segmented_route_rejects_malformed_first_segment_without_crashing():
    graph = _graph()
    state = _route(graph)["route_state"]
    state.update({"route_kind": "multimodal", "travel_mode": "walk"})
    result = {"route_state": state, "recommended_edge_ids": [],
              "recommended": [], "recommended_length_m": 0}

    report = verify_segmented_route_result(
        graph, result, road_conditions=[], expected_kind="multimodal",
        segments=[None],
    )

    assert report["status"] == "fail"
    assert "leg_1:invalid_result" in report["issues"]


def test_display_geometry_uses_selected_parallel_edge_shape_and_gcj_coordinates():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_edge(1, 2, key=0, geometry="LINESTRING (114.3600 30.5300, 114.3610 30.5300)")
    graph.add_edge(1, 2, key=1, geometry=(
        "LINESTRING (114.3610 30.5300, 114.3605 30.5305, 114.3600 30.5300)"
    ))
    selected = [(1, 2, 1)]
    points = [(114.3600, 30.5300), (114.3605, 30.5305), (114.3610, 30.5300)]
    displayed = []
    for lon, lat in points:
        converted = wgs84_to_gcj02(lon, lat)
        displayed.append({"lng": round(converted[0], 6), "lat": round(converted[1], 6)})
    assert verify_display_geometry(graph, selected, displayed)["status"] == "pass"
    assert "display_geometry_mismatch" in verify_display_geometry(
        graph, selected, [displayed[0], displayed[-1]],
    )["issues"]


def test_display_geometry_with_unreadable_source_is_unknown():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_edge(1, 2, key=0, geometry="LINESTRING INVALID")
    report = verify_display_geometry(graph, [(1, 2, 0)], [])
    assert report["status"] == "unknown"
    assert "source_geometry_unreadable" in report["issues"]
