from copy import deepcopy

import networkx as nx
import pytest

from experiments.e3_independent_audit import audit_route_response


@pytest.fixture
def case():
    graph = nx.MultiDiGraph()
    for u, v, length in ((1, 2, 150), (2, 4, 150), (1, 3, 200), (3, 4, 200)):
        graph.add_edge(u, v, key=0, length=length, highway="footway", slope_level=1)
    task = {"start_node": 1, "end_node": 4, "mode": "walk", "strategy": "shortest",
            "constraints": {}, "data_version": "graph-v1", "road_condition_version": "events-v1",
            "events": []}
    result = {"recommended_edge_ids": [[1, 2, 0], [2, 4, 0]],
              "recommended_length_m": 300.0, "distance_m": 300.0,
              "duration_min": 4.0, "mode": "walk", "strategy": {"name": "shortest"},
              "road_condition_version": "events-v1", "road_conditions_applied": 0,
              "route_events": [], "road_condition_overview": [],
              "route_state": {"travel_mode": "walk", "hard_constraints": {},
                              "strategy": {"name": "shortest"}, "data_version": "graph-v1",
                              "road_condition_version": "events-v1"},
              "route_verification": {"status": "fail"}}
    return graph, graph.copy(), task, result


def score(case, **kwargs):
    return audit_route_response(*case, **kwargs)


def test_valid_delivery_is_independent_of_product_verification_flag(case):
    report = score(case)
    assert report["status"] == "pass", report
    assert report["metrics"]["physical_length_m"] == 300
    assert report["metrics"]["oracle_cost"] == 300
    case[3]["route_verification"]["status"] = "pass"
    assert score(case) == report


def test_event_version_in_state_is_sufficient_for_replan_contract(case):
    del case[3]["road_condition_version"]
    assert score(case)["status"] == "pass"


def test_state_event_version_must_match_even_if_other_fields_match(case):
    case[3]["evidence"] = {"road_condition_version": "events-v1"}
    case[3]["route_state"]["road_condition_version"] = "stale"
    assert "road_condition_version_mismatch" in score(case)["issues"]


def test_present_top_level_event_version_must_match(case):
    case[3]["road_condition_version"] = "stale"
    assert "road_condition_version_mismatch" in score(case)["issues"]


def test_present_evidence_event_version_must_match(case):
    del case[3]["road_condition_version"]
    case[3]["evidence"] = {"road_condition_version": "stale"}
    assert "evidence_road_condition_version_mismatch" in score(case)["issues"]


@pytest.mark.parametrize("edges", [None, [[1, 2]], [[1, 2, 9], [2, 4, 0]],
                                  [[1, 2, 0], [3, 4, 0]], [[True, 2, 0], [2, 4, 0]]])
def test_missing_malformed_wrong_key_or_discontinuous_edge_chain_fails(case, edges):
    case[3]["recommended_edge_ids"] = edges
    assert score(case)["status"] == "fail"


def test_equal_cost_on_a_closed_edge_is_not_a_valid_route(case):
    case[1].remove_edge(1, 2, 0)
    case[1][1][3][0]["length"] = 150
    case[1][3][4][0]["length"] = 150
    report = score(case)
    assert report["status"] == "fail"
    assert "edge_not_in_effective_graph" in report["issues"]


@pytest.mark.parametrize("field,value", [("mode", "bike"), ("recommended_length_m", 299),
                                        ("distance_m", 500), ("duration_min", 2),
                                        ("road_condition_version", "stale")])
def test_tampered_delivery_fields_fail(case, field, value):
    case[3][field] = value
    assert score(case)["status"] == "fail"


def test_same_length_parallel_edge_and_reverse_direction_are_distinct(case):
    case[0].add_edge(1, 2, key=1, length=150)
    case[3]["recommended_edge_ids"][0] = [1, 2, 1]
    assert score(case)["status"] == "fail"
    case[3]["recommended_edge_ids"][0] = [2, 1, 0]
    assert score(case)["status"] == "fail"


def test_suboptimal_but_feasible_route_fails_distance_optimality(case):
    case[3].update(recommended_edge_ids=[[1, 3, 0], [3, 4, 0]],
                   recommended_length_m=400, distance_m=400, duration_min=5.3)
    report = score(case)
    assert report["checks"]["feasibility"]["status"] == "pass"
    assert report["checks"]["optimality"]["status"] == "fail"


def test_slowdown_changes_eta_but_not_shortest_distance_objective(case):
    case[1][1][2][0]["audit_duration_multiplier"] = 2.0
    report = score(case)
    assert report["checks"]["optimality"]["status"] == "pass"
    assert "duration_mismatch" in report["issues"]
    case[3]["duration_min"] = 6.0
    assert score(case)["status"] == "pass"


def test_steps_and_slope_eta_is_recomputed_on_exact_edges(case):
    case[0][1][2][0].update(highway="['steps']", slope_level=5)
    case[3]["duration_min"] = 12.0
    assert score(case)["status"] == "pass"


def test_constraint_semantics_and_actual_edge_constraint_are_checked(case):
    case[2]["constraints"] = {"avoid_steps": True}
    case[3]["route_state"]["hard_constraints"] = {"avoid_steps": True}
    case[0][1][2][0]["highway"] = "steps"
    assert "steps_constraint_violated" in score(case)["issues"]
    case[0][1][2][0]["highway"] = "footway"
    case[3]["route_state"]["hard_constraints"] = {}
    assert "task_constraints_mismatch" in score(case)["issues"]


def test_budget_check_uses_physical_length_and_recomputed_eta(case):
    case[2]["constraints"] = {"max_distance_m": 250, "max_duration_s": 200}
    case[3]["route_state"]["hard_constraints"] = deepcopy(case[2]["constraints"])
    report = score(case)
    assert "distance_budget_exceeded" in report["issues"]
    assert "duration_budget_exceeded" in report["issues"]


def test_nonshortest_without_independent_objective_is_unknown(case):
    case[2]["strategy"] = "scenery"
    case[3]["strategy"] = {"name": "scenery"}
    case[3]["route_state"]["strategy"] = {"name": "scenery"}
    report = score(case)
    assert report["status"] == "unknown"
    assert report["checks"]["optimality"]["status"] == "unknown"


def test_nonshortest_with_frozen_objective_can_be_verified(case):
    case[2]["strategy"] = "scenery"
    case[3]["strategy"] = {"name": "scenery"}
    case[3]["route_state"]["strategy"] = {"name": "scenery"}
    for u, v, key, attrs in case[1].edges(keys=True, data=True):
        attrs["audit_objective_cost"] = 3 if 2 in (u, v) else 1
    case[3].update(recommended_edge_ids=[[1, 3, 0], [3, 4, 0]],
                   recommended_length_m=400, distance_m=400, duration_min=5.3)
    assert score(case)["status"] == "pass"
    assert score(case)["metrics"]["oracle_cost"] == 2


def with_events(case):
    notice = {"id": "notice-a", "action": "notice", "edge_ids": [[1, 2, 0]],
              "name": "路面提醒", "description": "注意路面", "start_time": 10, "end_time": 20}
    closure = {"id": "closure-b", "action": "closure", "edge_ids": [[1, 3, 0]], "name": "封路"}
    case[2]["events"] = [notice, closure]
    case[3]["route_events"] = [{k: v for k, v in notice.items() if k != "edge_ids"}]
    case[3]["road_condition_overview"] = [
        {k: e[k] for k in ("id", "action", "name")} for e in (notice, closure)]
    case[3]["road_conditions_applied"] = 2
    return case


def test_structured_event_explanations_match_current_route_and_snapshot(case):
    assert score(with_events(case))["status"] == "pass"


def test_initial_route_contract_allows_missing_overview_when_explicitly_declared(case):
    with_events(case)
    del case[3]["road_condition_overview"]
    assert score(case)["status"] == "fail"
    case[2]["require_event_overview"] = False
    assert score(case)["status"] == "pass"
    case[3]["road_conditions_applied"] = 1
    assert "event_applied_count_mismatch" in score(case)["issues"]
    case[3]["road_conditions_applied"] = 2
    case[3]["route_events"] = []
    assert "route_events_ids_mismatch" in score(case)["issues"]


@pytest.mark.parametrize("overview", [None, [], [{"id": "stale", "action": "closure"}]])
def test_optional_overview_is_still_checked_when_present(case, overview):
    with_events(case)
    case[2]["require_event_overview"] = False
    case[3]["road_condition_overview"] = overview
    assert score(case)["status"] == "fail"


@pytest.mark.parametrize("mutation", ["missing", "expired", "action", "description", "count", "duplicate"])
def test_missing_stale_or_inconsistent_event_explanation_fails(case, mutation):
    with_events(case)
    if mutation == "missing":
        case[3]["route_events"] = []
    elif mutation == "expired":
        case[2]["events"] = []
    elif mutation == "count":
        case[3]["road_conditions_applied"] = 1
    elif mutation == "duplicate":
        case[3]["route_events"] *= 2
    else:
        case[3]["route_events"][0][mutation] = "wrong"
    assert score(case)["status"] == "fail"


def test_unreachable_is_distinct_from_error_and_recovery(case):
    case[1].remove_edges_from([(1, 2, 0), (1, 3, 0)])
    case = (*case[:3], {"error": "route_not_found", "message": "没有路线"})
    assert score(case, response_status=404)["status"] == "pass"
    assert score((*case[:3], {"error": "database_unavailable"}), response_status=503)["status"] == "fail"
    case[1].add_edge(1, 2, key=0, length=150)
    assert score(case, response_status=404)["status"] == "fail"


def test_expected_version_rejection_is_a_separate_success_kind(case):
    case[2].update(expected_error="route_state_version_conflict", expected_status=409)
    report = score((*case[:3], {"error": "route_state_version_conflict"}), response_status=409)
    assert report["status"] == "pass"
    assert report["metrics"]["outcome"] == "expected_rejection"
    assert score(case)["status"] == "fail"


def test_missing_response_is_not_accepted(case):
    assert score((*case[:3], None))["status"] == "fail"


def test_task_revision_if_requested_is_checked_in_proof_binding(case):
    case[2]["task_revision"] = 3
    case[3]["route_state"]["reuse_certificate"] = {"task_revision": 2}
    assert "task_revision_mismatch" in score(case)["issues"]
    case[3]["route_state"]["reuse_certificate"]["task_revision"] = 3
    assert score(case)["status"] == "pass"


def test_invalid_unvisited_edge_weight_cannot_hide_behind_early_goal(case):
    case[0][1][3][0]["length"] = 1000
    case[0][3][4][0]["length"] = -900
    assert score(case)["status"] == "unknown"


def test_nonshortest_missing_unvisited_objective_cost_is_unknown(case):
    case[2]["strategy"] = "scenery"
    case[3]["strategy"] = {"name": "scenery"}
    case[3]["route_state"]["strategy"] = {"name": "scenery"}
    case[1][1][2][0]["audit_objective_cost"] = 1
    case[1][2][4][0]["audit_objective_cost"] = 1
    case[1][1][3][0]["audit_objective_cost"] = 100
    assert score(case)["status"] == "unknown"


def test_verified_accessibility_requires_edges_and_endpoint_evidence(case):
    case[2]["constraints"] = {"require_verified_accessibility": True}
    case[3]["route_state"]["hard_constraints"] = deepcopy(case[2]["constraints"])
    assert score(case)["status"] == "unknown"
    for edge in ((1, 2, 0), (2, 4, 0)):
        case[0][edge[0]][edge[1]][edge[2]].update(
            step_free_accessible=True, accessibility_verification_status="field_verified")
    assert score(case)["status"] == "unknown"
    case[2]["endpoint_access_truth"] = {"start": "field_verified", "end": "institution_verified"}
    assert score(case)["status"] == "pass"
