"""Candidate maintenance must keep whole-trip budgets across prior legs."""

from copy import deepcopy

import pytest

from agents import planner, route_state
from agents.result_maintenance import capture_dependencies
from agents.route_state import current_road_condition_version
from tests.test_place_result_maintenance import _leg, _pending_trip
from tests.test_reachability_time_conditions import reachability_network  # noqa: F401


@pytest.fixture
def pending_with_budget(monkeypatch):
    pending = _pending_trip()
    pending["constraints"].update({"max_distance_m": 500, "max_duration_s": 600,
                                   "excluded_edge_ids": ["[9,10,0]"]})
    leg = pending["completed_legs"][0]
    leg["recommended_length_m"] = leg["distance_m"]
    leg["route_state"]["hard_constraints"].update(pending["constraints"])
    pending["result_dependencies"] = capture_dependencies([leg], task_revision=0)
    pending["result_dependencies"]["candidate_snapshot"] = {
        "data_version": "data-test", "road_condition_version": current_road_condition_version([]),
    }
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr(route_state, "current_data_version", lambda *_: "data-test")
    return pending


def test_revising_dining_uses_remaining_whole_trip_budget(pending_with_budget, monkeypatch):
    pending = pending_with_budget
    original = deepcopy(pending)
    calls = []
    restaurant = {"poi_id": "poi_311", "name": "工学部餐厅", "subcategory": "restaurant", "distance_m": 200}

    def execute(name, args, _context):
        calls.append((name, deepcopy(args)))
        assert name == "search_poi_candidates"
        return {"data_version": "data-test", "road_condition_version": current_road_condition_version([])}, {
            "candidates": [restaurant],
        }

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    result = planner.run_agent("不吃食堂，换成餐厅", context={"place_request": pending})
    assert result["response_kind"] == "candidates"
    assert calls[0][1]["constraints"] == {
        "avoid_steps": True, "max_distance_m": 380, "max_duration_s": 480,
        "excluded_edge_ids": ["[9,10,0]"],
    }
    assert "天气" in result["message"] and "整程" in result["message"]
    assert result["place_request"]["constraints"] == original["constraints"]
    assert pending == original


def test_stale_prefix_is_not_used_to_claim_remaining_budget(pending_with_budget, monkeypatch):
    pending = pending_with_budget
    pending["completed_legs"][0]["route_state"]["road_condition_version"] = "old-roads"
    original = deepcopy(pending)
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    result = planner.run_agent("不吃食堂，换成餐厅", context={"place_request": pending})
    assert result["response_kind"] == "clarify"
    assert not result.get("candidates") and not calls
    assert "前序" in result["message"] or "路况" in result["message"]
    assert result["place_request"]["constraints"] == original["constraints"]
    assert pending == original


def test_missing_prefix_dependencies_do_not_authorize_reusing_budget(pending_with_budget, monkeypatch):
    pending = pending_with_budget
    pending["result_dependencies"] = None
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    result = planner.run_agent("不吃食堂，换成餐厅", context={"place_request": pending})
    assert result["response_kind"] == "clarify" and not calls
    assert "前序" in result["message"] and "依赖" in result["message"]
    assert result["place_request"]["constraints"] == pending["constraints"]


def test_invalid_zero_time_budget_is_not_silently_dropped(pending_with_budget, monkeypatch):
    pending = pending_with_budget
    pending["constraints"] = {"avoid_steps": True, "max_duration_s": 0}
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    result = planner.run_agent("不吃食堂，换成餐厅", context={"place_request": pending})
    assert result["response_kind"] == "clarify" and not calls
    assert "上限" in result["message"]
    assert result["place_request"]["constraints"]["max_duration_s"] == 0


def test_distinct_nearby_anchor_is_not_mistaken_for_budget_origin(pending_with_budget, monkeypatch):
    pending = pending_with_budget
    pending["search_anchor"] = {"type": "poi", "name": "总图", "poi_id": "poi_001"}
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    result = planner.run_agent("不吃食堂，换成餐厅", context={"place_request": pending})
    assert result["response_kind"] == "clarify"
    assert not result.get("candidates") and not calls
    assert "参照" in result["message"] and "起点" in result["message"]
    assert result["place_request"]["constraints"] == pending["constraints"]


@pytest.mark.parametrize("tool_result", [
    {"error": "reachability_incomplete", "message": "路线约束下的地点尚无法完整核验。"},
    {"candidates": [], "route_budget_scope": "current_leg_only", "total_count": 0},
])
def test_unavailable_budget_candidates_keep_the_prior_trip(pending_with_budget, monkeypatch, tool_result):
    pending = pending_with_budget
    original = deepcopy(pending)
    artifact = None if tool_result.get("error") else {"candidates": []}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_: (tool_result, artifact))
    result = planner.run_agent("不吃食堂，换成餐厅", context={"place_request": pending})
    assert result["response_kind"] == "clarify" and result["needs_input"] is True
    assert not result.get("candidates")
    assert result["place_request"]["constraints"] == original["constraints"]
    assert result["place_request"]["completed_legs"] == original["completed_legs"]
    assert not result["place_request"]["candidate_poi_ids"]
    assert pending == original


def test_remaining_budget_filters_actual_candidates_after_dining_revision(
        pending_with_budget, reachability_network, monkeypatch):
    pending = pending_with_budget
    _graph, conditions = reachability_network
    pending["constraints"]["excluded_edge_ids"] = []
    leg = pending["completed_legs"][0]
    leg["route_state"]["hard_constraints"]["excluded_edge_ids"] = []
    leg["route_state"]["road_condition_version"] = current_road_condition_version(conditions)
    pending["result_dependencies"] = capture_dependencies([leg], task_revision=0)
    pending["result_dependencies"]["candidate_snapshot"] = {
        "data_version": "data-test", "road_condition_version": current_road_condition_version(conditions),
    }
    monkeypatch.setattr(planner.agent_tools, "_data_version_for_graph", lambda _graph: "data-test")

    result = planner.run_agent("换成食堂", context={"place_request": pending})
    assert result["response_kind"] == "candidates"
    assert [candidate["poi_id"] for candidate in result["candidates"]] == ["intermediate"]
    assert result["place_request"]["constraints"]["max_distance_m"] == 500
    assert result["place_request"]["constraints"]["max_duration_s"] == 600
    assert "天气" in result["message"] and "整程" in result["message"]


@pytest.mark.parametrize("next_minutes, expected", [(5, "route"), (9, "clarify")])
def test_selected_dining_candidate_keeps_and_checks_whole_trip_budget(
        pending_with_budget, monkeypatch, next_minutes, expected):
    pending = pending_with_budget
    restaurant = {"poi_id": "poi_311", "name": "工学部餐厅", "subcategory": "restaurant", "distance_m": 200}
    route_calls = []

    def execute(name, args, _context):
        if name == "search_poi_candidates":
            return {"data_version": "data-test", "road_condition_version": current_road_condition_version([])}, {
                "candidates": [restaurant],
            }
        assert name == "plan_route"
        route_calls.append(deepcopy(args))
        route = _leg(args["start"], args["end"], [[114.36, 30.54], [114.37, 30.54]], [2, 3, 0],
                     distance=200, minutes=next_minutes)
        route["recommended_length_m"] = 200
        route["route_state"]["hard_constraints"].update(pending["constraints"])
        return {}, {"route": route}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [{"id": "poi_311", "name": "工学部餐厅"}])
    changed = planner.run_agent("不吃食堂，换成餐厅", context={"place_request": pending})
    final = planner.run_agent("就去这里", context={
        "place_request": changed["place_request"], "selected_poi_id": "poi_311",
    }, proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert route_calls[0]["constraints"]["max_distance_m"] == 500
    assert route_calls[0]["constraints"]["max_duration_s"] == 600
    assert final["response_kind"] == expected
    if expected == "route":
        assert final["route"]["distance_m"] == 320
        assert final["route"]["duration_min"] == 7
        assert final["route"]["route_state"]["hard_constraints"]["max_duration_s"] == 600
    else:
        assert final.get("route") is None and final["needs_input"] is True
