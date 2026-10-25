"""Event replay derives observations from persisted tool evidence, not prose."""

import importlib

import pytest

from experiments.scenario_live_adapter import build_poi_lookup, project_live_run


LOOKUP = build_poi_lookup([
    {"id": "poi_001", "name": "起点"},
    {"id": "poi_015", "name": "终点"},
])


def _project(run):
    try:
        module = importlib.import_module("experiments.event_live_replay")
    except ImportError:
        pytest.fail("event-only live projector is missing")
    return module.project_event_run(run, LOOKUP)


def _run(kind="route", *, event_ids=None, query_outcome="success"):
    events = [{"event_type": "agent_tool_trace_started", "payload": {}}]
    if event_ids is not None:
        events.append({"event_type": "agent_tool_call", "payload": {
            "tool_name": "list_road_conditions", "argument_keys": [],
            "outcome": query_outcome, "artifact_kind": None,
            "road_event_ids": event_ids,
        }})
    events.append({"event_type": "agent_tool_call", "payload": {
        "tool_name": "plan_route", "argument_keys": ["start", "end"],
        "outcome": "success" if kind == "route" else "tool_error",
        "artifact_kind": "route" if kind == "route" else None,
    }})
    result = {
        "response_kind": kind, "message": "因封闭不可达。" if kind == "no_route" else "路线已规划。",
        "route_state": {
            "start": {"name": "起点"}, "end": {"name": "终点"},
            "travel_mode": "walk", "strategy": {"name": "shortest"},
            "hard_constraints": {},
        },
    }
    if kind == "route":
        result["recommended_edge_ids"] = [[1, 2, 0]]
    return {"status": "completed", "events": events, "result": result}


def test_event_projection_adds_ids_only_from_successful_condition_trace():
    run = _run(event_ids=["simulated_2", "simulated_1"])
    projected = _project(run)
    assert projected["road_event_ids"] == ["simulated_1", "simulated_2"]
    assert projected["outputs"] == ["route", "road_events"]
    assert "road_event_ids" not in project_live_run(run, LOOKUP)

    claimed = _run(event_ids=None)
    claimed["result"]["message"] = "我知道 simulated_1 正在封路。"
    assert "road_event_ids" not in _project(claimed)
    assert "road_events" not in _project(claimed)["outputs"]
    assert "road_event_ids" not in _project(_run(
        event_ids=["simulated_1"], query_outcome="tool_error"
    ))


def test_event_projection_maps_completed_no_route_without_inventing_edges():
    projected = _project(_run(kind="no_route", event_ids=["simulated_1"]))
    assert projected["status"] == "infeasible"
    assert projected["kind"] == "no_route"
    assert projected["route_edge_ids"] == []
    assert projected["road_event_ids"] == ["simulated_1"]
    assert "infeasibility_explanation" in projected["outputs"]
