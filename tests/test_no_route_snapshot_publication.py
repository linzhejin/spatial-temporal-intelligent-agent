"""A verified no-path result should survive the versioned publication guard."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from agents import route_state, tools
from spatial import road_conditions
from storage import database
from storage.task_repository import claim_next_run, create_conversation, create_task_run, finish_run, get_run
from tests.test_agent_tool_route_verification import _graph, _patch_common


pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_current_graph_no_path_is_published_with_its_snapshot(tmp_path, monkeypatch):
    event_file = tmp_path / "conditions.json"
    event_file.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(event_file))
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(event_file))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)

    graph = _graph()
    _patch_common(monkeypatch, graph)
    monkeypatch.setattr(tools, "_data_version_for_graph", lambda _graph: "data-test")
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-test")

    def no_path(**_kwargs):
        raise tools.RouteNoPathError("no graph path")

    monkeypatch.setattr(tools, "compute_route", no_path)
    answer, artifact = tools.execute_tool("plan_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"}, "strategy": "shortest",
    }, {"query": "从起点到终点", "_route_graph_snapshot": graph,
        "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert artifact is None and answer["reason_code"] == "graph_no_path"

    database.initialize(None)
    owner, _, conversation = create_conversation(None)
    queued = create_task_run(None, owner, conversation, "从起点到终点", str(uuid4()),
                             datetime.now(timezone.utc) + timedelta(minutes=2))
    assert claim_next_run(None, "no-route-test", 30)["run_id"] == queued["run_id"]
    finish_run(None, queued["run_id"], "no-route-test", status="completed", result={
        "task_type": "path_planning", "response_kind": "no_route", "message": "图内没有可通行路线。",
        "route_state": answer["route_request"], "no_path_evidence": {"reason_code": "graph_no_path"},
    })
    stored = get_run(None, owner, queued["run_id"])
    assert stored["status"] == "completed"
    assert stored["result"]["response_kind"] == "no_route"
    assert stored["result"]["route_state"]["data_version"] == "data-test"
    assert stored["result"]["route_state"]["road_condition_version"] == route_state.current_road_condition_version([])


@pytest.mark.parametrize("version_case", ["missing", "stale_event"])
def test_unversioned_or_stale_graph_no_path_is_not_published(tmp_path, monkeypatch, version_case):
    event_file = tmp_path / "conditions.json"
    event_file.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(event_file))
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(event_file))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-test")
    snapshot = {"start": {"name": "起点"}, "end": {"name": "终点"},
                "travel_mode": "walk", "strategy": {"name": "shortest"}, "hard_constraints": {}}
    if version_case == "stale_event":
        snapshot.update(data_version="data-test", road_condition_version="obsolete-event-version")

    database.initialize(None)
    owner, _, conversation = create_conversation(None)
    queued = create_task_run(None, owner, conversation, "从起点到终点", str(uuid4()),
                             datetime.now(timezone.utc) + timedelta(minutes=2))
    assert claim_next_run(None, "stale-no-route-test", 30)["run_id"] == queued["run_id"]
    finish_run(None, queued["run_id"], "stale-no-route-test", status="completed", result={
        "task_type": "path_planning", "response_kind": "no_route", "message": "图内没有可通行路线。",
        "route_state": snapshot, "no_path_evidence": {"reason_code": "graph_no_path"},
    })
    stored = get_run(None, owner, queued["run_id"])
    assert stored["status"] == "failed"
    assert stored["result"]["response_kind"] != "no_route"
