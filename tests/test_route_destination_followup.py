"""A named-place follow-up must retain the right completed route."""
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest

from agents import planner
from agents.context_builder import build_agent_context, is_route_followup
from storage import task_repository


PREVIOUS = {
    "schema_version": 1, "route_id": "previous-route", "route_kind": "via",
    "original_query": "从总图经过卓尔体育馆到梅园操场，不走台阶",
    "start": {"type": "poi", "name": "武汉大学图书馆(总馆)"},
    "end": {"type": "poi", "name": "武汉大学梅园操场"},
    "via": {"type": "poi", "name": "卓尔体育馆"},
    "tour": None, "legs": [], "travel_mode": "walk",
    "hard_constraints": {"avoid_steps": True},
    "strategy": {"name": "flat", "source": "explicit_nl"},
    "data_version": "test-data", "road_condition_version": "test-roads",
}


@pytest.mark.parametrize("query", [
    "桂园操场不行吗",
    "那桂园操场呢？",
    "换成桂园操场吧",
    "终点换成桂园操场",
])
def test_named_destination_followup_replans_saved_route(query):
    previous = deepcopy(PREVIOUS)
    context = build_agent_context(query, [
        {"role": "user", "content": previous["original_query"]},
        {"role": "assistant", "content": "路线已规划。"},
    ], previous)
    route = {"recommended": [[1, 2], [2, 3]], "start_name": previous["start"]["name"],
             "end_name": "武汉大学桂园操场", "mode": "walk"}
    with patch.object(planner.agent_tools, "execute_tool",
                      return_value=(route, {"route": route, "route_kind": "via"})) as execute:
        result = planner.run_agent(query, context=context)

    assert is_route_followup(query)
    assert context["previous_route_state"] == previous
    name, args, _ = execute.call_args.args
    assert name == "plan_via_route"
    assert args["start"] == previous["start"]
    assert "桂园操场" in args["end"]["name"]
    assert args["via_name"] == "卓尔体育馆"
    assert args["mode"] == "walk"
    assert args["strategy"] == "flat"
    assert args["constraints"] == {"avoid_steps": True}
    assert result["response_kind"] == "route"
    assert previous == PREVIOUS


@pytest.mark.parametrize("query", [
    "天气怎么样？", "桂园操场现在开放吗？", "介绍一下桂园操场", "从信图到樱顶",
])
def test_unrelated_questions_and_fresh_routes_do_not_inherit_previous_route(query):
    assert not is_route_followup(query)
    assert "previous_route_state" not in build_agent_context(query, [], PREVIOUS)


@pytest.mark.parametrize("latest,expected", [
    ({"status": "completed", "result": {"response_kind": "route", "route_state": PREVIOUS,
                                          "route_verification": {"status": "pass"}}}, True),
    ({"status": "completed", "result": {"response_kind": "no_route", "route_state": PREVIOUS}}, False),
    ({"status": "completed", "result": {"response_kind": "route", "route_state": PREVIOUS,
                                          "route_verification": {"status": "unknown"}}}, False),
    ({"status": "completed", "result": {"response_kind": "chat"}}, False),
    ({"status": "needs_input", "result": {}}, False),
    (None, False),
])
def test_elliptical_followup_does_not_reuse_a_route_from_an_older_task(monkeypatch, latest, expected):
    calls = []

    class Cursor:
        def __init__(self, value): self.value = value
        def fetchone(self): return self.value
        def fetchall(self): return self.value

    class Connection:
        def execute(self, sql, _params):
            calls.append(sql)
            values = [
                {"run_id": "run-2", "task_id": "task-2", "task_revision": 0,
                 "cancel_requested": False, "conversation_id": "conversation-1",
                 "revision": 0, "state": {"route_spec": {}}, "owner_id": "owner-1"},
                {"seq": 30, "content": "桂园操场不行吗"},
                [{"role": "assistant", "content": "上一轮的回答"}],
                {"route_state": PREVIOUS, "task_id": "task-1", "source_seq": 20},
                [],
                latest,
            ]
            return Cursor(values[len(calls) - 1])

    @contextmanager
    def connect(_url):
        yield Connection()

    monkeypatch.setattr(task_repository.database, "connect", connect)
    result = task_repository.get_run_input("postgres://unused", "run-2")
    assert bool(result["previous_route_state"]) is expected
    assert result["route_context_uncertain"] is (not expected)


def test_ambiguous_location_followup_asks_for_clarification_without_planning(monkeypatch):
    from agents import context_builder

    choices = [{"name": "文理三教", "id": "a"}, {"name": "工学三教", "id": "b"}]
    monkeypatch.setattr(context_builder, "named_destination_followup", lambda query: choices)
    with patch.object(planner.agent_tools, "execute_tool") as execute:
        result = planner.run_agent("三教不行吗", context={"previous_route_state": PREVIOUS})
    assert result["response_kind"] == "clarify"
    assert "多个地点" in result["message"]
    execute.assert_not_called()
