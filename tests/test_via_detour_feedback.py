"""A requested detour compares against a verified direct route, not the via shortest."""

import math
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents import planner


@pytest.mark.parametrize("direct,total,expected", [
    (677.1, 821.1, "多走约 144 米"),
    (821.1, 821.1, "没有增加路网距离"),
    (900.0, 821.1, "少走约 79 米"),
])
def test_explicit_detour_answer_uses_verified_direct_reference(direct, total, expected):
    route = {"via": {"name": "咖啡馆"}, "recommended_length_m": total,
             "direct_reference_length_m": direct, "direct_reference_verified": True}
    assert expected in planner._detour_feedback(route, "请告诉我途经咖啡馆绕行多少距离")


@pytest.mark.parametrize("direct,verified", [(None, True), (0, True), (math.nan, True),
                                               (677.1, False), (True, True)])
def test_unverified_direct_reference_never_fabricates_a_distance(direct, verified):
    route = {"via": {"name": "咖啡馆"}, "recommended_length_m": 821.1,
             "direct_reference_length_m": direct, "direct_reference_verified": verified}
    assert "无法核实" in planner._detour_feedback(route, "绕行多少距离")


def test_non_detour_request_has_no_extra_message():
    assert planner._detour_feedback({"via": {"name": "咖啡馆"}}, "我要去喝咖啡") == ""


def test_original_and_root_question_do_not_duplicate_detour_answer():
    question = "绕行多少距离"
    route = {"via": {"name": "咖啡馆"}, "recommended_length_m": 821.1,
             "direct_reference_length_m": 677.1, "direct_reference_verified": True,
             "route_state": {"original_query": question}}
    assert planner._route_message_with_detour(route, question).count("多走约 144 米") == 1


def test_legacy_place_state_recovers_original_question_from_completed_leg():
    previous = {"original_query": "去吃饭", "completed_legs": [
        {"route_state": {"original_query": "先打球再吃饭，告诉我绕行多少距离"}},
    ]}
    restored = planner._place_request_state("去吃饭", previous=previous)
    assert restored["root_query"] == "先打球再吃饭，告诉我绕行多少距离"


def test_failed_comparison_clears_older_reference(monkeypatch):
    route = {"via": {"name": "篮球场"}, "recommended_length_m": 821.1,
             "direct_reference_length_m": 677.1, "direct_reference_verified": True}
    monkeypatch.setattr(planner, "_place_route_join_issue", lambda *_: "快照不一致")
    legs = [{"route_state": {"hard_constraints": {}}}, {"route_state": {}}]
    planner._add_combined_direct_reference(route, legs, "绕行多少距离", None)
    assert "direct_reference_verified" not in route
    assert "无法核实" in planner._detour_feedback(route, "绕行多少距离")


def test_model_route_response_includes_requested_detour_when_model_omits_it():
    call = SimpleNamespace(id="route-1", function=SimpleNamespace(
        name="plan_via_route", arguments=json.dumps({"via_name": "咖啡馆"})))
    first = SimpleNamespace(content=None, tool_calls=[call])
    first.model_dump = lambda exclude_none=False: {"role": "assistant", "tool_calls": []}
    second = SimpleNamespace(content="路线已经规划。", tool_calls=[])
    replies = [SimpleNamespace(choices=[SimpleNamespace(message=first)]),
               SimpleNamespace(choices=[SimpleNamespace(message=second)])]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = {"start_name": "桂园食堂", "end_name": "樱顶", "via": {"name": "咖啡馆"},
             "mode": "walk", "recommended_length_m": 821.1,
             "direct_reference_length_m": 677.1, "direct_reference_verified": True}
    with patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", return_value=(route, {"route": route, "route_kind": "via"})):
        result = planner.run_agent("从桂园食堂到樱顶，途中顺便买咖啡，告诉我绕行多少距离")
    assert result["response_kind"] == "route"
    assert "多走约 144 米" in result["message"]
