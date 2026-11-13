"""Model-led tour requests must keep visited places separate from departure."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from agents import tools
from agents.dialogue import Dialogue
from tests.test_model_dialogue import reply


STOPS = ["老图书馆", "凌波门"]
QUERY = "带朋友去老图书馆和凌波门逛逛"


def make_dialogue(query, replies, *, context=None, history=None, coord_start=None):
    model = Mock(side_effect=replies)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=model)))
    calls = []

    def execute(name, args, _context):
        calls.append((name, deepcopy(args)))
        if name == "ask_user":
            return tools._tool_ask_user(args, {})
        start = args.get("start") or {"type": "poi", "name": STOPS[0]}
        state = {"start": start, "end": {"type": "poi", "name": STOPS[-1]}}
        if name == "plan_tour":
            state["tour"] = {"pois": [{"name": name} for name in args.get("poi_names", [])]}
        elif name == "plan_via_route":
            state["via"] = {"name": STOPS[0]}
        return {"status": "ok"}, {"route": {"route_state": state,
                "route_verification": {"status": "pass"}}, "route_kind": "tour"}

    dialogue = Dialogue(query, context=context, history=history, coord_start=coord_start,
                        execute_tool=execute, client_factory=lambda: client, trace=lambda _event: None)
    return dialogue, model, calls


@pytest.fixture(autouse=True)
def model_environment(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "unit-test-only")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))


def initial_task(**extra):
    return {"relation": "new", "operation": "itinerary", "goal": QUERY,
            "requirements": ["游览老图书馆", "游览凌波门"], "tour_stops": STOPS, **extra}


def test_missing_departure_does_not_turn_first_stop_into_origin():
    dialogue, _, calls = make_dialogue(QUERY, [
        reply(update_task=initial_task(), plan_tour={"poi_names": STOPS, "loop": False}),
        reply("路线已规划。"),
    ])

    result = dialogue.run()

    assert result["response_kind"] == "clarify"
    assert result["route"] is None
    assert result["clarify"]["slot"] == "start"
    assert result["place_request"]["pending_slot"] == "start"
    assert result["place_request"]["semantic_request"]["tour_stops"] == STOPS
    assert not result["place_request"].get("start")
    assert not calls


def test_direct_model_question_retains_stops_and_the_missing_slot():
    dialogue, _, calls = make_dialogue(QUERY, [
        reply(update_task=initial_task(), ask_user={"question": "你们从哪里出发？", "slot": "start"}),
    ])

    result = dialogue.run()

    assert result["response_kind"] == "clarify"
    assert result["place_request"]["pending_slot"] == "start"
    assert result["place_request"]["semantic_request"]["tour_stops"] == STOPS
    assert [item[0] for item in calls] == ["ask_user"]


def test_origin_answer_continues_same_task_and_keeps_both_stops():
    first, _, _ = make_dialogue(QUERY, [
        reply(update_task=initial_task(), plan_tour={"poi_names": STOPS, "loop": False}),
    ])
    pending = first.run()["place_request"]
    context = {"available_tasks": [{"task_id": "tour-task", "revision": 3,
                                    "state": {"place_request": pending}}]}
    second, _, calls = make_dialogue("从信图出发", [
        reply(update_task={"relation": "continue", "task_id": "tour-task", "task_revision": 3,
                           "operation": "itinerary", "start": {"type": "poi", "name": "信图"}},
              plan_tour={"loop": False}),
        reply("路线已规划。"),
    ], context=context, history=[{"role": "user", "content": QUERY},
                                {"role": "assistant", "content": "你们从哪里出发？"}])
    second.bind_task = lambda **_kwargs: {"task_id": "tour-task", "task_revision": 4}

    result = second.run()

    assert result["response_kind"] == "route"
    assert calls[0][0] == "plan_tour"
    assert calls[0][1]["start"]["name"] == "信图"
    assert calls[0][1]["poi_names"] == STOPS
    assert result["completed_place_request"]["pending_slot"] is None


def test_start_in_tool_call_alone_must_be_recorded_by_model_before_planning():
    dialogue, model, calls = make_dialogue("从信图出发去老图书馆和凌波门", [
        reply(update_task=initial_task(), plan_tour={"start": {"type": "poi", "name": "信图"},
                                                   "poi_names": STOPS}),
        reply(update_task={"relation": "continue", "operation": "itinerary",
                           "start": {"type": "poi", "name": "信图"}},
              plan_tour={"poi_names": STOPS}),
        reply("路线已规划。"),
    ])

    result = dialogue.run()

    assert result["response_kind"] == "route"
    assert len(calls) == 1 and calls[0][1]["start"]["name"] == "信图"
    assert "task_update_required" in str(model.call_args_list[1].kwargs["messages"])


def test_unconfirmed_start_cannot_be_reported_as_route_when_model_ignores_guard():
    dialogue, _, calls = make_dialogue(QUERY, [
        reply(update_task=initial_task(), plan_tour={"start": {"type": "poi", "name": STOPS[0]},
                                                   "poi_names": STOPS}),
        reply("路线已经规划好了。"),
    ])

    result = dialogue.run()

    assert result["response_kind"] != "route"
    assert result["route"] is None
    assert not calls


def test_explicit_map_start_is_used_as_real_departure():
    start = {"lng": 114.36, "lat": 30.54, "name": "地图起点"}
    dialogue, _, calls = make_dialogue(QUERY, [
        reply(update_task=initial_task(), plan_tour={"poi_names": STOPS}),
        reply("路线已规划。"),
    ], coord_start=start)

    result = dialogue.run()

    assert result["response_kind"] == "route"
    assert calls[0][1]["start"]["lng"] == start["lng"]


def test_explicit_independent_preview_is_not_saved_as_user_departure():
    dialogue, _, calls = make_dialogue("只看看两处景点之间的参考路线，不考虑我从哪出发", [
        reply(update_task=initial_task(tour_scope="independent"), plan_tour={"poi_names": STOPS, "loop": False}),
        reply("这条路线经过两处景点。"),
    ])

    result = dialogue.run()

    assert result["response_kind"] == "route"
    assert not calls[0][1].get("start")
    assert not result["completed_place_request"].get("start")
    assert result["completed_place_request"]["semantic_request"]["tour_scope"] == "independent"
    assert "参考" in result["message"] and "出发" in result["message"]


def test_independent_preview_keeps_the_same_boundary_with_a_via_route():
    dialogue, _, _ = make_dialogue("只看看景点之间的参考路线", [
        reply(update_task=initial_task(tour_scope="independent"), plan_via_route={
            "start": {"type": "poi", "name": STOPS[0]}, "end": {"type": "poi", "name": STOPS[1]},
            "via_points": [{"type": "poi", "name": STOPS[0]}]}),
        reply("这条路线经过两处景点。"),
    ])

    result = dialogue.run()

    assert result["response_kind"] == "route"
    assert not result["completed_place_request"].get("start")
    assert "参考" in result["message"] and "出发" in result["message"]


def test_alternative_route_tool_also_cannot_invent_departure():
    dialogue, _, calls = make_dialogue(QUERY, [
        reply(update_task=initial_task(), plan_via_route={
            "start": {"type": "poi", "name": STOPS[0]},
            "end": {"type": "poi", "name": STOPS[1]}, "via_points": []}),
        reply("路线已经规划好了。"),
    ])

    result = dialogue.run()

    assert result["response_kind"] != "route"
    assert result["route"] is None
    assert not calls


def test_route_cannot_be_delivered_when_it_omits_a_saved_stop():
    dialogue, _, calls = make_dialogue("从信图出发去老图书馆和凌波门", [
        reply(update_task=initial_task(start={"type": "poi", "name": "信图"}),
              plan_route={"end": {"type": "poi", "name": "凌波门"}, "end_turn_if_success": False}),
        reply("路线已经规划好了。"),
    ])

    result = dialogue.run()

    assert [name for name, _args in calls] == ["plan_route"]
    assert result["response_kind"] != "route"
    assert result["route"] is None
    assert result["failure_code"] == "itinerary_stops_missing"


def test_resuming_independent_preview_does_not_turn_its_anchor_into_user_origin():
    pending = {"request_id": "preview", "original_query": "只看景点间参考路线",
               "mode": "walk", "constraints": {},
               "semantic_request": {"operation": "itinerary", "tour_scope": "independent", "tour_stops": STOPS}}
    previous = {"start": {"type": "poi", "name": STOPS[0]}, "end": {"type": "poi", "name": STOPS[1]},
                "travel_mode": "walk", "hard_constraints": {}, "strategy": {"name": "shortest"}}
    context = {"available_tasks": [{"task_id": "preview-task", "revision": 2,
                                    "state": {"place_request": pending}, "route_state": previous}]}
    dialogue, _, _ = make_dialogue("把这条参考路线改平坦一点", [], context=context)
    dialogue.bind_task = lambda **_kwargs: {"task_id": "preview-task", "task_revision": 3}

    dialogue.update({"relation": "resume_route", "task_id": "preview-task", "task_revision": 2,
                     "operation": "itinerary", "strategy": "flat"})

    assert not dialogue.pending.get("start")
    assert dialogue.intent["tour_scope"] == "independent"


def test_model_understood_tour_stops_take_precedence_over_legacy_text_heuristic(monkeypatch):
    pois = {name: {"id": str(index), "name": name, "lon": 114.36, "lat": 30.54}
            for index, name in enumerate(STOPS)}
    monkeypatch.setattr(tools, "_prepare_route_context", lambda ctx: (ctx, None))
    monkeypatch.setattr(tools, "_plan_common", lambda _args, _ctx: (object(), object(), "walk", None))
    monkeypatch.setattr(tools, "_infer_excluded_tour_poi_names", lambda _query: set())
    monkeypatch.setattr(tools, "_infer_explicit_tour_poi_names",
                        lambda _query: pytest.fail("the legacy heuristic must not replace accepted model stops"))
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda name: (pois.get(name), []))
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "_snap_wgs_to_network", lambda *_args: (1, {}, None))
    monkeypatch.setattr(tools, "importance_score", lambda _poi: 0.5)
    monkeypatch.setattr(tools, "_strategy_for_args", lambda *_args: SimpleNamespace(
        name="shortest", weights={}, detour_cap=1.0))
    seen = []

    def capture(_graph, poi_nodes, **_kwargs):
        seen.extend(poi["name"] for poi, _node in poi_nodes)
        return {"legs": [], "ordered_pois": [], "dropped": [], "loop": False}

    monkeypatch.setattr(tools, "compute_tour_route", capture)
    tools._tool_plan_tour({"poi_names": ["错误地点"]}, {
        "query": "从信图出发", "_model_dialogue": True, "_semantic_request": {"tour_stops": STOPS},
        "_route_conditions_snapshot": [],
    })

    assert seen == STOPS


def test_old_implicit_first_stop_is_not_restored_as_confirmed_departure():
    previous = {"route_kind": "tour", "start": {"name": "武汉大学早期建筑图书馆", "type": "poi"},
                "end": {"name": "凌波门", "type": "poi"}, "travel_mode": "walk",
                "hard_constraints": {}, "strategy": {"name": "shortest"},
                "tour": {"pois": [{"name": "武汉大学早期建筑图书馆", "type": "poi"},
                                   {"name": "凌波门", "type": "poi"}], "loop": False}}
    pending = {"request_id": "old", "original_query": QUERY, "mode": "walk", "constraints": {},
               "start": deepcopy(previous["start"]),
               "semantic_request": {"operation": "itinerary", "requirements": ["游览老图书馆", "游览凌波门"],
                                    "route_to_modify": previous}}
    state = {"place_request": pending, "route_spec": {"start": deepcopy(previous["start"])},
             "field_provenance": {"place_request.goal": {"source": "inference"}}}
    context = {"available_tasks": [{"task_id": "old-tour", "revision": 2,
                                    "state": state, "route_state": previous}]}
    dialogue, _, calls = make_dialogue("继续这个行程", [
        reply(update_task={"relation": "continue", "task_id": "old-tour", "task_revision": 2,
                           "operation": "itinerary"}, plan_tour={"poi_names": STOPS, "loop": False}),
    ], context=context)
    dialogue.bind_task = lambda **_kwargs: {"task_id": "old-tour", "task_revision": 3}

    result = dialogue.run()

    assert result["response_kind"] == "clarify"
    assert not result["place_request"].get("start")
    assert result["place_request"]["pending_slot"] == "start"
    assert not calls
    assert context["available_tasks"][0]["state"]["place_request"]["start"] == previous["start"]


def test_explicitly_saved_origin_on_an_old_tour_is_preserved():
    previous = {"route_kind": "tour", "start": {"name": "老图书馆", "type": "poi"},
                "tour": {"pois": [{"name": "老图书馆"}, {"name": "凌波门"}]}}
    pending = {"request_id": "explicit", "original_query": "从老图书馆出发去凌波门", "mode": "walk",
               "start": deepcopy(previous["start"]), "semantic_request": {"operation": "itinerary"}}
    state = {"place_request": pending, "route_spec": {"start": deepcopy(previous["start"])},
             "field_provenance": {"place_request.start": {"source": "inference"}}}
    context = {"available_tasks": [{"task_id": "old-explicit", "revision": 1,
                                    "state": state, "route_state": previous}]}
    dialogue, _, _ = make_dialogue("继续", [], context=context)
    dialogue.bind_task = lambda **_kwargs: {"task_id": "old-explicit", "task_revision": 2}

    dialogue.update({"relation": "continue", "task_id": "old-explicit", "task_revision": 1,
                     "operation": "itinerary"})

    assert dialogue.pending["start"]["name"] == "老图书馆"
