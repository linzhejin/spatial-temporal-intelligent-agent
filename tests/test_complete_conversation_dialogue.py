"""Regression contracts for model-led turns and complete conversation input."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import config
import pytest
from agents.dialogue import Dialogue
from tests.test_model_dialogue import place, reply


def make_dialogue(query, *, history=None, context=None, replies=()):
    create = Mock(side_effect=list(replies))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    dialogue = Dialogue(
        query, history=history, context=context, execute_tool=Mock(),
        client_factory=lambda: client, trace=lambda _item: None,
    )
    return dialogue, create


def test_task_update_schema_requires_an_explicit_hard_constraint_decision():
    from agents.dialogue import tool_schemas
    update = next(item["function"] for item in tool_schemas() if item["function"]["name"] == "update_task")
    assert "constraints" in update["parameters"]["required"]
    assert "avoid_steps" in update["parameters"]["properties"]["constraints"]["properties"]


def test_read_only_place_resolution_can_precede_task_binding(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    dialogue, _ = make_dialogue("从甲地出发", replies=[
        reply(resolve_poi={"name": "甲地"}),
        reply(update_task={"relation": "new", "operation": "route", "start": {"type": "poi", "name": "甲地"}}),
        reply("已保存起点。"),
    ])
    dialogue.bind_task = lambda **_args: {"task_id": "task", "task_revision": 1}
    executed = []
    dialogue.execute_tool = lambda name, args, _ctx: (executed.append((name, args)) or
        ({"found": True, "poi": {"poi_id": "poi-test", "name": "甲地"}}, None))
    result = dialogue.run()
    assert executed == [("resolve_poi", {"name": "甲地"})]
    assert result.get("failure_code") is None


def test_prebound_resolved_online_start_survives_new_task_creation(monkeypatch):
    from agents import tools
    candidate = place("amap:RESOLVED123", "示例门店")
    dialogue, _ = make_dialogue("从示例门店出发")
    dialogue.execute_tool = lambda *_args: ({"found": True, "poi": candidate}, None)
    monkeypatch.setattr(tools.amap_poi, "lookup_by_id", lambda _ident: candidate)
    monkeypatch.setattr(tools.amap_poi, "choose_match", lambda _name, rows, **_kwargs: rows[0])
    monkeypatch.setattr(tools, "filter_reported_absent_provider_results", lambda rows: rows)
    dialogue.spatial("resolve_poi", {"name": candidate["name"]})
    dialogue.update({"relation": "new", "operation": "route",
                     "start": {"type": "poi", "poi_id": candidate["poi_id"], "name": candidate["name"]}})
    assert dialogue.pending["start"]["poi_id"] == candidate["poi_id"]
    assert candidate["poi_id"] in dialogue.known


def test_prebound_resolved_online_start_survives_existing_task_binding(monkeypatch):
    from agents import tools
    candidate = place("amap:RESOLVED123", "示例门店")
    pending = {"request_id": "request", "original_query": "先前路线", "mode": "walk",
               "semantic_request": {"operation": "route"}}
    dialogue, _ = make_dialogue("改从示例门店出发", context={"available_tasks": [{
        "task_id": "task", "revision": 4, "state": {"place_request": pending}}]})
    dialogue.bind_task = lambda **_args: {"task_id": "task", "task_revision": 5}
    dialogue.execute_tool = lambda *_args: ({"found": True, "poi": candidate}, None)
    monkeypatch.setattr(tools.amap_poi, "lookup_by_id", lambda _ident: candidate)
    monkeypatch.setattr(tools.amap_poi, "choose_match", lambda _name, rows, **_kwargs: rows[0])
    monkeypatch.setattr(tools, "filter_reported_absent_provider_results", lambda rows: rows)
    dialogue.spatial("resolve_poi", {"name": candidate["name"]})
    dialogue.update({"relation": "continue", "task_id": "task", "task_revision": 4, "operation": "route",
                     "start": {"type": "poi", "poi_id": candidate["poi_id"], "name": candidate["name"]}})
    assert dialogue.pending["start"]["poi_id"] == candidate["poi_id"]
    assert candidate["poi_id"] in dialogue.known


def test_route_tool_cannot_silently_replace_accepted_start(monkeypatch):
    from agents import tools

    pending = {"request_id": "request", "original_query": "从珞珈门出发",
               "start": {"type": "poi", "name": "珞珈门"}, "mode": "walk",
               "semantic_request": {"operation": "route"}}
    dialogue, _ = make_dialogue("继续步行", context={"place_request": pending})
    execute = Mock(return_value=({"status": "ok"}, {}))
    dialogue.execute_tool = execute
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    answer = dialogue.spatial("plan_route", {"start": {"type": "poi", "name": "凌波门"},
                                             "end": {"type": "poi", "name": "万林艺术博物馆"}})

    assert answer["error"] == "task_update_required"
    execute.assert_not_called()


def test_location_identity_never_equates_conflicting_stable_ids():
    from agents import tools
    from agents.dialogue import _same_accepted_location
    first = {"type": "poi", "poi_id": "poi_140", "name": "同名地点"}
    second = {"type": "poi", "poi_id": "poi_137", "name": "同名地点"}
    assert not _same_accepted_location(first, second, tools)


def test_location_identity_checks_nested_coordinates_instead_of_missing_fields():
    from agents import tools
    from agents.dialogue import _same_accepted_location
    first = {"type": "coord", "coordinates": {"lng": 114.36, "lat": 30.53}}
    other = {"type": "coord", "coordinates": {"lng": 114.37, "lat": 30.54}}
    assert not _same_accepted_location(first, other, tools)


def test_nearby_search_cannot_use_an_invented_time_to_expand_the_radius(monkeypatch):
    from agents import tools
    pending = {"request_id": "request", "original_query": "附近打球", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"}, "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近球场", context={"place_request": pending})
    execute = Mock(return_value=({"status": "ok", "candidates": []}, {}))
    dialogue.execute_tool = execute
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    dialogue.spatial("find_reachable_places", {"max_time_min": 20})

    arguments = execute.call_args.args[1]
    assert arguments["max_distance_m"] == 1000
    assert "max_time_min" not in arguments


def test_reachable_search_preserves_user_time_constraint(monkeypatch):
    from agents import tools
    pending = {"request_id": "request", "original_query": "步行二十分钟内找球场", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"}, "search_radius_m": 1000,
               "constraints": {"max_duration_s": 1200}, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("步行二十分钟内找球场", context={"place_request": pending})
    execute = Mock(return_value=({"status": "ok", "candidates": []}, {}))
    dialogue.execute_tool = execute
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    dialogue.spatial("find_reachable_places", {"max_time_min": 20})

    arguments = execute.call_args.args[1]
    assert arguments["max_time_min"] == 20
    assert "max_distance_m" not in arguments


def test_reachable_search_keeps_explicit_radius_and_time_together(monkeypatch):
    from agents import tools
    candidate = {**place("amap:A123", "范围外候选"), "network_distance_m": 700, "distance_m": 700}
    pending = {"request_id": "request", "original_query": "找五百米内、二十分钟可达的球场",
               "start": {"type": "poi", "name": "珞珈门"}, "mode": "walk", "search_radius_m": 500,
               "constraints": {"max_duration_s": 1200}, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找五百米内、二十分钟可达的球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": [candidate], "count": 1},
                                               {"candidates": [candidate]}))
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    dialogue.spatial("find_reachable_places", {"max_time_min": 20})

    assert dialogue.execute_tool.call_args.args[1]["max_time_min"] == 20
    assert dialogue.execute_tool.call_args.args[2]["_search_scope_distance_m"] == 500


def test_explicit_default_radius_is_preserved_across_a_later_time_revision():
    pending = {"request_id": "request", "original_query": "附近找球场", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"}, "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("范围一公里", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "search_radius_m": 1000})
    assert dialogue.pending["semantic_request"]["search_radius_explicit"] is True


def test_reachable_search_uses_saved_time_instead_of_a_model_invented_budget(monkeypatch):
    from agents import tools
    pending = {"request_id": "request", "original_query": "步行二十分钟内找球场", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"}, "search_radius_m": 1000,
               "constraints": {"max_duration_s": 1200}, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0}, {}))
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue.spatial("find_reachable_places", {"max_time_min": 30})
    assert dialogue.execute_tool.call_args.args[1]["max_time_min"] == 20


def test_time_only_place_search_does_not_apply_an_invented_one_km_radius(monkeypatch):
    from agents import tools
    pending = {"request_id": "request", "original_query": "步行二十分钟内找球场", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"}, "search_radius_m": 1000,
               "constraints": {"max_duration_s": 1200}, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0}, {}))
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue.spatial("search_poi_candidates", {"activities": ["basketball"]})
    assert dialogue.execute_tool.call_args.args[1]["max_distance_m"] > 1000


def test_time_only_search_with_same_origin_and_anchor_has_no_implicit_radius(monkeypatch):
    from agents import tools
    origin = {"type": "poi", "name": "珞珈门"}
    pending = {"request_id": "request", "original_query": "从珞珈门步行20分钟内找球场",
               "mode": "walk", "start": origin, "search_anchor": deepcopy(origin), "search_radius_m": 1000,
               "constraints": {"max_duration_s": 1200}, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("从珞珈门步行20分钟内找球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0}, {}))
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue.spatial("search_poi_candidates", {"activities": ["basketball"]})
    assert dialogue.execute_tool.call_args.args[1]["max_distance_m"] > 1000


def test_time_limit_does_not_expand_an_explicit_search_center_scope(monkeypatch):
    from agents import tools
    pending = {"request_id": "request", "original_query": "从珞珈门出发找凌波门附近二十分钟可达的球场",
               "mode": "walk", "start": {"type": "poi", "name": "珞珈门"},
               "search_anchor": {"type": "poi", "name": "凌波门"}, "search_radius_m": 1000,
               "constraints": {"max_duration_s": 1200}, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0}, {}))
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue.spatial("search_poi_candidates", {"activities": ["basketball"]})
    assert dialogue.execute_tool.call_args.args[1]["max_distance_m"] == 1000


def test_reachable_search_does_not_replace_a_different_search_center(monkeypatch):
    from agents import tools
    pending = {"request_id": "request", "original_query": "珞珈门出发，找凌波门附近球场",
               "start": {"type": "poi", "name": "珞珈门"},
               "search_anchor": {"type": "poi", "name": "凌波门"}, "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近球场", context={"place_request": pending})
    execute = Mock(return_value=({"status": "ok", "candidates": []}, {}))
    dialogue.execute_tool = execute
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    answer = dialogue.spatial("find_reachable_places", {"max_time_min": 20})

    assert answer["error"] == "search_center_mismatch"
    execute.assert_not_called()


def test_reachable_search_cannot_replace_the_actual_origin_with_search_center(monkeypatch):
    from agents import tools
    pending = {"request_id": "request", "original_query": "珞珈门出发，找凌波门附近球场",
               "start": {"type": "poi", "name": "珞珈门"},
               "search_anchor": {"type": "poi", "name": "凌波门"}, "mode": "walk",
               "constraints": {"max_duration_s": 1200},
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": []}, {}))
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    result = dialogue.spatial("find_reachable_places", {
        "start": {"type": "poi", "name": "凌波门"}, "max_time_min": 20})
    assert result["error"] == "task_update_required"
    dialogue.execute_tool.assert_not_called()


def test_rejected_search_arguments_do_not_consume_the_search_limit():
    pending = {"request_id": "request", "original_query": "找附近球场", "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场", context={"place_request": pending})
    for _ in range(3):
        answer = dialogue.spatial("search_poi_candidates", {"max_distance_m": 2000})
        assert answer["error"] == "task_update_required"
    assert dialogue.place_search_count == 0


def test_resolving_the_same_search_center_does_not_discard_search_evidence():
    candidate = place("amap:A123", "目录地点")
    pending = {"request_id": "request", "original_query": "珞珈门附近找球场",
               "search_anchor": {"type": "poi", "name": "珞珈门"},
               "candidate_pool": [candidate], "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("看看其他的", context={"place_request": pending})
    dialogue.last_search = {"candidates": [], "count": 0, "online_status": "ok"}

    dialogue.update({"relation": "continue", "operation": "place_search",
                     "search_anchor": {"type": "poi", "poi_id": "poi_140", "name": "珞珈门"}})

    assert dialogue.pending["candidate_pool"] == [candidate]
    assert dialogue.last_search["count"] == 0


def test_new_search_with_unshown_results_requests_selectable_delivery(monkeypatch):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    candidate = place("amap:B123", "目录地点")
    pending = {"request_id": "request", "original_query": "找球场", "mode": "walk",
               "semantic_request": {"operation": "place_search", "shown_ids": []}}
    dialogue, model = make_dialogue("找球场", context={"place_request": pending}, replies=[
        reply(search_poi_candidates={"keyword": "球场"}),
        reply("附近有一个候选。"),
        reply(present_places={"introduction": "找到一处。", "places": [
            {"poi_id": candidate["poi_id"], "reason": ""}]}),
    ])
    dialogue.execute_tool = Mock(return_value=({"status": "ok", "candidates": [candidate], "count": 1},
                                                {"candidates": [candidate]}))

    result = dialogue.run()

    assert model.call_count == 3
    assert [row["poi_id"] for row in result["candidates"]] == [candidate["poi_id"]]


def test_repeated_successful_empty_searches_finish_with_an_honest_empty_result(monkeypatch):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找球场", "mode": "walk",
               "semantic_request": {"operation": "place_search", "shown_ids": []}}
    dialogue, model = make_dialogue("找球场", context={"place_request": pending}, replies=[
        reply(search_poi_candidates={"keyword": "球场"}) for _ in range(9)])
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "ok"},
                                                {"candidates": []}))

    result = dialogue.run()

    assert result["response_kind"] == "chat"
    assert result.get("failure_code") is None
    assert "未找到" in (result.get("assistant_text") or result.get("message") or "")
    assert model.call_count <= 3
    assert dialogue.execute_tool.call_count == 1


def test_empty_search_deduplication_keeps_distinct_queries_and_failures(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找球场", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(side_effect=[
        ({"candidates": [], "count": 0, "online_status": "ok"}, {}),
        ({"candidates": [], "count": 0, "online_status": "unavailable"}, {})])
    assert not dialogue.spatial("search_poi_candidates", {"keyword": "篮球", "limit": 4}).get("error")
    assert dialogue.spatial("search_poi_candidates", {"keyword": "篮球", "limit": 8})["error"] == "duplicate_empty_search"
    assert dialogue.spatial("search_poi_candidates", {"keyword": "羽毛球"})["online_status"] == "unavailable"
    assert dialogue.execute_tool.call_count == 2


def test_unverified_resolved_place_cannot_become_a_search_task_route(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找附近球场", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"},
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近球场", context={"place_request": pending})
    dialogue.known["poi_206"] = place("poi_206", "工学部篮球场")
    dialogue.execute_tool = Mock(return_value=({"status": "ok"}, {}))
    answer = dialogue.spatial("plan_route", {"end": {"type": "poi", "poi_id": "poi_206", "name": "工学部篮球场"}})
    assert answer["error"] == "candidate_scope_unverified"
    assert not dialogue.pending.get("selected_poi_id")
    dialogue.execute_tool.assert_not_called()


def test_model_and_spatial_work_leave_time_to_commit_a_timeout(monkeypatch):
    from agents import deadline, dialogue_loop
    monkeypatch.setattr(dialogue_loop.time, "monotonic", lambda: 100.0)
    dialogue, _ = make_dialogue("继续")
    dialogue.deadline_monotonic = 110.0
    monkeypatch.setattr(dialogue_loop, "_run_dialogue_loop",
                        lambda *_args: deadline.limited_timeout(100))
    assert dialogue_loop.run_dialogue_loop(dialogue) == 5.0


def test_part_of_origin_name_is_not_a_separately_requested_search_center():
    pending = {"request_id": "request", "mode": "walk",
               "start": {"type": "poi", "name": "信息学部快递服务中心"},
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("我在信息学部快递服务中心，找附近的球场。", context={"place_request": pending})
    answer = dialogue.update({"relation": "continue", "operation": "place_search",
                              "search_anchor": {"type": "poi", "name": "信息学部"}})
    assert not dialogue.pending.get("search_anchor")
    assert answer.get("search_anchor_note")

    dialogue, _ = make_dialogue("我在信息学部快递服务中心，找信息学部附近的球场。", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search",
                     "search_anchor": {"type": "poi", "name": "信息学部"}})
    assert dialogue.pending["search_anchor"]["name"] == "信息学部"


def test_search_tool_cannot_invent_an_unaccepted_distinct_center():
    pending = {"request_id": "request", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"},
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近球场", context={"place_request": pending})
    answer = dialogue.spatial("search_poi_candidates", {"search_anchor": {"type": "poi", "name": "凌波门"}})
    assert answer["error"] == "task_update_required"
    dialogue.execute_tool.assert_not_called()


def test_changing_search_center_removes_only_the_duplicated_old_center_requirement():
    pending = {"request_id": "request", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"},
               "search_anchor": {"type": "poi", "name": "万林艺术博物馆"},
               "semantic_request": {"operation": "place_search",
                                    "requirements": ["万林艺术博物馆附近", "咖啡店", "避开台阶"]}}
    dialogue, _ = make_dialogue("改成在星湖体育馆附近找。", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search",
                     "search_anchor": {"type": "poi", "name": "星湖体育馆"}})
    assert dialogue.pending["semantic_request"]["requirements"] == ["咖啡店", "避开台阶"]
    assert dialogue.pending["start"]["name"] == "珞珈门"

    both, _ = make_dialogue("两处附近都要考虑。", context={"place_request": pending})
    both.update({"relation": "continue", "operation": "place_search",
                 "search_anchor": {"type": "poi", "name": "星湖体育馆"},
                 "requirements": ["万林艺术博物馆附近", "星湖体育馆附近"]})
    assert "万林艺术博物馆附近" in both.pending["semantic_request"]["requirements"]


@pytest.mark.parametrize(("query", "start", "end", "slot"), [
    ("我想去体育馆，从珞珈门步行出发。", "珞珈门", "武汉大学宋卿体育馆", "destination"),
    ("从体育馆步行到珞珈门。", "武汉大学宋卿体育馆", "珞珈门", "start"),
    ("从珞珈门去食堂。", "珞珈门", "枫园风味食堂", "destination"),
])
def test_unresolved_ambiguous_route_place_requires_a_choice(monkeypatch, query, start, end, slot):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "mode": "walk", "start": {"type": "poi", "name": start},
               "semantic_request": {"operation": "route"}}
    dialogue, _ = make_dialogue(query, context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"status": "ok"}, {}))
    answer = dialogue.spatial("plan_route", {"start": {"type": "poi", "name": start},
                                             "end": {"type": "poi", "name": end}})
    assert answer.get("needs_input") is True
    assert dialogue.clarify["slot"] == slot
    assert len(dialogue.clarify["options"]) >= 2
    dialogue.execute_tool.assert_not_called()


def test_explicitly_named_route_destination_does_not_trigger_ambiguity_guard(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "mode": "walk", "start": {"type": "poi", "name": "珞珈门"},
               "semantic_request": {"operation": "route"}}
    dialogue, _ = make_dialogue("从珞珈门去宋卿体育馆。", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"status": "ok"}, {}))
    answer = dialogue.spatial("plan_route", {"end": {"type": "poi", "name": "武汉大学宋卿体育馆"}})
    assert not answer.get("needs_input")
    dialogue.execute_tool.assert_called_once()


def test_many_distinct_empty_searches_offer_a_bounded_honest_answer(monkeypatch):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找球场", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    queries = ["球场", "运动场", "打球", "运动", "附近球场", "球类场地"]
    dialogue, model = make_dialogue("找球场", context={"place_request": pending}, replies=[
        reply(search_poi_candidates={"keyword": keyword}) for keyword in queries])
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "ok"}, {}))
    result = dialogue.run()
    assert result.get("failure_code") is None
    assert "未找到" in (result.get("assistant_text") or result.get("message") or "")
    assert model.call_count <= 6
    assert dialogue.execute_tool.call_count == 4


def test_changed_scope_at_search_limit_gets_one_bounded_recheck(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找附近球场", "mode": "walk",
               "activities": ["basketball"], "semantic_request": {"operation": "place_search", "area": "信息学部"}}
    dialogue, _ = make_dialogue("再查查", context={"place_request": pending})
    dialogue.place_search_count = 8
    dialogue.last_search = {"candidates": [], "count": 0, "online_status": "not_requested"}
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "not_requested"}, {}))
    dialogue.update({"relation": "continue", "operation": "place_search", "area": "武汉大学"})
    assert not dialogue.spatial("search_poi_candidates", {"area": "武汉大学"}).get("error")
    assert dialogue.execute_tool.call_count == 1
    dialogue.update({"relation": "continue", "operation": "place_search", "area": "工学部"})
    assert dialogue.spatial("search_poi_candidates", {"area": "工学部"})["error"] == "place_search_limit"
    assert dialogue.execute_tool.call_count == 1


def test_unchanged_scope_cannot_create_extra_search_credit(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找球场", "mode": "walk",
               "semantic_request": {"operation": "place_search", "area": "武汉大学"}}
    dialogue, _ = make_dialogue("继续", context={"place_request": pending})
    dialogue.place_search_count = 8
    dialogue.update({"relation": "continue", "operation": "place_search", "area": "武汉大学"})
    assert dialogue.spatial("search_poi_candidates", {"area": "武汉大学"})["error"] == "place_search_limit"


def test_ambiguous_ball_search_requests_sport_before_accepting_inferred_candidates(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue, _ = make_dialogue("从星湖园食堂出发，我想找地方打球，步行过去。")
    dialogue.update({"relation": "new", "operation": "place_search", "mode": "walk",
                     "start": {"type": "poi", "name": "星湖园食堂"},
                     "activities": ["basketball", "badminton"]})
    dialogue.execute_tool = Mock(return_value=({"candidates": []}, {}))
    answer = dialogue.spatial("find_reachable_places", {"activities": ["basketball", "badminton"]})
    assert answer.get("needs_input") is True
    assert dialogue.clarify["slot"] == "sport"
    assert dialogue.pending["start"]["name"] == "星湖园食堂"
    assert not dialogue.pending.get("activities")
    dialogue.execute_tool.assert_not_called()


def test_model_sport_question_does_not_save_unconfirmed_options_as_chosen_activities(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    query = "从星湖园食堂出发，我想找地方打球，步行过去。"
    dialogue, _ = make_dialogue(query, replies=[
        reply(update_task={"relation": "new", "operation": "place_search", "mode": "walk",
                           "start": {"type": "poi", "name": "星湖园食堂"},
                           "activities": ["basketball", "badminton"]}),
        reply(ask_user={"slot": "sport", "question": "想打哪种球？", "options": ["篮球", "羽毛球"]}),
    ])
    dialogue.execute_tool = lambda name, args, context: (
        {"question": args["question"], "options": args["options"], "slot": args["slot"]},
        {"clarify": {"question": args["question"], "options": args["options"], "slot": args["slot"]}})
    result = dialogue.run()
    assert result["response_kind"] == "clarify"
    assert result["place_request"]["activities"] == []
    assert result["place_request"]["pending_slot"] == "activity"


@pytest.mark.parametrize("as_tool", [False, True])
def test_sport_question_clears_unconfirmed_activities_even_without_a_slot(monkeypatch, as_tool):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    query = "从星湖园食堂出发，想找地方打球。"
    answer = (reply(ask_user={"question": "想打哪种球？", "options": ["篮球", "羽毛球"]})
              if as_tool else reply("想打哪种球？"))
    dialogue, _ = make_dialogue(query, replies=[
        reply(update_task={"relation": "new", "operation": "place_search",
                           "start": {"type": "poi", "name": "星湖园食堂"},
                           "activities": ["basketball", "badminton"]}), answer])
    dialogue.execute_tool = lambda name, args, context: (
        {"question": args["question"]}, {"clarify": {"question": args["question"]}})
    result = dialogue.run()
    assert result["response_kind"] == "clarify"
    assert result["place_request"]["activities"] == []
    assert result["place_request"]["pending_slot"] == "activity"


def test_ambiguous_wording_preserves_a_previously_selected_sport(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找篮球场", "mode": "walk",
               "activities": ["basketball"], "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("再找个打球的地方", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0}, {}))
    answer = dialogue.spatial("find_reachable_places", {})
    assert not answer.get("needs_input")
    dialogue.execute_tool.assert_called_once()


def test_user_can_explicitly_accept_all_ball_sports(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue, _ = make_dialogue("附近找地方打球，哪种球都可以。")
    dialogue.update({"relation": "new", "operation": "place_search", "activities": ["basketball", "badminton"]})
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0}, {}))
    answer = dialogue.spatial("find_reachable_places", {})
    assert not answer.get("needs_input")
    dialogue.execute_tool.assert_called_once()


def test_sport_clarification_is_cleared_when_a_specific_activity_is_supplied():
    pending = {"request_id": "request", "original_query": "找地方打球", "mode": "walk",
               "pending_slot": "activity", "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("打羽毛球", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "activities": ["badminton"]})
    assert dialogue.pending["pending_slot"] is None
    assert dialogue.pending["activities"] == ["badminton"]


def test_repeated_unverified_route_after_empty_search_returns_the_scoped_empty_result(monkeypatch):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找附近篮球场", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"},
               "activities": ["basketball"], "semantic_request": {"operation": "place_search"}}
    invalid_route = reply(plan_route={"end": {"type": "poi", "name": "工学部篮球场"}})
    dialogue, _ = make_dialogue("找附近篮球场", context={"place_request": pending}, replies=[
        reply(search_poi_candidates={"keyword": "篮球场"}), invalid_route, invalid_route, invalid_route])
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "ok"}, {}))
    result = dialogue.run()
    assert result.get("failure_code") is None
    assert "未找到" in (result.get("assistant_text") or result.get("message") or "")
    assert not result.get("route_state")
    assert dialogue.execute_tool.call_count == 1


def test_verified_empty_search_reply_explicitly_says_no_candidates_and_keeps_other_information():
    from agents.dialogue_loop import _bounded_empty_reply
    pending = {"request_id": "request", "original_query": "找篮球场", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找篮球场", context={"place_request": pending})
    dialogue.last_search = {"candidates": [], "count": 0, "online_status": "ok"}
    text = _bounded_empty_reply(dialogue, "以上是目前的检索结果，也可以换个方向。天气晴。")
    assert "未找到" in text
    assert "以上是目前的检索结果" not in text
    assert "换个方向" in text and "天气晴" in text


def test_empty_search_does_not_invent_a_reason_for_the_missing_candidates():
    from agents.dialogue_loop import _bounded_empty_reply
    pending = {"request_id": "request", "original_query": "找篮球场", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找篮球场", context={"place_request": pending})
    dialogue.last_search = {"candidates": [], "count": 0, "online_status": "ok"}
    text = _bounded_empty_reply(dialogue,
        "当前没有检索到篮球场（连食堂也返回为空，说明附近路网数据不足）。可以扩大范围。天气晴。")
    assert "没有检索到篮球场" in text
    assert "食堂" not in text and "路网数据不足" not in text
    assert "扩大范围" in text and "天气晴" in text


def test_empty_search_guard_still_allows_another_requested_read(monkeypatch):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找球场并看看天气", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场并看看天气", context={"place_request": pending}, replies=[
        *(reply(search_poi_candidates={"keyword": str(index)}) for index in range(4)),
        reply(get_weather={}), reply(conclude_without_places={"reason": "当前查询未找到球场。"})])
    def execute(name, *_args):
        if name == "get_weather":
            return {"weather": "晴", "temperature": 23}, {}
        return {"candidates": [], "count": 0, "online_status": "ok"}, {}
    dialogue.execute_tool = Mock(side_effect=execute)
    result = dialogue.run()
    assert result["weather"]["weather"] == "晴"
    assert "未找到" in result["message"]


def test_unseen_candidate_pages_do_not_count_as_a_verified_empty_search(monkeypatch):
    from agents import tools
    from agents.dialogue_loop import _has_verified_empty_search
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找球场", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场", context={"place_request": pending})
    dialogue.last_search = {"candidates": [], "count": 0, "online_status": "ok", "has_more": True}
    dialogue.place_search_count = 4
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "ok"}, {}))
    assert not _has_verified_empty_search(dialogue)
    assert not dialogue.spatial("search_poi_candidates", {"keyword": "篮球"}).get("error")
    dialogue.execute_tool.assert_called_once()


def test_explicit_search_radius_is_not_saved_as_a_route_distance_limit():
    pending = {"request_id": "request", "mode": "walk", "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找500米范围内、步行20分钟内能到的篮球场。", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search",
                     "constraints": {"max_distance_m": 500, "max_duration_s": 1200}})
    assert dialogue.pending["search_radius_m"] == 500
    assert dialogue.pending["constraints"]["max_distance_m"] is None
    assert dialogue.pending["constraints"]["max_duration_s"] == 1200

    later, _ = make_dialogue("搜索范围扩大到一公里。", context={"place_request": dialogue.pending})
    later.update({"relation": "continue", "operation": "place_search",
                  "constraints": {"max_distance_m": 1000}})
    assert later.pending["search_radius_m"] == 1000
    assert later.pending["constraints"]["max_distance_m"] is None
    assert later.pending["constraints"]["max_duration_s"] == 1200


def test_radius_and_independent_route_distance_constraints_are_both_preserved():
    pending = {"request_id": "request", "mode": "walk", "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("搜索半径500米，路线距离不超过500米。", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "constraints": {"max_distance_m": 500}})
    assert dialogue.pending["search_radius_m"] == 500
    assert dialogue.pending["constraints"]["max_distance_m"] == 500

    route_only, _ = make_dialogue("路线距离不超过500米。", context={"place_request": pending})
    route_only.update({"relation": "continue", "operation": "place_search", "constraints": {"max_distance_m": 500}})
    assert route_only.pending["search_radius_m"] == 1000
    assert route_only.pending["constraints"]["max_distance_m"] == 500

    route_range, _ = make_dialogue("路线长度限定在500米范围内。", context={"place_request": pending})
    route_range.update({"relation": "continue", "operation": "place_search", "constraints": {"max_distance_m": 500}})
    assert route_range.pending["search_radius_m"] == 1000
    assert route_range.pending["constraints"]["max_distance_m"] == 500

    both, _ = make_dialogue("搜索半径500米、路线距离不超过500米。", context={"place_request": pending})
    both.update({"relation": "continue", "operation": "place_search", "constraints": {"max_distance_m": 500}})
    assert both.pending["search_radius_m"] == 500
    assert both.pending["constraints"]["max_distance_m"] == 500


def test_changing_radius_does_not_relax_a_previously_explicit_route_limit():
    pending = {"request_id": "request", "mode": "walk", "search_radius_m": 500,
               "constraints": {"max_distance_m": 400}, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("范围扩大到一公里。", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "constraints": {"max_distance_m": 1000}})
    assert dialogue.pending["search_radius_m"] == 1000
    assert dialogue.pending["constraints"]["max_distance_m"] == 400


@pytest.mark.parametrize("query", ["不必限制在500米范围内。", "取消500米范围限制。"])
def test_negated_radius_is_not_added_as_a_new_search_condition(query):
    pending = {"request_id": "request", "mode": "walk", "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue(query, context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "constraints": {}})
    assert dialogue.pending["search_radius_m"] == 1000


def test_coarse_place_type_is_not_mistaken_for_a_subcategory(monkeypatch):
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "mode": "walk", "activities": ["basketball"],
               "start": {"type": "poi", "name": "珞珈门"},
               "subcategories": ["sports"], "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近篮球场", context={"place_request": pending})
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0}, {}))
    dialogue.spatial("search_poi_candidates", {"keyword": "篮球"})
    args = dialogue.execute_tool.call_args.args[1]
    assert args["poi_type"] == "sports"
    assert "sports" not in args.get("subcategories", [])
    assert args["activities"] == ["basketball"]


@pytest.mark.parametrize("terminal", [
    reply("地图已查询，500米范围内没有球场。"),
    reply(conclude_without_places={"reason": "地图已查询，500米范围内没有球场。"}),
])
def test_empty_search_answer_does_not_invent_provider_use_or_old_radius(monkeypatch, terminal):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "mode": "walk", "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("范围扩大到一公里。", context={"place_request": pending}, replies=[
        reply(search_poi_candidates={"keyword": "篮球"}), terminal])
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "not_requested"}, {}))
    result = dialogue.run()
    assert "未找到" in result["message"]
    assert "地图" not in result["message"] and "500米" not in result["message"]


@pytest.mark.parametrize(("query", "expected"), [
    ("找500米范围内、步行20分钟内能到的篮球场。", 1200),
    ("从这里步行不超过二十分钟找球场。", 1200),
    ("最多走15分钟。", 900),
    ("用时放宽到30分钟。", 1800),
    ("半小时以内能到的地点。", 1800),
])
def test_explicit_travel_time_limit_survives_a_model_omission(query, expected):
    pending = {"request_id": "request", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue(query, context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "constraints": {}})
    assert dialogue.pending["constraints"]["max_duration_s"] == expected


@pytest.mark.parametrize("query", ["到球场后停留20分钟。", "20分钟后出发。", "预计步行20分钟。",
                                   "不用限制在20分钟内。", "取消20分钟内到达的要求。",
                                   "路线距离不超过一公里、预计步行20分钟。"])
def test_dwell_departure_and_estimated_times_are_not_invented_as_travel_limits(query):
    pending = {"request_id": "request", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue(query, context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "constraints": {}})
    assert dialogue.pending["constraints"]["max_duration_s"] is None


def test_search_service_error_is_never_rewritten_as_an_empty_result(monkeypatch):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找球场", "mode": "walk",
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找球场", context={"place_request": pending}, replies=[
        reply(search_poi_candidates={"keyword": "球场"}), reply("没有。")])
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "unavailable"},
                                                {"candidates": []}))
    result = dialogue.run()
    assert result["failure_code"] == "place_service_unavailable"


def test_repeated_rejected_searches_preserve_the_last_valid_empty_result(monkeypatch):
    from agents import tools
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    pending = {"request_id": "request", "original_query": "找附近球场", "mode": "walk",
               "search_radius_m": 1000, "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近球场", context={"place_request": pending}, replies=[
        reply(search_poi_candidates={"keyword": "球场"}),
        *[reply(search_poi_candidates={"keyword": "球场", "max_distance_m": 2000}) for _ in range(3)],
    ])
    dialogue.execute_tool = Mock(return_value=({"candidates": [], "count": 0, "online_status": "ok"},
                                                {"candidates": []}))
    result = dialogue.run()
    assert result.get("failure_code") is None
    assert "未找到" in (result.get("assistant_text") or result.get("message") or "")
    assert dialogue.pending["search_radius_m"] == 1000


def test_present_nearby_place_requires_a_result_from_the_current_scoped_search():
    import pytest
    valid, resolved_only = place("amap:A123", "已核验地点"), place("amap:B123", "仅解析地点")
    pending = {"request_id": "request", "original_query": "附近找球场", "mode": "walk",
               "search_anchor": {"type": "poi", "name": "珞珈门"},
               "candidate_pool": [valid], "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("附近找球场", context={"place_request": pending})
    dialogue.known[resolved_only["poi_id"]] = resolved_only
    with pytest.raises(ValueError, match="candidate_scope_unverified"):
        dialogue.present({"places": [{"poi_id": resolved_only["poi_id"], "reason": ""}]})
    dialogue.present({"places": [{"poi_id": valid["poi_id"], "reason": ""}]})
    assert [row["poi_id"] for row in dialogue.candidates] == [valid["poi_id"]]


def test_scoped_candidate_pool_keeps_results_from_multiple_valid_searches(monkeypatch):
    from agents import tools
    first, second = place("amap:A123", "第一类候选"), place("amap:B123", "第二类候选")
    pending = {"request_id": "request", "original_query": "附近找吃饭地点", "mode": "walk",
               "start": {"type": "poi", "name": "珞珈门"},
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("附近找吃饭地点", context={"place_request": pending})
    dialogue.execute_tool = Mock(side_effect=[({"candidates": [first], "count": 1}, {"candidates": [first]}),
                                               ({"candidates": [second], "count": 1}, {"candidates": [second]})])
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue.spatial("search_poi_candidates", {"keyword": "餐厅"})
    dialogue.spatial("search_poi_candidates", {"keyword": "快餐"})
    assert {row["poi_id"] for row in dialogue.pending["candidate_pool"]} == {first["poi_id"], second["poi_id"]}


def test_unselected_recommendations_are_not_hidden_by_a_provisional_route():
    candidate = place("amap:A123", "目录候选")
    pending = {"request_id": "request", "original_query": "找附近球场",
               "start": {"type": "poi", "name": "珞珈门"},
               "candidate_pool": [candidate], "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找附近球场", context={"place_request": pending})
    dialogue.route = {"route_verification": {"status": "pass"}}
    dialogue.present({"places": [{"poi_id": candidate["poi_id"], "reason": ""}]})

    result = dialogue.result("可以看看这里。")

    assert result["response_kind"] == "candidates"
    assert result["route"] is None
    assert result["place_request"]["pending_slot"] == "selection"


def test_all_original_messages_reach_first_model_call_without_truncation(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    history = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": f"第{i}条" + "文" * 5100} for i in range(18)]
    dialogue, create = make_dialogue("回到最初的要求", history=history,
                                     replies=[reply("收到")])

    dialogue.run()

    sent = create.call_args_list[0].kwargs["messages"]
    assert [item for item in sent if item["role"] in {"user", "assistant"}][:-1] == history
    assert sent[-1] == {"role": "user", "content": "回到最初的要求"}


def test_candidate_paging_always_consults_model_with_unshown_candidate_ids(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    first = place("amap:A123", "已展示餐厅")
    second = place("amap:B123", "未展示餐厅")
    pending = {"request_id": "request", "original_query": "信图出发找总图附近吃饭",
               "pending_slot": "selection", "start": {"type": "poi", "name": "信图"},
               "search_anchor": {"type": "poi", "name": "总图"},
               "candidate_poi_ids": [first["poi_id"]], "candidate_places": [first],
               "candidate_pool": [first, second], "mode": "walk", "constraints": {},
               "semantic_request": {"operation": "place_search", "shown_ids": [first["poi_id"]]}}
    dialogue, create = make_dialogue(
        "换一批", context={"place_request": deepcopy(pending)},
        replies=[reply(update_task={"relation": "continue", "operation": "place_search",
                                    "candidate_action": "next_batch"}),
                 reply(present_places={"introduction": "还有一个候选。", "places": [
                     {"poi_id": second["poi_id"], "reason": "地图目录中的候选。"}]})],
    )

    result = dialogue.run()

    assert create.call_count >= 1
    first_call = create.call_args_list[0].kwargs
    assert {"update_task", "search_poi_candidates", "get_weather", "plan_route"} <= {
        schema["function"]["name"] for schema in first_call["tools"]}
    assert "amap:B123" in str(first_call["messages"]) or (
        create.call_count > 1 and "amap:B123" in str(create.call_args_list[1].kwargs["messages"]))
    assert [row["poi_id"] for row in result["candidates"]] == ["amap:B123"]
    assert result["place_request"]["start"]["name"] == "信图"
    assert result["place_request"]["search_anchor"]["name"] == "总图"


def test_model_next_batch_cannot_finish_with_stale_text_while_unseen_candidates_exist(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    first, second = place("amap:A123", "已展示餐厅"), place("amap:B123", "未展示餐厅")
    pending = {"request_id": "request", "original_query": "找餐厅", "mode": "walk",
               "candidate_pool": [first, second], "candidate_places": [first],
               "candidate_poi_ids": [first["poi_id"]],
               "semantic_request": {"operation": "place_search", "shown_ids": [first["poi_id"]]}}
    dialogue, model = make_dialogue("再看看其他家", context={"place_request": pending}, replies=[
        reply(update_task={"relation": "continue", "operation": "place_search", "candidate_action": "next_batch"}),
        reply(update_task={"relation": "continue", "operation": "place_search"}),
        reply("上面就是所有候选。"),
        reply(present_places={"introduction": "还有一家。", "places": [
            {"poi_id": second["poi_id"], "reason": "目录候选"}]}),
    ])
    result = dialogue.run()
    assert model.call_count == 4
    assert [row["poi_id"] for row in result["candidates"]] == [second["poi_id"]]
    assert "上面就是所有候选" not in result["message"]


def test_model_can_explain_that_unshown_candidates_do_not_satisfy_requirements(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    candidate = place("amap:B123", "未展示餐厅")
    pending = {"request_id": "request", "original_query": "找安静的餐厅", "candidate_pool": [candidate],
               "semantic_request": {"operation": "place_search", "shown_ids": []}}
    dialogue, _ = make_dialogue("再看看", context={"place_request": pending}, replies=[
        reply(update_task={"relation": "continue", "operation": "place_search", "candidate_action": "next_batch"}),
        reply(conclude_without_places={"reason": "现有候选缺少安静程度的依据，暂时无法推荐。"}),
    ])
    result = dialogue.run()
    assert result["response_kind"] == "chat"
    assert "缺少安静程度的依据" in result["message"]
    assert "候选池" not in result["message"]
    assert "条未展示记录" not in result["message"]


def test_exhausted_candidates_are_reported_and_cannot_be_represented():
    import pytest
    selected = place("amap:A123", "已展示餐厅")
    pending = {"request_id": "request", "original_query": "找吃饭地点",
               "candidate_pool": [selected], "candidate_places": [selected],
               "semantic_request": {"operation": "place_search", "shown_ids": [selected["poi_id"]]}}
    dialogue, _ = make_dialogue("还有别家吗", context={"place_request": pending})
    page = dialogue.browse_candidates({})
    assert page["candidates"] == [] and page["exhausted"] is True
    with pytest.raises(ValueError, match="already_shown"):
        dialogue.present({"places": [{"poi_id": selected["poi_id"], "reason": "目录候选"}]})


def test_next_batch_excludes_same_name_and_location_with_a_different_source_id():
    shown = place("poi:LOCAL", "同一场所")
    duplicate = place("amap:DUP", "同一场所")
    distinct = {**place("amap:FAR", "同一场所"), "coordinates": {"lng": 114.37, "lat": 30.54}}
    pending = {"request_id": "request", "original_query": "找附近场所", "mode": "walk",
               "candidate_pool": [shown, duplicate, distinct],
               "semantic_request": {"operation": "place_search", "shown_ids": [shown["poi_id"]]}}
    dialogue, _ = make_dialogue("换一批", context={"place_request": pending})
    page = dialogue.update({"relation": "continue", "operation": "place_search",
                            "candidate_action": "next_batch"})["candidate_page"]
    assert [row["poi_id"] for row in page["candidates"]] == [distinct["poi_id"]]
    assert [row["poi_id"] for row in dialogue.browse_candidates({})["candidates"]] == [distinct["poi_id"]]
    with pytest.raises(ValueError, match="already_shown"):
        dialogue.present({"places": [{"poi_id": duplicate["poi_id"], "reason": ""}]})


def test_next_batch_page_reports_its_actual_exhaustion():
    candidate = place("amap:B123", "未展示餐厅")
    pending = {"request_id": "request", "original_query": "找餐厅", "candidate_pool": [candidate],
               "semantic_request": {"operation": "place_search", "shown_ids": []}}
    dialogue, _ = make_dialogue("换几家", context={"place_request": pending})
    page = dialogue.update({"relation": "continue", "operation": "place_search",
                            "candidate_action": "next_batch"})["candidate_page"]
    assert page["count"] == page["next_offset"] == 1
    assert page["offset"] == 0 and page["exhausted"] is True


def test_presenting_candidates_preserves_all_prior_shown_ids():
    candidate = place("amap:NEW123", "新候选")
    old_ids = [f"amap:OLD{i}" for i in range(105)]
    pending = {"request_id": "request", "original_query": "找吃饭地点", "candidate_pool": [candidate],
               "semantic_request": {"operation": "place_search", "shown_ids": old_ids}}
    dialogue, _ = make_dialogue("还有别家吗", context={"place_request": pending})
    dialogue.present({"places": [{"poi_id": candidate["poi_id"], "reason": "目录候选"}]})
    assert dialogue.intent["shown_ids"] == [*old_ids, candidate["poi_id"]]


def test_present_places_removes_card_reasons_without_direct_evidence():
    hot_pot = place("amap:HOT123", "呷哺呷哺(武汉武大梅园店)")
    supported = place("amap:MEAL123", "周麻婆")
    supported.update({"description": "川菜小炒", "address": "梅园二路铺位"})
    address_distance = place("amap:DIST123", "巷口面馆")
    address_distance["address"] = "汉林东路与珞南一路交叉口东南60米"
    dialogue, _ = make_dialogue("想吃热食", context={"place_request": {
        "candidate_pool": [hot_pot, supported, address_distance],
    }})

    dialogue.present({"places": [
        {"poi_id": hot_pot["poi_id"], "reason": "火锅，可现煮热食"},
        {"poi_id": supported["poi_id"], "reason": "川菜小炒，梅园二路铺位"},
        {"poi_id": address_distance["poi_id"],
         "reason": "位于汉林东路与珞南一路交叉口东南60米"},
    ]})

    assert dialogue.candidates[0]["recommendation_reason"] == ""
    assert dialogue.candidates[1]["recommendation_reason"] == "川菜小炒，梅园二路铺位"
    assert dialogue.candidates[2]["recommendation_reason"] == ""


def test_present_places_can_ground_card_reasons_in_attached_web_source():
    candidate = place("amap:COFFEE123", "校园咖啡店")
    dialogue, _ = make_dialogue("想找咖啡", context={"place_request": {
        "candidate_pool": [candidate],
    }})
    dialogue.sources["Scoffee123"] = {"source_id": "Scoffee123", "excerpt": "现磨咖啡"}

    dialogue.present({"places": [{"poi_id": candidate["poi_id"],
                                   "reason": "现磨咖啡", "source_ids": ["Scoffee123"]}]})

    assert dialogue.candidates[0]["recommendation_reason"] == "现磨咖啡"
    assert dialogue.candidates[0]["source_ids"] == ["Scoffee123"]


def test_present_places_preserves_requested_detail_and_rejects_excessive_text():
    import pytest

    candidate = place("amap:SHORT123", "简短餐厅")
    dialogue, _ = make_dialogue("请详细介绍附近餐厅", context={"place_request": {
        "candidate_pool": [candidate],
    }})

    detailed_intro = "详细介绍" * 35
    assert dialogue.present({"introduction": detailed_intro, "places": [
        {"poi_id": candidate["poi_id"], "reason": "坐下吃饭"},
    ]}) == detailed_intro

    dialogue, _ = make_dialogue("请详细介绍附近餐厅", context={"place_request": {
        "candidate_pool": [candidate],
    }})
    with pytest.raises(ValueError, match="recommendation_intro_too_long"):
        dialogue.present({"introduction": "详细介绍" * 126, "places": [
            {"poi_id": candidate["poi_id"], "reason": "坐下吃饭"},
        ]})

    dialogue, _ = make_dialogue("请详细介绍附近餐厅", context={"place_request": {
        "candidate_pool": [candidate],
    }})
    with pytest.raises(ValueError, match="recommendation_reason_too_long"):
        dialogue.present({"introduction": "附近可以看看这家。", "places": [
            {"poi_id": candidate["poi_id"], "reason": "详细介绍" * 10},
        ]})


def test_worker_sends_weather_turn_and_full_history_to_model(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from langgraph.checkpoint.memory import InMemorySaver
    from agents import workflow

    seen = []
    monkeypatch.setattr(workflow, "append_run_event", lambda *_a, **_k: {"stored": True})
    monkeypatch.setattr(workflow, "get_run_input", lambda *_a: {
        "model_led": True, "query": "现在会下雨吗", "owner_id": "owner", "conversation_id": "conversation",
        "history": [{"role": "user", "content": str(i), "metadata": {}} for i in range(16)],
        "tasks": [], "request": {}, "deadline_at": datetime.now(timezone.utc) + timedelta(seconds=75),
    })
    monkeypatch.setattr(workflow, "_run_weather_tool", lambda: (_ for _ in ()).throw(AssertionError("rule bypass")))
    def agent(query, **kwargs):
        seen.append((query, kwargs))
        return {"response_kind": "chat", "message": "模型查询天气", "turns": 1}
    graph = workflow.build_agent_workflow(InMemorySaver(), agent=agent)
    result = graph.invoke({"run_id": "model-weather"}, {"configurable": {"thread_id": "model-weather"}},
                          context={"worker_id": "worker"})["result"]
    assert result["reply"] == "模型查询天气"
    assert len(seen[0][1]["history"]) == 16
    assert callable(seen[0][1]["bind_task_callback"])
    assert result["assistant_text"] == result["reply"]


def test_model_route_delivery_text_omits_generic_and_named_route_event_appendices():
    from agents.workflow import _model_delivery_text
    result = {"response_kind": "route", "explanation": "路线已规划。",
              "route_verification": {"status": "pass", "field_truth_verified": False},
              "route_events": [{"road_name": "科技路", "name": "人流较多", "action": "slowdown"}]}
    text = _model_delivery_text(result)
    assert text.startswith("路线已规划。")
    assert "路网和路线约束" not in text
    assert "未实地确认道路当前可通行" not in text
    assert "科技路" not in text and "人流较多" not in text


def test_model_route_delivery_keeps_only_the_specific_step_access_caveat():
    from agents.workflow import _model_delivery_text
    result = {"response_kind": "route", "explanation": "已按你的要求规划好。",
              "route_state": {"hard_constraints": {"avoid_steps": True}},
              "filter_status": "no_outside+steps_avoid+steps_unverified"}

    text = _model_delivery_text(result)

    assert "台阶" in text and "尚未核实" in text
    assert text.count("台阶") == 2
    assert "路网和路线约束" not in text


def test_model_route_delivery_does_not_append_generic_unknown_events_notice():
    from agents.workflow import _model_delivery_text
    result = {"response_kind": "route", "explanation": "路线已规划。", "route_events_status": "unknown"}

    assert "沿途路况暂未核实" not in _model_delivery_text(result)


def test_read_failure_summary_omits_record_count_and_keeps_named_closure():
    from agents.dialogue_loop import _finish_with_read_failures

    class DialogueResult:
        candidates = route = comparison = clarify = None

        @staticmethod
        def result(message):
            return {"message": message}

    result = _finish_with_read_failures(
        DialogueResult(), "", {"weather_unavailable": "天气暂时获取不到，请稍后重试。"},
        {"list_road_conditions": {"conditions": [{"action": "closure", "action_label": "禁止通行",
                                                    "road_name": "桂园路", "name": "施工封闭"}],
                                  "count": 1}},
    )

    assert "桂园路" in result["message"] and "禁止通行" in result["message"]
    assert "1 条记录" not in result["message"]
    assert "weather_unavailable" not in result["message"]


def test_provider_failure_records_sanitized_completion_diagnostics(monkeypatch, caplog):
    import logging
    import config

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    events = []
    dialogue, create = make_dialogue("从信图去总图", history=[{"role": "user", "content": "PRIVATE_TEXT"}])
    error = RuntimeError("PRIVATE_PROVIDER_BODY")
    error.status_code = 500
    error.request_id = "req-test-123"
    create.side_effect = error
    dialogue.trace = events.append

    with caplog.at_level(logging.WARNING, logger="agents.dialogue_loop"):
        result = dialogue.run()

    completion = next(event for event in events if event.get("phase") == "completed")
    assert completion["outcome"] == "provider_error"
    assert completion["provider_status"] == "500"
    assert completion["request_id"] == "req-test-123"
    assert completion["duration_ms"] >= 0
    assert completion["request_summary"]["message_count"] > 1
    assert "PRIVATE_TEXT" not in repr(completion)
    assert "model_request_failed" in caplog.text and "http_500" in caplog.text
    assert "PRIVATE_PROVIDER_BODY" not in caplog.text
    assert result["failure_code"] == "model_unavailable"
    assert "PRIVATE_PROVIDER_BODY" not in result["message"]


def test_model_client_initialization_failure_records_terminal_event(monkeypatch, caplog):
    import logging
    import config

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    events = []
    dialogue, _ = make_dialogue("查一下天气")
    dialogue.trace = events.append
    dialogue.client_factory = lambda: (_ for _ in ()).throw(RuntimeError("PRIVATE_INIT_DETAIL"))

    with caplog.at_level(logging.WARNING, logger="agents.dialogue_loop"):
        result = dialogue.run()

    completion = next(event for event in events if event.get("phase") == "completed")
    assert completion["outcome"] == "client_error"
    assert completion["error_code"] == "model_not_configured"
    assert "PRIVATE_INIT_DETAIL" not in caplog.text
    assert result["failure_code"] == "model_unavailable"


def test_diagnostic_call_budget_is_not_reported_as_provider_failure(monkeypatch, caplog):
    import logging
    import config
    from agents.dialogue_loop import DiagnosticCallBudgetExceeded

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    events = []
    dialogue, create = make_dialogue("查一下天气")
    create.side_effect = DiagnosticCallBudgetExceeded("diagnostic cap")
    dialogue.trace = events.append

    with caplog.at_level(logging.WARNING, logger="agents.dialogue_loop"):
        result = dialogue.run()

    assert result["failure_code"] == "diagnostic_call_budget_exhausted"
    assert result["model_status"] == "budget_exhausted"
    assert not caplog.text
    assert any(event.get("outcome") == "diagnostic_budget_exhausted" for event in events)


def test_expand_radius_keeps_activity_anchors_and_previously_shown_ids():
    first = place("amap:A123", "甲餐厅")
    pending = {"request_id": "request", "original_query": "信图出发到总图附近吃饭", "mode": "walk",
               "start": {"type": "poi", "name": "信图"}, "search_anchor": {"type": "poi", "name": "总图"},
               "activities": ["meal"], "search_radius_m": 1000, "candidate_pool": [first],
               "candidate_places": [first], "candidate_poi_ids": [first["poi_id"]],
               "semantic_request": {"operation": "place_search", "shown_ids": [first["poi_id"]]}}
    dialogue, _ = make_dialogue("再远一些", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "expand_search_radius": True})
    assert dialogue.pending["search_radius_m"] == 2000
    assert dialogue.pending["start"]["name"] == "信图"
    assert dialogue.pending["search_anchor"]["name"] == "总图"
    assert dialogue.pending["activities"] == ["meal"]
    assert dialogue.intent["shown_ids"] == [first["poi_id"]]
    dialogue.update({"relation": "continue", "operation": "place_search", "expand_search_radius": True})
    assert dialogue.pending["search_radius_m"] == 2000
    next_turn, _ = make_dialogue("再扩大", context={"place_request": deepcopy(dialogue.pending)})
    next_turn.update({"relation": "continue", "operation": "place_search", "expand_search_radius": True})
    assert next_turn.pending["search_radius_m"] == 4000
    last_turn, _ = make_dialogue("再扩大", context={"place_request": deepcopy(next_turn.pending)})
    last_turn.update({"relation": "continue", "operation": "place_search", "expand_search_radius": True})
    assert last_turn.pending["search_radius_m"] == 5000


def test_explicit_radius_increase_keeps_previously_shown_places():
    candidate = place("amap:A123", "甲餐厅")
    pending = {"request_id": "request", "original_query": "总图附近吃饭", "mode": "walk",
               "search_anchor": {"type": "poi", "name": "总图"}, "search_radius_m": 1000,
               "candidate_pool": [candidate], "candidate_poi_ids": [candidate["poi_id"]],
               "semantic_request": {"operation": "place_search", "shown_ids": [candidate["poi_id"]]}}
    dialogue, _ = make_dialogue("范围扩大到两公里", context={"place_request": pending})

    dialogue.update({"relation": "continue", "operation": "place_search", "search_radius_m": 2000})

    assert dialogue.pending["search_radius_m"] == 2000
    assert dialogue.pending["candidate_pool"] == []
    assert dialogue.intent["shown_ids"] == [candidate["poi_id"]]


def test_changing_start_invalidates_candidates_dependent_on_the_old_origin():
    candidate = place("amap:A123", "甲餐厅")
    pending = {"request_id": "request", "original_query": "信图附近吃饭", "mode": "walk",
               "start": {"type": "poi", "name": "信图"},
               "candidate_pool": [candidate], "candidate_poi_ids": [candidate["poi_id"]],
               "semantic_request": {"operation": "place_search", "shown_ids": [candidate["poi_id"]]}}
    dialogue, _ = make_dialogue("现在从总图出发", context={"place_request": pending})

    dialogue.update({"relation": "continue", "operation": "place_search",
                     "start": {"type": "poi", "name": "总图"}})

    assert dialogue.pending["start"]["name"] == "总图"
    assert dialogue.pending["candidate_pool"] == []
    assert dialogue.pending["candidate_poi_ids"] == []


def test_unrequested_model_radius_cannot_replace_default_or_existing_radius():
    new, _ = make_dialogue("我从信图出发，在总图附近找三家餐馆")
    accepted = new.update({"relation": "new", "operation": "place_search", "search_radius_m": 1500})
    assert new.pending["search_radius_m"] == 1000
    assert "半径" in accepted["radius_note"]
    existing = {"request_id": "request", "original_query": "找餐馆", "search_radius_m": 1000}
    followup, _ = make_dialogue("范围宽一些", context={"place_request": existing})
    followup.update({"relation": "continue", "operation": "place_search",
                     "search_radius_m": 1500, "expand_search_radius": True})
    assert followup.pending["search_radius_m"] == 2000


def test_radius_expansion_requires_a_user_request_and_respects_negation():
    for query in ("找附近的羽毛球场", "不要扩大搜索范围", "别再往远处找了"):
        dialogue, _ = make_dialogue(query)
        response = dialogue.update({"relation": "new", "operation": "place_search",
                                    "expand_search_radius": True})
        assert dialogue.pending["search_radius_m"] == 1000
        assert "半径" in response["radius_note"]
    for query in ("范围宽一些", "再远一些", "能不能扩大范围"):
        pending = {"request_id": "request", "original_query": "找球场", "search_radius_m": 1000}
        dialogue, _ = make_dialogue(query, context={"place_request": pending})
        dialogue.update({"relation": "continue", "operation": "place_search", "expand_search_radius": True})
        assert dialogue.pending["search_radius_m"] == 2000


def test_explicit_distance_or_change_can_support_the_model_selected_radius():
    for text, radius in [("范围限定1.5公里", 1500), ("附近一千五百米", 1500),
                         ("半公里内", 500), ("一公里半内", 1500)]:
        dialogue, _ = make_dialogue(text)
        dialogue.update({"relation": "new", "operation": "place_search", "search_radius_m": radius})
        assert dialogue.pending["search_radius_m"] == radius
    previous = {"request_id": "request", "original_query": "找餐馆", "search_radius_m": 1000}
    for text, radius in [("范围再加500米", 1500), ("范围变为三倍", 3000)]:
        dialogue, _ = make_dialogue(text, context={"place_request": deepcopy(previous)})
        dialogue.update({"relation": "continue", "operation": "place_search", "search_radius_m": radius})
        assert dialogue.pending["search_radius_m"] == radius


def test_search_radius_in_natural_within_phrase_is_not_a_route_limit():
    query = "我在万林艺术博物馆，找两公里内的篮球场，步行。"
    dialogue, _ = make_dialogue(query)
    dialogue.update({"relation": "new", "operation": "place_search", "search_radius_m": 2000,
                     "constraints": {"max_distance_m": 2000}})
    assert dialogue.pending["search_radius_m"] == 2000
    assert dialogue.pending["constraints"]["max_distance_m"] is None


def test_within_search_radius_does_not_erase_a_separate_route_distance():
    dialogue, _ = make_dialogue("附近找两公里内的球场，路线最多两公里。")
    dialogue.update({"relation": "new", "operation": "place_search", "search_radius_m": 2000,
                     "constraints": {"max_distance_m": 2000}})
    assert dialogue.pending["constraints"]["max_distance_m"] == 2000


def test_searching_a_route_with_a_distance_limit_does_not_create_a_radius():
    from agents.distance_evidence import explicit_search_radius
    assert explicit_search_radius("搜索路线距离两公里以内") == (None, False)
    assert explicit_search_radius("附近找步行两公里内能到的篮球场") == (None, False)
    assert explicit_search_radius("找两公里内的篮球场，步行最多两公里") == (2000, True)


def test_expansion_overrides_conflicting_next_batch_and_requires_fresh_search():
    candidate = place("amap:A123", "候选餐厅")
    pending = {"request_id": "request", "original_query": "找餐馆", "search_radius_m": 1000,
               "candidate_pool": [candidate], "candidate_places": [candidate],
               "candidate_poi_ids": [candidate["poi_id"],],
               "semantic_request": {"operation": "place_search", "shown_ids": [candidate["poi_id"]]}}
    dialogue, _ = make_dialogue("范围放宽一些", context={"place_request": pending})

    accepted = dialogue.update({"relation": "continue", "operation": "place_search",
                                 "candidate_action": "next_batch", "expand_search_radius": True})

    assert dialogue.pending["search_radius_m"] == 2000
    assert dialogue.turn_action == "recommend"
    assert dialogue.pending["candidate_pool"] == []
    assert dialogue.pending["candidate_places"] == []
    assert dialogue.pending["candidate_poi_ids"] == []
    assert dialogue.intent["shown_ids"] == [candidate["poi_id"]]
    assert accepted["search_required"] is True
    assert "candidate_page" not in accepted


def test_browse_candidates_can_bind_only_the_explicit_task_revision():
    import pytest

    shown = place("amap:A123", "已展示餐厅")
    unshown = place("amap:B123", "未展示餐厅")
    pending = {"request_id": "request", "original_query": "找餐馆", "search_radius_m": 1000,
               "candidate_pool": [shown, unshown], "candidate_places": [shown],
               "candidate_poi_ids": [shown["poi_id"]],
               "semantic_request": {"operation": "place_search", "shown_ids": [shown["poi_id"]]}}
    dialogue, _ = make_dialogue("换几家", context={"available_tasks": [{
        "task_id": "task-123", "revision": 4, "state": {"place_request": pending},
    }]})
    bound = []
    dialogue.bind_task = lambda **proposal: (bound.append(proposal) or {
        "task_id": proposal["task_id"], "task_revision": 5,
    })

    page = dialogue.browse_candidates({"task_id": "task-123", "task_revision": 4})

    assert bound[0]["relation"] == "continue"
    assert bound[0]["task_id"] == "task-123" and bound[0]["base_revision"] == 4
    assert dialogue.bound_task["task_revision"] == 5
    assert [row["poi_id"] for row in page["candidates"]] == [unshown["poi_id"]]
    assert page["exhausted"] is True

    stale, _ = make_dialogue("换几家", context={"available_tasks": [{
        "task_id": "task-123", "revision": 4, "state": {"place_request": pending},
    }]})
    stale.bind_task = lambda **_proposal: pytest.fail("stale candidate pages must not bind a task")
    with pytest.raises(ValueError, match="task_version_invalid"):
        stale.browse_candidates({"task_id": "task-123", "task_revision": 3})


def test_route_preference_preserves_selected_place_when_search_cache_is_invalidated():
    selected = place("amap:A123", "已选餐厅")
    pending = {"request_id": "request", "original_query": "信图出发去总图附近吃饭",
               "mode": "walk", "start": {"type": "poi", "name": "信图"},
               "search_anchor": {"type": "poi", "name": "总图"},
               "search_radius_m": 2000, "selected_poi_id": selected["poi_id"],
               "candidate_poi_ids": [selected["poi_id"]], "candidate_places": [selected],
               "candidate_pool": [selected], "semantic_request": {"operation": "route"}}
    dialogue, _ = make_dialogue("路线尽量平坦", context={"available_tasks": [{
        "task_id": "task", "revision": 2, "state": {"place_request": pending},
    }]})
    dialogue.bind_task = lambda **_kwargs: {"task_id": "task", "task_revision": 3}

    dialogue.update({"relation": "continue", "task_id": "task", "task_revision": 2,
                     "operation": "route", "strategy": "flat", "requirements": ["尽量平坦"]})

    assert dialogue.pending["selected_poi_id"] == selected["poi_id"]
    assert dialogue.pending["selected_place"]["poi_id"] == selected["poi_id"]
    assert selected["poi_id"] in dialogue.known
    assert dialogue.pending["candidate_poi_ids"] == []
    assert dialogue.pending["candidate_pool"] == []


def test_resuming_flat_route_preserves_search_and_normalizes_route_constraints():
    selected = place("amap:A123", "已选餐厅")
    pending = {"request_id": "request", "original_query": "从信图找总图附近的餐馆",
               "mode": "walk", "start": {"type": "poi", "name": "信图"},
               "search_anchor": {"type": "poi", "name": "总图"}, "search_radius_m": 2000,
               "selected_poi_id": selected["poi_id"], "selected_place": selected,
               "constraints": {"avoid_steps": True, "max_duration_s": 1200},
               "semantic_request": {"operation": "route", "requirements": ["不推荐食堂"]}}
    route = {"start": pending["start"], "end": {"type": "poi", "poi_id": selected["poi_id"]},
             "travel_mode": "walk", "hard_constraints": {"slope": "prefer", "avoid_steps": True},
             "strategy": {"name": "flat"}}
    dialogue, _ = make_dialogue("回到之前的路线，改走最短的", context={"available_tasks": [{
        "task_id": "task", "revision": 9, "state": {"place_request": pending}, "route_state": route,
    }]})
    dialogue.bind_task = lambda **_kwargs: {"task_id": "task", "task_revision": 10}

    dialogue.update({"relation": "resume_route", "task_id": "task", "task_revision": 9,
                     "operation": "route", "strategy": "shortest"})

    assert dialogue.pending["search_anchor"]["name"] == "总图"
    assert dialogue.pending["search_radius_m"] == 2000
    assert dialogue.pending["selected_poi_id"] == selected["poi_id"]
    assert dialogue.pending["semantic_request"]["requirements"] == ["不推荐食堂"]
    assert dialogue.pending["constraints"]["avoid_steps"] is True
    assert dialogue.pending["constraints"]["avoid_slope"] is False
    assert dialogue.pending["constraints"]["max_duration_s"] == 1200
    assert dialogue.pending["strategy"] == "shortest"


def test_resuming_route_restores_legacy_slope_hard_constraint():
    pending = {"request_id": "request", "original_query": "从信图去总图", "mode": "walk",
               "constraints": {"avoid_steps": True}, "semantic_request": {"operation": "route"}}
    route = {"start": {"type": "poi", "name": "信图"}, "travel_mode": "walk",
             "hard_constraints": {"slope": "avoid", "max_distance_m": 2500},
             "strategy": {"name": "shortest"}}
    dialogue, _ = make_dialogue("继续路线", context={"available_tasks": [{
        "task_id": "task", "revision": 4, "state": {"place_request": pending}, "route_state": route,
    }]})
    dialogue.bind_task = lambda **_kwargs: {"task_id": "task", "task_revision": 5}

    dialogue.update({"relation": "resume_route", "task_id": "task", "task_revision": 4,
                     "operation": "route"})

    assert dialogue.pending["constraints"]["avoid_slope"] is True
    assert dialogue.pending["constraints"]["avoid_steps"] is True
    assert dialogue.pending["constraints"]["max_distance_m"] == 2500


def test_soft_route_preference_does_not_silently_clear_existing_hard_constraints():
    pending = {"request_id": "request", "original_query": "规划无台阶且避坡的路线", "mode": "walk",
               "constraints": {"avoid_slope": True, "avoid_steps": True, "max_duration_s": 1200},
               "semantic_request": {"operation": "route"}}
    dialogue, _ = make_dialogue("尽量平坦", context={"place_request": pending})

    dialogue.update({"relation": "continue", "operation": "route", "strategy": "flat",
                     "constraints": {"slope": "prefer", "avoid_steps": False}})

    assert dialogue.pending["constraints"]["avoid_slope"] is True
    assert dialogue.pending["constraints"]["avoid_steps"] is True
    assert dialogue.pending["constraints"]["max_duration_s"] == 1200


def test_explicitly_clearing_route_constraints_preserves_other_conditions():
    pending = {"request_id": "request", "original_query": "规划无台阶且避坡的路线", "mode": "walk",
               "constraints": {"avoid_slope": True, "avoid_steps": True, "max_duration_s": 1200},
               "semantic_request": {"operation": "route"}}
    dialogue, _ = make_dialogue("不再要求避坡和无台阶", context={"place_request": pending})

    dialogue.update({"relation": "continue", "operation": "route",
                     "clear_fields": ["constraints.avoid_slope", "constraints.avoid_steps"]})

    assert dialogue.pending["constraints"]["avoid_slope"] is False
    assert dialogue.pending["constraints"]["avoid_steps"] is False
    assert dialogue.pending["constraints"]["max_duration_s"] == 1200


def test_provider_failure_is_reported_without_repeated_network_calls(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    calls = []
    dialogue, model = make_dialogue("附近找吃的", replies=[
        reply(update_task={"relation": "new", "operation": "place_search"},
              search_poi_candidates={"keyword": "餐饮"}),
        reply(search_poi_candidates={"keyword": "餐厅"}),
    ])
    dialogue.execute_tool = lambda name, args, context: (calls.append(name) or {
        "status": "provider_error", "online_status": "provider_error", "candidates": []}, {})
    result = dialogue.run()
    assert calls == ["search_poi_candidates"]
    assert model.call_count == 2
    assert result["execution_status"] == "failed"
    assert result["failure_code"] == "place_service_unavailable"
    assert result["response_kind"] != "clarify"


def test_weather_failure_is_retryable_even_if_model_answers_normally(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    dialogue, _ = make_dialogue("天气怎么样", replies=[
        reply(get_weather={}), reply("天气暂时查不到。"),
    ])
    dialogue.execute_tool = lambda name, args, context: (
        {"error": "weather_unavailable", "message": "天气暂时获取不到"}, None,
    )

    result = dialogue.run()

    assert result["execution_status"] == "failed"
    assert result["failure_code"] == "weather_unavailable"
    assert result["response_kind"] != "clarify"
    assert "天气" in result["message"]


def test_successful_tool_retry_clears_earlier_weather_failure(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    dialogue, _ = make_dialogue("天气怎么样", replies=[
        reply(get_weather={}), reply(get_weather={}), reply("当前晴朗。"),
    ])
    outcomes = iter([
        ({"error": "weather_unavailable", "message": "天气暂时获取不到"}, None),
        ({"weather": "晴"}, None),
    ])
    dialogue.execute_tool = lambda name, args, context: next(outcomes)

    result = dialogue.run()

    assert result.get("execution_status") != "failed"
    assert result["weather"]["weather"] == "晴"


def test_successful_weather_tool_returns_the_checked_result_without_a_restatement_call(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    dialogue, model = make_dialogue("武汉今天的天气怎么样？", replies=[
        reply(update_task={"relation": "new", "operation": "weather"},
              get_weather={"end_turn_if_success": True}),
    ])
    executed = []
    dialogue.bind_task = lambda **_kwargs: (
        executed.append("bind_task") or {"task_id": "task", "task_revision": 1}
    )
    dialogue.execute_tool = lambda _name, _args, _context: (
        executed.append("get_weather") or
        {"weather": "晴", "temperature": 22, "windpower": "微风", "slippery": False}, None,
    )

    result = dialogue.run()

    assert model.call_count == 1
    assert executed == ["bind_task", "get_weather"]
    assert result["weather"]["weather"] == "晴"
    assert result["message"] == "武汉今天晴。气温22℃。"


def test_weather_is_not_delivered_early_without_model_finish_signal(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    dialogue, model = make_dialogue("武汉今天的天气怎么样？", replies=[
        reply(get_weather={"end_turn_if_success": False}), reply("武汉今天晴。"),
    ])
    dialogue.execute_tool = lambda _name, _args, _context: (
        {"weather": "晴", "temperature": 22}, None,
    )

    result = dialogue.run()

    assert model.call_count == 2
    assert result["message"] == "武汉今天晴。"


def test_verified_direct_route_returns_the_checked_result_without_a_restatement_call(monkeypatch):
    from agents import tools

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue, model = make_dialogue("尽量走平坦些", replies=[
        reply(plan_route={"end_turn_if_success": True}), reply("已按平坦偏好规划好路线。"),
    ])
    route = {"route_verification": {"status": "pass"},
             "route_state": {"strategy": {"name": "flat"}}}
    dialogue.execute_tool = lambda _name, _args, _context: ({"status": "ok"}, {"route": route})

    result = dialogue.run()

    assert model.call_count == 1
    assert result["response_kind"] == "route"
    assert result["route"]["route_verification"]["status"] == "pass"
    assert result["message"] == "已按平坦偏好规划好路线。"


def test_direct_terminal_choice_is_required_on_weather_and_route_tool_calls():
    from agents.dialogue import tool_schemas

    schemas = {item["function"]["name"]: item["function"] for item in tool_schemas()}
    assert "finish_turn" not in schemas

    for name in ("get_weather", "plan_route"):
        parameters = schemas[name]["parameters"]
        assert "end_turn_if_success" in parameters["required"]
        decision = parameters["properties"]["end_turn_if_success"]
        assert decision["type"] == "boolean"
        assert "其他子任务" in decision["description"]

    browse = schemas["browse_candidates"]["parameters"]["properties"]
    assert browse["task_id"]["type"] == "string"
    assert browse["task_revision"]["type"] == "integer"
    present = schemas["present_places"]["parameters"]["properties"]["places"]["items"]["properties"]
    assert "空字符串" in present["reason"]["description"]
    assert "source_id" in present["source_ids"]["description"]
    assert "data_source" in present["source_ids"]["description"]
    update_properties = schemas["update_task"]["parameters"]["properties"]
    assert "搜索区域代替" in update_properties["start"]["description"]
    assert "不能复制start" in update_properties["search_anchor"]["description"]
    clearable = update_properties["clear_fields"]["items"]["enum"]
    assert "constraints.avoid_slope" in clearable
    assert "constraints.slope" not in clearable
    assert "修改路线偏好只更新strategy" in update_properties["clear_fields"]["description"]


def test_repeated_weather_failure_reports_service_cause_not_validation(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    dialogue, model = make_dialogue("天气如何", replies=[reply(get_weather={}) for _ in range(3)])
    dialogue.execute_tool = lambda name, args, context: (
        {"error": "weather_unavailable", "message": "天气暂时获取不到"}, None,
    )

    result = dialogue.run()

    assert model.call_count == 3
    assert result["failure_code"] == "weather_unavailable"
    assert result["execution_status"] == "failed"
    assert "天气" in result["message"]


def test_road_failure_keeps_successful_weather_as_partial_result(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    dialogue, _ = make_dialogue("天气和路况如何", replies=[
        reply(get_weather={"end_turn_if_success": True}, list_road_conditions={}),
        reply("天气晴，路况暂不可用。"),
    ])
    dialogue.execute_tool = lambda name, args, context: (
        ({"weather": "晴"}, None) if name == "get_weather"
        else ({"error": "road_conditions_unavailable", "message": "路况暂不可用"}, None)
    )

    result = dialogue.run()

    assert result["execution_status"] == "partial"
    assert result["failure_code"] == "road_conditions_unavailable"
    assert result["weather"]["weather"] == "晴"
    assert "路况" in result["message"]


def test_later_empty_search_does_not_discard_candidate_snapshot(monkeypatch):
    from agents import route_state, tools
    from storage.task_repository import _maintenance_snapshot_is_current
    candidate = place("amap:A123", "候选餐厅")
    pending = {"request_id": "request", "original_query": "信图出发，总图附近吃饭",
               "start": {"type": "poi", "name": "信图"}, "search_anchor": {"type": "poi", "name": "总图"},
               "subcategories": ["restaurant"], "search_radius_m": 1000,
               "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("找吃饭地点", context={"place_request": pending})
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue.execute_tool = Mock(side_effect=[
        ({"status": "ok", "candidates": [candidate], "data_version": "data-current",
          "road_condition_version": "roads-current"}, {"candidates": [candidate]}),
        ({"status": "ok", "candidates": [], "count": 0}, {"candidates": []}),
    ])
    dialogue.spatial("search_poi_candidates", {"keyword": "餐馆"})
    dialogue.spatial("search_poi_candidates", {"keyword": "另一关键词"})
    assert [row["poi_id"] for row in dialogue.pending["candidate_pool"]] == [candidate["poi_id"]]
    text = dialogue.present({"introduction": "候选", "places": [
        {"poi_id": candidate["poi_id"], "reason": "目录候选"}]})
    result = dialogue.result(text)
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-current")
    monkeypatch.setattr(route_state, "current_road_condition_version", lambda _conditions: "roads-current")
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])
    assert _maintenance_snapshot_is_current(result)


def test_repeated_identical_tool_validation_errors_stop_without_exhausting_the_budget(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    invalid = reply(present_places={"introduction": "候选", "places": [
        {"poi_id": "amap:UNKNOWN", "reason": "目录记录"}]})
    dialogue, model = make_dialogue("推荐餐馆", replies=[
        reply(update_task={"relation": "new", "operation": "place_search"}), invalid, invalid, invalid,
    ])
    result = dialogue.run()
    assert model.call_count == 4
    assert result["execution_status"] == "failed"
    assert result["failure_code"] == "repeated_tool_error"


def test_continued_task_restores_its_previously_accepted_web_sources():
    source = {"source_id": "S123", "url": "https://example.test/place", "title": "来源"}
    pending = {"request_id": "request", "original_query": "找餐馆",
               "semantic_request": {"sources": [source]}}
    dialogue, _ = make_dialogue("继续", context={"available_tasks": [{
        "task_id": "task", "revision": 2, "state": {"place_request": pending},
    }]})
    dialogue.bind_task = lambda **_kwargs: {"task_id": "task", "task_revision": 3}
    dialogue.update({"relation": "continue", "task_id": "task", "task_revision": 2, "operation": "place_search"})
    assert dialogue.sources["S123"] == source


def test_model_context_limit_explicitly_offers_a_new_conversation(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    class ContextLimit(Exception):
        status_code = 400
    dialogue, model = make_dialogue("继续")
    model.side_effect = ContextLimit("maximum context length exceeded")
    result = dialogue.run()
    assert result["failure_code"] == "context_length_exceeded"
    assert "新建会话" in result["message"]


def test_provider_timeout_uses_the_remaining_shared_deadline(monkeypatch):
    from agents import deadline
    from spatial.amap_poi import _provider_get
    client = Mock()
    monkeypatch.setattr(deadline.time, "monotonic", lambda: 100.0)
    with deadline.shared_deadline(102.0):
        _provider_get(client, "https://example.invalid", params={})
    assert client.get.call_args.kwargs["timeout"] == 2.0
    with deadline.shared_deadline(100.0):
        import pytest
        with pytest.raises(TimeoutError):
            _provider_get(client, "https://example.invalid", params={})


def test_unavailable_route_comparison_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    from agents import tools
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _ctx: ({}, None))
    dialogue, _ = make_dialogue("更平坦一些", replies=[
        reply(compare_routes={}), reply("路线比较已经完成。"),
    ])
    dialogue.execute_tool = lambda *_args: (
        {"error": "route_comparison_unavailable", "message": "路线缺少坡度证据，无法核验。"},
        {"route_comparison": {"status": "unavailable", "alternatives": []}})
    result = dialogue.run()
    assert result["execution_status"] == "failed"
    assert result["failure_code"] == "route_comparison_unavailable"
    assert "缺少坡度证据" in result["message"]


def test_partial_requirement_update_does_not_silently_drop_existing_exclusions():
    pending = {"request_id": "request", "original_query": "找餐馆", "mode": "walk",
               "semantic_request": {"operation": "place_search", "requirements": ["不要食堂档口"],
                                    "exclude_terms": ["食堂"]}}
    dialogue, _ = make_dialogue("安静一点", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search",
                     "requirements": ["安静"], "exclude_terms": []})
    assert dialogue.intent["requirements"] == ["不要食堂档口", "安静"]
    assert dialogue.intent["exclude_terms"] == ["食堂"]
    dialogue.update({"relation": "continue", "operation": "place_search",
                     "clear_fields": ["requirements"], "requirements": ["不用安静"]})
    assert dialogue.intent["requirements"] == ["不用安静"]


def test_changing_activity_removes_only_the_obsolete_plain_activity_requirement():
    pending = {"request_id": "request", "original_query": "附近找咖啡", "mode": "walk",
               "activities": ["coffee"], "subcategories": ["coffee"],
               "semantic_request": {"operation": "place_search",
                                    "requirements": ["咖啡店", "避开台阶", "咖啡店需要无障碍入口"]}}
    dialogue, _ = make_dialogue("改成吃饭的地方", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "activities": ["meal"],
                     "subcategories": ["canteen", "restaurant", "fastfood"],
                     "requirements": ["吃饭的地方"]})
    requirements = dialogue.intent["requirements"]
    assert "咖啡店" not in requirements
    assert "避开台阶" in requirements
    assert "咖啡店需要无障碍入口" in requirements
    assert "吃饭的地方" in requirements


def test_changing_activity_keeps_shared_qualifiers_without_the_old_sport():
    pending = {"request_id": "request", "original_query": "步行十五分钟内找篮球场", "mode": "walk",
               "activities": ["basketball"], "subcategories": ["field"],
               "constraints": {"avoid_steps": True, "max_duration_s": 900},
               "semantic_request": {"operation": "place_search", "requirements": [
                   "步行可达的篮球场", "步行15分钟内的篮球场", "避开台阶"]}}
    dialogue, _ = make_dialogue("不打篮球了，帮我找羽毛球场", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "activities": ["badminton"],
                     "subcategories": ["court"], "requirements": ["羽毛球场"]})
    assert dialogue.intent["requirements"] == ["步行可达", "步行15分钟内", "避开台阶", "羽毛球场"]
    assert dialogue.pending["mode"] == "walk"
    assert dialogue.pending["constraints"]["avoid_steps"] is True
    assert dialogue.pending["constraints"]["max_duration_s"] == 900


def test_reordering_the_same_stops_replaces_only_the_old_plain_order_requirement():
    pending = {"request_id": "request", "original_query": "先到甲，再到乙", "mode": "walk",
               "semantic_request": {"operation": "route", "requirements": ["先到甲，再到乙", "避开台阶"]}}
    dialogue, _ = make_dialogue("把顺序换一下，先到乙", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "route",
                     "requirements": ["先到乙，再到甲"]})
    assert dialogue.intent["requirements"] == ["避开台阶", "先到乙，再到甲"]


def test_reordering_keeps_the_new_order_if_model_also_repeats_the_old_one():
    pending = {"request_id": "request", "original_query": "先到甲，再到乙", "mode": "walk",
               "semantic_request": {"operation": "route", "requirements": ["先到甲，再到乙", "避开台阶"]}}
    dialogue, _ = make_dialogue("把顺序换一下，先到乙", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "route",
                     "requirements": ["先到甲，再到乙", "先到乙，再到甲"]})
    assert dialogue.intent["requirements"] == ["避开台阶", "先到乙，再到甲"]


def test_reordering_split_stop_requirements_removes_only_the_old_order():
    pending = {"request_id": "request", "original_query": "先到甲，再到乙", "mode": "walk",
               "semantic_request": {"operation": "route", "requirements": ["先到甲", "再到乙", "避开台阶"]}}
    dialogue, _ = make_dialogue("把顺序换一下，先到乙", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "route",
                     "requirements": ["先到乙", "再到甲"]})
    assert dialogue.intent["requirements"] == ["避开台阶", "先到乙", "再到甲"]


def test_reordering_arrow_notation_replaces_only_the_old_sequence():
    pending = {"request_id": "request", "original_query": "从起点先到甲再到乙", "mode": "walk",
               "semantic_request": {"operation": "route", "requirements": ["步行", "顺序：起点→甲→乙"]}}
    dialogue, _ = make_dialogue("把顺序换一下，先到乙", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "route",
                     "requirements": ["顺序：起点→乙→甲"]})
    assert dialogue.intent["requirements"] == ["步行", "顺序：起点→乙→甲"]


def test_a_new_task_cannot_invent_the_former_destination_as_current_start():
    route = {"start": {"type": "poi", "name": "甲地"}, "end": {"type": "poi", "name": "乙地"},
             "travel_mode": "walk", "hard_constraints": {}}
    context = {"previous_route_state": route}
    dialogue, _ = make_dialogue("继续步行", context=context)
    with pytest.raises(ValueError, match="unconfirmed_current_position"):
        dialogue.update({"relation": "new", "operation": "route", "start": route["end"]})
    arrived, _ = make_dialogue("我已经到乙地了，接着走", context=context)
    arrived.update({"relation": "new", "operation": "route", "start": route["end"]})
    assert arrived.pending["start"]["name"] == "乙地"


def test_an_explicit_current_position_can_use_a_verified_place_alias():
    route = {"start": {"type": "poi", "name": "凌波门"},
             "end": {"type": "poi", "name": "武汉大学网球场"},
             "travel_mode": "walk", "hard_constraints": {}}
    dialogue, _ = make_dialogue("我现在在工学部网球场，从这里继续", context={"previous_route_state": route})
    dialogue.update({"relation": "new", "operation": "route", "start": route["end"]})
    assert dialogue.pending["start"]["name"] == "武汉大学网球场"


def test_a_clarification_cannot_assume_the_user_reached_the_old_destination():
    from agents import tools
    route = {"start": {"type": "poi", "name": "甲地"}, "end": {"type": "poi", "name": "乙地"},
             "travel_mode": "walk", "hard_constraints": {}}
    pending = {"request_id": "request", "original_query": "从甲到乙", "mode": "walk",
               "start": route["start"], "semantic_request": {"operation": "route", "route_to_modify": route}}
    dialogue, _ = make_dialogue("继续步行", context={"place_request": pending, "previous_route_state": route})
    dialogue.execute_tool = tools.execute_tool
    result = dialogue.spatial("ask_user", {"question": "接着从乙地往前走，下一个去哪儿？", "slot": "destination"})
    assert result["error"] == "unconfirmed_current_position"
    assert dialogue.clarify is None

    arrived, _ = make_dialogue("我已经到了乙地，接下来去哪", context={"place_request": pending,
                                                               "previous_route_state": route})
    arrived.execute_tool = tools.execute_tool
    result = arrived.spatial("ask_user", {"question": "接着从乙地往前走，下一个去哪儿？", "slot": "destination"})
    assert result["question"] == "接着从乙地往前走，下一个去哪儿？"


def test_changing_category_removes_old_activity_even_if_model_never_saved_activities():
    pending = {"request_id": "request", "original_query": "附近找咖啡", "mode": "walk",
               "subcategories": ["coffee"],
               "semantic_request": {"operation": "place_search", "requirements": ["咖啡店", "避开台阶"]}}
    dialogue, _ = make_dialogue("不喝咖啡了，换成吃饭的地方", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "activities": [],
                     "subcategories": ["canteen", "restaurant", "fastfood"], "requirements": ["吃饭的地方"]})
    assert dialogue.intent["requirements"] == ["避开台阶", "吃饭的地方"]


def test_changing_categories_keeps_an_activity_still_among_the_options():
    pending = {"request_id": "request", "original_query": "附近找咖啡", "mode": "walk",
               "subcategories": ["coffee"],
               "semantic_request": {"operation": "place_search", "requirements": ["咖啡店"]}}
    dialogue, _ = make_dialogue("再加上餐厅", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search",
                     "subcategories": ["coffee", "restaurant"]})
    assert "咖啡店" in dialogue.intent["requirements"]


def test_generic_ball_search_includes_provider_courts_despite_a_narrow_inferred_category(monkeypatch):
    from agents import discovery, tools

    venue = {"id": "amap:SPORT001", "name": "校园篮球场", "source": "amap_online",
             "coordinates": {"lng": 114.36, "lat": 30.54}, "lon": 114.36, "lat": 30.54,
             "type": "sports", "category": "sports", "subcategory": "court",
             "type_name": "体育休闲服务;运动场馆;篮球场馆", "type_code": "080104"}
    provider_calls = []
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _ctx: ({}, None))
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "_endpoint_target_wgs", lambda *_args: (114.36, 30.54, "起点", None))
    monkeypatch.setattr(discovery, "wgs84_to_gcj02", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(discovery.amap_poi, "search_nearby", lambda _center, **kwargs: (
        provider_calls.append(kwargs) or {"status": "ok", "candidates": [venue]}))

    def reachable(_args, context):
        rows = [{"poi_id": row["poi_id"], "name": row["name"], "type": row["type"],
                 "subcategory": row["subcategory"], "network_distance_m": 400}
                for row in context["_candidate_pool"]]
        return {"candidates": rows, "count": len(rows), "start_name": "起点"}, {"candidates": rows}

    monkeypatch.setattr(tools, "_tool_find_reachable_places", reachable)
    dialogue, _ = make_dialogue("从星湖体育馆出发，找附近可以打篮球的地方。")
    dialogue.update({"relation": "new", "operation": "place_search", "activities": ["basketball"],
                     "subcategories": ["field"], "start": {"type": "poi", "name": "星湖体育馆"}})
    dialogue.execute_tool = lambda _name, args, context: discovery.search_places_for_dialogue(args, context)
    answer = dialogue.spatial("search_poi_candidates", {"subcategory": "field", "keyword": "篮球场"})
    assert [row["poi_id"] for row in answer["candidates"]] == ["amap:SPORT001"]
    assert provider_calls[0]["poi_type"] == "sports"
    assert "court" in dialogue.pending["subcategories"]


def test_explicit_venue_type_remains_a_restriction_for_a_sport_search():
    dialogue, _ = make_dialogue("从珞珈门出发，只找体育馆里的羽毛球场。")
    dialogue.update({"relation": "new", "operation": "place_search", "activities": ["badminton"],
                     "subcategories": ["gym"], "start": {"type": "poi", "name": "珞珈门"}})
    assert dialogue.pending["subcategories"] == ["gym"]


def test_sport_venue_type_remains_restricted_on_followup():
    pending = {"request_id": "request", "original_query": "只找体育馆里的羽毛球场",
               "mode": "walk", "activities": ["badminton"], "subcategories": ["gym"],
               "semantic_request": {"operation": "place_search", "requirements": ["体育馆里"]}}
    dialogue, _ = make_dialogue("范围扩大一点", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "activities": ["badminton"],
                     "subcategories": ["gym"], "expand_search_radius": True})
    assert dialogue.pending["subcategories"] == ["gym"]


def test_an_explicitly_retained_activity_is_not_removed_by_a_conflicting_category():
    pending = {"request_id": "request", "original_query": "附近找咖啡", "mode": "walk",
               "activities": ["coffee"], "subcategories": ["coffee"],
               "semantic_request": {"operation": "place_search", "requirements": ["咖啡店"]}}
    dialogue, _ = make_dialogue("再加上餐厅", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "place_search", "activities": ["coffee", "meal"],
                     "subcategories": ["restaurant"]})
    assert "咖啡店" in dialogue.intent["requirements"]


def test_removing_a_via_stop_removes_its_plain_requirement_but_preserves_others():
    via = {"type": "poi", "name": "梅园快递服务中心"}
    end = {"type": "poi", "name": "万林艺术博物馆"}
    pending = {"request_id": "request", "original_query": "先取快递再参观", "mode": "walk",
               "semantic_request": {"operation": "itinerary", "tour_stops": [via["name"], end["name"]],
                   "route_to_modify": {"via": via, "end": end},
                   "requirements": ["先去梅园快递服务中心", "最后到万林艺术博物馆", "避开台阶"]}}
    dialogue, _ = make_dialogue("快递先不取了", context={"place_request": pending})
    dialogue.update({"relation": "continue", "operation": "itinerary", "tour_stops": [end["name"]],
                     "requirements": ["最后到万林艺术博物馆"]})
    requirements = dialogue.intent["requirements"]
    assert "先去梅园快递服务中心" not in requirements
    assert "最后到万林艺术博物馆" in requirements
    assert "避开台阶" in requirements


def test_model_can_read_a_paged_result_reference_without_binding_a_task(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    calls = []
    dialogue, model = make_dialogue("看看之前那条路线", replies=[
        reply(read_result={"run_id": "prior-run", "section": "route_geometry", "offset": 0, "limit": 2}),
        reply("已查看路线记录。"),
    ])
    dialogue.bind_task = lambda **_args: (_ for _ in ()).throw(AssertionError("must remain unbound"))
    dialogue.read_result = lambda **kwargs: calls.append(kwargs) or {
        "items": [[114.3, 30.5], [114.4, 30.6]], "offset": 0, "exhausted": False}
    result = dialogue.run()
    assert result["response_kind"] == "chat"
    assert calls == [{"run_id": "prior-run", "section": "route_geometry", "offset": 0, "limit": 2}]
    assert "114.3" in str(model.call_args_list[1].kwargs["messages"])


def test_name_must_not_be_accepted_as_a_fabricated_local_poi_id():
    import pytest
    dialogue, _ = make_dialogue("总图附近找餐厅")
    with pytest.raises(ValueError, match="unknown_place_id"):
        dialogue.update({"relation": "new", "operation": "place_search",
                         "search_anchor": {"type": "poi", "name": "武汉大学图书馆(总馆)",
                                           "poi_id": "武汉大学图书馆(总馆)"}})


def test_resolving_local_poi_returns_the_stable_id():
    from agents.tools import _tool_resolve_poi
    result, _ = _tool_resolve_poi({"name": "武汉大学图书馆(总馆)"}, {})
    assert result["found"] is True
    assert result["poi"].get("poi_id", "").startswith("poi_")


def test_structured_restaurant_category_can_search_without_a_keyword(monkeypatch):
    from agents import discovery, tools
    received = []
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_a, **_k: [])
    monkeypatch.setattr(tools, "_endpoint_target_wgs", lambda *_a: (114.36, 30.54, "总图", None))
    monkeypatch.setattr(discovery.amap_poi, "search_nearby", lambda _center, **kwargs: (
        received.append(kwargs) or {"status": "ok", "candidates": []}))
    discovery.search_places_for_dialogue({"search_anchor": {"type": "poi", "name": "总图"},
                                         "subcategories": ["restaurant"]}, {"_semantic_request": {}})
    assert received and received[0]["keywords"] == "餐饮"


def test_independent_weather_and_road_reads_can_run_concurrently(monkeypatch):
    import threading
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    barrier = threading.Barrier(2, timeout=2)
    dialogue, _ = make_dialogue("天气和路况", replies=[
        reply(get_weather={}, list_road_conditions={}), reply("已查询天气和路况。"),
    ])
    def spatial(name, args):
        barrier.wait()
        return {"status": "ok", "tool": name}
    dialogue.spatial = spatial
    result = dialogue.run()
    assert result["response_kind"] == "chat"
    assert result["message"] == "已查询天气和路况。"
