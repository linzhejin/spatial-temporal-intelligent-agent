"""Regression contracts for model-led turns and complete conversation input."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import config
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
    assert "仍有 1 条未展示记录" in result["message"]


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


def test_model_route_delivery_text_includes_the_displayed_safety_notice():
    from agents.workflow import _model_delivery_text
    result = {"response_kind": "route", "explanation": "路线已规划。",
              "route_verification": {"status": "pass", "field_truth_verified": False},
              "route_events": [{"road_name": "科技路", "name": "人流较多", "action": "slowdown"}]}
    text = _model_delivery_text(result)
    assert text.startswith("路线已规划。")
    assert "未实地确认道路当前可通行" in text
    assert "科技路" in text and "人流较多" in text


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


def test_rejected_expand_and_next_batch_update_does_not_consume_the_expansion():
    import pytest
    candidate = place("amap:A123", "候选餐厅")
    pending = {"request_id": "request", "original_query": "找餐馆", "search_radius_m": 1000,
               "candidate_pool": [candidate], "semantic_request": {"operation": "place_search"}}
    dialogue, _ = make_dialogue("范围放宽一些", context={"place_request": pending})
    with pytest.raises(ValueError, match="next_batch requires"):
        dialogue.update({"relation": "continue", "operation": "place_search",
                         "candidate_action": "next_batch", "expand_search_radius": True})
    assert dialogue.pending["search_radius_m"] == 1000
    assert candidate["poi_id"] in dialogue.known
    dialogue.update({"relation": "continue", "operation": "place_search", "expand_search_radius": True})
    assert dialogue.pending["search_radius_m"] == 2000


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
