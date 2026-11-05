"""Natural language is interpreted by the model, with tool-grounded delivery."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import planner


def test_model_dialogue_is_the_default_mode(monkeypatch):
    from agents.dialogue import model_dialogue_enabled
    monkeypatch.delenv("AGENT_DIALOGUE_MODE", raising=False)
    assert model_dialogue_enabled()


def test_model_receives_compact_place_candidates_without_geometry():
    import json
    from agents.dialogue import _model_tool_content

    candidates = [{"poi_id": f"amap:{index}", "name": f"店铺{index}",
                  "coordinates": {"lng": 114.36, "lat": 30.54}, "lng": 114.36, "lat": 30.54,
                  "provider_cost": "人均50元", "distance_m": index * 100,
                  "description": "d" * 900} for index in range(15)]

    result = json.loads(_model_tool_content("search_poi_candidates", {
        "status": "ok", "count": 15, "candidates": candidates,
    }))

    assert len(result["candidates"]) == 12
    assert result["candidate_batch_truncated"] is True
    assert result["candidate_batch_has_more"] is True
    assert result["candidates"][0]["provider_cost"] == "人均50元"
    assert result["candidates"][0]["distance_m"] == 0
    assert len(result["candidates"][0]["description"]) == 500
    assert "coordinates" not in result["candidates"][0]
    assert "lng" not in result["candidates"][0]


def reply(text="", **calls):
    import json
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=text, reasoning_content=None,
        tool_calls=[SimpleNamespace(id=f"call_{name}", type="function", function=SimpleNamespace(
            name=name, arguments=json.dumps(args, ensure_ascii=False),
        )) for name, args in calls.items()],
    ))], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20))


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("AGENT_DIALOGUE_MODE", "model")
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "unit-test-only")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
    monkeypatch.setattr(planner, "_make_client", lambda: client)
    return client.chat.completions.create


def place(identifier, name):
    return {"poi_id": identifier, "id": identifier, "name": name, "source": "amap_online",
            "coordinates": {"lng": 114.36, "lat": 30.54}, "coordinate_system": "GCJ-02",
            "subcategory": "restaurant", "verification_status": "provider_only"}


def test_full_dining_requirements_reach_model_and_it_selects_the_delivered_places(model, monkeypatch):
    query = "推荐武大校内独立火锅店，不要食堂档口，人均50元以内，适合三个人聚餐"
    candidates = [place("amap:A123", "测试火锅甲"), place("amap:B123", "测试火锅乙")]
    tools = []
    def execute(name, args, _context):
        tools.append((name, args))
        return {"status": "ok", "candidates": candidates}, {"candidates": candidates}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "goal": query,
                           "requirements": ["独立门店", "不含食堂档口", "三人聚餐"],
                           "budget_per_person": 50, "keywords": ["火锅"], "subcategories": ["restaurant"]}),
        reply(search_poi_candidates={"keyword": "火锅", "area": "武汉大学", "subcategories": ["restaurant"]}),
        reply(present_places={"introduction": "找到一个可进一步核对的选择，人均消费暂未核实。",
                              "places": [{"poi_id": "amap:B123", "reason": "地图记录为火锅餐厅，是否独立门店仍需确认。"}]}),
    ]
    result = planner.run_agent(query)
    assert model.call_count == 3
    assert query in str(model.call_args_list[0].kwargs["messages"])
    assert result["turns"] == 3
    assert result["response_kind"] == "candidates"
    assert [p["poi_id"] for p in result["candidates"]] == ["amap:B123"]
    assert "测试火锅甲" not in result["message"]
    assert "测试火锅乙" in result["message"]
    intent = result["place_request"]["semantic_request"]
    assert intent["budget_per_person"] == 50
    assert "不含食堂档口" in intent["requirements"]
    assert tools[0][1]["keyword"] == "火锅"


def test_general_recommendation_does_not_require_origin_before_model(model, monkeypatch):
    model.side_effect = [reply("可以先比较校内不同区域的店铺，确定地点后再计算从你出发的路线。")]
    result = planner.run_agent("推荐几家武大校内适合聚餐的火锅店")
    assert model.call_count == 1
    assert result["response_kind"] == "chat"
    assert "你从哪里出发" not in result["message"]


def test_rich_followup_gets_durable_origin_and_requirements(model, monkeypatch):
    state = planner._place_request_state("我在信图，想去总图附近干饭", pending_slot="selection",
        start={"type": "poi", "name": "信息学部图书馆"},
        search_anchor={"type": "poi", "name": "总图"}, candidate_poi_ids=["amap:A123"],
        candidate_places=[place("amap:A123", "测试餐厅")])
    state["semantic_request"] = {"goal": "总图附近用餐", "requirements": ["不要食堂档口"], "budget_per_person": 50}
    model.side_effect = [reply("保留信图出发和总图附近的范围，继续核对适合聚餐的选择。")]
    result = planner.run_agent("换一家适合三个人坐下来聊的", context={"place_request": state})
    assert model.call_count == 1
    messages = str(model.call_args.kwargs["messages"])
    assert "信息学部图书馆" in messages and "总图" in messages and "不要食堂档口" in messages
    assert result["place_request"]["start"] == state["start"]


def test_accepted_card_selection_is_forwarded_to_the_route_tool(model, monkeypatch):
    candidate_id = "amap:B0K3M1Q398"
    state = planner._place_request_state(
        "找餐厅", pending_slot="selection", start={"type": "poi", "name": "信息学部图书馆"},
        candidate_poi_ids=[candidate_id],
        candidate_places=[{"poi_id": candidate_id, "name": "桂园餐厅", "source": "amap_online",
                           "lng": 114.36, "lat": 30.54}],
    )
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda _ctx: ({}, None))
    provider = planner.agent_tools.amap_poi.normalize_poi({
        "id": "B0K3M1Q398", "name": "桂园餐厅", "location": "114.36,30.54",
    })
    monkeypatch.setattr(planner.agent_tools.amap_poi, "lookup_by_id", lambda _poi_id: provider)
    monkeypatch.setattr(planner.agent_tools, "filter_reported_absent_provider_results", lambda rows: rows)
    route = {"start_name": "信息学部图书馆", "end_name": "桂园餐厅", "mode": "walk",
             "route_verification": {"status": "pass"},
             "route_state": {"start": {"type": "poi", "name": "信息学部图书馆"},
                             "travel_mode": "walk", "hard_constraints": {}, "strategy": {"name": "shortest"}}}
    captured = {}
    def execute(name, args, _context):
        captured.update({"name": name, "args": args})
        return route, {"route": route, "route_kind": "direct"}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    model.side_effect = [reply("已按你选的桂园餐厅规划路线。")]

    result = planner.run_agent("去这家", context={"place_request": state, "selected_poi_id": candidate_id})

    assert result["response_kind"] == "route"
    first_tool_names = {tool["function"]["name"] for tool in model.call_args_list[0].kwargs["tools"]}
    assert "resolve_poi" not in first_tool_names
    assert "search_poi_candidates" not in first_tool_names
    assert first_tool_names == {"get_weather", "list_road_conditions"}
    assert model.call_count == 1
    assert "plan_multimodal_route" not in first_tool_names
    assert captured["args"]["end"]["poi_id"] == candidate_id
    assert candidate_id in str(model.call_args_list[0].kwargs["messages"])


def test_changing_a_requirement_invalidates_old_candidate_set(model):
    old = place("amap:A123", "旧候选")
    state = planner._place_request_state("找餐厅", pending_slot="selection",
        candidate_poi_ids=["amap:A123"], candidate_places=[old])
    state["semantic_request"] = {"goal": "找餐厅", "requirements": ["不要食堂档口"]}
    model.side_effect = [
        reply(update_task={"relation": "continue", "operation": "place_search",
                           "requirements": ["适合三人聚餐"]}),
        reply("我会按适合聚餐的条件重新搜索。"),
    ]

    result = planner.run_agent("改成适合三人聚餐", context={"place_request": state})

    assert result["place_request"]["candidate_poi_ids"] == []
    assert result["place_request"]["candidate_places"] == []


def test_switching_from_place_search_to_route_keeps_the_accepted_candidate():
    from agents.dialogue import Dialogue

    candidate = place("amap:SHOP123", "已接受的店铺")
    state = planner._place_request_state("推荐店铺", pending_slot="selection",
        candidate_poi_ids=[candidate["poi_id"]], candidate_places=[candidate])
    state["semantic_request"] = {"operation": "place_search", "goal": "找店铺"}
    dialogue = Dialogue("去这家", context={"place_request": state,
        "selected_poi_id": candidate["poi_id"]}, execute_tool=lambda *_args: {},
        client_factory=lambda: None, trace=lambda *_args: None)

    dialogue.update({"relation": "continue", "operation": "route"})

    assert dialogue.pending["selected_poi_id"] == candidate["poi_id"]
    assert dialogue.pending["candidate_poi_ids"] == [candidate["poi_id"]]


def test_place_search_task_gets_only_relevant_tools_after_llm_interprets_it(model, monkeypatch):
    from agents import tools as spatial_tools

    candidates = [place("amap:SHOP123", "测试火锅")]
    monkeypatch.setattr(spatial_tools, "execute_tool", lambda *_args: (
        {"status": "ok", "candidates": candidates}, {"candidates": candidates}))
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "goal": "推荐火锅店",
                           "keywords": ["火锅"], "limit": 1},
              search_poi_candidates={"keyword": "火锅", "area": "武汉大学"}),
        reply(present_places={"introduction": "找到一条地图候选，价格待核实。",
                              "places": [{"poi_id": "amap:SHOP123", "reason": "地图目录匹配火锅检索词。"}]}),
    ]

    result = planner.run_agent("推荐一家武大校内火锅店")

    assert result["response_kind"] == "candidates"
    first_names = {tool["function"]["name"] for tool in model.call_args_list[0].kwargs["tools"]}
    second_names = {tool["function"]["name"] for tool in model.call_args_list[1].kwargs["tools"]}
    assert len(second_names) == 4
    assert "plan_multimodal_route" in first_names
    assert "present_places" in second_names
    assert "search_web" in second_names
    assert "search_poi_candidates" not in second_names
    assert "plan_multimodal_route" not in second_names


def test_place_search_stops_when_enough_candidates_are_already_available():
    from agents.dialogue import Dialogue

    calls = []
    candidates = [place("amap:SHOP123", "火锅甲"), place("amap:SHOP456", "火锅乙")]
    def execute(name, args, _context):
        calls.append((name, args))
        return {"status": "ok", "count": len(candidates), "candidates": candidates}, {"candidates": candidates}
    dialogue = Dialogue("推荐两家火锅", execute_tool=execute, client_factory=lambda: None,
                        trace=lambda *_args: None)
    dialogue.intent["operation"] = "place_search"
    dialogue.intent["limit"] = 2

    first = dialogue.spatial("search_poi_candidates", {"keyword": "火锅"})
    repeated = dialogue.spatial("search_poi_candidates", {"keyword": "武大火锅"})
    re_resolved = dialogue.spatial("resolve_poi", {"name": "火锅甲"})

    assert len(first["candidates"]) == 2
    assert repeated["error"] == "candidate_search_complete"
    assert re_resolved["error"] == "candidate_search_complete"
    assert len(calls) == 1


def test_model_cannot_create_coordinates_for_search_anchor(model, monkeypatch):
    called = []
    def execute(name, _args, _context):
        called.append(name)
        return {"candidates": []}, {"candidates": []}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    model.side_effect = [
        reply(search_poi_candidates={"keyword": "火锅", "area": "梅园",
                                     "search_anchor": {"type": "coord", "lng": 114.1, "lat": 30.1}}),
        reply("我不能用未核对的地图坐标搜索。"),
    ]

    result = planner.run_agent("找梅园附近火锅")

    assert "search_poi_candidates" not in called
    assert "untrusted_coordinates" in str(model.call_args_list[1].kwargs["messages"])
    assert not result.get("candidates")


def test_presenter_rejects_a_model_invented_poi(model, monkeypatch):
    model.side_effect = [reply(present_places={"introduction": "推荐这里", "places": [{"poi_id": "amap:FAKE", "reason": "好吃"}]}),
                         reply("暂时没有核对到具体门店，可以继续检索。")]
    result = planner.run_agent("推荐火锅店")
    assert model.call_count == 2
    assert not result.get("candidates")
    assert "amap:FAKE" not in result["message"]
    assert "unknown_place" in str(model.call_args.kwargs["messages"])


def test_model_failure_is_explicit_and_does_not_fall_back_to_keyword_recommendations(model):
    model.side_effect = TimeoutError("test")
    result = planner.run_agent("我在总图，想吃饭")
    assert model.call_count == 1
    assert result["execution_status"] == "failed"
    assert result["model_status"] == "unavailable"
    assert not result.get("candidates")


def test_semantic_state_roundtrips_the_existing_task_contract():
    from agents.state_models import PlaceRequestState
    state = planner._place_request_state("找火锅", pending_slot="selection")
    state["semantic_request"] = {"goal": "找火锅", "requirements": ["独立门店"], "budget_per_person": 50}
    assert PlaceRequestState.model_validate(state).model_dump()["semantic_request"] == state["semantic_request"]


def test_response_adapter_preserves_sources_for_cards():
    from api.routes import _agent_response_to_legacy
    sources = [{"source_id": "Sabc123", "title": "门店资料", "url": "https://example.org/shop"}]
    result = _agent_response_to_legacy({"response_kind": "candidates", "candidates": [],
                                        "sources": sources})
    assert result["sources"] == sources


def test_discovery_searches_a_specific_area_without_an_origin(monkeypatch):
    from agents import discovery
    from agents import tools
    candidate = {"id": "amap:A123", "name": "梅园测试火锅", "aliases": [],
                 "coordinates": {"lng": 114.36, "lat": 30.54}, "lon": 114.36, "lat": 30.54,
                 "coordinate_system": "GCJ-02", "source": "amap_online",
                 "subcategory": "restaurant", "campus": "", "verification_status": "provider_only"}
    legacy = {**candidate, "id": "legacy-restaurant", "poi_id": "legacy-restaurant",
              "name": "旧本地商户", "source": "campus_catalog"}
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [legacy])
    monkeypatch.setattr(discovery, "has_source_backed_record", lambda _poi: True)
    monkeypatch.setattr(discovery.amap_poi, "search_text", lambda *args, **_kwargs: {
        "status": "ok", "candidates": [candidate], "query": args[0],
    })

    result, artifact = discovery.search_places_for_dialogue(
        {"keyword": "火锅", "area": "梅园", "subcategories": ["restaurant"]},
        {"_semantic_request": {}},
    )

    assert [row["poi_id"] for row in result["candidates"]] == ["amap:A123"]
    assert result["search_area"] == "梅园"
    assert artifact["candidates"] == result["candidates"]


def test_search_center_can_differ_from_route_origin(monkeypatch):
    from agents import discovery
    from agents import tools
    candidate = {"id": "amap:A123", "name": "梅园在线餐厅", "source": "amap_online",
                 "coordinates": {"lng": 114.36, "lat": 30.54}, "lon": 114.36, "lat": 30.54,
                 "coordinate_system": "GCJ-02", "subcategory": "restaurant"}
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "_endpoint_target_wgs",
                        lambda *_args: (114.36, 30.54, "总图", None))
    monkeypatch.setattr(discovery, "wgs84_to_gcj02", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(discovery.amap_poi, "search_nearby", lambda *_args, **_kwargs: {
        "status": "ok", "candidates": [candidate],
    })
    def reachable(args, _context):
        assert _context["_excluded_candidate_ids"] == ["amap:old"]
        assert args["start"]["name"] == "信息学部图书馆"
        return {"candidates": [{"poi_id": "amap:A123", "name": "梅园在线餐厅"}],
                "count": 1}, {"candidates": [{"poi_id": "amap:A123", "name": "梅园在线餐厅"}]}
    monkeypatch.setattr(tools, "_tool_find_reachable_places", reachable)

    result, _artifact = discovery.search_places_for_dialogue(
        {"keyword": "吃饭", "area": "梅园", "start": {"type": "poi", "name": "信息学部图书馆"},
         "search_anchor": {"type": "poi", "name": "总图"}, "subcategories": ["restaurant"]},
        {"_semantic_request": {"shown_ids": ["amap:old"]}},
    )

    assert result["candidates"][0]["distance_reference_name"] == "信息学部图书馆"


def test_dialogue_bounds_repeated_place_searches():
    from agents.dialogue import Dialogue

    dialogue = Dialogue("找附近餐厅", execute_tool=lambda *_args: pytest.fail("search must be blocked"),
                        client_factory=lambda: None, trace=lambda *_args: None)
    dialogue.place_search_count = 6

    result = dialogue.spatial("search_poi_candidates", {"keyword": "火锅"})

    assert result["error"] == "place_search_limit"


def test_running_search_prefers_the_reviewed_guiyuan_track_over_meiyuan_square():
    from agents.discovery import search_places_for_dialogue

    result, _artifact = search_places_for_dialogue(
        {"area": "桂园", "subcategories": ["field"], "activities": ["running"]},
        {"_semantic_request": {}},
    )

    ids = [row["poi_id"] for row in result["candidates"]]
    assert "poi_085" in ids
    assert "poi_082" not in ids
