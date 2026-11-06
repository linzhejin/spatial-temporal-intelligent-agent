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


def test_run_agent_uses_model_dialogue_when_mode_is_unset(monkeypatch):
    from agents import dialogue

    monkeypatch.delenv("AGENT_DIALOGUE_MODE", raising=False)
    calls = []
    monkeypatch.setattr(dialogue, "run_dialogue", lambda *args, **kwargs: (
        calls.append((args, kwargs)) or {
            "response_kind": "chat", "message": "模型主流程", "turns": 1,
        }
    ))

    result = planner.run_agent("请推荐校园地点")

    assert len(calls) == 1
    assert result["response_kind"] == "chat"
    assert result["message"] == "模型主流程"


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


def test_new_task_commits_distinct_start_and_search_anchor_before_spatial_tools(model, monkeypatch):
    query = "我在信图，想去总图附近干饭"
    candidate = place("amap:ANCHOR123", "总图附近餐厅")
    calls = []

    def execute(name, args, context):
        calls.append((name, args, context))
        return {"status": "ok", "count": 1, "candidates": [candidate]}, {"candidates": [candidate]}

    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda _context: ({}, None))
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    model.side_effect = [
        reply(update_task={
            "relation": "new", "operation": "place_search", "goal": "总图附近用餐",
            "start": {"type": "poi", "name": "信息学部图书馆"},
            "search_anchor": {"type": "poi", "name": "总图"},
            "area": "总图", "keywords": ["餐饮"], "activities": ["meal"],
            "subcategories": ["restaurant", "canteen", "fastfood"], "limit": 1,
        }),
        reply(present_places={"introduction": "以下地点来自本次地图检索。", "places": [
            {"poi_id": candidate["poi_id"], "reason": "位于总图附近的餐饮候选。"},
        ]}),
    ]

    result = planner.run_agent(query)

    first_call = model.call_args_list[0].kwargs
    first_tool_names = {item["function"]["name"] for item in first_call["tools"]}
    assert first_tool_names == {"update_task"}
    assert first_call["tool_choice"] == {
        "type": "function", "function": {"name": "update_task"},
    }
    assert model.call_count == 2
    assert calls[0][0] == "search_poi_candidates"
    assert calls[0][1]["start"]["name"] == "信息学部图书馆"
    assert calls[0][1]["search_anchor"]["name"] == "总图"
    next_call = model.call_args_list[1].kwargs
    next_tool_names = {item["function"]["name"] for item in next_call["tools"]}
    assert next_tool_names == {"present_places"}
    assert next_call["tool_choice"] == {
        "type": "function", "function": {"name": "present_places"},
    }
    assert candidate["name"] in str(next_call["messages"])
    assert result["place_request"]["start"]["name"] == "信息学部图书馆"
    assert result["place_request"]["search_anchor"]["name"] == "总图"
    assert result["candidates"][0]["poi_id"] == candidate["poi_id"]


def test_new_task_cannot_execute_spatial_tool_before_intent_is_saved(model, monkeypatch):
    candidate = place("amap:ANCHOR123", "总图附近餐厅")
    executed = []
    def execute(name, args, context):
        executed.append((name, args, context))
        return {"status": "ok", "count": 1, "candidates": [candidate]}, {"candidates": [candidate]}
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda _context: ({}, None))
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "keywords": ["干饭"],
                           "area": "总图", "limit": 1,
                           "start": {"type": "poi", "name": "信息学部图书馆"},
                           "search_anchor": {"type": "poi", "name": "总图"}},
              search_poi_candidates={"keyword": "干饭", "area": "总图"}),
        reply(present_places={"introduction": "按本次地图检索结果推荐。", "places": [
            {"poi_id": candidate["poi_id"], "reason": "地图检索得到的候选。"},
        ]}),
    ]

    result = planner.run_agent("我在信图，想去总图附近干饭")

    assert result["response_kind"] == "candidates"
    assert len(executed) == 1
    assert executed[0][0] == "search_poi_candidates"
    assert executed[0][1]["start"]["name"] == "信息学部图书馆"
    assert executed[0][1]["search_anchor"]["name"] == "总图"
    assert model.call_count == 2


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
        reply(present_places={"introduction": "找到一个可进一步核对的选择，人均消费暂未核实。",
                              "places": [{"poi_id": "amap:B123", "reason": "地图记录为火锅餐厅，是否独立门店仍需确认。"}]}),
    ]
    result = planner.run_agent(query)
    assert model.call_count == 2
    assert query in str(model.call_args_list[0].kwargs["messages"])
    assert result["turns"] == 2
    assert result["response_kind"] == "candidates"
    assert [p["poi_id"] for p in result["candidates"]] == ["amap:B123"]
    assert "测试火锅甲" not in result["message"]
    assert "测试火锅乙" in result["message"]
    intent = result["place_request"]["semantic_request"]
    assert intent["budget_per_person"] == 50
    assert "不含食堂档口" in intent["requirements"]
    assert tools[0][1]["keyword"] == "火锅"


def test_general_recommendation_does_not_require_origin_before_model(model, monkeypatch):
    candidate = place("amap:GENERAL123", "校内火锅候选")
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_args: (
        {"status": "ok", "count": 1, "candidates": [candidate]}, {"candidates": [candidate]}))
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search",
                           "goal": "推荐校内适合聚餐的火锅店", "area": "武汉大学",
                           "keywords": ["火锅"]}),
        reply(present_places={"introduction": "先按校内目录与地图结果筛选。", "places": [
            {"poi_id": candidate["poi_id"], "reason": "符合本次火锅关键词检索。"},
        ]}),
    ]
    result = planner.run_agent("推荐几家武大校内适合聚餐的火锅店")
    assert model.call_count == 2
    assert result["response_kind"] == "candidates"
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


def test_continuing_a_place_task_preserves_its_web_research_choice():
    from agents.dialogue import Dialogue

    state = planner._place_request_state("推荐梅园餐饮", pending_slot="selection")
    state["semantic_request"] = {"operation": "place_search", "area": "梅园",
                                 "keywords": ["餐饮"], "web_research_requested": True}
    dialogue = Dialogue("换去桂园找餐厅", context={"place_request": state},
                        execute_tool=lambda *_args: {}, client_factory=lambda: None,
                        trace=lambda *_args: None)

    dialogue.update({"relation": "continue", "operation": "place_search", "area": "桂园"})

    assert dialogue.intent["web_research_requested"] is True


def test_next_batch_reuses_unseen_candidates_and_keeps_the_search_anchor(model, monkeypatch):
    from agents.state_models import PlaceRequestState

    places = [place(f"amap:MEAL{index}", f"餐饮候选{index}") for index in range(1, 7)]
    state = planner._place_request_state(
        "我在信图，想去总图附近干饭", pending_slot="selection",
        start={"type": "poi", "name": "信息学部图书馆"},
        search_anchor={"type": "poi", "name": "武汉大学图书馆(总馆)"},
        candidate_poi_ids=[row["poi_id"] for row in places[:2]],
        candidate_places=places[:2],
    )
    state["candidate_pool"] = places
    state["semantic_request"] = {
        "operation": "place_search", "goal": "总图附近吃饭", "area": "总图",
        "keywords": ["餐饮"], "web_research_requested": True,
        "web_page_read_requested": False, "shown_ids": [row["poi_id"] for row in places[:2]],
    }
    PlaceRequestState.model_validate(state)
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda name, *_args: pytest.fail(
        f"next batch must reuse saved candidates, not call {name}"))
    model.side_effect = [
        reply(update_task={"relation": "continue", "operation": "place_search",
                           "candidate_action": "next_batch"}),
        reply(present_places={"introduction": "继续展示尚未出现的候选。", "places": [
            {"poi_id": places[2]["poi_id"], "reason": "来自本轮已核对的地图候选。"},
            {"poi_id": places[3]["poi_id"], "reason": "来自本轮已核对的地图候选。"},
        ]}),
    ]

    result = planner.run_agent("换一批", context={"place_request": state})

    assert result["response_kind"] == "candidates"
    assert [row["poi_id"] for row in result["candidates"]] == ["amap:MEAL3", "amap:MEAL4"]
    assert result["place_request"]["start"] == state["start"]
    assert result["place_request"]["search_anchor"] == state["search_anchor"]
    assert [row["poi_id"] for row in result["place_request"]["candidate_pool"]] == [
        row["poi_id"] for row in places]
    assert model.call_count == 2
    assert {tool["function"]["name"] for tool in model.call_args_list[0].kwargs["tools"]} == {
        "update_task", "ask_user",
    }


def test_ambiguous_place_followup_asks_before_searching(model, monkeypatch):
    state = planner._place_request_state(
        "我在信图，想去总图附近干饭", pending_slot="selection",
        start={"type": "poi", "name": "信息学部图书馆"},
        search_anchor={"type": "poi", "name": "武汉大学图书馆(总馆)"},
    )
    state["semantic_request"] = {
        "operation": "place_search", "goal": "总图附近吃饭", "area": "总图",
        "keywords": ["餐饮"], "web_research_requested": True,
    }
    calls = []
    execute_tool = planner.agent_tools.execute_tool

    def execute_only_ask_user(name, *args):
        calls.append(name)
        if name != "ask_user":
            pytest.fail(f"clarification must precede {name}")
        return execute_tool(name, *args)

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute_only_ask_user)
    model.side_effect = [reply(ask_user={
        "question": "你是想继续找总图附近的餐饮，还是改找别的地点？",
        "options": ["继续找附近餐饮", "换个区域或类型"], "slot": "other",
    })]

    result = planner.run_agent("那个呢？", context={"place_request": state})

    assert result["response_kind"] == "clarify"
    assert result["clarify"]["question"] == "你是想继续找总图附近的餐饮，还是改找别的地点？"
    assert result["place_request"]["start"] == state["start"]
    assert model.call_count == 1
    assert calls == ["ask_user"]
    assert {tool["function"]["name"] for tool in model.call_args_list[0].kwargs["tools"]} == {
        "update_task", "ask_user",
    }


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


def test_changing_search_area_refreshes_before_ignoring_a_stale_search_call(model, monkeypatch):
    old_candidate = place("amap:MEIYUAN123", "梅园旧候选")
    gui_yuan_candidate = place("poi_085", "桂园操场")
    state = planner._place_request_state("去梅园操场跑步", pending_slot="selection",
        candidate_poi_ids=[old_candidate["poi_id"]], candidate_places=[old_candidate])
    state["semantic_request"] = {
        "operation": "place_search", "area": "梅园", "keywords": ["操场"],
        "activities": ["running"], "subcategories": ["field"], "shown_ids": [old_candidate["poi_id"]],
    }
    searches = []

    def execute(name, args, context):
        searches.append((name, args, context["_semantic_request"].get("area")))
        return {"status": "ok", "count": 1, "candidates": [gui_yuan_candidate]}, {
            "candidates": [gui_yuan_candidate],
        }

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    model.side_effect = [
        reply(update_task={"relation": "continue", "operation": "place_search", "area": "桂园",
                           "keywords": ["操场"], "activities": ["running"], "subcategories": ["field"]},
              search_poi_candidates={"keyword": "餐饮", "area": "梅园"}),
        reply(present_places={"introduction": "桂园有匹配的运动场候选。", "places": [
            {"poi_id": gui_yuan_candidate["poi_id"], "reason": "已按桂园和跑步条件重新查询。"},
        ]}),
    ]

    result = planner.run_agent("不去梅园，明确去桂园找操场跑步", context={"place_request": state})

    assert result["response_kind"] == "candidates"
    assert [item[2] for item in searches] == ["桂园"]
    assert searches[0][1]["keyword"] == "操场"
    assert result["candidates"][0]["poi_id"] == "poi_085"


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
                           "keywords": ["火锅"], "limit": 1}),
        reply(present_places={"introduction": "找到一条地图候选，价格待核实。",
                              "places": [{"poi_id": "amap:SHOP123", "reason": "地图目录匹配火锅检索词。"}]}),
    ]

    result = planner.run_agent("推荐一家武大校内火锅店")

    assert result["response_kind"] == "candidates"
    first_names = {tool["function"]["name"] for tool in model.call_args_list[0].kwargs["tools"]}
    second_names = {tool["function"]["name"] for tool in model.call_args_list[1].kwargs["tools"]}
    assert first_names == {"update_task"}
    assert model.call_args_list[0].kwargs["tool_choice"] == {
        "type": "function", "function": {"name": "update_task"},
    }
    assert second_names == {"present_places"}
    assert model.call_args_list[1].kwargs["tool_choice"] == {
        "type": "function", "function": {"name": "present_places"},
    }


def test_requested_web_facts_keep_a_bounded_search_fetch_then_delivery_flow(model, monkeypatch):
    from agents import web_evidence

    candidate = place("amap:WEB123", "地图检索火锅店")
    spatial_calls = []
    web_calls = []
    def execute_map(name, args, _context):
        spatial_calls.append((name, args))
        return {"status": "ok", "count": 1, "candidates": [candidate]}, {"candidates": [candidate]}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute_map)
    def execute_web(name, args, sources, *, original_question=""):
        web_calls.append((name, args, original_question))
        if name == "search_web":
            sources["Sfact123"] = {"source_id": "Sfact123", "title": "门店资料", "url": "https://example.org/shop",
                                   "excerpt": "网页摘要", "fetched": False}
            return {"status": "ok", "results": [{"source_id": "Sfact123", "title": "门店资料"}]}
        sources[args["source_id"]].update({"fetched": True, "content": "页面正文"})
        return {"source_id": args["source_id"], "content": "页面正文"}
    monkeypatch.setattr(web_evidence, "execute", execute_web)
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "keywords": ["火锅"],
                           "area": "武汉大学", "limit": 1, "web_research_requested": True,
                           "web_page_read_requested": True}),
        reply(search_poi_candidates={"candidate_names": ["测试火锅店"], "area": "武汉大学"}),
        reply(present_places={"introduction": "下面几家都是地图核对到的实际地点。门店消费信息参考网页来源。", "places": [
            {"poi_id": candidate["poi_id"], "reason": "地图目录中匹配到的候选记录；消费信息以来源为准。",
             "source_ids": ["Sfact123"]},
        ]}),
    ]

    result = planner.run_agent("推荐武大火锅，并核对人均和营业时间")

    assert result["response_kind"] == "candidates"
    assert model.call_count == 3
    expected = ["update_task", "search_poi_candidates", "present_places"]
    assert [{tool["function"]["name"] for tool in call.kwargs["tools"]} for call in model.call_args_list] == [
        {"update_task"}, {"search_poi_candidates"}, {"present_places"},
    ]
    assert [call.kwargs["tool_choice"]["function"]["name"] for call in model.call_args_list] == expected
    assert len(spatial_calls) == 1
    assert spatial_calls[0][1]["candidate_names"] == ["测试火锅店"]
    assert [call[0] for call in web_calls] == ["search_web", "read_web_page"]
    assert "火锅" in web_calls[0][1]["query"]
    assert "武汉大学" in web_calls[0][1]["query"]
    assert result["candidates"][0]["source_ids"] == ["Sfact123"]
    assert "地图核对到的实际地点" not in result["message"]
    assert "地图目录中匹配到的候选记录" in result["message"]
    assert "未现场核实存在性、入口或营业状态" in result["message"]
    system_prompt = model.call_args_list[0].kwargs["messages"][0]["content"]
    assert "地图返回记录不等于现场存在性核实" in system_prompt
    map_schema = next(tool for tool in model.call_args_list[1].kwargs["tools"]
                      if tool["function"]["name"] == "search_poi_candidates")
    assert "不证明现场存在" in map_schema["function"]["description"]
    assert any("evidence_scope" in message.get("content", "")
               for message in model.call_args_list[2].kwargs["messages"])


def test_general_food_search_uses_web_search_snippets_without_fetching_page(model, monkeypatch):
    from agents import web_evidence

    candidate = place("amap:QUICKMEAL123", "地图匹配餐饮候选")
    web_calls = []

    def execute_map(name, args, _context):
        return {"status": "ok", "count": 1, "candidates": [candidate]}, {"candidates": [candidate]}

    def execute_web(name, args, sources, *, original_question=""):
        web_calls.append(name)
        if name == "search_web":
            sources["Squick123"] = {"source_id": "Squick123", "title": "校园餐饮信息",
                                    "url": "https://example.org/food", "excerpt": "附近餐饮线索",
                                    "fetched": False}
            return {"status": "ok", "results": [{"source_id": "Squick123", "title": "校园餐饮信息"}]}
        pytest.fail("普通附近推荐不应等待网页正文")

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute_map)
    monkeypatch.setattr(web_evidence, "execute", execute_web)
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "keywords": ["餐饮"],
                           "area": "总图", "web_research_requested": True,
                           "web_page_read_requested": False}),
        reply(search_poi_candidates={"candidate_names": ["校园餐饮"], "area": "总图"}),
        reply(present_places={"introduction": "以下是地图匹配的附近候选。", "places": [
            {"poi_id": candidate["poi_id"], "reason": "位于本次查询范围内。"},
        ]}),
    ]

    result = planner.run_agent("我在信图，想去总图附近干饭")

    assert result["response_kind"] == "candidates"
    assert model.call_count == 3
    assert web_calls == ["search_web"]
    assert result["place_request"]["candidate_pool"][0]["poi_id"] == candidate["poi_id"]


def test_web_flow_falls_back_to_grounded_map_search_when_model_skips_map_step(model, monkeypatch):
    from agents import web_evidence

    candidate = place("amap:FALLBACK123", "附近火锅店")
    spatial_calls = []
    def execute_map(name, args, _context):
        spatial_calls.append((name, args))
        return {"status": "ok", "count": 1, "candidates": [candidate]}, {"candidates": [candidate]}
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda _context: ({}, None))
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute_map)

    def execute_web(name, args, sources, *, original_question=""):
        if name == "search_web":
            sources["Sfact987"] = {"source_id": "Sfact987", "title": "火锅资料",
                                   "url": "https://example.org/hotpot", "excerpt": "校园火锅店",
                                   "fetched": False}
            return {"status": "ok", "results": [{"source_id": "Sfact987", "title": "火锅资料"}]}
        sources[args["source_id"]].update({"fetched": True, "content": "火锅店资料正文"})
        return {"source_id": args["source_id"], "content": "火锅店资料正文"}
    monkeypatch.setattr(web_evidence, "execute", execute_web)

    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "keywords": ["火锅"],
                           "area": "武汉大学", "web_research_requested": True}),
        # The only available stage tool is search_poi_candidates; this stale
        # task edit must not consume another model turn or alter the request.
        reply(update_task={"relation": "continue", "operation": "place_search", "area": "工学部"}),
        reply(present_places={"introduction": "地图核对到附近候选，网页提供参考。", "places": [
            {"poi_id": candidate["poi_id"], "reason": "名称与位置已由地图核对。",
             "source_ids": ["Sfact987"]},
        ]}),
    ]

    result = planner.run_agent("推荐武大附近火锅店")

    assert result["response_kind"] == "candidates"
    assert result["turns"] == 3
    assert len(spatial_calls) == 1
    assert spatial_calls[0][0] == "search_poi_candidates"
    assert spatial_calls[0][1].get("candidate_names") is None
    assert result["candidates"][0]["source_ids"] == ["Sfact987"]


def test_empty_map_retries_end_with_a_clear_no_results_message(model, monkeypatch):
    executed = []

    def execute(name, _args, _context):
        executed.append(name)
        return {"status": "ok", "count": 0, "candidates": [], "online_status": "ok"}, {"candidates": []}

    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda _context: ({}, None))
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "keywords": ["干饭"],
                           "area": "总图", "limit": 1}),
        reply(search_poi_candidates={"keyword": "餐饮", "area": "总图"}),
    ]

    result = planner.run_agent("推荐总图附近吃饭的地方")

    assert model.call_count == 2
    assert executed == ["search_poi_candidates", "search_poi_candidates"]
    assert "没有找到可核对的地点" in result["message"]
    assert result.get("failed") is not True


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
        reply(update_task={"relation": "new", "operation": "place_search", "keywords": ["火锅"],
                           "area": "梅园", "search_anchor": {"type": "coord", "lng": 114.1, "lat": 30.1}}),
        reply("我不能用未核对的地图坐标搜索。"),
    ]

    result = planner.run_agent("找梅园附近火锅")

    assert "search_poi_candidates" not in called
    assert "untrusted_coordinates" in str(model.call_args_list[1].kwargs["messages"])
    assert not result.get("candidates")


def test_presenter_rejects_a_model_invented_poi(model, monkeypatch):
    candidates = [place("amap:REAL", "已检索门店")]
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_args: (
        {"status": "ok", "count": 1, "candidates": candidates}, {"candidates": candidates}))
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "keywords": ["火锅"], "limit": 1}),
        reply(present_places={"introduction": "推荐这里", "places": [{"poi_id": "amap:FAKE", "reason": "好吃"}]}),
        reply("暂时没有核对到具体门店，可以继续检索。"),
    ]
    result = planner.run_agent("推荐火锅店")
    assert model.call_count == 3
    assert not result.get("candidates")
    assert "amap:FAKE" not in result["message"]
    assert "unknown_place" in str(model.call_args_list[2].kwargs["messages"])


def test_model_failure_keeps_the_request_and_returns_a_recoverable_question(model):
    model.side_effect = TimeoutError("test")
    result = planner.run_agent("我在总图，想吃饭")
    assert model.call_count == 1
    assert result["execution_status"] == "needs_input"
    assert result["response_kind"] == "clarify"
    assert result["needs_input"] is True
    assert result["model_status"] == "unavailable"
    assert not result.get("candidates")
    assert result["clarify"]["options"]
    assert "任务没有完成" not in result["message"]


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
        assert args["start"]["name"] == "总图"
        return {"candidates": [{"poi_id": "amap:A123", "name": "梅园在线餐厅"}],
                "count": 1}, {"candidates": [{"poi_id": "amap:A123", "name": "梅园在线餐厅"}]}
    monkeypatch.setattr(tools, "_tool_find_reachable_places", reachable)

    result, _artifact = discovery.search_places_for_dialogue(
        {"keyword": "吃饭", "area": "梅园", "start": {"type": "poi", "name": "信息学部图书馆"},
         "search_anchor": {"type": "poi", "name": "总图"}, "subcategories": ["restaurant"]},
        {"_semantic_request": {"shown_ids": ["amap:old"]}},
    )

    assert result["candidates"][0]["distance_reference_name"] == "总图"


def test_web_discovered_names_are_map_verified_and_ranked_from_the_search_center(monkeypatch):
    from agents import discovery, tools

    store = {"id": "amap:WEB123", "poi_id": "amap:WEB123", "name": "周麻婆(梅园店)",
             "aliases": [], "coordinates": {"lng": 114.36, "lat": 30.54},
             "lon": 114.36, "lat": 30.54, "coordinate_system": "GCJ-02",
             "source": "amap_online", "subcategory": "restaurant", "campus": "",
             "verification_status": "provider_only"}
    near_calls = []
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "_endpoint_target_wgs", lambda *_args: (114.36, 30.54, "总图", None))
    monkeypatch.setattr(discovery, "wgs84_to_gcj02", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(discovery.amap_poi, "search_nearby", lambda _center, **kwargs: (
        near_calls.append(kwargs) or {"status": "ok", "candidates": []}))
    monkeypatch.setattr(discovery.amap_poi, "lookup", lambda name: store if name == store["name"] else None)

    def reachable(args, context):
        assert args["start"]["name"] == "总图"
        assert args["max_distance_m"] == 1000
        assert [row["poi_id"] for row in context["_candidate_pool"]] == ["amap:WEB123"]
        return {"start_name": "武汉大学图书馆(总馆)", "candidates": [
            {"poi_id": "amap:WEB123", "name": store["name"], "distance_m": 460,
             "network_distance_m": 460},
        ], "count": 1}, {"candidates": []}

    monkeypatch.setattr(tools, "_tool_find_reachable_places", reachable)
    result, _artifact = discovery.search_places_for_dialogue(
        {"area": "武汉大学", "keyword": "食堂 餐厅 吃饭", "candidate_names": [store["name"]],
         "start": {"type": "poi", "name": "信息学部图书馆"},
         "search_anchor": {"type": "poi", "name": "总图"}, "subcategories": ["restaurant"]},
        {"_semantic_request": {"area": "武汉大学", "keywords": ["食堂", "餐厅", "吃饭"],
                               "subcategories": ["restaurant"]}},
    )

    assert near_calls[0]["keywords"] == "餐饮"
    assert result["candidates"][0]["poi_id"] == "amap:WEB123"
    assert result["candidates"][0]["distance_reference_name"] == "武汉大学图书馆(总馆)"


def test_generic_meal_web_candidates_are_not_filtered_as_canteens_only(monkeypatch):
    from agents import discovery, tools

    store = {"id": "amap:WEB456", "poi_id": "amap:WEB456", "name": "周麻婆(梅园店)",
             "aliases": [], "coordinates": {"lng": 114.36, "lat": 30.54},
             "lon": 114.36, "lat": 30.54, "coordinate_system": "GCJ-02",
             "source": "amap_online", "subcategory": "restaurant", "campus": "",
             "verification_status": "provider_only"}
    nearby_calls = []
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "_endpoint_target_wgs", lambda *_args: (114.36, 30.54, "总图", None))
    monkeypatch.setattr(discovery, "wgs84_to_gcj02", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(discovery.amap_poi, "search_nearby", lambda _center, **kwargs: (
        nearby_calls.append(kwargs) or {"status": "ok", "candidates": []}))
    monkeypatch.setattr(discovery.amap_poi, "lookup", lambda name: store if name == store["name"] else None)

    def reachable(args, context):
        assert args["start"]["name"] == "总图"
        assert context["_candidate_pool"] == [store]
        return {"start_name": "总图", "candidates": [{
            "poi_id": "amap:WEB456", "name": store["name"], "distance_m": 460,
            "network_distance_m": 460,
        }], "count": 1}, {"candidates": []}

    monkeypatch.setattr(tools, "_tool_find_reachable_places", reachable)
    result, _artifact = discovery.search_places_for_dialogue(
        {"area": "武汉大学", "keyword": "食堂 餐厅 吃饭", "candidate_names": [store["name"]],
         "search_anchor": {"type": "poi", "name": "总图"}, "subcategories": ["canteen"]},
        {"_semantic_request": {"area": "武汉大学", "keywords": ["食堂", "餐厅", "吃饭"],
                               "subcategories": ["canteen"]}},
    )

    assert nearby_calls[0]["keywords"] == "餐饮"
    assert result["candidates"][0]["poi_id"] == "amap:WEB456"


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
