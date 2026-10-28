from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path

import networkx as nx
import pytest

import agents.planner as planner
import agents.tools as tools
import spatial.poi as poi_module


def test_recommendation_categories_cover_broad_food_and_narrow_requests():
    schema = next(item["function"] for item in tools.TOOL_SCHEMAS
                  if item["function"]["name"] == "search_poi_candidates")
    properties = schema["parameters"]["properties"]
    assert "start" in properties
    assert "subcategories" in properties
    assert "mode" in properties
    assert tools.infer_recommendation_subcategories("我想吃美食") == [
        "canteen", "restaurant", "fastfood",
    ]
    assert tools.infer_recommendation_subcategories("想吃点好吃的") == [
        "canteen", "restaurant", "fastfood",
    ]
    assert tools.infer_recommendation_subcategories("找食堂") == ["canteen"]
    assert tools.infer_recommendation_subcategories("附近有什么食堂好吃") == ["canteen"]
    assert tools.infer_recommendation_subcategories("想喝咖啡") == ["coffee"]
    assert tools.infer_recommendation_subcategories("附近有没有奶茶") == ["tea_drink"]
    assert tools.infer_recommendation_subcategories("我想喝咖啡和奶茶") == ["coffee", "tea_drink"]
    assert tools.infer_recommendation_subcategories("想喝一杯果汁") == ["tea_drink"]
    assert tools.infer_recommendation_subcategories("找个自习的地方") == ["library", "study_other"]
    assert tools.infer_recommendation_subcategories("我想找个安静的地方看书") == [
        "library", "study_other",
    ]
    assert tools.infer_recommendation_subcategories("我想买点零食") == ["supermarket"]
    assert tools.infer_recommendation_subcategories("想运动") is None
    assert tools.resolve_activity_query("想运动")["options"] == ["跑步", "打球", "健身", "其他运动"]


def test_location_ranked_candidates_search_full_categories_before_limiting(monkeypatch):
    graph = nx.MultiDiGraph()
    for node in range(13):
        graph.add_node(node, x=114.36 + node * 0.0001, y=30.53)
    for node in range(1, 13):
        graph.add_edge(0, node, length=float((13 - node) * 100))
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    categories_called = []

    canteens = []
    fastfood = []
    for node in range(1, 13):
        poi = {
            "id": f"poi-{node}", "name": f"餐饮点{node}", "type": "dining",
            "subcategory": "canteen" if node % 2 else "fastfood",
            "coordinates": {"lng": 114.36 + node * 0.0001, "lat": 30.53},
            "lat": 30.53, "lon": 114.36 + node * 0.0001,
            "source_refs": [f"source-{node}"],
        }
        (canteens if node % 2 else fastfood).append(poi)

    def category_search(subcategory=None, limit=None, **_kwargs):
        categories_called.append((subcategory, limit))
        # Deliberately return catalog order opposite to proximity order.
        return list(reversed(canteens if subcategory == "canteen" else fastfood))

    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (0, "信息学部起点", None))
    monkeypatch.setattr(tools, "search_by_category", category_search)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "地图起点", "lng": 114.36, "lat": 30.53},
        "subcategories": ["canteen", "fastfood"],
        "mode": "walk",
        "max_distance_m": 5000,
        "limit": 4,
    })

    assert {category for category, _limit in categories_called} == {"canteen", "fastfood"}
    assert all(limit >= 12 for _category, limit in categories_called)
    candidates = result["candidates"]
    assert [item["name"] for item in candidates] == ["餐饮点12", "餐饮点11", "餐饮点10", "餐饮点9"]
    assert [item["network_distance_m"] for item in candidates] == [100, 200, 300, 400]
    assert [item["distance_m"] for item in candidates] == [100, 200, 300, 400]
    assert candidates[0]["source_refs"] == ["source-12"]
    assert candidates[0]["access_link_verified"] is False
    assert artifact["candidates"] == candidates


def test_fuzzy_recommendation_without_origin_asks_instead_of_returning_global_favorites():
    client = FakeClient([_fake_response(tool_calls=[_fake_tool_call(
        "search_poi_candidates", {"subcategories": ["canteen", "restaurant", "fastfood"]}
    )]), _fake_response(content="找到热门食堂。")])
    with patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", return_value=(
             {"candidates": [{"name": "热门食堂"}]},
             {"candidates": [{"name": "热门食堂"}]},
         )):
        result = planner.run_agent("我想吃美食")

    assert result["response_kind"] == "clarify"
    assert "起点" in result["clarify"]["question"]
    assert result["candidates"] is None


def test_map_selected_origin_is_passed_to_fuzzy_recommendation_tool():
    selected_start = {"lng": 114.36, "lat": 30.53, "name": "地图起点"}
    candidate = {
        "name": "信息学部三食堂", "subcategory": "canteen",
        "distance_m": 340, "network_distance_m": 340,
        "access_link_verified": False,
    }
    client = FakeClient([_fake_response(tool_calls=[_fake_tool_call(
        "search_poi_candidates", {"subcategory": "canteen"}
    )]), _fake_response(content="找到附近食堂。")])
    with patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": [candidate]}, {"candidates": [candidate]},
    )) as execute:
        result = planner.run_agent("找食堂", coord_start=selected_start)

    assert result["response_kind"] == "candidates"
    assert result["candidates"] == [candidate]
    assert "340" in result["message"]
    name, args = execute.call_args.args[:2]
    assert name == "search_poi_candidates"
    assert args["start"] == {
        "name": "地图起点", "type": "coord", "lng": 114.36, "lat": 30.53,
    }
    assert args["subcategories"] == ["canteen"]


def test_explicit_nearest_route_goes_to_nearest_network_ranked_candidate():
    selected_start = {"lng": 114.36, "lat": 30.53, "name": "地图起点"}
    candidate = {"name": "附近食堂", "distance_m": 210, "subcategory": "canteen"}
    route = {"start_name": "地图起点", "end_name": "附近食堂", "mode": "walk",
             "recommended": [[114.36, 30.53]], "recommended_length_m": 210}
    with patch.object(planner.agent_tools, "execute_tool", side_effect=[
        ({"start_name": "地图起点", "candidates": [candidate]}, {"candidates": [candidate]}),
        (route, {"route": route, "route_kind": "direct"}),
    ]) as execute:
        result = planner.run_agent("带我去最近的食堂", coord_start=selected_start)
    assert result["response_kind"] == "route"
    assert [call.args[0] for call in execute.call_args_list] == [
        "search_poi_candidates", "plan_route",
    ]
    route_args = execute.call_args_list[1].args[1]
    assert route_args["start"]["name"] == "地图起点"
    assert route_args["end"] == {"type": "poi", "name": "附近食堂"}


def test_nearest_route_keeps_requested_shortlist_in_route_response():
    from api.routes import _agent_response_to_legacy

    selected_start = {"lng": 114.36, "lat": 30.53, "name": "地图起点"}
    candidates = [
        {"poi_id": "canteen_a", "name": "附近食堂甲", "distance_m": 210},
        {"poi_id": "canteen_b", "name": "附近食堂乙", "distance_m": 340},
        {"poi_id": "canteen_c", "name": "附近食堂丙", "distance_m": 510},
    ]
    route = {"start_name": "地图起点", "end_name": "附近食堂甲",
             "mode": "walk", "recommended": [[114.36, 30.53]],
             "recommended_length_m": 210}
    with patch.object(planner.agent_tools, "execute_tool", side_effect=[
        ({"candidates": candidates}, {"candidates": candidates}),
        (route, {"route": route, "route_kind": "direct"}),
    ]):
        response = planner.run_agent(
            "推荐附近3个食堂，并直接给出到其中最近一个的完整步行最短路线",
            coord_start=selected_start,
        )

    assert response["response_kind"] == "route"
    assert [item["poi_id"] for item in response["recommendation_candidates"]] == [
        "canteen_a", "canteen_b", "canteen_c",
    ]
    public = _agent_response_to_legacy(response)
    assert [item["poi_id"] for item in public["recommendation_candidates"]] == [
        "canteen_a", "canteen_b", "canteen_c",
    ]


def test_explicit_nearest_route_reports_routing_failure_instead_of_shortlist():
    selected_start = {"lng": 114.36, "lat": 30.53, "name": "地图起点"}
    candidate = {"name": "附近食堂", "distance_m": 210, "subcategory": "canteen"}
    with patch.object(planner.agent_tools, "execute_tool", side_effect=[
        ({"start_name": "地图起点", "candidates": [candidate]}, {"candidates": [candidate]}),
        ({"error": "route_unavailable", "message": "当前道路无法连通。"}, None),
    ]):
        result = planner.run_agent("带我去最近的食堂", coord_start=selected_start)

    assert result["response_kind"] == "chat"
    assert result["message"] == "当前道路无法连通。"
    assert result["route"] is None
    assert result["candidates"] is None


def test_named_current_location_is_used_for_nearby_recommendation():
    candidate = {"name": "工学部食堂", "distance_m": 210, "subcategory_label": "食堂"}
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"start_name": "信息学部", "candidates": [candidate]},
        {"candidates": [candidate]},
    )) as execute:
        result = planner.run_agent("我在信息学部附近，想吃美食")
    assert result["response_kind"] == "candidates"
    args = execute.call_args.args[1]
    assert args["start"] == {"type": "poi", "name": "信息学部"}
    assert args["subcategories"] == ["canteen", "restaurant", "fastfood"]
    assert candidate["recommendation_start_name"] == "信息学部"


@pytest.mark.parametrize(("query", "expected_limit"), [
    ("我在信息学部，推荐2个食堂", 2),
    ("我在信息学部，推荐三家咖啡店", 3),
])
def test_nearby_recommendation_honors_explicit_candidate_count(query, expected_limit):
    candidate = {"name": "附近地点", "distance_m": 210, "subcategory_label": "食堂"}
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"start_name": "信息学部", "candidates": [candidate]},
        {"candidates": [candidate]},
    )) as execute:
        result = planner.run_agent(query)
    assert result["response_kind"] == "candidates"
    assert execute.call_args.args[1]["limit"] == expected_limit


def test_recommendation_count_defaults_and_ignores_unrelated_building_number():
    assert planner._requested_recommendation_limit("找食堂", catalog=False) == 4
    assert planner._requested_recommendation_limit("学校有哪些食堂", catalog=True) == 20
    assert planner._requested_recommendation_limit(
        "从信息学部3号楼出发，推荐两家食堂", catalog=False,
    ) == 2


def test_frozen_development_recommendation_prompts_use_requested_limit():
    packet_path = (Path(__file__).resolve().parents[1] /
                   "experiments/datasets/controlled_recommendation_pilot_v2/"
                   "scenarios_recommendation_dev_v2.json")
    cases = json.loads(packet_path.read_text(encoding="utf-8"))["cases"]
    pois_path = Path(__file__).resolve().parents[1] / "data/pois.json"
    poi_names = {poi["id"]: poi["name"] for poi in
                 json.loads(pois_path.read_text(encoding="utf-8"))["pois"]}
    candidate = {"id": "test-place", "name": "候选地点", "distance_m": 210}
    for case in cases:
        prompt = case["messages"][0]["content"]
        requested = case["gold_derivation"]["parameters"]["limit"]
        with patch.object(planner.agent_tools, "execute_tool", return_value=(
            {"start_name": "校内起点", "candidates": [candidate]},
            {"candidates": [candidate]},
        )) as execute:
            result = planner.run_agent(prompt)
        assert result["response_kind"] == "candidates", case["id"]
        assert execute.call_args.args[0] == "search_poi_candidates", case["id"]
        assert execute.call_args.args[1]["limit"] == requested, case["id"]
        assert execute.call_args.args[1]["start"]["name"] == poi_names[
            case["gold_derivation"]["parameters"]["start_poi_id"]
        ], case["id"]


def test_distance_ordering_language_does_not_request_directions():
    assert not tools.query_requests_route(
        "从信息学部出发，按路网距离由近到远推荐2个食堂。"
    )
    assert tools.query_requests_route("从信息学部到桂园食堂，怎么走？")


def test_explicit_text_origin_takes_precedence_over_passed_coordinates():
    candidate = {"name": "信息学部食堂", "distance_m": 210, "subcategory_label": "食堂"}
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"start_name": "武汉大学信息学部图书馆", "candidates": [candidate]},
        {"candidates": [candidate]},
    )) as execute:
        result = planner.run_agent(
            "我在武汉大学信息学部图书馆附近，想吃美食",
            coord_start={"lng": 114.37, "lat": 30.54, "name": "自动定位"},
        )
    assert result["response_kind"] == "candidates"
    assert execute.call_args.args[1]["start"] == {
        "type": "poi", "name": "武汉大学信息学部图书馆"}


def test_place_answer_continues_pending_recommendation_with_that_place_as_origin():
    candidate = {"name": "工学部食堂", "distance_m": 210, "subcategory_label": "食堂"}
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"start_name": "工学部", "candidates": [candidate]},
        {"candidates": [candidate]},
    )) as execute:
        result = planner.run_agent("工学部", context={"active_task_request": "我想吃美食"})
    assert result["response_kind"] == "candidates"
    assert execute.call_args.args[1]["start"] == {"type": "poi", "name": "工学部"}
    assert candidate["recommendation_start_name"] == "工学部"


def test_location_recommendation_network_failure_does_not_fall_back_to_catalog(monkeypatch):
    monkeypatch.setattr(tools, "_ensure_graph", lambda: None)
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: pytest.fail("不得降级为全局热度检索"))

    result, artifact = tools.execute_tool("search_poi_candidates", {
        "start": {"type": "coord", "name": "地图起点", "lng": 114.36, "lat": 30.53},
        "subcategories": ["canteen", "restaurant", "fastfood"],
        "limit": 4,
    })

    assert result["error"] == "network_unavailable"
    assert artifact is None


def test_same_names_at_different_locations_are_not_collapsed(monkeypatch):
    places = [
        {"id": "west", "name": "三食堂", "type": "dining", "subcategory": "canteen",
         "coordinates": {"lng": 114.36, "lat": 30.53}, "lat": 30.53, "lon": 114.36},
        {"id": "east", "name": "三食堂", "type": "dining", "subcategory": "canteen",
         "coordinates": {"lng": 114.37, "lat": 30.54}, "lat": 30.54, "lon": 114.37},
    ]
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: places)

    result = tools._search_poi_categories(["canteen"], None, "", None, True)

    assert [poi["id"] for poi in result] == ["west", "east"]


def test_aliases_of_the_same_nearby_venue_do_not_duplicate_candidates(monkeypatch):
    places = [
        {"id": "primary", "name": "星湖餐饮点", "aliases": ["星湖食堂窗口"],
         "type": "dining", "subcategory": "canteen",
         "coordinates": {"lng": 114.36, "lat": 30.53}, "lat": 30.53, "lon": 114.36},
        {"id": "alias", "name": "星湖食堂窗口", "aliases": ["星湖餐饮点"],
         "type": "dining", "subcategory": "canteen",
         "coordinates": {"lng": 114.36001, "lat": 30.53001}, "lat": 30.53001, "lon": 114.36001},
    ]
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: places)

    result = tools._search_poi_categories(["canteen"], None, "", None, True)

    assert [poi["id"] for poi in result] == ["primary"]


def test_keyword_category_search_preserves_all_matches_before_distance_ranking(monkeypatch):
    places = [
        {"id": f"poi-{index}", "name": f"食堂候选{index}", "type": "dining",
         "subcategory": "canteen", "coordinates": {"lng": 114.36, "lat": 30.53}}
        for index in range(70)
    ]
    calls = []

    def search(keyword, **kwargs):
        calls.append((keyword, kwargs.get("limit")))
        return places

    monkeypatch.setattr(tools, "search_pois", search)

    result = tools._search_poi_categories(["canteen"], None, "食堂", None, True)

    assert calls == [("食堂", 2000)]
    assert len(result) == 70


def test_candidate_order_changes_with_origin(monkeypatch):
    graph = nx.MultiDiGraph()
    positions = {
        "信息学部": (1, 114.3600), "工学部": (2, 114.3700),
        "西侧食堂": (3, 114.3610), "东侧食堂": (4, 114.3690),
    }
    for name, (node, lng) in positions.items():
        graph.add_node(node, x=lng, y=30.53)
    for start, west, east in ((1, 100, 700), (2, 700, 100)):
        graph.add_edge(start, 3, length=west)
        graph.add_edge(start, 4, length=east)
    places = [
        {"id": "west", "name": "西侧食堂", "type": "dining", "subcategory": "canteen",
         "coordinates": {"lng": 114.3610, "lat": 30.53}, "lat": 30.53, "lon": 114.3610},
        {"id": "east", "name": "东侧食堂", "type": "dining", "subcategory": "canteen",
         "coordinates": {"lng": 114.3690, "lat": 30.53}, "lat": 30.53, "lon": 114.3690},
    ]
    mode_index = SimpleNamespace(graph=graph, mode_penalty={})
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: mode_index))
    monkeypatch.setattr(tools, "_resolve_endpoint",
                        lambda ref, _graph: (positions[ref["name"]][0], ref["name"], None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence",
                        lambda *_args: {"snap_distance_m": 0.0})
    monkeypatch.setattr(tools, "search_by_category", lambda **_kwargs: places)
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])

    def first(origin):
        result, _artifact = tools.execute_tool("search_poi_candidates", {
            "start": {"type": "poi", "name": origin},
            "subcategory": "canteen", "max_distance_m": 2000, "limit": 2,
        })
        return result["candidates"][0]["name"]

    assert first("信息学部") == "西侧食堂"
    assert first("工学部") == "东侧食堂"


def test_catalog_question_does_not_require_current_origin():
    candidates = [{"poi_id": "poi_307", "name": "星湖园食堂",
                   "subcategory": "canteen", "activity_labels": ["食堂"]}]
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": candidates, "count": len(candidates)}, {"candidates": candidates},
    )) as execute:
        result = planner.run_agent("武汉大学有哪些食堂")
    assert result["response_kind"] == "candidates"
    assert result["candidates"] == candidates
    assert "星湖园食堂" in result["candidates"][0]["name"]
    assert "start" not in execute.call_args.args[1]
    assert tools.is_poi_catalog_query("武汉大学有哪些食堂") is True
    assert tools.is_poi_catalog_query("我想吃美食") is False


def test_large_catalog_reports_total_and_truncation_instead_of_claiming_all_results():
    candidates = [{"poi_id": f"poi-{index}", "name": f"食堂{index}"} for index in range(20)]
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": candidates, "count": 20, "total_count": 25, "has_more": True},
        {"candidates": candidates},
    )):
        result = planner.run_agent("武汉大学有哪些食堂")

    assert "25" in result["message"]
    assert "先列出 20 个" in result["message"]


def test_catalog_summary_does_not_claim_campus_grouping_when_candidates_are_not_grouped():
    candidates = [
        {"poi_id": "a", "name": "文理食堂", "campus": "文理学部"},
        {"poi_id": "b", "name": "工学部食堂", "campus": "工学部"},
    ]
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": candidates, "count": 2, "total_count": 2},
        {"candidates": candidates},
    )):
        result = planner.run_agent("武汉大学有哪些食堂")

    assert "按学部列出" not in result["message"]


def test_explicit_poi_destination_is_not_rewritten_as_a_nearby_recommendation():
    client = FakeClient([_fake_response(content="正在确认目的地。")])
    with patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner.agent_tools, "execute_tool") as execute:
        result = planner.run_agent("带我去星湖园食堂")
    assert result["response_kind"] == "clarify"
    assert result["place_request"]["pending_slot"] == "start"
    assert result["place_request"]["selected_poi_id"] == "poi_307"
    execute.assert_not_called()


def test_candidate_route_continuation_keeps_the_text_origin_in_the_route_request():
    client = FakeClient([_fake_response(tool_calls=[_fake_tool_call("plan_route", {
        "start": {"name": "工学部", "type": "poi"},
        "end": {"name": "工学部清真食堂", "type": "poi"},
    })])])
    route = {"recommended": [[114.36, 30.54]], "start_name": "工学部",
             "end_name": "工学部清真食堂", "mode": "walk"}
    with patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.config, "DEEPSEEK_API_KEY", "test-key"), \
         patch.object(planner.agent_tools, "execute_tool",
                      return_value=(route, {"route": route})) as execute:
        result = planner.run_agent("从工学部到工学部清真食堂")

    assert result["response_kind"] == "route"
    args = execute.call_args.args[1]
    assert args["start"] == {"name": "工学部", "type": "poi"}
    assert args["end"] == {"name": "工学部清真食堂", "type": "poi"}


def _fake_msg(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    msg.model_dump = lambda **_kwargs: {
        "role": "assistant",
        **({"content": content} if content else {}),
        **({"tool_calls": [
            {"id": call.id, "type": "function", "function": {
                "name": call.function.name, "arguments": call.function.arguments,
            }} for call in (tool_calls or [])
        ]} if tool_calls else {}),
    }
    return msg


def _fake_response(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=_fake_msg(content, tool_calls))])


def _fake_tool_call(name, args):
    import json

    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )


class FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **_kwargs):
        return self.scripted.pop(0)


def test_real_poi_taxonomy_categorizes_dining_types_correctly():
    pois = poi_module.load_pois()
    by_id = {poi["id"]: poi for poi in pois}

    assert len(pois) >= 440
    assert (by_id["poi_307"]["type"], by_id["poi_307"]["subcategory"]) == ("dining", "canteen")
    assert (by_id["poi_311"]["type"], by_id["poi_311"]["subcategory"]) == ("dining", "tea_drink")
    assert (by_id["poi_365"]["type"], by_id["poi_365"]["subcategory"]) == ("dining", "tea_drink")
    assert (by_id["poi_405"]["type"], by_id["poi_405"]["subcategory"]) == ("dining", "coffee")
    assert by_id["poi_042"]["subcategory"] == "laboratory"
    assert by_id["poi_237"]["subcategory"] == "building"
    assert by_id["poi_258"]["subcategory"] == "sports_field"
    assert by_id["poi_428"]["subcategory"] == "post"
    assert by_id["poi_154"]["subcategory"] == "park"
    valid_subcategories = {
        "dining": {"canteen", "restaurant", "fastfood", "coffee", "tea_drink"},
        "study": {"college", "classroom", "laboratory", "library", "study_other", "building", "culture", "museum"},
        "sports": {"field", "sports_field", "gym", "court", "pool", "sports_centre", "sports_other"},
        "dorm": {"dormitory"}, "gate": {"gate"},
        "scenery": {"landmark", "lake", "sakura", "park", "hill", "square", "pavilion"},
        "area": {"area"},
        "service": {"service", "activity_center", "hospital", "supermarket", "bank", "post"},
    }
    assert all(poi["subcategory"] in valid_subcategories[poi["type"]] for poi in pois)
    assert all(poi["subcategory"] in tools._SUBCATEGORY_LABELS for poi in pois)
    assert len({poi["id"] for poi in pois}) == len(pois)
    assert len(pois) >= 440
