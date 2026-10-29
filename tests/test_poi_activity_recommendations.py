from unittest.mock import patch

import pytest

from agents import planner, tools
from agents.route_state import build_route_state
from agents.state_models import PlaceRequestState
from spatial.poi import _flatten_poi
from spatial.poi_taxonomy import resolve_activity_query


def _test_route_state(args):
    from agents.route_state import current_road_condition_version
    return build_route_state(route_kind="direct", original_query="测试路线",
        start=args["start"], end=args["end"], travel_mode=args.get("mode", "walk"),
        hard_constraints=args.get("constraints") or {}, strategy={"name": args.get("strategy") or "shortest"},
        data_version="test-data", road_condition_version=current_road_condition_version([]))


@pytest.mark.parametrize("query,expected_slot", [
    ("我在工学部排球场附近，找个运动的地方。", "needs_clarification"),
    ("我在信息学部游泳池附近，找个运动的地方。", "needs_clarification"),
    ("我从桂园一食堂出发，想走平坦的路。", None),
    ("我在篮球场附近想喝咖啡", "resolved"),
    ("从武汉大学文理学部第4教学楼出发去打球，别走台阶。", "needs_clarification"),
])
def test_activity_intent_ignores_poi_category_in_explicit_origin(query, expected_slot):
    result = resolve_activity_query(query)
    assert (result or {}).get("status") == expected_slot
    if expected_slot == "needs_clarification":
        assert result["question"] == ("想打什么球？" if "打球" in query else "想做哪种运动？")
    if "咖啡" in query:
        assert result["activities"] == ["coffee"]


def test_compound_nearby_activity_waits_for_choice_before_named_final_stop():
    query = "从武汉大学信息学部出发，先找附近能打篮球的地方，再到星湖园食堂。"
    with patch.object(planner, "_make_client", side_effect=AssertionError("must not skip selection")):
        result = planner.run_agent(query)
    assert result["response_kind"] == "candidates"
    assert result["route"] is None
    assert result["place_request"]["pending_slot"] == "selection"
    assert result["place_request"]["follow_up_requests"][0]["destination_poi_id"] == "poi_307"
    assert all("basketball" in item.get("activities", []) for item in result["candidates"])

    selected = result["candidates"][0]
    with patch.object(planner, "_make_client", side_effect=AssertionError("must not drop task")):
        completed = planner.run_agent(
            "就选这个篮球场，再去星湖园食堂。",
            context={"place_request": result["place_request"],
                     "selected_poi_id": selected["poi_id"]},
        )
    assert completed["response_kind"] == "route"
    assert completed["route_kind"] == "via"
    assert [leg["end_name"] for leg in completed["route"]["legs"]] == [
        selected["name"], "星湖园食堂",
    ]
    assert selected["name"] in completed["message"]


def test_two_activity_stops_remain_before_a_named_final_stop(monkeypatch):
    basketball = {"poi_id": "poi_206", "name": "工学部篮球场",
                  "activities": ["basketball"], "distance_m": 120}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_args: (
        {"candidates": [basketball], "start_name": "工学部"},
        {"candidates": [dict(basketball)]},
    ))
    with patch.object(planner, "_make_client", side_effect=AssertionError("unexpected LLM")):
        first = planner.run_agent(
            "从工学部出发，先打篮球，再吃饭，最后到武汉大学文理学部教3楼。",
        )
    followups = first["place_request"]["follow_up_requests"]
    assert first["response_kind"] == "candidates"
    assert len(followups) == 2
    assert followups[0]["activities"] == ["meal"]
    assert followups[1]["destination_poi_id"]


def test_custom_weights_do_not_make_saved_place_request_invalid():
    saved = planner._place_request_state(
        "从工学部出发，先找篮球场，再吃饭，距离坡度风景 0.5:0.4:0.1",
    )
    assert PlaceRequestState.model_validate(saved).strategy is None


def test_less_slope_phrase_keeps_flat_strategy_for_place_task():
    assert planner._explicit_place_strategy("少走坡") == "flat"
    saved = planner._place_request_state("从工学部出发先找篮球场，再吃饭，少走坡")
    assert PlaceRequestState.model_validate(saved).strategy == "flat"


def test_constraint_only_followup_keeps_preference_and_explicit_change_wins():
    saved = planner._place_request_state("从工学部出发，找食堂，平坦优先")
    constrained = planner._apply_place_request_correction(saved, "就去桂园一食堂，不走台阶")
    changed = planner._apply_place_request_correction(saved, "改成风景优先，就去桂园一食堂")
    assert constrained["strategy"] == "flat"
    assert constrained["constraints"]["avoid_steps"] is True
    assert changed["strategy"] == "scenery"


@pytest.mark.parametrize("query", [
    "我从武汉大学教1楼出发，帮我规划步行路线。",
    "我从桂园一食堂出发，想走平坦的路。",
    "我从桂园一食堂出发想走平坦的路。",
])
def test_named_origin_without_destination_asks_before_model(query):
    with patch.object(planner, "_make_client", side_effect=AssertionError("unexpected LLM call")):
        response = planner.run_agent(query)
    assert response["response_kind"] == "clarify"
    assert response["place_request"]["pending_slot"] == "destination"
    assert response["place_request"]["start"]["type"] == "poi"
    assert response["clarify"]["question"] == "你想去哪个地点？目的地明确后，我会从你说的起点规划路线。"


def test_flat_preference_survives_destination_answer():
    first = planner.run_agent("我从桂园一食堂出发，想走平坦的路。")
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"error": "route unavailable"}, {},
    )) as execute:
        planner.run_agent("卓尔体育馆", context={"place_request": first["place_request"]})
    name, args = execute.call_args.args[:2]
    assert name == "plan_route"
    assert args["start"]["name"] == "桂园一食堂"
    assert args["strategy"] == "flat"


@pytest.mark.parametrize("route_query,weights", [
    ("我从桂园一食堂出发，想走平坦的路。",
     {"distance": 0.5, "slope": 0.4, "scenery": 0.1}),
    ("我从桂园一食堂出发，风景优先。",
     {"distance": 0.5, "slope": 0.1, "scenery": 0.4}),
])
def test_preference_is_applied_by_actual_route_tool_after_destination_answer(route_query, weights):
    first = planner.run_agent(route_query)
    second = planner.run_agent("卓尔体育馆", context={"place_request": first["place_request"]})
    assert first["response_kind"] == "clarify"
    assert second["response_kind"] == "route"
    assert second["route"]["applied_weights"] == weights


def test_deictic_origin_does_not_replace_saved_selection_origin():
    assert tools.recommendation_origin_from_text(
        "吃饭选蔬三时轻食；请从刚才的起点出发，先到选好的篮球场，再到这家吃饭的地方"
    ) is None
    assert tools.recommendation_origin_from_text(
        "从武汉大学工学部出发，找附近食堂"
    ) == "武汉大学工学部"
    assert planner._explicit_start_name("请从刚才的起点出发，经过球场再到食堂") is None
    assert planner._explicit_start_name("请从武汉大学工学部出发，经过球场再到食堂") == "武汉大学工学部"


def test_activity_filter_is_applied_to_the_full_category_result():
    records = [
        {"id": "basketball", "name": "篮球场", "type": "sports", "subcategory": "field",
         "activities": ["basketball"]},
        {"id": "badminton", "name": "羽毛球场", "type": "sports", "subcategory": "field",
         "activities": ["badminton"]},
        {"id": "running", "name": "操场", "type": "sports", "subcategory": "field",
         "activities": ["running"]},
    ]
    calls = []

    def category_search(subcategory=None, limit=None, **_kwargs):
        calls.append((subcategory, limit))
        return records if subcategory == "field" else []

    with patch.object(tools, "search_by_category", side_effect=category_search):
        results = tools._search_poi_categories(
            ["field", "court"], "sports", "", None, True, activities=["basketball"],
        )

    assert calls == [("field", 2000), ("court", 2000)]
    assert [poi["id"] for poi in results] == ["basketball"]


def test_live_master_category_search_returns_basketball_and_canteens():
    basketball = tools._search_poi_categories(
        ["field", "sports_field", "court"], "sports", "", None, True,
        activities=["basketball"],
    )
    canteens = tools._search_poi_categories(
        ["canteen"], "dining", "", None, True, activities=["canteen_meal"],
    )

    assert {poi["id"] for poi in basketball} >= {"poi_083", "poi_206", "poi_262"}
    assert canteens
    assert all("canteen_meal" in poi["activities"] for poi in canteens)
    assert all(poi["subcategory"] == "canteen" for poi in canteens)


def test_badminton_search_includes_sourced_gym_courts_without_other_gyms():
    badminton = tools._search_poi_categories(
        ["field", "sports_field", "court", "gym"], "sports", "", None, True,
        activities=["badminton"],
    )

    ids = {poi["id"] for poi in badminton}
    assert "poi_257" in ids
    assert all("badminton" in poi["activities"] for poi in badminton)
    assert not {"poi_081", "poi_259"} & ids


def test_indoor_sport_search_returns_only_gym_with_matching_reviewed_activity():
    categories = ["gym"]
    gyms = tools._search_poi_categories(
        categories, "sports", "", None, True,
        activities=["badminton", "table_tennis"],
    )

    ids = {poi["id"] for poi in gyms}
    assert {"poi_086", "poi_204"} <= ids
    assert all(set(poi["activities"]) & {"badminton", "table_tennis"} for poi in gyms)
    assert "poi_259" not in ids


def test_generic_meal_matches_all_meal_venues_but_canteen_stays_specific():
    meal = tools._search_poi_categories(
        ["canteen", "restaurant", "fastfood"], "dining", "", None, True,
        activities=["meal"],
    )
    canteens = tools._search_poi_categories(
        ["canteen", "restaurant", "fastfood"], "dining", "", None, True,
        activities=["canteen_meal"],
    )

    assert meal
    assert {poi["subcategory"] for poi in meal} == {"canteen", "restaurant", "fastfood"}
    assert canteens
    assert all(poi["subcategory"] == "canteen" for poi in canteens)


def test_buy_water_from_named_campus_uses_location_ranked_shopping_tool():
    candidate = {
        "poi_id": "poi_231", "name": "珞珈自强超市", "subcategory": "supermarket",
        "activities": ["shopping"], "distance_m": 430,
    }
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": [candidate], "count": 1, "start_name": "工学部"},
        {"candidates": [candidate]},
    )) as execute:
        result = planner.run_agent("我在工学部，想买瓶水")

    assert result["response_kind"] == "candidates"
    name, args = execute.call_args.args[:2]
    assert name == "search_poi_candidates"
    assert args["activities"] == ["shopping"]
    assert args["subcategories"] == ["supermarket"]
    assert args["start"] == {"type": "poi", "name": "工学部"}


def test_flattened_poi_preserves_audited_activity_tags():
    source = {
        "id": "poi_262", "name": "信息学部篮球场", "type": "sports",
        "subcategory": "field", "coordinates": {"lng": 114.4, "lat": 30.5},
        "activities": ["basketball"], "activity_basis": "name_supported: basketball court",
        "activity_review_status": "source_supported", "activity_taxonomy_version": 1,
    }

    flattened = _flatten_poi(source)

    assert flattened["activities"] == ["basketball"]
    assert flattened["activity_basis"] == "name_supported: basketball court"
    assert flattened["activity_review_status"] == "source_supported"
    assert flattened["activity_taxonomy_version"] == 1


def test_ambiguous_ball_request_asks_before_calling_the_model():
    with patch.object(planner, "_make_client", side_effect=AssertionError("LLM should not choose a sport")):
        result = planner.run_agent("我想去打球")

    assert result["response_kind"] == "clarify"
    assert result["clarify"]["options"] == ["篮球", "羽毛球", "乒乓球", "足球", "其他球类"]
    assert result["place_request"]["pending_slot"] == "activity"


def test_sport_answer_that_matches_a_poi_does_not_replace_the_saved_origin():
    original_query = "从信息学部图书馆出发，想去打球"
    pending = planner.run_agent(original_query)
    candidate = {
        "poi_id": "poi_450", "name": "乒乓球馆", "subcategory": "court",
        "activities": ["table_tennis"], "distance_m": 120,
    }

    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": [candidate], "count": 1,
         "start_name": "武汉大学信息学部图书馆"},
        {"candidates": [candidate]},
    )) as execute:
        result = planner.run_agent("乒乓球", context={
            "active_task_request": original_query,
            "place_request": pending["place_request"],
        })

    assert result["response_kind"] == "candidates"
    tool_name, args = execute.call_args.args[:2]
    assert tool_name == "search_poi_candidates"
    assert args["start"] == {
        "type": "poi", "name": "武汉大学信息学部图书馆",
    }


def test_candidate_summary_formats_activity_labels_as_plain_text_without_repeating_poi_name():
    original_query = "从信息学部图书馆出发，想去打乒乓球"
    candidate = {
        "poi_id": "poi_450", "name": "乒乓球馆", "subcategory": "court",
        "activities": ["table_tennis"], "activity_labels": ["乒乓球馆"],
        "distance_m": 274,
    }
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": [candidate], "count": 1,
         "start_name": "武汉大学信息学部图书馆"},
        {"candidates": [candidate]},
    )):
        result = planner.run_agent(original_query)

    assert result["response_kind"] == "candidates"
    assert "['乒乓球馆']" not in result["message"]
    assert "乒乓球馆（乒乓球馆" not in result["message"]
    assert "约 274 米" in result["message"]


def test_basketball_catalog_uses_complete_activity_filtered_master_without_origin():
    basketball = [
        {"poi_id": "poi_083", "name": "文理学部篮球场", "activities": ["basketball"]},
        {"poi_id": "poi_206", "name": "工学部篮球场", "activities": ["basketball"]},
        {"poi_id": "poi_262", "name": "信息学部篮球场", "activities": ["basketball"]},
    ]
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": basketball, "count": 3}, {"candidates": basketball},
    )) as execute:
        result = planner.run_agent("武大有哪些篮球场")

    assert result["response_kind"] == "candidates"
    assert [row["poi_id"] for row in result["candidates"]] == ["poi_083", "poi_206", "poi_262"]
    name, args = execute.call_args.args[:2]
    assert name == "search_poi_candidates"
    assert args["activities"] == ["basketball"]
    assert "start" not in args


def test_basketball_catalog_tool_filters_by_explicit_query_without_model_activity_arg():
    result, _artifact = tools._tool_search_poi_candidates(
        {"subcategories": ["field", "sports_field", "court"]},
        {"query": "武大有哪些篮球场"},
    )

    assert {row["poi_id"] for row in result["candidates"]} == {
        "poi_083", "poi_206", "poi_262",
    }
    assert all("basketball" in row["activities"] for row in result["candidates"])


def test_catalog_tool_reports_total_matches_when_result_limit_truncates(monkeypatch):
    places = [{"id": f"poi-{index}", "name": f"食堂{index}", "subcategory": "canteen"}
              for index in range(25)]
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: places)

    result, _artifact = tools._tool_search_poi_candidates(
        {"subcategories": ["canteen"], "limit": 20},
        {"query": "武汉大学有哪些食堂"},
    )

    assert result["count"] == 20
    assert result["total_count"] == 25
    assert result["has_more"] is True


def test_catalog_tool_applies_inferred_categories_without_model_category_arg(monkeypatch):
    captured = {}
    places = [{"id": "library", "name": "图书馆", "subcategory": "library"}]

    def search(subcategories, poi_type, keyword, season, include_minor, *, activities=None):
        captured.update({"subcategories": subcategories, "activities": activities})
        return places

    monkeypatch.setattr(tools, "_search_poi_categories", search)
    result, _artifact = tools._tool_search_poi_candidates(
        {}, {"query": "武汉大学有哪些学习场所"},
    )

    assert captured == {
        "subcategories": ["library", "study_other"], "activities": ["study"],
    }
    assert [row["poi_id"] for row in result["candidates"]] == ["library"]


def test_study_place_catalog_does_not_include_unverified_open_buildings():
    result, _artifact = tools._tool_search_poi_candidates(
        {}, {"query": "武汉大学有哪些学习场所"},
    )

    assert all("study" in row["activities"] for row in result["candidates"])
    assert not {"poi_020", "poi_024", "poi_184", "poi_187"} & {
        row["poi_id"] for row in result["candidates"]
    }


def test_catalog_query_for_courier_points_uses_package_pickup_activity():
    result, _artifact = tools._tool_search_poi_candidates(
        {}, {"query": "武汉大学有哪些快递点"},
    )

    assert result["total_count"] == 2
    assert all("package_pickup" in row["activities"] for row in result["candidates"])


def test_reachability_tool_keeps_explicit_activity_for_catalog_query(monkeypatch):
    from types import SimpleNamespace
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_node(0, x=114.36, y=30.53)
    captured = {}
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(
        tools, "get_routing_index",
        lambda _graph: SimpleNamespace(for_mode=lambda _mode: SimpleNamespace(graph=graph)),
    )
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (0, "origin", None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_: {"snap_distance_m": 0})

    def categories(subcategories, poi_type, keyword, season, include_minor, *, activities):
        captured.update({"subcategories": subcategories, "activities": activities})
        return []

    monkeypatch.setattr(tools, "_search_poi_categories", categories)
    result, _artifact = tools._tool_find_reachable_places(
        {"start": {"type": "poi", "name": "origin"}, "max_distance_m": 1000,
         "subcategories": ["court"]},
        {"query": "武大有哪些篮球场", "_route_graph_snapshot": graph,
         "_route_conditions_snapshot": []},
    )

    assert result["candidates"] == []
    assert captured == {
        "subcategories": ["field", "sports_field", "court"],
        "activities": ["basketball"],
    }


def test_composite_sport_then_meal_keeps_the_second_request_through_route_selection():
    initial = planner.run_agent("我在工学部，想先打球再吃饭")
    assert initial["response_kind"] == "clarify"
    assert initial["place_request"]["follow_up_requests"] == [{
        "query": "吃饭", "activities": ["meal"],
        "subcategories": ["canteen", "restaurant", "fastfood"],
    }]

    candidate = {
        "poi_id": "poi_206", "name": "工学部篮球场", "subcategory": "field",
        "activities": ["basketball"], "distance_m": 120,
    }
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": [candidate], "count": 1, "start_name": "工学部"},
        {"candidates": [candidate]},
    )):
        selected_sport = planner.run_agent(
            "篮球", context={"place_request": initial["place_request"]},
        )
    assert selected_sport["response_kind"] == "candidates"
    assert selected_sport["place_request"]["follow_up_requests"] == initial["place_request"]["follow_up_requests"]

    route = {"recommended": [[114.36, 30.54]], "start_name": "工学部",
             "end_name": "工学部篮球场", "mode": "walk"}
    with patch.object(planner.agent_tools, "load_pois", return_value=[{
        "id": "poi_206", "name": "工学部篮球场",
    }]), patch.object(planner.agent_tools, "execute_tool", return_value=(
        route, {"route": route},
    )) as execute:
        completed = planner.run_agent("去这个地点", context={
            "place_request": selected_sport["place_request"],
            "selected_poi_id": "poi_206",
        })

    assert completed["response_kind"] == "route"
    assert completed["suggestions"] == [{
        "label": "下一步：找餐饮", "query": "从工学部篮球场出发找吃饭",
    }]
    route_args = next(call.args[1] for call in execute.call_args_list
                      if call.args[0] == "plan_route")
    assert route_args["end"]["poi_id"] == "poi_206"



def test_three_stage_request_preserves_unspecified_dorm_and_steps_constraint():
    basketball = {
        "poi_id": "poi_206", "name": "工学部篮球场", "subcategory": "field",
        "activities": ["basketball"], "distance_m": 120,
    }
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": [basketball], "count": 1, "start_name": "工学部"},
        {"candidates": [basketball]},
    )) as execute:
        result = planner.run_agent("我在工学部，先打篮球，再吃饭，最后回宿舍，不走台阶")

    assert result["response_kind"] == "candidates"
    assert [item["query"] for item in result["place_request"]["follow_up_requests"]] == [
        "吃饭", "回宿舍，不走台阶",
    ]
    assert result["place_request"]["constraints"]["avoid_steps"] is True
    assert execute.call_args.args[1]["constraints"]["avoid_steps"] is True


def test_named_destination_after_meal_selection_returns_the_full_ordered_route(monkeypatch):
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    meal = {"poi_id": "poi_meal", "name": "测试食堂", "activities": ["canteen_meal"],
            "subcategory": "canteen", "distance_m": 150}
    final = {"id": "poi_070", "name": "桂园食堂", "aliases": []}
    points = {
        "武汉大学科技门": [114.36, 30.53],
        "测试食堂": [114.361, 30.53],
        "桂园食堂": [114.362, 30.53],
    }
    route_calls = []

    def execute(name, args, _ctx):
        if name == "search_poi_candidates":
            return {"candidates": [meal], "start_name": "武汉大学科技门"}, {
                "candidates": [dict(meal)],
            }
        assert name == "plan_route"
        route_calls.append(args)
        start_name = args["start"]["name"]
        end_name = args["end"]["name"]
        edge_id = (1, 2, 0) if end_name == "测试食堂" else (2, 3, 0)
        route = {
            "start_name": start_name, "end_name": end_name, "mode": "walk",
            "recommended": [points[start_name], points[end_name]],
            "shortest": [points[start_name], points[end_name]],
            "recommended_edge_ids": [edge_id], "shortest_edge_ids": [edge_id],
            "recommended_length_m": 100.0, "shortest_length_m": 100.0,
            "duration_min": 2.0, "shortest_duration_min": 2.0,
            "steps": [], "pois": [], "timings_ms": {},
            "route_verification": {"status": "pass", "field_truth_verified": False},
            "route_state": _test_route_state(args),
        }
        return route, {"route": route}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": "poi_meal", "name": "测试食堂"}, final,
    ])
    monkeypatch.setattr(planner.agent_tools, "find_poi_ambiguous", lambda name:
                        (final, []) if name == "桂园食堂" else (None, []))

    first = planner.run_agent("从武汉大学科技门先找个附近能吃饭的地方，再去桂园食堂。")
    assert first["response_kind"] == "candidates"
    assert first["candidates"][0]["poi_id"] == "poi_meal"
    completed = planner.run_agent("去这个地点", context={
        "place_request": first["place_request"], "selected_poi_id": "poi_meal",
    })

    assert completed["response_kind"] == "route"
    assert completed["route_kind"] == "via"
    assert completed["route"]["start_name"] == "武汉大学科技门"
    assert completed["route"]["end_name"] == "桂园食堂"
    assert completed["route"]["recommended_edge_ids"] == [(1, 2, 0), (2, 3, 0)]
    assert completed["route"]["route_verification"]["scope"] == "verified_legs_and_checked_join"
    assert [leg["end_name"] for leg in completed["route"]["legs"]] == ["测试食堂", "桂园食堂"]
    assert [call["end"]["name"] for call in route_calls] == ["测试食堂", "桂园食堂"]

    def changed_execute(name, args, context):
        result, artifact = execute(name, args, context)
        if name == "plan_route" and args["end"]["name"] == "桂园食堂":
            artifact["route"]["route_state"]["road_condition_version"] = "changed-roads"
        return result, artifact

    monkeypatch.setattr(planner.agent_tools, "execute_tool", changed_execute)
    stale = planner.run_agent("去这个地点", context={
        "place_request": first["place_request"], "selected_poi_id": "poi_meal",
    })
    assert stale["response_kind"] == "clarify"
    assert stale["route"] is None and stale["needs_input"] is True
    assert stale["place_request"]["completed_legs"]
    assert "版本不同" in stale["message"]


@pytest.mark.parametrize("selection_query,expected_strategy", [
    ("就去桂园一食堂，当前有无道路管制？然后到刚才说的终点；请按顺序给出全程步行路线。", "flat"),
    ("改成风景优先，就去桂园一食堂，然后到刚才说的终点；请按顺序给出全程步行路线。", "scenery"),
])
def test_textual_meal_selection_keeps_named_non_activity_final_stop(
    monkeypatch, selection_query, expected_strategy,
):
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    meal = {"poi_id": "poi_068", "name": "桂园一食堂", "activities": ["canteen_meal"],
            "subcategory": "canteen", "distance_m": 150}
    route_calls = []
    points = {
        "武汉大学文理学部教3楼": [114.36, 30.53],
        "桂园一食堂": [114.361, 30.53],
        "武汉大学梅园操场": [114.362, 30.53],
    }

    def execute(name, args, _context):
        if name == "search_poi_candidates":
            return {"candidates": [meal], "start_name": "武汉大学文理学部教3楼"}, {
                "candidates": [dict(meal)],
            }
        assert name == "plan_route"
        route_calls.append(args)
        start_name = args["start"]["name"]
        end_name = args["end"]["name"]
        edge_id = (1, 2, 0) if end_name == "桂园一食堂" else (2, 3, 0)
        route = {
            "start_name": start_name, "end_name": end_name, "mode": "walk",
            "recommended": [points[start_name], points[end_name]],
            "shortest": [points[start_name], points[end_name]],
            "recommended_edge_ids": [edge_id], "shortest_edge_ids": [edge_id],
            "recommended_length_m": 100.0, "shortest_length_m": 100.0,
            "duration_min": 2.0, "shortest_duration_min": 2.0,
            "steps": [], "pois": [], "timings_ms": {},
            "route_verification": {"status": "pass", "field_truth_verified": False},
            "road_conditions_applied": 1,
            "route_state": _test_route_state(args),
        }
        return route, {"route": route}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    first = planner.run_agent(
        "从武汉大学文理学部教3楼出发，走平坦的路，先找个附近能吃饭的地方，再去武汉大学梅园操场。",
    )
    assert first["response_kind"] == "candidates"
    assert first["place_request"]["follow_up_requests"][-1]["destination_poi_id"] == "poi_082"

    completed = planner.run_agent(
        selection_query,
        context={"place_request": first["place_request"]},
    )
    assert completed["response_kind"] == "route"
    assert completed["route_kind"] == "via"
    assert [leg["end_name"] for leg in completed["route"]["legs"]] == [
        "桂园一食堂", "武汉大学梅园操场",
    ]
    assert [call["end"]["poi_id"] for call in route_calls] == ["poi_068", "poi_082"]
    assert [call.get("strategy") for call in route_calls] == [expected_strategy] * 2
    if "道路管制" in selection_query:
        # The mocked legs have no shared snapshot version, so their global
        # event count cannot safely be presented as the whole itinerary's.
        assert "未取得可核实的路况应用信息" in completed["message"]


def test_textual_candidate_choice_respects_rejection_and_ambiguity():
    candidate_ids = ["poi_068", "poi_069"]
    assert planner._selected_candidate_from_text(
        "不去桂园一食堂，换成桂园二食堂", candidate_ids,
    ) == "poi_069"
    assert planner._selected_candidate_from_text(
        "不去桂园一食堂", candidate_ids,
    ) is None
    assert planner._selected_candidate_from_text(
        "桂园一食堂还是桂园二食堂？", candidate_ids,
    ) is None
    assert planner._selected_candidate_from_text(
        "就去梅园操场", candidate_ids,
    ) is None
    assert planner._selected_candidate_from_text(
        "要去桂园一食堂吗？", candidate_ids,
    ) is None


def test_live_shaped_ordered_legs_join_at_the_same_point_with_undirected_edges():
    shared = {"lng": 114.363465, "lat": 30.53795}
    state = {"data_version": "data-1", "road_condition_version": "roads-1"}
    legs = [
        {"route_state": state, "recommended_edge_ids": [[10, 20, 0]],
         "recommended": [{"lng": 114.363, "lat": 30.537}, shared]},
        {"route_state": state, "recommended_edge_ids": [[20, 10, 0]],
         "recommended": [shared, {"lng": 114.364, "lat": 30.538}]},
    ]

    assert planner._place_route_join_issue(legs) is None


def test_ordered_legs_reject_disconnected_edge_sets_even_with_matching_poi_points():
    shared = {"lng": 114.363465, "lat": 30.53795}
    legs = [
        {"recommended_edge_ids": [[10, 20, 0]], "recommended": [shared]},
        {"recommended_edge_ids": [[30, 40, 0]], "recommended": [shared]},
    ]

    assert planner._place_route_join_issue(legs) is not None


@pytest.mark.parametrize("origin,final_name", [
    ("武汉大学科技门", "桂园食堂"),
    ("武汉大学茶港门", "武汉大学田园食堂"),
    ("玉兰2门", "星湖园食堂"),
])
def test_three_reviewed_food_then_named_destination_cases_keep_final_id(
    monkeypatch, origin, final_name,
):
    candidate = {"poi_id": "poi_meal", "name": "测试食堂",
                 "activities": ["meal"], "distance_m": 100}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_args:
                        ({"candidates": [dict(candidate)], "start_name": origin},
                         {"candidates": [dict(candidate)]}))
    response = planner.run_agent(
        f"从{origin}先找个附近能吃饭的地方，再去{final_name}。",
    )
    expected, alternatives = planner.agent_tools.find_poi_ambiguous(final_name)
    assert expected and not alternatives
    assert response["response_kind"] == "candidates"
    assert response["place_request"]["follow_up_requests"][-1]["destination_poi_id"] == expected["id"]


def test_selected_first_stop_equal_to_named_final_does_not_plan_empty_second_leg(monkeypatch):
    poi = {"id": "poi_070", "name": "桂园食堂"}
    route = {"start_name": "武汉大学科技门", "end_name": "桂园食堂",
             "mode": "walk", "recommended": [[114.36, 30.53], [114.37, 30.53]],
             "recommended_edge_ids": [(1, 2, 0)],
             "route_verification": {"status": "pass"}}
    calls = []

    def execute(name, args, _context):
        calls.append((name, args))
        return route, {"route": route}

    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [poi])
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    response = planner.run_agent("去这个地点", context={
        "selected_poi_id": "poi_070",
        "place_request": {
            "pending_slot": "selection", "start": {"type": "poi", "name": "武汉大学科技门"},
            "candidate_poi_ids": ["poi_070"],
            "follow_up_requests": [{"query": "桂园食堂", "destination_poi_id": "poi_070"}],
        },
    })
    assert response["response_kind"] == "route"
    assert response["needs_input"] is False
    assert len(calls) == 1


def test_map_overview_ball_then_food_clarification_reaches_real_candidates():
    with patch.object(planner, "_make_client", side_effect=AssertionError("unexpected LLM call")):
        first = planner.run_agent("我在总图，先找一个球场然后去吃饭")
        selected = planner.run_agent("篮球", context={
            "active_task_request": "我在总图，先找一个球场然后去吃饭",
            "place_request": first["place_request"],
        })
    assert first["response_kind"] == "clarify"
    assert first["place_request"]["pending_slot"] == "activity"
    assert selected["response_kind"] == "candidates"
    assert selected["candidates"]
    assert all("basketball" in item.get("activities", []) for item in selected["candidates"])
    assert selected["place_request"]["follow_up_requests"][0]["activities"]


def test_third_dorm_stage_survives_two_selected_routes(monkeypatch):
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    basketball = {"poi_id": "poi_206", "name": "工学部篮球场", "activities": ["basketball"],
                  "subcategory": "field", "distance_m": 120}
    canteen = {"poi_id": "poi_meal", "name": "测试食堂", "activities": ["canteen_meal"],
               "subcategory": "canteen", "distance_m": 180}
    points = {
        "工学部": {"lng": 114.36, "lat": 30.53},
        "工学部篮球场": {"lng": 114.361, "lat": 30.53},
        "测试食堂": {"lng": 114.362, "lat": 30.53},
        "武汉大学桂园6舍": {"lng": 114.363, "lat": 30.53},
    }
    route_calls = []

    def execute(name, args, _ctx):
        if name == "search_poi_candidates":
            candidate = basketball if "basketball" in args.get("activities", []) else canteen
            return {"candidates": [candidate], "start_name": args.get("start", {}).get("name")}, {
                "candidates": [dict(candidate)],
            }
        assert name == "plan_route"
        route_calls.append(args)
        start_name = args["start"]["name"]
        end_name = args["end"]["name"]
        edge_id = {"工学部篮球场": (1, 2, 0), "测试食堂": (2, 3, 0),
                   "武汉大学桂园6舍": (3, 4, 0)}[end_name]
        route = {
            "start_name": start_name, "end_name": end_name, "mode": "walk",
            "recommended": [points[start_name], points[end_name]],
            "shortest": [points[start_name], points[end_name]],
            "recommended_edge_ids": [edge_id], "shortest_edge_ids": [edge_id],
            "recommended_length_m": 100.0, "shortest_length_m": 100.0,
            "duration_min": 2.0, "shortest_duration_min": 2.0,
            "steps": [], "pois": [], "timings_ms": {},
            "route_state": _test_route_state(args),
        }
        return route, {"route": route}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": basketball["poi_id"], "name": basketball["name"]},
        {"id": canteen["poi_id"], "name": canteen["name"]},
        {"id": "poi_110", "name": "武汉大学桂园6舍"},
    ])
    initial = planner.run_agent("我在工学部，先打篮球，再吃饭，最后回宿舍，不走台阶，平坦优先")
    first = planner.run_agent("去这个地点", context={
        "place_request": initial["place_request"], "selected_poi_id": basketball["poi_id"],
    })
    second = planner.run_agent("吃饭选测试食堂；请从刚才的起点出发，先到选好的篮球场，再到这家吃饭的地方", context={
        "place_request": first["place_request"], "selected_poi_id": canteen["poi_id"],
    })
    third = planner.run_agent("桂园6舍", context={
        "place_request": second["place_request"],
    })

    assert first["candidates"][0]["poi_id"] == canteen["poi_id"]
    assert second["route_kind"] == "via"
    assert second["place_request"]["pending_slot"] == "destination"
    assert "宿舍" in second["place_request"]["original_query"]
    assert "宿舍" in second["clarify"]["question"]
    assert len(second["route"]["recommended_edge_ids"]) == 2
    assert third["route_kind"] == "via"
    assert third["place_request"] is None
    assert len(third["route"]["recommended_edge_ids"]) == 3
    assert route_calls[1]["start"]["name"] == basketball["name"]
    assert all(call["constraints"]["avoid_steps"] for call in route_calls)
    assert [call.get("strategy") for call in route_calls] == ["flat", "flat", "flat"]


def test_unrecognized_dorm_answer_keeps_the_pending_task(monkeypatch):
    saved = {
        "request_id": "req-dorm", "original_query": "回宿舍",
        "pending_slot": "destination", "start": {"type": "poi", "name": "测试食堂"},
        "constraints": {"avoid_steps": True}, "completed_legs": [{"route_id": "leg-1"}],
    }
    monkeypatch.setattr(planner.agent_tools, "find_poi_ambiguous", lambda _name: (None, []))

    response = planner.run_agent("我住的那栋楼", context={"place_request": saved})

    assert response["response_kind"] == "clarify"
    assert response["place_request"]["request_id"] == "req-dorm"
    assert response["place_request"]["completed_legs"] == [{"route_id": "leg-1"}]
    assert response["place_request"]["constraints"]["avoid_steps"] is True


def test_ambiguous_dorm_answer_uses_stable_candidate_ids(monkeypatch):
    saved = {
        "request_id": "req-dorm", "original_query": "回宿舍",
        "pending_slot": "destination", "start": {"type": "poi", "name": "测试食堂"},
        "constraints": {"avoid_steps": True}, "completed_legs": [{"route_id": "leg-1"}],
    }
    alternatives = [
        {"id": "dorm-a", "name": "桂园6舍", "campus": "文理学部"},
        {"id": "dorm-b", "name": "桂园6舍东区", "campus": "文理学部"},
    ]
    monkeypatch.setattr(planner.agent_tools, "find_poi_ambiguous",
                        lambda _name: (None, alternatives))

    response = planner.run_agent("桂园6舍", context={"place_request": saved})

    assert response["response_kind"] == "candidates"
    assert [item["poi_id"] for item in response["candidates"]] == ["dorm-a", "dorm-b"]
    assert response["place_request"]["candidate_poi_ids"] == ["dorm-a", "dorm-b"]
    assert response["place_request"]["completed_legs"] == [{"route_id": "leg-1"}]


def test_mode_change_during_dorm_clarification_keeps_pending_destination(monkeypatch):
    saved = {
        "request_id": "req-dorm", "original_query": "回宿舍",
        "pending_slot": "destination", "start": {"type": "poi", "name": "测试食堂"},
        "mode": "walk", "constraints": {"avoid_steps": True},
        "completed_legs": [{"route_id": "leg-1"}],
    }
    monkeypatch.setattr(planner.agent_tools, "find_poi_ambiguous", lambda _name: (None, []))

    response = planner.run_agent("改成骑行", context={
        "place_request": saved,
        "previous_route_state": {"route_kind": "direct", "route_id": "previous-route"},
    })

    assert response["response_kind"] == "clarify"
    assert response["place_request"]["pending_slot"] == "destination"
    assert response["place_request"]["mode"] == "bike"
    assert response["place_request"]["constraints"]["avoid_steps"] is True


def test_selected_place_honors_current_turn_origin_mode_and_steps_constraint():
    saved = {
        "request_id": "req-1", "original_query": "从工学部找篮球场，再去吃饭",
        "activities": ["basketball"], "subcategories": ["field"],
        "follow_up_requests": [{"query": "吃饭", "activities": ["meal"],
                                "subcategories": ["canteen", "restaurant", "fastfood"]}],
        "pending_slot": "selection", "start": {"type": "poi", "name": "工学部"},
        "candidate_poi_ids": ["poi_206"], "mode": "walk",
        "constraints": {"avoid_steps": False},
    }
    route = {"recommended": [[114.36, 30.54]], "start_name": "信息学部图书馆",
             "end_name": "工学部篮球场", "mode": "bike"}
    with patch.object(planner.agent_tools, "load_pois", return_value=[{
        "id": "poi_206", "name": "工学部篮球场",
    }]), patch.object(planner.agent_tools, "find_poi_ambiguous", return_value=(
        {"name": "武汉大学信息学部图书馆"}, [],
    )), patch.object(planner.agent_tools, "execute_tool", return_value=(
        route, {"route": route},
    )) as execute:
        completed = planner.run_agent(
            "改从信息学部图书馆出发，骑行，不走台阶",
            context={"place_request": saved, "selected_poi_id": "poi_206"},
            travel_mode="bike",
        )

    assert completed["response_kind"] == "route"
    args = next(call.args[1] for call in execute.call_args_list
                if call.args[0] == "plan_route")
    assert args["start"] == {"type": "poi", "name": "武汉大学信息学部图书馆"}
    assert args["mode"] == "bike"
    assert args["constraints"]["avoid_steps"] is True
    assert completed["suggestions"][0]["query"].startswith("从工学部篮球场出发找吃饭")
    assert "不走台阶" in completed["suggestions"][0]["query"]
    assert "骑行" in completed["suggestions"][0]["query"]


def test_category_via_candidates_use_current_open_and_constraint_filtered_network(monkeypatch):
    import networkx as nx

    graph = nx.MultiDiGraph()
    for node in range(4):
        graph.add_node(node, x=114.36 + node * 0.0001, y=30.53)
    graph.add_edge(0, 1, key=0, length=10, highway="footway")
    graph.add_edge(0, 2, key=0, length=10, highway="steps")
    graph.add_edge(0, 3, key=0, length=30, highway="footway")
    context = {"_route_graph_snapshot": graph, "_route_conditions_snapshot": [{"id": "closure"}]}
    candidate = {"id": "poi-via", "name": "顺路食堂", "type": "dining",
                 "subcategory": "canteen", "lon": 114.3601, "lat": 30.53,
                 "coordinates": {"lng": 114.3601, "lat": 30.53},
                 "activities": ["canteen_meal"]}
    ranked_graphs = []
    snapped_graphs = []

    monkeypatch.setattr(tools, "_prepare_route_context", lambda _ctx: (context, None))
    monkeypatch.setattr(tools, "_plan_common", lambda _args, _ctx: (graph, graph, "walk", None))
    monkeypatch.setattr(tools, "_resolve_endpoint",
                        lambda ref, _graph: (0 if ref["name"] == "起点" else 3, ref["name"], None))
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(tools, "gcj02_to_wgs84", lambda lng, lat: (lng, lat))
    def snap(_lng, _lat, candidate_graph, _name):
        snapped_graphs.append(candidate_graph)
        return 1, {"snap_distance_m": 0}, None
    monkeypatch.setattr(tools, "_snap_wgs_to_network", snap)
    def apply_closure(current_graph, _conditions, _mode, *, strict):
        assert strict
        open_graph = current_graph.copy()
        open_graph.remove_edge(0, 1, 0)
        return open_graph, {}, {(0, 1, 0)}, 1
    monkeypatch.setattr(tools, "apply_conditions_to_graph", apply_closure)
    def rank(current_graph, *_args, **_kwargs):
        ranked_graphs.append(current_graph)
        return [], []
    monkeypatch.setattr(tools, "rank_via_candidates", rank)

    result, _artifact = tools._tool_plan_via_route({
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "via_subcategory": "canteen", "constraints": {"avoid_steps": True},
    }, {})

    assert result["error"] == "no_on_the_way_via"
    assert snapped_graphs and ranked_graphs
    assert all(not candidate_graph.has_edge(0, 1)
               and not candidate_graph.has_edge(0, 2)
               for candidate_graph in [*snapped_graphs, *ranked_graphs])


def test_explicit_activity_survives_composite_query_when_search_delegates(monkeypatch):
    captured = {}
    def reachable(args, _ctx):
        captured.update(args)
        return {"candidates": []}, None
    monkeypatch.setattr(tools, "_tool_find_reachable_places", reachable)
    result, _artifact = tools._tool_search_poi_candidates(
        {"start": {"type": "poi", "name": "origin"},
         "activities": ["basketball"], "subcategories": ["field"],
         "constraints": {"avoid_steps": True}},
        {"query": "先打篮球再吃饭"},
    )
    assert result["candidates"] == []
    assert captured["activities"] == ["basketball"]
    assert captured["subcategories"] == ["field"]
    assert captured["constraints"] == {"avoid_steps": True}


def test_explicit_activity_survives_composite_query_in_reachability(monkeypatch):
    from types import SimpleNamespace
    import networkx as nx
    graph = nx.MultiDiGraph()
    graph.add_node(0, x=114.36, y=30.53)
    captured = {}
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "get_routing_index",
                        lambda _graph: SimpleNamespace(for_mode=lambda _mode: SimpleNamespace(graph=graph)))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda *_: (0, "origin", None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence",
                        lambda *_: {"snap_distance_m": 0})
    def categories(subcategories, poi_type, keyword, season, include_minor, *, activities):
        captured.update({"subcategories": subcategories, "activities": activities})
        return []
    monkeypatch.setattr(tools, "_search_poi_categories", categories)
    result, _artifact = tools._tool_find_reachable_places(
        {"start": {"type": "poi", "name": "origin"},
         "activities": ["basketball"], "subcategories": ["field"]},
        {"query": "先打篮球再吃饭",
         "_route_graph_snapshot": graph, "_route_conditions_snapshot": []},
    )
    assert result["candidates"] == []
    assert captured == {"subcategories": ["field"], "activities": ["basketball"]}


def test_recommendation_tool_schemas_expose_supported_constraints_and_categories():
    schemas = {item["function"]["name"]: item["function"]["parameters"]["properties"]
               for item in tools.TOOL_SCHEMAS}
    assert "constraints" in schemas["search_poi_candidates"]
    assert "subcategories" in schemas["find_reachable_places"]
