"""Common meal wording and an explicitly named nearby reference."""

from unittest.mock import patch

from agents import planner, tools
from spatial.poi_taxonomy import resolve_activity_query, resolve_activity_sequence


def test_ganfan_means_a_general_meal_but_explicit_canteen_stays_specific():
    assert resolve_activity_query("想去干饭")["activities"] == ["meal"]
    assert resolve_activity_query("去食堂干饭")["activities"] == ["canteen_meal"]
    assert resolve_activity_query("我想去食堂附近干饭")["activities"] == ["meal"]
    assert resolve_activity_query("我想去总图附近的食堂干饭")["activities"] == ["canteen_meal"]
    assert resolve_activity_query("不想干饭，想打篮球")["activities"] == ["basketball"]
    assert resolve_activity_query("不想去干饭") is None
    stages = resolve_activity_sequence("先打篮球再去干饭")
    assert [stage["intent"]["activities"] for stage in stages] == [["basketball"], ["meal"]]


def test_nearby_named_place_is_the_reference_when_no_separate_origin_is_given():
    assert tools.recommendation_origin_from_text("我想去总图附近干饭") == "总图"
    assert tools.recommendation_origin_from_text("我想在总图周边吃饭") == "总图"
    assert tools.recommendation_origin_from_text("从工学部出发去总图附近干饭") == "工学部"
    assert tools.recommendation_near_reference_from_text("从工学部出发去总图附近干饭") == "总图"


def test_named_nearby_meal_search_keeps_reference_and_returns_more_candidates():
    rows = [{"poi_id": f"poi_{i}", "name": f"餐饮地点{i}", "subcategory": "restaurant",
             "distance_m": 100 + i} for i in range(8)]

    def search(name, args, context):
        assert name == "search_poi_candidates"
        assert args["activities"] == ["meal"]
        assert args["start"]["name"] == "武汉大学图书馆(总馆)"
        assert args["limit"] == 20
        return {"candidates": rows, "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": rows}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=search):
        result = planner.run_agent("我想去总图附近干饭")
    assert result["response_kind"] == "candidates"
    assert len(result["candidates"]) == 8
    assert result["place_request"]["candidate_poi_ids"] == [row["poi_id"] for row in rows]
    assert result["place_request"]["start"] is None
    assert "总馆" in result["message"] and "附近" in result["message"]
    assert "参照点" in result["message"]
    # The browser can start on another page, so the shared introduction must
    # not single out a place that might not be among the visible cards.
    assert rows[0]["name"] not in result["message"]
    assert not rows[0].get("recommendation_start_name")


def test_nearby_search_center_and_trip_origin_are_not_conflated():
    rows = [{"poi_id": "poi_01", "name": "测试餐饮", "subcategory": "restaurant", "distance_m": 100}]

    def search(_name, args, _context):
        assert args["start"]["name"] == "武汉大学图书馆(总馆)"
        return {"candidates": rows, "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": rows}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=search):
        result = planner.run_agent("从工学部出发去总图附近干饭")
    assert result["response_kind"] == "candidates"
    assert result["place_request"]["start"]["name"] == "工学部"
    assert result["place_request"]["search_anchor"]["name"] == "武汉大学图书馆(总馆)"
    assert "最近的是测试餐饮" in result["message"]
    assert rows[0]["recommendation_start_name"] == "工学部"


def test_user_reported_nearby_query_keeps_library_origin():
    rows = [{"poi_id": "poi_347", "name": "肯德基(梅园店)", "subcategory": "fastfood", "distance_m": 100}]

    def search(_name, args, _context):
        assert args["start"]["name"] == "武汉大学图书馆(总馆)"
        return {"candidates": rows, "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": rows}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=search):
        result = planner.run_agent("我在信图，想去总图附近干饭")
    assert result["response_kind"] == "candidates"
    assert result["place_request"]["start"]["name"] in {"信图", "武汉大学信息学部图书馆"}
    assert result["place_request"]["search_anchor"]["name"] == "武汉大学图书馆(总馆)"


def test_unknown_nearby_anchor_is_not_replaced_by_the_trip_origin():
    with patch.object(planner.agent_tools, "execute_tool") as execute:
        result = planner.run_agent("从工学部出发去火星城附近干饭")
    assert result["response_kind"] == "clarify"
    assert "火星城" in result["message"]
    execute.assert_not_called()


def test_selecting_nearby_food_without_an_origin_asks_before_planning():
    rows = [{"poi_id": "meal_1", "name": "测试餐饮", "subcategory": "restaurant", "distance_m": 100}]
    def search(_name, args, _context):
        return {"candidates": rows, "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": rows}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=search):
        first = planner.run_agent("我想去总图附近干饭")
    with patch("agents.result_maintenance.candidate_snapshot_is_current", return_value=True), \
         patch.object(planner.agent_tools, "load_pois", return_value=[{"id": "meal_1", "name": "测试餐饮"}]), \
         patch.object(planner.agent_tools, "execute_tool") as execute:
        selected = planner.run_agent("去测试餐饮", context={"place_request": first["place_request"],
                                                            "selected_poi_id": "meal_1"})
    assert selected["response_kind"] == "clarify"
    assert "从哪里出发" in selected["message"]
    assert selected["place_request"]["selected_poi_id"] == "meal_1"
    execute.assert_not_called()


def test_expired_candidate_refresh_keeps_the_nearby_search_center():
    rows = [{"poi_id": "meal_1", "name": "测试餐饮", "subcategory": "restaurant", "distance_m": 100}]
    def search(_name, args, _context):
        assert args["start"]["name"] == "武汉大学图书馆(总馆)"
        return {"candidates": rows, "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": rows}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=search):
        first = planner.run_agent("从工学部出发去总图附近干饭")
        with patch("agents.result_maintenance.candidate_snapshot_is_current", side_effect=[False, True]):
            refreshed = planner.run_agent("去测试餐饮", context={"place_request": first["place_request"],
                                                                  "selected_poi_id": "meal_1"})
    assert refreshed["response_kind"] == "candidates"
    assert refreshed["place_request"]["start"]["name"] == "工学部"
    assert refreshed["place_request"]["search_anchor"]["name"] == "武汉大学图书馆(总馆)"


def test_explicit_small_count_is_not_expanded():
    assert planner._place_candidate_limit("推荐2个附近吃饭的地方", ["meal"], catalog=False) == 2
    assert planner._place_candidate_limit("附近干饭", ["meal"], catalog=False) == 20
    assert planner._place_candidate_limit("附近打篮球", ["basketball"], catalog=False) == 4


def test_retry_after_stale_publication_keeps_nearby_reference_without_assuming_origin():
    rows = [{"poi_id": "poi_068", "name": "桂园一食堂", "subcategory": "canteen", "distance_m": 100}]
    search_origins = []

    def search(name, args, _context):
        assert name == "search_poi_candidates"
        search_origins.append(args["start"]["name"])
        return {"candidates": rows, "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": rows}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=search):
        first = planner.run_agent("我想去总图附近干饭")
        pending = first["place_request"]
        # The publication guard clears an out-of-date candidate list and asks
        # for a fresh search while preserving the user's original request.
        pending.update({"pending_slot": "follow_up", "candidate_poi_ids": [], "selected_poi_id": None})
        retried = planner.run_agent("重新查询", context={"place_request": pending})

    assert retried["response_kind"] == "candidates"
    assert search_origins == ["武汉大学图书馆(总馆)"] * 2
    assert retried["place_request"]["search_anchor"]["name"] == search_origins[0]
    assert retried["place_request"]["start"] is None



def test_food_negation_does_not_select_the_rejected_type():
    for phrase, expected in [
        ("不想吃食堂，想吃餐厅", "restaurant_meal"),
        ("不去餐厅，换成快餐", "fast_food"),
        ("不想吃餐厅，想吃食堂", "canteen_meal"),
    ]:
        result = resolve_activity_query(phrase)
        assert result["status"] == "resolved"
        assert result["activities"] == [expected]


def test_food_exclusion_without_an_unambiguous_replacement_requires_clarification():
    for phrase in ("不想吃食堂", "不吃食堂，想吃食堂"):
        result = resolve_activity_query(phrase)
        assert result["status"] == "needs_clarification"


def test_nearby_food_revision_keeps_search_anchor_and_rejects_old_choice(monkeypatch):
    from copy import deepcopy

    calls = []
    def search(name, args, _context):
        calls.append((name, deepcopy(args)))
        assert name == "search_poi_candidates"
        is_restaurant = args["activities"] == ["restaurant_meal"]
        rows = [{"poi_id": "restaurant_1" if is_restaurant else "canteen_1",
                 "name": "测试餐厅" if is_restaurant else "测试食堂",
                 "subcategory": "restaurant" if is_restaurant else "canteen", "distance_m": 100}]
        return {"candidates": rows, "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": rows}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", search)
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", None)
    first = planner.run_agent("我想去总图附近干饭")
    before = deepcopy(first["place_request"])
    result = planner.run_agent("不想吃食堂，想吃餐厅", context={
        "place_request": first["place_request"], "selected_poi_id": "canteen_1",
    })
    assert first["place_request"] == before
    assert result["response_kind"] == "candidates"
    assert calls[-1][1]["start"]["name"] == "武汉大学图书馆(总馆)"
    assert calls[-1][1]["activities"] == ["restaurant_meal"]
    assert result["place_request"]["search_anchor"]["name"] == "武汉大学图书馆(总馆)"
    assert result["place_request"]["start"] is None
    assert result["place_request"]["candidate_poi_ids"] == ["restaurant_1"]
    assert result.get("route") is None



def test_new_food_request_cannot_drop_an_unhandled_restriction_on_followup(monkeypatch):
    calls = []
    def search(name, args, _context):
        calls.append((name, args))
        row = {"poi_id": "restaurant_1", "name": "测试餐厅", "distance_m": 100}
        return {"candidates": [row], "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": [row]}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", search)
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", None)
    query = "我想去总图附近吃饭，不想吃食堂，想吃餐厅，不吃辣"
    initial = planner.run_agent(query)
    assert initial["response_kind"] == "clarify"
    assert initial["place_request"]["unresolved_route_requirements"] == query
    followup = planner.run_agent("餐厅", context={"place_request": initial["place_request"],
                                                "active_task_request": query})
    assert followup["response_kind"] == "clarify"
    assert followup.get("candidates") is None
    assert calls == []



def test_new_nearby_food_search_uses_the_positive_category_and_named_anchor(monkeypatch):
    calls = []
    def search(name, args, _context):
        calls.append((name, args))
        row = {"poi_id": "restaurant_1", "name": "测试餐厅", "subcategory": "restaurant", "distance_m": 100}
        return {"candidates": [row], "start_name": args["start"]["name"],
                "data_version": "test-data", "road_condition_version": "test-roads"}, {"candidates": [row]}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", search)
    result = planner.run_agent("我想去总图附近干饭，不想吃食堂，想吃餐厅")
    assert result["response_kind"] == "candidates"
    assert calls[0][1]["activities"] == ["restaurant_meal"]
    assert calls[0][1]["start"]["name"] == "武汉大学图书馆(总馆)"
    assert result["place_request"]["search_anchor"]["name"] == "武汉大学图书馆(总馆)"
    assert result["place_request"]["start"] is None
