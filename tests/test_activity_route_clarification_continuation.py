"""A model's origin question must not erase the unfinished ordered task."""

from unittest.mock import patch

from agents import planner


def _sport_candidates(_name, args, _context):
    assert args["activities"] == ["basketball"]
    row = {"poi_id": "poi_206", "name": "工学部篮球场", "subcategory": "field",
           "activities": ["basketball"], "distance_m": 120}
    return {"candidates": [row], "start_name": "工学部体育馆",
            "data_version": "test-data", "road_condition_version": "test-events"}, {"candidates": [row]}


def test_origin_answer_retains_same_task_order_and_new_step_constraint():
    original = "我想先打篮球再去食堂，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool", side_effect=_sport_candidates) as execute:
        result = planner.run_agent("我在工学部体育馆，步行不走台阶", context={
            "active_task_request": original, "task_revision": 2,
        })

    assert result["response_kind"] == "candidates"
    pending = result["place_request"]
    assert pending["original_query"] == original
    assert pending["start"]["name"] == "工学部体育馆"
    assert pending["activities"] == ["basketball"]
    assert pending["follow_up_requests"][0]["subcategories"] == ["canteen"]
    assert pending["constraints"]["avoid_steps"] is True
    assert execute.call_args.args[1]["constraints"]["avoid_steps"] is True


def test_supplemented_step_constraint_merges_existing_task_constraints():
    original = "我想先打篮球再去食堂"
    first = planner.run_agent(original)
    pending = first["place_request"]
    assert pending["pending_slot"] == "start"
    pending["constraints"] = {"avoid_slope": True}
    with patch.object(planner.agent_tools, "execute_tool", side_effect=_sport_candidates) as execute:
        result = planner.run_agent("我在工学部体育馆，步行不走台阶", context={
            "active_task_request": original, "place_request": pending, "task_revision": 2,
        })
    assert result["response_kind"] == "candidates"
    assert result["place_request"]["constraints"]["avoid_steps"] is True
    assert result["place_request"]["constraints"]["avoid_slope"] is True
    assert execute.call_args.args[1]["constraints"] == {"avoid_steps": True, "slope": "avoid"}


def test_fresh_origin_only_still_asks_for_destination():
    result = planner.run_agent("我在工学部体育馆，步行不走台阶")
    assert result["response_kind"] == "clarify"
    assert result["place_request"]["pending_slot"] == "destination"


def test_explicit_new_mode_is_applied_when_no_structured_pending_place_exists():
    original = "我想先打篮球再去食堂，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool", side_effect=_sport_candidates) as execute:
        result = planner.run_agent("我在工学部体育馆，骑行不走台阶", context={
            "active_task_request": original, "task_revision": 2,
        }, travel_mode="walk")
    assert result["response_kind"] == "candidates"
    assert result["place_request"]["mode"] == "bike"
    assert execute.call_args.args[1]["mode"] == "bike"


def test_original_mode_is_retained_when_origin_answer_does_not_change_it():
    original = "我想先打篮球再去食堂，骑行，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool", side_effect=_sport_candidates) as execute:
        result = planner.run_agent("我在工学部体育馆，不走台阶", context={
            "active_task_request": original, "task_revision": 2,
        }, travel_mode="walk")
    assert result["response_kind"] == "candidates"
    assert result["place_request"]["mode"] == "bike"
    assert execute.call_args.args[1]["mode"] == "bike"


def test_global_scenery_and_casual_meal_survive_an_origin_question():
    original = "全程风景优先，我想先打篮球再去干饭，步行"
    with patch.object(planner.agent_tools, "execute_tool", side_effect=_sport_candidates):
        result = planner.run_agent("我在工学部体育馆", context={
            "active_task_request": original, "task_revision": 2,
        })
    assert result["response_kind"] == "candidates"
    assert result["place_request"]["strategy"] == "scenery"
    assert result["place_request"]["follow_up_requests"][0]["activities"] == ["meal"]


def test_latest_explicit_single_mode_overrides_the_original_mode():
    original = "我想先打篮球再去食堂，骑行，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool", side_effect=_sport_candidates) as execute:
        result = planner.run_agent("我在工学部体育馆，步行不走台阶", context={
            "active_task_request": original, "task_revision": 2,
        }, travel_mode="bike")
    assert result["response_kind"] == "candidates"
    assert result["place_request"]["mode"] == "walk"
    assert execute.call_args.args[1]["mode"] == "walk"


def test_original_stage_specific_modes_are_not_collapsed_into_one():
    original = "我想先步行打篮球再骑行去食堂，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool") as execute:
        result = planner.run_agent("我在工学部体育馆，不走台阶", context={
            "active_task_request": original, "task_revision": 2,
        })
    assert result["response_kind"] == "clarify"
    assert "暂不能可靠处理的条件" in result["message"]
    execute.assert_not_called()


def test_multiple_modes_in_origin_answer_are_not_collapsed_into_one():
    original = "我想先打篮球再去食堂，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool") as execute:
        result = planner.run_agent("我在工学部体育馆，步行骑行不走台阶", context={
            "active_task_request": original, "task_revision": 2,
        })
    assert result["response_kind"] == "clarify"
    assert "暂不能可靠处理的条件" in result["message"]
    execute.assert_not_called()


def test_unrecognized_additional_preference_is_not_silently_treated_as_origin_only():
    original = "我想先打篮球再去食堂，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool") as execute:
        result = planner.run_agent("我在工学部体育馆，步行不走台阶，不吃辣", context={
            "active_task_request": original, "task_revision": 2,
        })
    assert result["response_kind"] == "clarify"
    assert "暂不能可靠处理的条件" in result["message"]
    execute.assert_not_called()


def test_unrecognized_original_preference_is_not_silently_dropped():
    original = "我想先打篮球再去食堂，不吃辣，请帮我规划一条路线"
    with patch.object(planner.agent_tools, "execute_tool") as execute:
        result = planner.run_agent("我在工学部体育馆，步行不走台阶", context={
            "active_task_request": original, "task_revision": 2,
        })
    assert result["response_kind"] == "clarify"
    assert "暂不能可靠处理的条件" in result["message"]
    execute.assert_not_called()
