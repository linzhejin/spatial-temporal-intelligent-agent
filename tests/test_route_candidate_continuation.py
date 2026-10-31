"""A route plus a later fuzzy destination must retain a real continuation."""
from types import SimpleNamespace
from unittest.mock import patch
import shutil
from pathlib import Path

from agents import planner
from agents.route_state import build_route_state
from tests.test_vision_condition_feedback import _call, _event, _message


def _route():
    state = build_route_state(
        route_kind="direct", original_query="从总图到樱顶", start={"type": "poi", "name": "总图"},
        end={"type": "poi", "name": "樱顶"}, travel_mode="walk", hard_constraints={"avoid_steps": True},
        strategy={"name": "shortest", "source": "explicit_nl"}, data_version="data-1",
        road_condition_version="roads-1",
    )
    return {"start_name": "武汉大学图书馆(总馆)", "end_name": "武汉大学老斋舍", "mode": "walk",
            "route_state": state, "route_verification": {"status": "pass"},
            "recommended": [[114.3, 30.5], [114.31, 30.51]], "recommended_edge_ids": [[1, 2, 0]],
            "recommended_length_m": 500, "duration_min": 8}


def _run(query, route=None, *, recheck_error=False, model_search=True):
    model_calls = [_call("list_road_conditions"), _call("plan_route")]
    if model_search:
        model_calls.append(_call("search_poi_candidates", {"activities": ["canteen_meal"]}))
    replies = [_message(calls=model_calls),
               _message(content="路线和候选已经生成。")]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: replies.pop(0))))
    route = _route() if route is None else route
    calls = []

    def execute(name, args, _context):
        calls.append((name, args))
        if name == "list_road_conditions":
            return {"conditions": [_event()], "count": 1}, None
        if name == "plan_route":
            return route, {"route": route}
        if model_search and len([entry for entry in calls if entry[0] == "search_poi_candidates"]) == 1:
            stale = [{"poi_id": "poi_307", "name": "星湖园食堂"}]
            return {"candidates": stale}, {"candidates": stale}
        if recheck_error:
            return {"error": "road_conditions_unavailable", "message": "路况暂不可用"}, None
        current = [{"poi_id": "poi_070", "name": "桂园食堂", "distance_m": 200}]
        return {"candidates": current}, {"candidates": current}

    with patch.object(planner.config, "DEEPSEEK_API_KEY", "fixture"), \
         patch.object(planner, "_make_client", return_value=client), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        return planner.run_agent(query), calls


def test_ordered_fuzzy_destination_preserves_first_leg_and_rechecks_origin():
    result, calls = _run("从总图到樱顶，然后去附近食堂，不走台阶")
    assert result["response_kind"] == "route" and result["needs_input"] is True
    assert [x["poi_id"] for x in result["candidates"]] == ["poi_070"]
    pending = result["place_request"]
    assert pending["pending_slot"] == "selection"
    assert pending["start"]["poi_id"] == "poi_161"
    assert pending["completed_legs"][0]["start_name"] == "武汉大学图书馆(总馆)"
    assert pending["constraints"]["avoid_steps"] is True
    assert pending["activities"] == ["canteen_meal"]
    assert pending["candidate_poi_ids"] == ["poi_070"]
    searches = [args for name, args in calls if name == "search_poi_candidates"]
    assert len(searches) == 2 and searches[-1]["start"]["poi_id"] == "poi_161"
    assert searches[-1]["constraints"]["avoid_steps"] is True


def test_named_middle_stop_cannot_silently_disappear():
    result, _ = _run("从总图到樱顶，然后去附近食堂，再去梅园操场，最后去超市，不走台阶")
    pending = result.get("place_request") or {}
    followups = pending.get("follow_up_requests") or []
    names = [item.get("query", "") for item in followups]
    assert len(followups) == 2 and followups[0]["destination_poi_id"] == "poi_082"
    assert "梅园操场" in names[0] and "超市" in names[1]
    assert followups[1]["activities"] == ["shopping"]


def test_route_only_model_call_still_continues_explicit_activity():
    result, calls = _run("从总图到樱顶，然后去附近食堂，不走台阶", model_search=False)
    assert result.get("needs_input") is True
    assert result["place_request"]["start"]["poi_id"] == "poi_161"
    assert [item["poi_id"] for item in result["candidates"]] == ["poi_070"]
    assert len([name for name, _ in calls if name == "search_poi_candidates"]) == 1


def test_multiple_activity_stages_remain_in_order():
    result, _ = _run("从总图到樱顶，然后去附近食堂，再去超市，不走台阶")
    assert result.get("needs_input") is True
    assert result["place_request"]["activities"] == ["canteen_meal"]
    followups = result["place_request"]["follow_up_requests"]
    assert len(followups) == 1 and followups[0]["activities"] == ["shopping"]


def test_unrecognized_middle_stage_is_never_discarded():
    result, _ = _run("从总图到樱顶，然后去附近食堂，再去不存在的地点，最后去超市，不走台阶")
    assert result.get("needs_input") is True and result.get("place_request") is None
    assert "完整" in result["message"]


def test_excluded_named_road_is_not_discarded_when_not_in_route_state():
    result, _ = _run("从总图到樱顶，避开桂园路，然后去附近食堂，不走台阶")
    assert result.get("needs_input") is True and result.get("place_request") is None
    assert result.get("candidates") is None
    assert "避让" in result["message"] or "避开" in result["message"]


def test_avoiding_named_road_in_other_common_wording_is_not_discarded():
    for wording in ("避让桂园路", "不要经过桂园路", "不要从桂园路走"):
        result, _ = _run(f"从总图到樱顶，{wording}，然后去附近食堂，不走台阶")
        assert result.get("needs_input") is True and result.get("place_request") is None
        assert result.get("candidates") is None


def test_unrelated_excluded_edge_cannot_prove_named_road_is_avoided():
    route = _route()
    route["route_state"]["hard_constraints"]["excluded_edge_ids"] = [[1, 2, 0]]
    result, _ = _run("从总图到樱顶，避开桂园路，然后去附近食堂，不走台阶", route)
    assert result.get("needs_input") is True and result.get("place_request") is None
    assert result.get("candidates") is None


def test_wheelchair_accessibility_is_not_implicitly_claimed():
    result, _ = _run("从总图到樱顶，然后去附近食堂，我坐轮椅")
    assert result.get("needs_input") is True and result.get("place_request") is None
    assert result.get("candidates") is None


def test_stair_and_slope_avoidance_verbs_require_matching_route_constraints():
    for phrase in ("避让台阶", "别走楼梯", "绕过陡坡"):
        route = _route()
        route["route_state"]["hard_constraints"]["avoid_steps"] = False
        result, _ = _run(f"从总图到樱顶，{phrase}，然后去附近食堂", route)
        assert result.get("needs_input") is True and result.get("place_request") is None, phrase
        assert result.get("candidates") is None, phrase


def test_stair_and_slope_avoidance_allows_matching_route_constraints():
    for phrase in ("避让台阶", "别走楼梯"):
        result, _ = _run(f"从总图到樱顶，{phrase}，然后去附近食堂")
        assert result["place_request"]["constraints"]["avoid_steps"] is True
    route = _route()
    route["route_state"]["hard_constraints"]["slope"] = "avoid"
    result, _ = _run("从总图到樱顶，绕过陡坡，然后去附近食堂", route)
    assert result["place_request"]["constraints"]["avoid_slope"] is True


def test_accessibility_flag_alone_cannot_verify_candidate_access():
    route = _route()
    route["route_state"]["hard_constraints"]["require_verified_accessibility"] = True
    result, _ = _run("从总图到樱顶，然后去附近食堂，我坐轮椅", route)
    assert result.get("needs_input") is True and result.get("place_request") is None
    assert result.get("candidates") is None


def test_different_mode_on_a_later_stage_never_silently_disappears():
    result, _ = _run("从总图到樱顶，然后步行去附近食堂，再骑车去超市，不走台阶")
    assert result.get("needs_input") is True and result.get("place_request") is None
    assert "方式不同" in result["message"]


def test_explicit_venue_conditions_are_never_silently_dropped():
    for query in (
        "从总图到樱顶，然后去附近食堂，再去室内篮球场，不走台阶",
        "从总图到樱顶，然后去附近食堂，最后去24小时营业的超市，不走台阶",
        "从总图到樱顶，然后去附近食堂，最后去还开门的超市，不走台阶",
        "从总图到樱顶，然后去附近食堂，再去有空调的篮球场，不走台阶",
    ):
        result, _ = _run(query)
        assert result.get("needs_input") is True and result.get("place_request") is None
        assert result.get("candidates") is None
        assert "条件" in result["message"] or "核实" in result["message"]


def test_spatial_reference_does_not_replace_requested_activity():
    for reference in ("星湖园食堂旁的篮球场", "图书馆附近的篮球场"):
        result, _ = _run(f"从总图到樱顶，然后去附近食堂，再去{reference}，不走台阶")
        assert result.get("needs_input") is True and result.get("place_request") is None
        assert result.get("candidates") is None
        assert "参照" in result["message"] or "无法确认" in result["message"]


def test_spatial_reference_is_stopped_before_deterministic_route_branch():
    with patch.object(planner, "_make_client", side_effect=AssertionError("unexpected model")), \
         patch.object(planner.agent_tools, "execute_tool", side_effect=AssertionError("unexpected tool")):
        result = planner.run_agent("从总图到樱顶，再去图书馆附近的篮球场，不走台阶")
    assert result.get("needs_input") is True and result.get("route") is None
    assert "参照" in result["message"]


def test_simple_nearby_stage_is_not_mistaken_for_a_spatial_reference():
    result, _ = _run("从总图到樱顶，然后去附近食堂，不走台阶")
    assert result["place_request"]["activities"] == ["canteen_meal"]


def test_named_venue_is_not_replaced_by_nearest_canteen():
    result, _ = _run("从总图到樱顶，然后去桂园食堂，不走台阶")
    assert result.get("place_request") is None and result.get("candidates") is None
    assert result.get("needs_input") is True and "完整" in result["message"]


def test_information_only_query_does_not_become_a_planning_continuation():
    result, calls = _run("规划从总图到樱顶的路线，同时看看附近有哪些食堂")
    assert result["response_kind"] == "route"
    assert result.get("needs_input") is not True
    assert result.get("place_request") is None
    assert "星湖园食堂" in result["message"]
    assert len([name for name, _ in calls if name == "search_poi_candidates"]) == 1


def test_unverified_first_route_does_not_invent_continuation_start():
    route = _route()
    route.pop("route_state")
    result, calls = _run("从总图到樱顶，然后去附近食堂，不走台阶", route)
    assert result.get("place_request") is None
    assert result.get("route") is None and result.get("route_kind") is None
    assert result["response_kind"] == "clarify" and "未展示" in result["message"]
    assert len([name for name, _ in calls if name == "search_poi_candidates"]) == 1


def test_different_first_destination_does_not_start_at_an_unrequested_place():
    route = _route()
    route["route_state"]["end"] = {"type": "poi", "name": "桂园食堂"}
    result, _ = _run("从总图到樱顶，然后去附近食堂，不走台阶", route)
    assert result.get("place_request") is None and result.get("candidates") is None
    assert result.get("needs_input") is True
    assert result.get("route") is None and result.get("route_kind") is None


def test_different_mode_for_later_leg_is_not_silently_changed():
    result, _ = _run("从总图到樱顶，然后骑车去附近食堂，不走台阶")
    assert result.get("place_request") is None and result.get("candidates") is None
    assert result.get("needs_input") is True
    assert "方式不同" in result["message"]


def test_unavailable_recheck_does_not_reuse_stale_candidates():
    result, _ = _run("从总图到樱顶，然后去附近食堂，不走台阶", recheck_error=True)
    assert result.get("candidates") is None
    assert result.get("needs_input") is True
    assert result["place_request"]["pending_slot"] == "follow_up"
    assert result.get("route") is not None
    assert "路况暂不可用" in result["message"]


def test_first_leg_missing_hard_constraint_is_not_shown():
    route = _route()
    route["route_state"]["hard_constraints"]["avoid_steps"] = False
    result, _ = _run("从总图到樱顶，然后去附近食堂，不走台阶", route)
    assert result["response_kind"] == "clarify" and result.get("needs_input") is True
    assert result.get("route") is None and result.get("route_kind") is None
    assert "未展示" in result["message"]


def test_selected_candidate_unavailable_keeps_task_and_supports_retry():
    result, _ = _run("从总图到樱顶，然后去附近食堂，再去梅园操场，最后去超市，不走台阶")
    pending = result["place_request"]
    selected = result["candidates"][0]["poi_id"]
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"error": "route_unavailable", "message": "路线服务暂不可用"}, None,
    )):
        failed = planner.run_agent("桂园食堂", context={
            "place_request": pending, "selected_poi_id": selected,
        })
    assert failed["needs_input"] is True and failed["route"] is None
    assert "路线服务暂不可用" in failed["message"]
    assert failed["place_request"]["selected_poi_id"] == selected
    assert failed["place_request"]["completed_legs"] == pending["completed_legs"]
    assert failed["place_request"]["follow_up_requests"] == pending["follow_up_requests"]
    assert failed["place_request"]["constraints"] == pending["constraints"]
    assert "重试" in failed["clarify"]["options"]


def test_real_graph_rescans_from_verified_route_endpoint(monkeypatch, tmp_path):
    from agents import tools
    from spatial import network

    events = tmp_path / "conditions.json"
    events.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(events))
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    cache = tmp_path / "graph.graphml"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "data/whu_road_network.graphml", cache)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(network, "_cache_path", lambda: str(cache))
    monkeypatch.setattr(network, "_download_network", lambda *_: (_ for _ in ()).throw(AssertionError("network")))
    network.load_or_download_network()
    _, artifact = tools.execute_tool("plan_route", {
        "start": {"type": "poi", "name": "总图"},
        "end": {"type": "poi", "name": "樱顶"}, "mode": "walk",
        "constraints": {"avoid_steps": True}, "strategy": "shortest",
    }, {"query": "从总图到樱顶，不走台阶"})
    assert artifact and artifact.get("route")
    pending, candidates, failure = planner._recheck_route_candidate_stage(
        "从总图到樱顶，然后去附近食堂，不走台阶", artifact["route"],
        None, None, "walk", "isolated-test", tools.execute_tool,
    )
    assert failure is None and candidates
    assert pending["start"]["poi_id"] == "poi_161"
    assert pending["constraints"]["avoid_steps"] is True
    assert pending["candidate_poi_ids"] == [item["poi_id"] for item in candidates]
    retry_pending = dict(pending, pending_slot="follow_up", candidate_poi_ids=[])
    retried = planner.run_agent("重试", context={"place_request": retry_pending}, uid="isolated-test")
    assert retried.get("candidates"), retried.get("message")
    assert retried["place_request"]["completed_legs"], retried.get("message")
    selected = candidates[0]
    response = planner.run_agent(selected["name"], context={
        "place_request": pending, "selected_poi_id": selected["poi_id"],
    }, uid="isolated-test")
    assert response["response_kind"] == "route", response.get("message")
    assert response["route_kind"] == "via"
    assert response["route"]["route_state"]["start"]["name"] in {"总图", "武汉大学图书馆(总馆)"}
    assert response["route"]["route_state"]["end"]["name"] == selected["name"]

    mixed, choices, failure = planner._recheck_route_candidate_stage(
        "从总图到樱顶，然后去附近食堂，再去梅园操场，最后去超市，不走台阶",
        artifact["route"], None, None, "walk", "isolated-test", tools.execute_tool,
    )
    assert failure is None and choices
    meal = choices[0]
    next_stage = planner.run_agent(meal["name"], context={
        "place_request": mixed, "selected_poi_id": meal["poi_id"],
    }, uid="isolated-test")
    assert next_stage.get("candidates"), next_stage.get("message")
    assert next_stage["place_request"]["start"]["poi_id"] == "poi_082"
    shop = next_stage["candidates"][0]
    final = planner.run_agent(shop["name"], context={
        "place_request": next_stage["place_request"], "selected_poi_id": shop["poi_id"],
    }, uid="isolated-test")
    assert final["response_kind"] == "route", final.get("message")
    assert [leg["end_name"] for leg in final["route"]["legs"]] == [
        "武汉大学老斋舍", meal["name"], "武汉大学梅园操场", shop["name"],
    ]
    assert final["route"]["route_state"]["hard_constraints"]["avoid_steps"] is True
