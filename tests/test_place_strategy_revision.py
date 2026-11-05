"""A confirmed global route objective must replan the whole ordered trip."""

from copy import deepcopy

import pytest

from agents import planner, route_state
from agents.result_maintenance import capture_dependencies
from agents.route_state import build_route_state, current_road_condition_version


START = {"type": "poi", "poi_id": "poi_start", "name": "工学部"}
SPORT = {"type": "poi", "poi_id": "poi_sport", "name": "工学部篮球场"}
FOOD = {"type": "poi", "poi_id": "poi_food", "name": "工学部食堂"}
EVENT_VERSION = current_road_condition_version([])


def _state(start, end, strategy, *, via=None, constraints=None):
    return build_route_state(
        route_kind="via" if via else "direct", original_query="先运动再用餐",
        start=start, end=end, via=via, travel_mode="walk",
        hard_constraints={"avoid_steps": True} if constraints is None else constraints,
        strategy={"name": strategy, "source": "explicit_nl"},
        data_version="data-test", road_condition_version=EVENT_VERSION,
    )


def _leg(start, end, strategy, edge, coords):
    return {"mode": "walk", "start_name": start["name"], "end_name": end["name"],
            "recommended": coords, "recommended_edge_ids": [edge],
            "distance_m": 100, "duration_min": 2,
            "route_state": _state(start, end, strategy),
            "route_verification": {"status": "pass"}, "weather_input_sha256": "a" * 64}


def _pending():
    first = _leg(START, SPORT, "shortest", [1, 2, 0], [[114.35, 30.54], [114.36, 30.54]])
    dependencies = capture_dependencies([first], task_revision=0)
    dependencies["candidate_snapshot"] = {
        "data_version": "data-test", "road_condition_version": EVENT_VERSION,
    }
    return {"request_id": "strategy-test", "original_query": "吃饭",
            "root_query": "先运动再用餐，不走台阶", "activities": ["meal"],
            "subcategories": ["canteen"], "pending_slot": "selection", "start": SPORT,
            "completed_legs": [first], "follow_up_requests": [],
            "candidate_poi_ids": [FOOD["poi_id"]], "selected_poi_id": None,
            "mode": "walk", "strategy": "shortest", "constraints": {"avoid_steps": True},
            "result_dependencies": dependencies}


def _setup(monkeypatch, *, replan_result="valid"):
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr(route_state, "current_data_version", lambda *_: "data-test")
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": point["poi_id"], "name": point["name"]} for point in (START, SPORT, FOOD)
    ])
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda context: (
        {**context, "_route_graph_snapshot": object(), "_route_conditions_snapshot": []}, None))
    monkeypatch.setattr(planner.agent_tools, "_data_version_for_graph", lambda _: "data-test")
    calls = []

    def tool(name, args, _context):
        calls.append((name, deepcopy(args)))
        if name == "plan_route":
            route = _leg(SPORT, FOOD, args["strategy"], [2, 3, 0],
                         [[114.36, 30.54], [114.37, 30.54]])
            return {}, {"route": route}
        assert name == "plan_via_route"
        if replan_result == "unavailable":
            return {"error": "route_not_found", "message": "没有可用路线"}, None
        actual_strategy = "shortest" if replan_result == "wrong_strategy" else args["strategy"]
        constraints = {} if replan_result == "wrong_constraints" else {"avoid_steps": True}
        full = {"mode": "walk", "start_name": START["name"], "end_name": FOOD["name"],
                "recommended": [[114.35, 30.54], [114.36, 30.54], [114.37, 30.54]],
                "recommended_edge_ids": [[1, 2, 0], [2, 3, 0]], "distance_m": 200,
                "duration_min": 4, "route_verification": {"status": "pass"},
                "route_state": _state(START, FOOD, actual_strategy, via=SPORT, constraints=constraints)}
        return full, {"route": full}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tool)
    return calls


@pytest.mark.parametrize("policy", ["conservative", "full_recompute"])
@pytest.mark.parametrize("query", ["就去这里，全程风景优先", "就去这里，全程风景优先，不走台阶"])
def test_confirmed_scenery_goal_replans_all_stops_with_constraints(monkeypatch, policy, query):
    monkeypatch.setattr(planner.config, "RESULT_MAINTENANCE_POLICY", policy)
    calls = _setup(monkeypatch)
    pending = _pending()
    original = deepcopy(pending)
    result = planner.run_agent(query, context={
        "place_request": pending, "selected_poi_id": FOOD["poi_id"],
    }, proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert pending == original
    assert result["response_kind"] == "route"
    assert [name for name, _ in calls] == ["plan_route", "plan_via_route"]
    args = calls[-1][1]
    assert args["strategy"] == "scenery" and args["mode"] == "walk"
    assert args["constraints"] == {"avoid_steps": True}
    assert [args["start"]["poi_id"], args["via_points"][0]["poi_id"], args["end"]["poi_id"]] == [
        START["poi_id"], SPORT["poi_id"], FOOD["poi_id"]]
    assert result["route"]["route_state"]["strategy"]["name"] == "scenery"
    assert result["result_maintenance"]["prefix_action"] == "recomputed"


@pytest.mark.parametrize("failure", ["unavailable", "wrong_strategy", "wrong_constraints"])
def test_unusable_replan_retains_the_task_and_never_delivers_old_route(monkeypatch, failure):
    calls = _setup(monkeypatch, replan_result=failure)
    result = planner.run_agent("就去这里，全程风景优先", context={
        "place_request": _pending(), "selected_poi_id": FOOD["poi_id"],
    }, proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert [name for name, _ in calls] == ["plan_route", "plan_via_route"]
    assert result["needs_input"] is True and result.get("route") is None
    assert result["place_request"]["pending_slot"] == "route_retry"
    assert result["place_request"]["completed_legs"][0]["route_state"]["end"] == SPORT
    retry = planner.run_agent("重试", context={"place_request": result["place_request"]})
    assert retry["needs_input"] is True and retry.get("route") is None
    assert [name for name, _ in calls][-1] == "plan_via_route"


def test_missing_confirmed_goal_does_not_replan_mixed_strategy_legs(monkeypatch):
    calls = _setup(monkeypatch)
    pending = _pending()
    pending["pending_slot"] = "route_retry"
    pending["completed_legs"].append(_leg(SPORT, FOOD, "scenery", [2, 3, 0],
                                           [[114.36, 30.54], [114.37, 30.54]]))
    original = deepcopy(pending)
    result = planner.run_agent("重试", context={"place_request": pending})
    assert result["needs_input"] is True and result.get("route") is None
    assert result["place_request"] == original
    assert calls == []


def test_successful_retry_without_a_known_cause_does_not_claim_a_road_event(monkeypatch):
    _setup(monkeypatch)
    pending = _pending()
    pending["pending_slot"] = "route_retry"
    pending["completed_legs"].append(_leg(
        SPORT, FOOD, "shortest", [2, 3, 0], [[114.36, 30.54], [114.37, 30.54]],
    ))

    result = planner.run_agent("重试", context={"place_request": pending})

    assert result["response_kind"] == "route"
    assert "重新规划全程" in result["message"]
    assert "路况发生变化" not in result["message"]


@pytest.mark.parametrize("query", ["就去这里，风景优先", "就去这里，只有后一段风景优先", "就去这里，不要风景优先",
                                  "就去这里，看风景", "就去这里，风景好", "就去这里，少爬坡",
                                  "就去这里，赶时间", "就去这里，不绕路", "就去这里，少走坡"])
def test_ambiguous_or_stage_scoped_goal_is_not_applied_to_the_whole_trip(monkeypatch, query):
    calls = _setup(monkeypatch)
    pending = _pending()
    result = planner.run_agent(query, context={
        "place_request": pending, "selected_poi_id": FOOD["poi_id"],
    })
    assert result["needs_input"] is True and result.get("route") is None
    assert result["place_request"]["completed_legs"] == pending["completed_legs"]
    assert result["place_request"]["selected_poi_id"] == FOOD["poi_id"]
    if "少爬坡" in query or "少走坡" in query:
        assert result["place_request"]["constraints"] == pending["constraints"]
        assert result["place_request"]["unresolved_route_requirements"] == query
    assert calls == []


@pytest.mark.parametrize("name", ["赏樱亭", "游览服务点", "散步广场"])
def test_ordinary_place_names_do_not_replace_existing_trip_strategy(name):
    pending = _pending()
    corrected = planner._apply_place_request_correction(pending, "去" + name)
    assert corrected["strategy"] == "shortest"
    assert corrected.get("confirmed_strategy_override") is None


@pytest.mark.parametrize("root", ["第一段必须走最短路线，之后再吃饭", "先到篮球场少爬坡，然后吃饭"])
def test_previous_stage_specific_goal_cannot_be_silently_overridden(monkeypatch, root):
    calls = _setup(monkeypatch)
    pending = _pending()
    pending["root_query"] = root
    result = planner.run_agent("就去这里，全程风景优先", context={
        "place_request": pending, "selected_poi_id": FOOD["poi_id"],
    })
    assert result["needs_input"] is True and result.get("route") is None
    assert result["place_request"]["completed_legs"] == pending["completed_legs"]
    assert calls == []


def test_ambiguous_scope_can_be_confirmed_without_losing_selection(monkeypatch):
    from agents.state_models import PlaceRequestState

    calls = _setup(monkeypatch)
    question = planner.run_agent("就去这里，风景优先", context={
        "place_request": _pending(), "selected_poi_id": FOOD["poi_id"],
    })
    assert question["needs_input"] is True and question.get("route") is None
    assert "全程风景优先" in question["message"]
    assert question["place_request"]["selected_poi_id"] == FOOD["poi_id"]
    assert not calls
    saved = PlaceRequestState.model_validate(question["place_request"]).model_dump(mode="json")
    result = planner.run_agent("全程风景优先", context={"place_request": saved})
    assert result["response_kind"] == "route"
    assert result["route"]["route_state"]["strategy"]["name"] == "scenery"
    assert result["route"]["route_state"]["end"]["poi_id"] == FOOD["poi_id"]
    assert [name for name, _ in calls] == ["plan_route", "plan_via_route"]


@pytest.mark.parametrize("change", ["mode", "constraints"])
def test_incompatible_previous_mode_or_constraints_are_not_overridden(monkeypatch, change):
    calls = _setup(monkeypatch)
    pending = _pending()
    first_state = pending["completed_legs"][0]["route_state"]
    if change == "mode":
        first_state["travel_mode"] = "bike"
    else:
        first_state["hard_constraints"] = {}
    result = planner.run_agent("就去这里，全程风景优先", context={
        "place_request": pending, "selected_poi_id": FOOD["poi_id"],
    })
    assert result["needs_input"] is True and result.get("route") is None
    assert result["place_request"]["completed_legs"][0]["route_state"] == first_state
    assert [name for name, _ in calls] == ["plan_route"]


def test_unconfirmed_second_leg_strategy_cannot_override_first(monkeypatch):
    calls = _setup(monkeypatch)
    original_tool = planner.agent_tools.execute_tool

    def unexpected_strategy(name, args, context):
        if name == "plan_route":
            args = {**args, "strategy": "scenery"}
        return original_tool(name, args, context)

    monkeypatch.setattr(planner.agent_tools, "execute_tool", unexpected_strategy)
    result = planner.run_agent("就去这里", context={
        "place_request": _pending(), "selected_poi_id": FOOD["poi_id"],
    })
    assert result["needs_input"] is True and result.get("route") is None
    assert result["place_request"]["pending_slot"] == "route_retry"
    assert [name for name, _ in calls] == ["plan_route"]


def test_confirmed_goal_survives_task_serialization_before_retry(monkeypatch):
    from agents.state_models import PlaceRequestState

    _setup(monkeypatch, replan_result="unavailable")
    result = planner.run_agent("就去这里，全程风景优先", context={
        "place_request": _pending(), "selected_poi_id": FOOD["poi_id"],
    })
    saved = PlaceRequestState.model_validate(result["place_request"]).model_dump(mode="json")
    assert saved["confirmed_strategy_override"] == "scenery"
    retried = planner.run_agent("重试", context={"place_request": saved})
    assert retried["needs_input"] is True and retried.get("route") is None
    assert retried["place_request"]["confirmed_strategy_override"] == "scenery"


def test_unscoped_slope_and_global_goal_are_preserved_without_retry_loop(monkeypatch):
    from agents.state_models import PlaceRequestState

    calls = _setup(monkeypatch)
    original = _pending()
    first = planner.run_agent("就去这里，少爬坡", context={
        "place_request": original, "selected_poi_id": FOOD["poi_id"],
    })
    saved = PlaceRequestState.model_validate(first["place_request"]).model_dump(mode="json")
    second = planner.run_agent("全程风景优先", context={"place_request": saved})
    assert second["needs_input"] is True and second.get("route") is None
    assert second["place_request"]["unresolved_route_requirements"] == "就去这里，少爬坡"
    assert second["place_request"]["selected_poi_id"] == FOOD["poi_id"]
    assert second["place_request"]["confirmed_strategy_override"] == "scenery"
    assert second["place_request"]["constraints"]["avoid_slope"] is False
    assert second["place_request"]["completed_legs"] == original["completed_legs"]
    assert "不支持分段避坡" in second["message"] and "重试" not in second["message"]
    assert calls == []


@pytest.mark.parametrize("extra", ["不吃辣", "需要安静", "再去图书馆"])
def test_unhandled_compound_request_is_preserved_instead_of_dropped(monkeypatch, extra):
    from agents.state_models import PlaceRequestState

    calls = _setup(monkeypatch)
    query = "就去这里，全程风景优先，" + extra
    first = planner.run_agent(query, context={
        "place_request": _pending(), "selected_poi_id": FOOD["poi_id"],
    })
    assert first["needs_input"] is True and first.get("route") is None
    assert first["place_request"]["unresolved_route_requirements"] == query
    assert first["place_request"]["selected_poi_id"] == FOOD["poi_id"]
    assert first["place_request"]["confirmed_strategy_override"] == "scenery"
    saved = PlaceRequestState.model_validate(first["place_request"]).model_dump(mode="json")
    second = planner.run_agent("全程风景优先", context={"place_request": saved})
    assert second["needs_input"] is True and second.get("route") is None
    assert second["place_request"]["unresolved_route_requirements"] == query
    assert "附加要求" in second["message"] and calls == []


@pytest.mark.parametrize("change", ["不走台阶", "允许走台阶"])
def test_changed_step_constraint_is_preserved_without_retry_loop(monkeypatch, change):
    from agents.state_models import PlaceRequestState

    calls = _setup(monkeypatch)
    pending = _pending()
    if change == "不走台阶":
        pending["constraints"] = {}
        pending["completed_legs"][0]["route_state"]["hard_constraints"] = {}
        snapshot = pending["result_dependencies"]["candidate_snapshot"]
        pending["result_dependencies"] = capture_dependencies(pending["completed_legs"], task_revision=0)
        pending["result_dependencies"]["candidate_snapshot"] = snapshot
    original = deepcopy(pending)
    query = "就去这里，全程风景优先，" + change
    first = planner.run_agent(query, context={"place_request": pending, "selected_poi_id": FOOD["poi_id"]})
    assert first["needs_input"] is True and first.get("route") is None
    assert first["place_request"]["unresolved_route_requirements"] == query
    assert first["place_request"]["selected_poi_id"] == FOOD["poi_id"]
    assert first["place_request"]["confirmed_strategy_override"] == "scenery"
    assert first["place_request"]["constraints"] == original["constraints"]
    assert first["place_request"]["completed_legs"] == original["completed_legs"]
    saved = PlaceRequestState.model_validate(first["place_request"]).model_dump(mode="json")
    second = planner.run_agent("全程风景优先", context={"place_request": saved})
    assert second["needs_input"] is True and second.get("route") is None
    assert second["place_request"]["unresolved_route_requirements"] == query
    assert "通行约束" in second["message"] and "重试" not in second["message"]
    assert calls == []


def test_selected_place_name_is_not_interpreted_as_a_route_preference(monkeypatch):
    calls = _setup(monkeypatch)
    selected_name = "看风景食堂"
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": START["poi_id"], "name": START["name"]},
        {"id": SPORT["poi_id"], "name": SPORT["name"]},
        {"id": FOOD["poi_id"], "name": selected_name},
    ])
    original_tool = planner.agent_tools.execute_tool

    def tool(name, args, context):
        result, artifact = original_tool(name, args, context)
        if isinstance(artifact, dict) and isinstance(artifact.get("route"), dict):
            route = artifact["route"]
            route["end_name"] = selected_name
            route["route_state"]["end"]["name"] = selected_name
        return result, artifact

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tool)
    result = planner.run_agent("去看风景食堂，全程风景优先", context={
        "place_request": _pending(), "selected_poi_id": FOOD["poi_id"],
    })
    assert result["response_kind"] == "route"
    assert result["route"]["route_state"]["end"]["name"] == selected_name
    assert [name for name, _ in calls] == ["plan_route", "plan_via_route"]


def test_real_routing_recomputes_both_legs_for_confirmed_scenery_goal(monkeypatch):
    import networkx as nx
    from agents.routing_policy import select_route_strategy
    from spatial import network, result_verifier

    graph = nx.MultiDiGraph()
    coords = {1: (114.3600, 30.5300), 2: (114.3610, 30.5300),
              3: (114.3620, 30.5300), 4: (114.3605, 30.5302),
              5: (114.3615, 30.5302)}
    for node, (lng, lat) in coords.items():
        graph.add_node(node, x=lng, y=lat)
    for u, v, length, scenery in ((1, 2, 100, 1), (2, 3, 100, 1),
                                  (1, 4, 55, 5), (4, 2, 55, 5),
                                  (2, 5, 55, 5), (5, 3, 55, 5)):
        graph.add_edge(u, v, 0, length=length, highway="residential",
                       slope_level=1, scenery_level=scenery)

    def coord(node):
        lng, lat = coords[node]
        return {"type": "coord", "name": str(node), "coordinates": {"lng": lng, "lat": lat}}

    shortest = select_route_strategy(explicit_strategy="shortest").as_dict()
    scenery = select_route_strategy(query="全程风景优先", explicit_strategy="scenery").as_dict()

    def prior(start, end, strategy):
        return {"route_state": build_route_state(
            route_kind="direct", original_query="行程", start=coord(start), end=coord(end),
            travel_mode="walk", hard_constraints={"avoid_steps": True}, strategy=strategy,
            data_version="data-test", road_condition_version=EVENT_VERSION)}

    monkeypatch.setattr(planner.agent_tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(planner.agent_tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr(network, "get_annotation_coverage_rate", lambda: 1.0)
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [])
    monkeypatch.setattr(planner.agent_tools, "_data_version_for_graph", lambda _: "data-test")
    monkeypatch.setattr(route_state, "current_data_version", lambda *_: "data-test")
    monkeypatch.setattr(result_verifier, "current_data_version", lambda *_: "data-test")
    result, error = planner._replan_changed_place_route(
        [prior(1, 2, shortest), prior(2, 3, scenery)],
        {"query": "全程风景优先", "task_constraints": {"avoid_steps": True},
         "confirmed_strategy_override": "scenery"},
    )
    assert error is None
    assert result["route_verification"]["status"] == "pass"
    assert result["route_state"]["strategy"]["name"] == "scenery"
    assert result["route_state"]["hard_constraints"] == {"avoid_steps": True}
    assert [tuple(edge) for edge in result["recommended_edge_ids"]] == [
        (1, 4, 0), (4, 2, 0), (2, 5, 0), (5, 3, 0)]
    assert [tuple(edge) for edge in result["shortest_edge_ids"]] == [(1, 2, 0), (2, 3, 0)]
