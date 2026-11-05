"""A changed activity must update candidates without losing the earlier trip."""

from copy import deepcopy

import pytest

from agents import planner
from agents.route_state import build_route_state, current_road_condition_version


def _leg(start, end, path, edge, *, distance, minutes):
    state = build_route_state(
        route_kind="direct", original_query="先打篮球再吃饭",
        start=start, end=end, travel_mode="walk",
        hard_constraints={"avoid_steps": True}, strategy={"name": "shortest"},
        data_version="data-test", road_condition_version=current_road_condition_version([]),
    )
    return {
        "recommended": path, "recommended_edge_ids": [edge],
        "distance_m": distance, "duration_min": minutes,
        "start_name": start["name"], "end_name": end["name"],
        "mode": "walk", "route_state": state, "route_verification": {"status": "pass"},
        "weather_input_sha256": "a" * 64,
    }


def _pending_trip():
    from agents.result_maintenance import capture_dependencies

    sport = {"type": "poi", "poi_id": "poi_206", "name": "工学部篮球场"}
    first = _leg(
        {"type": "poi", "name": "工学部"}, sport,
        [[114.35, 30.54], [114.36, 30.54]], [1, 2, 0], distance=120, minutes=2,
    )
    pending = {
        "request_id": "test-trip", "root_query": "先打篮球再吃饭，不走台阶",
        "original_query": "吃饭", "activities": ["meal"],
        "subcategories": ["canteen", "restaurant", "fastfood"],
        "pending_slot": "selection", "start": sport, "completed_legs": [first],
        "follow_up_requests": [], "candidate_poi_ids": ["poi_310"],
        "selected_poi_id": None, "mode": "walk", "strategy": "shortest",
        "constraints": {"avoid_steps": True},
    }
    pending["result_dependencies"] = capture_dependencies([first], task_revision=0)
    pending["result_dependencies"]["candidate_snapshot"] = {
        "data_version": "data-test", "road_condition_version": current_road_condition_version([]),
    }
    return pending


@pytest.mark.parametrize("revision_query", [
    "不吃食堂，换成餐厅",
    "不想吃食堂，想吃餐厅",
    "我不想吃食堂了，想去餐厅",
])
def test_dining_revision_updates_candidates_and_retains_complete_route(monkeypatch, revision_query):
    pending = _pending_trip()
    saved = deepcopy(pending)
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr("agents.route_state.current_data_version", lambda: "data-test")
    restaurant = {"poi_id": "poi_311", "name": "工学部餐厅",
                  "activities": ["restaurant_meal"], "subcategory": "restaurant",
                  "distance_m": 300}
    calls = []

    def tools(name, args, _context):
        calls.append((name, deepcopy(args)))
        if name == "search_poi_candidates":
            return {"candidates": [restaurant], "start_name": "工学部篮球场", "data_version": "data-test",
                    "road_condition_version": current_road_condition_version([])}, {"candidates": [restaurant]}
        assert name == "plan_route"
        return {}, {"route": _leg(args["start"], args["end"],
                                 [[114.36, 30.54], [114.37, 30.54]], [2, 3, 0],
                                 distance=300, minutes=5)}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tools)
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": "poi_311", "name": "工学部餐厅"},
    ])

    changed = planner.run_agent(revision_query, context={"place_request": pending})
    assert pending == saved  # accepting a revision must not mutate its source
    assert calls[0][0] == "search_poi_candidates"
    assert calls[0][1]["subcategories"] == ["restaurant"]
    assert calls[0][1]["activities"] == ["restaurant_meal"]
    assert calls[0][1]["start"]["poi_id"] == "poi_206"
    assert calls[0][1]["constraints"]["avoid_steps"] is True
    assert [item["poi_id"] for item in changed["candidates"]] == ["poi_311"]
    assert changed["place_request"]["candidate_poi_ids"] == ["poi_311"]
    assert changed["place_request"]["completed_legs"] == saved["completed_legs"]

    final = planner.run_agent("就去这里", context={
        "place_request": changed["place_request"], "selected_poi_id": "poi_311",
    }, proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert final["response_kind"] == "route"
    assert final["route"]["recommended_edge_ids"] == [[1, 2, 0], [2, 3, 0]]
    assert final["route"]["distance_m"] == 420
    assert final["route"]["duration_min"] == 7
    assert final["route"]["end_name"] == "工学部餐厅"
    assert "工学部餐厅" in final["message"]
    assert final["result_maintenance"]["prefix_action"] == "retained"
    # Once delivered, the next explicit activity revision must still be able
    # to recover the verified prior stage without treating the whole trip as a
    # fresh, independent dining request.
    completed = final["completed_place_request"]
    assert completed["completed_legs"] == saved["completed_legs"]
    assert completed["start"]["poi_id"] == "poi_206"
    assert completed["candidate_poi_ids"] == []


def test_unsupported_dining_revision_does_not_reuse_old_candidates(monkeypatch):
    pending = _pending_trip()
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    changed = planner.run_agent("把用餐改成素食", context={"place_request": pending})
    assert changed["response_kind"] == "clarify"
    assert changed["needs_input"] is True
    assert not changed.get("candidates")
    assert changed["place_request"]["completed_legs"] == pending["completed_legs"]
    assert changed["place_request"]["candidate_poi_ids"] == []
    assert calls == []


@pytest.mark.parametrize("stale_field", ["data_version", "road_condition_version", "missing"])
def test_environment_change_researches_candidates_before_accepting_an_old_choice(monkeypatch, stale_field):
    pending = _pending_trip()
    event_version = current_road_condition_version([])
    snapshot = {"data_version": "data-test", "road_condition_version": event_version}
    if stale_field == "missing":
        pending["result_dependencies"].pop("candidate_snapshot", None)
    else:
        snapshot[stale_field] = "old-version"
        pending["result_dependencies"]["candidate_snapshot"] = snapshot
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr("agents.route_state.current_data_version", lambda: "data-test")
    calls = []
    updated = {"poi_id": "poi_311", "name": "工学部餐厅", "subcategory": "restaurant", "distance_m": 300}

    def tools(name, args, _context):
        calls.append((name, deepcopy(args)))
        assert name == "search_poi_candidates"
        return {"data_version": "data-test", "road_condition_version": event_version,
                "start_name": "工学部篮球场"}, {"candidates": [updated]}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tools)
    answer = planner.run_agent("就去这里", context={"place_request": pending, "selected_poi_id": "poi_310"})
    assert [call[0] for call in calls] == ["search_poi_candidates"]
    assert answer["response_kind"] == "candidates"
    assert answer.get("route") is None
    assert answer["place_request"]["candidate_poi_ids"] == ["poi_311"]
    assert answer["place_request"]["completed_legs"] == pending["completed_legs"]
    assert "变化" in answer["message"] or "重新" in answer["message"]


def test_first_stage_candidates_are_researched_when_graph_changes(monkeypatch):
    pending = _pending_trip()
    pending["completed_legs"] = []
    pending["result_dependencies"]["candidate_snapshot"]["data_version"] = "old-data"
    current_event = current_road_condition_version([])
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr("agents.route_state.current_data_version", lambda: "data-test")
    calls = []
    candidate = {"poi_id": "poi_311", "name": "工学部餐厅", "distance_m": 300}

    def tool(name, args, _context):
        calls.append(name)
        assert name == "search_poi_candidates"
        return {"data_version": "data-test", "road_condition_version": current_event}, {"candidates": [candidate]}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tool)
    answer = planner.run_agent("就去这里", context={"place_request": pending, "selected_poi_id": "poi_310"})
    assert calls == ["search_poi_candidates"]
    assert answer["response_kind"] == "candidates"
    assert answer["place_request"]["candidate_poi_ids"] == ["poi_311"]


def test_activity_revision_does_not_accept_a_simultaneous_stale_selection(monkeypatch):
    pending = _pending_trip()
    restaurant = {"poi_id": "poi_311", "name": "工学部餐厅", "distance_m": 300}
    calls = []

    def tools(name, args, _context):
        calls.append(name)
        assert name == "search_poi_candidates"
        return {"candidates": [restaurant], "start_name": "工学部篮球场"}, {"candidates": [restaurant]}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tools)
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": "poi_310", "name": "工学部食堂"},
    ])
    changed = planner.run_agent("换成餐厅", context={
        "place_request": pending, "selected_poi_id": "poi_310",
    })
    assert calls == ["search_poi_candidates"]
    assert changed["place_request"]["candidate_poi_ids"] == ["poi_311"]


@pytest.mark.parametrize("query", [
    "把用餐改成餐厅，然后再去图书馆",
    "把用餐改成餐厅，不吃辣",
    "换成素食餐厅",
    "把前面的运动改成羽毛球",
    "把用餐改成餐厅，骑行改为步行",
])
def test_compound_or_ambiguous_revision_is_not_silently_discarded(monkeypatch, query):
    pending = _pending_trip()
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    answer = planner.run_agent(query, context={"place_request": pending})
    assert answer["response_kind"] == "clarify"
    assert answer["needs_input"] is True
    assert answer["place_request"]["unresolved_revision"] == query
    assert answer["place_request"]["completed_legs"] == pending["completed_legs"]
    assert answer["place_request"]["candidate_poi_ids"] == []
    assert not calls


def test_old_candidate_is_rejected_while_a_revision_is_unresolved(monkeypatch):
    pending = _pending_trip()
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    answer = planner.run_agent("改成素食", context={"place_request": pending})
    stale = planner.run_agent("就去这里", context={
        "place_request": answer["place_request"], "selected_poi_id": "poi_310",
    })
    assert stale["response_kind"] == "clarify"
    assert stale["needs_input"] is True
    assert stale.get("route") is None
    assert not calls


def test_unresolved_revision_requires_explicit_choice_before_searching(monkeypatch):
    pending = _pending_trip()
    calls = []
    restaurant = {"poi_id": "poi_311", "name": "工学部餐厅", "distance_m": 300}

    def tools(name, args, _context):
        calls.append((name, args))
        return {"candidates": [restaurant], "start_name": "工学部篮球场"}, {"candidates": [restaurant]}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tools)
    unresolved = planner.run_agent("把用餐改成餐厅，不吃辣", context={"place_request": pending})
    still_unresolved = planner.run_agent("餐厅", context={"place_request": unresolved["place_request"]})
    assert still_unresolved["response_kind"] == "clarify"
    assert "刷新" in still_unresolved["message"]
    assert "重新说明完整行程" not in still_unresolved["message"]
    assert not calls
    accepted = planner.run_agent("只改成餐厅", context={"place_request": still_unresolved["place_request"]})
    assert accepted["response_kind"] == "candidates"
    assert accepted["place_request"]["unresolved_revision"] is None
    assert calls[0][1]["subcategories"] == ["restaurant"]


def test_ambiguous_combined_revision_keeps_the_previous_travel_mode(monkeypatch):
    pending = _pending_trip()
    pending["mode"] = "bike"
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_args: pytest.fail("must ask first"))
    answer = planner.run_agent("把用餐改成餐厅，骑行改为步行", context={"place_request": pending})
    assert answer["response_kind"] == "clarify"
    assert answer["place_request"]["mode"] == "bike"


def test_snapshot_check_rejects_changed_graph_even_when_events_match(monkeypatch):
    from agents import route_state

    pending = _pending_trip()
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-new")
    assert planner._place_route_snapshot_issue(pending["completed_legs"]) == "当前路网或地点数据已更新"


def test_old_or_tampered_dependencies_cannot_authorize_selective_reuse():
    from agents.result_maintenance import may_keep_prefix

    pending = _pending_trip()
    legs = pending["completed_legs"]
    dependencies = pending["result_dependencies"]
    assert may_keep_prefix(dependencies, legs, mode="walk", next_leg=legs[0]) is True
    assert may_keep_prefix(dependencies, legs, mode="walk") is False
    assert may_keep_prefix(None, legs, mode="walk") is False
    assert may_keep_prefix(dependencies, legs, mode="bike") is False
    edited = deepcopy(legs)
    edited[0]["recommended_edge_ids"][0][1] = 99
    assert may_keep_prefix(dependencies, edited, mode="walk") is False
    edited = deepcopy(legs)
    edited[0]["route_verification"] = {"status": "unknown"}
    assert may_keep_prefix(dependencies, edited, mode="walk") is False


def test_prefix_reuse_requires_matching_recorded_route_weather():
    from agents.result_maintenance import capture_dependencies, may_keep_prefix, weather_input_fingerprint

    pending = _pending_trip()
    prefix = pending["completed_legs"]
    dry = weather_input_fingerprint({"weather": "晴", "temperature": 22})
    wet = weather_input_fingerprint({"weather": "小雨", "temperature": 22})
    prefix[0]["weather_input_sha256"] = dry
    record = capture_dependencies(prefix, task_revision=0)
    next_leg = {"weather_input_sha256": dry}
    assert dry and wet and dry != wet
    assert may_keep_prefix(record, prefix, mode="walk", next_leg=next_leg)
    next_leg["weather_input_sha256"] = wet
    assert not may_keep_prefix(record, prefix, mode="walk", next_leg=next_leg)
    next_leg["weather_input_sha256"] = None
    assert not may_keep_prefix(record, prefix, mode="walk", next_leg=next_leg)
    prefix[0].pop("weather_input_sha256")
    assert capture_dependencies(prefix, task_revision=0) is None
    assert weather_input_fingerprint(None) is None
    assert weather_input_fingerprint({}) is None


@pytest.mark.parametrize("policy,drop_dependencies", [("full_recompute", False), ("conservative", True)])
def test_full_policy_and_legacy_records_recompute_the_whole_trip(monkeypatch, policy, drop_dependencies):
    pending = _pending_trip()
    if drop_dependencies:
        # A legacy prefix has no reuse proof; its candidates have already
        # been freshly searched in the current environment.
        pending["result_dependencies"] = {"candidate_snapshot": {
            "data_version": "data-test", "road_condition_version": current_road_condition_version([])}}
    monkeypatch.setattr(planner.config, "RESULT_MAINTENANCE_POLICY", policy)
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr("agents.route_state.current_data_version", lambda: "data-test")
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [{"id": "poi_311", "name": "工学部餐厅"}])
    route = _leg(pending["start"], {"type": "poi", "poi_id": "poi_311", "name": "工学部餐厅"},
                 [[114.36, 30.54], [114.37, 30.54]], [2, 3, 0], distance=300, minutes=5)
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_args: ({}, {"route": route}))
    replans = []
    recomputed = {"mode": "walk", "start_name": "工学部", "end_name": "工学部餐厅",
                  "recommended": [[114.35, 30.54], [114.37, 30.54]], "recommended_edge_ids": [[9, 10, 0]],
                  "distance_m": 450, "duration_min": 8, "route_verification": {"status": "pass"}}

    def replan(legs, _context):
        replans.append(deepcopy(legs))
        return deepcopy(recomputed), None

    monkeypatch.setattr(planner, "_replan_changed_place_route", replan)
    pending["candidate_poi_ids"] = ["poi_311"]
    result = planner.run_agent("就去这里", context={"place_request": pending, "selected_poi_id": "poi_311"},
                               proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert len(replans) == 1
    assert result["route"]["recommended_edge_ids"] == [[9, 10, 0]]
    assert result["result_maintenance"]["prefix_action"] == "recomputed"


def test_changed_graph_forces_full_replan_even_when_prefix_has_evidence(monkeypatch):
    pending = _pending_trip()
    # Candidate search happened after the graph update; the older prefix did not.
    pending["result_dependencies"]["candidate_snapshot"]["data_version"] = "data-new"
    monkeypatch.setattr(planner.config, "RESULT_MAINTENANCE_POLICY", "conservative")
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr("agents.route_state.current_data_version", lambda: "data-new")
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [{"id": "poi_311", "name": "工学部餐厅"}])
    route = _leg(pending["start"], {"type": "poi", "poi_id": "poi_311", "name": "工学部餐厅"},
                 [[114.36, 30.54], [114.37, 30.54]], [2, 3, 0], distance=300, minutes=5)
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_args: ({}, {"route": route}))
    replans = []
    replanned = {"mode": "walk", "start_name": "工学部", "end_name": "工学部餐厅",
                 "recommended": [[114.35, 30.54], [114.37, 30.54]], "recommended_edge_ids": [[9, 10, 0]],
                 "distance_m": 450, "duration_min": 8, "route_verification": {"status": "pass"}}
    monkeypatch.setattr(planner, "_replan_changed_place_route", lambda legs, _context: (
        replans.append(deepcopy(legs)) or deepcopy(replanned), None,
    ))
    pending["candidate_poi_ids"] = ["poi_311"]
    result = planner.run_agent("就去这里", context={"place_request": pending, "selected_poi_id": "poi_311"},
                               proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert len(replans) == 1
    assert result["route"]["recommended_edge_ids"] == [[9, 10, 0]]
    assert result["result_maintenance"]["prefix_action"] == "recomputed"


def test_named_final_stop_does_not_reassemble_old_legs_after_a_full_replan(monkeypatch):
    pending = _pending_trip()
    pending["candidate_poi_ids"] = ["poi_311"]
    pending["follow_up_requests"] = [{"query": "图书馆", "destination_poi_id": "poi_400"}]
    monkeypatch.setattr(planner.config, "RESULT_MAINTENANCE_POLICY", "full_recompute")
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr("agents.route_state.current_data_version", lambda: "data-test")
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": "poi_311", "name": "工学部餐厅"}, {"id": "poi_400", "name": "图书馆"},
    ])

    def tools(name, args, _context):
        assert name == "plan_route"
        is_library = args["end"]["poi_id"] == "poi_400"
        return {}, {"route": _leg(args["start"], args["end"],
                                  [[114.37, 30.54], [114.38, 30.54]] if is_library else
                                  [[114.36, 30.54], [114.37, 30.54]],
                                  [3, 4, 0] if is_library else [2, 3, 0], distance=100, minutes=2)}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tools)
    replans = []

    def replan(legs, _context):
        replans.append(deepcopy(legs))
        end = "图书馆" if len(legs) == 3 else "工学部餐厅"
        return {"mode": "walk", "start_name": "工学部", "end_name": end,
                "recommended": [[114.35, 30.54], [114.38, 30.54]],
                "recommended_edge_ids": [[8, len(legs), 0]], "distance_m": 450,
                "duration_min": 8, "route_verification": {"status": "pass"}}, None

    monkeypatch.setattr(planner, "_replan_changed_place_route", replan)
    result = planner.run_agent("就去这里", context={"place_request": pending, "selected_poi_id": "poi_311"},
                               proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert [len(legs) for legs in replans] == [2, 3]
    assert result["route"]["recommended_edge_ids"] == [[8, 3, 0]]
    assert result["route"]["end_name"] == "图书馆"
    assert result["result_maintenance"]["prefix_action"] == "recomputed"


@pytest.mark.parametrize("final_weather", [None, "b" * 64])
def test_named_final_stop_with_unmatched_weather_replans_all_legs(monkeypatch, final_weather):
    pending = _pending_trip()
    pending["candidate_poi_ids"] = ["poi_311"]
    pending["follow_up_requests"] = [{"query": "图书馆", "destination_poi_id": "poi_400"}]
    monkeypatch.setattr(planner.config, "RESULT_MAINTENANCE_POLICY", "conservative")
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr("agents.route_state.current_data_version", lambda: "data-test")
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [
        {"id": "poi_311", "name": "工学部餐厅"}, {"id": "poi_400", "name": "图书馆"},
    ])

    def tool(name, args, _context):
        assert name == "plan_route"
        is_final = args["end"]["poi_id"] == "poi_400"
        route = _leg(args["start"], args["end"],
                     [[114.37, 30.54], [114.38, 30.54]] if is_final else
                     [[114.36, 30.54], [114.37, 30.54]],
                     [3, 4, 0] if is_final else [2, 3, 0], distance=100, minutes=2)
        if is_final:
            route["weather_input_sha256"] = final_weather
        return {}, {"route": route}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", tool)
    replans = []

    def replan(legs, _context):
        replans.append(deepcopy(legs))
        return {"mode": "walk", "start_name": "工学部", "end_name": "图书馆",
                "recommended": [[114.35, 30.54], [114.38, 30.54]],
                "recommended_edge_ids": [[8, 9, 0]], "distance_m": 450,
                "duration_min": 8, "route_verification": {"status": "pass"}}, None

    monkeypatch.setattr(planner, "_replan_changed_place_route", replan)
    result = planner.run_agent("就去这里", context={"place_request": pending, "selected_poi_id": "poi_311"},
                               proof_identity={"owner_id": "owner", "task_id": "task", "task_revision": 1})
    assert [len(legs) for legs in replans] == [3]
    assert result["route"]["recommended_edge_ids"] == [[8, 9, 0]]
    assert result["result_maintenance"]["prefix_action"] == "recomputed"


def test_workflow_recovers_a_completed_trip_only_for_an_explicit_activity_change(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    from agents import workflow

    pending = _pending_trip()
    seen = []
    data = {
        "run_id": "completed-trip-followup", "task_id": "new-task", "owner_id": "owner",
        "query": "把用餐改成餐厅", "history": [], "task": {"route_spec": {}},
        "task_revision": 0, "previous_completed_place_request": pending,
    }
    monkeypatch.setattr(workflow, "get_run_input", lambda *_: data)
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: {"id": 1})

    def agent(_query, *, context, **_kwargs):
        seen.append(context)
        return {"response_kind": "candidates", "message": "请选择新餐厅。",
                "candidates": [{"poi_id": "poi_311", "name": "餐厅"}],
                "place_request": pending, "needs_input": True}

    graph = workflow.build_agent_workflow(MemorySaver(), agent=agent)
    graph.invoke({"run_id": data["run_id"], "worker_id": "worker"},
                 {"configurable": {"thread_id": "explicit-activity-change"}},
                 context={"worker_id": "worker"})
    assert seen[-1]["place_request"] == pending

    data["query"] = "把路线改成骑行"
    graph.invoke({"run_id": data["run_id"], "worker_id": "worker"},
                 {"configurable": {"thread_id": "separate-route-change"}},
                 context={"worker_id": "worker"})
    assert "place_request" not in seen[-1]


def test_workflow_asks_for_endpoints_when_last_route_context_is_uncertain(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    from agents import workflow

    data = {"run_id": "uncertain-route", "task_id": "new-task", "owner_id": "owner",
            "query": "改成骑行", "history": [], "task": {"route_spec": {}}, "task_revision": 0,
            "previous_route_state": None, "route_context_uncertain": True}
    monkeypatch.setattr(workflow, "get_run_input", lambda *_: data)
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: {"id": 1})
    graph = workflow.build_agent_workflow(MemorySaver(), agent=lambda *_args, **_kwargs: pytest.fail("must clarify first"))
    result = graph.invoke({"run_id": data["run_id"], "worker_id": "worker"},
                          {"configurable": {"thread_id": "uncertain-route"}}, context={"worker_id": "worker"})
    assert result["result"]["response_kind"] == "clarify"
    assert "起点" in result["result"]["message"] and "终点" in result["result"]["message"]



def test_rejected_food_with_additional_constraint_is_not_partially_applied(monkeypatch):
    pending = _pending_trip()
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", None)
    query = "不想吃食堂，想吃餐厅，不吃辣"
    result = planner.run_agent(query, context={"place_request": pending})
    assert result["response_kind"] == "clarify"
    assert result["place_request"]["unresolved_revision"] == query
    assert result["place_request"]["completed_legs"] == pending["completed_legs"]
    assert result["place_request"]["candidate_poi_ids"] == []
    assert calls == []



def test_unsupported_non_dining_revision_keeps_prior_trip_for_clarification(monkeypatch):
    pending = _pending_trip()
    pending["activities"] = ["basketball"]
    pending["subcategories"] = ["field"]
    calls = []
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *args: calls.append(args))
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", None)
    query = "把运动改成羽毛球"
    answer = planner.run_agent(query, context={"place_request": pending})
    assert answer["response_kind"] == "clarify"
    assert answer["place_request"]["unresolved_revision"] == query
    assert answer["place_request"]["completed_legs"] == pending["completed_legs"]
    assert calls == []
