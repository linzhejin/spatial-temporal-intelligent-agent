from unittest.mock import patch

from agents import planner


def test_activity_answer_preserves_follow_up_and_applies_new_mode_and_constraint():
    first = planner.run_agent("我在工学部，想先打球再吃饭")
    assert first["response_kind"] == "clarify"

    candidate = {"poi_id": "poi_206", "name": "工学部篮球场",
                 "subcategory": "field", "activities": ["basketball"], "distance_m": 120}
    with patch.object(planner.agent_tools, "execute_tool", return_value=(
        {"candidates": [candidate], "count": 1, "start_name": "工学部"},
        {"candidates": [candidate]},
    )) as execute:
        answer = planner.run_agent(
            "篮球，骑行，不走台阶", context={"place_request": first["place_request"]},
        )

    assert answer["response_kind"] == "candidates"
    assert answer["place_request"]["start"] == first["place_request"]["start"]
    assert answer["place_request"]["follow_up_requests"] == first["place_request"]["follow_up_requests"]
    assert answer["place_request"]["mode"] == "bike"
    assert answer["place_request"]["constraints"]["avoid_steps"] is True
    assert execute.call_args.args[1]["mode"] == "bike"


def test_composite_sport_and_meal_remains_pending_after_first_route_and_finishes_after_second():
    first = planner.run_agent("我在工学部，想先打球再吃饭")
    sport = {"poi_id": "poi_206", "name": "工学部篮球场", "activities": ["basketball"],
             "subcategory": "field", "distance_m": 120}
    meal = {"poi_id": "poi_310", "name": "工学部食堂", "activities": ["meal"],
            "subcategory": "canteen", "distance_m": 260}

    def search_sport(name, args, _ctx):
        assert name == "search_poi_candidates"
        return {"candidates": [sport], "start_name": "工学部"}, {"candidates": [sport]}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=search_sport):
        sports = planner.run_agent("篮球", context={"place_request": first["place_request"]})
    assert sports["response_kind"] == "candidates"

    def first_stage(name, args, _ctx):
        if name == "plan_route":
            return {}, {"route": {"recommended": [[114.35, 30.54], [114.36, 30.54]],
                                  "recommended_edge_ids": ["first-edge"],
                                  "distance_m": 120, "duration_min": 2,
                                  "start_name": "工学部", "end_name": "工学部篮球场",
                                  "mode": "walk"}}
        assert name == "search_poi_candidates"
        assert args["activities"] == ["meal"]
        assert args["start"] == {"type": "poi", "name": "工学部篮球场"}
        return {"candidates": [meal], "start_name": "工学部篮球场"}, {"candidates": [meal]}

    with patch.object(planner.agent_tools, "load_pois", return_value=[
        {"id": "poi_206", "name": "工学部篮球场"},
        {"id": "poi_310", "name": "工学部食堂"},
    ]), patch.object(planner.agent_tools, "execute_tool", side_effect=first_stage):
        routed_sport = planner.run_agent("去这个篮球场", context={
            "place_request": sports["place_request"], "selected_poi_id": "poi_206",
        })
    assert routed_sport["response_kind"] == "route"
    assert routed_sport["needs_input"] is True
    assert [item["poi_id"] for item in routed_sport["candidates"]] == ["poi_310"]
    assert routed_sport["place_request"]["candidate_poi_ids"] == ["poi_310"]
    assert routed_sport["place_request"]["start"] == {"type": "poi", "name": "工学部篮球场"}

    def second_stage(name, args, _ctx):
        assert name == "plan_route"
        assert args["start"] == {"type": "poi", "name": "工学部篮球场"}
        assert args["end"]["poi_id"] == "poi_310"
        return {}, {"route": {"recommended": [[114.36, 30.54], [114.37, 30.54]],
                              "recommended_edge_ids": ["second-edge"],
                              "distance_m": 260, "duration_min": 4,
                              "start_name": "工学部篮球场", "end_name": "工学部食堂",
                              "mode": "walk"}}

    with patch.object(planner.agent_tools, "load_pois", return_value=[
        {"id": "poi_310", "name": "工学部食堂"},
    ]), patch.object(planner.agent_tools, "execute_tool", side_effect=second_stage):
        routed_meal = planner.run_agent("去这个食堂", context={
            "place_request": routed_sport["place_request"], "selected_poi_id": "poi_310",
        })
    assert routed_meal["response_kind"] == "route"
    assert routed_meal.get("needs_input") is not True
    assert routed_meal["place_request"] is None
    assert routed_meal["route"]["recommended"] == [
        [114.35, 30.54], [114.36, 30.54], [114.37, 30.54],
    ]
    assert routed_meal["route"]["recommended_edge_ids"] == ["first-edge", "second-edge"]
    assert routed_meal["route"]["distance_m"] == 380
    assert routed_meal["route"]["duration_min"] == 6
    assert routed_meal["route"]["start_name"] == "工学部"
    assert routed_meal["route"]["end_name"] == "工学部食堂"
    assert len(routed_meal["route"]["legs"]) == 2


def test_nearest_first_stage_keeps_meal_follow_up_and_route_constraints():
    sport = {"poi_id": "poi_206", "name": "工学部篮球场",
             "activities": ["basketball"], "distance_m": 120}
    meal = {"poi_id": "poi_310", "name": "工学部食堂",
            "activities": ["meal"], "distance_m": 260}
    calls = []

    def execute(name, args, _ctx):
        calls.append((name, args))
        if name == "search_poi_candidates":
            candidates = [meal] if args["activities"] == ["meal"] else [sport]
            return {"candidates": candidates, "start_name": "工学部"}, {"candidates": candidates}
        assert name == "plan_route"
        return {}, {"route": {"recommended": [[114.36, 30.54]],
                              "start_name": "工学部", "end_name": "工学部篮球场",
                              "mode": "walk"}}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("我在工学部，带我去最近的篮球场，不走台阶、少爬坡，然后吃饭")

    assert result["response_kind"] == "route"
    assert result["needs_input"] is True
    assert result["place_request"]["original_query"] == "吃饭"
    assert result["place_request"]["start"] == {"type": "poi", "name": "工学部篮球场"}
    assert [candidate["poi_id"] for candidate in result["candidates"]] == ["poi_310"]
    assert calls[0][1]["constraints"]["avoid_steps"] is True
    assert calls[1][1]["constraints"]["avoid_steps"] is True
    assert calls[1][1]["constraints"]["slope"] == "avoid"
    assert calls[2][1]["constraints"]["avoid_steps"] is True


def test_nearest_flat_route_preserves_saved_step_avoidance():
    court = {"poi_id": "poi_206", "name": "工学部篮球场",
             "activities": ["basketball"], "distance_m": 120}
    saved = {"original_query": "从工学部去最近的篮球场", "activities": ["basketball"],
             "subcategories": ["field"], "pending_slot": "start",
             "start": {"type": "poi", "name": "工学部"},
             "constraints": {"avoid_steps": True}, "mode": "walk"}
    calls = []

    def execute(name, args, _ctx):
        calls.append((name, args))
        if name == "search_poi_candidates":
            return {"candidates": [court]}, {"candidates": [court]}
        return {}, {"route": {"recommended": [[114.36, 30.54]],
                             "start_name": "工学部", "end_name": "工学部篮球场",
                             "mode": "walk"}}

    with patch.object(planner.agent_tools, "execute_tool", side_effect=execute):
        result = planner.run_agent("带我去最近的篮球场，少爬坡", context={"place_request": saved})

    assert result["response_kind"] == "route"
    route_args = next(args for name, args in calls if name == "plan_route")
    assert route_args["constraints"] == {"avoid_steps": True, "slope": "avoid"}


def test_workflow_keeps_route_open_when_next_place_stage_needs_selection(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    from agents import workflow

    input_data = {"run_id": "run-composite-place", "query": "先打球再吃饭",
                  "owner_id": "owner", "history": [], "task": {"route_spec": {}},
                  "task_revision": 0}
    monkeypatch.setattr(workflow, "get_run_input", lambda _db, _run: input_data)
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: {"id": 1})
    pending = {"pending_slot": "selection", "original_query": "吃饭",
               "candidate_poi_ids": ["poi_310"]}
    candidates = [{"poi_id": "poi_310", "name": "工学部食堂"}]

    def agent(*_args, **_kwargs):
        return {"response_kind": "route", "message": "第一段路线已生成。",
                "route": {"recommended": [[114.36, 30.54]], "start_name": "工学部",
                          "end_name": "工学部篮球场", "mode": "walk"},
                "route_kind": "direct", "needs_input": True,
                "place_request": pending, "candidates": candidates,
                "clarify": {"question": "下一站选哪个食堂？", "options": []}}

    graph = workflow.build_agent_workflow(MemorySaver(), agent=agent)
    result = graph.invoke(
        {"run_id": "run-composite-place", "worker_id": "worker"},
        {"configurable": {"thread_id": "run-composite-place"}},
        context={"worker_id": "worker"},
    )["result"]
    assert result["task_type"] == "path_planning"
    assert result["execution_status"] == "needs_input"
    assert result["needs_input"] is True
    assert result["place_request"] == pending
    assert result["candidates"] == candidates
    assert result["clarify"]["question"] == "下一站选哪个食堂？"


def test_route_join_rejects_stale_versions_and_unconnected_geometry():
    from agents.route_state import build_route_state

    def leg(start, end, path, edge, version="roads-1", constraints=None):
        return {"recommended": path, "recommended_edge_ids": [edge],
                "start_name": start, "end_name": end, "mode": "walk",
                "route_state": build_route_state(
                    route_kind="direct", original_query=start + end,
                    start={"type": "poi", "name": start},
                    end={"type": "poi", "name": end}, travel_mode="walk",
                    hard_constraints=constraints or {}, strategy={"name": "shortest"},
                    data_version="data-1", road_condition_version=version,
                )}

    first = leg("起点", "球场", [[114.35, 30.54], [114.36, 30.54]], "edge-one")
    second = leg("球场", "食堂", [[114.36, 30.54], [114.37, 30.54]], "edge-two")
    assert planner._place_route_join_issue([first, second]) is None
    combined = planner._combined_place_route([first, second])
    assert combined["route_state"]["route_kind"] == "via"
    assert combined["route_state"]["via"] == first["route_state"]["end"]

    for part in (first, second):
        part.update(shortest_baseline_feasible=True,
                    shortest_length_m=80, shortest_duration_min=1.2)
        part["route_state"]["hard_constraints"] = {"max_duration_s": 120}
    limited = planner._combined_place_route([first, second])
    assert limited["shortest_baseline_feasible"] is False

    changed_roads = leg("球场", "食堂", second["recommended"], "edge-two", "roads-2")
    assert "版本不同" in planner._place_route_join_issue([first, changed_roads])
    disconnected = leg("球场", "食堂", [[114.365, 30.54], [114.37, 30.54]], "edge-two")
    assert "未在同一位置接通" in planner._place_route_join_issue([first, disconnected])
    different_constraints = leg("球场", "食堂", second["recommended"], "edge-two",
                                constraints={"avoid_steps": True})
    mixed = planner._combined_place_route([first, different_constraints])
    assert "route_state" not in mixed
    assert mixed["route_state_unavailable"]
