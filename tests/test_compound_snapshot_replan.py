from copy import deepcopy

import pytest

from agents import planner
from agents.route_state import build_route_state


def ref(name):
    return {"type": "poi", "name": name, "poi_id": "id_" + name}


def state(start, end, *, via=None, version="old", mode="walk", constraints=None):
    return build_route_state(route_kind="via" if via else "direct", original_query="行程",
        start=ref(start), end=ref(end), via=via, travel_mode=mode,
        hard_constraints=constraints or {"avoid_steps": True, "max_distance_m": 5000},
        strategy={"name": "shortest"}, data_version="data", road_condition_version=version)


def legs():
    via = {"type": "multi", "name": "B、C", "points": [ref("B"), ref("C")]}
    return [{"route_state": state("A", "D", via=via)},
            {"route_state": state("D", "E", version="new")}]


def test_snapshot_replan_retains_nested_order_identity_and_constraints(monkeypatch):
    values = legs()
    pois = [{"id": f"id_{x}", "name": x} for x in "ABCDE"]
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: pois)
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context",
                        lambda context: ({**context, "_route_graph_snapshot": object(),
                            "_route_conditions_snapshot": [], "_route_weather_snapshot": None}, None))
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    monkeypatch.setattr(planner.agent_tools, "_data_version_for_graph", lambda _: "data")
    calls = []
    def execute(name, args, ctx):
        calls.append((name, deepcopy(args), ctx))
        fresh = state("A", "E", via={"type": "multi", "name": "B、C、D",
                     "points": [ref("B"), ref("C"), ref("D")]}, version="fresh")
        route = {"route_state": fresh, "route_verification": {"status": "pass"}}
        return route, {"route": route}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    monkeypatch.setattr(planner.agent_tools.route_state_module, "current_road_condition_version", lambda _: "fresh")
    route, reason = planner._replan_changed_place_route(values, {"query": "先后去各处"})
    assert reason is None and route["route_state"]["end"]["poi_id"] == "id_E"
    assert calls[0][0] == "plan_via_route"
    assert [item["poi_id"] for item in calls[0][1]["via_points"]] == ["id_B", "id_C", "id_D"]
    assert calls[0][1]["constraints"] == {"avoid_steps": True, "max_distance_m": 5000}
    assert calls[0][2]["_route_conditions_snapshot"] == []


@pytest.mark.parametrize("variation", ["mixed_mode", "different_constraints", "tour", "repeated", "ambiguous"])
def test_snapshot_replan_rejects_unprovable_semantics_without_tool_call(monkeypatch, variation):
    values = legs()
    if variation == "mixed_mode":
        values[1]["route_state"]["travel_mode"] = "bike"
    elif variation == "different_constraints":
        values[1]["route_state"]["hard_constraints"] = {}
    elif variation == "tour":
        values[0]["route_state"]["route_kind"] = "tour"
    elif variation == "repeated":
        values[0]["route_state"]["via"]["points"][1] = ref("B")
    else:
        values[0]["route_state"]["via"]["points"][0] = {"type": "poi", "name": "B"}
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda:
        [{"id": f"id_{x}", "name": x} for x in "ABCDE"] + [{"id": "other", "name": "B"}])
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_: pytest.fail("must not plan"))
    route, reason = planner._replan_changed_place_route(values, {})
    assert route is None and reason


def test_snapshot_replan_refuses_lost_task_constraint(monkeypatch):
    values = legs()
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [{"id": f"id_{x}", "name": x} for x in "ABCDE"])
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_: pytest.fail("must not plan"))
    route, reason = planner._replan_changed_place_route(values, {"task_constraints": {
        "avoid_steps": True, "max_distance_m": 5000, "require_verified_accessibility": True}})
    assert route is None and reason


def test_return_to_start_is_not_silently_dropped(monkeypatch):
    import networkx as nx
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.36, y=30.53)
    graph.add_node(2, x=114.361, y=30.53)
    graph.add_edge(1, 2, 0, length=100, highway="residential")
    graph.add_edge(2, 1, 0, length=100, highway="residential")
    values = [{"route_state": state("A", "B")}, {"route_state": state("B", "A", version="new")}]
    pois = [{"id": f"id_{name}", "name": name, "coordinates": {"lng": graph.nodes[node]["x"], "lat": 30.53}}
            for node, name in [(1, "A"), (2, "B")]]
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: pois)
    monkeypatch.setattr(planner.agent_tools, "navigation_wgs", lambda poi, _: (
        poi["coordinates"]["lng"], poi["coordinates"]["lat"]))
    monkeypatch.setattr(planner.agent_tools, "find_poi_ambiguous", lambda name:
        (next((poi for poi in pois if poi["name"] == name), None), []))
    monkeypatch.setattr(planner.agent_tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(planner.agent_tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    route, reason = planner._replan_changed_place_route(values, {})
    assert reason is None
    assert route["recommended_edge_ids"] == [(1, 2, 0), (2, 1, 0)]
    assert route["route_state"]["start"]["poi_id"] == "id_A"
    assert route["route_state"]["via"]["poi_id"] == "id_B"
    assert route["route_state"]["end"]["poi_id"] == "id_A"


def test_more_than_ten_ordered_stops_is_not_silently_truncated(monkeypatch):
    letters = list("ABCDEFGHIJKLM")
    via = {"type": "multi", "name": "途经", "points": [ref(name) for name in letters[1:11]]}
    values = [{"route_state": state("A", "L", via=via)},
              {"route_state": state("L", "M", version="new")}]
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [{"id": f"id_{x}", "name": x} for x in letters])
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_: pytest.fail("must not plan"))
    route, reason = planner._replan_changed_place_route(values, {})
    assert route is None and reason


def test_combined_route_rejects_total_budget_and_missing_constraints():
    constraints = {"avoid_steps": True, "max_distance_m": 500, "max_duration_s": 600}
    route_legs = [{"route_state": state("A", "B", constraints=constraints),
                   "recommended_length_m": 400, "duration_min": 5},
                  {"route_state": state("B", "C", constraints=constraints),
                   "recommended_length_m": 400, "duration_min": 5}]
    assert planner._place_route_constraint_issue(route_legs, constraints)
    route_legs[0]["recommended_length_m"] = 200
    route_legs[1]["recommended_length_m"] = 200
    assert planner._place_route_constraint_issue(route_legs, constraints) is None
    route_legs[1]["route_state"]["hard_constraints"] = {"avoid_steps": True}
    assert planner._place_route_constraint_issue(route_legs, constraints)
    route_legs[1]["route_state"]["hard_constraints"] = constraints
    route_legs[1]["route_state"]["strategy"] = {"name": "flat", "source": "explicit_nl"}
    assert planner._place_route_constraint_issue(route_legs, constraints)
    route_legs[1]["route_state"]["strategy"] = {"name": "shortest", "source": "explicit_nl"}
    assert planner._place_route_constraint_issue(route_legs, constraints) is None


def test_route_retry_state_round_trips_without_extra_fields():
    from agents.state_models import PlaceRequestState
    pending = PlaceRequestState.model_validate({"request_id": "r", "original_query": "从A到B",
        "pending_slot": "route_retry", "completed_legs": [{"route_state": state("A", "B")}],
        "follow_up_requests": [{"destination_poi_id": "id_C"}]})
    recovered = PlaceRequestState.model_validate_json(pending.model_dump_json())
    assert recovered.pending_slot == "route_retry"
    assert recovered.follow_up_requests[0]["destination_poi_id"] == "id_C"


def test_combined_route_state_keeps_nested_via_identities():
    values = legs()
    values[1]["route_state"]["road_condition_version"] = "old"
    for item in values:
        item.update({"mode": "walk", "recommended": [], "steps": [], "pois": [],
                     "route_events": [], "road_condition_overview": [],
                     "route_verification": {"status": "pass"}})
    combined = planner._combined_place_route(values)
    assert [x["poi_id"] for x in combined["route_state"]["via"]["points"]] == ["id_B", "id_C", "id_D"]


def test_snapshot_replan_rejects_event_change_during_planning(monkeypatch):
    values = legs()
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [{"id": f"id_{x}", "name": x} for x in "ABCDE"])
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context",
        lambda context: ({**context, "_route_graph_snapshot": object(), "_route_conditions_snapshot": []}, None))
    monkeypatch.setattr(planner.agent_tools, "_data_version_for_graph", lambda _: "data")
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [{"id": "changed"}])
    monkeypatch.setattr(planner.agent_tools.route_state_module, "current_road_condition_version",
                        lambda events: "changed" if events else "fresh")
    fresh = state("A", "E", via={"type": "multi", "name": "B、C、D",
                  "points": [ref("B"), ref("C"), ref("D")]}, version="fresh")
    route = {"route_state": fresh, "route_verification": {"status": "pass"}}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_: (route, {"route": route}))
    result, reason = planner._replan_changed_place_route(values, {})
    assert result is None and reason


def test_snapshot_replan_uses_real_route_tool_on_isolated_graph(monkeypatch):
    import networkx as nx
    graph = nx.MultiDiGraph()
    for node in range(1, 5):
        graph.add_node(node, x=114.36 + (node - 1) * .001, y=30.53)
    for node in range(1, 4):
        graph.add_edge(node, node + 1, 0, length=100, highway="residential")
        graph.add_edge(node + 1, node, 0, length=100, highway="residential")
    def coord(node):
        return {"type": "coord", "name": str(node),
                "coordinates": {"lng": graph.nodes[node]["x"], "lat": graph.nodes[node]["y"]}}
    def route_state(start, end, via, version):
        return build_route_state(route_kind="via" if via else "direct", original_query="行程",
            start=coord(start), end=coord(end), via=coord(via) if via else None,
            travel_mode="walk", hard_constraints={}, strategy={"name": "shortest"},
            data_version="data", road_condition_version=version)
    monkeypatch.setattr(planner.agent_tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(planner.agent_tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    route, error = planner._replan_changed_place_route([
        {"route_state": route_state(1, 3, 2, "old")},
        {"route_state": route_state(3, 4, None, "new")}], {"task_constraints": {}})
    assert error is None
    assert route["route_verification"]["status"] == "pass"
    assert route["recommended_edge_ids"] == [(1, 2, 0), (2, 3, 0), (3, 4, 0)]


def test_snapshot_replan_preserves_custom_weights_with_real_route_tool(monkeypatch):
    import networkx as nx
    from agents.routing_policy import select_route_strategy
    graph = nx.MultiDiGraph()
    for node in range(1, 4):
        graph.add_node(node, x=114.36 + (node - 1) * .001, y=30.53)
    for node in (1, 2):
        graph.add_edge(node, node + 1, 0, length=100, highway="residential")
        graph.add_edge(node + 1, node, 0, length=100, highway="residential")
    def coord(node):
        return {"type": "coord", "name": str(node), "coordinates": {
            "lng": graph.nodes[node]["x"], "lat": graph.nodes[node]["y"]}}
    strategy = select_route_strategy(query="", explicit_strategy="custom",
        custom_weights={"distance": .5, "slope": .3, "scenery": .2}).as_dict()
    def leg(a, b, version):
        return {"route_state": build_route_state(route_kind="direct", original_query="行程",
            start=coord(a), end=coord(b), travel_mode="walk", hard_constraints={}, strategy=strategy,
            data_version="data", road_condition_version=version)}
    monkeypatch.setattr(planner.agent_tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(planner.agent_tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: [])
    route, reason = planner._replan_changed_place_route([leg(1, 2, "old"), leg(2, 3, "new")], {})
    assert reason is None
    assert route["route_state"]["strategy"]["weights"] == strategy["weights"]
    assert route["route_state"]["strategy"]["detour_cap"] == strategy["detour_cap"]


def test_real_default_and_explicit_shortest_have_equivalent_strategy():
    import networkx as nx
    graph = nx.MultiDiGraph()
    for node in range(1, 4):
        graph.add_node(node, x=114.36 + (node - 1) * .001, y=30.53)
    for node in (1, 2):
        graph.add_edge(node, node + 1, 0, length=100, highway="residential")
        graph.add_edge(node + 1, node, 0, length=100, highway="residential")
    def coord(node):
        return {"type": "coord", "name": str(node), "coordinates": {
            "lng": graph.nodes[node]["x"], "lat": graph.nodes[node]["y"]}}
    context = {"_route_graph_snapshot": graph, "_route_conditions_snapshot": [], "_route_weather_snapshot": None}
    first, first_artifact = planner.agent_tools.execute_tool("plan_route",
        {"start": coord(1), "end": coord(2), "mode": "walk"}, context)
    second, second_artifact = planner.agent_tools.execute_tool("plan_route",
        {"start": coord(2), "end": coord(3), "mode": "walk", "strategy": "shortest"}, context)
    assert first_artifact is not None, first
    assert second_artifact is not None, second
    legs = [first_artifact["route"], second_artifact["route"]]
    assert legs[0]["route_state"]["strategy"]["source"] != legs[1]["route_state"]["strategy"]["source"]
    assert planner._place_route_constraint_issue(legs, {}) is None


def test_via_tool_keeps_selected_poi_identity_in_returned_state(monkeypatch):
    import networkx as nx
    graph = nx.MultiDiGraph()
    pois = []
    for node in range(1, 4):
        lng = 114.36 + (node - 1) * .001
        graph.add_node(node, x=lng, y=30.53)
        pois.append({"id": f"poi_{node}", "name": f"地点{node}", "coordinates": {"lng": lng, "lat": 30.53}})
    for node in range(1, 3):
        graph.add_edge(node, node + 1, 0, length=100, highway="residential")
        graph.add_edge(node + 1, node, 0, length=100, highway="residential")
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: pois)
    monkeypatch.setattr(planner.agent_tools, "navigation_wgs", lambda poi, _: (
        poi["coordinates"]["lng"], poi["coordinates"]["lat"]))
    monkeypatch.setattr(planner.agent_tools, "find_poi_ambiguous", lambda name:
        (next((poi for poi in pois if poi["name"] == name), None), []))
    args = {"start": {"poi_id": "poi_1", "name": "地点1"},
            "end": {"poi_id": "poi_3", "name": "地点3"},
            "via_points": [{"poi_id": "poi_2", "name": "地点2"}],
            "mode": "walk", "strategy": "shortest"}
    result, artifact = planner.agent_tools.execute_tool("plan_via_route", args,
        {"_route_graph_snapshot": graph, "_route_conditions_snapshot": [], "_route_weather_snapshot": None})
    assert artifact is not None, result
    assert artifact["route"]["route_state"]["via"]["poi_id"] == "poi_2"


@pytest.mark.parametrize("replan_ok", [True, False])
def test_compound_user_flow_replans_or_preserves_pending_task(monkeypatch, replan_ok):
    pois = [{"id": f"id_{name}", "name": name} for name in "ADE"]
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: pois)
    def execute(name, args, _ctx):
        assert name == "plan_route"
        start, end = args["start"]["name"], args["end"]["name"]
        version = "old" if end == "D" else "new"
        route = {"start_name": start, "end_name": end, "mode": "walk",
                 "recommended": [[114.36, 30.53], [114.361, 30.53]],
                 "recommended_edge_ids": [[1, 2, 0]], "steps": [], "pois": [],
                 "route_state": state(start, end, version=version, constraints={"avoid_steps": True}),
                 "route_verification": {"status": "pass"}}
        return route, {"route": route}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    fresh_state = state("A", "E", via=ref("D"), version="fresh", constraints={"avoid_steps": True})
    fresh_state["route_id"] = "fresh"
    fresh = {"start_name": "A", "end_name": "E", "mode": "walk",
             "recommended": [[114.36, 30.53], [114.362, 30.53]],
             "route_verification": {"status": "pass"}, "route_state": fresh_state}
    seen = []
    ready = {"value": replan_ok}
    def replan(legs, context):
        seen.append((legs, context))
        return (fresh, None) if ready["value"] else (None, "当前路况再次变化")
    monkeypatch.setattr(planner, "_replan_changed_place_route", replan)
    pending = {"request_id": "request", "original_query": "从A到D再到E",
               "pending_slot": "selection", "start": ref("A"), "mode": "walk",
               "candidate_poi_ids": ["id_D"], "constraints": {"avoid_steps": True},
               "follow_up_requests": [{"destination_poi_id": "id_E", "query": "到E"}]}
    response = planner.run_agent("选这个", context={"place_request": pending, "selected_poi_id": "id_D"})
    assert len(seen) == 1 and seen[0][1]["task_constraints"] == {"avoid_steps": True}
    if replan_ok:
        assert response["response_kind"] == "route" and response["route"]["route_state"]["route_id"] == "fresh"
    else:
        assert response["response_kind"] == "clarify" and response["route"] is None
        assert response["clarify"]["options"] == ["重试"]
        assert "点击“重试”" in response["message"] and "重新提交完整需求" not in response["message"]
        assert response["place_request"]["pending_slot"] == "route_retry"
        from api.routes import _agent_response_to_legacy
        projected = _agent_response_to_legacy(response)
        assert projected["clarify"]["options"] == ["重试"]
        assert any(item.get("destination_poi_id") == "id_E" for item in response["place_request"]["follow_up_requests"])
        ready["value"] = True
        retried = planner.run_agent("重试", context={"place_request": response["place_request"]})
        assert len(seen) == 2 and retried["response_kind"] == "route"
        assert retried["route"]["route_state"]["route_id"] == "fresh"


def test_final_leg_failure_keeps_choice_and_all_remaining_requirements(monkeypatch):
    pois = [{"id": f"id_{name}", "name": name} for name in "ADE"]
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: pois)
    calls = []
    def execute(name, args, _ctx):
        calls.append((name, args))
        if args["end"]["poi_id"] == "id_E":
            return {"error": "route_not_found", "message": "终点路段当前无法通行"}, None
        route = {"start_name": "A", "end_name": "D", "mode": "walk",
                 "recommended": [[114.36, 30.53], [114.361, 30.53]],
                 "recommended_edge_ids": [[1, 2, 0]], "steps": [], "pois": [],
                 "route_state": state("A", "D", constraints={"avoid_steps": True, "max_distance_m": 5000}),
                 "route_verification": {"status": "pass"}}
        return route, {"route": route}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    pending = {"request_id": "request", "original_query": "从A到D再到E", "pending_slot": "selection",
               "start": ref("A"), "mode": "walk", "candidate_poi_ids": ["id_D"],
               "constraints": {"avoid_steps": True, "max_distance_m": 5000},
               "follow_up_requests": [{"destination_poi_id": "id_E", "query": "到E"}]}
    result = planner.run_agent("选这个", context={"place_request": pending, "selected_poi_id": "id_D"})
    assert result["route"] is None and result["needs_input"] is True
    assert result["place_request"]["selected_poi_id"] == "id_D"
    assert result["place_request"]["follow_up_requests"][0]["destination_poi_id"] == "id_E"
    assert result["place_request"]["constraints"]["max_distance_m"] == 5000
    assert calls[-1][1]["constraints"]["max_distance_m"] == 5000
    planner.run_agent("重试", context={"place_request": result["place_request"]})
    assert calls[-1][1]["end"]["poi_id"] == "id_E"


def test_multistage_combination_failure_retry_retains_next_target(monkeypatch):
    pois = [{"id": f"id_{name}", "name": name} for name in "ADEF"]
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: pois)
    monkeypatch.setattr(planner, "_place_route_snapshot_issue", lambda _: None)
    def make_route(start, end, version, first_edge, last_edge, via=None):
        route_state = state(start, end, via=via, version=version, constraints={"avoid_steps": True})
        return {"start_name": start, "end_name": end, "mode": "walk",
                "recommended": [[114.36 + first_edge * .001, 30.53],
                                [114.36 + last_edge * .001, 30.53]],
                "recommended_edge_ids": [[first_edge, first_edge + 1, 0],
                                         *([[first_edge + 1, last_edge, 0]] if last_edge > first_edge + 1 else [])],
                "recommended_length_m": 100 * (last_edge - first_edge), "duration_min": 2,
                "steps": [], "pois": [], "route_events": [], "road_condition_overview": [],
                "route_state": route_state, "route_verification": {"status": "pass"}}
    previous = make_route("A", "D", "old", 1, 2)
    newly_selected = make_route("D", "E", "new", 2, 3)
    rebuilt = make_route("A", "E", "fresh", 1, 3, via=ref("D"))
    final_leg = make_route("E", "F", "fresh", 3, 4)
    def execute(name, args, _ctx):
        assert name == "plan_route"
        result = newly_selected if args["end"]["poi_id"] == "id_E" else final_leg
        return result, {"route": result}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    attempts = []
    def replan(route_legs, context):
        attempts.append((route_legs, context))
        return (None, "路况正在更新") if len(attempts) == 1 else (rebuilt, None)
    monkeypatch.setattr(planner, "_replan_changed_place_route", replan)
    pending = {"request_id": "request", "original_query": "从A经D到E再到F",
               "pending_slot": "selection", "start": ref("D"), "mode": "walk",
               "candidate_poi_ids": ["id_E"], "completed_legs": [previous],
               "constraints": {"avoid_steps": True},
               "follow_up_requests": [{"destination_poi_id": "id_F", "query": "到F"}]}
    failed = planner.run_agent("选这个", context={"place_request": pending, "selected_poi_id": "id_E"})
    assert failed["response_kind"] == "clarify" and failed["route"] is None
    assert failed["clarify"]["options"] == ["重试"]
    assert failed["place_request"]["pending_slot"] == "route_retry"
    assert failed["place_request"]["follow_up_requests"][0]["destination_poi_id"] == "id_F"
    retried = planner.run_agent("重试", context={"place_request": failed["place_request"]})
    assert len(attempts) == 2 and retried["response_kind"] == "route"
    assert retried["route"]["route_state"]["end"]["poi_id"] == "id_F"
    assert [item["poi_id"] for item in retried["route"]["route_state"]["via"]["points"]] == ["id_D", "id_E"]
    assert retried["route"]["route_state"]["hard_constraints"]["avoid_steps"] is True


@pytest.mark.parametrize("event_change", ["before_second_leg", "after_second_leg"])
def test_total_library_basketball_meal_flow_replans_on_real_isolated_graph(monkeypatch, event_change):
    import time
    import networkx as nx
    graph = nx.MultiDiGraph()
    pois = []
    for node, name, category, activity in [(1, "总图", "library", "study"),
                                            (2, "篮球场", "field", "basketball"),
                                            (3, "食堂", "canteen", "canteen_meal")]:
        lng = 114.36 + (node - 1) * .001
        graph.add_node(node, x=lng, y=30.53)
        pois.append({"id": f"poi_{node}", "name": name, "type": "study" if node == 1 else "sport" if node == 2 else "dining",
                     "subcategory": category, "activities": [activity],
                     "coordinates": {"lng": lng, "lat": 30.53}})
    for node in (1, 2):
        graph.add_edge(node, node + 1, 0, length=100, highway="residential")
        graph.add_edge(node + 1, node, 0, length=100, highway="residential")
    monkeypatch.setattr(planner.agent_tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(planner.agent_tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: pois)
    monkeypatch.setattr(planner.agent_tools, "navigation_wgs", lambda poi, _: (
        poi["coordinates"]["lng"], poi["coordinates"]["lat"]))
    monkeypatch.setattr(planner.agent_tools, "find_poi_ambiguous", lambda name:
        (next((poi for poi in pois if poi["name"] == name), None), []))
    event = {"id": "new", "name": "模拟提示", "type": "event", "action": "notice",
             "affected_modes": ["walk"], "cost_multiplier": None,
             "start_time": time.time() - 10, "end_time": time.time() + 300,
             "edge": {"u": 1, "v": 2, "key": 0, "road_name": "测试路"}}
    current = {"events": []}
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_: deepcopy(current["events"]))
    actual_execute = planner.agent_tools.execute_tool
    called = []
    def execute(name, args, ctx):
        called.append(name)
        if name == "search_poi_candidates":
            basketball = "basketball" in args.get("activities", [])
            if not basketball:
                assert args["start"]["poi_id"] == "poi_2"
            poi = pois[1] if basketball else pois[2]
            candidate = {"poi_id": poi["id"], "name": poi["name"], "subcategory": poi["subcategory"],
                         "activities": poi["activities"], "distance_m": 100}
            return {"candidates": [candidate], "start_name": args["start"]["name"]}, {"candidates": [candidate]}
        second_leg = name == "plan_route" and args["end"].get("poi_id") == "poi_3"
        if second_leg and event_change == "before_second_leg":
            current["events"] = [event]
        output = actual_execute(name, args, ctx)
        if second_leg and event_change == "after_second_leg":
            current["events"] = [event]
        return output
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    first = planner.run_agent("我在总图，先找一个球场然后去吃饭，不走台阶")
    assert first["clarify"]["question"] == "想打什么球？"
    second = planner.run_agent("篮球", context={"place_request": first["place_request"]})
    assert second["response_kind"] == "candidates"
    third = planner.run_agent("去这个地点", context={"place_request": second["place_request"], "selected_poi_id": "poi_2"})
    assert third["needs_input"] is True
    fourth = planner.run_agent("去这个地点", context={"place_request": third["place_request"], "selected_poi_id": "poi_3"})
    assert fourth["response_kind"] == "route", fourth.get("message")
    route = fourth["route"]
    assert route["route_verification"]["status"] == "pass"
    assert route["recommended_edge_ids"] == [(1, 2, 0), (2, 3, 0)]
    assert route["route_state"]["via"]["poi_id"] == "poi_2"
    assert route["route_state"]["hard_constraints"]["avoid_steps"] is True
    assert called.count("plan_via_route") == 1
