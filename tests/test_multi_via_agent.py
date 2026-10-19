import json
from types import SimpleNamespace

import agents.planner as planner
import agents.tools as tools
import api.routes as routes


def _route_state(via):
    return {
        "schema_version": 1,
        "route_id": "route-multi-via",
        "route_kind": "via",
        "original_query": "从起点经过两个点到终点",
        "start": {"name": "起点", "type": "poi"},
        "end": {"name": "终点", "type": "poi"},
        "via": via,
        "tour": None,
        "legs": [],
        "travel_mode": "walk",
        "hard_constraints": {},
        "strategy": {"name": "shortest", "source": "commute_default"},
        "data_version": routes._current_data_version(),
        "road_condition_version": "roads-test",
    }


def test_every_declared_tool_has_an_executor():
    schema_names = [item["function"]["name"] for item in tools.TOOL_SCHEMAS]
    assert len(schema_names) == len(set(schema_names))
    assert set(schema_names) == set(tools._EXECUTORS)
    for schema in tools.TOOL_SCHEMAS:
        parameters = schema["function"]["parameters"]
        assert parameters.get("type") == "object"
        assert set(parameters.get("required", [])) <= set(parameters.get("properties", {}))


def test_via_tool_schema_accepts_an_ordered_list_of_multiple_points():
    params = next(item["function"]["parameters"] for item in tools.TOOL_SCHEMAS
                  if item["function"]["name"] == "plan_via_route")
    via_points = params["properties"]["via_points"]
    assert via_points["type"] == "array"
    assert via_points["minItems"] == 1
    assert via_points["items"]["type"] == "object"
    assert via_points["maxItems"] == 10


def test_via_tool_rejects_more_than_ten_points_with_a_user_readable_error(monkeypatch):
    monkeypatch.setattr(tools, "_plan_common", lambda args, ctx: (object(), object(), "walk", None))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, graph: (1, ref["name"], None))
    result, artifact = tools.execute_tool("plan_via_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "via_points": [{"name": f"点{i}"} for i in range(11)],
    })
    assert result["error"] == "invalid_via_points"
    assert "10" in result["message"]
    assert artifact is None


def test_map_selected_multiple_waypoints_execute_route_tool_without_llm(monkeypatch):
    route = {"recommended": [], "steps": [], "timings_ms": {}}
    calls = []

    def execute(name, args, ctx):
        calls.append((name, args, ctx))
        return route, {"route": route, "route_kind": "via"}

    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    monkeypatch.setattr(planner, "_make_client", lambda: (_ for _ in ()).throw(
        AssertionError("structured map waypoints must not wait for an LLM call")))
    result = planner.run_agent(
        "从地图标记的起点途经地图标记点到终点",
        coord_start={"lng": 114.36, "lat": 30.53, "name": "起点"},
        coord_end={"lng": 114.37, "lat": 30.54, "name": "终点"},
        coord_waypoints=[
            {"lng": 114.361, "lat": 30.531},
            {"lng": 114.362, "lat": 30.532},
        ],
        travel_mode="walk",
    )

    assert result["response_kind"] == "route"
    assert result["route_kind"] == "via"
    assert calls[0][0] == "plan_via_route"
    assert [point["lng"] for point in calls[0][1]["via_points"]] == [114.361, 114.362]


def test_natural_language_tool_call_keeps_all_named_stops_in_order(monkeypatch):
    points = [{"name": "卓尔体育馆", "type": "poi"}, {"name": "星湖园食堂", "type": "poi"}]
    route = {
        "start_name": "玉兰二门", "end_name": "信息学部",
        "recommended_length_m": 800, "shortest_length_m": 800,
        "duration_min": 10, "shortest_duration_min": 10, "mode": "walk",
    }
    function = SimpleNamespace(name="plan_via_route", arguments=json.dumps({
        "start": {"name": "玉兰二门"}, "end": {"name": "信息学部"},
        "via_points": points,
    }, ensure_ascii=False))
    tool_call = SimpleNamespace(id="multi-via-call", function=function)
    message = SimpleNamespace(content=None, tool_calls=[tool_call],
                              model_dump=lambda exclude_none=False: {"role": "assistant"})
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class Client:
        chat = None
        completions = None

        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            return response

    client = Client()
    calls = []

    def execute(name, args, ctx):
        calls.append((name, args))
        return route, {"route": route, "route_kind": "via"}

    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(planner, "_make_client", lambda: client)
    monkeypatch.setattr(planner.agent_tools, "execute_tool", execute)
    result = planner.run_agent("从玉兰二门经过卓尔体育馆和星湖园食堂到信息学部")

    assert result["response_kind"] == "route"
    assert calls[0][0] == "plan_via_route"
    assert calls[0][1]["via_points"] == points


def test_via_tool_routes_through_all_points_in_order(monkeypatch):
    graph = object()
    decision = SimpleNamespace(
        name="shortest", weights={"distance": 1.0, "slope": 0.0, "scenery": 0.0},
        detour_cap=1.0, as_dict=lambda: {"name": "shortest", "source": "button"},
    )
    names = {"起点": 1, "甲点": 2, "乙点": 3, "终点": 4}
    route_calls = []

    def fake_compute_route(_graph, start_node, end_node, **kwargs):
        route_calls.append((start_node, end_node))
        length = 300.0 if (start_node, end_node) == (1, 4) else 100.0
        return {
            "recommended": [start_node, end_node], "shortest": [start_node, end_node],
            "recommended_edges": [(start_node, end_node, 0)],
            "shortest_edges": [(start_node, end_node, 0)],
            "recommended_length_m": length, "shortest_length_m": length,
            "filter_status": "", "overlap_rate": 1.0, "duration_min": 2.0,
            "shortest_duration_min": 2.0, "applied_weights": kwargs.get("weights"),
            "degraded": False, "length_capped": False, "mode": "walk",
            "timings_ms": {},
        }

    def fake_payload(_graph, result, start_name, end_name, mode):
        a, b = result["recommended"]
        return {
            "start_name": start_name, "end_name": end_name,
            "recommended": [{"lng": a, "lat": 0}, {"lng": b, "lat": 0}],
            "shortest": [{"lng": a, "lat": 0}, {"lng": b, "lat": 0}],
            "recommended_edge_ids": result["recommended_edges"],
            "shortest_edge_ids": result["shortest_edges"],
            "steps": [{"type": "depart", "cumulative_m": 0},
                      {"type": "arrive", "cumulative_m": result["recommended_length_m"]}],
            "pois": [], "recommended_length_m": result["recommended_length_m"],
            "shortest_length_m": result["shortest_length_m"],
            "duration_min": result["duration_min"],
            "shortest_duration_min": result["shortest_duration_min"],
            "timings_ms": {},
        }

    monkeypatch.setattr(tools, "_plan_common", lambda args, ctx: (graph, graph, "walk", None))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, g: (names[ref["name"]], ref["name"], None))
    monkeypatch.setattr(tools, "_strategy_for_args", lambda args, ctx: decision)
    monkeypatch.setattr(tools, "compute_route", fake_compute_route)
    monkeypatch.setattr(tools, "_route_payload", fake_payload)
    monkeypatch.setattr(tools, "current_data_version", lambda graph: "data-test")
    monkeypatch.setattr(tools, "current_road_condition_version", lambda: "roads-test")
    # This ordering-only test mocks the graph as an object. The verifier has
    # dedicated tests against exact edges and a real MultiDiGraph.
    monkeypatch.setattr(tools, "_audit_tool_route", lambda *_args: None)

    payload, artifact = tools.execute_tool("plan_via_route", {
        "start": {"name": "起点"}, "end": {"name": "终点"},
        "via_points": [{"name": "甲点"}, {"name": "乙点"}],
    }, {"query": "依次经过甲点和乙点"})

    assert route_calls == [(1, 2), (2, 3), (3, 4), (1, 4)]
    assert [coord["lng"] for coord in payload["recommended"]] == [1, 2, 3, 4]
    assert len(payload["legs"]) == 3
    assert payload["via"]["name"] == "甲点、乙点"
    assert payload["route_state"]["via"]["type"] == "multi"
    assert [point["name"] for point in payload["route_state"]["via"]["points"]] == ["甲点", "乙点"]
    assert payload["duration_min"] == 6.0
    assert artifact["route_kind"] == "via"


def test_replan_preserves_multiple_via_points():
    via = {"name": "甲点、乙点", "type": "multi", "points": [
        {"name": "甲点", "type": "poi"},
        {"name": "乙点", "type": "coord", "coordinates": {"lng": 114.36, "lat": 30.53}},
    ]}
    tool_name, args = routes._replan_tool_request(_route_state(via))
    assert tool_name == "plan_via_route"
    assert args["via_points"] == via["points"]
