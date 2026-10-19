"""Ordered travel instructions must not acquire an invented itinerary budget."""

import pytest

from agents import planner, tools


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "test-key")


@pytest.mark.parametrize("query,start,via,end", [
    ("从玉兰2门先到卓尔体育馆，再到桂园食堂，最后樱顶。",
     "玉兰2门", ["卓尔体育馆", "桂园食堂"], "樱顶"),
    ("从南门先去图书馆，然后到实验楼，最后到北门",
     "南门", ["图书馆", "实验楼"], "北门"),
    ("从东门依次到甲楼、乙楼、丙楼", "东门", ["甲楼", "乙楼"], "丙楼"),
    ("从南门先到图书馆再到实验楼最后到北门", "南门", ["图书馆", "实验楼"], "北门"),
    ("从南门出发，先到图书馆，然后到实验楼，最后北门", "南门", ["图书馆", "实验楼"], "北门"),
    ("从南门先到新材料再生中心，最后北门", "南门", ["新材料再生中心"], "北门"),
])
def test_ordered_route_executes_all_stops_without_model_or_itinerary(
        monkeypatch, query, start, via, end):
    calls = []
    route = {"start_name": start, "end_name": end, "duration_min": 20}

    def execute(name, args, ctx):
        calls.append((name, args, ctx))
        return route, {"route": route, "route_kind": "via"}

    monkeypatch.setattr(tools, "execute_tool", execute)
    monkeypatch.setattr(planner, "_make_client", lambda: pytest.fail("ordered route called LLM"))
    result = planner.run_agent(query)
    assert result["response_kind"] == "route"
    assert result["route_kind"] == "via"
    name, args, _ctx = calls[0]
    assert name == "plan_via_route"
    assert args["start"]["name"] == start
    assert args["end"]["name"] == end
    assert [point["name"] for point in args["via_points"]] == via
    assert "time_budget_min" not in args
    assert "stop_duration_min" not in args


def test_wrong_itinerary_call_cannot_invent_budget_dwell_or_return(monkeypatch):
    query = "从南门先到图书馆，再到实验楼，最后北门。"
    calls = []
    route = {"start_name": "南门", "end_name": "北门"}

    def via(args, ctx):
        calls.append(args)
        return route, {"route": route, "route_kind": "via"}

    monkeypatch.setattr(tools, "_tool_plan_via_route", via)
    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "南门"},
        "stops": [{"name": "图书馆"}, {"name": "实验楼"}, {"name": "北门"}],
        "end": {"name": "南门"}, "time_budget_min": 1, "stop_duration_min": 15,
        "constraints": {"max_duration_s": 60},
    }, {"query": query})
    assert artifact and artifact["route_kind"] == "via"
    assert calls[0]["end"]["name"] == "北门"
    assert [point["name"] for point in calls[0]["via_points"]] == ["图书馆", "实验楼"]
    assert not calls[0].get("constraints")
    assert "itinerary" not in result


@pytest.mark.parametrize("suffix", [
    "，总共90分钟", "，每站停留5分钟", "，最后返回起点", "，规划游览行程",
])
def test_explicit_itinerary_intent_is_not_reinterpreted(suffix):
    assert tools.ordered_via_request("从南门先到图书馆，再到实验楼，最后北门" + suffix) is None


@pytest.mark.parametrize("suffix", [
    "，不超过2公里", "，无障碍通行", "，轮椅可达", "，途经点全部要室内",
    "，骑行，距离不超过1000米", "，需要安静的路", "，北门附近取快递",
])
def test_unhandled_extra_requirements_stay_on_model_path(suffix):
    assert tools.ordered_via_request("从南门先到图书馆，再到实验楼，最后北门" + suffix) is None


def test_list_unknown_suffix_is_not_another_destination():
    assert tools.ordered_via_request("从南门依次到甲楼、乙楼、北门，需要安静的路") is None


def test_itinerary_guard_uses_explicit_mode_and_keeps_strategy(monkeypatch):
    calls = []

    def via(args, ctx):
        calls.append((args, ctx))
        return {}, {"route": {"mode": args["mode"]}, "route_kind": "via"}

    monkeypatch.setattr(tools, "_tool_plan_via_route", via)
    tools.execute_tool("plan_itinerary", {"mode": "drive"}, {
        "query": "从南门先到图书馆，再到实验楼，最后北门，骑行，平坦优先，避开台阶",
    })
    args, ctx = calls[0]
    assert args["mode"] == "bike"
    assert args["constraints"] == {"avoid_steps": True}
    assert tools._strategy_for_args(args, ctx).name == "flat"


def test_ordered_route_preserves_explicit_mode_and_steps_constraint(monkeypatch):
    calls = []

    def execute(name, args, ctx):
        calls.append(args)
        route = {"start_name": "南门", "end_name": "北门"}
        return route, {"route": route, "route_kind": "via"}

    monkeypatch.setattr(tools, "execute_tool", execute)
    monkeypatch.setattr(planner, "_make_client", lambda: pytest.fail("ordered route called LLM"))
    planner.run_agent("从南门先到图书馆，再到实验楼，最后北门，骑行，避开台阶")
    assert calls[0]["mode"] == "bike"
    assert calls[0]["constraints"] == {"avoid_steps": True}
    assert calls[0]["end"]["name"] == "北门"


def test_explicit_budget_dwell_and_return_still_use_itinerary(monkeypatch):
    calls = []
    route = {
        "start_name": "南门", "end_name": "南门", "duration_min": 10,
        "recommended_length_m": 600, "legs": [{"duration_min": 10}],
        "route_state": {"route_kind": "via"},
        "route_verification": {"status": "pass", "measured_length_m": 600},
    }

    def via(args, ctx):
        calls.append(args)
        return route, {"route": route, "route_kind": "via"}

    monkeypatch.setattr(tools, "_tool_plan_via_route", via)
    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "南门"}, "end": {"name": "南门"},
        "stops": [{"name": "图书馆"}, {"name": "实验楼"}],
        "time_budget_min": 30, "stop_duration_min": 5,
    }, {"query": "从南门先到图书馆，再到实验楼，最后返回南门，总共30分钟，每站停留5分钟"})
    assert result["total_duration_min"] == 20
    assert result["time_budget_min"] == 30
    assert calls[0]["end"]["name"] == "南门"
    assert artifact["route_kind"] == "itinerary"


def test_ordered_route_verification_failure_cannot_fall_back_to_a_direct_route(monkeypatch):
    monkeypatch.setattr(tools, "execute_tool", lambda *_args: (
        {"error": "route_verification_failed", "message": "途经顺序无法核验"}, None,
    ))
    monkeypatch.setattr(planner, "_make_client", lambda: pytest.fail("ordered route called LLM"))
    result = planner.run_agent("从南门先到图书馆，再到实验楼，最后北门")
    assert result["route"] is None
    assert result["response_kind"] == "chat"
    assert "未通过独立核验" in result["message"]


@pytest.mark.parametrize("inherited_mode", ["bike", "drive"])
def test_explicit_spoken_walk_overrides_inherited_mode(monkeypatch, inherited_mode):
    calls = []

    def execute(name, args, ctx):
        calls.append(args)
        route = {"start_name": "南门", "end_name": "北门"}
        return route, {"route": route, "route_kind": "via"}

    monkeypatch.setattr(tools, "execute_tool", execute)
    planner.run_agent("从南门先到图书馆，最后北门，走路", travel_mode=inherited_mode)
    assert calls[0]["mode"] == "walk"


@pytest.mark.parametrize("failure", [False, True])
def test_deterministic_ordered_route_emits_sanitized_tool_trace(monkeypatch, failure):
    traces = []
    route = {"start_name": "南门", "end_name": "北门"}
    monkeypatch.setattr(tools, "execute_tool", lambda *_args: (
        ({"error": "route_verification_failed", "message": "unverified"}, None)
        if failure else (route, {"route": route, "route_kind": "via"})
    ))
    planner.run_agent("从南门先到图书馆，最后北门", uid="private-user-id",
                      tool_trace_callback=traces.append)
    assert traces == [{
        "tool_name": "plan_via_route",
        "argument_keys": ["end", "mode", "start", "via_points"],
        "outcome": "tool_error" if failure else "success",
        "artifact_kind": None if failure else "route",
    }]


def test_deterministic_ordered_route_records_tool_exception(monkeypatch):
    traces = []

    def execute(*_args):
        raise RuntimeError("private diagnostic")

    monkeypatch.setattr(tools, "execute_tool", execute)
    with pytest.raises(RuntimeError, match="private diagnostic"):
        planner.run_agent("从南门先到图书馆，最后北门", tool_trace_callback=traces.append)
    assert traces == [{
        "tool_name": "plan_via_route",
        "argument_keys": ["end", "mode", "start", "via_points"],
        "outcome": "exception", "artifact_kind": None,
    }]
