import agents.tools as tools


def _route_result(distance=1000, duration=15):
    return {
        "start_name": "星湖园",
        "end_name": "宿舍",
        "recommended_length_m": distance,
        "distance_m": distance,
        "duration_min": duration,
        "legs": [{"duration_min": duration}],
        "route_state": {"route_kind": "via"},
        "route_verification": {
            "status": "pass", "measured_length_m": distance,
            "scope": "internal_graph_and_event_snapshot",
        },
    }


def test_plan_itinerary_counts_travel_and_stop_time_and_returns_route(monkeypatch):
    calls = []

    def plan_via(args, ctx):
        calls.append((args, ctx))
        return _route_result(), {"route": _route_result(), "route_kind": "via"}

    monkeypatch.setattr(tools, "_tool_plan_via_route", plan_via)
    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园", "type": "poi"},
        "stops": [{"name": "星湖园食堂", "type": "poi"}, {"name": "图书馆", "type": "poi"}],
        "end": {"name": "宿舍", "type": "poi"},
        "time_budget_min": 30,
        "stop_duration_min": 5,
    }, {"query": "半小时内逛两个地方"})

    assert len(calls) == 1
    assert calls[0][0]["via_points"] == [
        {"name": "星湖园食堂", "type": "poi"}, {"name": "图书馆", "type": "poi"},
    ]
    assert result["status"] == "feasible"
    assert result["travel_duration_min"] == 15
    assert result["stop_duration_min"] == 10
    assert result["total_duration_min"] == 25
    assert artifact["route"]["itinerary"]["time_budget_min"] == 30
    assert artifact["route"]["route_state"]["route_kind"] == "itinerary"
    assert artifact["route"]["route_state"]["itinerary"] == {
        "time_budget_min": 30,
        "stop_duration_min": 5,
    }


def test_plan_itinerary_does_not_publish_route_when_budget_is_exceeded(monkeypatch):
    monkeypatch.setattr(tools, "_tool_plan_via_route",
                        lambda *_args: (_route_result(duration=25), {"route": _route_result(duration=25)}))

    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园", "type": "poi"},
        "stops": [{"name": "食堂", "type": "poi"}, {"name": "图书馆", "type": "poi"}],
        "time_budget_min": 30,
        "stop_duration_min": 5,
    })

    assert result["status"] == "over_budget"
    assert result["total_duration_min"] == 35
    assert result["over_by_min"] == 5
    assert artifact is None


def test_itinerary_hard_duration_limit_includes_stops(monkeypatch):
    monkeypatch.setattr(tools, "_tool_plan_via_route",
                        lambda *_args: (_route_result(), {"route": _route_result()}))

    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园"},
        "stops": [{"name": "食堂"}, {"name": "图书馆"}],
        "time_budget_min": 30,
        "stop_duration_min": 5,
        "constraints": {"max_duration_s": 1200},
    })

    assert result["status"] == "over_budget"
    assert result["hard_max_duration_min"] == 20
    assert result["total_duration_min"] == 25
    assert artifact is None


def test_plan_itinerary_requires_budget_and_at_least_one_stop():
    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园", "type": "poi"},
    })

    assert result["error"] == "invalid_itinerary"
    assert artifact is None


def test_plan_itinerary_rejects_boolean_budget_values():
    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园", "type": "poi"},
        "stops": [{"name": "图书馆", "type": "poi"}],
        "time_budget_min": True,
        "stop_duration_min": False,
    })

    assert result["error"] == "invalid_itinerary"
    assert artifact is None


def test_plan_itinerary_schema_and_executor_are_registered():
    schema = next((item for item in tools.TOOL_SCHEMAS
                   if item["function"]["name"] == "plan_itinerary"), None)

    assert schema is not None
    assert schema["function"]["parameters"]["required"] == ["start", "stops", "time_budget_min"]
    assert "plan_itinerary" in tools._EXECUTORS


def test_plan_itinerary_rejects_unverified_via_route(monkeypatch):
    route = _route_result()
    route.pop("route_verification")
    monkeypatch.setattr(tools, "_tool_plan_via_route", lambda *_args: (route, {"route": route}))

    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园"},
        "stops": [{"name": "图书馆"}],
        "time_budget_min": 30,
    })
    assert result["error"] == "itinerary_verification_failed"
    assert artifact is None


def test_plan_itinerary_rejects_underreported_travel_time(monkeypatch):
    route = _route_result(distance=1200, duration=1)
    monkeypatch.setattr(tools, "_tool_plan_via_route", lambda *_args: (route, {"route": route}))

    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园"},
        "stops": [{"name": "图书馆"}],
        "time_budget_min": 30,
    })
    assert result["error"] == "itinerary_verification_failed"
    assert artifact is None


def test_plan_itinerary_rejects_travel_time_that_disagrees_with_verified_legs(monkeypatch):
    route = _route_result(distance=900, duration=15)
    route["legs"] = [{"duration_min": 8.0}, {"duration_min": 12.0}]
    monkeypatch.setattr(tools, "_tool_plan_via_route", lambda *_args: (route, {"route": route}))

    result, artifact = tools.execute_tool("plan_itinerary", {
        "start": {"name": "星湖园"},
        "stops": [{"name": "图书馆"}],
        "time_budget_min": 30,
        "stop_duration_min": 5,
    })
    assert result["error"] == "itinerary_verification_failed"
    assert artifact is None
