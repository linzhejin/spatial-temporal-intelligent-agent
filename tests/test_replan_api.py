from copy import deepcopy
from unittest.mock import patch

import pytest

import api.routes as routes


def _state_for(route_kind):
    state = {
        "schema_version": 1,
        "route_id": "route-test",
        "route_kind": route_kind,
        "original_query": "测试路线",
        "start": {"name": "珞珈门", "type": "poi"},
        "end": {"name": "樱顶", "type": "poi"},
        "via": None,
        "tour": None,
        "legs": [],
        "travel_mode": "walk",
        "hard_constraints": {"slope": "normal"},
        "strategy": {
            "name": "shortest",
            "source": "commute_default",
            "task_class": "commute",
            "weights": {"distance": 1.0, "slope": 0.0, "scenery": 0.0},
            "detour_cap": 1.0,
        },
        "data_version": routes._current_data_version(),
        "road_condition_version": "old-road-version",
    }
    if route_kind == "via":
        state["via"] = {"name": "老图书馆", "type": "poi"}
    elif route_kind == "tour":
        state["tour"] = {
            "theme": "scenery",
            "pois": [{"name": "老图书馆", "type": "poi"}],
            "loop": False,
        }
    elif route_kind == "itinerary":
        state["via"] = {"type": "multi", "points": [
            {"name": "星湖园食堂", "type": "poi"},
            {"name": "图书馆", "type": "poi"},
        ]}
        state["itinerary"] = {"time_budget_min": 45, "stop_duration_min": 8}
    elif route_kind == "multimodal":
        state["legs"] = [
            {"end": state["end"], "travel_mode": "walk"},
            {"end": {"name": "教五", "type": "poi"}, "travel_mode": "bike"},
        ]
    return state


@pytest.fixture
def client():
    import app as app_module

    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.mark.parametrize(
    "route_kind,tool_name",
    [
        ("direct", "plan_route"),
        ("via", "plan_via_route"),
        ("itinerary", "plan_itinerary"),
        ("tour", "plan_tour"),
        ("multimodal", "plan_multimodal_route"),
    ],
)
def test_replan_preserves_route_kind_and_skips_llm(client, route_kind, tool_name):
    state = _state_for(route_kind)
    route_payload = {
        "recommended": [], "distance_m": 0, "mode": "bike",
        "timings_ms": {
            "agent": 91.0,
            "poi_resolution": 1.0,
            "graph_prepare": 2.0,
            "path_search": 3.0,
            "response_build": 4.0,
        },
    }
    if route_kind == "itinerary":
        route_payload["status"] = "feasible"
        route_payload["itinerary"] = {"status": "feasible", "stop_count": 2}

    with patch("agents.planner.run_agent", side_effect=AssertionError("LLM called")), \
         patch("agents.tools.execute_tool", return_value=(route_payload, {"route": route_payload})) as execute:
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"travel_mode": "bike"}},
        )

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["route_state"]["route_kind"] == route_kind
    assert data["route_state"]["travel_mode"] == "bike"
    assert data["route_state"]["start"] == state["start"]
    assert data["route_state"]["end"] == state["end"]
    assert data["route_state"]["hard_constraints"] == state["hard_constraints"]
    assert data["route_state"]["road_condition_version"] == routes._current_road_condition_version()
    assert set(data["timings_ms"]) == {
        "agent", "poi_resolution", "graph_prepare", "path_search", "response_build"
    }
    assert data["timings_ms"]["agent"] == 0.0
    assert execute.call_args.args[0] == tool_name
    if route_kind == "itinerary":
        assert execute.call_args.args[1]["time_budget_min"] == 45
        assert execute.call_args.args[1]["stop_duration_min"] == 8
        assert execute.call_args.args[1]["stops"] == state["via"]["points"]


def test_replan_after_road_condition_update_preserves_route_and_uses_latest_snapshot(client):
    state = _state_for("via")
    state["road_condition_version"] = "old-road-version"
    road_conditions = [{"condition_id": "event-new", "action": "slowdown"}]
    payload = {"recommended": [], "distance_m": 600, "mode": "walk"}

    with patch("api.routes.list_conditions", return_value=road_conditions), \
         patch("api.routes._current_road_condition_version", return_value="new-road-version"), \
         patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})) as execute:
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"refresh_road_conditions": True}},
        )

    assert response.status_code == 200
    updated = response.get_json()["data"]["route_state"]
    assert updated["road_condition_version"] == "new-road-version"
    assert updated["start"] == state["start"]
    assert updated["end"] == state["end"]
    assert updated["via"] == state["via"]
    assert updated["hard_constraints"] == state["hard_constraints"]
    assert execute.call_args.args[2]["_route_conditions_snapshot"] == road_conditions


def test_itinerary_replan_refuses_a_route_that_exceeds_budget(client):
    state = _state_for("itinerary")
    with patch("agents.tools.execute_tool", return_value=(
        {"status": "over_budget", "over_by_min": 4}, None
    )):
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"travel_mode": "bike"}},
        )

    assert response.status_code == 422
    assert response.get_json()["error"] == "itinerary_over_budget"


def test_replan_rejects_stale_data_version_before_running_tool(client):
    state = _state_for("direct")
    state["data_version"] = "stale"

    with patch("agents.tools.execute_tool") as execute:
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"strategy": "flat"}},
        )

    assert response.status_code == 409
    assert response.get_json()["error"] == "route_state_version_conflict"
    execute.assert_not_called()


def test_replan_rejects_client_computed_fields_and_multiple_change(client):
    state = deepcopy(_state_for("direct"))
    state["recommended_edge_ids"] = [[1, 2, 0]]
    state["distance_m"] = 1

    response = client.post(
        "/api/route/replan",
        json={
            "route_state": state,
            "change": {"travel_mode": "bike", "strategy": "flat"},
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_route_state"


def test_chat_context_uses_canonical_route_state_when_no_pending_intent():
    state = _state_for("via")

    context = routes._normalize_chat_context({
        "history": [{"role": "user", "content": "上一轮"}],
        "previous_route_state": state,
    })

    assert context["previous_intent"]["start"] == state["start"]
    assert context["previous_intent"]["end"] == state["end"]
    assert context["previous_intent"]["mode"] == "walk"
    assert context["previous_intent"]["strategy_hint"] == "shortest"
    assert context["constraints"] == state["hard_constraints"]
