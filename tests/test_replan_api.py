from copy import deepcopy
from tests.test_route_reuse import mock_proof_identity  # noqa: F401 - register shared pytest fixture
from unittest.mock import patch

import pytest
import networkx as nx

import api.routes as routes
import agents.tools as tools
from agents.route_state import apply_change, current_data_version, current_road_condition_version
from agents.routing_policy import select_route_strategy


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
        state["end"] = state["legs"][-1]["end"]
    return state


def _verified_replan_state(state, change, conditions=None):
    updated = apply_change(state, change)
    current = updated["strategy"]
    updated["strategy"] = select_route_strategy(
        query=updated["original_query"], explicit_strategy=current["name"],
        strategy_source=current.get("source", "button"),
        custom_weights=current.get("weights"),
    ).as_dict()
    updated["road_condition_version"] = current_road_condition_version(conditions or [])
    return updated


@pytest.fixture
def client():
    import app as app_module

    app_module.app.config["TESTING"] = True
    app_module.app.config["SECRET_KEY"] = "replan-api-test-secret-with-enough-entropy"
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
    change = {"travel_mode": "bike"}
    route_payload = {
        "recommended": [], "distance_m": 0, "mode": "bike",
        "route_state": _verified_replan_state(state, change),
        "route_verification": {"status": "pass"},
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
        route_payload["route_verification"]["itinerary_budget"] = {"status": "pass"}
    if route_kind == "tour":
        route_payload["tour"] = {"mode": "bike"}

    tool_result = ({"status": "feasible"} if route_kind == "itinerary" else route_payload)

    with patch("agents.planner.run_agent", side_effect=AssertionError("LLM called")), \
         patch("api.routes.list_conditions", return_value=[]), \
         patch("agents.tools.execute_tool", return_value=(tool_result, {"route": route_payload})) as execute:
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": change},
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
    produced_state = _verified_replan_state(
        state, {"refresh_road_conditions": True}, road_conditions,
    )
    payload = {"recommended": [], "distance_m": 600, "mode": "walk",
               "route_state": produced_state,
               "route_verification": {"status": "pass"}}

    with patch("api.routes.list_conditions", return_value=road_conditions), \
         patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})) as execute:
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"refresh_road_conditions": True}},
        )

    assert response.status_code == 200
    updated = response.get_json()["data"]["route_state"]
    assert updated["road_condition_version"] == current_road_condition_version(road_conditions)
    assert updated["start"] == state["start"]
    assert updated["end"] == state["end"]
    assert updated["via"] == state["via"]
    assert updated["hard_constraints"] == state["hard_constraints"]
    assert execute.call_args.args[2]["_route_conditions_snapshot"] == road_conditions


def test_replan_forwards_prior_state_only_for_road_refresh_reuse_check(client, monkeypatch, mock_proof_identity):  # noqa: F811
    monkeypatch.setattr(routes.route_reuse_module, "authenticated_proof_identity", lambda: mock_proof_identity)
    state = _state_for("direct")
    certificate = {"proof_id": "fd099645-66eb-4e3a-bb6f-d8504da4b1fb"}
    state["reuse_certificate"] = certificate
    produced = _verified_replan_state(state, {"refresh_road_conditions": True}, [])
    payload = {
        "recommended": [], "distance_m": 10, "mode": "walk",
        "route_state": produced,
        "route_verification": {"status": "pass"},
    }

    with patch("api.routes.has_valid_reuse_certificate", return_value=True), \
         patch("api.routes.list_conditions", return_value=[]), \
         patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})) as execute:
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"refresh_road_conditions": True}},
        )

    assert response.status_code == 200
    context = execute.call_args.args[2]
    assert context["_reuse_route_state"] == state


def test_replan_does_not_forward_invalid_reuse_certificate(client):
    state = _state_for("direct")
    state["reuse_certificate"] = {"signature": "invalid"}
    produced = _verified_replan_state(state, {"refresh_road_conditions": True}, [])
    payload = {
        "recommended": [], "distance_m": 10, "mode": "walk",
        "route_state": produced,
        "route_verification": {"status": "pass"},
    }

    with patch("api.routes.has_valid_reuse_certificate", return_value=False), \
         patch("api.routes.list_conditions", return_value=[]), \
         patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})) as execute:
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"refresh_road_conditions": True}},
        )

    assert response.status_code == 200
    assert "_reuse_route_state" not in execute.call_args.args[2]


def test_api_refresh_reuses_persistent_exact_shortest_path_with_mock_repository(client, monkeypatch, mock_proof_identity):  # noqa: F811
    monkeypatch.setattr(routes.route_reuse_module, "authenticated_proof_identity", lambda: mock_proof_identity)
    graph = nx.MultiDiGraph()
    for node, x, y in (
        (1, 114.3600, 30.5350), (2, 114.3605, 30.5350),
        (3, 114.3610, 30.5350), (4, 114.3603, 30.5355),
    ):
        graph.add_node(node, x=x, y=y)
    graph.add_edge(1, 2, key=0, length=60.0, highway="footway", slope_level=2)
    graph.add_edge(2, 3, key=0, length=40.0, highway="footway", slope_level=2)
    graph.add_edge(1, 4, key=0, length=80.0, highway="footway", slope_level=2)
    graph.add_edge(4, 3, key=0, length=80.0, highway="footway", slope_level=2)
    start = {"type": "coord", "coordinates": {"lng": 114.3600, "lat": 30.5350}}
    end = {"type": "coord", "coordinates": {"lng": 114.3610, "lat": 30.5350}}
    request_body = {"start": start, "end": end, "strategy": "shortest", "travel_mode": "walk"}
    condition = {
        "id": "off-shortest-route", "type": "event", "action": "closure",
        "affected_modes": ["walk"], "start_time": 0, "end_time": 0,
        "edge": {"u": 1, "v": 4, "key": 0, "edges": [[1, 4, 0]]},
    }

    monkeypatch.setattr(routes, "_ensure_network", lambda: (graph, None))
    monkeypatch.setattr(routes, "get_nearest_node", lambda _graph, lng, _lat: 1 if lng < 114.3605 else 3)
    monkeypatch.setattr(routes, "_current_data_version", lambda _graph=None: current_data_version(graph))
    monkeypatch.setattr(routes, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(routes, "list_conditions", lambda **_kwargs: [])
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _graph: (
        1 if ref["coordinates"]["lng"] < 114.3605 else 3,
        "起点" if ref["coordinates"]["lng"] < 114.3605 else "终点", None,
    ))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_args: {
        "status": "unverified_nearby_network_node", "snap_distance_m": 0,
    })
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda _name: (None, []))
    monkeypatch.setattr("spatial.road_conditions._resolve_edge_keys",
                        lambda _graph, _condition, strict=False: {(1, 4, 0)})
    monkeypatch.setattr("spatial.road_conditions._expand_coincident_corridor_edges",
                        lambda _graph, keys, _edge: keys)

    initial = client.post("/api/route", json=request_body)
    assert initial.status_code == 200, initial.get_json()
    state = initial.get_json()["data"]["route_state"]
    assert set(state["reuse_certificate"]) == {"proof_id"}
    monkeypatch.setattr(routes, "list_conditions", lambda **_kwargs: [condition])

    refreshed = client.post(
        "/api/route/replan",
        json={"route_state": state, "change": {"refresh_road_conditions": True}},
    )

    assert refreshed.status_code == 200, refreshed.get_json()
    payload = refreshed.get_json()["data"]
    assert payload["route_verification"]["status"] == "pass"
    assert payload["execution_optimization"]["path_reused"] is True
    assert payload["execution_optimization"]["dijkstra_runs"] == 0
    assert payload["recommended_edge_ids"] == [[1, 2, 0], [2, 3, 0]]

    dijkstra_calls = []
    import spatial.routing as routing_module

    original_run_path = routing_module._run_path

    def count_full_search(*args, **kwargs):
        dijkstra_calls.append(1)
        return original_run_path(*args, **kwargs)

    monkeypatch.setattr(routing_module, "_run_path", count_full_search)
    tampered = deepcopy(state)
    tampered["reuse_certificate"]["proof_id"] = "tampered"
    fallback = client.post(
        "/api/route/replan",
        json={"route_state": tampered, "change": {"refresh_road_conditions": True}},
    )

    assert fallback.status_code == 200, fallback.get_json()
    assert len(dijkstra_calls) == 1
    assert "execution_optimization" not in fallback.get_json()["data"]


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


def test_replan_rejects_tool_route_without_verification(client):
    state = _state_for("direct")
    payload = {"recommended": [], "distance_m": 0, "mode": "walk"}
    with patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})):
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"strategy": "flat"}},
        )
    assert response.status_code == 503
    assert response.get_json()["error"] == "route_verification_failed"


def test_replan_rejects_tool_state_for_different_travel_mode(client):
    state = _state_for("direct")
    produced_state = deepcopy(state)
    produced_state["travel_mode"] = "walk"
    payload = {
        "recommended": [], "distance_m": 0, "mode": "walk",
        "route_state": produced_state,
        "route_verification": {"status": "pass"},
    }
    with patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})):
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"travel_mode": "bike"}},
        )
    assert response.status_code == 503
    assert response.get_json()["error"] == "route_verification_failed"


def test_multimodal_change_applies_mode_to_every_leg():
    from agents.route_state import apply_change

    state = _state_for("multimodal")
    changed = apply_change(state, {"travel_mode": "drive"})
    tool_name, arguments = routes._replan_tool_request(changed)
    assert tool_name == "plan_multimodal_route"
    assert [leg["mode"] for leg in arguments["legs"]] == ["drive", "drive"]


def test_replan_keeps_verified_tour_order_from_tool(client):
    state = _state_for("tour")
    state["tour"]["pois"] = [
        {"name": "老图书馆", "type": "poi"},
        {"name": "樱顶", "type": "poi"},
    ]
    changed = _verified_replan_state(state, {"strategy": "flat"})
    changed["route_id"] = "verified-tool-route"
    changed["tour"]["pois"].reverse()
    changed["end"] = changed["tour"]["pois"][-1]
    payload = {
        "recommended": [], "distance_m": 0,
        "tour": {"mode": "walk"},
        "route_state": changed,
        "route_verification": {"status": "pass"},
    }
    with patch("api.routes.list_conditions", return_value=[]), \
         patch("agents.tools.execute_tool", return_value=(payload, {"route": payload})):
        response = client.post(
            "/api/route/replan",
            json={"route_state": state, "change": {"strategy": "flat"}},
        )
    assert response.status_code == 200
    returned = response.get_json()["data"]["route_state"]
    assert returned["route_id"] == "verified-tool-route"
    assert returned["tour"]["pois"] == changed["tour"]["pois"]
    assert returned["end"] == changed["end"]


def test_replan_direct_uses_real_verified_tool_result(client, monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.3600, y=30.5300)
    graph.add_node(2, x=114.3610, y=30.5300)
    graph.add_edge(1, 2, key=0, length=100.0, highway="footway",
                   slope_level=2, scenery_level=3)
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "_plan_common", lambda _args, _context: (
        graph, graph, "walk", None,
    ))
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _graph: (
        1 if ref["name"] == "珞珈门" else 2, ref["name"], None,
    ))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_args: {
        "status": "unverified_nearby_network_node", "snap_distance_m": 0,
    })
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda _name: (None, []))
    monkeypatch.setattr(routes, "_current_data_version", lambda: current_data_version(graph))
    monkeypatch.setattr(routes, "list_conditions", lambda **_kwargs: [])

    state = _state_for("direct")
    state["data_version"] = current_data_version(graph)
    response = client.post(
        "/api/route/replan",
        json={"route_state": state, "change": {"strategy": "flat"}},
    )
    assert response.status_code == 200, response.get_json()
    payload = response.get_json()["data"]
    assert payload["route_verification"]["status"] == "pass"
    assert payload["recommended_edge_ids"] == [[1, 2, 0]]
    assert payload["route_state"]["strategy"]["name"] == "flat"


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
