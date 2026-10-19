"""P3 API/tool integration against explicit, guarded PostgreSQL storage."""
import uuid

from flask import Flask
import pytest

from agents import route_reuse, tools
from agents.route_state import current_data_version
import api.routes as routes
from storage import database, task_repository
from tests.test_route_reuse import _graph, _state, _payload, _condition

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


@pytest.fixture
def case(monkeypatch):
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute(
            "TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE"
        )
    owner, token, conversation = task_repository.create_conversation(DATABASE_URL)
    graph = _graph()
    app = Flask(__name__)
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    app.register_blueprint(routes.api_bp)
    client = app.test_client()
    client.set_cookie("whu_conversation_token", token)
    monkeypatch.setattr(routes, "_ensure_network", lambda: (graph, None))
    monkeypatch.setattr(routes, "get_nearest_node", lambda _g, lng, _lat: 1 if lng < 114.361 else 4)
    monkeypatch.setattr(routes, "_current_data_version", lambda _graph=None: current_data_version(graph))
    monkeypatch.setattr(routes, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(routes, "list_conditions", lambda **_kw: [])
    monkeypatch.setattr(routes, "_find_pois_along_route", lambda *_a, **_kw: [])
    monkeypatch.setattr(tools, "_ensure_graph", lambda: graph)
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(tools, "_resolve_endpoint", lambda ref, _g: (
        1 if ref["coordinates"]["lng"] < 114.361 else 4, "endpoint", None))
    monkeypatch.setattr(tools, "_endpoint_access_evidence", lambda *_a: {
        "status": "unverified_nearby_network_node", "snap_distance_m": 0})
    monkeypatch.setattr(tools, "find_poi_ambiguous", lambda _name: (None, []))
    monkeypatch.setattr("spatial.routing._should_degrade_annotations", lambda: None)
    body = {"start": {"type": "coord", "coordinates": {"lng": 114.360, "lat": 30.530}},
            "end": {"type": "coord", "coordinates": {"lng": 114.362, "lat": 30.530}},
            "strategy": "shortest", "travel_mode": "walk"}
    return client, graph, body, owner, token, conversation


def initial(case):
    response = case[0].post("/api/route", json=case[2])
    assert response.status_code == 200, response.get_json()
    return response.get_json()["data"]["route_state"]


def refresh(client, state):
    response = client.post("/api/route/replan", json={
        "route_state": state, "change": {"refresh_road_conditions": True},
        "uid": "client-supplied-owner-is-not-authentication"})
    assert response.status_code == 200, response.get_json()
    return response.get_json()["data"]


def test_api_persists_only_an_opaque_handle_and_reuses_across_requests(case):
    state = initial(case)
    assert set(state["reuse_certificate"]) == {"proof_id"}
    proof_id = state["reuse_certificate"]["proof_id"]
    with database.connect(DATABASE_URL) as conn:
        row = conn.execute("SELECT * FROM route_proof WHERE proof_id=%s", (proof_id,)).fetchone()
    assert str(row["owner_id"]) == case[3]
    assert row["selected_edges"] == [[1, 2, 0], [2, 4, 0]]
    result = refresh(case[0], state)
    assert result["execution_optimization"]["path_reused"] is True
    assert result["execution_optimization"]["dijkstra_runs"] == 0
    assert result["route_verification"]["status"] == "pass"


def test_persistent_proof_off_path_closure_matches_full_search(case, monkeypatch):
    state = initial(case)
    event = _condition("closure", (1, 3, 0))
    monkeypatch.setattr(routes, "list_conditions", lambda **_kw: [event])
    monkeypatch.setattr("spatial.road_conditions._resolve_edge_keys", lambda *_a, **_kw: {(1, 3, 0)})
    monkeypatch.setattr("spatial.road_conditions._expand_coincident_corridor_edges", lambda _g, keys, _edge: keys)
    reused = refresh(case[0], state)
    assert reused["execution_optimization"]["path_reused"] is True
    assert reused["execution_optimization"]["dijkstra_runs"] == 0
    assert reused["execution_optimization"]["edge_change_set"]["removed_edge_ids"] == [[1, 3, 0]]
    assert reused["execution_optimization"]["event_change_set"]["added_event_ids"] == ["test-event"]
    state.pop("reuse_certificate")
    full = refresh(case[0], state)
    assert reused["recommended_edge_ids"] == full["recommended_edge_ids"]
    assert reused["recommended_length_m"] == full["recommended_length_m"]
    assert reused["duration_min"] == full["duration_min"]
    assert reused["route_verification"]["status"] == full["route_verification"]["status"] == "pass"


def test_off_path_closure_reuse_does_not_scan_all_graph_edges(case, monkeypatch):
    state = initial(case)
    event = _condition("closure", (1, 3, 0))
    monkeypatch.setattr(routes, "list_conditions", lambda **_kw: [event])
    monkeypatch.setattr("spatial.road_conditions._resolve_edge_keys", lambda *_a, **_kw: {(1, 3, 0)})
    monkeypatch.setattr("spatial.road_conditions._expand_coincident_corridor_edges", lambda _g, keys, _edge: keys)

    def forbidden_full_graph_scan(*_args, **_kwargs):
        raise AssertionError("closure-only refresh should compare the stored closure edge sets")

    monkeypatch.setattr("agents.execution_dependencies.effective_graph_delta", forbidden_full_graph_scan)

    refreshed = refresh(case[0], state)

    assert refreshed["execution_optimization"]["path_reused"] is True
    assert refreshed["execution_optimization"]["dijkstra_runs"] == 0


def test_removed_closure_opens_shortcut_and_forces_full_search(case, monkeypatch):
    state = initial(case)
    closure = {
        "id": "gate-a", "type": "event", "action": "closure",
        "affected_modes": ["walk"], "start_time": 0, "end_time": 0,
        "edge": {"u": 1, "v": 2, "key": 0, "edges": [[1, 2, 0]]},
    }
    monkeypatch.setattr(routes, "list_conditions", lambda **_kw: [closure])
    monkeypatch.setattr("spatial.road_conditions._resolve_edge_keys", lambda *_a, **_kw: {(1, 2, 0)})
    monkeypatch.setattr("spatial.road_conditions._expand_coincident_corridor_edges", lambda _g, keys, _edge: keys)
    closed = refresh(case[0], state)
    assert closed["execution_optimization"]["path_reused"] is False
    assert closed["recommended_length_m"] == 12

    monkeypatch.setattr(routes, "list_conditions", lambda **_kw: [])
    opened = refresh(case[0], closed["route_state"])
    assert opened["execution_optimization"]["path_reused"] is False
    assert opened["execution_optimization"]["reason"] == "new_edge_available"
    assert opened["execution_optimization"]["edge_change_set"]["added_edge_ids"] == [[1, 2, 0]]
    assert opened["execution_optimization"]["event_change_set"]["removed_event_ids"] == ["gate-a"]

    opened_state = dict(opened["route_state"])
    opened_state.pop("reuse_certificate", None)
    full = refresh(case[0], opened_state)
    assert opened["recommended_edge_ids"] == full["recommended_edge_ids"] == [[1, 2, 0], [2, 4, 0]]
    assert opened["recommended_length_m"] == full["recommended_length_m"] == 10
    assert opened["duration_min"] == full["duration_min"]


def test_expired_closure_opens_shortcut_and_forces_full_search(case, monkeypatch, tmp_path):
    import time
    from spatial import road_conditions

    now = time.time()
    clock = [now]
    monkeypatch.setattr(road_conditions.time, "time", lambda: clock[0])
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(tmp_path / "conditions.json"))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", None)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    closure = _condition("closure", (1, 2, 0))
    closure.update({"start_time": now - 10, "end_time": now + 10})
    road_conditions._save_conditions([closure])
    monkeypatch.setattr(routes, "list_conditions", road_conditions.list_conditions)
    monkeypatch.setattr("spatial.road_conditions._resolve_edge_keys", lambda *_a, **_kw: {(1, 2, 0)})
    monkeypatch.setattr("spatial.road_conditions._expand_coincident_corridor_edges", lambda _g, keys, _edge: keys)

    state = initial(case)
    assert state["reuse_certificate"]["proof_id"]
    assert case[1].has_edge(1, 2, 0)
    closed = refresh(case[0], state)
    assert closed["execution_optimization"]["path_reused"] is True

    clock[0] = now + 11
    assert road_conditions.list_conditions() == []
    opened = refresh(case[0], closed["route_state"])
    assert opened["execution_optimization"]["path_reused"] is False
    assert opened["execution_optimization"]["reason"] == "new_edge_available"
    assert opened["recommended_edge_ids"] == [[1, 2, 0], [2, 4, 0]]
    assert opened["recommended_length_m"] == 10


def test_new_slowdown_records_change_and_falls_back_to_full_search(case, monkeypatch):
    state = initial(case)
    slowdown = {
        "id": "work-zone", "type": "event", "action": "slowdown",
        "affected_modes": ["walk"], "cost_multiplier": 1.5,
        "start_time": 0, "end_time": 0,
        "edge": {"u": 1, "v": 3, "key": 0, "edges": [[1, 3, 0]]},
    }
    monkeypatch.setattr(routes, "list_conditions", lambda **_kw: [slowdown])
    monkeypatch.setattr("spatial.road_conditions._resolve_edge_keys", lambda *_a, **_kw: {(1, 3, 0)})
    monkeypatch.setattr("spatial.road_conditions._expand_coincident_corridor_edges", lambda _g, keys, _edge: keys)

    updated = refresh(case[0], state)
    optimization = updated["execution_optimization"]
    assert optimization["path_reused"] is False
    assert optimization["reason"] == "road_slowdown_requires_full_search"
    assert optimization["event_change_set"]["added_event_ids"] == ["work-zone"]
    without_proof = dict(updated["route_state"])
    without_proof.pop("reuse_certificate", None)
    full = refresh(case[0], without_proof)
    assert updated["recommended_edge_ids"] == full["recommended_edge_ids"]
    assert updated["recommended_length_m"] == full["recommended_length_m"]
    assert updated["duration_min"] == full["duration_min"]


@pytest.mark.parametrize("identity", ["missing", "invalid", "foreign"])
def test_missing_invalid_or_foreign_cookie_falls_back(case, identity):
    state = initial(case)
    if identity == "missing":
        case[0].delete_cookie("whu_conversation_token")
    else:
        token = "invalid" if identity == "invalid" else task_repository.create_conversation(DATABASE_URL)[1]
        case[0].set_cookie("whu_conversation_token", token)
    result = refresh(case[0], state)
    assert not result.get("execution_optimization", {}).get("path_reused")
    if identity != "foreign":
        assert "reuse_certificate" not in result["route_state"]


def test_client_owner_uuid_without_cookie_cannot_issue_a_proof(case):
    case[0].delete_cookie("whu_conversation_token")
    case[2]["uid"] = case[3]
    assert "reuse_certificate" not in initial(case)


@pytest.mark.parametrize("mutation", ["digest", "event", "graph", "expired", "invalidated", "unverified", "proof_id", "legacy"])
def test_mismatched_or_unavailable_proof_full_searches(case, mutation, monkeypatch):
    state = initial(case)
    handle = state["reuse_certificate"]
    if mutation in {"digest", "event", "graph", "expired", "invalidated", "unverified"}:
        sql = {"digest": "semantic_digest=repeat('a',64)", "event": "event_version='stale'",
               "graph": "graph_version='stale'", "expired": "created_at=now()-interval '2 minutes', expires_at=now()-interval '1 minute'",
               "invalidated": "invalidated_at=now()", "unverified": "verified=false"}[mutation]
        with database.connect(DATABASE_URL) as conn:
            conn.execute("UPDATE route_proof SET " + sql + " WHERE proof_id=%s", (handle["proof_id"],))
    elif mutation == "proof_id":
        handle["proof_id"] = str(uuid.uuid4())
    else:
        state["reuse_certificate"] = {"signature": "old-hmac", "edge_ids": [[1, 3, 0], [3, 4, 0]]}
    calls = []
    original = tools.compute_route
    def compute(**kwargs):
        calls.append(kwargs.get("reuse_exact_shortest_edges"))
        return original(**kwargs)
    monkeypatch.setattr(tools, "compute_route", compute)
    result = refresh(case[0], state)
    assert calls == [None]
    assert not result.get("execution_optimization", {}).get("path_reused")


def test_task_bound_tool_proof_preserves_binding_and_rejects_current_revision_change(case):
    client, graph, _body, owner, _token, conversation = case
    task_id = task_repository.create_task(DATABASE_URL, owner, conversation, {})
    identity = {"database_url": DATABASE_URL, "owner_id": owner, "task_id": task_id, "task_revision": 0}
    state = initial(case)
    with client.application.app_context():
        state = route_reuse.issue_shortest_path_reuse_certificate(
            state, _payload([[1, 2, 0], [2, 4, 0]]), graph, [], verified=True,
            proof_identity=identity)
    assert state["reuse_certificate"]["task_id"] == task_id
    # API knows owner from cookie and takes task hints only as DB match inputs.
    result = refresh(client, state)
    assert result["execution_optimization"]["path_reused"] is True
    assert result["route_state"]["reuse_certificate"]["task_id"] == task_id
    with database.connect(DATABASE_URL) as conn:
        conn.execute("UPDATE task SET revision=1 WHERE task_id=%s", (task_id,))
    assert not refresh(client, state)["execution_optimization"]["path_reused"]
    with client.application.app_context():
        refused = route_reuse.issue_shortest_path_reuse_certificate(
            _state(graph), _payload([[1, 2, 0], [2, 4, 0]]), graph, [], verified=True,
            proof_identity=identity)
    assert "reuse_certificate" not in refused


def test_database_outage_full_searches_and_omits_new_proof(case, monkeypatch):
    from storage import route_proof_repository
    state = initial(case)
    def outage(*_a, **_kw):
        raise RuntimeError("storage offline")
    monkeypatch.setattr(route_proof_repository, "load_verified_proof", outage)
    monkeypatch.setattr(route_proof_repository, "create_verified_proof", outage)
    result = refresh(case[0], state)
    assert result["execution_optimization"]["path_reused"] is False
    assert result["route_verification"]["status"] == "pass"
    assert "reuse_certificate" not in result["route_state"]


@pytest.mark.parametrize("failure", ["no_database", "cookie_lookup_outage"])
def test_optional_storage_failure_does_not_block_normal_routing(case, monkeypatch, failure):
    if failure == "no_database":
        case[0].application.config["DATABASE_URL"] = None
    else:
        def outage(*_a):
            raise RuntimeError("identity lookup unavailable")
        monkeypatch.setattr(task_repository, "get_owner_by_conversation_token", outage)
    state = initial(case)
    assert "reuse_certificate" not in state
    assert "reuse_certificate" not in refresh(case[0], state)["route_state"]


@pytest.mark.parametrize("change", ["query", "extra_edges", "task_revision", "task_id", "strip_task"])
def test_client_semantics_or_binding_tampering_cannot_reuse(case, change):
    state = initial(case)
    if change in {"task_revision", "task_id", "strip_task"}:
        task_id = task_repository.create_task(DATABASE_URL, case[3], case[5], {})
        state = route_reuse.issue_shortest_path_reuse_certificate(
            state, _payload([[1, 2, 0], [2, 4, 0]]), case[1], [], verified=True,
            proof_identity={"database_url": DATABASE_URL, "owner_id": case[3],
                            "task_id": task_id, "task_revision": 0})
        if change == "task_revision":
            state["reuse_certificate"]["task_revision"] = 1
        elif change == "task_id":
            state["reuse_certificate"]["task_id"] = str(uuid.uuid4())
        else:
            state["reuse_certificate"] = {"proof_id": state["reuse_certificate"]["proof_id"]}
    elif change == "query":
        state["original_query"] = "changed client query"
    else:
        state["reuse_certificate"]["selected_edges"] = [[1, 3, 0], [3, 4, 0]]
    result = refresh(case[0], state)
    assert not result.get("execution_optimization", {}).get("path_reused")


def test_worker_trusted_run_input_issues_task_bound_tool_proof(case, monkeypatch):
    from agents import workflow
    from datetime import datetime, timedelta, timezone
    from langgraph.checkpoint.memory import InMemorySaver
    from jobs.worker import process_next_run
    submitted = task_repository.create_task_run(
        DATABASE_URL, case[3], case[5], "从起点到终点走最短路线",
        str(uuid.uuid4()), datetime.now(timezone.utc) + timedelta(minutes=1))
    def actual_tool_agent(query, *, proof_identity, **_kwargs):
        args = {"start": case[2]["start"], "end": case[2]["end"], "strategy": "shortest"}
        result, artifact = tools.execute_tool("plan_route", args, {
            "query": query, "_route_proof_identity": proof_identity,
            "_route_graph_snapshot": case[1], "_route_conditions_snapshot": [],
            "_route_weather_snapshot": None})
        assert artifact, result
        return {"response_kind": "route", "route": result, "message": "planned"}
    graph = workflow.build_agent_workflow(InMemorySaver(), agent=actual_tool_agent, database_url=DATABASE_URL)
    result = process_next_run(DATABASE_URL, "proof-worker", graph)
    assert result["run_id"] == submitted["run_id"]
    assert result["status"] == "completed"
    completed = task_repository.get_run(DATABASE_URL, case[3], submitted["run_id"])
    route_state = completed["result"]["route_state"]
    assert route_state["reuse_certificate"]["task_revision"] == completed["current_task_revision"]
    assert refresh(case[0], route_state)["execution_optimization"]["path_reused"] is True
    task_repository.apply_task_update(
        DATABASE_URL, case[3], submitted["task_id"], completed["current_task_revision"],
        {"status": "draft"},
    )
    stale = refresh(case[0], route_state)
    assert stale["execution_optimization"]["path_reused"] is False
    with database.connect(DATABASE_URL) as conn:
        row = conn.execute("SELECT * FROM route_proof WHERE task_id=%s", (submitted["task_id"],)).fetchone()
    assert row and str(row["owner_id"]) == case[3]
    assert row["task_revision"] == completed["current_task_revision"]
    assert row["selected_edges"] == [[1, 2, 0], [2, 4, 0]]
    assert completed["status"] == result["status"] == "completed"


@pytest.mark.parametrize("malformation", ["proof_id", "handle_type"])
def test_malformed_optional_worker_proof_does_not_block_route_replan(
    case, malformation, monkeypatch,
):
    from agents import workflow
    from datetime import datetime, timedelta, timezone
    from langgraph.checkpoint.memory import InMemorySaver
    from jobs.worker import process_next_run

    submitted = task_repository.create_task_run(
        DATABASE_URL, case[3], case[5], "从起点到终点走最短路线",
        str(uuid.uuid4()), datetime.now(timezone.utc) + timedelta(minutes=1))

    def actual_tool_agent(query, *, proof_identity, **_kwargs):
        result, _artifact = tools.execute_tool("plan_route", {
            "start": case[2]["start"], "end": case[2]["end"], "strategy": "shortest",
        }, {
            "query": query, "_route_proof_identity": proof_identity,
            "_route_graph_snapshot": case[1], "_route_conditions_snapshot": [],
            "_route_weather_snapshot": None,
        })
        if malformation == "proof_id":
            result["route_state"]["reuse_certificate"]["proof_id"] = "malformed-proof-id"
        else:
            result["route_state"]["reuse_certificate"] = "malformed-proof-handle"
        return {"response_kind": "route", "route": result, "message": "planned"}

    graph = workflow.build_agent_workflow(
        InMemorySaver(), agent=actual_tool_agent, database_url=DATABASE_URL)
    worker_result = process_next_run(DATABASE_URL, "proof-worker", graph)
    assert worker_result["run_id"] == submitted["run_id"]
    completed = task_repository.get_run(DATABASE_URL, case[3], submitted["run_id"])

    assert worker_result["status"] == completed["status"] == "completed"
    assert "reuse_certificate" not in completed["result"]["route_state"]
    from spatial import routing as routing_module

    full_searches = []
    original_run_path = routing_module._run_path

    def count_full_search(*args, **kwargs):
        full_searches.append(1)
        return original_run_path(*args, **kwargs)

    monkeypatch.setattr(routing_module, "_run_path", count_full_search)
    replanned = refresh(case[0], completed["result"]["route_state"])
    assert replanned["route_verification"]["status"] == "pass"
    assert "execution_optimization" not in replanned
    assert full_searches == [1]
