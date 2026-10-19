from copy import deepcopy
import uuid

import networkx as nx
import pytest

from agents import route_reuse
from agents.route_state import build_route_state, current_data_version, current_road_condition_version


def _graph():
    graph = nx.MultiDiGraph()
    for node, x in ((1, 114.360), (2, 114.361), (3, 114.360), (4, 114.362)):
        graph.add_node(node, x=x, y=30.530)
    graph.add_edge(1, 2, key=0, length=5.0, highway="footway")
    graph.add_edge(2, 4, key=0, length=5.0, highway="footway")
    graph.add_edge(1, 3, key=0, length=6.0, highway="footway")
    graph.add_edge(3, 4, key=0, length=6.0, highway="footway")
    return graph


def _state(graph, conditions=(), **overrides):
    values = {
        "route_kind": "direct",
        "original_query": "走最短路线",
        "start": {"name": "起点", "type": "poi"},
        "end": {"name": "终点", "type": "poi"},
        "travel_mode": "walk",
        "hard_constraints": {"slope": "normal"},
        "strategy": {
            "name": "shortest", "source": "button",
            "weights": {"distance": 1.0, "slope": 0.0, "scenery": 0.0},
            "detour_cap": 1.0,
        },
        "data_version": current_data_version(graph),
        "road_condition_version": current_road_condition_version(list(conditions)),
    }
    values.update(overrides)
    return build_route_state(**values)


def _payload(edge_ids, **overrides):
    payload = {
        "recommended_edge_ids": edge_ids,
        "shortest_edge_ids": edge_ids,
        "hard_budget_fallback": False,
        "shortest_baseline_feasible": True,
        "route_verification": {"status": "pass"},
    }
    payload.update(overrides)
    return payload


def _condition(action, edge):
    condition = {
        "id": "test-event", "type": "event", "action": action,
        "affected_modes": ["walk"], "start_time": 0, "end_time": 0,
        "edge": {"u": edge[0], "v": edge[1], "key": edge[2], "edges": [list(edge)]},
    }
    if action == "slowdown":
        condition["cost_multiplier"] = 1.5
    return condition


def _require_reuse_api():
    assert callable(getattr(route_reuse, "issue_shortest_path_reuse_certificate", None))
    assert callable(getattr(route_reuse, "has_valid_reuse_certificate", None))
    assert callable(getattr(route_reuse, "assess_shortest_path_reuse", None))


@pytest.fixture
def mock_proof_identity(monkeypatch):
    """In-memory repository double; persistence is tested separately on PG."""
    from storage import route_proof_repository
    identity = {"database_url": "mock-only", "owner_id": str(uuid.uuid4())}
    rows = {}
    def create(_url, **kwargs):
        proof_id = str(uuid.uuid4())
        rows[proof_id] = deepcopy(kwargs)
        return proof_id
    def load(_url, **kwargs):
        row = rows.get(kwargs.pop("proof_id"))
        return deepcopy(row) if row and all(row.get(key) == value for key, value in kwargs.items()) else None
    monkeypatch.setattr(route_proof_repository, "create_verified_proof", create)
    monkeypatch.setattr(route_proof_repository, "load_verified_proof", load)
    return identity


@pytest.fixture
def app_context(tmp_path, monkeypatch, mock_proof_identity):
    import app as app_module
    import config

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    app_module.app.config["SECRET_KEY"] = "route-reuse-test-secret-with-enough-entropy"
    with app_module.app.app_context():
        yield mock_proof_identity


def test_opaque_handle_loads_server_edges_and_binds_route_semantics(app_context):
    _require_reuse_api()
    graph = _graph()
    state = _state(graph)
    path = [[1, 2, 0], [2, 4, 0]]

    certified = route_reuse.issue_shortest_path_reuse_certificate(
        state, _payload(path), graph, [], verified=True, proof_identity=app_context,
    )

    assert set(certified["reuse_certificate"]) == {"proof_id"}
    assert route_reuse.assess_shortest_path_reuse(
        certified, graph, [], proof_identity=app_context)["edge_ids"] == [tuple(edge) for edge in path]
    changed_path = deepcopy(certified)
    changed_path["reuse_certificate"]["edge_ids"] = [[1, 3, 0]]
    assert not route_reuse.has_valid_reuse_certificate(changed_path)
    changed_task = deepcopy(certified)
    changed_task["end"] = {"name": "other", "type": "poi"}
    assert not route_reuse.assess_shortest_path_reuse(
        changed_task, graph, [], proof_identity=app_context)["reuse_path"]


def test_even_a_valid_legacy_hmac_cannot_authorize_reuse(app_context):
    import hashlib
    import hmac
    from flask import current_app

    graph = _graph()
    state = _state(graph)
    unsigned = {
        "version": 1, "semantic_digest": route_reuse._semantic_digest(state),
        "road_condition_version": state["road_condition_version"],
        "edge_ids": [[1, 2, 0], [2, 4, 0]], "previously_closed_edge_ids": [],
    }
    signature = hmac.new(current_app.config["SECRET_KEY"].encode(),
                         route_reuse._canonical_json(unsigned), hashlib.sha256).hexdigest()
    state["reuse_certificate"] = {**unsigned, "signature": signature}
    assert not route_reuse.has_valid_reuse_certificate(state)
    assert not route_reuse.assess_shortest_path_reuse(
        state, graph, [], proof_identity=app_context)["reuse_path"]


def test_certificate_is_not_issued_for_preference_budget_or_unverified_route(app_context):
    _require_reuse_api()
    graph = _graph()
    path = [[1, 2, 0], [2, 4, 0]]

    for state, payload, verified in (
        (_state(graph, strategy={"name": "flat", "source": "button"}), _payload(path), True),
        (_state(graph, hard_constraints={"max_distance_m": 50}), _payload(path), True),
        (_state(graph), _payload(path, shortest_edge_ids=[[1, 3, 0], [3, 4, 0]]), True),
        (_state(graph), _payload(path, route_verification={"status": "fail"}), True),
        (_state(graph), _payload(path), False),
    ):
        result = route_reuse.issue_shortest_path_reuse_certificate(
            state, payload, graph, [], verified=verified, proof_identity=app_context,
        )
        assert not route_reuse.has_valid_reuse_certificate(result)


def test_off_path_closure_reuses_path_but_used_closure_and_slowdown_fall_back(
    app_context, monkeypatch,
):
    _require_reuse_api()
    graph = _graph()
    path = [[1, 2, 0], [2, 4, 0]]
    state = _state(graph)
    certified = route_reuse.issue_shortest_path_reuse_certificate(
        state, _payload(path), graph, [], verified=True, proof_identity=app_context,
    )
    monkeypatch.setattr(
        "spatial.road_conditions._resolve_edge_keys",
        lambda _graph, condition, strict=False: {
            tuple(condition["edge"]["edges"][0])
        },
    )
    monkeypatch.setattr(
        "spatial.road_conditions._expand_coincident_corridor_edges",
        lambda _graph, keys, _edge: keys,
    )

    off_path = route_reuse.assess_shortest_path_reuse(
        certified, graph, [_condition("closure", (1, 3, 0))],
        proof_identity=app_context, expected_state={**certified, "road_condition_version": current_road_condition_version(
            [_condition("closure", (1, 3, 0))]
        )},
    )
    used_edge = route_reuse.assess_shortest_path_reuse(
        certified, graph, [_condition("closure", (1, 2, 0))],
        proof_identity=app_context, expected_state={**certified, "road_condition_version": current_road_condition_version(
            [_condition("closure", (1, 2, 0))]
        )},
    )
    slowdown = route_reuse.assess_shortest_path_reuse(
        certified, graph, [_condition("slowdown", (1, 3, 0))],
        proof_identity=app_context, expected_state={**certified, "road_condition_version": current_road_condition_version(
            [_condition("slowdown", (1, 3, 0))]
        )},
    )

    assert off_path["reuse_path"] is True, off_path
    assert off_path["edge_ids"] == [(1, 2, 0), (2, 4, 0)]
    assert off_path["change_set"]["removed_edge_ids"] == [[1, 3, 0]]
    assert used_edge["reuse_path"] is False
    assert slowdown["reuse_path"] is False
    assert slowdown["event_change_set"]["added_event_ids"] == ["test-event"]


def test_opening_previously_closed_edge_requires_full_search(app_context, monkeypatch):
    _require_reuse_api()
    graph = _graph()
    previous = [_condition("closure", (1, 3, 0))]
    monkeypatch.setattr(
        "spatial.road_conditions._resolve_edge_keys",
        lambda _graph, condition, strict=False: {
            tuple(condition["edge"]["edges"][0])
        },
    )
    monkeypatch.setattr(
        "spatial.road_conditions._expand_coincident_corridor_edges",
        lambda _graph, keys, _edge: keys,
    )
    state = _state(graph, previous)
    certified = route_reuse.issue_shortest_path_reuse_certificate(
        state, _payload([[1, 2, 0], [2, 4, 0]]), graph, previous, verified=True, proof_identity=app_context,
    )

    decision = route_reuse.assess_shortest_path_reuse(
        certified, graph, [], proof_identity=app_context, expected_state={
            **certified, "road_condition_version": current_road_condition_version([]),
        },
    )

    assert decision["reuse_path"] is False
    assert decision["reason"] == "new_edge_available"
    assert decision["event_change_set"]["removed_event_ids"] == ["test-event"]
