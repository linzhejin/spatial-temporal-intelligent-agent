"""Persistent, owner-bound evidence for conservative shortest-path reuse."""

from __future__ import annotations

import hashlib
import json
import uuid

from flask import current_app, request

from agents.execution_dependencies import (
    can_reuse_exact_shortest_path_after_closure_delta,
    condition_change_set,
    condition_snapshot,
)
from agents.route_state import current_data_version, current_road_condition_version, validate_route_state


_TASK_FIELDS = (
    "route_kind", "original_query", "start", "end", "via", "itinerary",
    "tour", "legs", "travel_mode", "hard_constraints", "strategy",
)
_BUDGET_FIELDS = {"max_distance_m", "max_duration_s"}


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def authenticated_proof_identity() -> dict | None:
    """Resolve anonymous ownership only from the validated HttpOnly cookie."""
    try:
        url = current_app.config.get("DATABASE_URL")
        token = request.cookies.get("whu_conversation_token")
        if not url or not token:
            return None
        from storage.task_repository import get_owner_by_conversation_token
        owner = get_owner_by_conversation_token(url, token)
        return {"database_url": url, "owner_id": owner} if owner else None
    except Exception:
        # Proofs are optional; a storage outage must never prevent full routing.
        return None


def _semantic_state(raw_state) -> dict:
    state = validate_route_state(raw_state)
    state.pop("reuse_certificate", None)
    return state


def _semantic_digest(raw_state) -> str:
    return hashlib.sha256(_canonical_json(_semantic_state(raw_state))).hexdigest()


def _valid_edge_ids(value, *, allow_empty=False) -> bool:
    if not isinstance(value, list) or (not allow_empty and not value):
        return False
    return all(
        isinstance(edge, (list, tuple)) and len(edge) == 3
        for edge in value
    )


def _eligibility_error(state: dict) -> str | None:
    if state.get("route_kind") != "direct":
        return "unsupported_route_kind"
    if state.get("strategy", {}).get("name") != "shortest":
        return "unsupported_strategy"
    constraints = state.get("hard_constraints") or {}
    if any(key in constraints and constraints[key] is not None for key in _BUDGET_FIELDS):
        return "route_budget_present"
    return None


def has_valid_reuse_certificate(raw_state) -> bool:
    """Check handle shape only; authorization always requires a database load.

    The historical name is kept for callers. Legacy HMAC certificates are not
    proof handles, even if their signatures would have been valid.
    """
    try:
        handle = validate_route_state(raw_state).get("reuse_certificate")
        if not isinstance(handle, dict) or set(handle) not in (
            {"proof_id"}, {"proof_id", "task_id", "task_revision"},
        ):
            return False
        for key in ("proof_id", "task_id"):
            if key in handle and str(uuid.UUID(handle[key])) != handle[key]:
                return False
        if "task_revision" in handle:
            return type(handle["task_revision"]) is int and 0 <= handle["task_revision"] <= 2**63 - 1
        return True
    except (TypeError, ValueError, AttributeError):
        return False


def _json_edge_ids(edges) -> list[list]:
    return [list(edge) for edge in edges]


def issue_shortest_path_reuse_certificate(
    raw_state, route_payload: dict, graph, conditions: list, *, verified: bool,
    proof_identity: dict | None = None,
) -> dict:
    """Persist exact verified evidence, exposing only its opaque database handle."""
    state = _semantic_state(raw_state)
    eligibility_error = _eligibility_error(state)
    if (not isinstance(proof_identity, dict) or not proof_identity.get("owner_id")
            or not proof_identity.get("database_url")
            or eligibility_error or verified is not True
            or not isinstance(route_payload, dict)
            or (route_payload.get("route_verification") or {}).get("status") != "pass"
            or route_payload.get("hard_budget_fallback") is True
            or route_payload.get("shortest_baseline_feasible") is False):
        return state

    recommended = route_payload.get("recommended_edge_ids")
    shortest = route_payload.get("shortest_edge_ids")
    if recommended is None:
        recommended = route_payload.get("recommended_edges")
    if shortest is None:
        shortest = route_payload.get("shortest_edges")
    if not _valid_edge_ids(recommended) or not _valid_edge_ids(shortest):
        return state
    recommended = _json_edge_ids(recommended)
    shortest = _json_edge_ids(shortest)
    if recommended != shortest:
        return state

    data_version = current_data_version(graph)
    condition_version = current_road_condition_version(conditions)
    if (state.get("data_version") != data_version
            or state.get("road_condition_version") != condition_version):
        return state

    try:
        from spatial.road_conditions import apply_conditions_to_graph
        from spatial.routing_index import get_routing_index

        mode_graph = get_routing_index(graph).for_mode(state["travel_mode"]).graph
        _effective_graph, penalties, closed_edges, _count = apply_conditions_to_graph(
            mode_graph, conditions, state["travel_mode"], strict=True,
        )
    except Exception:
        return state
    if penalties:
        return state

    try:
        from storage.route_proof_repository import create_verified_proof
        task_binding = {key: proof_identity.get(key) for key in ("task_id", "task_revision")}
        proof_id = create_verified_proof(
            proof_identity["database_url"], owner_id=proof_identity["owner_id"],
            semantic_digest=_semantic_digest(state), graph_version=data_version,
            event_version=condition_version, selected_edges=recommended,
            closed_edge_ids=_json_edge_ids(sorted(closed_edges, key=lambda edge: tuple(map(repr, edge)))),
            ttl_seconds=3600, event_snapshot=condition_snapshot(conditions), **task_binding,
        )
    except Exception:
        return state
    handle = {"proof_id": proof_id}
    if task_binding["task_id"] is not None:
        handle.update(task_binding)
    state["reuse_certificate"] = handle
    return state


def _same_route_task(prior: dict, expected: dict) -> bool:
    try:
        prior_state = _semantic_state(prior)
        expected_state = _semantic_state(expected)
    except (TypeError, ValueError):
        return False
    return all(prior_state.get(field) == expected_state.get(field) for field in _TASK_FIELDS)


def assess_shortest_path_reuse(
    raw_state, graph, conditions: list, *, expected_state=None,
    proof_identity: dict | None = None,
) -> dict:
    """Prove an old direct shortest path is still optimal after route refresh."""
    def decline(reason, change_set=None, event_change_set=None):
        return {"reuse_path": False, "reason": reason, "edge_ids": None,
                "change_set": change_set, "event_change_set": event_change_set}

    try:
        state = validate_route_state(raw_state)
    except (TypeError, ValueError):
        return decline("invalid_route_state")
    eligibility_error = _eligibility_error(state)
    if eligibility_error:
        return decline(eligibility_error)
    if not has_valid_reuse_certificate(state):
        return decline("missing_or_invalid_proof_handle")
    if not isinstance(proof_identity, dict) or not proof_identity.get("owner_id") or not proof_identity.get("database_url"):
        return decline("missing_authenticated_owner")
    if expected_state is not None and not _same_route_task(state, expected_state):
        return decline("route_task_changed")
    if state.get("data_version") != current_data_version(graph):
        return decline("stale_data_version")
    if expected_state is not None:
        expected = validate_route_state(expected_state)
        if expected.get("road_condition_version") != current_road_condition_version(conditions):
            return decline("road_condition_snapshot_mismatch")

    handle = state["reuse_certificate"]
    task_binding = {key: handle.get(key) for key in ("task_id", "task_revision")}
    if any(key in proof_identity and proof_identity[key] != task_binding[key] for key in task_binding):
        return decline("task_binding_mismatch")
    try:
        from storage.route_proof_repository import load_verified_proof
        certificate = load_verified_proof(
            proof_identity["database_url"], proof_id=handle["proof_id"],
            owner_id=proof_identity["owner_id"], semantic_digest=_semantic_digest(state),
            graph_version=current_data_version(graph), event_version=state["road_condition_version"],
            **task_binding,
        )
        if certificate is None:
            return decline("proof_not_current_or_not_owned")
        edge_ids = [tuple(edge) for edge in certificate["selected_edges"]]
        event_changes = condition_change_set(certificate.get("event_snapshot", []), conditions)
    except Exception:
        return decline("proof_storage_unavailable")
    if not edge_ids:
        return decline("invalid_old_path")
    start_node, end_node = edge_ids[0][0], edge_ids[-1][1]

    try:
        from spatial.road_conditions import resolve_condition_effects
        from spatial.routing_index import get_routing_index

        base_graph = get_routing_index(graph).for_mode(state["travel_mode"]).graph
        penalties, closed_edges, _count = resolve_condition_effects(
            base_graph, conditions, state["travel_mode"], strict=True,
        )
    except Exception:
        return decline("effective_graph_unavailable")

    if penalties:
        closure_delta = can_reuse_exact_shortest_path_after_closure_delta(
            old_edge_ids=edge_ids,
            start_node=start_node,
            end_node=end_node,
            previously_closed_edge_ids=certificate["closed_edge_ids"],
            current_closed_edge_ids=closed_edges,
            strategy=state["strategy"]["name"],
            old_exact_optimality_proven=True,
            same_mode_constraints_and_endpoints=True,
        )
        return decline(
            "road_slowdown_requires_full_search",
            closure_delta.get("change_set"), event_changes,
        )

    proof = can_reuse_exact_shortest_path_after_closure_delta(
        old_edge_ids=edge_ids,
        start_node=start_node,
        end_node=end_node,
        previously_closed_edge_ids=certificate["closed_edge_ids"],
        current_closed_edge_ids=closed_edges,
        strategy=state["strategy"]["name"],
        old_exact_optimality_proven=True,
        same_mode_constraints_and_endpoints=True,
    )
    if not proof["reuse_path"]:
        return decline(proof["reason"], proof.get("change_set"), event_changes)
    return {
        "reuse_path": True,
        "reason": proof["reason"],
        "change_set": proof.get("change_set"),
        "event_change_set": event_changes,
        "edge_ids": edge_ids,
        "requires_reverification": True,
        "proof_identity": {**proof_identity, **task_binding},
    }
