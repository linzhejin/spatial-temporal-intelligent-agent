"""Self-contained simulated-event scorer with explicit no-path evidence.

The v1 scorer and v1/v2 packets remain unchanged for historical replay.
This module is the complete event scorer included in the v3 source inventory.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from experiments.controlled_event_cases import replay_event_case
from experiments.controlled_route_replay import score_controlled_route_case
from experiments.independent_source_snapshot import active_closed_edges


_ROUTE_TOOLS = {"plan_route", "plan_via_route", "plan_itinerary",
                "plan_tour", "plan_multimodal_route"}


def _active_event_ids(snapshot: dict[str, Any], mode: str,
                      closed: set[tuple[Any, Any, Any]]) -> list[str]:
    epoch = snapshot["as_of_epoch"]
    return sorted(item["id"] for item in snapshot["conditions"]
                  if item["action"] == "closure"
                  and mode in item["affected_modes"]
                  and item["start_time"] <= epoch
                  and (item["end_time"] == 0 or epoch < item["end_time"])
                  and (item.get("revoked_at") is None or epoch < item["revoked_at"])
                  and (item["edge"]["u"], item["edge"]["v"],
                       item["edge"]["key"]) in closed)


def score_controlled_event_case(
    case: dict[str, Any], observed: dict[str, Any] | None,
    graph: Any, anchors: dict[str, dict[str, Any]],
    edge_evidence: dict[str, set[tuple[Any, Any, Any]]],
    event_snapshot: dict[str, Any], *, template: str,
) -> dict[str, Any]:
    """Score exact active closures and require typed graph-no-path evidence."""
    if case.get("category") not in {"compound_tools", "event_infeasible"}:
        raise ValueError("unsupported controlled event category")
    replay_event_case(case, graph, anchors, edge_evidence, event_snapshot,
                      template=template)
    mode = case["gold_derivation"]["parameters"]["mode"]
    closed = active_closed_edges(graph, event_snapshot, mode)
    projected = deepcopy(case)
    projected["category"] = "mode_access"
    score = score_controlled_route_case(
        projected, observed, graph, anchors,
        permitted_edge_ids=edge_evidence["permitted"],
        unknown_edge_ids=edge_evidence["unknown"],
        closed_edge_ids=closed,
    )
    if not isinstance(observed, dict):
        return score
    turns = observed.get("turns")
    actual = turns[0] if isinstance(turns, list) and len(turns) == 1 else None
    trace = actual.get("tool_trace") if isinstance(actual, dict) else None
    trace = trace if isinstance(trace, list) else []
    query_indexes = [index for index, call in enumerate(trace)
                     if isinstance(call, dict)
                     and call.get("tool_name") == "list_road_conditions"
                     and call.get("outcome") == "success"]
    route_indexes = [index for index, call in enumerate(trace)
                     if isinstance(call, dict)
                     and call.get("tool_name") in _ROUTE_TOOLS]
    successful_route = any(trace[index].get("outcome") == "success"
                           for index in route_indexes)
    if not query_indexes:
        score["failures"].append("road_query_missing")
    if route_indexes and query_indexes and min(query_indexes) > min(route_indexes):
        score["failures"].append("road_query_after_route")
    event_ids = actual.get("road_event_ids") if isinstance(actual, dict) else None
    if not isinstance(event_ids, list) or not event_ids:
        score["failures"].append("road_event_ids_missing")
    elif (any(not isinstance(value, str) or not value for value in event_ids)
          or len(set(event_ids)) != len(event_ids)
          or sorted(event_ids) != _active_event_ids(event_snapshot, mode, closed)):
        score["failures"].append("road_event_mismatch")
    if case["category"] == "event_infeasible":
        params = case["gold_derivation"]["parameters"]
        request = {
            "start_node": int(anchors[params["start_poi_id"]]["node_id"]),
            "end_node": int(anchors[params["end_poi_id"]]["node_id"]),
            "mode": params["mode"],
            "avoid_steps": params["avoid_steps"],
        }
        encoded = json.dumps(request, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        expected_digest = hashlib.sha256(encoded).hexdigest()
        no_path_attempt = any(
            isinstance(call, dict)
            and call.get("tool_name") in _ROUTE_TOOLS
            and call.get("outcome") == "tool_error"
            and call.get("error_code") == "graph_no_path"
            and {"start", "end"}.issubset(call.get("argument_keys") or [])
            and call.get("request_digest") == expected_digest
            for call in trace
        )
        if not no_path_attempt:
            score["failures"].append("graph_no_path_evidence_missing")
    elif not successful_route:
        score["failures"].append("route_plan_trace_missing")
    score["task_success"] = not score["failures"] and not score["unknowns"]
    return score
