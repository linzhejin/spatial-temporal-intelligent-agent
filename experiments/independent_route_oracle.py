"""Narrow independent graph-level check for projected route edges.

This module does not call the product planner, graph filter or result verifier.
Callers must supply mode eligibility and active closures from a separately
frozen source. It cannot establish real-world road passability by itself.
"""
from __future__ import annotations

import ast
import math
from typing import Any


def _edge(raw: Any) -> tuple[Any, Any, Any] | None:
    if (not isinstance(raw, (list, tuple)) or len(raw) != 3
            or any(isinstance(part, (list, tuple, dict, set, bool)) for part in raw)):
        return None
    try:
        hash(tuple(raw))
    except TypeError:
        return None
    return tuple(raw)


def _highways(raw: Any) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.startswith("["):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if isinstance(raw, (list, tuple, set)):
        return {str(value).lower().strip() for value in raw}
    return {str(raw).lower().strip()}


def _ordered_nodes(path: list[Any], vias: list[Any]) -> bool:
    cursor = 0
    for via in vias:
        try:
            cursor = path.index(via, cursor) + 1
        except ValueError:
            return False
    return True


def inspect_route_edges(
    graph: Any, route_edge_ids: list[Any], *,
    start_node: Any, end_node: Any, via_nodes: list[Any], mode: str,
    permitted_edge_ids: set[tuple[Any, Any, Any]] | None,
    closed_edge_ids: set[tuple[Any, Any, Any]] | None,
    uncertain_edge_ids: set[tuple[Any, Any, Any]] | None = None,
    avoid_steps: bool = False, max_distance_m: float | None = None,
) -> dict[str, Any]:
    """Return pass/fail/unknown only for facts supported by frozen graph inputs."""
    if mode not in {"walk", "bike", "drive"}:
        raise ValueError("unsupported travel mode")
    if not isinstance(via_nodes, list) or not isinstance(route_edge_ids, list):
        raise ValueError("route edges and via nodes must be lists")
    if (max_distance_m is not None
            and (isinstance(max_distance_m, bool) or not isinstance(max_distance_m, (int, float))
                 or not math.isfinite(max_distance_m) or max_distance_m <= 0)):
        raise ValueError("max_distance_m must be positive and finite")
    issues: list[str] = []
    unknowns: list[str] = []

    def issue(code: str) -> None:
        if code not in issues:
            issues.append(code)

    def unknown(code: str) -> None:
        if code not in unknowns:
            unknowns.append(code)

    edges = [_edge(raw) for raw in route_edge_ids]
    if any(edge is None for edge in edges):
        issue("invalid_edge_id")
        return {"scope": "frozen_graph_only", "mode": mode, "verdict": "fail",
                "issues": issues, "unknowns": unknowns, "length_m": None}
    if not edges:
        if start_node != end_node or via_nodes:
            issue("missing_edge_sequence")
        path = [start_node]
    else:
        path = [edges[0][0]] + [edge[1] for edge in edges]
        if path[0] != start_node or path[-1] != end_node:
            issue("endpoint_mismatch")
        if any(left[1] != right[0] for left, right in zip(edges, edges[1:])):
            issue("discontinuous_edges")
    if not _ordered_nodes(path, via_nodes):
        issue("ordered_via_mismatch")
    if permitted_edge_ids is None:
        unknown("mode_evidence_missing")
    if closed_edge_ids is None:
        unknown("event_snapshot_missing")

    length = 0.0
    length_known = True
    for u, v, key in edges:
        edge = (u, v, key)
        if not graph.has_edge(u, v, key):
            issue("missing_edge")
            length_known = False
            continue
        if permitted_edge_ids is not None and edge not in permitted_edge_ids:
            if uncertain_edge_ids is not None and edge in uncertain_edge_ids:
                unknown("mode_edge_uncertain")
            else:
                issue("mode_forbidden_edge")
        if closed_edge_ids is not None and edge in closed_edge_ids:
            issue("closed_edge_used")
        attrs = graph[u][v][key]
        if avoid_steps:
            highways = _highways(attrs.get("highway"))
            if highways is None or not highways:
                unknown("step_tag_missing")
            elif "steps" in highways:
                issue("marked_steps_used")
        raw_length = attrs.get("length")
        if (isinstance(raw_length, bool) or not isinstance(raw_length, (int, float))
                or not math.isfinite(raw_length) or raw_length < 0):
            unknown("edge_length_missing")
            length_known = False
        else:
            length += raw_length
    if max_distance_m is not None:
        if not length_known:
            unknown("distance_budget_unverifiable")
        elif length > max_distance_m + 1e-6:
            issue("distance_budget_exceeded")
    return {
        "scope": "frozen_graph_only", "mode": mode,
        "verdict": "fail" if issues else "unknown" if unknowns else "pass",
        "issues": issues, "unknowns": unknowns,
        "length_m": round(length, 3) if length_known else None,
    }
