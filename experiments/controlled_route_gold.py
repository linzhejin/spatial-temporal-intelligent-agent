"""Independent graph-only existence and distance-budget oracle.

Only explicitly permitted frozen edges establish a feasible route. Unknown
edges can make an apparent disconnection or budget failure undecidable.
This module neither calls the product router nor asserts field passability.
"""
from __future__ import annotations

import math
from typing import Any

import networkx as nx

from experiments.independent_route_oracle import _highways


def _edge_order(edge: tuple[Any, Any, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((type(part).__name__, str(part)) for part in edge)


def _add_min_edge(graph: nx.DiGraph, u: Any, v: Any, key: Any,
                  weight: float, *, uncertain: bool) -> None:
    rank = (weight, _edge_order((u, v, key)))
    existing = graph.get_edge_data(u, v)
    if existing is None or rank < existing["rank"]:
        graph.add_edge(u, v, weight=weight, edge_id=(u, v, key),
                       uncertain=uncertain, rank=rank)


def _ordered_path(graph: nx.DiGraph, stops: list[Any]) -> tuple[list[tuple[Any, Any, Any]], float] | None:
    edges = []
    length = 0.0
    for start, end in zip(stops, stops[1:]):
        try:
            nodes = nx.shortest_path(graph, start, end, weight="weight")
        except nx.NetworkXNoPath:
            return None
        for u, v in zip(nodes, nodes[1:]):
            attrs = graph[u][v]
            edges.append(attrs["edge_id"])
            length += attrs["weight"]
    return edges, length


def derive_graph_route(
    graph: Any, *, start_node: Any, end_node: Any, via_nodes: list[Any],
    mode: str, permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]],
    avoid_steps: bool = False, max_distance_m: float | None = None,
) -> dict[str, Any]:
    """Derive a route, or explain why feasibility is unknown/infeasible."""
    if mode not in {"walk", "bike", "drive"}:
        raise ValueError("unsupported travel mode")
    if not isinstance(via_nodes, list) or any(node not in graph for node in
                                                 [start_node, *via_nodes, end_node]):
        raise ValueError("all route nodes must exist")
    if not all(isinstance(value, set) for value in
               (permitted_edge_ids, unknown_edge_ids, closed_edge_ids)):
        raise ValueError("explicit edge-evidence sets required")
    if permitted_edge_ids & unknown_edge_ids:
        raise ValueError("permitted and unknown evidence overlap")
    for edge in permitted_edge_ids | unknown_edge_ids | closed_edge_ids:
        if (not isinstance(edge, tuple) or len(edge) != 3
                or not graph.has_edge(*edge)):
            raise ValueError("edge evidence references missing graph edge")
    if (max_distance_m is not None and
            (isinstance(max_distance_m, bool) or not isinstance(max_distance_m, (int, float))
             or not math.isfinite(max_distance_m) or max_distance_m <= 0)):
        raise ValueError("max_distance_m must be positive and finite")

    confirmed = nx.DiGraph()
    potential = nx.DiGraph()
    confirmed.add_nodes_from(graph.nodes)
    potential.add_nodes_from(graph.nodes)
    for u, v, key in sorted(permitted_edge_ids | unknown_edge_ids, key=_edge_order):
        edge = (u, v, key)
        if edge in closed_edge_ids:
            continue
        attrs = graph[u][v][key]
        highways = _highways(attrs.get("highway")) if avoid_steps else set()
        if avoid_steps and highways is not None and "steps" in highways:
            continue
        raw_length = attrs.get("length")
        known_length = (not isinstance(raw_length, bool)
                        and isinstance(raw_length, (int, float))
                        and math.isfinite(raw_length) and raw_length >= 0)
        uncertain = (edge in unknown_edge_ids or not known_length
                     or (avoid_steps and not highways))
        _add_min_edge(potential, u, v, key, 0 if uncertain else float(raw_length),
                      uncertain=uncertain)
        if not uncertain:
            _add_min_edge(confirmed, u, v, key, float(raw_length), uncertain=False)

    stops = [start_node, *via_nodes, end_node]
    confirmed_result = _ordered_path(confirmed, stops)
    if confirmed_result is not None:
        edges, length = confirmed_result
        if max_distance_m is None or length <= max_distance_m + 1e-6:
            return {"scope": "frozen_graph_only", "verdict": "feasible",
                    "reason": "confirmed_edge_path", "route_edge_ids": [list(edge) for edge in edges],
                    "length_m": round(length, 3)}
    potential_result = _ordered_path(potential, stops)
    if potential_result is not None:
        _, lower_bound = potential_result
        if max_distance_m is None or lower_bound <= max_distance_m + 1e-6:
            reason = ("budget_may_use_unknown_edges" if confirmed_result is not None
                      else "possible_only_via_unknown_edges")
            return {"scope": "frozen_graph_only", "verdict": "unknown",
                    "reason": reason, "route_edge_ids": [], "length_m": None}
    reason = ("distance_budget_exceeded" if confirmed_result is not None
              else "no_path_in_frozen_eligible_graph")
    return {"scope": "frozen_graph_only", "verdict": "infeasible",
            "reason": reason, "route_edge_ids": [], "length_m": None}
