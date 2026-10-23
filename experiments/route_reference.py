"""Independent graph reachability reference for authored campus route tasks.

This checks the recorded graph, mode, steps and ordered stops without invoking
the product router. Nearest-node access and on-site passability remain unknown.
"""
from __future__ import annotations

import math
from typing import Any

import networkx as nx

from experiments.candidate_reference import _meters, _point
from experiments.scenario_v2 import _access_denied, _blocked_modes


def _mode_graph(graph: nx.MultiDiGraph, mode: str, *, avoid_steps: bool) -> nx.DiGraph:
    if mode not in {"walk", "bike", "drive"}:
        raise ValueError("unsupported travel mode")
    eligible = nx.DiGraph()
    eligible.add_nodes_from(str(node) for node in graph.nodes)
    for u, v, _key, attrs in graph.edges(keys=True, data=True):
        if (_access_denied(graph.nodes[u], mode, node=True)
                or _access_denied(graph.nodes[v], mode, node=True)
                or _access_denied(attrs, mode)):
            continue
        if avoid_steps and "steps" in str(attrs.get("highway", "")):
            continue
        allowed = attrs.get("allowed_modes")
        allowed_modes = _blocked_modes(allowed) if allowed is not None else None
        blocked = _blocked_modes(attrs.get("blocked_modes"))
        if ((allowed is not None and (allowed_modes is None or mode not in allowed_modes))
                or blocked is None or mode in blocked or "all" in blocked):
            continue
        try:
            length = float(attrs["length"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(length) or length <= 0:
            continue
        a, b = str(u), str(v)
        existing = eligible.get_edge_data(a, b)
        if existing is None or length < existing["weight"]:
            eligible.add_edge(a, b, weight=length)
    return eligible


def build_route_reference(graph: nx.MultiDiGraph, pois: list[dict[str, Any]], *,
                          start_poi_id: str, end_poi_id: str,
                          via_poi_ids: list[str], mode: str,
                          constraints: dict[str, Any],
                          coordinates: str = "gcj02") -> dict[str, Any]:
    """Return graph feasibility and explicit nearest-node access uncertainty."""
    if set(constraints) - {"avoid_steps"}:
        raise ValueError("unsupported independent hard constraint")
    by_id = {poi.get("id") or poi.get("poi_id"): poi for poi in pois}
    ids = [start_poi_id, *via_poi_ids, end_poi_id]
    if any(poi_id not in by_id for poi_id in ids):
        raise ValueError("route reference POI missing")
    eligible = _mode_graph(graph, mode, avoid_steps=bool(constraints.get("avoid_steps")))
    nodes = [(str(node), (float(attrs["x"]), float(attrs["y"])))
             for node, attrs in graph.nodes(data=True) if eligible.degree(str(node)) > 0]
    if not nodes:
        raise ValueError("mode has no georeferenced routable nodes")
    snapped = []
    for poi_id in ids:
        location = _point(by_id[poi_id], coordinates)
        node, access = min(((node, _meters(location, xy)) for node, xy in nodes),
                           key=lambda pair: pair[1])
        snapped.append({"poi_id": poi_id, "node": node,
                        "access_m": round(access, 2),
                        "access_over_30m": access > 30})
    lengths = []
    for origin, destination in zip(snapped, snapped[1:]):
        try:
            lengths.append(nx.shortest_path_length(eligible, origin["node"],
                                                   destination["node"], weight="weight"))
        except nx.NetworkXNoPath:
            lengths.append(None)
    feasible = all(length is not None for length in lengths)
    return {
        "start_node": snapped[0]["node"], "end_node": snapped[-1]["node"],
        "via_nodes": [row["node"] for row in snapped[1:-1]],
        "snapped_pois": snapped, "graph_feasible": feasible,
        "graph_distance_m": round(sum(lengths), 2) if feasible else None,
        "access_uncertain": any(row["access_over_30m"] for row in snapped),
        "segments_m": [round(length, 2) if length is not None else None
                       for length in lengths],
        "method": "independent_mode_filtered_node_graph_and_nearest_poi",
        "status": "authoring_reference_not_field_truth",
    }
