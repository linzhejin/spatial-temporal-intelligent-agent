"""Independent graph-distance reference for source-tagged POI candidates.

This is an authoring aid, not field truth or automatic formal gold. It uses
the frozen graph and POI activity tags without calling the product recommender.
"""
from __future__ import annotations

import math
from typing import Any

import networkx as nx

from experiments.scenario_v2 import _access_denied, _blocked_modes
from spatial.coord_transform import gcj02_to_wgs84


def _meters(a: tuple[float, float], b: tuple[float, float]) -> float:
    mean_lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * 111_320 * math.cos(mean_lat),
                      (a[1] - b[1]) * 111_320)


def _point(poi: dict[str, Any], coordinates: str) -> tuple[float, float]:
    raw = poi["coordinates"]
    lng, lat = float(raw["lng"]), float(raw["lat"])
    if coordinates == "gcj02":
        return gcj02_to_wgs84(lng, lat)
    if coordinates == "wgs84":
        return lng, lat
    raise ValueError("coordinates must be gcj02 or wgs84")


def _walk_graph(graph: nx.MultiDiGraph,
                closed_edges: set[str]) -> nx.DiGraph:
    walk = nx.DiGraph()
    walk.add_nodes_from(str(node) for node in graph.nodes)
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        if f"{u}|{v}|{key}" in closed_edges:
            continue
        if (_access_denied(graph.nodes[u], "walk", node=True)
                or _access_denied(graph.nodes[v], "walk", node=True)
                or _access_denied(attrs, "walk")):
            continue
        allowed = attrs.get("allowed_modes")
        allowed_modes = _blocked_modes(allowed) if allowed is not None else None
        blocked_modes = _blocked_modes(attrs.get("blocked_modes"))
        if ((allowed is not None and (allowed_modes is None or "walk" not in allowed_modes))
                or blocked_modes is None or "walk" in blocked_modes or "all" in blocked_modes):
            continue
        try:
            length = float(attrs["length"])
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isfinite(length) or length <= 0:
            continue
        source, target = str(u), str(v)
        existing = walk.get_edge_data(source, target)
        if existing is None or length < existing["weight"]:
            walk.add_edge(source, target, weight=length)
    return walk


def rank_nearby_activities(
    graph: nx.MultiDiGraph, pois: list[dict[str, Any]], *,
    start_poi_id: str, activities: set[str], coordinates: str = "gcj02",
    closed_edges: set[str] | None = None,
) -> dict[str, Any]:
    """Rank every supported venue by walk distance plus approximate end access."""
    if not activities:
        raise ValueError("activities cannot be empty")
    by_id = {poi.get("id") or poi.get("poi_id"): poi for poi in pois}
    if start_poi_id not in by_id:
        raise ValueError("start POI missing")
    walk = _walk_graph(graph, closed_edges or set())
    node_points = [(str(node), (float(attrs["x"]), float(attrs["y"])))
                   for node, attrs in graph.nodes(data=True)
                   if walk.degree(str(node)) > 0]
    if not node_points:
        raise ValueError("graph has no walk-routable nodes")

    def nearest(point: tuple[float, float]) -> tuple[str, float]:
        return min(((node, _meters(point, coords)) for node, coords in node_points),
                   key=lambda pair: pair[1])

    origin_node, origin_access = nearest(_point(by_id[start_poi_id], coordinates))
    distances = nx.single_source_dijkstra_path_length(walk, origin_node, weight="weight")
    reachable = []
    unreachable = []
    for poi in pois:
        poi_id = poi.get("id") or poi.get("poi_id")
        if not set(poi.get("activities") or []) & activities:
            continue
        node, access = nearest(_point(poi, coordinates))
        if node not in distances:
            unreachable.append(poi_id)
            continue
        reachable.append({"poi_id": poi_id, "network_distance_m": round(
            origin_access + distances[node] + access, 2),
            "start_access_m": round(origin_access, 2),
            "destination_access_m": round(access, 2),
            "access_over_30m": origin_access > 30 or access > 30})
    reachable.sort(key=lambda row: (row["network_distance_m"], row["poi_id"]))
    return {"start_poi_id": start_poi_id, "activities": sorted(activities),
            "reachable": reachable, "unreachable": sorted(unreachable),
            "method": "independent_node_graph_walk_distance_plus_straight_access",
            "status": "authoring_reference_not_field_truth"}


def build_candidate_reference(graph: nx.MultiDiGraph, pois: list[dict[str, Any]], *,
                              start_poi_id: str, spec: dict[str, Any],
                              closed_edges: set[str] | None = None) -> dict[str, Any]:
    """Recompute full eligible distances from frozen sources during scoring."""
    ranked = rank_nearby_activities(
        graph, pois, start_poi_id=start_poi_id,
        activities=set(spec["activities"]), coordinates=spec["coordinates"],
        closed_edges=closed_edges,
    )
    rows = ranked["reachable"]
    if not rows:
        raise ValueError("candidate reference has no reachable eligible POI")
    return {
        "distances_m": {row["poi_id"]: row["network_distance_m"] for row in rows},
        "max_display": spec["max_display"],
        "rank_tolerance_m": spec["rank_tolerance_m"],
        "uncertain_access_ids": [row["poi_id"] for row in rows
                                 if row["access_over_30m"]],
    }
