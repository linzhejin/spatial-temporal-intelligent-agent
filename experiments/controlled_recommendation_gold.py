"""Catalog-and-graph gold for location-ranked place recommendations.

The eligible universe is explicitly the frozen POIs with reviewed activity
tags and conservative graph anchors. Distances are graph lengths, not observed
walking time or proof that the last metres to a venue are accessible.
"""
from __future__ import annotations

from typing import Any

from experiments.controlled_route_gold import derive_graph_route


_GROUPS = {"meal": frozenset({"canteen_meal", "restaurant_meal", "fast_food"})}


def rank_reachable_activity_pois(
    graph: Any, pois: list[dict[str, Any]],
    anchors: dict[str, dict[str, Any]], *,
    start_poi_id: str, activity: str, mode: str, limit: int,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Rank the entire eligible pool before taking a display-sized prefix."""
    if not isinstance(activity, str) or not activity.strip():
        raise ValueError("activity required")
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("display limit must be 1 to 5")
    if mode not in {"walk", "bike", "drive"}:
        raise ValueError("unsupported travel mode")
    if (not isinstance(pois, list) or not isinstance(anchors, dict)
            or not all(isinstance(value, set) for value in
                       (permitted_edge_ids, unknown_edge_ids, closed_edge_ids))):
        raise ValueError("explicit frozen catalog and edge evidence required")
    if unknown_edge_ids - closed_edge_ids:
        raise ValueError("unknown edge may change distance ranking")
    start = anchors.get(start_poi_id)
    if (not isinstance(start, dict) or start.get("poi_id") != start_poi_id
            or start.get("access_status") != "last_meter_unverified"
            or start.get("node_id") not in graph):
        raise ValueError("valid frozen start anchor required")

    matching = _GROUPS.get(activity, frozenset({activity}))
    seen_ids = set()
    ranked: list[tuple[float, str]] = []
    excluded = []
    for poi in pois:
        if not isinstance(poi, dict):
            raise ValueError("invalid catalog POI")
        poi_id = poi.get("poi_id") or poi.get("id")
        if not isinstance(poi_id, str) or not poi_id or poi_id in seen_ids:
            raise ValueError("catalog POI IDs must be unique")
        seen_ids.add(poi_id)
        activities = poi.get("activities")
        if not isinstance(activities, list) or any(not isinstance(value, str)
                                                  for value in activities):
            raise ValueError(f"invalid activity tags: {poi_id}")
        if poi_id == start_poi_id or not matching.intersection(activities):
            continue
        if (poi.get("activity_review_status") != "source_supported"
                or not isinstance(poi.get("activity_basis"), str)
                or not poi["activity_basis"].strip()):
            raise ValueError(f"activity evidence missing: {poi_id}")
        anchor = anchors.get(poi_id)
        if (not isinstance(anchor, dict) or anchor.get("poi_id") != poi_id
                or anchor.get("access_status") != "last_meter_unverified"):
            excluded.append({"poi_id": poi_id, "reason": "not_in_anchored_catalog"})
            continue
        result = derive_graph_route(
            graph, start_node=start["node_id"], end_node=anchor["node_id"],
            via_nodes=[], mode=mode, permitted_edge_ids=permitted_edge_ids,
            unknown_edge_ids=unknown_edge_ids, closed_edge_ids=closed_edge_ids,
        )
        if result["verdict"] == "unknown":
            raise ValueError(f"unknown graph distance: {poi_id}")
        if result["verdict"] == "infeasible":
            excluded.append({"poi_id": poi_id, "reason": "unreachable"})
            continue
        ranked.append((result["length_m"], poi_id))
    if not ranked:
        raise ValueError("no reachable reviewed activity POI in anchored catalog")
    ranked.sort(key=lambda value: (value[0], value[1]))
    return {
        "scope": "frozen_anchored_catalog_graph_only",
        "activity": activity, "start_poi_id": start_poi_id, "mode": mode,
        "candidate_poi_ids": [poi_id for _, poi_id in ranked[:limit]],
        "all_reachable_poi_ids": [poi_id for _, poi_id in ranked],
        "graph_distance_m": {poi_id: distance for distance, poi_id in ranked},
        "excluded": sorted(excluded, key=lambda item: item["poi_id"]),
        "last_meter_status": "unverified",
    }
