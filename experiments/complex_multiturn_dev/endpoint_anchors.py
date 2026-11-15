"""Anchors for online endpoints, bounded by the declared 60 m snap budget."""
from __future__ import annotations

import math

from experiments.controlled_anchor_snapshot import build_anchor_candidates
from experiments.complex_multiturn_dev.semantic_checks import _provider_record_matches


def permitted_walk_nodes(graph, evidence, boundary_wkt):
    """Nodes that both admit walking and lie inside the frozen campus boundary."""
    from shapely import wkt
    from shapely.geometry import Point
    boundary = wkt.loads(boundary_wkt)
    nodes = {node for u, v, _ in evidence["walk"]["permitted"] for node in (u, v)}
    return {node for node in nodes if boundary.covers(
        Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])))}


def _point_distance_m(lng1, lat1, lng2, lat2):
    radius = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    return radius * math.hypot(math.radians(lng1 - lng2) * math.cos((p1 + p2) / 2), p1 - p2)


def provider_endpoint_anchor(graph, permitted_nodes, candidate, source_rows, *, to_wgs84,
                             product_node=None, tolerance_m=60.0):
    """Anchor an online endpoint from provider coordinates, keeping the declared snap budget."""
    raw = next((row for row in source_rows if _provider_record_matches(candidate, row)), None)
    if not raw:
        return None, None
    lng, lat = (float(value) for value in raw["location"].split(","))
    poi = {"id": candidate["poi_id"], "name": candidate["name"],
           "coordinates": {"lng": lng, "lat": lat}}
    result = build_anchor_candidates(graph, [poi], permitted_nodes, to_wgs84=to_wgs84,
                                     max_snap_m=tolerance_m)
    anchor = result["anchors"][0] if result["anchors"] else None
    if anchor is None:
        return None, None
    if product_node is None or str(product_node) == str(anchor["node_id"]):
        return anchor, "provider_center_nearest_node"
    if product_node not in graph:
        return anchor, "provider_center_nearest_node"
    wgs_lng, wgs_lat = to_wgs84(lng, lat)
    distance = _point_distance_m(float(graph.nodes[product_node]["x"]),
                                 float(graph.nodes[product_node]["y"]), wgs_lng, wgs_lat)
    if distance > tolerance_m:
        return anchor, "provider_center_snap_mismatch"
    snapped = dict(anchor, node_id=product_node, snap_distance_m=round(distance, 3))
    return snapped, f"provider_center_within_{tolerance_m:g}m"
