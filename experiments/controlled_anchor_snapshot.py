"""Conservative POI-to-graph candidate anchors for controlled experiments.

Nearest-node geometry does not establish an entrance or walkable last metres.
The coordinate converter is an explicit, separately frozen input.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Callable


_EARTH_RADIUS_M = 6371008.8


def _node_sort_key(node: Any) -> tuple[str, str]:
    return type(node).__name__, str(node)


def _xy(attrs: dict[str, Any]) -> tuple[float, float] | None:
    try:
        lon, lat = float(attrs["x"]), float(attrs["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(lon) or not math.isfinite(lat) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return math.radians(lon), math.radians(lat)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return _EARTH_RADIUS_M * math.hypot(
        (a[0] - b[0]) * math.cos((a[1] + b[1]) / 2), a[1] - b[1],
    )


def build_anchor_candidates(
    graph: Any, pois: list[dict[str, Any]], permitted_nodes: set[Any], *,
    to_wgs84: Callable[[float, float], tuple[float, float]], max_snap_m: float = 20,
) -> dict[str, Any]:
    """Return deterministic candidates and every exclusion reason."""
    if (isinstance(max_snap_m, bool) or not isinstance(max_snap_m, (int, float))
            or not math.isfinite(max_snap_m) or max_snap_m <= 0):
        raise ValueError("max_snap_m must be positive and finite")
    if not callable(to_wgs84):
        raise ValueError("to_wgs84 converter required")
    if not isinstance(pois, list) or any(not isinstance(poi, dict) for poi in pois):
        raise ValueError("pois must be a list")
    if not permitted_nodes:
        raise ValueError("permitted graph nodes required")
    nodes = []
    for node in sorted(permitted_nodes, key=_node_sort_key):
        if node not in graph:
            raise ValueError("permitted node missing from graph")
        point = _xy(graph.nodes[node])
        if point is not None:
            nodes.append((node, point))
    if not nodes:
        raise ValueError("permitted graph has no valid node coordinates")

    names = Counter(
        " ".join(poi.get("name", "").casefold().split())
        for poi in pois if isinstance(poi, dict) and isinstance(poi.get("name"), str)
    )
    ids = Counter(poi.get("id") for poi in pois if isinstance(poi.get("id"), str))
    anchors = []
    excluded = []
    for poi in sorted(pois, key=lambda item: str(item.get("id", ""))):
        poi_id = poi.get("id")
        name = poi.get("name")
        reason = None
        if not isinstance(poi_id, str) or not poi_id or not isinstance(name, str) or not name.strip():
            reason = "invalid_identity"
        elif ids[poi_id] != 1:
            reason = "duplicate_poi_id"
        elif names[" ".join(name.casefold().split())] != 1:
            reason = "ambiguous_name"
        else:
            coords = poi.get("coordinates")
            try:
                gcj_lon, gcj_lat = float(coords["lng"]), float(coords["lat"])
                if not all(math.isfinite(v) for v in (gcj_lon, gcj_lat)):
                    raise ValueError
                wgs_lon, wgs_lat = to_wgs84(gcj_lon, gcj_lat)
                point = _xy({"x": wgs_lon, "y": wgs_lat})
                if point is None:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                reason = "invalid_coordinates"
        if reason is not None:
            excluded.append({"poi_id": poi_id, "reason": reason})
            continue
        nearest, distance_m = min(
            ((node, _distance(point, node_point)) for node, node_point in nodes),
            key=lambda entry: (round(entry[1], 6), _node_sort_key(entry[0])),
        )
        if distance_m > max_snap_m:
            excluded.append({"poi_id": poi_id, "reason": "too_far_from_permitted_graph",
                             "nearest_distance_m": round(distance_m, 3)})
            continue
        anchors.append({
            "poi_id": poi_id, "name": name, "node_id": nearest,
            "snap_distance_m": round(distance_m, 3),
            "access_status": "last_meter_unverified",
        })
    return {
        "format": "controlled-anchor-candidates-v1",
        "scope": "graph_node_proximity_only",
        "distance_method": "local_equirectangular_m",
        "max_snap_m": max_snap_m,
        "anchors": anchors, "excluded": excluded,
    }
