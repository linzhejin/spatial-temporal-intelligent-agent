"""Conservative mode and closure evidence from frozen raw graph/event records.

No product routing, road-condition resolver, or result-verifier functions are
imported. Unknown tags remain unknown instead of being treated as open roads.
This establishes only rules within the frozen corpus, not field passability.
"""
from __future__ import annotations

import ast
import math
from collections import Counter
from typing import Any

from shapely.geometry import LineString, Point
from shapely import wkt


_MODES = {"walk", "bike", "drive"}
_MODE_ACCESS = {
    "walk": ("foot", "access"),
    "bike": ("bicycle", "vehicle", "access"),
    "drive": ("motorcar", "motor_vehicle", "vehicle", "access"),
}
_DENIED_ACCESS = {"no", "private", "permit", "destination", "customers",
                  "delivery", "agricultural", "forestry"}
_AFFIRMATIVE_ACCESS = {"yes", "designated", "permissive", "official"}
_DRIVE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "service", "living_street", "road", "busway",
}
_INACTIVE_HIGHWAYS = {"construction", "proposed", "abandoned", "razed"}


def _values(raw: Any) -> set[str] | None:
    if raw is None:
        return set()
    if isinstance(raw, str) and raw.startswith("["):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if isinstance(raw, (list, tuple, set)):
        if any(not isinstance(value, (str, int, float)) for value in raw):
            return None
        return {str(value).strip().lower() for value in raw}
    if isinstance(raw, (str, int, float)):
        return {str(raw).strip().lower()}
    return None


def _permission(tags: dict[str, Any], mode: str) -> tuple[str, str | None]:
    for key in _MODE_ACCESS[mode]:
        if key not in tags or tags[key] is None:
            continue
        values = _values(tags[key])
        if values is None or not values:
            return "unknown", key
        if values & _DENIED_ACCESS:
            return "denied", key
        if values <= _AFFIRMATIVE_ACCESS:
            return "permitted", key
        return "unknown", key
    return "unspecified", None


def _node_status(attrs: dict[str, Any], mode: str) -> str:
    permission, key = _permission(attrs, mode)
    if permission in {"denied", "unknown"}:
        return permission
    specific = permission == "permitted" and key != "access"
    entrance = _values(attrs.get("entrance"))
    barrier = _values(attrs.get("barrier"))
    if entrance is None or barrier is None:
        return "unknown"
    if not specific and entrance & {"emergency", "private"}:
        return "denied"
    if not specific and barrier & {"wall", "fence", "hedge", "retaining_wall"}:
        return "denied"
    if mode == "drive" and not specific and barrier & {
        "bollard", "block", "stile", "turnstile", "kissing_gate", "cycle_barrier",
    }:
        return "denied"
    if mode == "bike" and not specific and barrier & {
        "stile", "turnstile", "kissing_gate",
    }:
        return "denied"
    return "permitted"


def _edge_status(attrs: dict[str, Any], mode: str) -> str:
    permission, key = _permission(attrs, mode)
    if permission in {"denied", "unknown"}:
        return permission
    highways = _values(attrs.get("highway"))
    allowed = _values(attrs.get("allowed_modes"))
    blocked = _values(attrs.get("blocked_modes"))
    if (highways is None or not highways or allowed is None or blocked is None):
        return "unknown"
    if (highways & _INACTIVE_HIGHWAYS or (allowed and mode not in allowed)
            or blocked & {"all", mode}):
        return "denied"
    if mode == "walk":
        if highways & {"motorway", "motorway_link"} and key != "foot":
            return "denied"
        if str(attrs.get("indoor", "")).lower() in {"yes", "room"} and key != "foot":
            return "denied"
    elif mode == "bike":
        if highways & {"steps", "elevator", "escalator"}:
            return "denied"
        if highways & {"corridor", "motorway", "motorway_link"} and key != "bicycle":
            return "denied"
    else:
        if highways & {"steps", "elevator", "escalator", "corridor", "platform"}:
            return "denied"
        explicit_motor = permission == "permitted" and key in {
            "motorcar", "motor_vehicle", "vehicle",
        }
        if not highways & _DRIVE_HIGHWAYS and not explicit_motor:
            return "denied"
    return "permitted"


def _boundary_status(graph: Any, u: Any, v: Any, attrs: dict[str, Any], boundary: Any) -> str:
    points = []
    for node in (u, v):
        data = graph.nodes[node]
        try:
            x, y = float(data["x"]), float(data["y"])
        except (KeyError, TypeError, ValueError):
            return "unknown"
        if not math.isfinite(x) or not math.isfinite(y):
            return "unknown"
        points.append(Point(x, y))
    if any(boundary.covers(point) for point in points):
        return "permitted"
    raw_geometry = attrs.get("geometry")
    try:
        line = (wkt.loads(raw_geometry) if isinstance(raw_geometry, str)
                else raw_geometry if raw_geometry is not None
                else LineString(points))
        if not line.is_valid:
            return "unknown"
        return "permitted" if boundary.intersects(line) else "denied"
    except (TypeError, ValueError):
        return "unknown"


def mode_edge_evidence(graph: Any, mode: str, *, require_boundary: bool = False,
                       boundary_wkt: str | None = None) -> dict[str, Any]:
    """Partition exact directed graph edges into permitted, denied and unknown."""
    if mode not in _MODES:
        raise ValueError("unsupported travel mode")
    boundary = None
    if require_boundary:
        raw = boundary_wkt if boundary_wkt is not None else graph.graph.get("routing_boundary_wkt")
        if not isinstance(raw, str) or not raw:
            raise ValueError("frozen graph lacks a routing boundary")
        try:
            boundary = wkt.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid routing boundary") from exc
        if boundary.is_empty or not boundary.is_valid:
            raise ValueError("invalid routing boundary")
    buckets: dict[str, set[tuple[Any, Any, Any]]] = {
        "permitted": set(), "denied": set(), "unknown": set(),
    }
    node_status = {node: _node_status(attrs, mode) for node, attrs in graph.nodes(data=True)}
    reasons = Counter()
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        edge = (u, v, key)
        statuses = {node_status[u], node_status[v], _edge_status(attrs, mode)}
        if boundary is not None:
            statuses.add(_boundary_status(graph, u, v, attrs, boundary))
        status = "denied" if "denied" in statuses else "unknown" if "unknown" in statuses else "permitted"
        buckets[status].add(edge)
        reasons[status] += 1
    return {**buckets, "mode": mode, "counts": dict(reasons),
            "scope": "frozen_graph_tag_and_boundary_rules" if require_boundary
                     else "frozen_graph_tag_rules_only"}


def active_closed_edges(graph: Any, snapshot: dict[str, Any], mode: str) -> set[tuple[Any, Any, Any]]:
    """Resolve exact active closures from one explicit, immutable event snapshot."""
    if mode not in _MODES:
        raise ValueError("unsupported travel mode")
    epoch = snapshot.get("as_of_epoch")
    conditions = snapshot.get("conditions")
    if type(epoch) is not int or epoch < 0 or not isinstance(conditions, list):
        raise ValueError("invalid event snapshot")
    closed = set()
    seen_ids: set[str] = set()
    for item in conditions:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("invalid event record")
        if item["id"] in seen_ids:
            raise ValueError("duplicate event id")
        seen_ids.add(item["id"])
        action = item.get("action")
        modes = item.get("affected_modes")
        edge = item.get("edge")
        if (action not in {"closure", "slowdown", "notice"}
                or not isinstance(modes, list) or not modes
                or any(value not in _MODES for value in modes)
                or not isinstance(edge, dict)):
            raise ValueError("invalid event action, modes or edge")
        exact = (edge.get("u"), edge.get("v"), edge.get("key"))
        if any(part is None or isinstance(part, (dict, list, set, bool)) for part in exact):
            raise ValueError("invalid event edge identity")
        if not graph.has_edge(*exact):
            raise ValueError("event references missing graph edge")
        start = item.get("start_time")
        end = item.get("end_time")
        revoked = item.get("revoked_at")
        if (type(start) is not int or type(end) is not int or start < 0 or end < 0
                or (end != 0 and end <= start)
                or (revoked is not None and (type(revoked) is not int or revoked < 0))):
            raise ValueError("invalid event time")
        if (action == "closure" and mode in modes and start <= epoch
                and (end == 0 or epoch < end)
                and (revoked is None or epoch < revoked)):
            closed.add(exact)
    return closed
