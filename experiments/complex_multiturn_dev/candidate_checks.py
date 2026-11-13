"""Independent graph-distance bounds for already delivered local candidates."""
from __future__ import annotations

import math
from experiments.controlled_route_gold import derive_graph_route
from experiments.controlled_anchor_snapshot import build_anchor_candidates
from experiments.independent_route_oracle import _highways
from experiments.complex_multiturn_dev.semantic_checks import _provider_record_matches


def provider_anchor(graph, permitted_nodes, candidate, source_rows, *, to_wgs84):
    """Use provider-recorded coordinates only when identity and output agree."""
    raw = next((row for row in source_rows if _provider_record_matches(candidate, row)), None)
    if not raw:
        return None
    lng, lat = (float(value) for value in raw["location"].split(","))
    poi = {"id": candidate["poi_id"], "name": candidate["name"],
           "coordinates": {"lng": lng, "lat": lat}}
    result = build_anchor_candidates(graph, [poi], permitted_nodes, to_wgs84=to_wgs84, max_snap_m=60)
    return result["anchors"][0] if result["anchors"] else None


def time_distance_limit(observed):
    """Use a radius for the origin path only when it was explicitly requested."""
    limit = observed.get("route_max_distance_m")
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        limit = None
    radius = observed.get("radius_m")
    same_center = (observed.get("search_anchor") or observed.get("origin")) == observed.get("origin")
    explicit = observed.get("search_radius_explicit") is True or type(radius) is int and radius != 1000
    if same_center and explicit and type(radius) is int and radius > 0:
        limit = min(limit, radius) if limit is not None else radius
    return limit


def inspect_candidate_scope(graph, anchors, evidence, center, candidate, mode, radius, *, avoid_steps=False):
    if center not in anchors or candidate not in anchors or mode not in evidence or type(radius) is not int:
        return {"verdict": "unknown", "reason": "candidate_center_mode_or_radius_not_anchored"}
    first, last = anchors[center], anchors[candidate]
    access = float(first["snap_distance_m"]) + float(last["snap_distance_m"])
    remaining = radius - access
    if remaining <= 0:
        return {"verdict": "fail", "reason": "straight_line_access_exceeds_radius", "access_lower_bound_m": access}
    answer = derive_graph_route(graph, start_node=first["node_id"], end_node=last["node_id"],
        via_nodes=[], mode=mode, permitted_edge_ids=evidence[mode]["permitted"],
        unknown_edge_ids=evidence[mode]["unknown"], closed_edge_ids=set(),
        avoid_steps=avoid_steps, max_distance_m=remaining)
    verdict = {"feasible": "pass", "infeasible": "fail", "unknown": "unknown"}[answer["verdict"]]
    return {"verdict": verdict, "reason": answer["reason"], "graph_length_m": answer["length_m"],
            "access_lower_bound_m": round(access, 3),
            "boundary": "frozen graph and straight-line connector estimates; physical access unverified"}


def inspect_candidate_time(graph, anchors, evidence, origin, candidate, mode, max_duration_s, *,
                           radius=None, avoid_steps=False):
    """Find an independently derived witness; do not reject unexamined alternative paths."""
    speeds = {"walk": 4.5, "bike": 14.0, "drive": 25.0}
    if (origin not in anchors or candidate not in anchors or mode not in evidence or mode not in speeds
            or type(max_duration_s) not in (int, float) or not math.isfinite(max_duration_s) or max_duration_s <= 0
            or radius is not None and (type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0)):
        return {"verdict": "unknown", "reason": "candidate_origin_mode_or_time_unverified"}
    first, last = anchors[origin], anchors[candidate]
    access = float(first["snap_distance_m"]) + float(last["snap_distance_m"])
    remaining = None if radius is None else float(radius) - access
    if remaining is not None and remaining <= 0:
        return {"verdict": "fail", "reason": "access_exceeds_distance_limit"}
    route = derive_graph_route(graph, start_node=first["node_id"], end_node=last["node_id"],
        via_nodes=[], mode=mode, permitted_edge_ids=evidence[mode]["permitted"],
        unknown_edge_ids=evidence[mode]["unknown"], closed_edge_ids=set(),
        avoid_steps=avoid_steps, max_distance_m=remaining)
    if route["verdict"] != "feasible":
        return {"verdict": "fail" if route["verdict"] == "infeasible" else "unknown", "reason": route["reason"]}
    meters_per_minute = speeds[mode] * 1000 / 60
    minutes = access / meters_per_minute
    for u, v, key in route["route_edge_ids"]:
        edge = graph[u][v][key]
        factor, level = 1.0, edge.get("slope_level")
        if mode == "walk":
            if "steps" in (_highways(edge.get("highway")) or set()):
                factor *= 0.4
            if level == 5:
                factor *= 0.5
            elif level == 4:
                factor *= 0.7
        elif mode == "bike":
            if level == 5:
                factor *= 0.4
            elif level == 4:
                factor *= 0.6
        minutes += float(edge["length"]) / (meters_per_minute * factor)
    seconds = minutes * 60
    return {"verdict": "pass" if seconds <= max_duration_s + 1e-6 else "unknown",
            "reason": "independent_shortest_path_witness" if seconds <= max_duration_s + 1e-6
                      else "this_path_is_too_slow_alternatives_not_ruled_out",
            "estimated_seconds": round(seconds, 3), "route_edge_ids": route["route_edge_ids"],
            "boundary": "frozen graph speed model and straight-line connectors; physical access unverified"}
