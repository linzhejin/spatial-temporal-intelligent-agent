"""Check a direct route against exact graph edges and a frozen event snapshot.

This is a result audit, not another planner or a source of field truth. It does
not accept the planner's success flag as evidence. A caller must supply the same
active road-condition snapshot used for planning; ``None`` is never interpreted
as an empty set of events. Composite routes need per-leg verification instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from agents.route_state import current_data_version, current_road_condition_version
from spatial.coord_transform import wgs84_to_gcj02
from spatial.osm_access import edge_access, node_access, values
from spatial.road_conditions import apply_conditions_to_graph
from spatial.route_constraints import excluded_edge_ids, positive_budget
from spatial.routing import MODE_SPEEDS_KMH

_VALID_MODES = {"walk", "bike", "drive"}
_UNKNOWN_ISSUES = {
    "endpoint_unknown", "missing_event_snapshot", "event_snapshot_unusable",
    "missing_duration_for_budget", "budget_measurement_unknown",
    "accessibility_unverified", "duration_context_unavailable",
    "slope_evidence_unknown",
}
_SCOPE = "internal_graph_and_event_snapshot"
_CHECK_ISSUES = {
    "route_structure": {
        "invalid_result", "missing_route_state", "unsupported_route_kind",
        "invalid_route_state", "invalid_constraints", "missing_segments",
        "missing_legs", "leg_count_mismatch",
    },
    "road_conditions": {
        "data_version_mismatch", "event_version_mismatch",
        "missing_event_snapshot", "event_snapshot_unusable",
    },
    "endpoints": {"endpoint_unknown", "endpoint_mismatch", "empty_edge_sequence"},
    "exact_edges": {
        "missing_edge_sequence", "invalid_edge_id", "discontinuous_edges",
        "missing_edge", "display_edge_mismatch",
    },
    "mode_and_constraints": {
        "mode_mismatch", "mode_forbidden", "constraint_violation",
        "excluded_edge_used", "closed_edge", "accessibility_unverified",
        "slope_evidence_unknown",
    },
    "length_and_geometry": {
        "invalid_edge_length", "invalid_reported_length", "length_mismatch",
        "display_geometry_mismatch",
    },
    "duration_and_budget": {
        "budget_measurement_unknown", "duration_context_unavailable",
        "duration_underreported", "duration_overreported",
        "invalid_reported_duration", "duration_sum_mismatch",
        "invalid_segment_duration", "missing_reported_duration",
        "missing_duration_for_budget", "distance_budget_exceeded",
        "duration_budget_exceeded",
    },
}


def _base_issue(issue):
    if isinstance(issue, str) and issue.startswith("leg_") and ":" in issue:
        return issue.rsplit(":", 1)[-1]
    return issue


def _structured_report(report, *, task_revision=None, checked_groups=None):
    """Add a stable, machine-readable summary without removing legacy fields."""
    issues = list(report.get("issues") or [])
    base_issues = {issue: _base_issue(issue) for issue in issues}
    unknowns = [issue for issue in issues if base_issues[issue] in _UNKNOWN_ISSUES]
    violations = [issue for issue in issues if issue not in unknowns]
    allowed_groups = set(checked_groups or _CHECK_ISSUES)
    checks = {}
    assigned = set()
    for name, codes in _CHECK_ISSUES.items():
        if name not in allowed_groups:
            continue
        matching = [issue for issue in issues if base_issues[issue] in codes]
        assigned.update(matching)
        if not matching:
            status = "pass"
        elif all(issue in unknowns for issue in matching):
            status = "unknown"
        else:
            status = "fail"
        checks[name] = {"status": status, "issues": matching}
    unmatched = [issue for issue in issues if issue not in assigned]
    if unmatched and "other" in allowed_groups:
        status = "unknown" if all(issue in unknowns for issue in unmatched) else "fail"
        checks["other"] = {"status": status, "issues": unmatched}
    report.update({
        "checks": checks,
        "violations": violations,
        "unknowns": unknowns,
        "event_version": report.get("road_condition_version",
                                     report.get("event_version")),
        "task_revision": (task_revision if isinstance(task_revision, int)
                          and not isinstance(task_revision, bool) and task_revision >= 0
                          else None),
    })
    return report


def verify_display_geometry(graph, edge_ids, displayed):
    """Check GCJ-02 display points against the exact selected directed edges.

    This reconstructs coordinates from graph nodes and each selected edge's
    geometry. It does not call the route renderer, so a rendering mistake is
    visible as a mismatch rather than being repeated by the check.
    """
    if not isinstance(edge_ids, list) or not isinstance(displayed, list):
        return {"status": "fail", "issues": ["missing_display_geometry"]}
    expected_wgs = []
    previous_end = None
    for raw in edge_ids:
        edge = _edge_tuple(raw)
        if edge is None or not graph.has_edge(*edge):
            return {"status": "fail", "issues": ["missing_edge"]}
        u, v, key = edge
        if previous_end is not None and previous_end != u:
            return {"status": "fail", "issues": ["discontinuous_edges"]}
        try:
            start = (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]))
            end = (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"]))
            raw_geometry = graph[u][v][key].get("geometry")
            if raw_geometry:
                from shapely.wkt import loads as load_wkt
                shape = load_wkt(raw_geometry) if isinstance(raw_geometry, str) else raw_geometry
                points = [(float(point[0]), float(point[1])) for point in shape.coords]
                if len(points) < 2:
                    raise ValueError("edge geometry has fewer than two points")
                if _point_gap_m(points[0], start) > _point_gap_m(points[-1], start):
                    points.reverse()
                if _point_gap_m(points[0], start) > 3 or _point_gap_m(points[-1], end) > 3:
                    return {"status": "fail", "issues": ["source_geometry_endpoint_mismatch"]}
            else:
                points = [start, end]
        except Exception:
            # Shapely raises GEOSException for malformed WKT; malformed source
            # geometry is insufficient evidence for a publishable display line.
            return {"status": "unknown", "issues": ["source_geometry_unreadable"]}
        if not expected_wgs:
            expected_wgs.append(start)
        expected_wgs.extend(points[1:])
        previous_end = v

    expected = []
    for longitude, latitude in expected_wgs:
        gcj_lon, gcj_lat = wgs84_to_gcj02(round(longitude, 6), round(latitude, 6))
        expected.append((round(gcj_lon, 6), round(gcj_lat, 6)))
    if len(expected) != len(displayed):
        return {"status": "fail", "issues": ["display_geometry_mismatch"]}
    for actual, (expected_lon, expected_lat) in zip(displayed, expected):
        try:
            if isinstance(actual, Mapping):
                lon, lat = float(actual["lng"]), float(actual["lat"])
            else:
                lon, lat = float(actual[0]), float(actual[1])
        except (IndexError, KeyError, TypeError, ValueError):
            return {"status": "fail", "issues": ["display_geometry_mismatch"]}
        if (not math.isfinite(lon) or not math.isfinite(lat)
                or abs(lon - expected_lon) > 0.000002
                or abs(lat - expected_lat) > 0.000002):
            return {"status": "fail", "issues": ["display_geometry_mismatch"]}
    return {"status": "pass", "issues": [], "point_count": len(expected)}


def _point_gap_m(left, right):
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((left[1] + right[1]) / 2))
    return math.hypot((left[0] - right[0]) * lon_scale, (left[1] - right[1]) * lat_scale)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _edge_tuple(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    # Preserve exact typed IDs and keys; guessing an int conversion could
    # silently bind a response to a different MultiDiGraph edge.
    if any(isinstance(part, (dict, list, tuple, bool)) for part in raw):
        return None
    return tuple(raw)


def _mode_allowed(graph, edge, mode):
    u, v, key = edge
    attrs = graph[u][v][key]
    if not edge_access(attrs, mode)[0]:
        return False
    if not node_access(graph.nodes[u], mode)[0] or not node_access(graph.nodes[v], mode)[0]:
        return False
    allowed = attrs.get("allowed_modes")
    if allowed is not None and mode not in values(allowed):
        return False
    if {"all", mode} & set(values(attrs.get("blocked_modes"))):
        return False
    return True


def _minimum_duration_min(graph, edges, mode, road_penalties):
    """Rebuild a safe lower bound from selected edges and confirmed slowdowns.

    Weather and optional routing penalties can only increase the estimate, so
    omitting them here cannot reject a valid route for being too slow.
    """
    base_m_per_min = MODE_SPEEDS_KMH[mode] * 1000 / 60
    total = 0.0
    for u, v, key in edges:
        attrs = graph[u][v][key]
        length = _number(attrs.get("length"))
        if length is None:
            return None
        factor = 1.0
        slope = attrs.get("slope_level")
        if mode == "walk":
            if "steps" in values(attrs.get("highway")):
                factor *= 0.4
            if slope == 5:
                factor *= 0.5
            elif slope == 4:
                factor *= 0.7
        elif mode == "bike":
            if slope == 5:
                factor *= 0.4
            elif slope == 4:
                factor *= 0.6
        penalty = road_penalties.get((u, v, key), 1.0)
        if penalty > 1:
            factor /= penalty
        total += length / (base_m_per_min * max(factor, 0.05))
    return total


def _recomputed_duration_min(graph, edges, mode, road_conditions, weather_info):
    """Independently rebuild the public ETA from its frozen routing inputs.

    This deliberately does not call ``estimate_duration_min``. It rebuilds
    physical slope/step effects and confirmed road/weather slowdowns, then
    sums duration on the exact directed edges returned by the planner.
    """
    try:
        from spatial.routing import filter_graph_for_mode
        from spatial.weather import weather_slope_penalty

        mode_graph, _, _ = filter_graph_for_mode(graph, mode)
        condition_graph, road_penalties, closed_edges, _ = apply_conditions_to_graph(
            mode_graph, road_conditions, mode, strict=True,
        )
        slope_penalties = weather_slope_penalty(weather_info) if weather_info else {}
    except Exception:
        return None

    base_m_per_min = MODE_SPEEDS_KMH[mode] * 1000 / 60
    total = 0.0
    for edge in edges:
        u, v, key = edge
        if not graph.has_edge(u, v, key) or edge in closed_edges:
            return None
        attrs = graph[u][v][key]
        length = _number(attrs.get("length"))
        if length is None:
            return None
        factor = 1.0
        slope = attrs.get("slope_level")
        if mode == "walk":
            if "steps" in values(attrs.get("highway")):
                factor *= 0.4
            if slope == 5:
                factor *= 0.5
            elif slope == 4:
                factor *= 0.7
        elif mode == "bike":
            if slope == 5:
                factor *= 0.4
            elif slope == 4:
                factor *= 0.6
        multiplier = (
            road_penalties.get(edge, 1.0)
            * slope_penalties.get(slope, 1.0)
        )
        if multiplier > 1:
            factor /= multiplier
        total += length / (base_m_per_min * max(factor, 0.05))
    return round(total, 1)


def _check_segment_duration_sum(reported, segments, flag):
    if reported is None:
        return
    reported_value = _number(reported)
    segment_values = [
        _number(segment.get("duration_min")) if isinstance(segment, Mapping) else None
        for segment in segments
    ]
    if reported_value is None or not segments or any(value is None for value in segment_values):
        flag("invalid_segment_duration")
    elif not math.isclose(reported_value, sum(segment_values),
                          abs_tol=0.05 + 0.1 * len(segments)):
        flag("duration_sum_mismatch")


def _check_route_budgets(constraints, measured_length, reported_duration, flag):
    if not isinstance(constraints, Mapping):
        flag("invalid_constraints")
        return
    try:
        max_distance = positive_budget(constraints, "max_distance_m")
        max_duration = positive_budget(constraints, "max_duration_s")
    except ValueError:
        flag("invalid_constraints")
        return
    if max_distance is not None:
        if measured_length is None:
            flag("budget_measurement_unknown")
        elif measured_length > max_distance + 0.15:
            flag("distance_budget_exceeded")
    if max_duration is not None:
        duration = _number(reported_duration)
        if duration is None:
            flag("missing_duration_for_budget")
        elif duration * 60 > max_duration + 3:
            # Product estimates are rounded to one decimal minute.
            flag("duration_budget_exceeded")


def _edge_verified_accessible(attrs):
    return (attrs.get("step_free_accessible") is True
            and attrs.get("accessibility_verification_status") in {
                "field_verified", "institution_verified",
            })


def _constraint_issue(attrs, constraints):
    highway = set(values(attrs.get("highway")))
    if constraints.get("avoid_steps") and "steps" in highway:
        return "constraint_violation"
    if constraints.get("slope") == "avoid":
        slope = _number(attrs.get("slope_level"))
        if slope is None:
            return "slope_evidence_unknown"
        if slope >= 5:
            return "constraint_violation"
    return None


def verify_route_result(
    graph, result, *, road_conditions,
    expected_start_node=None, expected_end_node=None,
    _frozen_data_version=None, enforce_budgets=True,
    duration_context_available=False, weather_info=None, task_revision=None,
):
    """Return ``valid``, issue codes and physical length for one direct route.

    Expected API fields are ``recommended_edge_ids``, ``recommended_length_m``,
    ``mode`` and ``route_state``. Exact edge IDs use directed ``(u, v, key)``.
    Data and event versions are re-derived from the supplied graph and explicit
    event snapshot. Both expected endpoint nodes must come from an independent
    endpoint resolver; missing endpoints yield ``endpoint_status=unknown`` and
    cannot pass. The graph/conditions must be frozen together by the caller.
    """
    issues = []

    def flag(code):
        if code not in issues:
            issues.append(code)

    if not isinstance(result, Mapping):
        return _structured_report({
            "valid": False, "status": "fail", "issues": ["invalid_result"],
            "measured_length_m": None, "edge_count": 0,
            "endpoint_status": "unknown", "scope": _SCOPE,
            "data_version_scope": "graph_and_local_file_content", "field_truth_verified": False,
        }, task_revision=task_revision, checked_groups={"route_structure"})

    state = result.get("route_state")
    if not isinstance(state, Mapping):
        flag("missing_route_state")
        state = {}
    if state.get("route_kind") != "direct":
        flag("unsupported_route_kind")

    mode = result.get("mode")
    mode_valid = isinstance(mode, str) and mode in _VALID_MODES
    if not mode_valid or state.get("travel_mode") != mode:
        flag("mode_mismatch")
    constraints = state.get("hard_constraints")
    if not isinstance(constraints, Mapping):
        flag("invalid_constraints")
        constraints = {}
    try:
        excluded_edges = excluded_edge_ids(constraints)
    except ValueError:
        flag("invalid_constraints")
        excluded_edges = set()
    require_accessibility = constraints.get("require_verified_accessibility", False)
    if not isinstance(require_accessibility, bool):
        flag("invalid_constraints")
        require_accessibility = False

    actual_data_version = (_frozen_data_version if _frozen_data_version is not None
                           else current_data_version(graph))
    if not state.get("data_version") or state.get("data_version") != actual_data_version:
        flag("data_version_mismatch")
    if not isinstance(road_conditions, list):
        flag("missing_event_snapshot")
        road_conditions = None
    if road_conditions is not None:
        event_version = current_road_condition_version(road_conditions)
        if not state.get("road_condition_version") or state.get("road_condition_version") != event_version:
            flag("event_version_mismatch")
        published_event_version = result.get("road_condition_version")
        if published_event_version is not None and published_event_version != event_version:
            flag("event_version_mismatch")
    else:
        event_version = None

    endpoints_known = (
        expected_start_node is not None and expected_end_node is not None
        and expected_start_node in graph and expected_end_node in graph
    )
    if not endpoints_known:
        flag("endpoint_unknown")
    raw_edges = result.get("recommended_edge_ids")
    edge_sequence_present = isinstance(raw_edges, list)
    if not edge_sequence_present:
        flag("missing_edge_sequence")
        raw_edges = []
    edges = []
    for raw in raw_edges:
        edge = _edge_tuple(raw)
        if edge is None:
            flag("invalid_edge_id")
            continue
        if edges and edges[-1][1] != edge[0]:
            flag("discontinuous_edges")
        edges.append(edge)

    endpoint_status = "unknown"
    if endpoints_known and edge_sequence_present and len(edges) == len(raw_edges):
        if edges:
            if edges[0][0] == expected_start_node and edges[-1][1] == expected_end_node:
                endpoint_status = "matched"
            else:
                endpoint_status = "mismatch"
                flag("endpoint_mismatch")
        elif expected_start_node == expected_end_node:
            endpoint_status = "matched"
        else:
            endpoint_status = "mismatch"
            flag("empty_edge_sequence")
    elif endpoints_known and edge_sequence_present and not edges:
        flag("empty_edge_sequence")

    measured_length = 0.0
    all_lengths_known = (edge_sequence_present and len(edges) == len(raw_edges)
                         and (bool(edges) or endpoint_status == "matched"))
    for edge in edges:
        u, v, key = edge
        if not graph.has_edge(u, v, key):
            flag("missing_edge")
            all_lengths_known = False
            continue
        attrs = graph[u][v][key]
        if edge in excluded_edges:
            flag("excluded_edge_used")
        if require_accessibility and not _edge_verified_accessible(attrs):
            flag("accessibility_unverified")
        length = _number(attrs.get("length"))
        if length is None:
            flag("invalid_edge_length")
            all_lengths_known = False
        else:
            measured_length += length
        if mode_valid:
            if not _mode_allowed(graph, edge, mode):
                flag("mode_forbidden")
            constraint_issue = _constraint_issue(attrs, constraints)
            if constraint_issue:
                flag(constraint_issue)

    if all_lengths_known:
        declared_length = _number(result.get("recommended_length_m"))
        if declared_length is None:
            flag("invalid_reported_length")
        elif not math.isclose(measured_length, declared_length, abs_tol=0.15):
            flag("length_mismatch")

    road_penalties = None
    if road_conditions is not None and mode_valid:
        try:
            _, road_penalties, closed_edges, _ = apply_conditions_to_graph(
                graph, road_conditions, mode, strict=True,
            )
        except Exception:
            # An unbound or invalid active event cannot be treated as clear.
            flag("event_snapshot_unusable")
        else:
            if any(edge in closed_edges for edge in edges):
                flag("closed_edge")

    minimum_duration = None
    recomputed_duration = None
    if all_lengths_known and mode_valid and road_penalties is not None and not any(
        issue in issues for issue in ("missing_edge", "invalid_edge_length")
    ):
        minimum_duration = _minimum_duration_min(graph, edges, mode, road_penalties)
        if duration_context_available:
            recomputed_duration = _recomputed_duration_min(
                graph, edges, mode, road_conditions, weather_info,
            )
            if recomputed_duration is None:
                flag("duration_context_unavailable")
    if "duration_min" in result:
        reported_duration = _number(result.get("duration_min"))
        if reported_duration is None:
            flag("invalid_reported_duration")
        elif recomputed_duration is not None:
            if reported_duration + 0.11 < recomputed_duration:
                flag("duration_underreported")
            elif reported_duration > recomputed_duration + 0.11:
                flag("duration_overreported")
        elif duration_context_available and minimum_duration is None:
            flag("duration_context_unavailable")
        elif minimum_duration is not None and reported_duration + 0.11 < minimum_duration:
            flag("duration_underreported")
    elif duration_context_available and all_lengths_known:
        flag("missing_reported_duration")
    if enforce_budgets:
        _check_route_budgets(
            constraints, measured_length if all_lengths_known else None,
            result.get("duration_min"), flag,
        )
    if require_accessibility:
        access = result.get("endpoint_access")
        if (not isinstance(access, Mapping)
                or not isinstance(access.get("start"), Mapping)
                or not isinstance(access.get("end"), Mapping)
                or not access["start"].get("access_link_verified")
                or not access["end"].get("access_link_verified")):
            flag("accessibility_unverified")

    status = (
        "pass" if not issues else
        "unknown" if set(issues) <= _UNKNOWN_ISSUES else
        "fail"
    )
    return _structured_report({
        "valid": status == "pass",
        "status": status,
        "scope": _SCOPE,
        "data_version_scope": "graph_and_local_file_content",
        "field_truth_verified": False,
        "issues": issues,
        "measured_length_m": measured_length if all_lengths_known else None,
        "minimum_duration_min": round(minimum_duration, 1) if minimum_duration is not None else None,
        "expected_duration_min": recomputed_duration,
        "edge_count": len(edges),
        "endpoint_status": endpoint_status,
        "data_version": actual_data_version,
        "road_condition_version": event_version,
        "budgets_enforced": enforce_budgets,
    }, task_revision=task_revision)


def verify_ordered_route_result(graph, result, *, road_conditions, expected_nodes,
                                enforce_budgets=True,
                                duration_context_available=False, weather_info=None,
                                task_revision=None):
    """Audit a via route leg by leg against independently resolved stop nodes.

    The flattened edge and geometry arrays must be the concatenation of the
    verified legs. This catches a result that visits the wrong stop or draws a
    different route from the one used to compute its distance.
    """
    issues = []
    leg_reports = []

    def flag(code):
        if code not in issues:
            issues.append(code)

    if not isinstance(result, Mapping):
        return _structured_report({
            "valid": False, "status": "fail", "issues": ["invalid_result"],
            "leg_reports": [], "measured_length_m": None, "scope": _SCOPE,
            "data_version_scope": "graph_and_local_file_content", "field_truth_verified": False,
        }, task_revision=task_revision, checked_groups={"route_structure"})
    state = result.get("route_state")
    if not isinstance(state, Mapping) or state.get("route_kind") != "via":
        flag("invalid_route_state")
        state = state if isinstance(state, Mapping) else {}
    if not isinstance(expected_nodes, (list, tuple)) or len(expected_nodes) < 3:
        flag("endpoint_unknown")
        expected_nodes = []
    legs = result.get("legs")
    if not isinstance(legs, list):
        flag("missing_legs")
        legs = []
    if len(legs) != max(0, len(expected_nodes) - 1):
        flag("leg_count_mismatch")

    frozen_data_version = current_data_version(graph)

    joined_edges = []
    joined_geometry = []
    measured_length = 0.0
    lengths_known = bool(legs) and len(legs) == len(expected_nodes) - 1
    for index, leg in enumerate(legs):
        if not isinstance(leg, Mapping):
            flag(f"leg_{index + 1}:invalid_result")
            lengths_known = False
            continue
        raw_edges = leg.get("recommended_edge_ids")
        if isinstance(raw_edges, list):
            joined_edges.extend(raw_edges)
        else:
            lengths_known = False
        part = leg.get("recommended")
        if isinstance(part, list):
            joined_geometry.extend(part[1:] if joined_geometry and part and joined_geometry[-1] == part[0]
                                   else part)
        else:
            flag(f"leg_{index + 1}:missing_geometry")
        if index + 1 >= len(expected_nodes):
            continue
        direct_result = {
            **leg,
            "route_state": {**state, "route_kind": "direct"},
        }
        report = verify_route_result(
            graph, direct_result, road_conditions=road_conditions,
            expected_start_node=expected_nodes[index],
            expected_end_node=expected_nodes[index + 1],
            _frozen_data_version=frozen_data_version,
            enforce_budgets=enforce_budgets,
            duration_context_available=duration_context_available,
            weather_info=weather_info,
            task_revision=task_revision,
        )
        leg_reports.append(report)
        for issue in report["issues"]:
            flag(f"leg_{index + 1}:{issue}")
        if report["measured_length_m"] is None:
            lengths_known = False
        else:
            measured_length += report["measured_length_m"]

    top_edges = result.get("recommended_edge_ids")
    if not isinstance(top_edges, list) or [
        _edge_tuple(edge) for edge in top_edges
    ] != [_edge_tuple(edge) for edge in joined_edges]:
        flag("display_edge_mismatch")
    if result.get("recommended") != joined_geometry:
        flag("display_geometry_mismatch")
    if lengths_known:
        declared_length = _number(result.get("recommended_length_m"))
        if declared_length is None:
            flag("invalid_reported_length")
        elif not math.isclose(measured_length, declared_length,
                              abs_tol=max(0.15, 0.15 * len(legs))):
            flag("length_mismatch")
    _check_segment_duration_sum(result.get("duration_min"), legs, flag)
    if enforce_budgets:
        _check_route_budgets(
            state.get("hard_constraints") or {},
            measured_length if lengths_known else None, result.get("duration_min"), flag,
        )

    unknown = bool(issues) and all(
        issue == "endpoint_unknown" or (
            issue.startswith("leg_") and issue.split(":", 1)[-1] in _UNKNOWN_ISSUES
        ) for issue in issues
    )
    status = "pass" if not issues else "unknown" if unknown else "fail"
    return _structured_report({
        "valid": status == "pass", "status": status, "issues": issues,
        "leg_reports": leg_reports,
        "measured_length_m": measured_length if lengths_known else None,
        "scope": _SCOPE, "data_version_scope": "graph_and_local_file_content",
        "field_truth_verified": False,
        "road_condition_version": current_road_condition_version(road_conditions)
        if isinstance(road_conditions, list) else None,
        "budgets_enforced": enforce_budgets,
    }, task_revision=task_revision)


def verify_segmented_route_result(
    graph, result, *, road_conditions, expected_kind, segments,
    duration_context_available=False, weather_info=None, task_revision=None,
):
    """Audit a flattened tour or mixed-mode route from explicit segment facts.

    Each segment supplies its payload, independently resolved endpoint nodes,
    and travel mode. Transfers are checked by the caller before publication;
    this verifier checks only road edges and the displayed concatenation.
    """
    issues = []
    leg_reports = []

    def flag(code):
        if code not in issues:
            issues.append(code)

    if not isinstance(result, Mapping):
        return _structured_report({
            "valid": False, "status": "fail", "issues": ["invalid_result"],
            "leg_reports": [], "measured_length_m": None, "scope": _SCOPE,
            "data_version_scope": "graph_and_local_file_content", "field_truth_verified": False,
        }, task_revision=task_revision, checked_groups={"route_structure"})
    state = result.get("route_state")
    if not isinstance(state, Mapping) or state.get("route_kind") != expected_kind:
        flag("invalid_route_state")
        state = state if isinstance(state, Mapping) else {}
    if not isinstance(segments, list) or not segments:
        flag("missing_segments")
        segments = []
    elif (isinstance(segments[0], Mapping)
          and state.get("travel_mode") != segments[0].get("mode")):
        flag("mode_mismatch")

    frozen_data_version = current_data_version(graph)

    joined_edges = []
    joined_geometry = []
    measured_length = 0.0
    lengths_known = bool(segments)
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, Mapping) or not isinstance(segment.get("route"), Mapping):
            flag(f"leg_{index}:invalid_result")
            lengths_known = False
            continue
        route = segment["route"]
        raw_edges = route.get("recommended_edge_ids")
        if isinstance(raw_edges, list):
            joined_edges.extend(raw_edges)
        else:
            lengths_known = False
        part = route.get("recommended")
        if isinstance(part, list):
            joined_geometry.extend(part)
        else:
            flag(f"leg_{index}:missing_geometry")
        direct_result = {
            **route,
            "route_state": {**state, "route_kind": "direct",
                            "travel_mode": segment.get("mode")},
        }
        report = verify_route_result(
            graph, direct_result, road_conditions=road_conditions,
            expected_start_node=segment.get("start_node"),
            expected_end_node=segment.get("end_node"),
            _frozen_data_version=frozen_data_version,
            duration_context_available=duration_context_available,
            weather_info=weather_info,
            task_revision=task_revision,
        )
        leg_reports.append(report)
        for issue in report["issues"]:
            flag(f"leg_{index}:{issue}")
        if report["measured_length_m"] is None:
            lengths_known = False
        else:
            measured_length += report["measured_length_m"]

    top_edges = result.get("recommended_edge_ids")
    if not isinstance(top_edges, list) or [
        _edge_tuple(edge) for edge in top_edges
    ] != [_edge_tuple(edge) for edge in joined_edges]:
        flag("display_edge_mismatch")
    if result.get("recommended") != joined_geometry:
        flag("display_geometry_mismatch")
    if lengths_known:
        declared_length = _number(result.get("recommended_length_m"))
        if declared_length is None:
            flag("invalid_reported_length")
        elif not math.isclose(measured_length, declared_length,
                              abs_tol=max(0.15, 0.15 * len(segments))):
            flag("length_mismatch")
    reported_duration = result.get("duration_min")
    if expected_kind == "tour" and isinstance(result.get("tour"), Mapping):
        reported_duration = result["tour"].get("duration_min")
    _check_segment_duration_sum(
        reported_duration,
        [segment.get("route") if isinstance(segment, Mapping) else None
         for segment in segments],
        flag,
    )
    _check_route_budgets(
        state.get("hard_constraints") or {},
        measured_length if lengths_known else None, reported_duration, flag,
    )
    unknown = bool(issues) and all(
        issue.startswith("leg_") and issue.split(":", 1)[-1] in _UNKNOWN_ISSUES
        for issue in issues
    )
    status = "pass" if not issues else "unknown" if unknown else "fail"
    return _structured_report({
        "valid": status == "pass", "status": status, "issues": issues,
        "leg_reports": leg_reports,
        "measured_length_m": measured_length if lengths_known else None,
        "scope": _SCOPE, "data_version_scope": "graph_and_local_file_content",
        "field_truth_verified": False,
        "road_condition_version": current_road_condition_version(road_conditions)
        if isinstance(road_conditions, list) else None,
    }, task_revision=task_revision)
