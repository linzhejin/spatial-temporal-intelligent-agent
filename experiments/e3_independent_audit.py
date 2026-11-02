"""Independent scoring of E3 product responses against frozen graph truth.

No product planner, verifier, event resolver, or claimed verification status is
used. The caller supplies a mode/constraint/event-filtered effective graph. Its
construction is a separate audit responsibility, recorded by the experiment.
``audit_duration_multiplier`` on an effective edge contains the independently
resolved road/weather slowdown; it never changes the pure-distance objective.
Non-distance objectives require ``audit_objective_cost`` on every effective edge.
Structured event explanations are scored; unrestricted natural-language answers
and the accuracy of geographic source data are outside this component's scope.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
import heapq
import itertools
import json
import math


_SPEED_KMH = {"walk": 4.5, "bike": 14.0, "drive": 25.0}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _edge(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, str)) for x in value)):
        return None
    return tuple(value)


def _tags(value):
    if isinstance(value, str):
        if value.startswith(("[", "(")):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {value}
        else:
            return {x.strip() for x in value.split(";")}
    return set(value) if isinstance(value, (list, tuple, set)) else set()


def _objective_cost(raw_graph, effective_graph, edge, strategy):
    attrs = (raw_graph[edge[0]][edge[1]][edge[2]] if strategy == "shortest"
             else effective_graph[edge[0]][edge[1]][edge[2]])
    value = _number(attrs.get("length" if strategy == "shortest" else "audit_objective_cost"))
    if value is None:
        raise ValueError("independent_objective_unavailable")
    return value


def _oracle(raw_graph, effective_graph, start, end, strategy):
    """Independent nonnegative shortest-cost search, without product imports."""
    costs = {edge: _objective_cost(raw_graph, effective_graph, edge, strategy)
             for edge in effective_graph.edges(keys=True)}
    if start not in effective_graph or end not in effective_graph:
        return math.inf, 0
    counter = itertools.count()
    queue = [(0.0, next(counter), start)]
    best, settled = {start: 0.0}, set()
    while queue:
        distance, _, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        if node == end:
            return distance, len(settled)
        for _, target, key in effective_graph.out_edges(node, keys=True):
            cost = costs[node, target, key]
            candidate = distance + cost
            if candidate < best.get(target, math.inf):
                best[target] = candidate
                heapq.heappush(queue, (candidate, next(counter), target))
    return math.inf, len(settled)


def _eta(raw_graph, effective_graph, edges, mode):
    speed = _SPEED_KMH[mode] * 1000 / 60
    minutes = 0.0
    for u, v, key in edges:
        attrs = raw_graph[u][v][key]
        length = _number(attrs.get("length"))
        multiplier = _number(effective_graph[u][v][key].get("audit_duration_multiplier", 1.0))
        if length is None or multiplier is None or multiplier < 1:
            raise ValueError("duration_truth_unavailable")
        slope, speed_factor = attrs.get("slope_level"), 1.0
        if mode == "walk":
            if "steps" in _tags(attrs.get("highway")):
                speed_factor *= 0.4
            speed_factor *= {4: 0.7, 5: 0.5}.get(slope, 1.0)
        elif mode == "bike":
            speed_factor *= {4: 0.6, 5: 0.4}.get(slope, 1.0)
        minutes += length / (speed * max(speed_factor / multiplier, 0.05))
    return minutes


def _error_code(data):
    error = data.get("error") if isinstance(data, Mapping) else None
    return error.get("code") if isinstance(error, Mapping) else error


def audit_route_response(raw_graph, effective_graph, expected_task, response_data,
                         response_status=200):
    """Score a direct-route API payload using independently supplied truth.

    Required task keys: start_node, end_node, mode, strategy, constraints.
    Optional checks: data_version, road_condition_version, task_revision,
    expected_error and expected_status. ``events`` is the list of currently
    applicable, independently resolved events with id, action and edge_ids.
    Omitting event truth produces an unknown check, rather than assuming none.
    The state always carries the expected environment version. Initial route
    responses also expose it at top level; replan responses expose it in the
    evidence object. Optional copies are checked whenever they are supplied.
    ``require_event_overview=False`` permits an absent overview on the initial
    /api/route contract; a supplied overview is always checked. Route notices
    and the applied-event count remain required for both API contracts.

    Return ``status`` plus per-check statuses, issues and measured metrics.
    Expected rejections and correct no-path responses have distinct outcomes.
    The caller must count failed/unknown execution outcomes in its denominator.
    """
    required = {"start_node", "end_node", "mode", "strategy", "constraints"}
    if not isinstance(expected_task, Mapping) or not required.issubset(expected_task):
        raise ValueError("expected_task is missing required frozen inputs")
    start, end = expected_task["start_node"], expected_task["end_node"]
    mode, strategy = expected_task["mode"], expected_task["strategy"]
    constraints = expected_task["constraints"]
    if mode not in _SPEED_KMH or not isinstance(constraints, Mapping):
        raise ValueError("invalid frozen mode or constraints")
    report = {"status": "pass", "checks": {}, "issues": [], "metrics": {}}

    def check(name, errors=(), unknowns=(), **measurements):
        errors, unknowns = list(dict.fromkeys(errors)), list(dict.fromkeys(unknowns))
        status = "fail" if errors else "unknown" if unknowns else "pass"
        report["checks"][name] = {"status": status, "issues": errors + unknowns, **measurements}
        report["issues"] = list(dict.fromkeys(report["issues"] + errors + unknowns))
        if status == "fail" or (status == "unknown" and report["status"] == "pass"):
            report["status"] = status

    data = response_data
    if not isinstance(data, Mapping):
        check("delivery", ["missing_response"])
        return report
    expected_error = expected_task.get("expected_error")
    if expected_error:
        valid = (_error_code(data) == expected_error and response_status >= 400
                 and response_status == expected_task.get("expected_status", response_status))
        check("expected_rejection", [] if valid else ["expected_rejection_not_returned"])
        report["metrics"]["outcome"] = "expected_rejection" if valid else "unexpected_delivery"
        return report

    # A distance oracle is also used to distinguish actual disconnection from
    # an execution error; non-distance objective availability is checked below.
    try:
        reach_cost, visited = _oracle(raw_graph, effective_graph, start, end, "shortest")
    except (ValueError, KeyError, TypeError):
        check("source_truth", unknowns=["effective_graph_truth_unavailable"])
        return report
    reachable = math.isfinite(reach_cost)
    report["metrics"].update(reachable=reachable, oracle_reachability_settled_nodes=visited)
    if response_status >= 400 or _error_code(data):
        correct_no_path = (not reachable and response_status == 404
                           and _error_code(data) == "route_not_found")
        check("delivery", [] if correct_no_path else ["unexpected_error_or_false_unreachable"])
        report["metrics"]["outcome"] = "correct_unreachable" if correct_no_path else "execution_error"
        return report
    report["metrics"]["outcome"] = "route_delivery"
    check("delivery", [] if reachable else ["route_returned_for_unreachable_task"])

    state = data.get("route_state")
    errors = []
    if not isinstance(state, Mapping):
        errors.append("missing_task_state")
        state = {}
    if data.get("mode") != mode or state.get("travel_mode") != mode:
        errors.append("mode_mismatch")
    if state.get("hard_constraints") != constraints:
        errors.append("task_constraints_mismatch")
    for container in (data, state):
        actual_strategy = container.get("strategy")
        actual_strategy = actual_strategy.get("name") if isinstance(actual_strategy, Mapping) else actual_strategy
        if actual_strategy != strategy:
            errors.append("strategy_mismatch")
    for key in ("data_version", "road_condition_version"):
        if key in expected_task and state.get(key) != expected_task[key]:
            errors.append(f"{key}_mismatch")
    if "road_condition_version" in expected_task:
        version = expected_task["road_condition_version"]
        if "road_condition_version" in data and data["road_condition_version"] != version:
            errors.append("road_condition_version_mismatch")
        if "evidence" in data:
            evidence = data["evidence"]
            if not isinstance(evidence, Mapping) or evidence.get("road_condition_version") != version:
                errors.append("evidence_road_condition_version_mismatch")
    if "task_revision" in expected_task:
        proof = state.get("reuse_certificate")
        actual_revision = proof.get("task_revision") if isinstance(proof, Mapping) else None
        if actual_revision != expected_task["task_revision"]:
            errors.append("task_revision_mismatch")
    check("task_and_versions", errors)

    raw_edges = data.get("recommended_edge_ids")
    edges, errors = [], []
    if not isinstance(raw_edges, list):
        errors.append("missing_edge_sequence")
    else:
        edges = [_edge(value) for value in raw_edges]
        if any(edge is None for edge in edges):
            errors.append("invalid_edge_id")
        else:
            if not edges and start != end:
                errors.append("empty_edge_sequence")
            if edges and (edges[0][0] != start or edges[-1][1] != end):
                errors.append("endpoint_mismatch")
            if any(a[1] != b[0] for a, b in zip(edges, edges[1:])):
                errors.append("discontinuous_edges")
            if any(not raw_graph.has_edge(*edge) for edge in edges):
                errors.append("missing_raw_edge")
            if any(not effective_graph.has_edge(*edge) for edge in edges):
                errors.append("edge_not_in_effective_graph")
    check("feasibility", errors)
    if errors:
        return report

    errors, unknowns = [], []
    excluded = {_edge(value) for value in constraints.get("excluded_edge_ids", [])}
    if None in excluded:
        unknowns.append("invalid_exclusion_truth")
    for edge in edges:
        attrs = raw_graph[edge[0]][edge[1]][edge[2]]
        if constraints.get("require_verified_accessibility") and not (
            attrs.get("step_free_accessible") is True
            and attrs.get("accessibility_verification_status") in {"field_verified", "institution_verified"}
        ):
            unknowns.append("accessibility_evidence_unknown")
        if constraints.get("avoid_steps") and "steps" in _tags(attrs.get("highway")):
            errors.append("steps_constraint_violated")
        if constraints.get("slope") == "avoid":
            slope = _number(attrs.get("slope_level"))
            if slope is None:
                unknowns.append("slope_evidence_unknown")
            elif slope >= 5:
                errors.append("slope_constraint_violated")
        if edge in excluded:
            errors.append("excluded_edge_used")
    if constraints.get("require_verified_accessibility"):
        endpoints = expected_task.get("endpoint_access_truth")
        if (not isinstance(endpoints, Mapping)
                or any(endpoints.get(endpoint) not in {"field_verified", "institution_verified"}
                       for endpoint in ("start", "end"))):
            unknowns.append("endpoint_accessibility_evidence_unknown")
    check("hard_constraints", errors, unknowns)

    lengths = [_number(raw_graph[u][v][key].get("length")) for u, v, key in edges]
    if any(value is None for value in lengths):
        check("physical_length", unknowns=["physical_length_truth_unavailable"])
        return report
    length = sum(lengths)
    report["metrics"]["physical_length_m"] = length
    reported_length = _number(data.get("recommended_length_m"))
    errors = [] if reported_length is not None and abs(reported_length - length) <= 0.051 else ["length_mismatch"]
    if "distance_m" in data:
        alias = _number(data["distance_m"])
        if alias is None or abs(alias - length) > 0.051:
            errors.append("distance_alias_mismatch")
    check("physical_length", errors)

    try:
        duration_unrounded = _eta(raw_graph, effective_graph, edges, mode)
        duration = round(duration_unrounded, 1)
        report["metrics"].update(duration_min=duration, unrounded_duration_min=duration_unrounded)
        actual = _number(data.get("duration_min"))
        check("duration", [] if actual is not None and abs(actual - duration) <= 1e-8 else ["duration_mismatch"])
    except (KeyError, ValueError, TypeError):
        duration_unrounded = None
        check("duration", unknowns=["duration_truth_unavailable"])
    errors, unknowns = [], []
    for key, actual, tolerance in (("max_distance_m", length, 0.15),
                                    ("max_duration_s", None if duration_unrounded is None else duration_unrounded * 60, 3.0)):
        if key in constraints and constraints[key] is not None:
            bound = _number(constraints[key])
            if bound is None or bound <= 0:
                unknowns.append("invalid_budget_truth")
            elif actual is None:
                unknowns.append("duration_budget_unknown")
            elif actual > bound + tolerance:
                errors.append("distance_budget_exceeded" if key == "max_distance_m" else "duration_budget_exceeded")
    check("budgets", errors, unknowns)

    # A time-resource constraint requires a separate resource-constrained oracle.
    # Do not mark an unconstrained shortest distance as the constrained optimum.
    if constraints.get("max_duration_s") is not None:
        check("optimality", unknowns=["resource_constrained_optimum_not_supplied"])
    else:
        try:
            oracle_cost, settled = _oracle(raw_graph, effective_graph, start, end, strategy)
            actual_cost = sum(_objective_cost(raw_graph, effective_graph, edge, strategy) for edge in edges)
            report["metrics"].update(oracle_cost=oracle_cost, selected_objective_cost=actual_cost,
                                     oracle_objective_settled_nodes=settled)
            check("optimality", [] if math.isclose(actual_cost, oracle_cost, rel_tol=1e-9, abs_tol=1e-6)
                  else ["suboptimal_route"])
        except (KeyError, ValueError, TypeError):
            check("optimality", unknowns=["independent_objective_unavailable"])

    _audit_events(
        expected_task.get("events"), data, edges, check,
        require_overview=expected_task.get("require_event_overview", True) is not False,
    )
    return report


def _audit_events(events, data, edges, check, *, require_overview=True):
    if not isinstance(events, list):
        check("event_explanations", unknowns=["event_truth_unavailable"])
        return
    errors, expected_overview, expected_route = [], {}, {}
    selected = set(edges)
    for event in events:
        if (not isinstance(event, Mapping) or event.get("id") is None
                or event.get("action") not in {"notice", "slowdown", "closure"}
                or not isinstance(event.get("edge_ids"), list)):
            check("event_explanations", unknowns=["invalid_event_truth"])
            return
        identifier = event["id"]
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or identifier in expected_overview:
            check("event_explanations", unknowns=["invalid_event_truth"])
            return
        bound = {_edge(edge) for edge in event["edge_ids"]}
        if None in bound:
            check("event_explanations", unknowns=["invalid_event_truth"])
            return
        expected_overview[identifier] = event
        if event["action"] in {"notice", "slowdown"} and selected.intersection(bound):
            expected_route[identifier] = event

    for field, expected in (("road_condition_overview", expected_overview), ("route_events", expected_route)):
        if field == "road_condition_overview" and field not in data and not require_overview:
            continue
        actual = data.get(field)
        if not isinstance(actual, list):
            errors.append(f"missing_{field}")
            continue
        actual_by_id = {}
        for event in actual:
            if not isinstance(event, Mapping) or not isinstance(event.get("id"), (str, int)) or isinstance(event.get("id"), bool):
                errors.append(f"invalid_{field}")
                continue
            if event["id"] in actual_by_id:
                errors.append(f"duplicate_{field}")
            actual_by_id[event["id"]] = event
        if set(actual_by_id) != set(expected):
            errors.append(f"{field}_ids_mismatch")
        fields = ("action", "name", "description", "start_time", "end_time", "cost_multiplier") if field == "route_events" else ("action", "name")
        for identifier in set(actual_by_id).intersection(expected):
            for key in fields:
                if key not in expected[identifier]:
                    continue
                expected_value = expected[identifier][key]
                if key == "description" and isinstance(expected_value, str):
                    expected_value = expected_value[:100]
                if key == "name" and field == "road_condition_overview" and isinstance(expected_value, str):
                    expected_value = " ".join(expected_value.split())[:60]
                if actual_by_id[identifier].get(key) != expected_value:
                    errors.append(f"{field}_{key}_mismatch")
    count = data.get("road_conditions_applied")
    if isinstance(count, bool) or count != len(expected_overview):
        errors.append("event_applied_count_mismatch")
    check("event_explanations", errors)
