"""Independent frozen-graph validation of user-visible route edge sequences."""
from __future__ import annotations

from experiments.complex_multiturn_dev.observation import _id, _refs
from experiments.independent_route_oracle import inspect_route_edges


def _unknown(reason):
    return {"verdict": "unknown", "issues": [], "unknowns": [reason]}


def timeout_evidence(result):
    code = result.get("failure_code") if isinstance(result, dict) else None
    if code in {"deadline_exceeded", "model_timeout", "tool_timeout", "timeout"}:
        return "timeout:" + code
    return None


def inspect_radius_transition(previous, current, user_text):
    """Flag a changed radius that has no corresponding user-supplied evidence."""
    from agents.distance_evidence import expansion_supported_by_message, radius_supported_by_message
    if type(previous) is not int or type(current) is not int or previous == current:
        return None
    if radius_supported_by_message(user_text, current, previous):
        return None
    if current == min(5000, previous * 2) and expansion_supported_by_message(user_text):
        return None
    return "search_radius_change_unrequested"


def inspect_expected_route_fields(turn, result, lookup):
    """Compare actual delivered route, not just separately saved task fields."""
    state = result.get("route_state") if isinstance(result.get("route_state"), dict) else {}
    actual = {"origin": _id(state.get("start"), lookup),
              "mode": state.get("travel_mode") or result.get("mode")}
    report = {"failures": [], "unknowns": []}
    for check in turn.get("checks") or []:
        field = check.get("field")
        if field not in actual or check.get("op") != "eq":
            continue
        if actual[field] is None:
            report["unknowns"].append(f"route_{field}_missing")
        elif actual[field] != check.get("value"):
            report["failures"].append(f"route_{field}_mismatch")
    return report


def inspect_reference_endpoints(rubric, result, lookup):
    state = result.get("route_state") if isinstance(result.get("route_state"), dict) else {}
    report = {"failures": [], "unknowns": []}
    for field, key in (("origin", "start"), ("destination", "end")):
        actual = _id(state.get(key), lookup)
        if actual is None:
            report["unknowns"].append(f"reference_{field}_missing")
        elif actual != rubric[f"reference_{field}"]:
            report["failures"].append(f"reference_{field}_mismatch")
    return report


def inspect_delivery_route(graph, anchors, evidence, lookup, result, closed_edges, *, avoid_steps=None):
    state = result.get("route_state") if isinstance(result.get("route_state"), dict) else {}
    edges = result.get("recommended_edge_ids")
    if not isinstance(edges, list) or not edges:
        return _unknown("route_edge_sequence_missing")
    mode = state.get("travel_mode") or result.get("mode")
    if mode not in evidence:
        return _unknown("mode_evidence_missing")
    start, end = _id(state.get("start"), lookup), _id(state.get("end"), lookup)
    via_refs = _refs(state.get("via"))
    via_ids = [_id(ref, lookup) for ref in via_refs]
    if (not isinstance(edges[0], list) or not isinstance(edges[-1], list)
            or len(edges[0]) != 3 or len(edges[-1]) != 3):
        return _unknown("route_edge_format_invalid")
    unknowns = []
    if start not in anchors or end not in anchors:
        unknowns.append("endpoint_anchor_missing")
    if any(ident not in anchors for ident in via_ids):
        unknowns.append("via_anchor_missing")
    if ((start in anchors and edges[0][0] != anchors[start]["node_id"])
            or (end in anchors and edges[-1][1] != anchors[end]["node_id"])):
        unknowns.append("product_snap_differs_from_independent_anchor")
    saved = state.get("hard_constraints") if isinstance(state.get("hard_constraints"), dict) else {}
    use_steps = avoid_steps if type(avoid_steps) is bool else bool(saved.get("avoid_steps"))
    checked = inspect_route_edges(
        graph, edges, start_node=edges[0][0], end_node=edges[-1][1],
        via_nodes=[anchors[ident]["node_id"] for ident in via_ids if ident in anchors], mode=mode,
        permitted_edge_ids=evidence[mode]["permitted"],
        uncertain_edge_ids=evidence[mode]["unknown"],
        closed_edge_ids=closed_edges, avoid_steps=use_steps,
        max_distance_m=saved.get("max_distance_m"),
    )
    checked["unknowns"] = list(dict.fromkeys([*checked["unknowns"], *unknowns]))
    if checked["verdict"] == "pass" and checked["unknowns"]:
        checked["verdict"] = "unknown"
    return checked
