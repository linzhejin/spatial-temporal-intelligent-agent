"""Source-derived development cases for simulated road closures.

The event snapshot is a controlled input, not evidence of an actual campus
closure. Gold establishes only decisions within the frozen graph.
"""
from __future__ import annotations

from typing import Any

from experiments.controlled_route_cases import _MODE_TEXT, _anchor, _digest, _turn
from experiments.controlled_route_gold import derive_graph_route
from experiments.independent_route_oracle import inspect_route_edges
from experiments.independent_source_snapshot import active_closed_edges


_CATEGORIES = {"compound_tools": "feasible", "event_infeasible": "infeasible"}


def _edge_order(edge: tuple[Any, Any, Any]) -> tuple[str, ...]:
    return tuple(f"{type(part).__name__}:{part}" for part in edge)


def make_event_case(
    graph: Any, anchors: dict[str, dict[str, Any]],
    edge_evidence: dict[str, set[tuple[Any, Any, Any]]],
    event_snapshot: dict[str, Any], *, template: str, case_id: str,
    category: str, template_group: str, od_group: str,
    snapshot_id: str, generator_sha256: str, template_sha256: str,
    parameters: dict[str, Any],
    template_id: str = "event_route_v1",
) -> dict[str, Any]:
    """Build one decidable event-plus-route task from a frozen graph."""
    if category not in _CATEGORIES:
        raise ValueError("unsupported controlled event category")
    if (not isinstance(parameters, dict) or parameters.get("expected_verdict") !=
            _CATEGORIES[category] or not isinstance(parameters.get("via_poi_ids"), list)
            or parameters["via_poi_ids"] or parameters.get("mode") not in _MODE_TEXT
            or not isinstance(parameters.get("avoid_steps"), bool)
            or not isinstance(event_snapshot, dict)
            or event_snapshot.get("scope") != "synthetic_development_only"
            or event_snapshot.get("campus_truth") is not False
            or parameters.get("event_snapshot_id") != event_snapshot.get("id")
            or not isinstance(event_snapshot.get("id"), str)):
        raise ValueError("invalid simulated event case parameters or source")
    if (not isinstance(template, str) or any(marker not in template for marker in
            ("{start}", "{end}", "{mode_text}"))
            or "路况" not in template or "全程" not in template):
        raise ValueError("event template must state task and endpoints")
    if not isinstance(edge_evidence, dict) or not all(
            isinstance(edge_evidence.get(key), set) for key in ("permitted", "unknown")):
        raise ValueError("explicit graph edge evidence required")
    start = _anchor(anchors, parameters["start_poi_id"])
    end = _anchor(anchors, parameters["end_poi_id"])
    if start["poi_id"] == end["poi_id"]:
        raise ValueError("event case needs distinct endpoints")
    try:
        prompt = template.format(start=start["name"], end=end["name"],
                                 mode_text=_MODE_TEXT[parameters["mode"]])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("event template cannot be rendered") from exc
    if not prompt.strip():
        raise ValueError("empty event prompt")
    permitted = edge_evidence["permitted"]
    unknown = edge_evidence["unknown"]
    closed = active_closed_edges(graph, event_snapshot, parameters["mode"])
    if not closed or not closed <= permitted:
        raise ValueError("active closure must affect permitted route edges")
    common = dict(
        start_node=start["node_id"], end_node=end["node_id"], via_nodes=[],
        mode=parameters["mode"], permitted_edge_ids=permitted,
        unknown_edge_ids=unknown, avoid_steps=parameters["avoid_steps"],
    )
    baseline = derive_graph_route(graph, closed_edge_ids=set(), **common)
    result = derive_graph_route(graph, closed_edge_ids=closed, **common)
    if baseline["verdict"] != "feasible":
        raise ValueError("baseline route must be confirmed feasible")
    if result["verdict"] != parameters["expected_verdict"]:
        raise ValueError("event verdict differs from independent graph oracle")
    if result["verdict"] == "feasible":
        if (result["length_m"] <= baseline["length_m"] + 0.001
                or result["route_edge_ids"] == baseline["route_edge_ids"]):
            raise ValueError("compound event did not force a detour")
        checked = inspect_route_edges(
            graph, result["route_edge_ids"], start_node=start["node_id"],
            end_node=end["node_id"], via_nodes=[], mode=parameters["mode"],
            permitted_edge_ids=permitted, uncertain_edge_ids=unknown,
            closed_edge_ids=closed, avoid_steps=parameters["avoid_steps"],
        )
        if checked["verdict"] != "pass" or checked["length_m"] != result["length_m"]:
            raise ValueError("event route disagrees with independent edge check")
    elif result["reason"] != "no_path_in_frozen_eligible_graph":
        raise ValueError("event infeasibility must mean no graph path")
    turn = _turn(parameters, result["verdict"])
    turn["required_outputs"] = (["road_events", "route"] if
                                 result["verdict"] == "feasible" else
                                 ["road_events", "infeasibility_explanation"])
    snapshot_sha = _digest(event_snapshot)
    parameter_sha = _digest(parameters)
    oracle_sha = _digest(result)
    return {
        "id": case_id, "category": category,
        "template_group": template_group, "od_group": od_group,
        "snapshot_id": snapshot_id,
        "messages": [{"role": "user", "content": prompt}],
        "expected_turns": [turn],
        "gold_derivation": {
            "kind": "controlled_template", "generator_id": "controlled_event_cases_v1",
            "template_id": template_id, "generator_sha256": generator_sha256,
            "template_sha256": template_sha256,
            "parameters": parameters, "parameters_sha256": parameter_sha,
            "unresolved_ambiguity": False,
            "evidence_basis": ["template_parameter", "independent_graph_oracle",
                               "frozen_event_snapshot"],
            "oracle_result": result, "baseline_result": baseline,
            "event_snapshot_sha256": snapshot_sha,
            "active_closed_edge_ids": [list(edge) for edge in sorted(closed, key=_edge_order)],
        },
        "adjudication": {
            "status": "generated_verified", "method": "deterministic_source_rules",
            "evidence": [
                {"kind": "template_parameter", "sha256": parameter_sha},
                {"kind": "independent_graph_oracle", "sha256": oracle_sha},
                {"kind": "frozen_event_snapshot", "sha256": snapshot_sha},
            ],
        },
    }


def replay_event_case(
    case: dict[str, Any], graph: Any, anchors: dict[str, dict[str, Any]],
    edge_evidence: dict[str, set[tuple[Any, Any, Any]]],
    event_snapshot: dict[str, Any], *, template: str,
) -> None:
    """Fail if the question, label, closure or provenance changed."""
    derivation = case.get("gold_derivation")
    if not isinstance(derivation, dict):
        raise ValueError("missing event gold derivation")
    rebuilt = make_event_case(
        graph, anchors, edge_evidence, event_snapshot, template=template,
        case_id=case["id"], category=case["category"],
        template_group=case["template_group"], od_group=case["od_group"],
        snapshot_id=case["snapshot_id"],
        generator_sha256=derivation["generator_sha256"],
        template_sha256=derivation["template_sha256"],
        parameters=derivation["parameters"],
        template_id=derivation["template_id"],
    )
    for key in ("messages", "expected_turns", "gold_derivation", "adjudication"):
        if case.get(key) != rebuilt[key]:
            raise ValueError(f"event case {key} differs from source-derived gold")
