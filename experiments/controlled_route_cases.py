"""Generate and replay unambiguous route pilot cases from frozen graph data.

These cases are development material until a blind split, full source manifest,
and the formal runner are in place. They make no field-access claims.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from experiments.controlled_route_gold import derive_graph_route
from experiments.independent_route_oracle import inspect_route_edges


_MODE_TEXT = {"walk": "步行", "bike": "骑行", "drive": "驾车"}


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _anchor(anchors: dict[str, dict[str, Any]], poi_id: str) -> dict[str, Any]:
    anchor = anchors.get(poi_id)
    if (not isinstance(anchor, dict) or anchor.get("poi_id") != poi_id
            or not isinstance(anchor.get("name"), str) or not anchor["name"].strip()
            or anchor.get("access_status") != "last_meter_unverified"
            or anchor.get("node_id") is None):
        raise ValueError(f"invalid or missing POI anchor: {poi_id}")
    return anchor


def _prompt(template: str, parameters: dict[str, Any],
            anchors: dict[str, dict[str, Any]]) -> str:
    start = _anchor(anchors, parameters["start_poi_id"])
    end = _anchor(anchors, parameters["end_poi_id"])
    vias = [_anchor(anchors, value) for value in parameters["via_poi_ids"]]
    if any(marker not in template for marker in
           ("{start}", "{end}", "{mode_text}", "{constraint_text}")):
        raise ValueError("route template must state endpoints, mode and constraints")
    if vias and "{vias}" not in template:
        raise ValueError("via template must state the ordered destinations")
    constraints = []
    if parameters.get("avoid_steps"):
        constraints.append("不要走台阶")
    if parameters.get("max_distance_m") is not None:
        constraints.append(f"全程不超过{parameters['max_distance_m']:g}米")
    constraint_text = "且" + "、".join(constraints) if constraints else ""
    try:
        prompt = template.format(
            start=start["name"], end=end["name"],
            vias="、".join(item["name"] for item in vias),
            mode_text=_MODE_TEXT[parameters["mode"]],
            constraint_text=constraint_text,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("route template cannot be rendered") from exc
    if not prompt.strip():
        raise ValueError("empty route prompt")
    return prompt


def _turn(parameters: dict[str, Any], verdict: str) -> dict[str, Any]:
    constraints = {}
    if parameters.get("avoid_steps"):
        constraints["avoid_steps"] = True
    if parameters.get("max_distance_m") is not None:
        constraints["max_distance_m"] = parameters["max_distance_m"]
    feasible = verdict == "feasible"
    return {
        "status": "completed" if feasible else "infeasible",
        "kind": "route" if feasible else "no_route",
        "feasibility": verdict,
        "start_poi_id": parameters["start_poi_id"],
        "end_poi_id": parameters["end_poi_id"],
        "via_poi_ids": list(parameters["via_poi_ids"]),
        "hard_constraints": constraints,
        "required_outputs": ["route"] if feasible else ["infeasibility_explanation"],
        "candidate_poi_ids": [], "clarification_slot": None,
        "allowed_unknowns": ["last_meter_unverified"],
        "travel_mode": parameters["mode"], "strategy": "shortest",
        "intent_label": "route_planning",
    }


def make_route_case(
    graph: Any, anchors: dict[str, dict[str, Any]], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]],
    templates: dict[str, str], template_id: str,
    case_id: str, category: str, template_group: str, od_group: str,
    snapshot_id: str, generator_sha256: str, template_sha256: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Generate a single graph-scoped case; reject undecidable labels."""
    if (not isinstance(parameters, dict) or not isinstance(parameters.get("via_poi_ids"), list)
            or parameters.get("mode") not in _MODE_TEXT
            or not isinstance(parameters.get("avoid_steps"), bool)
            or not isinstance(templates.get(template_id), str)):
        raise ValueError("invalid route case parameters or template")
    if category not in {"mode_access", "multi_via_itinerary"}:
        raise ValueError("unsupported route pilot category")
    if category == "multi_via_itinerary" and not parameters["via_poi_ids"]:
        raise ValueError("multi-via case needs at least one via POI")
    if category == "mode_access" and parameters["via_poi_ids"]:
        raise ValueError("mode-access case cannot silently include a via")
    start = _anchor(anchors, parameters["start_poi_id"])
    end = _anchor(anchors, parameters["end_poi_id"])
    vias = [_anchor(anchors, value) for value in parameters["via_poi_ids"]]
    prompt = _prompt(templates[template_id], parameters, anchors)
    result = derive_graph_route(
        graph, start_node=start["node_id"], end_node=end["node_id"],
        via_nodes=[item["node_id"] for item in vias], mode=parameters["mode"],
        permitted_edge_ids=permitted_edge_ids, unknown_edge_ids=unknown_edge_ids,
        closed_edge_ids=closed_edge_ids,
        avoid_steps=parameters["avoid_steps"],
        max_distance_m=parameters.get("max_distance_m"),
    )
    if result["verdict"] == "unknown":
        raise ValueError("unknown graph feasibility cannot be generated_verified gold")
    if result["verdict"] == "feasible":
        check = inspect_route_edges(
            graph, result["route_edge_ids"], start_node=start["node_id"],
            end_node=end["node_id"], via_nodes=[item["node_id"] for item in vias],
            mode=parameters["mode"], permitted_edge_ids=permitted_edge_ids,
            uncertain_edge_ids=unknown_edge_ids, closed_edge_ids=closed_edge_ids,
            avoid_steps=parameters["avoid_steps"],
            max_distance_m=parameters.get("max_distance_m"),
        )
        if check["verdict"] != "pass" or check["length_m"] != result["length_m"]:
            raise ValueError("independent route edge check disagrees with graph oracle")
    parameter_sha = _digest(parameters)
    oracle_sha = _digest(result)
    return {
        "id": case_id, "category": category,
        "template_group": template_group, "od_group": od_group,
        "snapshot_id": snapshot_id,
        "messages": [{"role": "user", "content": prompt}],
        "expected_turns": [_turn(parameters, result["verdict"])],
        "gold_derivation": {
            "kind": "controlled_template", "generator_id": "controlled_route_cases_v1",
            "template_id": template_id, "generator_sha256": generator_sha256,
            "template_sha256": template_sha256,
            "parameters": parameters, "parameters_sha256": parameter_sha,
            "unresolved_ambiguity": False,
            "evidence_basis": ["template_parameter", "independent_graph_oracle"],
            "oracle_result": result,
        },
        "adjudication": {
            "status": "generated_verified", "method": "deterministic_source_rules",
            "evidence": [
                {"kind": "template_parameter", "sha256": parameter_sha},
                {"kind": "independent_graph_oracle", "sha256": oracle_sha},
            ],
        },
    }


def replay_route_case(
    case: dict[str, Any], graph: Any, anchors: dict[str, dict[str, Any]], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]], templates: dict[str, str],
) -> None:
    """Rebuild question and label from declared parameters, then compare."""
    derivation = case.get("gold_derivation")
    if not isinstance(derivation, dict):
        raise ValueError("missing gold derivation")
    rebuilt = make_route_case(
        graph, anchors, permitted_edge_ids=permitted_edge_ids,
        unknown_edge_ids=unknown_edge_ids, closed_edge_ids=closed_edge_ids,
        templates=templates, template_id=derivation["template_id"],
        case_id=case["id"], category=case["category"],
        template_group=case["template_group"], od_group=case["od_group"],
        snapshot_id=case["snapshot_id"],
        generator_sha256=derivation["generator_sha256"],
        template_sha256=derivation["template_sha256"],
        parameters=derivation["parameters"],
    )
    if case.get("messages") != rebuilt["messages"]:
        raise ValueError("prompt differs from source-derived template")
    if case.get("expected_turns") != rebuilt["expected_turns"]:
        raise ValueError("gold label differs from source-derived route")
    if case.get("gold_derivation") != rebuilt["gold_derivation"]:
        raise ValueError("gold derivation differs from source-derived route")
    if case.get("adjudication") != rebuilt["adjudication"]:
        raise ValueError("gold evidence differs from source-derived route")
