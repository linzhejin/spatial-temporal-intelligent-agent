"""Generate and replay location-ranked recommendation development cases.

Answers are derived from reviewed POI activity tags and frozen graph distance.
They do not assert opening hours, public access, ratings or field walkability.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from experiments.controlled_recommendation_gold import rank_reachable_activity_pois


_LABELS = {
    "meal": "吃饭地点", "canteen_meal": "食堂", "restaurant_meal": "餐厅",
    "fast_food": "快餐", "coffee": "咖啡店", "tea_drink": "茶饮店",
    "study": "学习场所", "badminton": "羽毛球场地",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_recommendation_case(
    graph: Any, pois: list[dict[str, Any]], anchors: dict[str, dict[str, Any]], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]],
    templates: dict[str, str], template_id: str,
    case_id: str, template_group: str, od_group: str,
    snapshot_id: str, generator_sha256: str, template_sha256: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Freeze a prompt and its complete eligible-pool distance ranking."""
    if not isinstance(parameters, dict):
        raise ValueError("recommendation parameters required")
    activity = parameters.get("activity")
    if activity not in _LABELS or parameters.get("activity_label") != _LABELS[activity]:
        raise ValueError("reviewed activity label required")
    if parameters.get("mode") != "walk":
        raise ValueError("recommendation pilot supports walk only")
    template = templates.get(template_id)
    if (not isinstance(template, str)
            or not all(token in template for token in
                       ("{start}", "{activity_label}", "{limit}"))
            or not all(marker in template for marker in
                       ("已收录", "路网", "由近到远"))):
        raise ValueError("recommendation template must state catalog scope and ordering")
    start_id = parameters.get("start_poi_id")
    start = anchors.get(start_id)
    if not isinstance(start, dict) or not isinstance(start.get("name"), str):
        raise ValueError("frozen start anchor required")
    ranked = rank_reachable_activity_pois(
        graph, pois, anchors, start_poi_id=start_id, activity=activity,
        mode="walk", limit=parameters.get("limit"),
        permitted_edge_ids=permitted_edge_ids,
        unknown_edge_ids=unknown_edge_ids, closed_edge_ids=closed_edge_ids,
    )
    try:
        prompt = template.format(
            start=start["name"], activity_label=parameters["activity_label"],
            limit=parameters["limit"],
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("recommendation template cannot be rendered") from exc
    if not prompt.strip():
        raise ValueError("empty recommendation prompt")
    parameter_sha = _digest(parameters)
    catalog_sha = _digest(pois)
    oracle_sha = _digest(ranked)
    return {
        "id": case_id, "category": "place_recommendation",
        "template_group": template_group, "od_group": od_group,
        "snapshot_id": snapshot_id,
        "messages": [{"role": "user", "content": prompt}],
        "expected_turns": [{
            "status": "completed", "kind": "candidates",
            "feasibility": "not_applicable",
            "start_poi_id": start_id, "end_poi_id": None,
            "via_poi_ids": [], "hard_constraints": {},
            "required_outputs": ["candidates"],
            "candidate_poi_ids": ranked["candidate_poi_ids"],
            "clarification_slot": None,
            "allowed_unknowns": ["last_meter_unverified"],
            "travel_mode": "walk", "strategy": "nearest",
            "intent_label": "activity_recommendation",
        }],
        "gold_derivation": {
            "kind": "controlled_template",
            "generator_id": "controlled_recommendation_cases_v1",
            "template_id": template_id,
            "generator_sha256": generator_sha256,
            "template_sha256": template_sha256,
            "parameters": parameters, "parameters_sha256": parameter_sha,
            "unresolved_ambiguity": False,
            "evidence_basis": [
                "template_parameter", "catalog_attribute", "independent_graph_oracle",
            ],
            "oracle_result": ranked,
        },
        "adjudication": {
            "status": "generated_verified", "method": "deterministic_source_rules",
            "evidence": [
                {"kind": "template_parameter", "sha256": parameter_sha},
                {"kind": "catalog_attribute", "sha256": catalog_sha},
                {"kind": "independent_graph_oracle", "sha256": oracle_sha},
            ],
        },
    }


def replay_recommendation_case(
    case: dict[str, Any], graph: Any, pois: list[dict[str, Any]],
    anchors: dict[str, dict[str, Any]], *,
    permitted_edge_ids: set[tuple[Any, Any, Any]],
    unknown_edge_ids: set[tuple[Any, Any, Any]],
    closed_edge_ids: set[tuple[Any, Any, Any]],
    templates: dict[str, str],
) -> dict[str, Any]:
    """Recompute one case; packet replay binds groups and digests to the spec."""
    derivation = case.get("gold_derivation")
    if not isinstance(derivation, dict):
        raise ValueError("recommendation case differs from source specification")
    expected = make_recommendation_case(
        graph, pois, anchors,
        permitted_edge_ids=permitted_edge_ids,
        unknown_edge_ids=unknown_edge_ids, closed_edge_ids=closed_edge_ids,
        templates=templates, template_id=derivation["template_id"],
        case_id=case["id"], template_group=case["template_group"],
        od_group=case["od_group"], snapshot_id=case["snapshot_id"],
        generator_sha256=derivation["generator_sha256"],
        template_sha256=derivation["template_sha256"],
        parameters=derivation["parameters"],
    )
    if expected != case:
        raise ValueError("recommendation case differs from source specification")
    return expected
