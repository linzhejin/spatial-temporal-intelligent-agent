"""Promote only graph-decidable visible drafts to a development scoring set.

The result is never a blind holdout or field-passability gold. Scenic/flat
optimality and near-tied/uncertain recommendation cases remain unadjudicated.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from experiments.scenario_v2 import audit_candidate_order, validate_dataset


_SOURCE_NAMES = ("draft", "pois", "graph", "overrides", "reviews")


def build_graph_development_set(
    draft: dict[str, Any], route_report: dict[str, Any],
    candidate_report: dict[str, Any], source_sha256: dict[str, str],
) -> dict[str, Any]:
    """Copy cases whose current graph-only questions have decisive references."""
    if draft.get("version") != "scenarios-test-v2-authoring-draft":
        raise ValueError("only the visible v2 authoring draft is accepted")
    if any(not isinstance(source_sha256.get(name), str)
           or len(source_sha256[name]) != 64 for name in _SOURCE_NAMES):
        raise ValueError("all five source digests are required")
    for report in (route_report, candidate_report):
        if report.get("source_sha256") != source_sha256:
            raise ValueError("reference report has stale source digests")
    route_rows = {row["case_id"]: row for row in route_report["cases"]}
    candidate_rows = {row["case_id"]: row for row in candidate_report["cases"]}
    if len(route_rows) != len(route_report["cases"]) or len(candidate_rows) != len(candidate_report["cases"]):
        raise ValueError("duplicate reference case ID")
    fingerprint = hashlib.sha256(json.dumps(
        source_sha256, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    cases = []
    exclusions = []
    for raw in draft["cases"]:
        case_id = raw["id"]
        category = raw["category"]
        reason = None
        ref = None
        if category == "T1":
            row = route_rows.get(case_id)
            turn = raw["expected_turns"][0]
            if row is None:
                raise ValueError(f"missing T1 reference: {case_id}")
            if turn["strategy"] in {"scenic", "flat"}:
                reason = "preference_optimality_not_independently_scored"
            elif row["needs_review"] or not row["proposed_route_reference"]["graph_feasible"]:
                reason = "route_graph_or_access_uncertain"
            else:
                ref = row["proposed_route_reference"]
        elif category == "T2":
            row = candidate_rows.get(case_id)
            if row is None:
                raise ValueError(f"missing T2 reference: {case_id}")
            ref = row["proposed_candidate_reference"]
            ids = [item["poi_id"] for item in row["top_four"]]
            if row["review_needed"] or audit_candidate_order(ids, ref)["verdict"] != "pass":
                reason = "candidate_access_or_rank_boundary_uncertain"
        else:
            continue
        if reason:
            exclusions.append({"case_id": case_id, "reason": reason})
            continue
        case = deepcopy(raw)
        case["snapshot_id"] = fingerprint
        case["source_case_id"] = case_id
        case["id"] = "dev_graph_" + case_id
        case["source_group"] = raw["source_group"]
        case["adjudication"] = {
            "status": "adjudicated", "method": "graph_computed",
            "reviewer": None,
            "evidence": [f"{name}:sha256:{source_sha256[name]}"
                         for name in _SOURCE_NAMES],
        }
        if category == "T1":
            case["route_reference"] = ref
            case["expected_turns"][0]["feasibility"] = "feasible"
        else:
            case["expected_turns"][0]["candidate_poi_ids"] = ids
        cases.append(case)
    result = {
        "protocol_version": "scenario-v2",
        "version": "graph-development-v2",
        "split": "development", "repeats": 1,
        "authoring_status": "Visible development only; graph and POI evidence, not field or blind-test gold.",
        "snapshot_id": fingerprint,
        "source_sha256": source_sha256,
        "cases": cases, "excluded_cases": exclusions,
    }
    validate_dataset(result)
    return result
