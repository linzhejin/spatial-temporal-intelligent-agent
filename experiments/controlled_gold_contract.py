"""Fail-closed contract for generated, source-derived scenario gold.

This checks the shape and provenance declarations. A later formal gate must
also verify the referenced source bytes and reproduce every derived label.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from experiments.controlled_source_manifest import verify_source_manifest
from experiments.scenario_evaluator import (
    CATEGORIES, INTENT_LABELS, KINDS, REQUIRED_GOLD, STATUSES,
    _validate_expected_tools,
)


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_EVIDENCE_BASIS = {
    "template_parameter", "catalog_attribute", "independent_graph_oracle",
    "frozen_event_snapshot", "independent_information_source",
}


def _has_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_controlled_dataset(
    dataset: dict[str, Any], *, formal: bool = False,
    source_manifest_bytes: bytes | None = None,
    source_root: Path | None = None,
) -> None:
    """Validate controlled labels without turning declarations into proof."""
    if not isinstance(dataset, dict) or dataset.get("protocol_version") != "scenario-v2":
        raise ValueError("unsupported protocol_version")
    if dataset.get("gold_layer") != "controlled_graph":
        raise ValueError("controlled dataset needs gold_layer=controlled_graph")
    if dataset.get("split") not in {"development", "test"}:
        raise ValueError("split must be development or test")
    if not _has_digest(dataset.get("source_manifest_sha256")):
        raise ValueError("source_manifest_sha256 is required")
    if not _has_digest(dataset.get("snapshot_id")):
        raise ValueError("frozen snapshot_id is required")
    cases = dataset.get("cases")
    repeats = dataset.get("repeats")
    if not isinstance(cases, list) or not cases:
        raise ValueError("controlled dataset requires nonempty cases")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if formal and (dataset["split"] != "test" or len(cases) != 300 or repeats != 3):
        raise ValueError("formal controlled test requires 300 cases and three repeats")
    ids = set()
    category_counts = Counter()
    for case in cases:
        if not isinstance(case, dict) or not _nonempty_text(case.get("id")) or case["id"] in ids:
            raise ValueError("case IDs must be unique nonempty strings")
        case_id = case["id"]
        ids.add(case_id)
        category = case.get("category")
        if not isinstance(category, str) or category not in CATEGORIES:
            raise ValueError(f"{case_id}: invalid category")
        category_counts[category] += 1
        for field in ("template_group", "od_group"):
            if not _nonempty_text(case.get(field)):
                raise ValueError(f"{case_id}: missing {field}")
        if case.get("snapshot_id") != dataset["snapshot_id"]:
            raise ValueError(f"{case_id}: snapshot_id differs from dataset")
        messages = case.get("messages")
        turns = case.get("expected_turns")
        if (not isinstance(messages, list) or not messages
                or not isinstance(turns, list) or len(messages) != len(turns)):
            raise ValueError(f"{case_id}: one expected turn is required per user message")
        if any(not isinstance(message, dict) or message.get("role") != "user"
               or not _nonempty_text(message.get("content")) for message in messages):
            raise ValueError(f"{case_id}: invalid user message")
        derivation = case.get("gold_derivation")
        if not isinstance(derivation, dict) or derivation.get("kind") != "controlled_template":
            raise ValueError(f"{case_id}: controlled gold_derivation is required")
        for field in ("generator_id", "template_id"):
            if not _nonempty_text(derivation.get(field)):
                raise ValueError(f"{case_id}: missing {field}")
        for field in ("generator_sha256", "template_sha256", "parameters_sha256"):
            if not _has_digest(derivation.get(field)):
                raise ValueError(f"{case_id}: missing {field}")
        parameters = derivation.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f"{case_id}: template parameters required")
        try:
            encoded_parameters = json.dumps(parameters, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{case_id}: invalid template parameters") from exc
        if hashlib.sha256(encoded_parameters).hexdigest() != derivation["parameters_sha256"]:
            raise ValueError(f"{case_id}: parameters_sha256 mismatch")
        if derivation.get("unresolved_ambiguity") is not False:
            raise ValueError(f"{case_id}: unresolved ambiguity cannot be automatic gold")
        basis = derivation.get("evidence_basis")
        if (not isinstance(basis, list) or not basis
                or any(not isinstance(value, str) or value not in _EVIDENCE_BASIS for value in basis)
                or len(set(basis)) != len(basis)):
            raise ValueError(f"{case_id}: invalid evidence_basis")
        if (any(isinstance(turn, dict) and turn.get("kind") in {"route", "no_route"}
                for turn in turns) and "independent_graph_oracle" not in basis):
            raise ValueError(f"{case_id}: route gold needs independent_graph_oracle")
        if category == "event_infeasible" and "frozen_event_snapshot" not in basis:
            raise ValueError(f"{case_id}: event gold needs frozen_event_snapshot")
        adjudication = case.get("adjudication")
        if not isinstance(adjudication, dict):
            raise ValueError(f"{case_id}: generated_verified adjudication required")
        if adjudication.get("status") != "generated_verified":
            raise ValueError(f"{case_id}: generated_verified adjudication required")
        if adjudication.get("method") != "deterministic_source_rules":
            raise ValueError(f"{case_id}: deterministic_source_rules required")
        if "reviewer" in adjudication:
            raise ValueError(f"{case_id}: reviewer belongs to separate human review layer")
        evidence = adjudication.get("evidence")
        if (not isinstance(evidence, list) or not evidence
                or any(not isinstance(item, dict) or item.get("kind") not in basis
                       or not _has_digest(item.get("sha256")) for item in evidence)):
            raise ValueError(f"{case_id}: verifiable evidence references required")
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError(f"{case_id}: invalid expected turn")
            missing = REQUIRED_GOLD - turn.keys()
            if missing:
                raise ValueError(f"{case_id}: missing {', '.join(sorted(missing))}")
            if (not isinstance(turn["status"], str) or turn["status"] not in STATUSES
                    or not isinstance(turn["kind"], str) or turn["kind"] not in KINDS):
                raise ValueError(f"{case_id}: invalid status or kind")
            if turn["feasibility"] not in {"feasible", "infeasible", "not_applicable"}:
                raise ValueError(f"{case_id}: controlled feasibility must be decidable")
            if (turn["kind"] == "candidates" and "catalog_attribute" not in basis):
                raise ValueError(f"{case_id}: candidates need catalog_attribute evidence")
            if (not isinstance(turn["via_poi_ids"], list)
                    or not isinstance(turn["candidate_poi_ids"], list)
                    or not isinstance(turn["hard_constraints"], dict)
                    or not isinstance(turn["required_outputs"], list)
                    or not isinstance(turn["allowed_unknowns"], list)):
                raise ValueError(f"{case_id}: invalid requirements")
            if ("weather" in turn["required_outputs"]
                    and "independent_information_source" not in basis):
                raise ValueError(f"{case_id}: weather needs independent_information_source")
            if ("road_events" in turn["required_outputs"]
                    and "frozen_event_snapshot" not in basis):
                raise ValueError(f"{case_id}: road_events need frozen_event_snapshot")
            intent = turn.get("intent_label")
            if intent is not None and intent not in INTENT_LABELS:
                raise ValueError(f"{case_id}: invalid intent_label")
            expected_tools = turn.get("expected_tools")
            if expected_tools is not None:
                _validate_expected_tools(case_id, expected_tools)
    if formal and any(category_counts[category] != 50 for category in CATEGORIES):
        raise ValueError("formal controlled test requires 50 cases per category")
    if formal:
        if not isinstance(source_manifest_bytes, bytes) or source_root is None:
            raise ValueError("source_manifest_bytes and source_root are required")
        if hashlib.sha256(source_manifest_bytes).hexdigest() != dataset["source_manifest_sha256"]:
            raise ValueError("source_manifest_sha256 differs from supplied bytes")
        try:
            manifest = json.loads(source_manifest_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("invalid source_manifest_bytes") from exc
        if manifest.get("profile") != "full":
            raise ValueError("formal controlled test requires full source profile")
        if verify_source_manifest(manifest, Path(source_root)) != dataset["snapshot_id"]:
            raise ValueError("snapshot_id differs from verified source inputs")
        for case in cases:
            derivation = case["gold_derivation"]
            for field, source in (("generator_sha256", "generator"),
                                  ("template_sha256", "templates")):
                if derivation[field] != manifest["inputs"][source]["sha256"]:
                    raise ValueError(f"{case['id']}: {field} differs from verified source")
