"""Audit scenario split leakage and formal readiness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
from typing import Any

from experiments.scenario_evaluator import CATEGORIES, validate_dataset
from experiments.scenario_v2 import GROUPS as V2_GROUPS, VERSION as V2_VERSION
from experiments.scenario_v2 import validate_dataset as validate_dataset_v2

_SNAPSHOT = re.compile(r"[0-9a-f]{64}\Z")
_SOURCES = ("source_group", "source_event_id", "flight_id", "source_video_id")
_GATES = ("frozen_snapshot_manifest", "agent_readiness_gate", "matching_git_commit", "clean_tracked_worktree")


def _strings(cases: list[dict[str, Any]], key: str) -> set[str]:
    return {v.strip() for c in cases if isinstance(c, dict)
            if isinstance((v := c.get(key)), str) and v.strip()}


def _prompts(cases: list[dict[str, Any]]) -> set[str]:
    out = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        messages = case.get("messages")
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                normalized = " ".join(msg["content"].casefold().split())
                if normalized:
                    out.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return out


def _has_user_prompt(case: Any) -> bool:
    if not isinstance(case, dict) or not isinstance(case.get("messages"), list):
        return False
    return any(isinstance(message, dict) and message.get("role") == "user"
               and isinstance(message.get("content"), str) and message["content"].strip()
               for message in case["messages"])


def _source_groups(cases: list[dict[str, Any]]) -> set[str]:
    return {value.strip() for case in cases if isinstance(case, dict)
            for key in _SOURCES if isinstance((value := case.get(key)), str) and value.strip()}


def _first_spatial_key(case: dict[str, Any]) -> tuple[str, str | None] | None:
    """Use actual gold endpoints; a template-prefixed OD label can hide reuse."""
    turns = case.get("expected_turns")
    if not isinstance(turns, list) or not turns or not isinstance(turns[0], dict):
        return None
    start, end = turns[0].get("start_poi_id"), turns[0].get("end_poi_id")
    return (start, end if isinstance(end, str) and end else None) if isinstance(start, str) and start else None


def _independently_adjudicated(case: dict[str, Any]) -> bool:
    adjudication = case.get("adjudication") or {}
    if not isinstance(adjudication, dict):
        return False
    evidence = adjudication.get("evidence")
    reviewer = adjudication.get("reviewer")
    return (adjudication.get("status") == "adjudicated"
            and adjudication.get("method") == "manual_independent"
            and isinstance(reviewer, str) and bool(reviewer.strip())
            and isinstance(evidence, list) and bool(evidence))


def _source_backed(case: dict[str, Any]) -> bool:
    try:
        validate_dataset_v2({"protocol_version": V2_VERSION, "split": "development",
                             "cases": [case]})
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
    return (case.get("adjudication") or {}).get("status") == "adjudicated"


def audit_scenario_splits(
    development: dict[str, Any], test: dict[str, Any], *,
    external_gates: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Audit leakage and readiness. External gates default to unverified."""
    dev = development.get("cases") if isinstance(development.get("cases"), list) else []
    tst = test.get("cases") if isinstance(test.get("cases"), list) else []
    v2 = test.get("protocol_version") == V2_VERSION
    overlaps = {
        "scenario_ids": _strings(dev, "id") & _strings(tst, "id"),
        "template_groups": _strings(dev, "template_group") & _strings(tst, "template_group"),
        "od_groups": _strings(dev, "od_group") & _strings(tst, "od_group"),
        "prompt_fingerprints": _prompts(dev) & _prompts(tst),
        "source_groups": _source_groups(dev) & _source_groups(tst),
    }
    blockers: list[dict[str, str]] = []

    def block(code: str, detail: str) -> None:
        blockers.append({"code": code, "detail": detail})

    if development.get("split") != "development":
        block("development_split_label", "Development dataset must declare split=development.")
    if test.get("split") != "test":
        block("test_split_label", "Test dataset must declare split=test.")
    if v2 and development.get("protocol_version") != V2_VERSION:
        block("development_protocol_mismatch", "Both splits must use scenario-v2.")
    if not dev:
        block("development_split_empty", "The development split has no cases.")
    if not tst:
        block("test_split_empty", "The blind test split has no cases.")
    for name, cases in (("development", dev), ("test", tst)):
        ids = [c.get("id") for c in cases if isinstance(c, dict)]
        if len(ids) != len(cases) or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            block(f"{name}_ids_invalid", f"{name.title()} case IDs must be unique, nonempty strings.")

    missing_groups = {}
    missing_sources = {}
    for name, cases in (("development", dev), ("test", tst)):
        missing = sum(not isinstance(case, dict)
                      or not isinstance(case.get("template_group"), str) or not case.get("template_group", "").strip()
                      or not isinstance(case.get("od_group"), str) or not case.get("od_group", "").strip()
                      or not _has_user_prompt(case)
                      for case in cases)
        missing_groups[name] = missing
        if missing:
            block(f"{name}_leakage_keys_missing",
                  f"{missing} {name} cases lack template/OD groups or user prompts needed for leakage checks.")
        if v2:
            source_missing = sum(
                not isinstance(case, dict) or not _source_groups([case])
                for case in cases
            )
            missing_sources[name] = source_missing
            if source_missing:
                block(f"{name}_source_keys_missing",
                      f"{source_missing} {name} cases lack a source group or event/flight ID; source leakage cannot be checked.")

    codes = {"scenario_ids": "scenario_id_overlap", "template_groups": "template_group_overlap",
             "od_groups": "od_group_overlap", "prompt_fingerprints": "prompt_overlap",
             "source_groups": "source_group_overlap"}
    for name, values in overlaps.items():
        if values:
            block(codes[name], f"{len(values)} {name.replace('_', ' ')} are shared between splits.")

    if type(test.get("repeats")) is not int or test.get("repeats") != 3:
        block("test_repeats_invalid", "Formal test requires exactly three repeats per scenario.")
    expected_count = 120 if v2 else 300
    categories = V2_GROUPS if v2 else CATEGORIES
    per_category = 24 if v2 else 50
    if len(tst) != expected_count:
        block("test_case_count_invalid", f"Formal test requires {expected_count} cases; found {len(tst)}.")
    category_counts = Counter()
    for case in tst:
        category = case.get("category") if isinstance(case, dict) else None
        if isinstance(category, str):
            category_counts[category] += 1
    for category in sorted(categories):
        if category_counts[category] != per_category:
            block("test_category_quota_invalid", f"Formal test requires {per_category} {category} cases; found {category_counts[category]}.")
    spatial_diversity = {}
    if v2:
        for category in sorted(V2_GROUPS):
            group_cases = [case for case in tst if isinstance(case, dict)
                           and case.get("category") == category]
            anchors = {_first_spatial_key(case) for case in group_cases}
            anchors.discard(None)
            distinct_starts = len({start for start, _ in anchors})
            distinct_od = len({(start, end) for start, end in anchors if end is not None})
            spatial_diversity[category] = {"distinct_start_pois": distinct_starts,
                                           "distinct_first_turn_od_pairs": distinct_od}
            if len(group_cases) == 24 and (distinct_starts < 8 if category in {"T2", "T3"}
                                           else distinct_od < 8):
                block("test_spatial_diversity_low",
                      f"{category} repeats too few actual starts or endpoint pairs across 24 cases; "
                      "template-specific OD labels do not establish spatial diversity.")
    invalid_categories = sum(not isinstance(c.get("category"), str)
                             or c["category"] not in categories
                             for c in tst if isinstance(c, dict))
    if invalid_categories:
        block("test_category_invalid", f"{invalid_categories} test cases have an unknown category.")
    unreviewed = sum(not _source_backed(c) if v2 else not _independently_adjudicated(c)
                     for c in tst if isinstance(c, dict))
    if unreviewed:
        source = "valid source-backed gold" if v2 else "manual independent review evidence"
        block("test_adjudication_incomplete", f"{unreviewed} test cases lack {source}.")
    invalid_snapshots = sum(not isinstance(c.get("snapshot_id"), str)
                            or not _SNAPSHOT.fullmatch(c["snapshot_id"])
                            for c in tst if isinstance(c, dict))
    if invalid_snapshots:
        block("test_snapshot_id_invalid", f"{invalid_snapshots} test cases lack a frozen input fingerprint.")
    elif len({c["snapshot_id"] for c in tst if isinstance(c, dict)}) > 1:
        block("test_snapshot_mismatch", "Test cases refer to more than one frozen input snapshot.")

    try:
        (validate_dataset_v2 if v2 else validate_dataset)(test, formal=True)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        block("test_dataset_contract_invalid", f"Formal dataset contract failed: {exc}")

    external_gates = external_gates or {}
    for gate in _GATES:
        if external_gates.get(gate) is not True:
            block(f"external_gate_unverified:{gate}", f"Formal run gate '{gate}' has not been verified.")
    leakage_evaluable = (bool(dev and tst) and not any(missing_groups.values())
                         and not any(missing_sources.values()))
    if not leakage_evaluable:
        block("split_comparison_unavailable",
              "Both nonempty splits need template/OD groups, user prompts and (for v2) source IDs to evaluate leakage.")
    leakage_free = leakage_evaluable and not any(overlaps.values())
    adjudication_counts = Counter()
    for case in tst:
        if not isinstance(case, dict):
            adjudication_counts["invalid_case"] += 1
            continue
        adjudication = case.get("adjudication") or {}
        status = adjudication.get("status", "missing") if isinstance(adjudication, dict) else "invalid"
        adjudication_counts[status if isinstance(status, str) else "invalid"] += 1
    return {
        "audit_version": "scenario-split-audit-v2" if v2 else "scenario-split-audit-v1",
        "leakage_evaluable": leakage_evaluable,
        "leakage_free": leakage_free,
        "formal_ready": leakage_free and not blockers,
        "blockers": blockers,
        "dataset_counts": {"development": len(dev), "test": len(tst),
                            "test_by_category": {k: category_counts[k] for k in sorted(categories)},
                            "test_adjudication_status": dict(sorted(adjudication_counts.items()))},
        "overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "test_spatial_diversity": spatial_diversity,
        "external_gates": {k: external_gates.get(k) is True for k in _GATES},
        "privacy": "Prompt overlaps are reported as counts only; prompt text and fingerprints are not emitted.",
    }


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit development/test scenario split hygiene.")
    parser.add_argument("--development", type=Path, default=Path("experiments/datasets/scenarios_dev_v1.json"))
    parser.add_argument("--test", type=Path, default=Path("experiments/datasets/scenarios_test_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("scripts/audit_output/scenario_split_readiness.json"))
    args = parser.parse_args()
    dev = json.loads(args.development.read_text(encoding="utf-8"))
    test = json.loads(args.test.read_text(encoding="utf-8"))
    report = audit_scenario_splits(dev, test)
    report["input_sha256"] = {"development": _hash(args.development), "test": _hash(args.test)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "leakage_free": report["leakage_free"],
                      "formal_ready": report["formal_ready"], "development_cases": len(dev.get("cases", [])),
                      "test_cases": len(test.get("cases", [])), "blocker_count": len(report["blockers"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
