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


def audit_scenario_splits(
    development: dict[str, Any], test: dict[str, Any], *,
    external_gates: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Audit leakage and readiness. External gates default to unverified."""
    dev = development.get("cases") if isinstance(development.get("cases"), list) else []
    tst = test.get("cases") if isinstance(test.get("cases"), list) else []
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
    if not dev:
        block("development_split_empty", "The development split has no cases.")
    if not tst:
        block("test_split_empty", "The blind test split has no cases.")
    for name, cases in (("development", dev), ("test", tst)):
        ids = [c.get("id") for c in cases if isinstance(c, dict)]
        if len(ids) != len(cases) or any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
            block(f"{name}_ids_invalid", f"{name.title()} case IDs must be unique, nonempty strings.")

    missing_groups = {}
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

    codes = {"scenario_ids": "scenario_id_overlap", "template_groups": "template_group_overlap",
             "od_groups": "od_group_overlap", "prompt_fingerprints": "prompt_overlap",
             "source_groups": "source_group_overlap"}
    for name, values in overlaps.items():
        if values:
            block(codes[name], f"{len(values)} {name.replace('_', ' ')} are shared between splits.")

    if type(test.get("repeats")) is not int or test.get("repeats") != 3:
        block("test_repeats_invalid", "Formal test requires exactly three repeats per scenario.")
    if len(tst) != 300:
        block("test_case_count_invalid", f"Formal test requires 300 cases; found {len(tst)}.")
    category_counts = Counter()
    for case in tst:
        category = case.get("category") if isinstance(case, dict) else None
        if isinstance(category, str):
            category_counts[category] += 1
    for category in sorted(CATEGORIES):
        if category_counts[category] != 50:
            block("test_category_quota_invalid", f"Formal test requires 50 {category} cases; found {category_counts[category]}.")
    invalid_categories = sum(not isinstance(c.get("category"), str)
                             or c["category"] not in CATEGORIES
                             for c in tst if isinstance(c, dict))
    if invalid_categories:
        block("test_category_invalid", f"{invalid_categories} test cases have an unknown category.")
    unreviewed = sum(not _independently_adjudicated(c) for c in tst if isinstance(c, dict))
    if unreviewed:
        block("test_adjudication_incomplete", f"{unreviewed} test cases lack manual independent review evidence.")
    invalid_snapshots = sum(not isinstance(c.get("snapshot_id"), str)
                            or not _SNAPSHOT.fullmatch(c["snapshot_id"])
                            for c in tst if isinstance(c, dict))
    if invalid_snapshots:
        block("test_snapshot_id_invalid", f"{invalid_snapshots} test cases lack a frozen input fingerprint.")
    elif len({c["snapshot_id"] for c in tst if isinstance(c, dict)}) > 1:
        block("test_snapshot_mismatch", "Test cases refer to more than one frozen input snapshot.")

    try:
        validate_dataset(test, formal=True)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        block("test_dataset_contract_invalid", f"Formal dataset contract failed: {exc}")

    external_gates = external_gates or {}
    for gate in _GATES:
        if external_gates.get(gate) is not True:
            block(f"external_gate_unverified:{gate}", f"Formal run gate '{gate}' has not been verified.")
    leakage_evaluable = bool(dev and tst) and not any(missing_groups.values())
    if not leakage_evaluable:
        block("split_comparison_unavailable",
              "Both nonempty splits with template groups, OD groups and user prompts are required to evaluate leakage.")
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
        "audit_version": "scenario-split-audit-v1",
        "leakage_evaluable": leakage_evaluable,
        "leakage_free": leakage_free,
        "formal_ready": leakage_free and not blockers,
        "blockers": blockers,
        "dataset_counts": {"development": len(dev), "test": len(tst),
                            "test_by_category": {k: category_counts[k] for k in sorted(CATEGORIES)},
                            "test_adjudication_status": dict(sorted(adjudication_counts.items()))},
        "overlap_counts": {k: len(v) for k, v in overlaps.items()},
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
