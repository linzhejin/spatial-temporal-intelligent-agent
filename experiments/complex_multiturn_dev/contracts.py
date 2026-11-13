"""Validate and score development scenarios without consulting product verdicts."""
from __future__ import annotations

import json
from pathlib import Path


GROUPS = {"search", "itinerary", "clarification", "events"}
OPS = {"eq", "contains", "not_contains", "ordered_contains", "exists", "absent"}


def read_dataset(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_dataset(data):
    if (not isinstance(data, dict) or data.get("scope") != "development_only"
            or data.get("formal_test") is not False or not isinstance(data.get("cases"), list)):
        raise ValueError("development-only dataset required")
    cases = data["cases"]
    if len(cases) != 32 or len({c.get("id") for c in cases}) != 32:
        raise ValueError("expected 32 uniquely identified cases")
    for group in GROUPS:
        items = [c for c in cases if c.get("group") == group]
        if len(items) != 8 or sum(c.get("phase") == "core" for c in items) != 6:
            raise ValueError("each group needs six core cases and two variants")
        if sum(c.get("phase") == "variant" for c in items) != 2:
            raise ValueError("invalid variant allocation")
    _validate_cases(cases)


def validate_supplement(data):
    """Allow separately identified development regressions without changing the 32-case protocol."""
    if (not isinstance(data, dict) or data.get("scope") != "development_supplement"
            or data.get("formal_test") is not False or not isinstance(data.get("cases"), list)):
        raise ValueError("development-only supplement required")
    cases = data["cases"]
    if not 1 <= len(cases) <= 10 or any(
            not isinstance(c.get("id"), str) or not c["id"].startswith("supplement_")
            or c.get("group") != "search" or c.get("phase") != "core" for c in cases):
        raise ValueError("invalid supplementary case")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("supplementary case ids must be unique")
    _validate_cases(cases)


def _validate_cases(cases):
    for case in cases:
        if (case.get("group") not in GROUPS or not isinstance(case.get("title"), str)
                or not isinstance(case.get("turns"), list) or not 3 <= len(case["turns"]) <= 5):
            raise ValueError("invalid scenario")
        for turn in case["turns"]:
            if (not isinstance(turn.get("user"), str) or not turn["user"].strip()
                    or not isinstance(turn.get("checks"), list) or not turn["checks"]):
                raise ValueError("each user turn needs independent checks")
            for check in turn["checks"]:
                if (not isinstance(check.get("field"), str) or check.get("op") not in OPS
                        or (check.get("op") not in {"exists", "absent"} and "value" not in check)):
                    raise ValueError("invalid check")


def assess_turn(expected, observed):
    """Unknown input stays unknown; only present explicit contradictions fail."""
    failures, unknowns = [], []
    if observed.get("status") in {"failed", "agent_failed", "cancelled", "timeout", "transport_error"}:
        failures.append("run:" + observed["status"])
    for check in expected["checks"]:
        field, op = check["field"], check["op"]
        value = observed.get(field)
        tag = field + ":" + op
        if value is None:
            if op == "absent" and field in observed:
                continue
            if field in observed and (op == "exists" or op == "eq" and check.get("value") is not None):
                failures.append(tag)
                continue
            unknowns.append(tag)
            continue
        target = check.get("value")
        if op == "eq":
            matches = value == target
        elif op == "contains":
            matches = isinstance(value, (str, list, tuple)) and target in value
        elif op == "not_contains":
            matches = isinstance(value, (str, list, tuple)) and target not in value
        elif op == "exists":
            matches = bool(value)
        elif op == "absent":
            matches = not value
        else:
            if not isinstance(value, list) or not isinstance(target, list):
                unknowns.append(tag)
                continue
            cursor = iter(value)
            matches = all(any(item == wanted for item in cursor) for wanted in target)
        if not matches:
            failures.append(tag)
    return {"verdict": "fail" if failures else "unknown" if unknowns else "pass",
            "failures": failures, "unknowns": unknowns}
