"""Independent, evidence-aware scoring for full conversation scenarios.

This module consumes pre-registered gold and projected observations. It never
uses the product's `completed` or verifier `pass` flag as route ground truth.
An independent reviewer must supply route and constraint findings separately.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import statistics
from typing import Any

from agents.tools import TOOL_NAMES


PROTOCOL_VERSION = "scenario-v1"
_SNAPSHOT_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
CATEGORIES = {
    "place_recommendation", "mode_access", "multi_via_itinerary",
    "compound_tools", "multi_turn_revision", "event_infeasible",
}
REQUIRED_GOLD = {
    "status", "kind", "feasibility", "start_poi_id", "end_poi_id",
    "via_poi_ids", "hard_constraints", "required_outputs",
    "candidate_poi_ids", "clarification_slot", "allowed_unknowns",
    "travel_mode", "strategy",
}
STATUSES = {"completed", "needs_input", "infeasible", "failed"}
KINDS = {"route", "candidates", "clarify", "information", "no_route"}
INTENT_LABELS = {
    "place_recommendation", "activity_recommendation", "route_planning",
    "route_revision", "route_comparison", "information_lookup",
    "clarification", "compound_request", "other",
}


def validate_dataset(dataset: dict[str, Any], *, formal: bool = False) -> None:
    """Reject incomplete gold and distinguish drafts from formal test data."""
    if dataset.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")
    if dataset.get("split") not in {"development", "test"}:
        raise ValueError("split must be development or test")
    if formal and dataset["split"] != "test":
        raise ValueError("formal runs require adjudicated test split")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset requires nonempty cases")
    repeats = dataset.get("repeats", 1)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if formal and (len(cases) != 300 or repeats != 3):
        raise ValueError("formal E1 requires 300 test scenarios and 3 repeats")
    ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case IDs must be unique and nonempty")
        ids.add(case_id)
        for field in ("category", "template_group", "od_group", "snapshot_id"):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ValueError(f"{case_id}: missing {field}")
        if case["category"] not in CATEGORIES:
            raise ValueError(f"{case_id}: invalid category")
        messages = case.get("messages")
        gold = case.get("expected_turns")
        adjudication = case.get("adjudication") or {}
        adjudicated = adjudication.get("status") == "adjudicated"
        if not isinstance(messages, list) or not messages or not isinstance(gold, list) or len(messages) != len(gold):
            raise ValueError(f"{case_id}: one expected turn is required per user message")
        for message in messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"{case_id}: invalid user message")
        for turn in gold:
            missing = REQUIRED_GOLD - turn.keys()
            if missing:
                raise ValueError(f"{case_id}: missing {', '.join(sorted(missing))}")
            if turn["status"] not in STATUSES or turn["kind"] not in KINDS:
                raise ValueError(f"{case_id}: invalid status or kind")
            if turn["feasibility"] not in {"feasible", "infeasible", "unknown", "not_applicable"}:
                raise ValueError(f"{case_id}: invalid feasibility")
            if formal and turn["kind"] in {"route", "no_route"} and turn["feasibility"] not in {"feasible", "infeasible"}:
                raise ValueError(f"{case_id}: formal spatial gold needs adjudicated feasibility")
            if not isinstance(turn["via_poi_ids"], list) or not isinstance(turn["candidate_poi_ids"], list):
                raise ValueError(f"{case_id}: POI order must be a list")
            if not isinstance(turn["hard_constraints"], dict) or not isinstance(turn["required_outputs"], list) or not isinstance(turn["allowed_unknowns"], list):
                raise ValueError(f"{case_id}: invalid requirements")
            intent = turn.get("intent_label")
            if intent is not None and intent not in INTENT_LABELS:
                raise ValueError(f"{case_id}: invalid intent_label")
            expected_tools = turn.get("expected_tools")
            if expected_tools is not None:
                _validate_expected_tools(case_id, expected_tools)
            if adjudicated and (intent is None or expected_tools is None):
                raise ValueError(f"{case_id}: adjudicated turns require intent_label and expected_tools")
        if adjudication.get("status") not in {"draft", "adjudicated"}:
            raise ValueError(f"{case_id}: adjudication status required")
        if formal and (adjudication["status"] != "adjudicated"
                       or adjudication.get("method") != "manual_independent"
                       or not adjudication.get("reviewer")
                       or not adjudication.get("evidence")):
            raise ValueError(f"{case_id}: formal runs require manually adjudicated gold")
    if formal and any(sum(case["category"] == category for case in cases) != 50
                      for category in CATEGORIES):
        raise ValueError("formal E1 requires 50 test scenarios per category")


def _validate_expected_tools(case_id: str, expected_tools: Any) -> None:
    if not isinstance(expected_tools, list) or not expected_tools:
        raise ValueError(f"{case_id}: expected_tools must be a nonempty list of acceptable sequences")
    if any(not isinstance(sequence, list) for sequence in expected_tools):
        raise ValueError(f"{case_id}: expected_tools must be a two-dimensional JSON array")
    seen: set[tuple[str, ...]] = set()
    for sequence in expected_tools:
        if any(not isinstance(name, str) or name not in TOOL_NAMES for name in sequence):
            raise ValueError(f"{case_id}: expected_tools contains an unknown tool")
        key = tuple(sequence)
        if key in seen:
            raise ValueError(f"{case_id}: expected_tools contains a duplicate sequence")
        seen.add(key)


def _append_once(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _tool_use_verdict(expected: dict[str, Any], actual: dict[str, Any] | None) -> str:
    """Compare only tool names/order; absence of opt-in trace is unknown."""
    alternatives = expected.get("expected_tools")
    if not isinstance(alternatives, list) or not alternatives:
        return "unadjudicated"
    if not isinstance(actual, dict) or not isinstance(actual.get("tool_trace"), list):
        return "unknown"
    sequence = []
    for call in actual["tool_trace"]:
        if not isinstance(call, dict) or not isinstance(call.get("tool_name"), str):
            return "unknown"
        sequence.append(call["tool_name"])
    return "pass" if sequence in alternatives else "fail"


def _tool_use_summary(verdicts: list[str]) -> dict[str, Any]:
    counts = Counter(verdicts)
    if counts["fail"]:
        status = "fail"
    elif counts["unknown"]:
        status = "unknown"
    elif counts["unadjudicated"] and counts["pass"]:
        status = "partial"
    elif counts["unadjudicated"]:
        status = "unadjudicated"
    else:
        status = "pass"
    return {
        "status": status,
        "pass_turns": counts["pass"],
        "fail_turns": counts["fail"],
        "unknown_turns": counts["unknown"],
        "unadjudicated_turns": counts["unadjudicated"],
        "turn_verdicts": verdicts,
    }


def evaluate_scenario(case: dict[str, Any], observed: dict[str, Any] | None, *,
                      _trusted_program_review: bool = False) -> dict[str, Any]:
    """Score all turns; absence of independent route evidence remains unknown."""
    adjudication = case.get("adjudication") or {}
    if not isinstance(adjudication, dict) or adjudication.get("status") != "adjudicated":
        execution_failures = []
        if observed is None:
            execution_failures.append("missing_observation")
        else:
            if observed.get("transport_error"):
                execution_failures.append("transport_error")
            if observed.get("input_messages") != case.get("messages"):
                execution_failures.append("input_mismatch")
        turns = observed.get("turns") if isinstance(observed, dict) else None
        spatial_output = any(
            isinstance(turn, dict) and isinstance(turn.get("route_edge_ids"), list)
            and bool(turn["route_edge_ids"])
            for turn in (turns if isinstance(turns, list) else [])
        )
        turn_count = len(case.get("expected_turns") or [])
        tool_use = _tool_use_summary(["unadjudicated"] * turn_count)
        return {
            "evaluation_status": "unadjudicated",
            "task_success": None,
            "failures": [],
            "execution_failures": execution_failures,
            "unknowns": ["gold_not_adjudicated"],
            "spatial_output": spatial_output,
            "hard_constraint_decidable": False,
            "hard_constraint_violation": None,
            "tool_use": tool_use,
        }

    failures: list[str] = []
    unknowns: list[str] = []
    spatial_output = False
    route_reviews: list[str] = []
    constraint_verdicts: list[str] = []
    tool_verdicts: list[str] = []
    if observed is None or observed.get("scenario_id") != case["id"]:
        tool_use = _tool_use_summary([
            _tool_use_verdict(expected, None) for expected in case["expected_turns"]
        ])
        return {"evaluation_status": "adjudicated", "task_success": False,
                "failures": ["missing_observation"],
                "unknowns": [], "spatial_output": False,
                "hard_constraint_decidable": False, "hard_constraint_violation": None,
                "tool_use": tool_use}
    if observed.get("input_messages") != case["messages"]:
        _append_once(failures, "input_mismatch")
    snapshot_id = case.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not _SNAPSHOT_FINGERPRINT.fullmatch(snapshot_id):
        _append_once(unknowns, "snapshot_not_frozen")
    elif observed.get("snapshot_id") != snapshot_id:
        _append_once(failures, "snapshot_mismatch")
    if observed.get("transport_error"):
        _append_once(failures, "transport_error")
    turns = observed.get("turns")
    if not isinstance(turns, list) or len(turns) != len(case["expected_turns"]):
        _append_once(failures, "turn_count_mismatch")
        turns = turns if isinstance(turns, list) else []
    for turn_index, (expected, actual) in enumerate(zip(case["expected_turns"], turns)):
        if not isinstance(actual, dict):
            _append_once(failures, "missing_turn")
            tool_verdicts.append(_tool_use_verdict(expected, None))
            continue
        tool_verdicts.append(_tool_use_verdict(expected, actual))
        if actual.get("status") != expected["status"]:
            if expected["feasibility"] == "feasible" and actual.get("status") == "infeasible":
                _append_once(failures, "false_refusal")
            elif expected["feasibility"] == "infeasible" and actual.get("status") == "completed":
                _append_once(failures, "infeasible_missed")
            else:
                _append_once(failures, "wrong_status")
        if actual.get("kind") != expected["kind"]:
            _append_once(failures, "wrong_kind")
        for field in ("start_poi_id", "end_poi_id"):
            if actual.get(field) != expected[field]:
                if (field == "start_poi_id" and expected[field] is not None
                        and actual.get(field) is None and actual.get("kind") == "candidates"):
                    _append_once(unknowns, "candidate_origin_unobserved")
                else:
                    _append_once(failures, "wrong_poi")
        if actual.get("travel_mode") != expected["travel_mode"]:
            _append_once(failures, "wrong_travel_mode")
        if actual.get("strategy") != expected["strategy"]:
            _append_once(failures, "wrong_strategy")
        if actual.get("via_poi_ids") != expected["via_poi_ids"]:
            _append_once(failures, "wrong_poi_order")
        program_candidate_review = (
            _trusted_program_review
            and isinstance(actual.get("independent_review"), dict)
            and actual["independent_review"].get("source") == "independent_program"
            and isinstance(actual["independent_review"].get("candidate_order"), dict)
        )
        computed_candidate = (adjudication.get("method") == "graph_computed"
                              and expected["kind"] == "candidates")
        if computed_candidate and not program_candidate_review:
            _append_once(unknowns, "candidate_evidence_missing")
        elif not program_candidate_review and actual.get("candidate_poi_ids") != expected["candidate_poi_ids"]:
            _append_once(failures, "wrong_candidate_order")
        if actual.get("clarification_slot") != expected["clarification_slot"]:
            if (actual.get("kind") == "clarify"
                    and expected["clarification_slot"] is not None
                    and actual.get("clarification_slot") is None):
                _append_once(unknowns, "clarification_slot_unobserved")
            else:
                _append_once(failures, "wrong_clarification")
        actual_constraints = actual.get("hard_constraints") or {}
        for key, value in expected["hard_constraints"].items():
            if actual_constraints.get(key) != value:
                _append_once(failures, "dropped_constraint")
        if any(key not in expected["hard_constraints"] and value for key, value in actual_constraints.items()):
            _append_once(failures, "unexpected_constraint")
        missing_outputs = set(expected["required_outputs"]) - set(actual.get("outputs") or [])
        if missing_outputs:
            _append_once(failures, "partial_completion")

        for unknown in actual.get("unknowns") or []:
            if unknown not in expected["allowed_unknowns"]:
                _append_once(failures, "unexpected_unknown")

        review = actual.get("independent_review")
        valid_review = (isinstance(review, dict)
                        and (review.get("source") == "independent_reviewer"
                             or (_trusted_program_review and review.get("source") == "independent_program"))
                        and bool(review.get("reviewer")) and bool(review.get("evidence")))
        for output in expected["required_outputs"]:
            if output == "route":
                continue
            if not valid_review:
                _append_once(unknowns, "output_review_missing")
                continue
            verdict = (review.get("outputs") or {}).get(output, "unknown")
            if verdict == "fail":
                _append_once(failures, "output_invalid")
            elif verdict != "pass":
                _append_once(unknowns, f"output_unknown:{output}")

        if "preserve_route_from_turn" in expected:
            previous_index = expected["preserve_route_from_turn"]
            previous = turns[previous_index] if (type(previous_index) is int
                                                 and 0 <= previous_index < turn_index
                                                 and isinstance(turns[previous_index], dict)) else {}
            previous_edges = previous.get("route_edge_ids") or []
            current_edges = actual.get("route_edge_ids") or []
            if current_edges and current_edges != previous_edges:
                _append_once(failures, "route_changed")
            # The server-only replay cannot observe the route still displayed
            # in the client. Require separate independent UI evidence.
            preservation = ((review.get("outputs") or {}).get("route_preservation")
                            if valid_review and review.get("source") == "independent_reviewer"
                            else None)
            if preservation == "fail":
                _append_once(failures, "route_changed")
            elif preservation != "pass" or not previous_edges:
                _append_once(unknowns, "route_preservation_unverified")

        has_route = bool(actual.get("route_edge_ids"))
        spatial_output |= has_route
        if not has_route:
            if expected["kind"] == "route" and actual.get("status") == "completed":
                _append_once(unknowns, "route_evidence_missing")
            continue
        if not valid_review:
            _append_once(unknowns, "route_review_missing")
            route_reviews.append("unknown")
            constraint_verdicts.append("unknown")
            continue
        route_verdict = review.get("route", "unknown")
        route_reviews.append(route_verdict)
        if route_verdict == "fail":
            _append_once(failures, "route_invalid")
        elif route_verdict != "pass":
            _append_once(unknowns, "route_review_unknown")
        for key, value in expected["hard_constraints"].items():
            if not value:
                continue
            verdict = (review.get("hard_constraints") or {}).get(key, "unknown")
            constraint_verdicts.append(verdict)
            if verdict == "fail":
                _append_once(failures, "hard_constraint_violation")
            elif verdict != "pass":
                _append_once(unknowns, f"hard_constraint_unknown:{key}")
    decidable = (spatial_output and bool(route_reviews)
                 and all(v in {"pass", "fail"} for v in route_reviews + constraint_verdicts))
    violation = any(v == "fail" for v in constraint_verdicts) if decidable else None
    tool_verdicts.extend(
        _tool_use_verdict(expected, None)
        for expected in case["expected_turns"][len(turns):]
    )
    return {"evaluation_status": "adjudicated",
            "task_success": not failures and not unknowns,
            "failures": failures, "unknowns": unknowns,
            "spatial_output": spatial_output,
            "hard_constraint_decidable": decidable,
            "hard_constraint_violation": violation,
            "tool_use": _tool_use_summary(tool_verdicts)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose every denominator; repeated attempts stay grouped by scenario."""
    by_case: dict[str, list[int]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    execution_failure_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    tool_use_counts: Counter[str] = Counter()
    tool_use_by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_ids: set[str] = set()
    adjudicated_attempts = 0
    unadjudicated_attempts = 0
    for record in records:
        score = record["score"]
        scenario_id = record["scenario_id"]
        scenario_ids.add(scenario_id)
        task_success = score.get("task_success")
        if score.get("evaluation_status") == "unadjudicated" or task_success is None:
            unadjudicated_attempts += 1
        else:
            adjudicated_attempts += 1
            by_case[scenario_id].append(int(bool(task_success)))
        failure_counts.update(score["failures"])
        execution_failure_counts.update(score.get("execution_failures") or [])
        unknown_counts.update(score["unknowns"])
        tool_use = score.get("tool_use") or {}
        for status, field in (("pass", "pass_turns"), ("fail", "fail_turns"),
                              ("unknown", "unknown_turns"),
                              ("unadjudicated", "unadjudicated_turns")):
            tool_use_counts[status] += int(tool_use.get(field, 0))
        intent_labels = record.get("intent_labels") or []
        verdicts = tool_use.get("turn_verdicts") or []
        for intent, verdict in zip(intent_labels, verdicts):
            if intent not in INTENT_LABELS or verdict not in {"pass", "fail", "unknown", "unadjudicated"}:
                continue
            tool_use_by_intent[intent][verdict] += 1
    spatial = sum(bool(r["score"]["spatial_output"]) for r in records)
    decidable = sum(bool(r["score"]["hard_constraint_decidable"]) for r in records)
    violations = sum(r["score"]["hard_constraint_violation"] is True for r in records)
    latencies = sorted(float(r["observation"]["latency_s"])
                       for r in records if r.get("observation") is not None
                       and r["observation"].get("latency_s") is not None)
    cost = {name: [r["observation"][name] for r in records
                   if r.get("observation") is not None and r["observation"].get(name) is not None]
            for name in ("tool_calls", "input_tokens", "output_tokens")}
    return {
        "n_scenarios": len(scenario_ids), "n_attempts": len(records),
        "adjudicated_scenarios": len(by_case),
        "adjudicated_attempts": adjudicated_attempts,
        "unadjudicated_scenarios": len(scenario_ids - set(by_case)),
        "unadjudicated_attempts": unadjudicated_attempts,
        "complete_task_success_attempts": sum(sum(v) for v in by_case.values()),
        "complete_task_success_denominator_attempts": adjudicated_attempts,
        "complete_task_success_rate_attempt": round(sum(sum(v) for v in by_case.values()) / adjudicated_attempts, 4) if adjudicated_attempts else None,
        "complete_task_success_rate_scenario_mean": round(sum(sum(v) / len(v) for v in by_case.values()) / len(by_case), 4) if by_case else None,
        "spatial_outputs": spatial, "spatial_output_coverage": round(spatial / len(records), 4) if records else None,
        "hard_constraint_decidable_outputs": decidable,
        "hard_constraint_violations": violations,
        "hard_constraint_violation_rate": round(violations / decidable, 4) if decidable else None,
        "tool_selection_pass_turns": tool_use_counts["pass"],
        "tool_selection_fail_turns": tool_use_counts["fail"],
        "tool_selection_unknown_turns": tool_use_counts["unknown"],
        "tool_selection_unadjudicated_turns": tool_use_counts["unadjudicated"],
        "tool_selection_decidable_turns": tool_use_counts["pass"] + tool_use_counts["fail"],
        "tool_selection_accuracy": round(
            tool_use_counts["pass"] / (tool_use_counts["pass"] + tool_use_counts["fail"]), 4
        ) if tool_use_counts["pass"] + tool_use_counts["fail"] else None,
        "tool_selection_by_intent": {
            intent: {
                "pass_turns": counts["pass"],
                "fail_turns": counts["fail"],
                "unknown_turns": counts["unknown"],
                "unadjudicated_turns": counts["unadjudicated"],
                "decidable_turns": counts["pass"] + counts["fail"],
                "accuracy": round(counts["pass"] / (counts["pass"] + counts["fail"]), 4)
                if counts["pass"] + counts["fail"] else None,
            }
            for intent, counts in sorted(tool_use_by_intent.items())
        },
        "latency_observed_attempts": len(latencies),
        "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95_s": round(latencies[math.ceil(0.95 * len(latencies)) - 1], 3) if latencies else None,
        "tool_calls_total": sum(cost["tool_calls"]),
        "input_tokens_total": sum(cost["input_tokens"]),
        "output_tokens_total": sum(cost["output_tokens"]),
        "cost_observed_attempts": {name: len(values) for name, values in cost.items()},
        "failure_categories": dict(sorted(failure_counts.items())),
        "execution_failure_categories": dict(sorted(execution_failure_counts.items())),
        "unknown_categories": dict(sorted(unknown_counts.items())),
    }
