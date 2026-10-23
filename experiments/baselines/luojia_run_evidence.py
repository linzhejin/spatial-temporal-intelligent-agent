"""Privacy-limited model/tool evidence projection from one actual Luojia run.

The source must be the persisted run-event stream. This module does not
manufacture a model call for deterministic turns or infer success from text.
"""
from __future__ import annotations

from typing import Any


_MODEL_KEYS = {
    "requested_model_id", "resolved_model_id", "tool_schema_sha256",
    "temperature", "max_tokens", "input_tokens", "output_tokens", "outcome",
}
_TOOL_OUTCOMES = {"success", "tool_error", "invalid_args", "exception"}
_HEADER_KEYS = {"data_snapshot_id", "event_snapshot_sha256", "event_clock_epoch",
                "enabled_tool_names", "extra_tool_names", "max_model_rounds_per_turn",
                "time_budget_s"}


def project_luojia_run_evidence(run: dict[str, Any], header: dict[str, Any]
                                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return the exact gate vocabulary without copying prompts or raw args."""
    events = run.get("events")
    if not isinstance(events, list) or not any(
            isinstance(event, dict) and event.get("event_type") == "agent_tool_trace_started"
            for event in events):
        raise ValueError("run lacks enabled model/tool trace")
    if set(header) != _HEADER_KEYS:
        raise ValueError("run evidence header must contain only comparison fields")
    records = [{**header, "kind": "run", "condition": "luojia"}]
    requests = input_tokens = output_tokens = 0
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("run event must be an object")
        kind = event.get("event_type")
        if kind not in {"agent_model_request", "agent_tool_call"}:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("model/tool event lacks structured payload")
        if kind == "agent_model_request":
            if not _MODEL_KEYS <= payload.keys():
                raise ValueError("model request lacks identity or usage")
            if any(type(payload[key]) is not int or payload[key] < 0
                   for key in ("input_tokens", "output_tokens")):
                raise ValueError("model request usage must be nonnegative integers")
            record = {"kind": "request", **{key: payload[key] for key in _MODEL_KEYS}}
            records.append(record)
            requests += 1
            input_tokens += payload["input_tokens"]
            output_tokens += payload["output_tokens"]
            continue
        outcome = payload.get("outcome")
        if outcome not in _TOOL_OUTCOMES:
            # Retry-verification skips are traces of policy decisions, not
            # executed tool calls. They must not inflate model tool usage.
            if isinstance(outcome, str) and outcome.startswith("skipped_after_"):
                continue
            raise ValueError("tool event has unsupported outcome")
        name = payload.get("tool_name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool event lacks a tool name")
        records.append({"kind": "tool", "name": name,
                        "outcome": "tool_error" if outcome == "exception" else outcome})
    return records, {"requests": requests, "input_tokens": input_tokens,
                     "output_tokens": output_tokens}
