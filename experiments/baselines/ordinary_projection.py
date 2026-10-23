"""Project the ordinary tool Agent into the same observable task fields as Luojia.

Only tool artifacts are treated as spatial evidence. A plausible final model
sentence cannot by itself establish a route or a nearby-place recommendation.
"""
from __future__ import annotations

from typing import Any

from experiments.scenario_live_adapter import project_live_run


def project_ordinary_turn(turn: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any]:
    artifacts = turn.get("artifacts") if isinstance(turn.get("artifacts"), list) else []
    chosen = next((item for item in reversed(artifacts)
                   if isinstance(item, dict) and any(
                       key in item for key in ("route", "candidates", "clarify"))), None)
    result: dict[str, Any] = {"response_kind": "chat"}
    status = turn.get("status")
    if chosen is not None and isinstance(chosen.get("route"), dict):
        result = {**chosen["route"], "response_kind": "route"}
    elif chosen is not None and isinstance(chosen.get("candidates"), list):
        result = {"response_kind": "candidates", "candidates": chosen["candidates"]}
        start_name = turn.get("selected_start_name")
        result["place_request"] = {
            "start": {"name": start_name} if isinstance(start_name, str) else None,
            "mode": turn.get("selected_mode"),
        }
    elif chosen is not None and isinstance(chosen.get("clarify"), dict):
        result = {"response_kind": "clarify", "place_request": {
            "pending_slot": chosen["clarify"].get("slot"),
            "start": {"name": turn["selected_start_name"]}
            if isinstance(turn.get("selected_start_name"), str) else None,
        }}
        if status == "completed":
            status = "needs_input"
    if isinstance(turn.get("weather_observation"), dict):
        result["weather"] = turn["weather_observation"]
        result["explanation"] = turn.get("message")
    projected = project_live_run({"status": status, "result": result}, lookup)
    trace = turn.get("tool_trace")
    if isinstance(trace, list):
        projected["tool_trace"] = [
            {key: call.get(key) for key in (
                "tool_name", "argument_keys", "outcome", "artifact_kind",
            )} for call in trace if isinstance(call, dict)
        ]
    return projected
