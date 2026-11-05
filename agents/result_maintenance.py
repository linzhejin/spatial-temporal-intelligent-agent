"""Evidence for reusing the already completed prefix of a place itinerary.

The record identifies inputs; it is not a route validity or optimality proof.
Callers must still check current graph/events, ownership and task constraints.
"""

from __future__ import annotations

import hashlib
import json
import re


def _fingerprint(value) -> str | None:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(raw).hexdigest()


def weather_input_fingerprint(weather_info) -> str | None:
    """Identify the actual routing input, without claiming freshness or truth."""
    if not isinstance(weather_info, dict) or not weather_info:
        return None
    return _fingerprint(weather_info)


def capture_dependencies(legs: list[dict], *, task_revision: int | None) -> dict | None:
    if not isinstance(legs, list) or not legs:
        return None
    snapshots = []
    for leg in legs:
        if not isinstance(leg, dict):
            return None
        state, report = leg.get("route_state"), leg.get("route_verification")
        if (not isinstance(state, dict) or not isinstance(report, dict)
                or report.get("status") != "pass"):
            return None
        weather = leg.get("weather_input_sha256")
        if not isinstance(weather, str) or not re.fullmatch(r"[0-9a-f]{64}", weather):
            return None
        snapshot = (state.get("data_version"), state.get("road_condition_version"), state.get("travel_mode"), weather)
        if not all(isinstance(value, str) and value and value != "unavailable" for value in snapshot):
            return None
        snapshots.append(snapshot)
    if len(set(snapshots)) != 1:
        return None
    digest = _fingerprint(legs)
    if digest is None:
        return None
    data_version, event_version, mode, weather = snapshots[0]
    return {
        "schema_version": 1, "prefix_sha256": digest,
        "data_version": data_version, "road_condition_version": event_version,
        "travel_mode": mode, "weather_input_sha256": weather,
        "task_revision": task_revision if type(task_revision) is int else None,
        "depends_on": {
            "prefix_route": ["previous_stages", "mode", "constraints", "graph", "events", "weather_input"],
            "candidates": ["current_activity", "prefix_endpoint", "mode", "constraints", "graph", "events"],
            "itinerary": ["prefix_route", "selected_candidate", "mode", "constraints", "graph", "events", "weather_input"],
            "time": ["itinerary", "mode", "events", "weather_input"],
            "explanation": ["itinerary", "time", "events", "task"],
        },
    }


def may_keep_prefix(record: dict | None, legs: list[dict], *, mode: str, next_leg: dict | None = None) -> bool:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        return False
    current = capture_dependencies(legs, task_revision=record.get("task_revision"))
    if (current is None or mode != current["travel_mode"] or not isinstance(next_leg, dict)
            or next_leg.get("weather_input_sha256") != current["weather_input_sha256"]):
        return False
    return all(record.get(key) == current[key] for key in (
        "prefix_sha256", "data_version", "road_condition_version", "travel_mode",
        "weather_input_sha256", "depends_on",
    ))


def record_candidate_snapshot(pending: dict, result: dict) -> None:
    """Keep candidate input versions even if the older prefix lacks a proof."""
    record = pending.get("result_dependencies")
    if not isinstance(record, dict):
        record = {}
        pending["result_dependencies"] = record
    record["candidate_snapshot"] = {
        "data_version": result.get("data_version"),
        "road_condition_version": result.get("road_condition_version"),
    }


def candidate_snapshot_is_current(record: dict | None) -> bool:
    from agents import route_state, tools

    snapshot = record.get("candidate_snapshot") if isinstance(record, dict) else None
    if not isinstance(snapshot, dict):
        return False
    try:
        data = route_state.current_data_version()
        events = route_state.current_road_condition_version(tools.list_conditions(strict=True))
    except Exception:
        return False
    return (isinstance(data, str) and data not in {"", "unavailable"}
            and snapshot.get("data_version") == data
            and snapshot.get("road_condition_version") == events)
