"""Event-only projection of real local Agent runs for development scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

from experiments.event_replay_guard import REPLAY_ROOT
from experiments.scenario_live_adapter import (
    TERMINAL, _data, project_live_run, require_local_service,
)


_EVENT_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def project_event_run(run: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any]:
    """Add only verified road-condition tool IDs to the generic projection."""
    projected = project_live_run(run, lookup)
    events = run.get("events") if isinstance(run.get("events"), list) else []
    trace_enabled = any(isinstance(event, dict)
                        and event.get("event_type") == "agent_tool_trace_started"
                        for event in events)
    observed: list[list[str]] = []
    if trace_enabled and isinstance(projected.get("tool_trace"), list):
        for event in events:
            if not isinstance(event, dict) or event.get("event_type") != "agent_tool_call":
                continue
            payload = event.get("payload")
            if (not isinstance(payload, dict)
                    or payload.get("tool_name") != "list_road_conditions"
                    or payload.get("outcome") != "success"):
                continue
            ids = payload.get("road_event_ids")
            if (not isinstance(ids, list) or len(ids) > 128
                    or any(not isinstance(value, str) or not _EVENT_ID.fullmatch(value)
                           for value in ids)
                    or len(set(ids)) != len(ids)):
                continue
            observed.append(sorted(ids))
    if observed and all(ids == observed[0] for ids in observed):
        projected["road_event_ids"] = observed[0]
        projected["outputs"].append("road_events")
    if projected["kind"] == "no_route" and projected["status"] == "completed":
        projected["status"] = "infeasible"
    return projected


def prepare_event_fixture(case: dict[str, Any], snapshot: dict[str, Any],
                          run_id: str, *, root: Path = REPLAY_ROOT) -> dict[str, Any]:
    """Create a never-overwritten product-shaped fixture for one frozen case."""
    case_id = case.get("id")
    if (not isinstance(run_id, str) or not _RUN_NAME.fullmatch(run_id)
            or not isinstance(case_id, str) or not _RUN_NAME.fullmatch(case_id)
            or snapshot.get("id") != (case.get("gold_derivation") or {})
            .get("parameters", {}).get("event_snapshot_id")
            or snapshot.get("scope") != "synthetic_development_only"
            or snapshot.get("campus_truth") is not False
            or type(snapshot.get("as_of_epoch")) is not int
            or not isinstance(snapshot.get("conditions"), list)
            or not snapshot["conditions"]):
        raise ValueError("invalid synthetic event case or snapshot")
    rows = []
    for item in snapshot["conditions"]:
        if not isinstance(item, dict) or item.get("action") != "closure":
            raise ValueError("event replay fixture requires closure records")
        row = {key: item[key] for key in (
            "id", "action", "affected_modes", "edge", "start_time", "end_time"
        )}
        row.update(type="closure", name="模拟封闭（仅研究隔离环境）",
                   source={"kind": "research_simulated",
                           "scope": "synthetic_development_only",
                           "campus_truth": False})
        from spatial.road_conditions import _validate_condition_record

        _validate_condition_record(row)
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate simulated event ID")
    root = Path(root).resolve()
    target_dir = root / run_id / case_id
    if target_dir.exists():
        raise FileExistsError(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / "road_conditions.json"
    content = (json.dumps(rows, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    with target.open("xb") as stream:
        stream.write(content)
    return {"path": str(target.resolve()),
            "as_of_epoch": snapshot["as_of_epoch"],
            "fixture_sha256": hashlib.sha256(content).hexdigest()}


def require_isolated_event_service(session, base_url: str,
                                   fixture: dict[str, Any]) -> None:
    """Refuse conversation writes unless both local stores and time match."""
    base = require_local_service(base_url)
    try:
        response = session.get(f"{base}/api/research/event-replay-readiness", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
    except Exception as exc:
        raise ValueError("event service isolation proof is unavailable") from exc
    if (not isinstance(payload, dict)
            or payload.get("status") != "ready"
            or payload.get("service") != "local_development"
            or payload.get("database_isolated") is not True
            or payload.get("road_conditions_isolated") is not True
            or payload.get("as_of_epoch") != fixture.get("as_of_epoch")
            or payload.get("fixture_sha256") != fixture.get("fixture_sha256")):
        raise ValueError("event service isolation proof does not match fixture")


def collect_event_case(case: dict[str, Any], session, base_url: str,
                       lookup: dict[str, Any], fixture: dict[str, Any], *,
                       poll_interval_s: float = 0.5, timeout_s: float = 90,
                       ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect one real run after checking the isolated fixture and DB."""
    base = require_local_service(base_url)
    if len(case.get("messages") or []) != 1:
        raise ValueError("event pilot collector expects exactly one user turn")
    require_isolated_event_service(session, base, fixture)
    started = time.perf_counter()
    deadline = started + timeout_s
    conversation = _data(session.post(
        f"{base}/api/conversations", json={}, timeout=20,
    ))["conversation_id"]
    accepted = _data(session.post(
        f"{base}/api/conversations/{conversation}/messages",
        json={"query": case["messages"][0]["content"],
              "request_id": str(uuid4())}, timeout=30,
    ))
    timed_out = False
    while True:
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        run = _data(session.get(
            f"{base}/api/runs/{accepted['run_id']}",
            params={"conversation_id": conversation}, timeout=25,
        ))
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        if run.get("status") in TERMINAL:
            break
        if poll_interval_s:
            time.sleep(poll_interval_s)
    metadata = {"run_id": accepted["run_id"],
                "conversation_id": conversation,
                "fixture_sha256": fixture["fixture_sha256"],
                "as_of_epoch": fixture["as_of_epoch"]}
    if timed_out:
        return ({"scenario_id": case["id"], "attempt": 1,
                 "snapshot_id": case["snapshot_id"],
                 "input_messages": case["messages"], "turns": [],
                 "latency_s": round(time.perf_counter() - started, 3),
                 "transport_error": "timeout"}, metadata)
    require_isolated_event_service(session, base, fixture)
    turn = project_event_run(run, lookup)
    row = {"scenario_id": case["id"], "attempt": 1,
           "snapshot_id": case["snapshot_id"],
           "input_messages": case["messages"], "turns": [turn],
           "latency_s": round(time.perf_counter() - started, 3),
           "transport_error": None}
    if isinstance(turn.get("tool_trace"), list):
        row["tool_calls"] = len(turn["tool_trace"])
    return row, metadata
