"""Collect development observations from a local full conversation service.

The adapter only projects stable task facts. It does not score its own output,
claim field truth, or synthesize the independent review required by E1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import requests

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "partial", "failed", "agent_failed", "needs_input", "cancelled"}


def build_poi_lookup(pois: list[dict[str, Any]]) -> dict[str, Any]:
    """Index exact, unique names and aliases from the same POI snapshot."""
    by_name: dict[str, str] = {}
    ambiguous: set[str] = set()
    ids = set()
    for poi in pois:
        poi_id = poi.get("poi_id") or poi.get("id")
        if not isinstance(poi_id, str) or not poi_id:
            continue
        ids.add(poi_id)
        for raw in [poi.get("name"), *(poi.get("aliases") or [])]:
            if not isinstance(raw, str) or not (name := raw.strip()):
                continue
            if name in by_name and by_name[name] != poi_id:
                ambiguous.add(name)
            else:
                by_name[name] = poi_id
    for name in ambiguous:
        by_name.pop(name, None)
    return {"names": by_name, "ids": ids}


def _poi_id(ref: Any, lookup: dict[str, Any]) -> str | None:
    if not isinstance(ref, dict):
        return None
    raw_id = ref.get("poi_id") or ref.get("id")
    if isinstance(raw_id, str) and raw_id in lookup["ids"]:
        return raw_id
    name = ref.get("name")
    return lookup["names"].get(name.strip()) if isinstance(name, str) else None


def _via_refs(ref: Any) -> list[dict[str, Any]]:
    if not isinstance(ref, dict):
        return []
    raw = ref.get("points") if ref.get("type") == "multi" else [ref]
    return raw if isinstance(raw, list) else []


def _project_tool_trace(run: dict[str, Any]) -> list[dict[str, Any]] | None:
    events = run.get("events") if isinstance(run.get("events"), list) else []
    enabled = any(event.get("event_type") == "agent_tool_trace_started"
                  for event in events if isinstance(event, dict))
    if not enabled:
        return None
    trace = []
    allowed_outcomes = {
        "success", "tool_error", "invalid_args", "exception",
        "skipped_after_route_verification_retry",
        "skipped_after_route_verification_failure",
    }
    allowed_artifacts = {"route", "candidates", "clarify", "suggestions", "route_comparison", None}
    for event in events:
        if not isinstance(event, dict) or event.get("event_type") != "agent_tool_call":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        name = payload.get("tool_name")
        keys = payload.get("argument_keys")
        outcome = payload.get("outcome")
        artifact = payload.get("artifact_kind")
        if (not isinstance(name, str) or not name
                or not isinstance(keys, list)
                or any(not isinstance(key, str) for key in keys)
                or outcome not in allowed_outcomes or artifact not in allowed_artifacts):
            return None
        trace.append({
            "tool_name": name,
            "argument_keys": sorted(set(keys)),
            "outcome": outcome,
            "artifact_kind": artifact,
        })
    return trace


def project_live_run(run: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any]:
    """Convert one actual API run into the strict replay observation schema."""
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    state = result.get("route_state") if isinstance(result.get("route_state"), dict) else {}
    request = result.get("place_request") if isinstance(result.get("place_request"), dict) else {}
    kind = result.get("response_kind") or result.get("task_type")
    kind = {"path_planning": "route", "unknown": "clarify", "chat": "information"}.get(kind, kind)
    if kind not in {"route", "candidates", "clarify", "information", "no_route"}:
        kind = "information"
    status = run.get("status")
    if status == "needs_input":
        status = "needs_input"
    elif status in {"partial", "failed", "agent_failed", "cancelled"}:
        status = "failed"
    elif status != "completed":
        status = "failed"
    start = state.get("start") or result.get("start") or request.get("start")
    end = state.get("end") or result.get("end")
    vias = _via_refs(state.get("via") or result.get("via"))
    candidates = result.get("candidates") or []
    strategy = state.get("strategy") or result.get("strategy") or {}
    if isinstance(strategy, dict):
        strategy = strategy.get("name")
    constraints = state.get("hard_constraints") or request.get("constraints") or {}
    edges = result.get("recommended_edge_ids")
    unknowns = []
    refs = [ref for ref in (start, end, *vias) if isinstance(ref, dict)]
    if any(_poi_id(ref, lookup) is None and not ref.get("coordinates") for ref in refs):
        unknowns.append("poi_identity_unresolved")
    if kind == "route" and not isinstance(edges, list):
        unknowns.append("route_edge_sequence_missing")
    outputs = []
    if kind == "route":
        outputs.append("route")
    weather = result.get("weather")
    user_reply = result.get("explanation") or result.get("message") or result.get("reply")
    if (isinstance(weather, dict) and isinstance(weather.get("weather"), str)
            and weather["weather"] and isinstance(user_reply, str)
            and weather["weather"] in user_reply):
        outputs.append("weather")
    if kind == "candidates" and isinstance(candidates, list) and candidates:
        outputs.append("candidates")
    projected = {
        "status": status, "kind": kind,
        "start_poi_id": _poi_id(start, lookup),
        "end_poi_id": _poi_id(end, lookup),
        "via_poi_ids": [_poi_id(ref, lookup) for ref in vias],
        "travel_mode": state.get("travel_mode") or result.get("mode") or request.get("mode"),
        "strategy": strategy, "hard_constraints": constraints,
        "candidate_poi_ids": [_poi_id(ref, lookup) for ref in candidates]
                             if isinstance(candidates, list) else [],
        "clarification_slot": request.get("pending_slot"),
        "outputs": outputs, "route_edge_ids": edges if isinstance(edges, list) else [],
        "unknowns": unknowns,
    }
    tool_trace = _project_tool_trace(run)
    if tool_trace is not None:
        projected["tool_trace"] = tool_trace
    return projected


def require_local_service(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise ValueError("live development replay requires a local HTTP service")
    return url.rstrip("/")


def _data(response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ValueError("service returned no data object")
    return data


def require_isolated_replay_service(session, base_url: str) -> None:
    """Refuse write-capable development replay unless the local app proves DB isolation."""
    base = require_local_service(base_url)
    try:
        response = session.get(f"{base}/api/research/replay-readiness", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "local service must prove it uses an isolated development database"
        ) from exc
    if (not isinstance(payload, dict) or payload.get("status") != "ready"
            or payload.get("service") != "local_development"
            or payload.get("database_isolated") is not True):
        raise ValueError(
            "local service must prove it uses an isolated development database"
        )


def run_case(case: dict[str, Any], session, base_url: str,
             lookup: dict[str, Any], *, poll_interval_s: float = 0.5,
             timeout_s: float = 90, _service_verified: bool = False
             ) -> tuple[list[dict[str, Any]], float]:
    """Run all user turns in one fresh conversation, preserving continuation."""
    base = require_local_service(base_url)
    if not _service_verified:
        require_isolated_replay_service(session, base)
    started = time.perf_counter()
    conversation = _data(session.post(f"{base}/api/conversations", json={}, timeout=20))["conversation_id"]
    continuation = None
    turns = []
    for message in case["messages"]:
        body = {"query": message["content"], "request_id": str(uuid4())}
        if continuation:
            body.update(continuation_task_id=continuation["task_id"],
                        base_revision=continuation["current_task_revision"])
        accepted = _data(session.post(
            f"{base}/api/conversations/{conversation}/messages", json=body, timeout=30,
        ))
        deadline = time.perf_counter() + timeout_s
        while True:
            run = _data(session.get(
                f"{base}/api/runs/{accepted['run_id']}",
                params={"conversation_id": conversation}, timeout=25,
            ))
            if run.get("status") in TERMINAL:
                break
            if time.perf_counter() >= deadline:
                raise TimeoutError("local run exceeded replay deadline")
            if poll_interval_s:
                time.sleep(poll_interval_s)
        turns.append(project_live_run(run, lookup))
        continuation = None
        if run["status"] == "needs_input":
            if not isinstance(run.get("task_id"), str) or type(run.get("current_task_revision")) is not int:
                raise ValueError("pending run lacks trusted task revision")
            continuation = run
    return turns, time.perf_counter() - started


def collect_development_replay(dataset: dict[str, Any], base_url: str,
                               lookup: dict[str, Any], session_factory=requests.Session
                               ) -> list[dict[str, Any]]:
    """Collect attempts; failures stay explicit and never become invented turns."""
    from experiments.scenario_evaluator import validate_dataset
    from experiments.scenario_v2 import VERSION as V2_VERSION
    from experiments.scenario_v2 import validate_dataset as validate_dataset_v2

    require_local_service(base_url)
    (validate_dataset_v2 if dataset.get("protocol_version") == V2_VERSION else validate_dataset)(dataset)
    if dataset["split"] != "development":
        raise ValueError("live adapter only collects development scenarios")
    rows = []
    verified_service = False
    for case in dataset["cases"]:
        for attempt in range(1, dataset.get("repeats", 1) + 1):
            with session_factory() as session:
                started = time.perf_counter()
                if not verified_service:
                    require_isolated_replay_service(session, base_url)
                    verified_service = True
                try:
                    turns, latency = run_case(
                        case, session, base_url, lookup, _service_verified=verified_service,
                    )
                    error = None
                except (requests.RequestException, TimeoutError, ValueError, KeyError) as exc:
                    turns, latency = [], time.perf_counter() - started
                    error = ("timeout" if isinstance(exc, TimeoutError) else
                             "network_error" if isinstance(exc, requests.RequestException) else
                             "adapter_error")
                row = {"scenario_id": case["id"], "attempt": attempt,
                       "snapshot_id": case["snapshot_id"],
                       "input_messages": case["messages"], "turns": turns,
                       "latency_s": round(latency, 3), "transport_error": error}
                if turns and all(isinstance(turn.get("tool_trace"), list) for turn in turns):
                    row["tool_calls"] = sum(len(turn["tool_trace"]) for turn in turns)
                rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--service", required=True, help="local isolated development service URL")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file; never overwritten")
    args = parser.parse_args(argv)
    require_local_service(args.service)
    if args.output.exists():
        raise FileExistsError(f"replay output already exists: {args.output}")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    pois = json.loads((ROOT / "data" / "pois.json").read_text(encoding="utf-8"))["pois"]
    rows = collect_development_replay(dataset, args.service, build_poi_lookup(pois))
    with args.output.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "attempts": len(rows),
                      "service_database_identity": "verified_isolated_local_test_database"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
