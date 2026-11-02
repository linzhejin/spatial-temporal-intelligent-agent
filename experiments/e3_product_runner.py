"""Execute the frozen E3 subset through real loopback HTTP and PostgreSQL.

No model calls, business database, production event writes or oracle substitution.
New output directories are mandatory. Initial route creation is common setup;
timed requests include the full update/verification/proof/serialization path.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from experiments.e3_product_cases import ROOT, build_campus_cases, effective_snapshot
from experiments.e3_independent_audit import audit_route_response
from experiments.e3_measurement import capture_execution_metrics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def input_hashes():
    paths = []
    for folder in ("agents", "api", "spatial", "storage", "experiments"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend((ROOT / "storage/migrations").glob("*.sql"))
    paths.extend((ROOT / "data").glob("*.json"))
    paths.extend([ROOT / "data/whu_road_network.graphml", ROOT / "config.py",
                  ROOT / "requirements.txt", ROOT / "experiments/E3_PRODUCT_COMPARISON.md"])
    # Runtime user logs are never experimental inputs or archived provenance.
    excluded = {"telemetry.json", "user_profiles.json", "experience_cards.json"}
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file() and p.name not in excluded}


@contextmanager
def product_server(graph, event_path, database_url):
    from flask import Flask
    from psycopg.conninfo import conninfo_to_dict
    from werkzeug.serving import WSGIRequestHandler, make_server
    from agents import tools
    from api import routes
    from spatial import network, road_conditions
    from storage import database, task_repository
    from tests.db_safety import verify_isolated_test_database

    config = conninfo_to_dict(database_url)
    if config.get("host") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("E3 requires a loopback-only dedicated test database")
    verify_isolated_test_database(database_url)
    database.initialize(database_url)
    with database.connect(database_url) as conn:
        server_version = conn.execute("SHOW server_version").fetchone()["server_version"]
    owner, token, conversation = task_repository.create_conversation(database_url)
    app = Flask(__name__)
    app.config.update(TESTING=True, DATABASE_URL=database_url)
    app.register_blueprint(routes.api_bp)
    holder = {"graph": graph, "clock": 0.0}

    class QuietHandler(WSGIRequestHandler):
        def log_request(self, *args, **kwargs):
            pass

    with ExitStack() as stack:
        for target, name, value in (
            (routes, "_ensure_network", lambda: (holder["graph"], None)),
            (tools, "_ensure_graph", lambda: holder["graph"]),
            (network, "get_network", lambda: holder["graph"]),
            (routes, "_weather_snapshot", lambda: None),
            (tools, "_weather_snapshot", lambda: None),
            (road_conditions, "time", SimpleNamespace(time=lambda: holder["clock"])),
            (road_conditions, "_CONDITIONS_FILE", str(event_path)),
            (road_conditions, "_cache", None),
            (road_conditions, "_cache_mtime", None),
            (road_conditions, "_cache_error", None),
        ):
            stack.enter_context(patch.object(target, name, value))
        # A preexisting replay environment must not override this frozen clock.
        from experiments.event_replay_guard import certified_event_snapshot
        if certified_event_snapshot(str(event_path)) is not None:
            raise ValueError("another replay fixture is active")
        server = make_server("127.0.0.1", 0, app, request_handler=QuietHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def post(path, body):
            req = Request(f"http://127.0.0.1:{server.server_port}/api{path}",
                          data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Cookie": f"whu_conversation_token={token}"})
            try:
                with urlopen(req, timeout=90) as response:
                    return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)

        def set_snapshot(current_graph, events, clock):
            holder.update(graph=current_graph, clock=clock)
            road_conditions._save_conditions([
                {k: v for k, v in event.items() if k != "bindings"} for event in events])

        try:
            yield post, set_snapshot, {"database": config["dbname"], "owner_id": owner,
                                       "conversation_id": conversation, "port": server.server_port,
                                       "server_version": server_version}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)


def expected_task(graph, case):
    from agents.route_state import current_data_version, current_road_condition_version
    from spatial.network import get_nearest_node
    from spatial.routing_index import get_routing_index

    body = case["updated_body"]
    mode, constraints = body["travel_mode"], body["constraints"]
    mode_graph = get_routing_index(graph).for_mode(mode).graph
    endpoints = []
    for field in ("start", "end"):
        c = body[field]["coordinates"]
        endpoints.append(get_nearest_node(mode_graph, c["lng"], c["lat"]))
    effective, events = effective_snapshot(graph, case["new_events"], case["updated_clock"],
                                            mode, constraints)
    # Environment version includes all active events, even those for other modes.
    active = [{k: v for k, v in event.items() if k != "bindings"}
              for event in case["new_events"] if not event.get("revoked_at")
              and (not event.get("start_time") or event["start_time"] <= case["updated_clock"])
              and (not event.get("end_time") or case["updated_clock"] < event["end_time"])]
    expected = {"start_node": endpoints[0], "end_node": endpoints[1], "mode": mode,
                "strategy": body["strategy"], "constraints": constraints, "events": events,
                "data_version": current_data_version(graph),
                "road_condition_version": current_road_condition_version(active),
                "require_event_overview": case["entry"] != "new_task" and not case["graph_edit"]}
    return effective, expected


def response_semantics(status, body):
    """Ignore random proof/route IDs and timings when matching count probes."""
    data = body.get("data", body)
    return {"status": status, **{key: data.get(key) for key in (
        "error", "recommended_edge_ids", "recommended_length_m", "duration_min",
        "mode", "route_events", "road_condition_overview", "road_conditions_applied")}}


def audit_execution_policy(case, strategy, data, status, metrics):
    reusable = {"unchanged", "off_path_closure", "future_event", "expired_event",
                "other_mode", "notice", "recovery"}
    expected_reuse = strategy == "conservative_reuse" and case["variant"] in reusable
    actual_reuse = bool(data.get("execution_optimization", {}).get("path_reused"))
    searches = metrics["phases"]["dijkstra"]["calls"]
    issues = []
    if expected_reuse != actual_reuse:
        issues.append("unexpected_reuse_decision")
    if actual_reuse and searches != 0:
        issues.append("reused_route_still_searched")
    if not actual_reuse and status == 200 and searches < 1:
        issues.append("recomputed_route_has_no_observed_search")
    return {"status": "fail" if issues else "pass", "issues": issues,
            "expected_reuse": expected_reuse, "actual_reuse": actual_reuse,
            "observed_searches": searches}


def summarize_records(protocol, records, *, input_hashes_unchanged, terminal_error=None):
    planned = {(c["id"], strategy, attempt)
               for c in protocol["cases"] for strategy in protocol["strategies"]
               for attempt in range(1, protocol["repeats"] + protocol["operation_probe_repeats"] + 1)}
    actual = {(r["case_id"], r["strategy"], r["attempt"]) for r in records}
    semantic_mismatches = []
    for case in protocol["cases"]:
        for strategy in protocol["strategies"]:
            group = [r for r in records if r["case_id"] == case["id"] and r["strategy"] == strategy]
            values = [response_semantics(r["http_status"], r["response"])
                      for r in group if "response" in r]
            if values and any(v != values[0] for v in values[1:]):
                semantic_mismatches.append({"case_id": case["id"], "strategy": strategy})
    counts = {}
    for group_name, probe in (("timed", False), ("operation_probe", True)):
        rows = [r for r in records if r["operation_probe"] == probe]
        counts[group_name] = {
            strategy: {status: sum(r["strategy"] == strategy and r["audit"]["status"] == status for r in rows)
                       for status in ("pass", "fail", "unknown")}
            for strategy in protocol["strategies"]}
    missing = [{"case_id": c, "strategy": s, "attempt": a} for c, s, a in sorted(planned - actual)]
    policy_failures = [{"case_id": r["case_id"], "strategy": r["strategy"], "attempt": r["attempt"]}
                       for r in records if r.get("execution_policy_audit", {}).get("status") != "pass"]
    return {"planned_timed_executions": protocol["planned_timed_executions"],
            "actual_timed_executions": sum(not r["operation_probe"] for r in records),
            "actual_operation_probes": sum(r["operation_probe"] for r in records),
            "audit_counts": counts, "missing_executions": missing,
            "duplicate_execution_keys": len(records) - len(actual),
            "execution_policy_failures": policy_failures,
            "semantic_mismatches_across_repeats_and_probes": semantic_mismatches,
            "input_hashes_unchanged": input_hashes_unchanged,
            "terminal_error": terminal_error,
            "acceptance_passed": not (missing or semantic_mismatches or terminal_error or policy_failures)
            and input_hashes_unchanged and len(records) == len(actual)
            and all(r["audit"]["status"] == "pass" for r in records),
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def run(output, database_url, *, repeats=3, case_ids=None, probes=True):
    from spatial import network

    if output.exists():
        raise FileExistsError(f"preserve existing batch: {output}")
    output.mkdir(parents=True)
    graph = network.load_or_download_network()
    cases = build_campus_cases(graph)
    if case_ids:
        wanted = set(case_ids)
        if wanted - {c["id"] for c in cases}:
            raise ValueError("unknown requested case")
        cases = [c for c in cases if c["id"] in wanted]
    protocol = {
        "schema": "e3-product-fixed-subset-v1", "scope": "seen_development_subset",
        "formal_unseen_test": False, "strategies": ["full_recompute", "conservative_reuse"],
        "repeats": repeats, "operation_probe_repeats": int(probes), "seed": 20261102,
        "cases": cases, "planned_timed_executions": len(cases) * repeats * 2,
        "planned_operation_probes": len(cases) * 2 if probes else 0,
        "timing": "warm process, real loopback HTTP including response serialization; initial route setup excluded",
        "binding_scope": "road bindings and static mode filters shared/frozen; active event application and output scoring independent",
        "model_calls": 0, "weather": None,
    }
    write_json(output / "protocol.json", protocol)
    frozen_hashes = input_hashes()
    write_json(output / "source_manifest.json", {
        "source_hashes": frozen_hashes, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "networkx", "Flask", "Werkzeug", "Shapely", "psycopg", "numpy")},
        "platform": platform.platform(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
    })
    rng, records = random.Random(protocol["seed"]), []
    terminal_error = None
    try:
        with product_server(graph, output / "isolated_events.json", database_url) as (post, set_snapshot, identity):
            write_json(output / "isolation.json", identity)
            with (output / "executions.jsonl").open("x", encoding="utf-8") as stream:
                for case_index, case in enumerate(cases):
                    changed_graph = graph
                    if case["graph_edit"]:
                        changed_graph = graph.copy()
                        u, v, k = case["graph_edit"]["edge"]
                        changed_graph[u][v][k]["length"] = case["graph_edit"]["length"]
                    effective, expectation = expected_task(changed_graph, case)
                    initial_events = [] if case["entry"] == "recovery" else case["old_events"]
                    initial_case = {**case, "updated_body": case["initial_body"],
                                    "new_events": initial_events, "updated_clock": case["initial_clock"],
                                    "entry": "new_task", "graph_edit": None}
                    initial_effective, initial_expectation = expected_task(graph, initial_case)
                    write_json(output / f"expected_{case['id']}.json", expectation)
                    for attempt in range(1, repeats + int(probes) + 1):
                        probe = attempt > repeats
                        set_snapshot(graph, initial_events, case["initial_clock"])
                        initial_status, initial = post("/route", case["initial_body"])
                        initial_audit = audit_route_response(
                            graph, initial_effective, initial_expectation,
                            initial.get("data", initial), initial_status)
                        write_json(output / f"initial_{case['id']}_{attempt}.json", {
                            "status": initial_status, "response": initial, "audit": initial_audit})
                        if initial_status != 200 or "reuse_certificate" not in initial["data"]["route_state"]:
                            write_json(output / f"setup_failure_{case['id']}_{attempt}.json", initial)
                            raise RuntimeError(f"initial proof was not issued: {case['id']}, status={initial_status}")
                        if initial_audit["status"] != "pass":
                            raise RuntimeError(f"initial independent audit failed: {case['id']}")
                        prior = initial["data"]["route_state"]
                        if case["entry"] == "recovery":
                            set_snapshot(graph, case["old_events"], case["initial_clock"])
                            blocked_status, blocked = post("/route/replan", {
                                "route_state": prior, "change": {"refresh_road_conditions": True}})
                            write_json(output / f"recovery_before_{case['id']}_{attempt}.json", blocked)
                            if blocked_status != 404 or blocked.get("error") != "route_not_found":
                                raise RuntimeError("unreachable phase did not produce the expected refusal")
                        order = list(protocol["strategies"])
                        rng.shuffle(order)
                        for strategy in order:
                            set_snapshot(changed_graph, case["new_events"], case["updated_clock"])
                            state = deepcopy(prior)
                            if strategy == "full_recompute":
                                state.pop("reuse_certificate", None)
                            request_body = {"route_state": state, "change": case["change"]}
                            requests = []
                            with capture_execution_metrics(expansion_probe=probe) as metrics:
                                started = time.perf_counter()
                                if case["entry"] == "new_task":
                                    status, body = post("/route", case["updated_body"])
                                    requests.append({"path": "/route", "status": status, "body": body})
                                else:
                                    status, body = post("/route/replan", request_body)
                                    requests.append({"path": "/route/replan", "status": status, "body": body})
                                    if case["graph_edit"]:
                                        if status != 409 or body.get("error") != "route_state_version_conflict":
                                            raise RuntimeError("base graph change must reject old route state")
                                        status, body = post("/route", case["updated_body"])
                                        requests.append({"path": "/route", "status": status, "body": body})
                                elapsed_ms = (time.perf_counter() - started) * 1000
                            data = body.get("data", body)
                            audit = audit_route_response(changed_graph, effective, expectation, data, status)
                            record = {"case_id": case["id"], "family": case["family"],
                                      "variant": case["variant"], "attempt": attempt,
                                      "strategy": strategy, "operation_probe": probe,
                                      "http_status": status, "elapsed_ms": elapsed_ms,
                                      "requests": requests, "response": body, "audit": audit,
                                      "metrics": metrics, "initial_proof_issued": True,
                                      "execution_policy_audit": audit_execution_policy(
                                          case, strategy, data, status, metrics)}
                            records.append(record)
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                            stream.flush()
                    print(json.dumps({"case": case["id"], "completed_cases": case_index + 1,
                                      "total_cases": len(cases), "executions": len(records)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        terminal_error = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output / "terminal_error.json", terminal_error)
    unchanged = frozen_hashes == input_hashes()
    summary = summarize_records(protocol, records, input_hashes_unchanged=unchanged,
                                terminal_error=terminal_error)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--no-probes", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        parser.error("explicit TEST_DATABASE_URL required")
    summary = run(args.output, database_url, repeats=args.repeats,
                  case_ids=args.case_ids, probes=not args.no_probes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(not summary["acceptance_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
