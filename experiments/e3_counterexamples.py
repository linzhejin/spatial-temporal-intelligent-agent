"""Product-backed, controlled E3 counterexamples with hand-checkable truth.

Only full recomputation and conservative reuse execute as product strategies.
The weak path-only listener is an offline diagnostic of a real prior response,
not an unsafe production override. Its elapsed time is never compared with API
latency. All graphs, slopes and road events here are controlled inputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
import traceback
from unittest.mock import patch

import networkx as nx

from experiments.e3_independent_audit import audit_route_response
from experiments.e3_measurement import capture_execution_metrics
from experiments.e3_product_cases import CLOCK, ROOT, coordinate_ref, path_listener_decision
from experiments.e3_product_runner import input_hashes, product_server, write_json


STRATEGIES = ("full_recompute", "conservative_reuse")


def build_counterexamples():
    """Declare every expected outcome before either product policy runs."""
    common = {
        "initial_strategy": "shortest", "updated_strategy": "shortest",
        "initial_closure": False, "graph_cost_change": False,
        "expected_initial_distance_m": 200.0,
        "expected_initial_eta_min": 5.3,
        "expected_updated_objective_cost": 200.0,
        "expected_updated_distance_m": 200.0,
        "expected_updated_eta_min": 5.3,
    }
    return [
        {**common, "id": "reopen_shortcut", "initial_closure": True,
         "expected_initial_distance_m": 220.0, "expected_initial_eta_min": 2.9},
        {**common, "id": "off_path_cost_decrease", "graph_cost_change": True,
         "expected_updated_objective_cost": 160.0, "expected_updated_distance_m": 160.0,
         "expected_updated_eta_min": 2.1},
        {**common, "id": "flat_weight_change", "updated_strategy": "flat",
         "expected_updated_objective_cost": 1.24, "expected_updated_distance_m": 220.0,
         "expected_updated_eta_min": 2.9},
    ]


def _graph():
    graph = nx.MultiDiGraph()
    graph.graph.update(
        crs="epsg:4326",
        routing_boundary_wkt="POLYGON ((114.35 30.52, 114.37 30.52, 114.37 30.54, 114.35 30.54, 114.35 30.52))",
    )
    for node, x, y in ((1, 114.360, 30.530), (2, 114.361, 30.531),
                       (3, 114.361, 30.529), (4, 114.362, 30.530)):
        graph.add_node(node, x=x, y=y)
    for u, v, length, slope in ((1, 2, 100.0, 5), (2, 4, 100.0, 5),
                                (1, 3, 110.0, 1), (3, 4, 110.0, 1)):
        graph.add_edge(u, v, key=0, length=length, highway="footway",
                       slope_level=slope, scenery_level=3, name="E3受控道路")
    return graph


def case_graphs(case):
    old = _graph()
    new = old.copy()
    if case["graph_cost_change"]:
        new[1][3][0]["length"] = 80.0
        new[3][4][0]["length"] = 80.0
    return old, new


def _closure():
    return {"id": "synthetic-upper-closure", "type": "event",
            "name": "上支受控封闭", "description": "E3受控小图封闭事件",
            "action": "closure", "affected_modes": ["walk"],
            "start_time": CLOCK - 100, "end_time": CLOCK + 100,
            "edge": {"edges": [[1, 2, 0]]}}


def case_truth(case, graph, *, initial=False):
    """Build truth directly from the declared four edges and arithmetic."""
    from agents.route_state import current_data_version, current_road_condition_version

    events = [_closure()] if initial and case["initial_closure"] else []
    effective = graph.copy()
    if events:
        effective.remove_edge(1, 2, 0)
    strategy = case["initial_strategy"] if initial else case["updated_strategy"]
    if strategy == "flat":
        # Full attribute coverage, no weather, event or road-name penalties.
        # The 220 m optimum is below the 250 m detour cap, so interpolation is
        # not triggered. This computes the declared objective independently.
        max_length = max(a["length"] for _, _, a in graph.edges(data=True))
        for _u, _v, _key, attributes in effective.edges(keys=True, data=True):
            attributes["audit_objective_cost"] = (
                0.5 * attributes["length"] / max_length
                + 0.4 * attributes["slope_level"] / 5
                + 0.1 * (1 - attributes["scenery_level"] / 5))
    expected = {
        "start_node": 1, "end_node": 4, "mode": "walk", "strategy": strategy,
        "constraints": {}, "data_version": current_data_version(graph),
        "road_condition_version": current_road_condition_version(events),
        "events": [{**event, "edge_ids": [[1, 2, 0]]} for event in events],
        "require_event_overview": not initial and not case["graph_cost_change"],
    }
    return effective, expected


def listener_diagnostic(old_effective, new_effective, task, old_edges, optimum):
    """Measure the counterfactual retained path, without faking an API result."""
    edges = [tuple(edge) for edge in old_edges]
    reuse = path_listener_decision(old_effective, new_effective, edges)
    feasible = bool(edges) and edges[0][0] == task["start_node"]
    previous = task["start_node"]
    for edge in edges:
        feasible = feasible and edge[0] == previous and new_effective.has_edge(*edge)
        previous = edge[1]
    feasible = feasible and previous == task["end_node"]
    field = "length" if task["strategy"] == "shortest" else "audit_objective_cost"
    cost = sum(new_effective[u][v][k][field] for u, v, k in edges) if feasible else None
    gap = cost - optimum if cost is not None else None
    return {
        "strategy": "path_listener_diagnostic", "source": "actual_initial_response_edges",
        "would_reuse": reuse, "old_path_still_feasible": feasible,
        "retained_edge_ids": [list(e) for e in edges],
        "objective": task["strategy"], "retained_objective_cost": cost,
        "independent_optimum": optimum, "objective_gap": gap,
        "false_optimality_reuse": bool(reuse and feasible and gap > 1e-8),
        "latency_comparison": "not_applicable_offline_diagnostic",
    }


def semantic_result(status, body):
    data = body.get("data", body)
    state = data.get("route_state") or {}
    return deepcopy({
        "http_status": status,
        "result": {k: data.get(k) for k in (
            "error", "recommended_edge_ids", "recommended_length_m", "duration_min",
            "mode", "strategy", "route_events", "road_condition_overview",
            "road_conditions_applied", "road_condition_version")},
        "task": {k: state.get(k) for k in (
            "start", "end", "travel_mode", "hard_constraints", "strategy",
            "data_version", "road_condition_version")},
    })


def runtime_versions(database_url):
    """Called after product_server has verified the dedicated test database."""
    from storage import database
    with database.connect(database_url) as conn:
        version = conn.execute("SHOW server_version").fetchone()["server_version"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "postgresql": version,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("networkx", "Flask", "Werkzeug", "shapely", "psycopg")}}


def _body(graph, strategy="shortest"):
    return {"start": coordinate_ref(graph, 1), "end": coordinate_ref(graph, 4),
            "strategy": strategy, "travel_mode": "walk", "constraints": {}, "query": ""}


def _failure(error, stage):
    # Exception messages may contain connection parameters. Keep a usable
    # location/type record without echoing credentials or arbitrary arguments.
    return {"stage": stage, "type": type(error).__name__,
            "frames": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                       for frame in traceback.extract_tb(error.__traceback__)]}


def _failed_record(case, attempt, strategy, probe, status, failure):
    return {"case_id": case["id"], "attempt": attempt, "strategy": strategy,
            "operation_probe": probe, "execution_status": status,
            "elapsed_ms": None, "http_status": None, "response": None, "requests": [],
            "failure": failure, "metrics": None,
            "audit": {"status": "fail" if status != "not_executed" else "unknown",
                      "issues": [status], "checks": {}, "metrics": {}}}


def _proof_snapshot(database_url, identity, response):
    from storage import database
    data = response["data"]
    proof_id = data["route_state"]["reuse_certificate"]["proof_id"]
    with database.connect(database_url) as conn:
        row = conn.execute(
            "SELECT selected_edges, graph_version, event_version, verified "
            "FROM route_proof WHERE proof_id=%s AND owner_id=%s",
            (proof_id, identity["owner_id"])).fetchone()
    if row is None or row["verified"] is not True or row["selected_edges"] != data["recommended_edge_ids"]:
        raise RuntimeError("actual stored proof does not match the initial response")
    return {"proof_id": proof_id, **dict(row)}


def _verify_binding(graph):
    from spatial.road_conditions import resolve_condition_effects
    _penalties, closed, applied, details = resolve_condition_effects(
        graph, [_closure()], "walk", strict=True, include_event_details=True)
    if applied != 1 or closed != {(1, 2, 0)} or set(details[0][1]) != {(1, 2, 0)}:
        raise RuntimeError("controlled event binding differs from preregistered edge truth")


def _execute(post, case, prior, graph, effective, expected, attempt, strategy, probe):
    state = deepcopy(prior)
    if strategy == "full_recompute":
        state.pop("reuse_certificate", None)
    change = ({"strategy": "flat"} if case["updated_strategy"] == "flat"
              else {"refresh_road_conditions": True})
    record = _failed_record(case, attempt, strategy, probe, "execution_error", {})
    requests, status, body, request_failure = [], None, None, None
    with capture_execution_metrics(expansion_probe=probe) as metrics:
        started = time.perf_counter()
        try:
            status, body = post("/route/replan", {"route_state": state, "change": change})
            requests.append({"path": "/route/replan", "status": status, "response": body})
            if case["graph_cost_change"] and status == 409:
                # This branch is the client recovery procedure. The independent
                # check of both responses happens after timing and instrumentation.
                status, body = post("/route", _body(graph, case["updated_strategy"]))
                requests.append({"path": "/route", "status": status, "response": body})
        except Exception as error:
            request_failure = _failure(error, "update_request")
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
    record.update(elapsed_ms=elapsed_ms, http_status=status, response=body,
                  requests=requests, metrics=metrics, failure=request_failure)
    rejection_audit = None
    if case["graph_cost_change"] and requests:
        rejected = {**expected, "expected_error": "route_state_version_conflict", "expected_status": 409}
        first = requests[0]
        rejection_audit = audit_route_response(
            graph, effective, rejected, first["response"], first["status"])
        first["audit"] = rejection_audit
    if request_failure is None:
        record["execution_status"] = "completed"
        record["audit"] = audit_route_response(graph, effective, expected, body.get("data", body), status)
        if record["audit"]["status"] == "pass":
            actual = record["audit"]["metrics"].get("selected_objective_cost")
            if actual is None or abs(actual - case["expected_updated_objective_cost"]) > 1e-8:
                record["audit"].update(status="fail")
                record["audit"]["issues"].append("declared_hand_computed_optimum_mismatch")
            if record["audit"]["metrics"].get("duration_min") != case["expected_updated_eta_min"]:
                record["audit"].update(status="fail")
                record["audit"]["issues"].append("declared_hand_computed_eta_mismatch")
        if rejection_audit is not None:
            record["audit"]["checks"]["stale_graph_rejection"] = rejection_audit
            if rejection_audit["status"] != "pass":
                record["audit"].update(status="fail")
                record["audit"]["issues"].append("expected_graph_version_rejection_not_returned")
        record["semantics"] = semantic_result(status, body)
        record["semantics"]["transition"] = [
            semantic_result(request["status"], request["response"]) for request in requests]
    return record


def _summary(protocol, records, weak, unchanged, fatal):
    groups = {label: [r for r in records if r["operation_probe"] is probe]
              for label, probe in (("timed", False), ("operation_probe", True))}
    summary = {
        "planned_timed_executions": protocol["planned_timed_executions"],
        "planned_operation_probes": protocol["planned_operation_probes"],
        "actual_timed_executions": sum(r["execution_status"] in {"completed", "execution_error"}
                                       for r in groups["timed"]),
        "actual_operation_probes": sum(r["execution_status"] in {"completed", "execution_error"}
                                       for r in groups["operation_probe"]),
        "accounted_planned_records": len(records), "input_hashes_unchanged": unchanged,
        "fatal_error": fatal, "probe_comparisons": [], "per_case": [],
    }
    for label, group in groups.items():
        summary[f"{label}_audit_counts"] = {status: sum(r["audit"]["status"] == status for r in group)
                                             for status in ("pass", "fail", "unknown")}
    for case in protocol["cases"]:
        for strategy in STRATEGIES:
            timed = [r for r in groups["timed"] if r["case_id"] == case["id"] and r["strategy"] == strategy]
            probes = [r for r in groups["operation_probe"] if r["case_id"] == case["id"] and r["strategy"] == strategy]
            consistent = (len(timed) == protocol["repeats"] and len(probes) == 1
                          and all(r["execution_status"] == "completed" for r in timed + probes)
                          and all(r.get("semantics") == probes[0].get("semantics") for r in timed))
            summary["probe_comparisons"].append({"case_id": case["id"], "strategy": strategy,
                                                  "matches_all_timed_repeats": consistent})
            times = [r["elapsed_ms"] for r in timed if r["execution_status"] == "completed"]
            summary["per_case"].append({"case_id": case["id"], "strategy": strategy,
                                          "planned_timed_runs": protocol["repeats"],
                                          "completed_timed_runs": len(times),
                                          "median_request_ms": statistics.median(times) if times else None})
    false_reuse = {r["case_id"] for r in weak if r["diagnostic"]["false_optimality_reuse"]}
    summary["weak_listener_false_reuse_case_ids"] = sorted(false_reuse)
    summary["completed_successfully"] = (
        not fatal and unchanged
        and len(records) == protocol["planned_timed_executions"] + protocol["planned_operation_probes"]
        and all(r["execution_status"] == "completed" and r["audit"]["status"] == "pass" for r in records)
        and all(row["matches_all_timed_repeats"] for row in summary["probe_comparisons"])
        and false_reuse == {c["id"] for c in protocol["cases"]})
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    return summary


def run_counterexamples(output, database_url, repeats=3):
    """Execute three fixed counterexamples; preserve every planned denominator."""
    from spatial import network

    output = Path(output)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if output.exists():
        raise FileExistsError(f"preserve existing batch: {output}")
    output.mkdir(parents=True)
    cases = build_counterexamples()
    graph = _graph()
    protocol = {
        "schema": "e3-controlled-counterexamples-v1", "scope": "controlled_mechanism_diagnostic",
        "formal_unseen_test": False, "cases": cases, "strategies": list(STRATEGIES),
        "repeats": repeats, "operation_probe_repeats": 1, "seed": 20261102,
        "planned_timed_executions": len(cases) * len(STRATEGIES) * repeats,
        "planned_operation_probes": len(cases) * len(STRATEGIES),
        "graph": {"attributes": graph.graph,
                  "nodes": [{"id": n, **attrs} for n, attrs in graph.nodes(data=True)],
                  "edges": [{"u": u, "v": v, "key": k, **attrs}
                            for u, v, k, attrs in graph.edges(keys=True, data=True)]},
        "clock": CLOCK, "initial_event": _closure(), "event_binding_truth": [[1, 2, 0]],
        "annotation_coverage_input": 1.0, "weather": None, "model_calls": 0,
        "timing": "loopback HTTP including request handling and serialization; shared setup excluded; probe excluded",
        "flat_objective": "0.5*(length/110)+0.4*(slope_level/5)+0.1*(1-scenery_level/5), detour cap 1.25 inactive",
        "eta_truth": "walk 75 m/min; upper slope level 5 speed factor 0.5, lower slope level 1 factor 1; sum and round to 0.1 min",
        "weak_listener": "offline diagnostic using actual initial response edges; not a production API strategy or timed arm",
    }
    write_json(output / "protocol.json", protocol)
    frozen = input_hashes()
    write_json(output / "source_manifest.json", {
        "source_hashes": frozen,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat()})
    records, weak, fatal, completed_keys = [], [], None, set()
    rng = random.Random(protocol["seed"])

    with (output / "executions.jsonl").open("x", encoding="utf-8") as stream, \
            (output / "weak_diagnostics.jsonl").open("x", encoding="utf-8") as weak_stream:
        def append(record):
            records.append(record)
            completed_keys.add((record["case_id"], record["attempt"], record["strategy"]))
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        try:
            _verify_binding(graph)
            with patch.object(network, "get_annotation_coverage_rate", lambda: 1.0), \
                    product_server(graph, output / "isolated_events.json", database_url) as (post, set_snapshot, identity):
                write_json(output / "isolation.json", identity)
                write_json(output / "runtime.json", runtime_versions(database_url))
                for case in cases:
                    old_graph, new_graph = case_graphs(case)
                    old_effective, old_task = case_truth(case, old_graph, initial=True)
                    new_effective, expected = case_truth(case, new_graph)
                    for attempt in range(1, repeats + 2):
                        probe = attempt > repeats
                        setup = {"case_id": case["id"], "attempt": attempt}
                        try:
                            set_snapshot(old_graph, [_closure()] if case["initial_closure"] else [], CLOCK)
                            initial_status, initial = post("/route", _body(old_graph))
                            setup.update(http_status=initial_status, response=initial)
                            initial_audit = audit_route_response(
                                old_graph, old_effective, old_task, initial.get("data", initial), initial_status)
                            setup["audit"] = initial_audit
                            if initial_audit["status"] != "pass":
                                raise RuntimeError("initial response failed independent scoring")
                            if (initial_audit["metrics"].get("selected_objective_cost")
                                    != case["expected_initial_distance_m"]
                                    or initial_audit["metrics"].get("duration_min")
                                    != case["expected_initial_eta_min"]):
                                raise RuntimeError("initial response differs from hand-computed truth")
                            setup["proof_snapshot"] = _proof_snapshot(database_url, identity, initial)
                            prior = initial["data"]["route_state"]
                            old_edges = initial["data"]["recommended_edge_ids"]
                            diagnostic = listener_diagnostic(old_effective, new_effective, expected, old_edges,
                                                             case["expected_updated_objective_cost"])
                            weak_row = {"case_id": case["id"], "attempt": attempt, "diagnostic": diagnostic}
                            weak.append(weak_row)
                            weak_stream.write(json.dumps(weak_row, ensure_ascii=False, allow_nan=False) + "\n")
                            weak_stream.flush()
                        except Exception as error:
                            failure = _failure(error, "initial_route_and_proof")
                            setup["failure"] = failure
                            for strategy in STRATEGIES:
                                append(_failed_record(case, attempt, strategy, probe, "setup_failed", failure))
                            continue
                        finally:
                            write_json(output / f"setup_{case['id']}_{attempt}.json", setup)
                        order = list(STRATEGIES)
                        rng.shuffle(order)
                        for strategy in order:
                            try:
                                set_snapshot(new_graph, [], CLOCK)
                                record = _execute(post, case, prior, new_graph, new_effective,
                                                  expected, attempt, strategy, probe)
                            except Exception as error:
                                record = _failed_record(case, attempt, strategy, probe, "execution_error",
                                                        _failure(error, "execution_or_scoring"))
                            record["execution_order"] = order.index(strategy) + 1
                            append(record)
                    print(json.dumps({"case": case["id"], "records": len(records)}, ensure_ascii=False), flush=True)
        except Exception as error:
            fatal = _failure(error, "batch_setup_or_execution")
        finally:
            for case in cases:
                for attempt in range(1, repeats + 2):
                    for strategy in STRATEGIES:
                        if (case["id"], attempt, strategy) not in completed_keys:
                            append(_failed_record(case, attempt, strategy, attempt > repeats,
                                                  "not_executed", fatal))
    summary = _summary(protocol, records, weak, frozen == input_hashes(), fatal)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        parser.error("explicit TEST_DATABASE_URL required")
    summary = run_counterexamples(args.output, database_url, args.repeats)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(not summary["completed_successfully"])


if __name__ == "__main__":
    raise SystemExit(main())
