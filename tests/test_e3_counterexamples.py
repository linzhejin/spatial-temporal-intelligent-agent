"""Deterministic truth and bookkeeping for the separate E3 counterexamples."""

from contextlib import contextmanager
import json

import networkx as nx
import pytest

from experiments import e3_counterexamples as counter


def test_case_optima_and_flat_objective_are_known_before_product_execution():
    cases = counter.build_counterexamples()
    assert len(cases) == 3
    by_id = {case["id"]: case for case in cases}
    for case in cases:
        old_graph, new_graph = counter.case_graphs(case)
        old_effective, old_task = counter.case_truth(case, old_graph, initial=True)
        new_effective, new_task = counter.case_truth(case, new_graph)
        assert old_task["strategy"] == "shortest"
        initial = nx.shortest_path_length(old_effective, 1, 4, weight="length")
        assert initial == case["expected_initial_distance_m"]
        weight = "audit_objective_cost" if new_task["strategy"] == "flat" else "length"
        optimum = nx.shortest_path_length(new_effective, 1, 4, weight=weight)
        assert optimum == pytest.approx(case["expected_updated_objective_cost"])
    assert by_id["reopen_shortcut"]["expected_initial_distance_m"] == 220
    assert by_id["off_path_cost_decrease"]["expected_updated_objective_cost"] == 160
    assert by_id["flat_weight_change"]["expected_updated_objective_cost"] == pytest.approx(1.24)
    old_graph, new_graph = counter.case_graphs(by_id["flat_weight_change"])
    assert old_graph[1][2][0]["length"] == new_graph[1][2][0]["length"] == 100
    assert 220 < 200 * 1.25


@pytest.mark.parametrize("case_id", ["reopen_shortcut", "off_path_cost_decrease", "flat_weight_change"])
def test_path_only_listener_misses_each_counterexample(case_id):
    case = next(c for c in counter.build_counterexamples() if c["id"] == case_id)
    old_graph, new_graph = counter.case_graphs(case)
    old_effective, _ = counter.case_truth(case, old_graph, initial=True)
    new_effective, task = counter.case_truth(case, new_graph)
    old_edges = ([[1, 3, 0], [3, 4, 0]] if case_id == "reopen_shortcut"
                 else [[1, 2, 0], [2, 4, 0]])
    result = counter.listener_diagnostic(old_effective, new_effective, task, old_edges,
                                         case["expected_updated_objective_cost"])
    assert result["would_reuse"] is True
    assert result["old_path_still_feasible"] is True
    assert result["false_optimality_reuse"] is True
    assert result["objective_gap"] > 0
    if case_id == "flat_weight_change":
        assert result["retained_objective_cost"] == pytest.approx(1.789090909090909)
    else:
        assert result["objective_gap"] == (20 if case_id == "reopen_shortcut" else 40)


def test_semantics_exclude_random_ids_but_keep_task_and_result():
    body = {"data": {"mode": "walk", "duration_min": 2.7,
                     "route_state": {"route_id": "one", "reuse_certificate": {"proof_id": "one"},
                                     "strategy": {"name": "shortest"}, "hard_constraints": {}},
                     "recommended_edge_ids": [[1, 2, 0], [2, 4, 0]],
                     "timings_ms": {"agent": 0}}}
    signature = counter.semantic_result(200, body)
    body["data"]["route_state"]["route_id"] = "two"
    body["data"]["route_state"]["reuse_certificate"]["proof_id"] = "two"
    body["data"]["timings_ms"] = {"agent": 12}
    assert counter.semantic_result(200, body) == signature
    body["data"]["duration_min"] = 3.0
    assert counter.semantic_result(200, body) != signature


def test_setup_errors_remain_in_all_planned_denominators(tmp_path, monkeypatch):
    @contextmanager
    def unavailable_product(*_args):
        def post(*_args):
            raise ConnectionError("simulated failure")
        yield post, lambda *_args: None, {"owner_id": "test-owner"}

    monkeypatch.setattr(counter, "product_server", unavailable_product)
    monkeypatch.setattr(counter, "input_hashes", lambda: {"input": "frozen"})
    monkeypatch.setattr(counter, "runtime_versions", lambda _url: {"test": True})
    output = tmp_path / "batch"
    summary = counter.run_counterexamples(output, "unused-test-url", repeats=1)
    records = [json.loads(line) for line in (output / "executions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["planned_timed_executions"] == 6
    assert summary["planned_operation_probes"] == 6
    assert len(records) == 12
    assert len({(r["case_id"], r["strategy"], r["attempt"]) for r in records}) == 12
    assert summary["completed_successfully"] is False
    assert summary["actual_timed_executions"] == 0
    assert all(r["execution_status"] == "setup_failed" for r in records)
    assert summary["timed_audit_counts"]["fail"] == 6
    assert summary["operation_probe_audit_counts"]["fail"] == 6


def test_independent_scoring_is_outside_request_timing_and_search_instrumentation(monkeypatch):
    case = next(c for c in counter.build_counterexamples() if c["id"] == "off_path_cost_decrease")
    _old, graph = counter.case_graphs(case)
    effective, expected = counter.case_truth(case, graph)
    clock = {"seconds": 0.0, "instrumented": False, "audits": 0}

    @contextmanager
    def capture(*, expansion_probe):
        assert expansion_probe is False
        clock["instrumented"] = True
        try:
            yield {"test_metrics": True}
        finally:
            clock["instrumented"] = False

    def post(path, _body):
        assert clock["instrumented"]
        clock["seconds"] += 0.01
        if path == "/route/replan":
            return 409, {"error": {"code": "route_state_version_conflict"}}
        return 200, {"data": {"duration_min": 2.1}}

    def audit(_raw, _effective, _expected, _body, _status):
        assert clock["instrumented"] is False
        clock["seconds"] += 100
        clock["audits"] += 1
        return {"status": "pass", "issues": [], "checks": {},
                "metrics": {"selected_objective_cost": 160.0, "duration_min": 2.1}}

    monkeypatch.setattr(counter, "capture_execution_metrics", capture)
    monkeypatch.setattr(counter.time, "perf_counter", lambda: clock["seconds"])
    monkeypatch.setattr(counter, "audit_route_response", audit)
    record = counter._execute(post, case, {}, graph, effective, expected,
                              1, "full_recompute", False)
    assert record["execution_status"] == "completed"
    assert record["audit"]["status"] == "pass"
    assert record["elapsed_ms"] == pytest.approx(20)
    assert clock["audits"] == 2
    assert [request["status"] for request in record["requests"]] == [409, 200]
    assert len(record["semantics"]["transition"]) == 2


def test_existing_batch_is_never_overwritten(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        counter.run_counterexamples(tmp_path, "unused-test-url", repeats=1)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_invalid_repeats_do_not_create_output(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="positive"):
        counter.run_counterexamples(output, "unused-test-url", repeats=0)
    assert not output.exists()
