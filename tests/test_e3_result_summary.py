"""Protect E3 result denominators and paired comparisons from reporting errors."""
from copy import deepcopy

import pytest

from scripts.research.summarize_e3_product import (
    aggregates, archive_batches, check_batch, flat_record, paired_rows, percentile,
)


def record(policy, attempt, elapsed, *, status=200, cost=100.0):
    probe = attempt == 2
    reuse = policy == "conservative_reuse"
    return {
        "case_id": "od_01__unchanged", "family": "od_01", "variant": "unchanged",
        "strategy": policy, "attempt": attempt, "operation_probe": probe,
        "http_status": status, "elapsed_ms": elapsed,
        "response": {"data": {"recommended_length_m": cost, "duration_min": 2.0,
                              "execution_optimization": {"path_reused": reuse}}},
        "requests": [{"status": status}],
        "audit": {"status": "pass", "metrics": {"oracle_cost": cost, "selected_objective_cost": cost}},
        "execution_policy_audit": {"status": "pass"},
        "metrics": {"phases": {"dijkstra": {"calls": 0 if reuse else 1}},
                    "searches": [] if reuse else [{"adjacency_expanded_nodes": 3 if probe else None,
                                                  "weight_evaluations": 4 if probe else None}]},
    }


def sample():
    return [record("full_recompute", 1, 100), record("conservative_reuse", 1, 120),
            record("full_recompute", 2, 10000), record("conservative_reuse", 2, 20000)]


def protocol():
    return {"cases": [{"id": "od_01__unchanged"}], "repeats": 1,
            "operation_probe_repeats": 1}


def test_probe_overhead_does_not_enter_request_statistics():
    rows = [flat_record(r) for r in sample()]
    result = aggregates(rows, paired_rows(rows), None)
    full, reuse = result
    assert full["timed_executions"] == full["operation_probes"] == 1
    assert full["request_ms_p50"] == 100
    assert reuse["request_ms_p50"] == 120
    assert full["probe_dijkstra_adjacency_expansions"] == 3
    assert reuse["probe_dijkstra_adjacency_expansions"] == 0
    assert full["paired_reuse_minus_full_ms_p50"] == 20
    assert full["paired_reuse_over_full_ratio_p50"] == 1.2


def test_missing_or_duplicated_executions_cannot_disappear_from_denominator():
    assert check_batch(protocol(), sample()) == []
    assert check_batch(protocol(), sample()[:-1])[0]["type"] == "execution_set_mismatch"
    assert check_batch(protocol(), sample() + sample()[:1])[0]["type"] == "duplicate_execution_keys"


def test_probe_label_and_failed_score_are_checked():
    rows = sample()
    rows[0]["operation_probe"] = True
    rows[1]["audit"]["status"] = "unknown"
    kinds = {i["type"] for i in check_batch(protocol(), rows)}
    assert kinds == {"wrong_probe_label", "independent_audit_not_pass"}


def test_correct_no_route_is_not_counted_as_route_delivery():
    rows = sample()
    for r in rows:
        r["http_status"] = 404
        r["response"] = {"error": "route_not_found"}
        r["audit"]["metrics"] = {"reachable": False}
    flat = [flat_record(r) for r in rows]
    pairs = paired_rows(flat)
    assert all(r["objective_delta"] is None and r["same_outcome"] for r in pairs)
    assert all(r["cost_equivalent"] is None and not r["cost_comparison_applicable"] for r in pairs)
    for r in aggregates(flat, pairs, None):
        assert r["correct_no_route_404"] == 1
        assert r["route_deliveries_200"] == 0
        assert r["cost_comparable_pairs"] == r["cost_equivalent_pairs"] == 0
        assert r["cost_not_applicable_pairs"] == 1


def test_version_rejection_then_rebuild_is_one_flow_with_two_requests():
    rows = sample()
    for r in rows:
        r["requests"] = [{"status": 409}, {"status": 200}]
    flat = [flat_record(r) for r in rows]
    for r in aggregates(flat, paired_rows(flat), None):
        assert r["timed_executions"] == r["route_deliveries_200"] == 1
        assert r["rejected_then_rebuilt_flows"] == r["version_rejections_409"] == 1
        assert r["update_http_requests"] == 2


def test_pair_cost_disagreement_is_not_hidden_by_equal_route_length():
    rows = sample()
    rows[1]["audit"]["metrics"]["selected_objective_cost"] = 101
    pair = paired_rows([flat_record(r) for r in rows])[0]
    assert pair["objective_delta"] == 1
    assert pair["cost_equivalent"] is False


def test_unpaired_condition_is_rejected():
    with pytest.raises(ValueError, match="unpaired"):
        paired_rows([flat_record(r) for r in sample()[:-1]])


def test_percentiles_are_explicitly_interpolated():
    assert percentile([100, 200, 300], .95) == 290
    assert percentile([], .95) is None


def test_archive_roundtrip_preserves_original_bytes(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    original = '{"说明":"实测"}\n'.encode()
    (batch / "record.json").write_bytes(original)
    result = archive_batches([batch], tmp_path / "raw.zip")
    assert result[0]["path"] == "batch/record.json"
    assert result[0]["bytes"] == len(original)
    assert (batch / "record.json").read_bytes() == original


def test_pairing_does_not_depend_on_execution_order():
    rows = [flat_record(r) for r in sample()]
    reversed_rows = list(reversed(deepcopy(rows)))
    assert paired_rows(rows) == paired_rows(reversed_rows)
