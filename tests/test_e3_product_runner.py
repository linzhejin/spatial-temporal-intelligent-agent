from copy import deepcopy

from experiments.e3_product_runner import audit_execution_policy, summarize_records


def _protocol():
    return {"cases": [{"id": "a"}], "strategies": ["full", "reuse"],
            "repeats": 1, "operation_probe_repeats": 1, "planned_timed_executions": 2}


def _records():
    return [{"case_id": "a", "strategy": s, "attempt": a,
             "operation_probe": a == 2, "http_status": 200,
             "response": {"data": {"recommended_edge_ids": [[1, 2, 0]], "duration_min": 1}},
             "audit": {"status": "pass"}, "execution_policy_audit": {"status": "pass"}}
            for s in ["full", "reuse"] for a in [1, 2]]


def test_missing_runs_remain_in_planned_denominator_and_fail_acceptance():
    result = summarize_records(_protocol(), _records()[:1], input_hashes_unchanged=True,
                               terminal_error={"type": "TimeoutError"})
    assert result["planned_timed_executions"] == 2
    assert result["actual_timed_executions"] == 1
    assert len(result["missing_executions"]) == 3
    assert not result["acceptance_passed"]


def test_probe_response_change_cannot_silently_pass_timing_batch():
    records = _records()
    records[-1]["response"]["data"]["duration_min"] = 9
    result = summarize_records(_protocol(), records, input_hashes_unchanged=True)
    assert not result["acceptance_passed"]
    assert result["semantic_mismatches_across_repeats_and_probes"] == [{"case_id": "a", "strategy": "reuse"}]


def test_complete_repeat_and_probe_batch_and_duplicate_detection():
    records = _records()
    result = summarize_records(_protocol(), records, input_hashes_unchanged=True)
    assert result["acceptance_passed"]
    assert result["audit_counts"]["timed"]["reuse"]["pass"] == 1
    result = summarize_records(_protocol(), records + [deepcopy(records[0])], input_hashes_unchanged=True)
    assert not result["acceptance_passed"]
    assert result["duplicate_execution_keys"] == 1


def test_identical_full_search_in_both_groups_cannot_pass_reuse_check():
    metric = {"phases": {"dijkstra": {"calls": 1}}}
    result = audit_execution_policy({"variant": "off_path_closure"}, "conservative_reuse", {}, 200, metric)
    assert result["status"] == "fail"
    assert "unexpected_reuse_decision" in result["issues"]
