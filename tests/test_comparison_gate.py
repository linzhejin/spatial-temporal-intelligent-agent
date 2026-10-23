from copy import deepcopy
import json
import hashlib

from experiments.comparison_gate import audit_same_model_conditions


def _manifests():
    return [{"condition": condition, "resolved_model_id": "deepseek-v4-flash",
             "tool_schema_sha256": "a" * 64, "data_snapshot_id": "b" * 64,
             "event_snapshot_sha256": "c" * 64, "event_clock_epoch": 1500,
             "temperature": 0, "max_tokens_per_request": 800,
             "max_model_rounds_per_turn": 6, "time_budget_s": 40,
             "enabled_tool_names": ["resolve_poi", "plan_route"],
             "extra_tool_names": [], "pilot_status": "passed",
             "implementation_revision": "commit-1",
             "measured_usage": {"requests": 2}}
            for condition in ("luojia", "ordinary_tool_agent", "deepseek_harness")]


def test_fairness_gate_requires_same_actual_model_and_tool_surface():
    base = _manifests()
    assert audit_same_model_conditions(base)["formal_ready"] is False
    assert "run_evidence_missing" in audit_same_model_conditions(base)["blockers"]
    unequal = deepcopy(base)
    unequal[2]["extra_tool_names"] = ["shell"]
    unequal[1]["resolved_model_id"] = "different"
    report = audit_same_model_conditions(unequal)
    assert report["formal_ready"] is False
    assert "extra_tools_enabled" in report["blockers"]
    assert "mismatch:resolved_model_id" in report["blockers"]


def test_fairness_gate_distinguishes_zero_calls_from_missing_usage():
    manifests = _manifests()
    manifests[0]["measured_usage"] = {"requests": 0, "input_tokens": 0,
                                      "output_tokens": 0}
    blockers = audit_same_model_conditions(manifests)["blockers"]
    assert "measured_usage_missing" not in blockers
    assert "run_evidence_missing" in blockers


def test_fairness_gate_checks_actual_request_and_tool_records(tmp_path):
    manifests = _manifests()
    for row in manifests:
        records = [
            {"kind": "run", "condition": row["condition"],
             "data_snapshot_id": "b" * 64,
             "event_snapshot_sha256": "c" * 64,
             "event_clock_epoch": 1500,
             "enabled_tool_names": ["resolve_poi", "plan_route"],
             "extra_tool_names": [], "max_model_rounds_per_turn": 6,
             "time_budget_s": 40},
            {"kind": "request", "requested_model_id": "deepseek-v4-flash",
             "resolved_model_id": "deepseek-v4-flash",
             "tool_schema_sha256": "a" * 64,
             "temperature": 0, "max_tokens": 800,
             "input_tokens": 10, "output_tokens": 3},
            {"kind": "tool", "name": "resolve_poi", "outcome": "success"},
            {"kind": "request", "requested_model_id": "deepseek-v4-flash",
             "resolved_model_id": "deepseek-v4-flash",
             "tool_schema_sha256": "a" * 64,
             "temperature": 0, "max_tokens": 800,
             "input_tokens": 10, "output_tokens": 3},
        ]
        path = tmp_path / f'{row["condition"]}.json'
        path.write_text(json.dumps(records), encoding="utf-8")
        row["run_evidence_path"] = str(path)
        row["run_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row["measured_usage"] = {"requests": 2, "input_tokens": 20,
                                 "output_tokens": 6}
    assert audit_same_model_conditions(manifests)["formal_ready"] is True
    path = tmp_path / "deepseek_harness.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records[2]["name"] = "shell"
    path.write_text(json.dumps(records), encoding="utf-8")
    bad = audit_same_model_conditions(manifests)
    assert bad["formal_ready"] is False
    assert "run_evidence_hash_mismatch:deepseek_harness" in bad["blockers"]
    manifests[2]["run_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    bad = audit_same_model_conditions(manifests)
    assert "observed_extra_tool:deepseek_harness:shell" in bad["blockers"]
