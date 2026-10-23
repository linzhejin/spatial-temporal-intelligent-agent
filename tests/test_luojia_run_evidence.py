import json

import pytest

from experiments.baselines.luojia_run_evidence import project_luojia_run_evidence
from scripts.analysis.export_luojia_run_evidence import main


HEADER = {
    "data_snapshot_id": "a" * 64, "event_snapshot_sha256": "b" * 64,
    "event_clock_epoch": 1500, "enabled_tool_names": ["plan_route"],
    "extra_tool_names": [], "max_model_rounds_per_turn": 6,
    "time_budget_s": 40,
}


def test_luojia_trace_projects_actual_requests_without_prompt_text():
    run = {"events": [
        {"event_type": "agent_tool_trace_started", "payload": {}},
        {"event_type": "agent_model_request", "payload": {
            "requested_model_id": "deepseek-chat", "resolved_model_id": "deepseek-flash",
            "tool_schema_sha256": "c" * 64, "temperature": 0,
            "max_tokens": 800, "input_tokens": 12, "output_tokens": 4,
            "outcome": "success", "prompt": "private text",
        }},
        {"event_type": "agent_tool_call", "payload": {
            "tool_name": "plan_route", "outcome": "success", "args": {"private": 1},
        }},
        {"event_type": "agent_tool_call", "payload": {
            "tool_name": "plan_route", "outcome": "skipped_after_route_verification_retry",
        }},
    ]}
    evidence, usage = project_luojia_run_evidence(run, HEADER)
    assert evidence[0]["condition"] == "luojia"
    assert [item["kind"] for item in evidence] == ["run", "request", "tool"]
    assert usage == {"requests": 1, "input_tokens": 12, "output_tokens": 4}
    assert "private text" not in str(evidence)


def test_luojia_trace_refuses_missing_events_or_usage():
    with pytest.raises(ValueError, match="enabled"):
        project_luojia_run_evidence({"events": []}, HEADER)
    with pytest.raises(ValueError, match="usage"):
        project_luojia_run_evidence({"events": [
            {"event_type": "agent_tool_trace_started", "payload": {}},
            {"event_type": "agent_model_request", "payload": {
                "requested_model_id": "m", "resolved_model_id": "m",
                "tool_schema_sha256": "c" * 64, "temperature": 0,
                "max_tokens": 800, "input_tokens": -1,
                "output_tokens": 0, "outcome": "success",
            }},
        ]}, HEADER)


def test_luojia_evidence_export_scrubs_report_and_refuses_overwrite(tmp_path):
    source = tmp_path / "local-run.json"
    source.write_text(json.dumps({"records": [
        {"kind": "agent_model_request", "payload": {
            "requested_model_id": "deepseek-chat", "resolved_model_id": "deepseek-flash",
            "tool_schema_sha256": "c" * 64, "temperature": 0,
            "max_tokens": 800, "input_tokens": 12, "output_tokens": 4,
            "outcome": "success", "prompt": "private text"}},
    ]}), encoding="utf-8")
    header = tmp_path / "header.json"
    header.write_text(json.dumps(HEADER), encoding="utf-8")
    output = tmp_path / "evidence.json"
    args = ["--run-json", str(source), "--header-json", str(header),
            "--output", str(output)]
    assert main(args) == 0
    assert "private text" not in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(args)
