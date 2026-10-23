import pytest

from experiments.scenario_live_adapter import build_poi_lookup
from scripts.analysis.audit_development_pilots import _digest, audit_pilot


def _case():
    return {"id": "dev_1", "messages": [{"role": "user", "content": "test"}]}


def _components():
    return {"data/pois.json": "a" * 64}


def _report(artifact, *, selected_start_name=None, resolved_model_id="deepseek-flash"):
    return {
        "status": "development_pilot_only", "case_id": "dev_1",
        "development_dataset_sha256": "b" * 64,
        "case_messages_sha256": _digest(_case()["messages"]),
        "data_component_sha256": _components(),
        "tool_schema_sha256": "d" * 64,
        "baseline_system_prompt_sha256": "e" * 64,
        "resolved_model_id": resolved_model_id,
        "tool_calls": 1,
        "turns": [{"status": "completed", "artifacts": [artifact],
                   "selected_start_name": selected_start_name,
                   "tool_trace": [{"tool_name": "ask_user", "argument_keys": ["question"],
                                   "outcome": "success", "artifact_kind": "clarify"}]}],
    }


def _audit(report):
    lookup = build_poi_lookup([{"id": "poi_start", "name": "Start"},
                               {"id": "poi_food", "name": "Food"}])
    return audit_pilot(report, report_sha256="c" * 64, case=_case(),
                       dataset_sha256="b" * 64, lookup=lookup,
                       component_sha256=_components(),
                       tool_schema_sha256="d" * 64, prompt_sha256="e" * 64)


def test_candidate_origin_from_tool_result_is_projected():
    report = _report({"candidates": [{"poi_id": "poi_food", "name": "Food"}]},
                     selected_start_name="Start")
    row = _audit(report)
    assert row["turns"][0]["start_poi_id"] == "poi_start"
    assert row["turns"][0]["candidate_poi_ids"] == ["poi_food"]
    assert row["projection_gaps"] == []
    assert row["assessment"] == "projection_only_not_scored"


def test_missing_candidate_origin_and_clarification_slot_are_explicit():
    candidates = _audit(_report({"candidates": [{"poi_id": "poi_food"}]}))
    assert "turn_1:candidate_origin_unobserved" in candidates["projection_gaps"]
    clarify = _audit(_report({"clarify": {"question": "which?", "options": []}},
                             resolved_model_id=None))
    assert "turn_1:clarification_slot_unobserved" in clarify["projection_gaps"]
    assert "provider_response_model_unobserved" in clarify["projection_gaps"]


def test_missing_input_binding_and_trace_mismatch_fail_closed():
    report = _report({"clarify": {"question": "which?", "options": []}})
    report.pop("case_messages_sha256")
    assert "input_hash_absent_or_mismatch" in _audit(report)["projection_gaps"]
    report.pop("baseline_system_prompt_sha256")
    assert "baseline_prompt_hash_absent_or_mismatch" in _audit(report)["projection_gaps"]
    report["tool_calls"] = 2
    with pytest.raises(ValueError, match="trace"):
        _audit(report)
