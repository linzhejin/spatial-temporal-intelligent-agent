"""Focused contract checks for the new full-task experiment scaffold."""

import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.scenario_evaluator import evaluate_scenario, summarize, validate_dataset
from experiments.scenario_runner import (
    _validate_observation_projection, run_replay, validate_snapshot_manifest,
)


def _case(*, expected_status="completed", expected_kind="route"):
    return {
        "id": "dev_01", "category": "multi_via_itinerary",
        "template_group": "t1", "od_group": "od1", "snapshot_id": "a" * 64,
        "adjudication": {"status": "draft", "reviewer": None, "evidence": []},
        "messages": [{"role": "user", "content": "从甲经过乙到丙，不走台阶"}],
        "expected_turns": [{
            "status": expected_status, "kind": expected_kind,
            "feasibility": "feasible", "start_poi_id": "poi_a",
            "end_poi_id": "poi_c", "via_poi_ids": ["poi_b"],
            "travel_mode": "walk", "strategy": "shortest",
            "hard_constraints": {"avoid_steps": True},
            "required_outputs": ["route", "weather"],
            "candidate_poi_ids": [], "clarification_slot": None,
            "allowed_unknowns": [], "intent_label": None, "expected_tools": None,
        }],
    }


def _adjudicated_case(*, expected_status="completed", expected_kind="route"):
    case = _case(expected_status=expected_status, expected_kind=expected_kind)
    case["adjudication"] = {"status": "adjudicated"}
    return case


def _observed(*, checks=None):
    return {"scenario_id": "dev_01", "attempt": 1,
            "snapshot_id": _case()["snapshot_id"],
            "input_messages": _case()["messages"], "turns": [{
        "status": "completed", "kind": "route", "start_poi_id": "poi_a",
        "end_poi_id": "poi_c", "via_poi_ids": ["poi_b"],
        "travel_mode": "walk", "strategy": "shortest",
        "hard_constraints": {"avoid_steps": True}, "candidate_poi_ids": [],
        "clarification_slot": None, "outputs": ["route", "weather"],
        "route_edge_ids": ["1|2|0"],
        "independent_review": checks,
    }]}


def test_route_requires_independent_checks_and_unknown_is_explicit():
    case = _adjudicated_case()
    score = evaluate_scenario(case, _observed())
    assert score["task_success"] is False
    assert "route_review_missing" in score["unknowns"]
    assert score["spatial_output"] is True
    assert score["hard_constraint_decidable"] is False
    assert score["hard_constraint_violation"] is None


def test_completed_route_without_edges_is_not_a_success():
    row = _observed()
    row["turns"][0]["route_edge_ids"] = []
    score = evaluate_scenario(_adjudicated_case(), row)
    assert score["task_success"] is False
    assert "route_evidence_missing" in score["unknowns"]


def test_complete_reviewed_route_can_pass():
    checks = {"source": "independent_reviewer", "reviewer": "human-1",
              "evidence": ["edge-audit-1"], "route": "pass",
              "hard_constraints": {"avoid_steps": "pass"},
              "outputs": {"weather": "pass"}}
    score = evaluate_scenario(_adjudicated_case(), _observed(checks=checks))
    assert score["task_success"] is True
    assert score["hard_constraint_decidable"] is True
    assert score["hard_constraint_violation"] is False


def test_mode_mismatch_fails_the_full_task():
    row = _observed()
    row["turns"][0]["travel_mode"] = "bike"
    assert "wrong_travel_mode" in evaluate_scenario(_adjudicated_case(), row)["failures"]


def test_replay_input_must_match_preregistered_full_history():
    row = _observed()
    row["input_messages"] = [{"role": "user", "content": "不同的问题"}]
    assert "input_mismatch" in evaluate_scenario(_adjudicated_case(), row)["failures"]


def test_replay_snapshot_must_match_gold():
    row = _observed()
    row["snapshot_id"] = "other-snapshot"
    assert "snapshot_mismatch" in evaluate_scenario(_adjudicated_case(), row)["failures"]


def test_placeholder_snapshot_is_explicitly_unknown_even_when_replay_matches():
    case = _adjudicated_case()
    case["snapshot_id"] = "PENDING_FROZEN_SNAPSHOT"
    row = _observed()
    row["snapshot_id"] = case["snapshot_id"]
    score = evaluate_scenario(case, row)
    assert score["task_success"] is False
    assert "snapshot_not_frozen" in score["unknowns"]


def test_transport_error_has_explicit_failure_category():
    row = _observed()
    row["transport_error"] = "timeout"
    assert "transport_error" in evaluate_scenario(_adjudicated_case(), row)["failures"]


def test_nonspatial_output_needs_independent_review():
    case = _adjudicated_case(expected_kind="information")
    case["expected_turns"][0]["required_outputs"] = ["weather"]
    row = _observed()
    row["turns"][0].update(kind="information", outputs=["weather"], route_edge_ids=[])
    score = evaluate_scenario(case, row)
    assert score["task_success"] is False
    assert "output_review_missing" in score["unknowns"]


def test_route_detects_violation_even_when_product_claims_completion():
    checks = {"source": "independent_reviewer", "reviewer": "human-1",
              "evidence": ["edge-audit-1"], "route": "pass",
              "hard_constraints": {"avoid_steps": "fail"}}
    score = evaluate_scenario(_adjudicated_case(), _observed(checks=checks))
    assert score["task_success"] is False
    assert "hard_constraint_violation" in score["failures"]
    assert score["hard_constraint_decidable"] is True
    assert score["hard_constraint_violation"] is True


def test_order_and_partial_output_are_distinct_failures():
    checks = {"source": "independent_reviewer", "reviewer": "human-1",
              "evidence": ["edge-audit-1"], "route": "pass",
              "hard_constraints": {"avoid_steps": "pass"}}
    row = _observed(checks=checks)
    row["turns"][0]["via_poi_ids"] = ["poi_x", "poi_b"]
    row["turns"][0]["outputs"] = ["route"]
    score = evaluate_scenario(_adjudicated_case(), row)
    assert "wrong_poi_order" in score["failures"]
    assert "partial_completion" in score["failures"]


def test_clarification_checks_slot_and_retained_constraints():
    case = _adjudicated_case(expected_status="needs_input", expected_kind="clarify")
    expected = case["expected_turns"][0]
    expected["clarification_slot"] = "sport"
    expected["required_outputs"] = []
    row = _observed()
    row["turns"][0].update(status="needs_input", kind="clarify",
                           clarification_slot="sport", outputs=[], route_edge_ids=[])
    assert evaluate_scenario(case, row)["task_success"] is True
    row["turns"][0]["hard_constraints"] = {}
    assert "dropped_constraint" in evaluate_scenario(case, row)["failures"]


def test_adjudicated_gold_requires_controlled_intent_and_acceptable_tool_sequences():
    case = _case()
    case["expected_turns"][0].update(
        intent_label="route_planning", expected_tools=[["resolve_poi", "plan_route"]]
    )
    dataset = {"protocol_version": "scenario-v1", "split": "development",
               "cases": [case]}
    validate_dataset(dataset)

    case["expected_turns"][0]["intent_label"] = "invented_intent"
    with pytest.raises(ValueError, match="intent_label"):
        validate_dataset(dataset)

    case["expected_turns"][0].update(
        intent_label="route_planning", expected_tools=[["unknown_tool"]]
    )
    with pytest.raises(ValueError, match="tool"):
        validate_dataset(dataset)


def test_tool_choice_is_scored_separately_and_accepts_any_adjudicated_sequence():
    case = _adjudicated_case()
    expected = case["expected_turns"][0]
    expected.update(intent_label="route_planning",
                    expected_tools=[["resolve_poi", "plan_route"], ["plan_route"]])
    observed = _observed()
    observed["turns"][0]["tool_trace"] = [
        {"tool_name": "plan_route", "argument_keys": [],
         "outcome": "success", "artifact_kind": "route"},
    ]
    score = evaluate_scenario(case, observed)
    assert score["tool_use"]["status"] == "pass"
    assert score["tool_use"]["pass_turns"] == 1
    assert score["task_success"] is False  # Route review is still missing.

    observed["turns"][0]["tool_trace"] = []
    score = evaluate_scenario(case, observed)
    assert score["tool_use"]["status"] == "fail"
    assert score["task_success"] is False


def test_tool_choice_without_trace_is_unknown_and_draft_gold_is_unadjudicated():
    case = _adjudicated_case()
    case["expected_turns"][0].update(intent_label="route_planning",
                                    expected_tools=[["plan_route"]])
    assert evaluate_scenario(case, _observed())["tool_use"]["status"] == "unknown"

    case["adjudication"]["status"] = "draft"
    case["expected_turns"][0].update(intent_label=None, expected_tools=None)
    score = evaluate_scenario(case, _observed())
    assert score["tool_use"]["status"] == "unadjudicated"
    assert score["task_success"] is None


def test_draft_gold_is_not_used_to_claim_task_success_or_failure():
    case = _case()
    observed = _observed()
    observed["turns"][0]["travel_mode"] = "bike"
    observed["turns"][0]["route_edge_ids"] = []

    score = evaluate_scenario(case, observed)

    assert score["evaluation_status"] == "unadjudicated"
    assert score["task_success"] is None
    assert score["failures"] == []
    assert "gold_not_adjudicated" in score["unknowns"]
    assert score["spatial_output"] is False
    assert score["hard_constraint_decidable"] is False
    assert score["hard_constraint_violation"] is None
    assert score["tool_use"]["status"] == "unadjudicated"


def test_summary_excludes_draft_attempts_from_success_denominator():
    draft = {"scenario_id": "draft", "intent_labels": [], "observation": {
        "latency_s": 2.0, "tool_calls": 0,
    }, "score": evaluate_scenario(_case(), _observed())}
    adjudicated = {"scenario_id": "reviewed", "intent_labels": [], "observation": {
        "latency_s": 3.0, "tool_calls": 1,
    }, "score": {
        "evaluation_status": "adjudicated", "task_success": True,
        "failures": [], "unknowns": [], "spatial_output": True,
        "hard_constraint_decidable": True, "hard_constraint_violation": False,
        "tool_use": {"pass_turns": 0, "fail_turns": 0, "unknown_turns": 0,
                     "unadjudicated_turns": 0, "turn_verdicts": []},
    }}

    report = summarize([draft, adjudicated])

    assert report["n_attempts"] == 2
    assert report["adjudicated_attempts"] == 1
    assert report["unadjudicated_attempts"] == 1
    assert report["complete_task_success_attempts"] == 1
    assert report["complete_task_success_denominator_attempts"] == 1
    assert report["complete_task_success_rate_attempt"] == 1.0
    assert report["complete_task_success_rate_scenario_mean"] == 1.0


def test_missing_observation_in_draft_is_execution_failure_not_gold_failure():
    score = evaluate_scenario(_case(), None)

    assert score["evaluation_status"] == "unadjudicated"
    assert score["task_success"] is None
    assert score["failures"] == []
    assert score["execution_failures"] == ["missing_observation"]


def test_summary_reports_tool_selection_accuracy_with_its_own_denominator():
    records = [{"scenario_id": "a", "intent_labels": ["route_planning", "route_planning"], "score": {
        "task_success": True, "failures": [], "unknowns": [],
        "spatial_output": False, "hard_constraint_decidable": False,
        "hard_constraint_violation": None,
        "tool_use": {"status": "pass", "pass_turns": 2, "fail_turns": 0,
                      "unknown_turns": 0, "unadjudicated_turns": 0,
                      "turn_verdicts": ["pass", "pass"]},
    }}, {"scenario_id": "b", "intent_labels": ["route_planning"], "score": {
        "task_success": False, "failures": [], "unknowns": [],
        "spatial_output": False, "hard_constraint_decidable": False,
        "hard_constraint_violation": None,
        "tool_use": {"status": "fail", "pass_turns": 0, "fail_turns": 1,
                      "unknown_turns": 0, "unadjudicated_turns": 0,
                      "turn_verdicts": ["fail"]},
    }}]
    report = summarize(records)
    assert report["tool_selection_pass_turns"] == 2
    assert report["tool_selection_fail_turns"] == 1
    assert report["tool_selection_decidable_turns"] == 3
    assert report["tool_selection_accuracy"] == 0.6667
    assert report["tool_selection_by_intent"]["route_planning"]["accuracy"] == 0.6667


def test_dataset_requires_complete_gold_and_manual_adjudication_for_formal():
    dataset = {"protocol_version": "scenario-v1", "split": "development",
               "cases": [_case()]}
    validate_dataset(dataset)
    with pytest.raises(ValueError, match="adjudicated"):
        validate_dataset(dataset, formal=True)
    dataset["cases"][0]["expected_turns"][0].pop("via_poi_ids")
    with pytest.raises(ValueError, match="via_poi_ids"):
        validate_dataset(dataset)


def test_formal_dataset_cannot_claim_full_e1_with_one_case():
    case = _case()
    case["adjudication"] = {"status": "adjudicated", "method": "manual_independent",
                            "reviewer": "human-1", "evidence": ["review-sheet-1"]}
    dataset = {"protocol_version": "scenario-v1", "split": "test",
               "repeats": 3, "cases": [case]}
    with pytest.raises(ValueError, match="300"):
        validate_dataset(dataset, formal=True)


def test_summary_groups_repeated_attempts_by_scenario():
    def record(case_id, passed):
        return {"scenario_id": case_id,
                "score": {"task_success": passed, "failures": [], "unknowns": [],
                          "spatial_output": False, "hard_constraint_decidable": False,
                          "hard_constraint_violation": None}}
    report = summarize([record("a", True), record("a", False), record("a", False),
                        record("b", True)])
    assert report["complete_task_success_rate_attempt"] == 0.5
    assert report["complete_task_success_rate_scenario_mean"] == 0.6667


def test_summary_reports_latency_and_cost_without_inventing_missing_values():
    records = [
        {"scenario_id": "a", "observation": {"latency_s": 2.0, "tool_calls": 3,
                                              "input_tokens": 10, "output_tokens": 20},
         "score": {"task_success": False, "failures": [], "unknowns": [],
                   "spatial_output": False, "hard_constraint_decidable": False,
                   "hard_constraint_violation": None}},
        {"scenario_id": "b", "observation": None,
         "score": {"task_success": False, "failures": ["missing_observation"],
                   "unknowns": [], "spatial_output": False,
                   "hard_constraint_decidable": False,
                   "hard_constraint_violation": None}},
    ]
    report = summarize(records)
    assert report["latency_p50_s"] == 2.0
    assert report["latency_p95_s"] == 2.0
    assert report["latency_observed_attempts"] == 1
    assert report["tool_calls_total"] == 3
    assert report["input_tokens_total"] == 10


def _development_dataset():
    path = Path(__file__).resolve().parents[1] / "experiments" / "datasets" / "scenarios_dev_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_development_dataset_has_exact_protocol_quota():
    dataset = _development_dataset()
    protocol_path = Path(__file__).resolve().parents[1] / "experiments" / "protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert len(dataset["cases"]) == 60
    assert Counter(case["category"] for case in dataset["cases"]) == {
        category: 10 for category in protocol["categories"]
    }
    from agents.tools import TOOL_NAMES
    assert set(protocol["gold_contract"]["available_tools"]) == TOOL_NAMES


def test_shipped_development_dataset_is_explicitly_draft_and_schema_valid():
    dataset = _development_dataset()
    validate_dataset(dataset)
    assert {case["adjudication"]["status"] for case in dataset["cases"]} == {"draft"}


def test_development_drafts_have_no_adjudication_or_empirical_gold():
    dataset = _development_dataset()
    assert "NOT SCOREABLE AS RESEARCH EVIDENCE" in dataset["authoring_status"]
    assert dataset["authoring_notes"]
    for case in dataset["cases"]:
        assert case["snapshot_id"].startswith("PENDING_"), case["id"]
        assert case["adjudication"] == {
            "status": "draft", "method": None, "reviewer": None, "evidence": []
        }, case["id"]
        assert len(case["messages"]) == len(case["expected_turns"])
        for turn in case["expected_turns"]:
            assert turn["feasibility"] == "unknown", case["id"]
            assert turn["candidate_poi_ids"] == [], case["id"]


def test_development_conversations_are_unique_and_preserve_seed_messages():
    cases = _development_dataset()["cases"]
    conversations = [tuple(message["content"].strip() for message in case["messages"])
                     for case in cases]
    assert len(set(conversations)) == len(conversations)
    seeds = {
        "seed_place_01": ("我在玉兰2门附近，推荐一个能吃饭的地方",),
        "seed_mode_01": ("从玉兰2门到樱顶，步行且不要走已标记的台阶",),
        "seed_via_01": ("从玉兰2门先到卓尔体育馆，再到桂园食堂，最后到樱顶",),
        "seed_compound_01": ("查一下天气，再规划从玉兰2门到樱顶的步行路线",),
        "seed_revision_01": (
            "从玉兰2门到樱顶，别走已标记台阶", "顺便查一下天气",
            "刚才路线改为平坦优先，别走台阶仍然保留",
        ),
        "seed_event_01": (
            "如果活动封闭了唯一可通行入口，从玉兰2门到樱顶并且不能穿越封闭路段，该怎么走？",
        ),
    }
    by_id = {case["id"]: case for case in cases}
    for case_id, messages in seeds.items():
        assert tuple(message["content"] for message in by_id[case_id]["messages"]) == messages


def test_development_poi_references_exist_in_current_catalog():
    poi_path = Path(__file__).resolve().parents[1] / "data" / "pois.json"
    catalog = json.loads(poi_path.read_text(encoding="utf-8"))
    poi_ids = {poi["id"] for poi in catalog["pois"]}
    for case in _development_dataset()["cases"]:
        for turn in case["expected_turns"]:
            references = [turn["start_poi_id"], turn["end_poi_id"],
                          *turn["via_poi_ids"], *turn["candidate_poi_ids"]]
            for poi_id in references:
                assert poi_id is None or poi_id in poi_ids, (case["id"], poi_id)


def test_development_templates_vary_within_each_category():
    groups = {}
    for case in _development_dataset()["cases"]:
        groups.setdefault(case["category"], set()).add(case["template_group"])
    assert all(len(templates) == 10 for templates in groups.values())


def test_formal_snapshot_requires_ready_complete_inputs_and_matching_fingerprint():
    from scripts.validate.check_agent_research_readiness import REQUIRED_INPUTS

    fingerprint = "a" * 64
    versions = {name: {"sha256": "b" * 64, "files": [name], "missing": []}
                for name in REQUIRED_INPUTS}
    versions["effective_inputs_sha256"] = fingerprint
    versions["event_state"]["active_sha256"] = "c" * 64
    versions["runtime_model_config"] = {"sha256": "d" * 64}
    report = {"scope": "agent_main_experiment", "ready": True,
              "versions": versions}
    validate_snapshot_manifest(report, [dict(_case(), snapshot_id=fingerprint)])
    report["versions"]["poi"]["missing"] = ["data/pois.json"]
    with pytest.raises(ValueError, match="poi"):
        validate_snapshot_manifest(report, [dict(_case(), snapshot_id=fingerprint)])


def test_replay_creates_new_run_and_never_overwrites(tmp_path):
    case = _case()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"protocol_version": "scenario-v1",
                                   "split": "development", "cases": [case]}), encoding="utf-8")
    replay = tmp_path / "replay.jsonl"
    replay.write_text(json.dumps(_observed()) + "\n", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"protocol_version": "scenario-v1"}), encoding="utf-8")
    first = run_replay(dataset, replay, protocol, tmp_path / "out", "trial-a")
    assert (first / "manifest.json").exists()
    assert (first / "records.jsonl").exists()
    assert (first / "summary.json").exists()
    with pytest.raises(FileExistsError):
        run_replay(dataset, replay, protocol, tmp_path / "out", "trial-a")
    second = run_replay(dataset, replay, protocol, tmp_path / "out", "trial-b")
    assert first != second
    assert json.loads((first / "summary.json").read_text(encoding="utf-8"))["n_attempts"] == 1


def test_replay_rejects_unprojected_sensitive_fields(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"protocol_version": "scenario-v1", "split": "development",
                                   "cases": [_case()]}), encoding="utf-8")
    replay = tmp_path / "replay.jsonl"
    replay.write_text(json.dumps({**_observed(), "session_cookie": "secret"}) + "\n",
                      encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"protocol_version": "scenario-v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unprojected"):
        run_replay(dataset, replay, protocol, tmp_path / "out", "trial-a")
    assert not (tmp_path / "out" / "trial-a").exists()


@pytest.mark.parametrize("artifact_kind", ["candidates", "route_comparison"])
@pytest.mark.parametrize("raw_field", ["raw_arguments", "raw_result"])
def test_replay_accepts_allowlisted_tool_trace_without_raw_arguments(artifact_kind, raw_field):
    observed = _observed()
    observed["tool_calls"] = 1
    observed["turns"][0]["tool_trace"] = [{
        "tool_name": "search_poi_candidates",
        "argument_keys": ["activities", "start"],
        "outcome": "success", "artifact_kind": artifact_kind,
    }]

    _validate_observation_projection(observed)

    observed["tool_calls"] = 2
    with pytest.raises(ValueError, match="does not match"):
        _validate_observation_projection(observed)
    observed["tool_calls"] = 1

    observed["turns"][0]["tool_trace"][0][raw_field] = {"start": "secret"}
    with pytest.raises(ValueError, match="unprojected"):
        _validate_observation_projection(observed)


def test_replay_rejects_unknown_tool_trace_artifact_kind():
    observed = _observed()
    observed["turns"][0]["tool_trace"] = [{
        "tool_name": "compare_routes", "argument_keys": [],
        "outcome": "success", "artifact_kind": "unknown_artifact",
    }]

    with pytest.raises(ValueError, match="invalid values"):
        _validate_observation_projection(observed)


@pytest.mark.parametrize("outcome", [
    "skipped_after_route_verification_retry",
    "skipped_after_route_verification_failure",
])
def test_replay_accepts_planner_skipped_tool_trace(outcome):
    observed = _observed()
    observed["tool_calls"] = 1
    observed["turns"][0]["tool_trace"] = [{
        "tool_name": "plan_route", "argument_keys": [],
        "outcome": outcome, "artifact_kind": None,
    }]

    _validate_observation_projection(observed)


def test_replay_rejects_unknown_tool_trace_outcome():
    observed = _observed()
    observed["turns"][0]["tool_trace"] = [{
        "tool_name": "plan_route", "argument_keys": [],
        "outcome": "unknown_outcome", "artifact_kind": None,
    }]

    with pytest.raises(ValueError, match="invalid values"):
        _validate_observation_projection(observed)


def test_formal_replay_requires_development_split_for_leakage_audit(tmp_path):
    dataset = tmp_path / "test.json"
    dataset.write_text(json.dumps({"protocol_version": "scenario-v1", "split": "test", "cases": []}), encoding="utf-8")
    replay = tmp_path / "replay.jsonl"
    replay.write_text("", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"protocol_version": "scenario-v1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="development dataset.*leakage"):
        run_replay(dataset, replay, protocol, tmp_path / "out", "formal", formal=True)

    assert not (tmp_path / "out" / "formal").exists()

def test_formal_replay_runs_split_audit_before_creating_result_dir(tmp_path, monkeypatch):
    import experiments.scenario_runner as scenario_runner

    case = _case()
    test = tmp_path / "test.json"
    test.write_text(json.dumps({"protocol_version": "scenario-v1", "split": "test",
                                "repeats": 3, "cases": [case]}), encoding="utf-8")
    dev = tmp_path / "development.json"
    dev.write_text(json.dumps({"split": "development", "cases": [_case()]}), encoding="utf-8")
    replay = tmp_path / "replay.jsonl"
    replay.write_text("", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"protocol_version": "scenario-v1"}), encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"versions": {"git_commit": "fake-commit"}}), encoding="utf-8")

    monkeypatch.setattr(scenario_runner, "validate_dataset", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scenario_runner, "validate_snapshot_manifest", lambda *_args: "a" * 64)
    monkeypatch.setattr(scenario_runner, "_git_dirty", lambda: False)
    monkeypatch.setattr(scenario_runner, "_git_head", lambda: "fake-commit")
    monkeypatch.setattr(scenario_runner, "audit_scenario_splits", lambda *_args, **_kwargs: {
        "formal_ready": False, "blockers": [{"code": "template_group_overlap"}],
    })

    with pytest.raises(ValueError, match="split leakage/readiness audit failed.*template_group_overlap"):
        scenario_runner.run_replay(
            test, replay, protocol, tmp_path / "out", "formal", formal=True,
            snapshot_manifest=snapshot, development_dataset_path=dev,
        )

    assert not (tmp_path / "out").exists()
