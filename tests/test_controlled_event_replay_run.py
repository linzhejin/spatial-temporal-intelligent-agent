"""The development scorer must record missing attempts as failures."""

import json

from experiments.controlled_event_replay import PILOT, ROOT, run_controlled_event_replay


def test_empty_replay_counts_both_cases_as_failed(tmp_path):
    replay = tmp_path / "empty.jsonl"
    replay.write_text("", encoding="utf-8")
    result_dir = run_controlled_event_replay(replay, tmp_path / "runs", "empty")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 2
    assert summary["observed_attempt_count"] == 0
    assert summary["task_success_count"] == 0
    assert summary["failure_counts"] == {"missing_observation": 2}
    assert summary["formal_ready"] is False


def test_projected_dev_answers_pass_but_are_not_model_results(tmp_path):
    dataset = json.loads((ROOT / PILOT / "scenarios_event_pilot_dev_v2.json")
                         .read_text(encoding="utf-8"))
    rows = []
    for case in dataset["cases"]:
        expected = case["expected_turns"][0]
        turn = {key: expected[key] for key in (
            "status", "kind", "start_poi_id", "end_poi_id", "via_poi_ids",
            "travel_mode", "strategy", "hard_constraints", "candidate_poi_ids",
            "clarification_slot")}
        source_event = ("simulated_detour_edge_001" if case["category"] == "compound_tools"
                        else "simulated_blocked_edge_001")
        turn.update(outputs=expected["required_outputs"], unknowns=[],
                    road_event_ids=[source_event],
                    route_edge_ids=case["gold_derivation"]["oracle_result"]["route_edge_ids"],
                    tool_trace=[
                        {"tool_name": "list_road_conditions", "argument_keys": [],
                         "outcome": "success", "artifact_kind": None},
                        {"tool_name": "plan_route", "argument_keys": [],
                         "outcome": "success", "artifact_kind": "route"},
                    ])
        rows.append({"scenario_id": case["id"], "attempt": 1,
                     "snapshot_id": dataset["snapshot_id"],
                     "input_messages": case["messages"], "turns": [turn]})
    replay = tmp_path / "projected.jsonl"
    replay.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                              for row in rows), encoding="utf-8")
    result_dir = run_controlled_event_replay(replay, tmp_path / "runs", "projected")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_success_count"] == 2
    assert summary["formal_ready"] is False
