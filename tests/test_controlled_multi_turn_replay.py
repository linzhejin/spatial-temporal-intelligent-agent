"""Frozen multi-turn observations must be scored, including missing attempts."""

import json
from pathlib import Path

import pytest

from experiments.controlled_multi_turn_replay import run_controlled_multi_turn_replay


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/datasets/controlled_multi_turn_pilot_v2"


def test_missing_attempts_are_failures_and_output_is_immutable(tmp_path):
    replay = tmp_path / "empty.jsonl"
    replay.write_text("", encoding="utf-8")
    result = run_controlled_multi_turn_replay(replay, tmp_path, "empty")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 2
    assert summary["task_success_count"] == 0
    assert summary["failure_counts"] == {"missing_observation": 2}
    assert summary["formal_ready"] is False
    run_manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    frozen_manifest = json.loads((PILOT / "source_manifest_multi_turn_dev_v1.json").read_text(encoding="utf-8"))
    assert run_manifest["delegated_route_scorer_sha256"] == frozen_manifest["inputs"]["route_scorer"]["sha256"]
    assert run_manifest["projection_validator_sha256"] == frozen_manifest["inputs"]["observation_projection"]["sha256"]
    assert run_manifest["replay_collector_verified"] is False
    with pytest.raises(FileExistsError):
        run_controlled_multi_turn_replay(replay, tmp_path, "empty")


def test_unknown_attempt_is_rejected_before_result_creation(tmp_path):
    packet = json.loads((PILOT / "scenarios_multi_turn_dev_v2.json").read_text(encoding="utf-8"))
    replay = tmp_path / "wrong.jsonl"
    replay.write_text(json.dumps({"scenario_id": "made_up", "attempt": 1,
                                  "snapshot_id": packet["snapshot_id"],
                                  "input_messages": packet["cases"][0]["messages"],
                                  "turns": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown scenario"):
        run_controlled_multi_turn_replay(replay, tmp_path, "wrong")
    assert not (tmp_path / "wrong").exists()


def test_constructed_two_turn_observations_score_success_then_detect_lost_origin(tmp_path):
    packet = json.loads((PILOT / "scenarios_multi_turn_dev_v2.json").read_text(encoding="utf-8"))
    assert all(turn["hard_constraints"] == {"avoid_steps": True}
               for case in packet["cases"] for turn in case["expected_turns"])
    rows = []
    for case in packet["cases"]:
        turns = []
        for index, expected in enumerate(case["expected_turns"]):
            turns.append({key: expected[key] for key in (
                "status", "kind", "start_poi_id", "end_poi_id", "via_poi_ids",
                "travel_mode", "strategy", "hard_constraints", "candidate_poi_ids",
                "clarification_slot",
            )} | {
                "outputs": expected["required_outputs"],
                "route_edge_ids": case["gold_derivation"]["oracle_result"]["turns"][index]["route_edge_ids"],
                "unknowns": [],
            })
        rows.append({"scenario_id": case["id"], "attempt": 1,
                     "snapshot_id": packet["snapshot_id"],
                     "input_messages": case["messages"], "turns": turns,
                     "transport_error": None})
    replay = tmp_path / "constructed.jsonl"
    replay.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    result = run_controlled_multi_turn_replay(replay, tmp_path, "constructed")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_success_count"] == 2

    rows[0]["turns"][1]["start_poi_id"] = rows[0]["turns"][0]["end_poi_id"]
    replay.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    result = run_controlled_multi_turn_replay(replay, tmp_path, "lost_origin")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_success_count"] == 1
    assert summary["failure_counts"]["wrong_poi"] == 1
