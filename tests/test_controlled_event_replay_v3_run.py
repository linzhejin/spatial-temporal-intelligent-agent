"""The v3 runner scores complete projected event attempts without raw payloads."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments import controlled_event_replay_v3_run as replay_module
from experiments.controlled_event_replay_v3_run import run_controlled_event_replay_v3


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/datasets/controlled_event_pilot_v3"


def _projected_rows():
    dataset = json.loads((PILOT / "scenarios_event_pilot_dev_v3.json")
                         .read_text(encoding="utf-8"))
    source = json.loads((PILOT / "event_snapshots_dev_v1.json")
                        .read_text(encoding="utf-8"))
    snapshots = {item["id"]: item for item in source["snapshots"]}
    anchor_rows = json.loads((ROOT / "experiments/datasets/controlled_spatial_v1/"
                              "walk_anchors.json").read_text(encoding="utf-8"))["anchors"]
    anchors = {item["poi_id"]: item for item in anchor_rows}
    rows = []
    for case in dataset["cases"]:
        expected = case["expected_turns"][0]
        turn = {key: deepcopy(expected[key]) for key in (
            "status", "kind", "start_poi_id", "end_poi_id", "via_poi_ids",
            "travel_mode", "strategy", "hard_constraints", "candidate_poi_ids",
            "clarification_slot",
        )}
        snapshot_id = case["gold_derivation"]["parameters"]["event_snapshot_id"]
        snapshot = snapshots[snapshot_id]
        route_call = {"tool_name": "plan_route", "argument_keys": ["start", "end"],
                      "outcome": "success", "artifact_kind": "route"}
        if case["category"] == "event_infeasible":
            params = case["gold_derivation"]["parameters"]
            request = {"start_node": int(anchors[params["start_poi_id"]]["node_id"]),
                       "end_node": int(anchors[params["end_poi_id"]]["node_id"]),
                       "mode": params["mode"], "avoid_steps": params["avoid_steps"]}
            digest = hashlib.sha256(json.dumps(
                request, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            route_call.update(outcome="tool_error", artifact_kind=None,
                              error_code="graph_no_path", request_digest=digest)
        turn.update(
            outputs=expected["required_outputs"], unknowns=[],
            road_event_ids=sorted(item["id"] for item in snapshot["conditions"]),
            route_edge_ids=case["gold_derivation"]["oracle_result"]["route_edge_ids"],
            tool_trace=[
                {"tool_name": "list_road_conditions", "argument_keys": [],
                 "outcome": "success", "artifact_kind": None},
                route_call,
            ],
        )
        rows.append({"scenario_id": case["id"], "attempt": 1,
                     "snapshot_id": dataset["snapshot_id"],
                     "input_messages": case["messages"], "turns": [turn]})
    return rows


def _write_replay(tmp_path, rows):
    replay = tmp_path / "projected.jsonl"
    replay.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                              for row in rows), encoding="utf-8")
    return replay


def test_v3_runner_scores_no_route_evidence_and_records_sources(tmp_path):
    replay = _write_replay(tmp_path, _projected_rows())
    result = run_controlled_event_replay_v3(
        replay, tmp_path / "runs", "projected",
        observation_origin="synthetic_projection",
    )
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert summary["projection_score_pass_count"] == 2
    assert summary["agent_result_verified"] is False
    assert summary["observation_origin"] == "synthetic_projection"
    assert summary["formal_ready"] is False
    assert manifest["campus_truth"] is False
    assert manifest["source_snapshot_id"]
    assert manifest["base_source_snapshot_id"]
    assert manifest["base_source_manifest_sha256"]
    assert manifest["runner_sha256"]
    assert manifest["runtime_source_tree_sha256"]
    assert manifest["event_guard_sha256"]
    with pytest.raises(FileExistsError):
        run_controlled_event_replay_v3(
            replay, tmp_path / "runs", "projected",
            observation_origin="synthetic_projection",
        )


def test_v3_runner_rejects_false_no_route_and_unprojected_fields(tmp_path):
    rows = _projected_rows()
    failed = next(row for row in rows if row["turns"][0]["tool_trace"][1]
                  .get("error_code") == "graph_no_path")
    failed["turns"][0]["tool_trace"][1]["error_code"] = "invalid_args"
    with pytest.raises(ValueError, match="error_code"):
        run_controlled_event_replay_v3(
            _write_replay(tmp_path, rows), tmp_path / "runs", "bad",
            observation_origin="synthetic_projection",
        )
    failed["turns"][0]["tool_trace"][1]["error_code"] = "graph_no_path"
    failed["turns"][0]["tool_trace"][1]["raw_error"] = "private payload"
    with pytest.raises(ValueError, match="unprojected"):
        run_controlled_event_replay_v3(
            _write_replay(tmp_path, rows), tmp_path / "runs", "bad2",
            observation_origin="synthetic_projection",
        )


def test_v3_runner_requires_honest_origin_label(tmp_path):
    replay = _write_replay(tmp_path, _projected_rows())
    with pytest.raises(ValueError, match="observation_origin"):
        run_controlled_event_replay_v3(
            replay, tmp_path / "runs", "fake_agent",
            observation_origin="agent_collector",
        )


def test_v3_runner_does_not_score_no_path_for_different_od(tmp_path):
    rows = _projected_rows()
    failed = next(row for row in rows if row["turns"][0]["tool_trace"][1]
                  .get("error_code") == "graph_no_path")
    failed["turns"][0]["tool_trace"][1]["request_digest"] = "0" * 64
    result = run_controlled_event_replay_v3(
        _write_replay(tmp_path, rows), tmp_path / "runs", "wrong_od",
        observation_origin="synthetic_projection",
    )
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["projection_score_pass_count"] == 1
    assert summary["failure_counts"]["graph_no_path_evidence_missing"] == 1


def test_v3_runner_rechecks_base_builder_and_adapter_inventory(tmp_path, monkeypatch):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    source = json.loads((ROOT / "experiments/datasets/controlled_event_pilot_v2/"
                         "source_manifest_event_pilot_dev_v1.json")
                        .read_text(encoding="utf-8"))
    source["inputs"]["dataset_builder"]["sha256"] = "0" * 64
    (base_dir / "source_manifest_event_pilot_dev_v1.json").write_text(
        json.dumps(source), encoding="utf-8",
    )
    monkeypatch.setattr(replay_module, "BASE_PILOT", base_dir)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        run_controlled_event_replay_v3(
            _write_replay(tmp_path, _projected_rows()), tmp_path / "runs",
            "bad_base", observation_origin="synthetic_projection",
        )
