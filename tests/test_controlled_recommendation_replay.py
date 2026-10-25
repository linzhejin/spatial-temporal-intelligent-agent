"""Independent scoring must not accept a product's own recommendation claim."""

from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import pytest

from experiments.controlled_recommendation_cases import make_recommendation_case
from experiments.controlled_recommendation_replay import (
    run_controlled_recommendation_replay, score_controlled_recommendation_case,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/datasets/controlled_recommendation_pilot_v2"


def _fixture():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(1, 3, key=0, highway="footway", length=20)
    anchors = {
        key: {"poi_id": key, "name": name, "node_id": node,
              "snap_distance_m": 1, "access_status": "last_meter_unverified"}
        for key, name, node in (("start", "出发点", 1), ("near", "近食堂", 2),
                                ("far", "远食堂", 3))
    }
    pois = [
        {"id": key, "activities": ["canteen_meal"] if key != "start" else [],
         "activity_review_status": "source_supported", "activity_basis": "category=canteen"}
        for key in anchors
    ]
    permitted = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    case = make_recommendation_case(
        graph, pois, anchors, permitted_edge_ids=permitted,
        unknown_edge_ids=set(), closed_edge_ids=set(),
        templates={"meal": "从{start}出发，在已收录且可接入校园路网的地点里，按路网距离由近到远推荐{limit}个{activity_label}。"},
        template_id="meal", case_id="rec_dev_1", template_group="meal_dev",
        od_group="start_meal", snapshot_id="a" * 64,
        generator_sha256="b" * 64, template_sha256="c" * 64,
        parameters={"start_poi_id": "start", "activity": "canteen_meal",
                    "activity_label": "食堂", "mode": "walk", "limit": 2},
    )
    turn = case["expected_turns"][0]
    observation = {
        "scenario_id": case["id"], "attempt": 1,
        "snapshot_id": case["snapshot_id"], "input_messages": case["messages"],
        "turns": [{
            "status": "completed", "kind": "candidates",
            "start_poi_id": "start", "end_poi_id": None,
            "via_poi_ids": [], "travel_mode": "walk", "strategy": None,
            "hard_constraints": {}, "candidate_poi_ids": turn["candidate_poi_ids"],
            "clarification_slot": None, "outputs": ["candidates"],
            "route_edge_ids": [], "unknowns": [],
        }],
    }
    return graph, anchors, pois, permitted, case, observation


def _score(graph, anchors, pois, permitted, case, observed):
    return score_controlled_recommendation_case(
        case, observed, graph, anchors, pois,
        permitted_edge_ids=permitted, unknown_edge_ids=set(),
        closed_edge_ids=set(),
    )


def test_recommendation_replay_accepts_order_and_rejects_reversed_or_missing():
    graph, anchors, pois, permitted, case, observed = _fixture()
    assert _score(graph, anchors, pois, permitted, case, observed)["task_success"] is True
    reversed_result = deepcopy(observed)
    reversed_result["turns"][0]["candidate_poi_ids"].reverse()
    reversed_result["turns"][0]["independent_review"] = {"route": "pass"}
    verdict = _score(graph, anchors, pois, permitted, case, reversed_result)
    assert verdict["task_success"] is False
    assert "wrong_candidate_order" in verdict["failures"]
    assert "missing_observation" in _score(
        graph, anchors, pois, permitted, case, None,
    )["failures"]


def test_recommendation_replay_marks_distance_metric_disagreement_unknown():
    graph, anchors, pois, permitted, case, observed = _fixture()
    anchors["near"]["snap_distance_m"] = 100
    verdict = _score(graph, anchors, pois, permitted, case, observed)
    assert verdict["task_success"] is False
    assert "distance_metric_order_ambiguous" in verdict["unknowns"]
    assert "wrong_candidate_order" not in verdict["failures"]


def test_recommendation_replay_rejects_changed_input_even_with_correct_order():
    graph, anchors, pois, permitted, case, observed = _fixture()
    changed = deepcopy(observed)
    changed["input_messages"] = [{"role": "user", "content": "different"}]
    assert "input_mismatch" in _score(
        graph, anchors, pois, permitted, case, changed,
    )["failures"]


def test_frozen_recommendation_replay_counts_missing_attempts(tmp_path):
    replay = tmp_path / "empty.jsonl"
    replay.write_text("", encoding="utf-8")
    result_dir = run_controlled_recommendation_replay(
        replay, tmp_path / "results", "missing_fixture", root=ROOT,
    )
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 12
    assert summary["failure_counts"] == {"missing_observation": 12}
    assert summary["task_success_count"] == 0
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["formal_ready"] is False
    with pytest.raises(FileExistsError):
        run_controlled_recommendation_replay(
            replay, tmp_path / "results", "missing_fixture", root=ROOT,
        )


def test_recommendation_replay_rejects_changed_frozen_packet_before_scoring(tmp_path):
    dataset = json.loads((PILOT / "scenarios_recommendation_dev_v2.json").read_text(encoding="utf-8"))
    dataset["cases"][0]["messages"][0]["content"] = "tampered"
    dataset_path = tmp_path / "changed.json"
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    replay = tmp_path / "empty.jsonl"
    replay.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="packet differs"):
        run_controlled_recommendation_replay(
            replay, tmp_path / "results", "changed_fixture", root=ROOT,
            dataset_path=dataset_path,
        )
