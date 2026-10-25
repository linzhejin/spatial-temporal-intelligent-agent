"""Controlled route observations are judged from frozen edge evidence."""

from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import pytest

from experiments.controlled_route_cases import make_route_case
from experiments.controlled_route_replay import (
    run_controlled_route_replay, score_controlled_route_case,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/datasets/controlled_route_pilot_v1"


def _fixture(*, disconnected=False):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=5)
    if not disconnected:
        graph.add_edge(2, 3, key=0, highway="footway", length=5)
        graph.add_edge(1, 4, key=0, highway="footway", length=10)
        graph.add_edge(4, 3, key=0, highway="footway", length=10)
    anchors = {
        "a": {"poi_id": "a", "name": "甲", "node_id": 1,
              "access_status": "last_meter_unverified"},
        "b": {"poi_id": "b", "name": "乙", "node_id": 3,
              "access_status": "last_meter_unverified"},
    }
    graph.add_node(3)
    permitted = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    case = make_route_case(
        graph, anchors, permitted_edge_ids=permitted, unknown_edge_ids=set(),
        closed_edge_ids=set(), templates={"direct": "从{start}到{end}，{mode_text}{constraint_text}。"},
        template_id="direct", case_id="case_1", category="mode_access",
        template_group="direct", od_group="a_b", snapshot_id="a" * 64,
        generator_sha256="b" * 64, template_sha256="c" * 64,
        parameters={"start_poi_id": "a", "end_poi_id": "b",
                    "via_poi_ids": [], "mode": "walk", "avoid_steps": False},
    )
    turn = case["expected_turns"][0]
    observation = {
        "scenario_id": case["id"], "attempt": 1,
        "snapshot_id": case["snapshot_id"], "input_messages": case["messages"],
        "turns": [{
            "status": turn["status"], "kind": turn["kind"],
            "start_poi_id": "a", "end_poi_id": "b", "via_poi_ids": [],
            "travel_mode": "walk", "strategy": "shortest", "hard_constraints": {},
            "candidate_poi_ids": [], "clarification_slot": None,
            "outputs": turn["required_outputs"],
            "route_edge_ids": [[1, 2, 0], [2, 3, 0]] if not disconnected else [],
            "unknowns": [],
        }],
        "transport_error": None,
    }
    return graph, anchors, permitted, case, observation


def _score(graph, anchors, permitted, case, observation):
    return score_controlled_route_case(
        case, observation, graph, anchors,
        permitted_edge_ids=permitted, unknown_edge_ids=set(),
        closed_edge_ids=set(),
    )


def test_graph_scoring_accepts_shortest_path_and_rejects_valid_detour():
    graph, anchors, permitted, case, observation = _fixture()
    assert _score(graph, anchors, permitted, case, observation)["task_success"] is True
    longer = deepcopy(observation)
    longer["turns"][0]["route_edge_ids"] = [[1, 4, 0], [4, 3, 0]]
    verdict = _score(graph, anchors, permitted, case, longer)
    assert verdict["task_success"] is False
    assert "not_shortest" in verdict["failures"]


def test_product_claim_cannot_override_invalid_or_missing_edges():
    graph, anchors, permitted, case, observation = _fixture()
    bad = deepcopy(observation)
    bad["turns"][0]["route_edge_ids"] = [[1, 2, 0], [4, 3, 0]]
    bad["turns"][0]["independent_review"] = {
        "source": "independent_reviewer", "reviewer": "product",
        "evidence": ["claimed pass"], "route": "pass",
    }
    verdict = _score(graph, anchors, permitted, case, bad)
    assert verdict["task_success"] is False
    assert "discontinuous_edges" in verdict["route_issues"]
    missing = deepcopy(observation)
    missing["turns"][0]["route_edge_ids"] = []
    assert "route_edges_missing" in _score(graph, anchors, permitted, case, missing)["failures"]


def test_infeasible_case_requires_explicit_no_route_and_explanation():
    graph, anchors, permitted, case, observation = _fixture(disconnected=True)
    assert case["gold_derivation"]["oracle_result"]["verdict"] == "infeasible"
    assert _score(graph, anchors, permitted, case, observation)["task_success"] is True
    missing = deepcopy(observation)
    missing["turns"][0]["outputs"] = []
    assert "missing_output" in _score(graph, anchors, permitted, case, missing)["failures"]
    fake = deepcopy(observation)
    fake["turns"][0]["route_edge_ids"] = [[1, 2, 0]]
    assert "unexpected_route_edges" in _score(graph, anchors, permitted, case, fake)["failures"]


def test_snapshot_or_prompt_drift_is_failure_even_with_valid_edges():
    graph, anchors, permitted, case, observation = _fixture()
    altered = deepcopy(observation)
    altered["snapshot_id"] = "d" * 64
    altered["input_messages"] = [{"role": "user", "content": "different"}]
    verdict = _score(graph, anchors, permitted, case, altered)
    assert verdict["task_success"] is False
    assert {"snapshot_mismatch", "input_mismatch"} <= set(verdict["failures"])


def _gold_observation(case):
    expected = case["expected_turns"][0]
    return {
        "scenario_id": case["id"], "attempt": 1,
        "snapshot_id": case["snapshot_id"], "input_messages": case["messages"],
        "turns": [{
            "status": expected["status"], "kind": expected["kind"],
            "start_poi_id": expected["start_poi_id"],
            "end_poi_id": expected["end_poi_id"],
            "via_poi_ids": expected["via_poi_ids"],
            "travel_mode": expected["travel_mode"],
            "strategy": expected["strategy"],
            "hard_constraints": expected["hard_constraints"],
            "candidate_poi_ids": [], "clarification_slot": None,
            "outputs": expected["required_outputs"],
            "route_edge_ids": case["gold_derivation"]["oracle_result"]["route_edge_ids"],
            "unknowns": [],
        }],
    }


def test_frozen_pilot_replay_counts_missing_attempt_as_failure(tmp_path):
    dataset = json.loads((PILOT / "scenarios_route_pilot_dev_v2.json").read_text(encoding="utf-8"))
    rows = [_gold_observation(case) for case in dataset["cases"][:-1]]
    replay = tmp_path / "observations.jsonl"
    replay.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = run_controlled_route_replay(
        replay, tmp_path / "results", "fixture_1", root=ROOT,
        dataset_path=PILOT / "scenarios_route_pilot_dev_v2.json",
        manifest_path=PILOT / "source_manifest_route_pilot_dev_v1.json",
        spec_path=PILOT / "route_pilot_spec.json",
    )
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 6
    assert summary["task_success_count"] == 5
    assert summary["failure_counts"]["missing_observation"] == 1
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_type"] == "development_diagnostic"
    assert manifest["formal_ready"] is False
    assert manifest["scorer_content_encoding"] == "utf8_lf_canonical_v1"
    with pytest.raises(FileExistsError):
        run_controlled_route_replay(
            replay, tmp_path / "results", "fixture_1", root=ROOT,
            dataset_path=PILOT / "scenarios_route_pilot_dev_v2.json",
            manifest_path=PILOT / "source_manifest_route_pilot_dev_v1.json",
            spec_path=PILOT / "route_pilot_spec.json",
        )


def test_frozen_pilot_replay_rejects_unregistered_observation(tmp_path):
    replay = tmp_path / "observations.jsonl"
    replay.write_text(json.dumps({"scenario_id": "not_a_case", "attempt": 1,
                                  "turns": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown scenario"):
        run_controlled_route_replay(
            replay, tmp_path / "results", "fixture_2", root=ROOT,
            dataset_path=PILOT / "scenarios_route_pilot_dev_v2.json",
            manifest_path=PILOT / "source_manifest_route_pilot_dev_v1.json",
            spec_path=PILOT / "route_pilot_spec.json",
        )


def test_replay_rejects_noninteger_attempt_before_scoring(tmp_path):
    replay = tmp_path / "observations.jsonl"
    replay.write_text(json.dumps({"scenario_id": "route_dev_direct_001",
                                  "attempt": True, "turns": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid scenario or attempt"):
        run_controlled_route_replay(
            replay, tmp_path / "results", "fixture_3", root=ROOT,
            dataset_path=PILOT / "scenarios_route_pilot_dev_v2.json",
            manifest_path=PILOT / "source_manifest_route_pilot_dev_v1.json",
            spec_path=PILOT / "route_pilot_spec.json",
        )
