"""A revision must preserve the prior origin and hard constraints."""

from copy import deepcopy

import networkx as nx
import pytest

from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_multi_turn_cases import (
    make_multi_turn_route_case, replay_multi_turn_route_case,
    score_controlled_multi_turn_case,
)


def _fixture():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="residential", length=10)
    graph.add_edge(1, 3, key=0, highway="residential", length=20)
    anchors = {
        poi_id: {"poi_id": poi_id, "name": name, "node_id": node,
                 "access_status": "last_meter_unverified"}
        for poi_id, name, node in (("start", "起点", 1), ("old", "旧终点", 2),
                                   ("new", "新终点", 3))
    }
    permitted = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    evidence = {mode: {"permitted": permitted, "unknown": set()}
                for mode in ("walk", "bike")}
    templates = {
        "first": "从{start}{initial_mode_text}到{old_end}，按最短路径，{constraint_text}。",
        "revision": "改成{revised_mode_text}，终点换成{new_end}；起点、原有约束和最短路径要求保持不变。",
    }
    params = {"start_poi_id": "start", "initial_end_poi_id": "old",
              "revised_end_poi_id": "new", "initial_mode": "walk",
              "revised_mode": "bike", "avoid_steps": True}
    case = make_multi_turn_route_case(
        graph, anchors, evidence, templates=templates, case_id="revision_dev_1",
        template_group="route_revision", od_group="start_old_new",
        snapshot_id="a" * 64, generator_sha256="b" * 64,
        template_sha256="c" * 64, parameters=params,
    )
    observation = {
        "scenario_id": case["id"], "attempt": 1,
        "snapshot_id": case["snapshot_id"], "input_messages": case["messages"],
        "turns": [
            {**turn, "outputs": turn["required_outputs"],
             "route_edge_ids": case["gold_derivation"]["oracle_result"]["turns"][index]["route_edge_ids"],
             "unknowns": []}
            for index, turn in enumerate(case["expected_turns"])
        ],
    }
    return graph, anchors, evidence, templates, case, observation


def test_multi_turn_gold_replays_and_requires_kept_origin_constraint():
    graph, anchors, evidence, templates, case, observation = _fixture()
    assert len(case["messages"]) == len(case["expected_turns"]) == 2
    assert case["expected_turns"][1]["start_poi_id"] == "start"
    assert case["expected_turns"][1]["hard_constraints"] == {"avoid_steps": True}
    replay_multi_turn_route_case(case, graph, anchors, evidence, templates=templates)
    verdict = score_controlled_multi_turn_case(case, observation, graph, anchors, evidence)
    assert verdict["task_success"] is True
    changed = deepcopy(observation)
    changed["turns"][1]["start_poi_id"] = "old"
    changed["turns"][1]["hard_constraints"] = {}
    verdict = score_controlled_multi_turn_case(case, changed, graph, anchors, evidence)
    assert verdict["task_success"] is False
    assert {"wrong_poi", "dropped_constraint"} <= set(verdict["failures"])


def test_multi_turn_gold_rejects_changed_revision_and_missing_second_turn():
    graph, anchors, evidence, templates, case, observation = _fixture()
    packet = {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
              "split": "development", "repeats": 1,
              "source_manifest_sha256": "d" * 64, "snapshot_id": "a" * 64,
              "cases": [case]}
    validate_controlled_dataset(packet)
    changed_case = deepcopy(case)
    changed_case["messages"][1]["content"] = "换个地方"
    with pytest.raises(ValueError, match="differs"):
        replay_multi_turn_route_case(
            changed_case, graph, anchors, evidence, templates=templates,
        )
    missing = deepcopy(observation)
    missing["turns"].pop()
    assert "turn_count_mismatch" in score_controlled_multi_turn_case(
        case, missing, graph, anchors, evidence,
    )["failures"]
