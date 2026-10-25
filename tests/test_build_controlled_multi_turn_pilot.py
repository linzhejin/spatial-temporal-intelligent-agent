"""The development packet must replay both turns from frozen inputs."""

from copy import deepcopy

import networkx as nx
import pytest

from experiments.build_controlled_multi_turn_pilot import (
    build_multi_turn_pilot_dataset, verify_multi_turn_pilot_dataset,
)
from experiments.controlled_gold_contract import validate_controlled_dataset


def _fixture():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="residential", length=10)
    graph.add_edge(1, 3, key=0, highway="residential", length=20)
    anchors = {poi_id: {"poi_id": poi_id, "name": name, "node_id": node,
                        "access_status": "last_meter_unverified"}
               for poi_id, name, node in (("start", "起点", 1),
                                          ("old", "旧终点", 2),
                                          ("new", "新终点", 3))}
    permitted = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    evidence = {mode: {"permitted": permitted, "unknown": set()}
                for mode in ("walk", "bike")}
    spec = {
        "format": "controlled-multi-turn-pilot-spec-v2", "scope": "development_only",
        "templates": {
            "first": "从{start}{initial_mode_text}到{old_end}，按最短路径，{constraint_text}。",
            "revision": "改成{revised_mode_text}，终点换成{new_end}；起点、原有约束和最短路径要求保持不变。",
        },
        "cases": [{"id": "revision_1", "od_group": "start_old_new",
                   "parameters": {"start_poi_id": "start", "initial_end_poi_id": "old",
                                  "revised_end_poi_id": "new", "initial_mode": "walk",
                                  "revised_mode": "bike", "avoid_steps": True}}],
    }
    manifest = {"profile": "multi_turn_pilot", "effective_inputs_sha256": "a" * 64,
                "inputs": {"generator": {"sha256": "b" * 64},
                           "templates": {"sha256": "c" * 64}}}
    return graph, anchors, evidence, spec, manifest


def test_multi_turn_packet_rebuilds_and_passes_contract():
    graph, anchors, evidence, spec, manifest = _fixture()
    packet = build_multi_turn_pilot_dataset(
        graph, anchors, spec, manifest, "d" * 64, evidence,
    )
    assert packet["split"] == "development"
    assert packet["repeats"] == 1
    assert len(packet["cases"][0]["messages"]) == 2
    validate_controlled_dataset(packet)
    verify_multi_turn_pilot_dataset(
        packet, graph, anchors, spec, manifest, "d" * 64, evidence,
    )


def test_multi_turn_packet_rejects_modified_revision_and_duplicate_ids():
    graph, anchors, evidence, spec, manifest = _fixture()
    packet = build_multi_turn_pilot_dataset(
        graph, anchors, spec, manifest, "d" * 64, evidence,
    )
    tampered = deepcopy(packet)
    tampered["cases"][0]["expected_turns"][1]["start_poi_id"] = "old"
    with pytest.raises(ValueError, match="differs"):
        verify_multi_turn_pilot_dataset(
            tampered, graph, anchors, spec, manifest, "d" * 64, evidence,
        )
    duplicated = deepcopy(spec)
    duplicated["cases"].append(deepcopy(duplicated["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_multi_turn_pilot_dataset(
            graph, anchors, duplicated, manifest, "d" * 64, evidence,
        )
