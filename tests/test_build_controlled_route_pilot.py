"""A route development packet is regenerated from its frozen case specification."""

from copy import deepcopy

import networkx as nx
import pytest

from experiments.build_controlled_route_pilot import (
    build_route_pilot_dataset, verify_route_pilot_dataset,
)
from experiments.controlled_gold_contract import validate_controlled_dataset


def _fixture():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(2, 3, key=0, highway="footway", length=5)
    anchors = {
        "a": {"poi_id": "a", "name": "甲门", "node_id": 1,
              "access_status": "last_meter_unverified"},
        "b": {"poi_id": "b", "name": "乙楼", "node_id": 2,
              "access_status": "last_meter_unverified"},
        "c": {"poi_id": "c", "name": "丙馆", "node_id": 3,
              "access_status": "last_meter_unverified"},
    }
    spec = {
        "format": "controlled-route-pilot-spec-v1", "scope": "development_only",
        "templates": {
            "direct": {"category": "mode_access", "template_group": "direct_dev",
                       "text": "从{start}到{end}，{mode_text}{constraint_text}。"},
            "via": {"category": "multi_via_itinerary", "template_group": "via_dev",
                    "text": "从{start}出发，先经过{vias}，再到{end}，{mode_text}{constraint_text}。"},
        },
        "cases": [
            {"id": "direct_1", "template_id": "direct", "od_group": "a_c",
             "parameters": {"start_poi_id": "a", "end_poi_id": "c",
                            "via_poi_ids": [], "mode": "walk", "avoid_steps": False}},
            {"id": "via_1", "template_id": "via", "od_group": "a_b_c",
             "parameters": {"start_poi_id": "a", "end_poi_id": "c",
                            "via_poi_ids": ["b"], "mode": "walk", "avoid_steps": True}},
        ],
    }
    manifest = {"profile": "route_pilot", "effective_inputs_sha256": "a" * 64,
                "inputs": {"generator": {"sha256": "b" * 64},
                           "templates": {"sha256": "c" * 64}}}
    evidence = {"walk": {"permitted": {(1, 2, 0), (2, 3, 0)}, "unknown": set()}}
    return graph, anchors, spec, manifest, evidence


def test_dev_packet_has_two_replayable_cases_and_passes_contract():
    graph, anchors, spec, manifest, evidence = _fixture()
    dataset = build_route_pilot_dataset(graph, anchors, spec, manifest,
                                        "d" * 64, evidence)
    assert dataset["split"] == "development"
    assert dataset["repeats"] == 1
    assert [case["category"] for case in dataset["cases"]] == [
        "mode_access", "multi_via_itinerary",
    ]
    validate_controlled_dataset(dataset)
    verify_route_pilot_dataset(dataset, graph, anchors, spec, manifest,
                               "d" * 64, evidence)


def test_packet_replay_rejects_changed_group_or_answer():
    graph, anchors, spec, manifest, evidence = _fixture()
    dataset = build_route_pilot_dataset(graph, anchors, spec, manifest,
                                        "d" * 64, evidence)
    changed = deepcopy(dataset)
    changed["cases"][0]["template_group"] = "fake_group"
    with pytest.raises(ValueError, match="case specification"):
        verify_route_pilot_dataset(changed, graph, anchors, spec, manifest,
                                   "d" * 64, evidence)
    changed = deepcopy(dataset)
    changed["cases"][1]["expected_turns"][0]["via_poi_ids"] = []
    with pytest.raises(ValueError, match="case specification"):
        verify_route_pilot_dataset(changed, graph, anchors, spec, manifest,
                                   "d" * 64, evidence)


def test_unknown_or_duplicate_spec_case_fails_closed():
    graph, anchors, spec, manifest, evidence = _fixture()
    spec["cases"][1]["id"] = spec["cases"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        build_route_pilot_dataset(graph, anchors, spec, manifest,
                                  "d" * 64, evidence)
    spec["cases"][1]["id"] = "via_1"
    evidence["walk"]["unknown"] = {(2, 3, 0)}
    evidence["walk"]["permitted"].remove((2, 3, 0))
    with pytest.raises(ValueError, match="unknown"):
        build_route_pilot_dataset(graph, anchors, spec, manifest,
                                  "d" * 64, evidence)
