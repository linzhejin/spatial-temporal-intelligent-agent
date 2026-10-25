"""Recommendation pilot must be reproducible from its own frozen sources."""

from copy import deepcopy

import networkx as nx
import pytest

from experiments.build_controlled_recommendation_pilot import (
    build_recommendation_pilot_dataset, verify_recommendation_pilot_dataset,
)
from experiments.controlled_recommendation_manifest import (
    RECOMMENDATION_INPUTS, make_recommendation_manifest,
    verify_recommendation_manifest,
)


def _manifest(tmp_path):
    paths = {}
    for name in RECOMMENDATION_INPUTS:
        path = tmp_path / f"{name}.txt"
        path.write_text(name + "\n", encoding="utf-8")
        paths[name] = path.name
    return make_recommendation_manifest(tmp_path, paths)


def test_recommendation_manifest_detects_source_drift(tmp_path):
    manifest = _manifest(tmp_path)
    assert verify_recommendation_manifest(manifest, tmp_path) == manifest["effective_inputs_sha256"]
    (tmp_path / "poi.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="poi.*mismatch"):
        verify_recommendation_manifest(manifest, tmp_path)


def _sources(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(1, 3, key=0, highway="footway", length=20)
    anchors = {
        key: {"poi_id": key, "name": name, "node_id": node,
              "access_status": "last_meter_unverified"}
        for key, name, node in (("start", "出发点", 1), ("near", "近食堂", 2),
                                ("far", "远食堂", 3))
    }
    pois = [
        {"id": key, "activities": ["canteen_meal"] if key != "start" else [],
         "activity_review_status": "source_supported", "activity_basis": "category=canteen"}
        for key in anchors
    ]
    edges = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    spec = {
        "format": "controlled-recommendation-pilot-spec-v1",
        "scope": "development_only",
        "templates": {"meal": {"template_group": "meal_dev",
                  "text": "从{start}步行出发，在已收录且可接入校园路网的地点里，按路网距离由近到远推荐{limit}个{activity_label}。"}},
        "cases": [{"id": "rec_dev_1", "template_id": "meal", "od_group": "start_meal",
                   "parameters": {"start_poi_id": "start", "activity": "canteen_meal",
                                  "activity_label": "食堂", "mode": "walk", "limit": 2}}],
    }
    return graph, anchors, pois, edges, spec, _manifest(tmp_path)


def test_recommendation_packet_replays_full_gold_and_rejects_changed_case(tmp_path):
    graph, anchors, pois, edges, spec, manifest = _sources(tmp_path)
    packet = build_recommendation_pilot_dataset(
        graph, anchors, pois, spec, manifest,
        permitted_edge_ids=edges, unknown_edge_ids=set(),
    )
    assert packet["cases"][0]["expected_turns"][0]["candidate_poi_ids"] == ["near", "far"]
    verify_recommendation_pilot_dataset(
        packet, graph, anchors, pois, spec, manifest,
        permitted_edge_ids=edges, unknown_edge_ids=set(),
    )
    changed = deepcopy(packet)
    changed["cases"][0]["template_group"] = "tampered"
    with pytest.raises(ValueError, match="packet differs"):
        verify_recommendation_pilot_dataset(
            changed, graph, anchors, pois, spec, manifest,
            permitted_edge_ids=edges, unknown_edge_ids=set(),
        )
