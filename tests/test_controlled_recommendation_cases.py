"""Recommendation cases keep prompt, rank, and source evidence aligned."""

from copy import deepcopy

import networkx as nx
import pytest

from experiments.controlled_gold_contract import validate_controlled_dataset
from experiments.controlled_recommendation_cases import (
    make_recommendation_case, replay_recommendation_case,
)


def _fixture():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(1, 3, key=0, highway="footway", length=20)
    anchors = {
        key: {"poi_id": key, "name": name, "node_id": node,
              "access_status": "last_meter_unverified"}
        for key, name, node in (("start", "起点", 1), ("near", "近食堂", 2),
                                ("far", "远食堂", 3))
    }
    pois = [
        {"id": key, "name": anchor["name"],
         "activities": ["canteen_meal"] if key != "start" else [],
         "activity_review_status": "source_supported",
         "activity_basis": "category=canteen"}
        for key, anchor in anchors.items()
    ]
    evidence = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    template = ("从{start}步行出发，在已收录且可接入校园路网的地点里，"
                "按路网距离由近到远推荐{limit}个{activity_label}。")
    parameters = {"start_poi_id": "start", "activity": "canteen_meal",
                  "activity_label": "食堂", "mode": "walk", "limit": 2}
    return graph, pois, anchors, evidence, template, parameters


def _case(graph, pois, anchors, evidence, template, parameters):
    return make_recommendation_case(
        graph, pois, anchors, permitted_edge_ids=evidence,
        unknown_edge_ids=set(), closed_edge_ids=set(),
        templates={"dining": template}, template_id="dining",
        case_id="recommend_1", template_group="dining_dev",
        od_group="start_canteen", snapshot_id="a" * 64,
        generator_sha256="b" * 64, template_sha256="c" * 64,
        parameters=parameters,
    )


def test_recommendation_case_ranks_all_reviewed_places_and_passes_contract():
    graph, pois, anchors, evidence, template, params = _fixture()
    case = _case(graph, pois, anchors, evidence, template, params)
    assert case["category"] == "place_recommendation"
    assert case["expected_turns"][0]["candidate_poi_ids"] == ["near", "far"]
    assert case["messages"][0]["content"].endswith("推荐2个食堂。")
    assert {item["kind"] for item in case["adjudication"]["evidence"]} == {
        "template_parameter", "catalog_attribute", "independent_graph_oracle",
    }
    dataset = {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
               "split": "development", "repeats": 1,
               "source_manifest_sha256": "d" * 64, "snapshot_id": "a" * 64,
               "cases": [case]}
    validate_controlled_dataset(dataset)
    assert replay_recommendation_case(
        case, graph, pois, anchors, permitted_edge_ids=evidence,
        unknown_edge_ids=set(), closed_edge_ids=set(),
        templates={"dining": template},
    ) == case


def test_replay_detects_changed_rank_or_prompt():
    graph, pois, anchors, evidence, template, params = _fixture()
    case = _case(graph, pois, anchors, evidence, template, params)
    changed = deepcopy(case)
    changed["expected_turns"][0]["candidate_poi_ids"].reverse()
    with pytest.raises(ValueError, match="recommendation case differs"):
        replay_recommendation_case(
            changed, graph, pois, anchors, permitted_edge_ids=evidence,
            unknown_edge_ids=set(), closed_edge_ids=set(),
            templates={"dining": template},
        )


def test_template_must_expose_catalog_scope_and_ordering():
    graph, pois, anchors, evidence, template, params = _fixture()
    with pytest.raises(ValueError, match="template"):
        _case(graph, pois, anchors, evidence,
              "从{start}出发找{activity_label}。", params)
