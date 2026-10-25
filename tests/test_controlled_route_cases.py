"""Route pilot cases must be source-derived and replayable before model calls."""

import copy

import networkx as nx
import pytest

from experiments.controlled_route_cases import make_route_case, replay_route_case
from experiments.controlled_gold_contract import validate_controlled_dataset


TEMPLATES = {
    "direct": "从{start}到{end}，{mode_text}{constraint_text}。",
    "ordered": "从{start}出发，先经过{vias}，再到{end}，{mode_text}{constraint_text}。",
}


def _inputs():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(2, 3, key=0, highway="footway", length=20)
    graph.add_edge(1, 3, key=0, highway="footway", length=15)
    anchors = {"start": {"poi_id": "start", "name": "甲门", "node_id": 1,
                         "access_status": "last_meter_unverified"},
               "via": {"poi_id": "via", "name": "乙楼", "node_id": 2,
                       "access_status": "last_meter_unverified"},
               "end": {"poi_id": "end", "name": "丙馆", "node_id": 3,
                       "access_status": "last_meter_unverified"}}
    evidence = {(1, 2, 0), (2, 3, 0), (1, 3, 0)}
    return graph, anchors, evidence


def _case(**overrides):
    graph, anchors, evidence = _inputs()
    parameters = {"start_poi_id": "start", "end_poi_id": "end",
                  "via_poi_ids": [], "mode": "walk", "avoid_steps": False}
    parameters.update(overrides.pop("parameters", {}))
    case = make_route_case(
        graph, anchors, permitted_edge_ids=evidence, unknown_edge_ids=set(),
        closed_edge_ids=set(), templates=TEMPLATES,
        template_id=overrides.pop("template_id", "direct"),
        case_id="dev_route_1", category=overrides.pop("category", "mode_access"),
        template_group="direct_pilot",
        od_group="start_end", snapshot_id="a" * 64,
        generator_sha256="b" * 64, template_sha256="c" * 64,
        parameters=parameters, **overrides,
    )
    return case, graph, anchors, evidence


def test_direct_case_passes_v2_contract_and_replays_from_sources():
    case, graph, anchors, evidence = _case()
    assert case["messages"] == [{"role": "user", "content": "从甲门到丙馆，步行。"}]
    assert case["expected_turns"][0]["feasibility"] == "feasible"
    assert case["gold_derivation"]["oracle_result"]["route_edge_ids"] == [[1, 3, 0]]
    dataset = {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
               "split": "development", "repeats": 1, "source_manifest_sha256": "d" * 64,
               "snapshot_id": "a" * 64, "cases": [case]}
    validate_controlled_dataset(dataset)
    replay_route_case(case, graph, anchors, permitted_edge_ids=evidence,
                      unknown_edge_ids=set(), closed_edge_ids=set(), templates=TEMPLATES)


def test_ordered_via_is_preserved_in_prompt_and_oracle():
    case, graph, anchors, evidence = _case(
        template_id="ordered", category="multi_via_itinerary",
        parameters={"via_poi_ids": ["via"], "avoid_steps": True},
    )
    assert "先经过乙楼，再到丙馆" in case["messages"][0]["content"]
    assert "不要走台阶" in case["messages"][0]["content"]
    assert case["expected_turns"][0]["via_poi_ids"] == ["via"]
    assert case["gold_derivation"]["oracle_result"]["route_edge_ids"] == [
        [1, 2, 0], [2, 3, 0],
    ]
    replay_route_case(case, graph, anchors, permitted_edge_ids=evidence,
                      unknown_edge_ids=set(), closed_edge_ids=set(), templates=TEMPLATES)


def test_two_vias_keep_full_order_and_all_route_legs():
    graph, anchors, evidence = _inputs()
    graph.add_edge(3, 4, key=0, highway="footway", length=7)
    anchors["via2"] = {"poi_id": "via2", "name": "丙馆", "node_id": 3,
                       "access_status": "last_meter_unverified"}
    anchors["end2"] = {"poi_id": "end2", "name": "丁门", "node_id": 4,
                      "access_status": "last_meter_unverified"}
    evidence.add((3, 4, 0))
    case = make_route_case(
        graph, anchors, permitted_edge_ids=evidence, unknown_edge_ids=set(),
        closed_edge_ids=set(), templates=TEMPLATES, template_id="ordered",
        case_id="dev_two_via", category="multi_via_itinerary",
        template_group="ordered_pilot", od_group="start_via_via2_end2",
        snapshot_id="a" * 64, generator_sha256="b" * 64,
        template_sha256="c" * 64,
        parameters={"start_poi_id": "start", "end_poi_id": "end2",
                    "via_poi_ids": ["via", "via2"], "mode": "walk", "avoid_steps": False},
    )
    assert "先经过乙楼、丙馆，再到丁门" in case["messages"][0]["content"]
    assert case["gold_derivation"]["oracle_result"]["route_edge_ids"] == [
        [1, 2, 0], [2, 3, 0], [3, 4, 0],
    ]


def test_replay_rejects_changed_answer_and_changed_prompt():
    case, graph, anchors, evidence = _case()
    altered = copy.deepcopy(case)
    altered["expected_turns"][0]["end_poi_id"] = "via"
    with pytest.raises(ValueError, match="gold label"):
        replay_route_case(altered, graph, anchors, permitted_edge_ids=evidence,
                          unknown_edge_ids=set(), closed_edge_ids=set(), templates=TEMPLATES)
    altered = copy.deepcopy(case)
    altered["messages"][0]["content"] = "去别的地方"
    with pytest.raises(ValueError, match="prompt"):
        replay_route_case(altered, graph, anchors, permitted_edge_ids=evidence,
                          unknown_edge_ids=set(), closed_edge_ids=set(), templates=TEMPLATES)


def test_unknown_connection_cannot_become_generated_verified_gold():
    graph, anchors, _ = _inputs()
    with pytest.raises(ValueError, match="unknown"):
        make_route_case(
            graph, anchors, permitted_edge_ids={(1, 2, 0)},
            unknown_edge_ids={(2, 3, 0)}, closed_edge_ids=set(),
            templates=TEMPLATES, template_id="direct", case_id="bad",
            category="mode_access", template_group="direct_pilot", od_group="start_end",
            snapshot_id="a" * 64, generator_sha256="b" * 64,
            template_sha256="c" * 64,
            parameters={"start_poi_id": "start", "end_poi_id": "end",
                        "via_poi_ids": [], "mode": "walk", "avoid_steps": False},
        )


def test_template_cannot_hide_mode_or_constraints_from_user():
    graph, anchors, evidence = _inputs()
    with pytest.raises(ValueError, match="template must state"):
        make_route_case(
            graph, anchors, permitted_edge_ids=evidence, unknown_edge_ids=set(),
            closed_edge_ids=set(), templates={"bad": "从{start}到{end}。"},
            template_id="bad", case_id="bad", category="mode_access",
            template_group="bad", od_group="start_end", snapshot_id="a" * 64,
            generator_sha256="b" * 64, template_sha256="c" * 64,
            parameters={"start_poi_id": "start", "end_poi_id": "end",
                        "via_poi_ids": [], "mode": "walk", "avoid_steps": True},
        )
