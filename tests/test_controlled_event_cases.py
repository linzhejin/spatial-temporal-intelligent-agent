"""Simulated closures must produce independent detour or no-route gold."""

from copy import deepcopy

import networkx as nx
import pytest

from experiments.controlled_event_cases import make_event_case, replay_event_case
from experiments.controlled_gold_contract import validate_controlled_dataset


def _fixture():
    graph = nx.MultiDiGraph()
    for u, v, length in ((1, 2, 5), (2, 3, 5), (1, 4, 7), (4, 3, 7)):
        graph.add_edge(u, v, key=0, highway="residential", length=length)
    anchors = {poi_id: {"poi_id": poi_id, "name": name, "node_id": node,
                        "access_status": "last_meter_unverified"}
               for poi_id, name, node in (("a", "甲门", 1), ("b", "乙楼", 3))}
    permitted = {(u, v, key) for u, v, key in graph.edges(keys=True)}
    evidence = {"permitted": permitted, "unknown": set()}
    params = {"start_poi_id": "a", "end_poi_id": "b", "via_poi_ids": [],
              "mode": "walk", "avoid_steps": False}
    template = "先查路况，再给我从{start}{mode_text}到{end}的全程最短路线。"
    return graph, anchors, evidence, params, template


def _event_snapshot(edge_ids, *, action="closure", affected_modes=None,
                    end_time=200, as_of_epoch=150):
    return {"id": "simulated_event", "scope": "synthetic_development_only",
            "campus_truth": False, "as_of_epoch": as_of_epoch,
            "conditions": [
                {"id": f"event_{index}", "action": action,
                 "affected_modes": affected_modes or ["walk"],
                 "edge": {"u": u, "v": v, "key": key},
                 "start_time": 100, "end_time": end_time}
                for index, (u, v, key) in enumerate(edge_ids)
            ]}


def _case(graph, anchors, evidence, params, template, snapshot, *, category,
          expected_verdict):
    return make_event_case(
        graph, anchors, evidence, snapshot, template=template,
        case_id=category, category=category, template_group="event_dev",
        od_group="a_b", snapshot_id="a" * 64,
        generator_sha256="b" * 64, template_sha256="c" * 64,
        parameters={**params, "event_snapshot_id": snapshot["id"],
                    "expected_verdict": expected_verdict},
    )


def test_active_closure_forces_detour_and_source_derived_compound_gold():
    graph, anchors, evidence, params, template = _fixture()
    snapshot = _event_snapshot([(2, 3, 0)])
    case = _case(graph, anchors, evidence, params, template, snapshot,
                 category="compound_tools", expected_verdict="feasible")
    assert case["expected_turns"][0]["required_outputs"] == ["road_events", "route"]
    assert case["gold_derivation"]["oracle_result"]["length_m"] == 14
    assert case["gold_derivation"]["oracle_result"]["route_edge_ids"] == [
        [1, 4, 0], [4, 3, 0],
    ]
    replay_event_case(case, graph, anchors, evidence, snapshot, template=template)
    packet = {"protocol_version": "scenario-v2", "gold_layer": "controlled_graph",
              "split": "development", "repeats": 1,
              "source_manifest_sha256": "d" * 64, "snapshot_id": "a" * 64,
              "cases": [case]}
    validate_controlled_dataset(packet)


def test_cut_closure_yields_no_route_but_expired_or_wrong_mode_does_not():
    graph, anchors, evidence, params, template = _fixture()
    snapshot = _event_snapshot([(2, 3, 0), (4, 3, 0)])
    case = _case(graph, anchors, evidence, params, template, snapshot,
                 category="event_infeasible", expected_verdict="infeasible")
    assert case["expected_turns"][0]["kind"] == "no_route"
    assert case["expected_turns"][0]["required_outputs"] == [
        "road_events", "infeasibility_explanation",
    ]
    assert case["gold_derivation"]["oracle_result"]["route_edge_ids"] == []
    replay_event_case(case, graph, anchors, evidence, snapshot, template=template)
    for wrong_snapshot in (
        _event_snapshot([(2, 3, 0), (4, 3, 0)], end_time=120),
        _event_snapshot([(2, 3, 0), (4, 3, 0)], affected_modes=["bike"]),
    ):
        with pytest.raises(ValueError, match="active closure"):
            _case(graph, anchors, evidence, params, template, wrong_snapshot,
                  category="event_infeasible", expected_verdict="infeasible")
    tampered = deepcopy(case)
    tampered["messages"][0]["content"] = "请随便带我走"
    with pytest.raises(ValueError, match="differs"):
        replay_event_case(tampered, graph, anchors, evidence, snapshot,
                          template=template)
