"""Event packet builder must bind cases to their own snapshot."""

import networkx as nx
import pytest

from experiments.build_controlled_event_pilot import (
    build_event_pilot_dataset, verify_event_pilot_dataset,
)
from experiments.controlled_event_replay import score_controlled_event_case


def _inputs():
    graph = nx.MultiDiGraph()
    for u, v, length in ((1, 2, 5), (2, 3, 5), (1, 4, 7), (4, 3, 7)):
        graph.add_edge(u, v, key=0, length=length, highway="residential")
    anchors = {"a": {"poi_id": "a", "name": "甲", "node_id": 1,
                     "access_status": "last_meter_unverified"},
               "b": {"poi_id": "b", "name": "乙", "node_id": 3,
                     "access_status": "last_meter_unverified"}}
    evidence = {"walk": {"permitted": set(graph.edges(keys=True)), "unknown": set()}}
    snapshots = [
        {"id": snapshot_id, "scope": "synthetic_development_only", "campus_truth": False,
         "as_of_epoch": 150, "conditions": [
             {"id": f"{snapshot_id}_{index}", "action": "closure",
              "affected_modes": ["walk"], "edge": {"u": u, "v": v, "key": key},
              "start_time": 100, "end_time": 200}
             for index, (u, v, key) in enumerate(edges)]}
        for snapshot_id, edges in (("detour", [(2, 3, 0)]),
                                   ("blocked", [(2, 3, 0), (4, 3, 0)]))
    ]
    source = {"format": "controlled-simulated-events-v1",
              "scope": "synthetic_development_only", "campus_truth": False,
              "snapshots": snapshots}
    spec = {"format": "controlled-event-pilot-spec-v1",
            "scope": "synthetic_development_only",
            "templates": {
                "detour": {"category": "compound_tools", "template_group": "event_dev",
                           "text": "先查路况，再给我从{start}{mode_text}到{end}的全程最短路线。"},
                "blocked": {"category": "event_infeasible", "template_group": "event_dev",
                            "text": "先查路况，给我从{start}{mode_text}到{end}的全程路线或不可达原因。"}},
            "cases": [
                {"id": category, "template_id": template_id, "od_group": "a_b",
                 "parameters": {"start_poi_id": "a", "end_poi_id": "b",
                                "via_poi_ids": [], "mode": "walk", "avoid_steps": False,
                                "event_snapshot_id": snapshot_id,
                                "expected_verdict": verdict}}
                for category, template_id, snapshot_id, verdict in (
                    ("compound_tools", "detour", "detour", "feasible"),
                    ("event_infeasible", "blocked", "blocked", "infeasible"))]}
    manifest = {"profile": "event_pilot", "effective_inputs_sha256": "a" * 64,
                "inputs": {"generator": {"sha256": "b" * 64},
                           "templates": {"sha256": "c" * 64}}}
    return graph, anchors, evidence, spec, source, manifest


def test_build_and_replay_both_event_cases():
    graph, anchors, evidence, spec, source, manifest = _inputs()
    packet = build_event_pilot_dataset(graph, anchors, evidence, spec, source,
                                       manifest, "d" * 64)
    assert [case["category"] for case in packet["cases"]] == [
        "compound_tools", "event_infeasible"]
    assert [case["gold_derivation"]["template_id"] for case in packet["cases"]] == [
        "detour", "blocked"]
    verify_event_pilot_dataset(packet, graph, anchors, evidence, spec, source,
                               manifest, "d" * 64)
    packet["cases"][0]["messages"][0]["content"] = "tampered"
    with pytest.raises(ValueError, match="differs"):
        verify_event_pilot_dataset(packet, graph, anchors, evidence, spec, source,
                                   manifest, "d" * 64)


def test_event_scorer_requires_road_query_and_checks_exact_route():
    graph, anchors, evidence, spec, source, manifest = _inputs()
    packet = build_event_pilot_dataset(graph, anchors, evidence, spec, source,
                                       manifest, "d" * 64)
    case = packet["cases"][0]
    snapshot = source["snapshots"][0]
    turn = case["expected_turns"][0]
    actual = {key: turn[key] for key in (
        "status", "kind", "start_poi_id", "end_poi_id", "via_poi_ids",
        "travel_mode", "strategy", "hard_constraints", "candidate_poi_ids",
        "clarification_slot")}
    actual.update(outputs=["road_events", "route"], unknowns=[],
                  road_event_ids=["detour_0"],
                  route_edge_ids=[[1, 4, 0], [4, 3, 0]],
                  tool_trace=[
                      {"tool_name": "list_road_conditions", "argument_keys": [],
                       "outcome": "success", "artifact_kind": None},
                      {"tool_name": "plan_route", "argument_keys": [],
                       "outcome": "success", "artifact_kind": "route"},
                  ])
    observed = {"scenario_id": case["id"], "snapshot_id": case["snapshot_id"],
                "input_messages": case["messages"], "turns": [actual]}
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence["walk"], snapshot,
                                        template=spec["templates"]["detour"]["text"])
    assert score["task_success"] is True
    actual["route_edge_ids"] = [[1, 2, 0], [2, 3, 0]]
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence["walk"], snapshot,
                                        template=spec["templates"]["detour"]["text"])
    assert score["task_success"] is False
    assert "route_invalid" in score["failures"]
    actual["route_edge_ids"] = [[1, 4, 0], [4, 3, 0]]
    actual["road_event_ids"] = []
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence["walk"], snapshot,
                                        template=spec["templates"]["detour"]["text"])
    assert "road_event_ids_missing" in score["failures"]
    actual["road_event_ids"] = ["wrong_event"]
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence["walk"], snapshot,
                                        template=spec["templates"]["detour"]["text"])
    assert "road_event_mismatch" in score["failures"]
    actual["road_event_ids"] = ["detour_0"]
    actual["tool_trace"].reverse()
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence["walk"], snapshot,
                                        template=spec["templates"]["detour"]["text"])
    assert "road_query_after_route" in score["failures"]
    actual["tool_trace"].reverse()
    actual["tool_trace"][1]["outcome"] = "tool_error"
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence["walk"], snapshot,
                                        template=spec["templates"]["detour"]["text"])
    assert "route_plan_trace_missing" in score["failures"]
    actual["tool_trace"] = [
        {"tool_name": "plan_route", "argument_keys": [],
         "outcome": "tool_error", "artifact_kind": None},
        {"tool_name": "list_road_conditions", "argument_keys": [],
         "outcome": "success", "artifact_kind": None},
        {"tool_name": "plan_route", "argument_keys": [],
         "outcome": "success", "artifact_kind": "route"},
    ]
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence["walk"], snapshot,
                                        template=spec["templates"]["detour"]["text"])
    assert "road_query_after_route" in score["failures"]
