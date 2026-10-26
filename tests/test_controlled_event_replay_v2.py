"""A true no-path tool error can satisfy the infeasible event task."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import networkx as nx

from experiments.controlled_event_replay_v2 import score_controlled_event_case
from experiments.independent_source_snapshot import mode_edge_evidence


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/datasets/controlled_event_pilot_v2"


def _infeasible_input():
    dataset = json.loads((PILOT / "scenarios_event_pilot_dev_v2.json").read_text(
        encoding="utf-8"
    ))
    source = json.loads((PILOT / "event_snapshots_dev_v1.json").read_text(
        encoding="utf-8"
    ))
    spec = json.loads((PILOT / "event_pilot_spec.json").read_text(
        encoding="utf-8"
    ))
    case = next(row for row in dataset["cases"]
                if row["category"] == "event_infeasible")
    params = case["gold_derivation"]["parameters"]
    snapshot = next(row for row in source["snapshots"]
                    if row["id"] == params["event_snapshot_id"])
    graph = nx.read_graphml(ROOT / "data/whu_road_network.graphml",
                            node_type=int, force_multigraph=True)
    anchor_data = json.loads((ROOT / "experiments/datasets/controlled_spatial_v1/"
                              "walk_anchors.json").read_text(encoding="utf-8"))
    anchors = {row["poi_id"]: row for row in anchor_data["anchors"]}
    boundary = (ROOT / "experiments/datasets/controlled_spatial_v1/"
                "boundary_wgs84.wkt").read_text(encoding="utf-8")
    evidence = mode_edge_evidence(graph, "walk", require_boundary=True,
                                  boundary_wkt=boundary)
    template = spec["templates"][case["gold_derivation"]["template_id"]]["text"]
    request = {"start_node": int(anchors[params["start_poi_id"]]["node_id"]),
               "end_node": int(anchors[params["end_poi_id"]]["node_id"]),
               "mode": params["mode"], "avoid_steps": params["avoid_steps"]}
    request_digest = hashlib.sha256(json.dumps(
        request, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    expected = case["expected_turns"][0]
    actual = {key: deepcopy(expected[key]) for key in (
        "status", "kind", "start_poi_id", "end_poi_id", "via_poi_ids",
        "travel_mode", "strategy", "hard_constraints", "candidate_poi_ids",
        "clarification_slot",
    )}
    actual.update(outputs=expected["required_outputs"], unknowns=[],
                  route_edge_ids=[],
                  road_event_ids=sorted(row["id"] for row in snapshot["conditions"]),
                  tool_trace=[
                      {"tool_name": "list_road_conditions", "argument_keys": [],
                       "outcome": "success", "artifact_kind": None},
                      {"tool_name": "plan_route", "argument_keys": ["start", "end"],
                       "outcome": "tool_error", "artifact_kind": None,
                       "error_code": "graph_no_path",
                       "request_digest": request_digest},
                  ])
    observed = {"scenario_id": case["id"], "attempt": 1,
                "snapshot_id": case["snapshot_id"],
                "input_messages": case["messages"], "turns": [actual]}
    return case, observed, graph, anchors, evidence, snapshot, template


def test_explicit_graph_no_path_is_valid_infeasible_attempt():
    case, observed, graph, anchors, evidence, snapshot, template = _infeasible_input()
    score = score_controlled_event_case(case, observed, graph, anchors,
                                        evidence, snapshot, template=template)
    assert score["task_success"] is True, score

    wrong = deepcopy(observed)
    wrong["turns"][0]["tool_trace"][1]["error_code"] = "invalid_args"
    score = score_controlled_event_case(case, wrong, graph, anchors,
                                        evidence, snapshot, template=template)
    assert score["task_success"] is False
    assert "graph_no_path_evidence_missing" in score["failures"]

    wrong_od = deepcopy(observed)
    wrong_od["turns"][0]["tool_trace"][1]["request_digest"] = "0" * 64
    score = score_controlled_event_case(case, wrong_od, graph, anchors,
                                        evidence, snapshot, template=template)
    assert score["task_success"] is False
    assert "graph_no_path_evidence_missing" in score["failures"]
