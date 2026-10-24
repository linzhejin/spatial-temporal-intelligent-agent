"""The revised experiment protocol must not turn drafts into formal evidence."""

import copy
import json

import networkx as nx
import pytest

from experiments.scenario_v2 import (audit_route_edges, score_case, validate_dataset,
                                     load_frozen_graph, closed_edges_at)
from experiments.scenario_split_audit import audit_scenario_splits
from experiments.scenario_runner import run_replay


def _turn():
    return {
        "status": "completed", "kind": "route", "feasibility": "feasible",
        "start_poi_id": "a", "end_poi_id": "b", "via_poi_ids": [],
        "travel_mode": "walk", "strategy": "shortest",
        "hard_constraints": {"avoid_steps": True}, "required_outputs": ["route"],
        "candidate_poi_ids": [], "clarification_slot": None,
        "allowed_unknowns": [],
    }


def _case():
    return {
        "id": "T1-001", "category": "T1", "template_group": "route-a",
        "od_group": "od-a", "snapshot_id": "a" * 64,
        "messages": [{"role": "user", "content": "从甲到乙，避开台阶"}],
        "expected_turns": [_turn()],
        "adjudication": {
            "status": "adjudicated", "method": "graph_computed",
            "evidence": ["frozen-graph-sha256:a"],
        },
        "route_reference": {"start_node": "1", "end_node": "3", "via_nodes": []},
    }


def _observation(edges):
    return {
        "scenario_id": "T1-001", "snapshot_id": "a" * 64,
        "input_messages": _case()["messages"],
        "turns": [{
            "status": "completed", "kind": "route", "start_poi_id": "a",
            "end_poi_id": "b", "via_poi_ids": [], "travel_mode": "walk",
            "strategy": "shortest", "hard_constraints": {"avoid_steps": True},
            "candidate_poi_ids": [], "clarification_slot": None,
            "outputs": ["route"], "route_edge_ids": edges,
            "independent_review": {
                "source": "independent_reviewer", "reviewer": "self-report",
                "evidence": ["claimed-pass"], "route": "pass",
                "hard_constraints": {"avoid_steps": "pass"},
            },
        }],
    }


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edge("1", "2", key=0, length=10, highway="steps")
    graph.add_edge("2", "3", key=0, length=10, highway="path")
    graph.add_edge("1", "4", key=0, length=11, highway="path")
    graph.add_edge("4", "3", key=0, length=11, highway="path")
    return graph


def test_v2_route_review_rejects_self_report_and_step_edge():
    result = score_case(_case(), _observation(["1|2|0", "2|3|0"]), graph=_graph())
    assert result["task_success"] is False
    assert "hard_constraint_violation" in result["failures"]


def test_v2_route_review_accepts_connected_allowed_path():
    result = score_case(_case(), _observation(["1|4|0", "4|3|0"]), graph=_graph())
    assert result["task_success"] is True


def _two_route_case_and_observation():
    case = _case()
    changed = {**_turn(), "end_poi_id": "c"}
    case["messages"].append({"role": "user", "content": "终点改成丙，起点不变"})
    case["expected_turns"].append(changed)
    observed = _observation(["1|4|0", "4|3|0"])
    observed["input_messages"] = case["messages"]
    second = copy.deepcopy(observed["turns"][0])
    second["end_poi_id"] = "c"
    observed["turns"].append(second)
    return case, observed


def test_v2_multiturn_route_review_uses_each_turn_endpoint_reference():
    case, observed = _two_route_case_and_observation()
    case["route_references"] = {
        "0": {"start_node": "1", "end_node": "3", "via_nodes": []},
        "1": {"start_node": "1", "end_node": "4", "via_nodes": []},
    }
    observed["turns"][1]["route_edge_ids"] = ["1|4|0"]
    assert score_case(case, observed, graph=_graph())["task_success"] is True


def test_v2_multiturn_route_review_never_reuses_first_turn_reference():
    case, observed = _two_route_case_and_observation()
    score = score_case(case, observed, graph=_graph())
    assert score["task_success"] is False
    assert "route_review_missing" in score["unknowns"]


def test_v2_route_after_clarification_uses_second_turn_reference():
    case = _case()
    case["messages"].insert(0, {"role": "user", "content": "带我去乙，起点稍后说"})
    case["expected_turns"].insert(0, {
        **_turn(), "status": "needs_input", "kind": "clarify",
        "feasibility": "not_applicable", "start_poi_id": None,
        "strategy": None, "hard_constraints": {}, "required_outputs": [],
        "clarification_slot": "start",
    })
    case["route_references"] = {"1": case.pop("route_reference")}
    observed = _observation(["1|4|0", "4|3|0"])
    observed["input_messages"] = case["messages"]
    observed["turns"].insert(0, {
        "status": "needs_input", "kind": "clarify", "start_poi_id": None,
        "end_poi_id": "b", "via_poi_ids": [], "travel_mode": "walk",
        "strategy": None, "hard_constraints": {}, "candidate_poi_ids": [],
        "clarification_slot": "start", "outputs": [], "route_edge_ids": [],
    })
    validate_dataset({"protocol_version": "scenario-v2", "split": "development",
                      "cases": [case]})
    assert score_case(case, observed, graph=_graph())["task_success"] is True


def test_v2_multiturn_graph_gold_requires_complete_turn_references():
    case, _ = _two_route_case_and_observation()
    dataset = {"protocol_version": "scenario-v2", "split": "development",
               "cases": [case]}
    with pytest.raises(ValueError, match="per-turn route reference"):
        validate_dataset(dataset)
    case["route_references"] = {
        "0": {"start_node": "1", "end_node": "3", "via_nodes": []},
    }
    with pytest.raises(ValueError, match="per-turn route reference"):
        validate_dataset(dataset)
    case["route_references"]["1"] = {
        "start_node": "1", "end_node": "4", "via_nodes": [],
    }
    validate_dataset(dataset)


def test_v2_route_gold_rejects_stale_extra_turn_reference():
    case = _case()
    case["route_references"] = {
        "0": case["route_reference"],
        "1": {"start_node": "1", "end_node": "3", "via_nodes": []},
    }
    dataset = {"protocol_version": "scenario-v2", "split": "development",
               "cases": [case]}
    with pytest.raises(ValueError, match="per-turn route reference"):
        validate_dataset(dataset)


def test_v2_multiturn_route_references_match_independent_poi_snaps():
    from experiments.scenario_v2 import verify_route_reference_sources

    case, _ = _two_route_case_and_observation()
    case["route_coordinates"] = "wgs84"
    case["route_references"] = {
        "0": {"start_node": "1", "end_node": "3", "via_nodes": []},
        "1": {"start_node": "1", "end_node": "4", "via_nodes": []},
    }
    graph = nx.MultiDiGraph()
    for node, x in (("1", 114.0), ("4", 114.001), ("3", 114.002)):
        graph.add_node(node, x=x, y=30.0)
    graph.add_edge("1", "4", key=0, length=100, highway="path")
    graph.add_edge("4", "3", key=0, length=100, highway="path")
    pois = [{"id": poi_id, "coordinates": {"lng": x, "lat": 30.0}}
            for poi_id, x in (("a", 114.0), ("c", 114.001), ("b", 114.002))]

    assert verify_route_reference_sources(case, graph, pois) == 2
    case["route_references"]["1"]["end_node"] = "3"
    with pytest.raises(ValueError, match="route reference differs from frozen sources"):
        verify_route_reference_sources(case, graph, pois)


def test_v2_graph_route_and_manual_weather_preservation_can_be_reviewed_together():
    case = _case()
    case["route_references"] = {"0": case.pop("route_reference")}
    case["messages"].append({"role": "user", "content": "再告诉我天气，不要改路线。"})
    case["expected_turns"].append({**_turn(), "kind": "information",
                                   "feasibility": "not_applicable", "start_poi_id": None,
                                   "end_poi_id": None, "via_poi_ids": [],
                                   "travel_mode": None, "strategy": None,
                                   "hard_constraints": {}, "required_outputs": ["weather"],
                                   "preserve_route_from_turn": 0})
    observed = _observation(["1|4|0", "4|3|0"])
    observed["input_messages"] = case["messages"]
    observed["turns"].append({
        "status": "completed", "kind": "information", "start_poi_id": None,
        "end_poi_id": None, "via_poi_ids": [], "travel_mode": None,
        "strategy": None, "hard_constraints": {}, "candidate_poi_ids": [],
        "clarification_slot": None, "outputs": ["weather"], "route_edge_ids": [],
    })
    review = {"source": "independent_reviewer", "reviewer": "ui-1",
              "evidence": ["client-route-state-hash-before-after"],
              "outputs": {"weather": "pass", "route_preservation": "pass"}}
    score = score_case(case, observed, graph=_graph(), independent_reviews=[{}, review])
    assert score["task_success"] is True


def test_v2_rejects_route_preservation_reference_to_nonroute_turn():
    case = _case()
    case["expected_turns"][0]["preserve_route_from_turn"] = 0
    dataset = {"protocol_version": "scenario-v2", "split": "development",
               "cases": [case]}
    with pytest.raises(ValueError, match="invalid route preservation reference"):
        validate_dataset(dataset)


def test_v2_route_review_fails_closed_when_graph_missing():
    result = score_case(_case(), _observation(["1|4|0", "4|3|0"]), graph=None)
    assert result["task_success"] is False
    assert "route_review_missing" in result["unknowns"]


def test_v2_candidate_order_uses_full_reference_and_tie_tolerance():
    from experiments.scenario_v2 import audit_candidate_order

    reference = {"distances_m": {"a": 100, "b": 103, "c": 200, "d": 250,
                                 "e": 252}, "max_display": 4,
                 "rank_tolerance_m": 5, "uncertain_access_ids": []}
    assert audit_candidate_order(["b", "a", "c", "e"], reference)["verdict"] == "pass"
    assert audit_candidate_order(["a", "b", "c", "d"], reference)["verdict"] == "pass"
    assert audit_candidate_order(["c", "b", "a", "d"], reference)["verdict"] == "fail"
    assert audit_candidate_order(["a", "b", "c", "c"], reference)["verdict"] == "fail"
    assert audit_candidate_order(["a", "b", "c", "d"], {
        **reference, "uncertain_access_ids": ["a"]})["verdict"] == "unknown"
    assert audit_candidate_order(["a", "b", "c", "d"], {
        **reference, "uncertain_access_ids": ["e"]})["verdict"] == "unknown"


def test_v2_candidate_review_does_not_require_exact_tie_order():
    case = _case()
    case["category"] = "T2"
    case["expected_turns"][0].update({
        "kind": "candidates", "feasibility": "not_applicable",
        "end_poi_id": None, "hard_constraints": {},
        "required_outputs": ["candidates"], "candidate_poi_ids": [],
    })
    reference = {
        "distances_m": {"a": 100, "b": 103, "c": 220}, "max_display": 2,
        "rank_tolerance_m": 5, "uncertain_access_ids": [],
    }
    actual = _observation([])
    actual["turns"][0].update({
        "kind": "candidates", "end_poi_id": None, "hard_constraints": {},
        "candidate_poi_ids": ["b", "a"], "outputs": ["candidates"],
    })
    case["candidate_reference"] = {**reference, "distances_m": {"c": 1}}
    missing = score_case(case, actual)
    assert "candidate_evidence_missing" in missing["unknowns"]
    result = score_case(case, actual, candidate_reference=reference)
    assert result["task_success"] is True
    assert "wrong_candidate_order" not in result["failures"]


def test_v2_candidate_delivery_accepts_selection_pending_or_completed_status():
    case = _case()
    case["category"] = "T2"
    case["expected_turns"][0].update({
        "kind": "candidates", "feasibility": "not_applicable",
        "end_poi_id": None, "hard_constraints": {},
        "required_outputs": ["candidates"], "candidate_poi_ids": [],
    })
    observed = _observation([])
    observed["turns"][0].update({
        "status": "needs_input", "kind": "candidates", "end_poi_id": None,
        "hard_constraints": {}, "candidate_poi_ids": ["a", "b"],
        "outputs": ["candidates"],
    })
    reference = {"distances_m": {"a": 100, "b": 200}, "max_display": 2,
                 "rank_tolerance_m": 0, "uncertain_access_ids": []}
    score = score_case(case, observed, candidate_reference=reference)
    assert score["task_success"] is True
    assert "wrong_status" not in score["failures"]


def test_v2_two_candidate_turns_use_distinct_location_references():
    case = _case()
    case["category"] = "T5"
    case["messages"] = [{"role": "user", "content": "先找篮球场"},
                        {"role": "user", "content": "选球场甲，再找附近吃饭处"}]
    first = {**_turn(), "status": "needs_input", "kind": "candidates",
             "feasibility": "not_applicable", "end_poi_id": None,
             "required_outputs": ["candidates"], "hard_constraints": {}}
    second = {**first, "start_poi_id": "court"}
    case["expected_turns"] = [first, second]
    observed = _observation([])
    observed["input_messages"] = case["messages"]
    observed["turns"] = [{**observed["turns"][0], "status": "needs_input",
                          "kind": "candidates", "end_poi_id": None,
                          "hard_constraints": {}, "outputs": ["candidates"],
                          "candidate_poi_ids": ["court"]},
                         {**observed["turns"][0], "status": "needs_input",
                          "kind": "candidates", "start_poi_id": "court",
                          "end_poi_id": None, "hard_constraints": {},
                          "outputs": ["candidates"],
                          "candidate_poi_ids": ["food"]}]
    references = {
        "0": {"distances_m": {"court": 100}, "max_display": 4,
              "rank_tolerance_m": 0, "uncertain_access_ids": []},
        "1": {"distances_m": {"food": 50}, "max_display": 4,
              "rank_tolerance_m": 0, "uncertain_access_ids": []},
    }
    assert score_case(case, observed, candidate_references=references)["task_success"] is True
    observed["turns"][1]["candidate_poi_ids"] = ["court"]
    failed = score_case(case, observed, candidate_references=references)
    assert failed["task_success"] is False
    assert "output_invalid" in failed["failures"]


def test_v2_formal_multiturn_candidate_gold_requires_each_turn_spec():
    dataset = _formal_dataset()
    case = next(row for row in dataset["cases"] if row["category"] == "T2")
    candidate = {**_turn(), "kind": "candidates", "status": "needs_input",
                 "feasibility": "not_applicable", "end_poi_id": None,
                 "required_outputs": ["candidates"], "hard_constraints": {}}
    case["expected_turns"] = [candidate, {**candidate, "start_poi_id": "court"}]
    case["messages"].append({"role": "user", "content": "选球场甲，再找食堂"})
    spec = {"activities": ["basketball"], "coordinates": "wgs84",
            "max_display": 4, "rank_tolerance_m": 5}
    case["candidate_specs"] = {"0": spec}
    with pytest.raises(ValueError, match="per-turn candidate specs"):
        validate_dataset(dataset, formal=True)
    case["candidate_specs"]["1"] = {**spec, "activities": ["canteen_meal"]}
    validate_dataset(dataset, formal=True)


def test_v2_replay_recomputes_candidates_from_frozen_pois_and_graph(tmp_path):
    graph = nx.MultiDiGraph()
    for node, x in (("s", 114.0), ("a", 114.001), ("b", 114.002),
                    ("c", 114.003)):
        graph.add_node(node, x=x, y=30.0)
    for u, v in (("s", "a"), ("a", "b"), ("b", "c")):
        graph.add_edge(u, v, key="walk", length=100, highway="footway")
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    pois_path = tmp_path / "pois.json"
    pois_path.write_text(json.dumps({"pois": [
        {"id": poi_id, "coordinates": {"lng": x, "lat": 30.0},
         "activities": [] if poi_id == "origin" else ["canteen_meal"]}
        for poi_id, x in (("origin", 114.0), ("near", 114.001),
                          ("middle", 114.002), ("far", 114.003))
    ]}), encoding="utf-8")
    case = _case()
    case["category"] = "T2"
    case["expected_turns"][0].update({
        "kind": "candidates", "feasibility": "not_applicable",
        "start_poi_id": "origin", "end_poi_id": None,
        "hard_constraints": {}, "required_outputs": ["candidates"],
        "candidate_poi_ids": [],
    })
    case["candidate_spec"] = {"activities": ["canteen_meal"],
                              "coordinates": "wgs84", "max_display": 3,
                              "rank_tolerance_m": 5}
    case["candidate_reference"] = {"distances_m": {"far": 1}}
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"protocol_version": "scenario-v2",
                                        "split": "development", "cases": [case]}),
                            encoding="utf-8")
    observation = _observation([])
    observation["attempt"] = 1
    observation["turns"][0].update({
        "kind": "candidates", "start_poi_id": "origin", "end_poi_id": None,
        "hard_constraints": {}, "candidate_poi_ids": ["near", "middle", "far"],
        "outputs": ["candidates"],
    })
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps(observation) + "\n", encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text('{"protocol_version":"scenario-v2"}', encoding="utf-8")
    run = run_replay(dataset_path, replay_path, protocol_path, tmp_path / "results",
                     "candidate-check", graph_path=graph_path, pois_path=pois_path)
    record = json.loads((run / "records.jsonl").read_text(encoding="utf-8"))
    assert record["score"]["task_success"] is True
    assert "independent_review" not in record["observation"]["turns"][0]
    assert len(json.loads((run / "manifest.json").read_text(encoding="utf-8"))
               ["poi_sha256"]) == 64


def test_v2_replay_recomputes_two_candidate_origins_then_a_full_ordered_route(tmp_path):
    graph = nx.MultiDiGraph()
    for node, x in (("origin", 114.0), ("court", 114.001), ("food", 114.002)):
        graph.add_node(node, x=x, y=30.0)
    graph.add_edge("origin", "court", key=0, length=100, highway="footway")
    graph.add_edge("court", "food", key=0, length=100, highway="footway")
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    pois_path = tmp_path / "pois.json"
    pois_path.write_text(json.dumps({"pois": [
        {"id": poi_id, "coordinates": {"lng": x, "lat": 30.0},
         "activities": activities}
        for poi_id, x, activities in (("origin", 114.0, []),
                                      ("court", 114.001, ["basketball"]),
                                      ("food", 114.002, ["canteen_meal"]))
    ]}), encoding="utf-8")
    case = _case()
    case["category"] = "T5"
    case["messages"] = [{"role": "user", "content": "从起点找球场"},
                        {"role": "user", "content": "选球场，再找吃饭处"},
                        {"role": "user", "content": "选食堂，从原起点经球场走完全程"}]
    first = {**_turn(), "kind": "candidates", "status": "needs_input",
             "feasibility": "not_applicable", "start_poi_id": "origin",
             "end_poi_id": None, "hard_constraints": {},
             "required_outputs": ["candidates"]}
    final = {**_turn(), "start_poi_id": "origin", "end_poi_id": "food",
             "via_poi_ids": ["court"], "hard_constraints": {}}
    case["expected_turns"] = [first, {**first, "start_poi_id": "court"}, final]
    case["route_references"] = {
        "2": {"start_node": "origin", "via_nodes": ["court"],
              "end_node": "food"},
    }
    case.pop("route_reference")
    case["route_coordinates"] = "wgs84"
    case["candidate_specs"] = {
        "0": {"activities": ["basketball"], "coordinates": "wgs84",
              "max_display": 4, "rank_tolerance_m": 0},
        "1": {"activities": ["canteen_meal"], "coordinates": "wgs84",
              "max_display": 4, "rank_tolerance_m": 0},
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"protocol_version": "scenario-v2",
                                        "split": "development", "cases": [case]}),
                            encoding="utf-8")
    observed = _observation([])
    observed["attempt"] = 1
    observed["input_messages"] = case["messages"]
    actual = {**observed["turns"][0], "kind": "candidates", "status": "needs_input",
              "start_poi_id": "origin", "end_poi_id": None,
              "hard_constraints": {}, "outputs": ["candidates"],
              "candidate_poi_ids": ["court"]}
    observed["turns"] = [actual, {**actual, "start_poi_id": "court",
                                 "candidate_poi_ids": ["food"]},
                         {**actual, "status": "completed", "kind": "route",
                          "start_poi_id": "origin", "end_poi_id": "food",
                          "via_poi_ids": ["court"], "candidate_poi_ids": [],
                          "outputs": ["route"],
                          "route_edge_ids": ["origin|court|0", "court|food|0"]}]
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps(observed) + "\n", encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text('{"protocol_version":"scenario-v2"}', encoding="utf-8")
    run = run_replay(dataset_path, replay_path, protocol_path, tmp_path / "results",
                     "two-candidate-origins", graph_path=graph_path, pois_path=pois_path)
    record = json.loads((run / "records.jsonl").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert record["score"]["task_success"] is True
    assert manifest["candidate_reference_turns_recomputed"] == 2
    assert manifest["route_reference_turns_verified_against_sources"] == 1


def test_v2_replay_rejects_route_gold_with_poisoned_poi_anchor(tmp_path):
    graph = nx.MultiDiGraph()
    for node, x in (("1", 114.0), ("4", 114.001), ("3", 114.002)):
        graph.add_node(node, x=x, y=30.0)
    graph.add_edge("1", "4", key=0, length=100, highway="path")
    graph.add_edge("4", "3", key=0, length=100, highway="path")
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)
    pois_path = tmp_path / "pois.json"
    pois_path.write_text(json.dumps({"pois": [
        {"id": "a", "coordinates": {"lng": 114.0, "lat": 30.0}},
        {"id": "b", "coordinates": {"lng": 114.002, "lat": 30.0}},
    ]}), encoding="utf-8")
    case = _case()
    case["route_coordinates"] = "wgs84"
    case["route_reference"]["end_node"] = "4"
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"protocol_version": "scenario-v2",
                                        "split": "development", "cases": [case]}),
                            encoding="utf-8")
    observed = _observation(["1|4|0"])
    observed["attempt"] = 1
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps(observed) + "\n", encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text('{"protocol_version":"scenario-v2"}', encoding="utf-8")

    with pytest.raises(ValueError, match="route reference differs from frozen sources"):
        run_replay(dataset_path, replay_path, protocol_path, tmp_path / "results",
                   "poisoned-route-gold", graph_path=graph_path, pois_path=pois_path)
    assert not (tmp_path / "results/poisoned-route-gold").exists()

    case["route_reference"]["end_node"] = "3"
    dataset_path.write_text(json.dumps({"protocol_version": "scenario-v2",
                                        "split": "development", "cases": [case]}),
                            encoding="utf-8")
    observed["turns"][0]["route_edge_ids"] = ["1|4|0", "4|3|0"]
    replay_path.write_text(json.dumps(observed) + "\n", encoding="utf-8")
    run = run_replay(dataset_path, replay_path, protocol_path, tmp_path / "results",
                     "source-backed-route", graph_path=graph_path, pois_path=pois_path)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_reference_turns_verified_against_sources"] == 1


def test_v2_route_audit_rejects_discontinuity_and_closure():
    graph = _graph()
    assert audit_route_edges(graph, ["1|4|0", "2|3|0"], "1", "3",
                             [], "walk", {})["route"] == "fail"
    assert audit_route_edges(graph, ["1|4|0", "4|3|0"], "1", "3",
                             [], "walk", {}, closed_edges={"4|3|0"})["route"] == "fail"


def test_v2_route_audit_checks_mode_and_node_access():
    graph = _graph()
    graph["1"]["4"][0]["allowed_modes"] = '["walk"]'
    assert audit_route_edges(graph, ["1|4|0", "4|3|0"], "1", "3",
                             [], "bike", {})["route"] == "fail"
    graph["1"]["4"][0].pop("allowed_modes")
    graph.nodes["4"]["barrier"] = "stile"
    assert audit_route_edges(graph, ["1|4|0", "4|3|0"], "1", "3",
                             [], "bike", {})["route"] == "fail"
    graph.nodes["4"].pop("barrier")
    graph["1"]["4"][0]["foot"] = "private"
    assert audit_route_edges(graph, ["1|4|0", "4|3|0"], "1", "3",
                             [], "walk", {})["route"] == "fail"


def test_v2_frozen_layers_and_timed_closure(tmp_path):
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(_graph(), graph_path)
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps({"edges": [
        {"edge_id": ["1", "4", 0], "blocked_modes": ["bike"]}
    ]}), encoding="utf-8")
    reviews_path = tmp_path / "reviews.json"
    reviews_path.write_text(json.dumps({"road_decisions": [
        {"edge_id": ["1", "4", 0], "verification_status": "field_verified",
         "verified_at": "2026-01-01", "evidence": "field-sheet-1",
         "blocked_modes": ["walk"], "passable_modes": ["bike"]}
    ]}), encoding="utf-8")
    graph = load_frozen_graph(graph_path, overrides_path, reviews_path)
    assert audit_route_edges(graph, ["1|4|0", "4|3|0"], "1", "3",
                             [], "walk", {})["route"] == "fail"
    assert audit_route_edges(graph, ["1|4|0", "4|3|0"], "1", "3",
                             [], "bike", {})["route"] == "pass"
    events = [{"action": "closure", "affected_modes": ["bike"],
               "start_time": 100, "end_time": 200,
               "edge": {"u": "1", "v": "4", "key": 0}}]
    assert closed_edges_at(graph, events, "bike", 150) == {"1|4|0"}
    assert closed_edges_at(graph, events, "bike", 200) == set()
    assert closed_edges_at(graph, events, "walk", 150) == set()


def test_frozen_layer_rejects_stale_parallel_edge_key(tmp_path):
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(_graph(), graph_path)
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps({"edges": [
        {"u": "1", "v": "4", "k": "missing", "blocked_modes": ["walk"]}
    ]}), encoding="utf-8")
    reviews_path = tmp_path / "reviews.json"
    reviews_path.write_text('{"road_decisions": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="override edge"):
        load_frozen_graph(graph_path, overrides_path, reviews_path)


def _formal_dataset():
    cases = []
    for group in ("T1", "T2", "T3", "T4", "T5"):
        for index in range(24):
            case = _case()
            case["id"] = f"{group}-{index:03d}"
            case["category"] = group
            case["template_group"] = f"family-{group}-{index // 3}"
            case["od_group"] = f"od-{group}-{index // 3}"
            case["expected_turns"][0]["start_poi_id"] = f"start-{group}-{index}"
            case["expected_turns"][0]["end_poi_id"] = f"end-{group}-{index}"
            case["messages"][0]["content"] += f" {group} {index}"
            if group == "T3":
                case["adjudication"] = {
                    "status": "adjudicated", "method": "manual_independent",
                    "reviewer": "reviewer-1", "evidence": ["review-sheet-1"],
                }
            cases.append(case)
    return {"protocol_version": "scenario-v2", "split": "test", "graph_sha256": "b" * 64,
            "poi_sha256": "c" * 64, "repeats": 3, "cases": cases}


def test_v2_formal_accepts_mixed_evidence_without_faking_manual_review():
    dataset = _formal_dataset()
    validate_dataset(dataset, formal=True)
    bad = copy.deepcopy(dataset)
    bad["cases"][0]["adjudication"] = {"status": "draft"}
    with pytest.raises(ValueError, match="adjudicat"):
        validate_dataset(bad, formal=True)


def test_v2_formal_graph_route_gold_requires_frozen_poi_digest():
    dataset = _formal_dataset()
    dataset.pop("poi_sha256", None)
    with pytest.raises(ValueError, match="poi_sha256"):
        validate_dataset(dataset, formal=True)


def test_v2_formal_rejects_empty_candidate_gold():
    dataset = _formal_dataset()
    candidate = next(case for case in dataset["cases"] if case["category"] == "T2")
    candidate["expected_turns"][0].update({
        "kind": "candidates", "feasibility": "not_applicable",
        "required_outputs": ["candidates"], "candidate_poi_ids": [],
    })
    dataset.pop("poi_sha256")
    with pytest.raises(ValueError, match="poi_sha256"):
        validate_dataset(dataset, formal=True)
    dataset["poi_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="candidate spec"):
        validate_dataset(dataset, formal=True)
    candidate["candidate_spec"] = {
        "activities": ["canteen_meal"], "coordinates": "gcj02",
        "max_display": 4, "rank_tolerance_m": 30,
    }
    validate_dataset(dataset, formal=True)
    candidate["expected_turns"][0]["required_outputs"] = []
    with pytest.raises(ValueError, match="candidate output must be scored"):
        validate_dataset(dataset, formal=True)


def test_v2_split_audit_uses_120_source_backed_cases_and_leakage_gates():
    test = _formal_dataset()
    dev_case = _case()
    dev_case["id"] = "development-1"
    dev_case["template_group"] = "development-only"
    dev_case["od_group"] = "development-only"
    dev_case["messages"][0]["content"] = "独立开发样本"
    dev_case["source_group"] = "development_prompt_source"
    for case in test["cases"]:
        case["source_group"] = "heldout_prompt_source"
    dev = {"protocol_version": "scenario-v2", "split": "development", "cases": [dev_case]}
    gates = {name: True for name in (
        "frozen_snapshot_manifest", "agent_readiness_gate", "matching_git_commit",
        "clean_tracked_worktree",
    )}
    report = audit_scenario_splits(dev, test, external_gates=gates)
    assert report["formal_ready"] is True
    assert report["audit_version"] == "scenario-split-audit-v2"
    assert report["dataset_counts"]["test_by_category"] == {
        group: 24 for group in ("T1", "T2", "T3", "T4", "T5")
    }


def test_v2_split_audit_cannot_claim_leakage_free_when_source_provenance_missing():
    dev = {"protocol_version": "scenario-v2", "split": "development", "cases": [
        {"id": "dev", "template_group": "dev_template", "od_group": "dev_od",
         "messages": [{"role": "user", "content": "开发题"}]}
    ]}
    test = {"protocol_version": "scenario-v2", "split": "test", "repeats": 3, "cases": [
        {"id": "test", "template_group": "test_template", "od_group": "test_od",
         "source_group": "test_author", "messages": [{"role": "user", "content": "留出题"}]}
    ]}
    report = audit_scenario_splits(dev, test)
    assert report["leakage_evaluable"] is False
    assert report["leakage_free"] is False
    assert "development_source_keys_missing" in {row["code"] for row in report["blockers"]}


def test_v1_quota_cannot_be_silently_relabelled_as_v2():
    dataset = {"protocol_version": "scenario-v2", "split": "test", "graph_sha256": "b" * 64,
               "repeats": 3, "cases": [_case()]}
    with pytest.raises(ValueError, match="120"):
        validate_dataset(dataset, formal=True)


def test_v2_formal_needs_frozen_graph_digest():
    dataset = {"protocol_version": "scenario-v2", "split": "test",
               "repeats": 3, "cases": [_case()]}
    with pytest.raises(ValueError, match="graph_sha256"):
        validate_dataset(dataset, formal=True)


def test_computed_case_cannot_override_graph_check_with_external_review():
    fake = {"source": "independent_reviewer", "reviewer": "fake", "evidence": ["fake"],
            "route": "pass", "hard_constraints": {"avoid_steps": "pass"}}
    result = score_case(_case(), _observation(["1|2|0", "2|3|0"]),
                        graph=_graph(), independent_reviews=[fake])
    assert "hard_constraint_violation" in result["failures"]


def test_v2_replay_uses_separate_graph_and_records_its_hash(tmp_path):
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(_graph(), graph_path)
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(json.dumps({"protocol_version": "scenario-v2",
                                        "split": "development", "cases": [_case()]}), encoding="utf-8")
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps({**_observation(["1|4|0", "4|3|0"]), "attempt": 1}) + "\n",
                           encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"protocol_version": "scenario-v2"}), encoding="utf-8")
    run = run_replay(dataset_path, replay_path, protocol_path, tmp_path / "results",
                     "v2-check", graph_path=graph_path)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    record = json.loads((run / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["complete_task_success_attempts"] == 1
    assert len(manifest["graph_sha256"]) == 64
    assert "independent_review" not in record["observation"]["turns"][0]


def test_v2_replay_applies_frozen_closure_and_records_input_hashes(tmp_path):
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(_graph(), graph_path)
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(json.dumps({"protocol_version": "scenario-v2",
                                        "split": "development", "cases": [_case()]}), encoding="utf-8")
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(json.dumps({**_observation(["1|4|0", "4|3|0"]), "attempt": 1}) + "\n",
                           encoding="utf-8")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"protocol_version": "scenario-v2"}), encoding="utf-8")
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text('{"edges": []}', encoding="utf-8")
    reviews_path = tmp_path / "reviews.json"
    reviews_path.write_text('{"road_decisions": []}', encoding="utf-8")
    conditions_path = tmp_path / "conditions.json"
    conditions_path.write_text(json.dumps([{"action": "closure", "affected_modes": ["walk"],
                                            "start_time": 100, "end_time": 200,
                                            "edge": {"u": "1", "v": "4", "key": 0}}]),
                               encoding="utf-8")
    run = run_replay(dataset_path, replay_path, protocol_path, tmp_path / "results",
                     "closed", graph_path=graph_path, overrides_path=overrides_path,
                     reviews_path=reviews_path, conditions_path=conditions_path,
                     as_of_epoch=150)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert summary["complete_task_success_attempts"] == 0
    assert len(manifest["overrides_sha256"]) == 64
    assert len(manifest["reviews_sha256"]) == 64
    assert len(manifest["conditions_sha256"]) == 64
    assert manifest["as_of_epoch"] == 150
