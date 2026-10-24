"""Visible authoring references must never be promoted to formal gold."""

import pytest


def _case(case_id, category, turns):
    return {
        "id": case_id,
        "category": category,
        "expected_turns": turns,
        "adjudication": {"status": "draft"},
    }


def _route(start, end, via=()):
    return {"kind": "route", "start_poi_id": start,
            "end_poi_id": end, "via_poi_ids": list(via)}


def test_visible_turn_ledger_keeps_graph_reference_and_unknown_separate():
    from scripts.analysis.audit_v2_turn_evidence_coverage import audit_turns

    dataset = {"cases": [
        _case("t1", "T1", [_route("a", "b")]),
        _case("t2", "T2", [{"kind": "candidates"}]),
        _case("t3", "T3", [{"kind": "clarify"}, _route("a", "c")]),
        _case("t4", "T4", [_route("a", "d"), {"kind": "information"}]),
        _case("t5", "T5", [{"kind": "candidates"}, {"kind": "candidates"},
                             _route("a", "f", ("e",))]),
    ]}
    t1 = {"cases": [{"case_id": "t1", "proposed_route_reference": {
        "graph_feasible": True, "access_uncertain": False}}]}
    t2 = {"cases": [{"case_id": "t2", "proposed_candidate_reference": {}}]}
    uncertainty = {"cases": [{"case_id": "t2",
                               "reference_top_four_verdict": "unknown"}]}
    multi = {"routes": [
        {"case_id": "t3", "turn_index": 1, "ordered_poi_ids": ["a", "c"],
         "reference": {"graph_feasible": True, "access_uncertain": False}},
        {"case_id": "t4", "turn_index": 0, "ordered_poi_ids": ["a", "d"],
         "reference": {"graph_feasible": True, "access_uncertain": False}},
        {"case_id": "t5", "turn_index": 2, "ordered_poi_ids": ["a", "e", "f"],
         "reference": {"graph_feasible": True, "access_uncertain": False}},
    ]}
    t5 = {"cases": [{"case_id": "t5", "selected_in_first_four": True,
                     "meal_selected_in_first_four": True,
                     "ordered_poi_ids": ["a", "e", "f"]}]}

    report = audit_turns(dataset, t1, t2, uncertainty, multi, t5)

    assert report["turn_count"] == 9
    assert report["by_reference_scope"] == {
        "graph_route": 4, "graph_candidate_rank": 1,
        "candidate_selection_membership": 2, "none": 2,
    }
    assert report["formal_adjudicated_cases"] == 0
    rows = {(row["case_id"], row["turn_index"]): row for row in report["turns"]}
    assert rows[("t2", 0)]["reference_verdict"] == "unknown"
    assert rows[("t3", 0)]["reference_scope"] == "none"
    assert rows[("t5", 1)]["reference_verdict"] == "selected_in_first_four"
    assert rows[("t5", 2)]["reference_verdict"] == "graph_feasible"


def test_turn_ledger_rejects_stale_ordered_route_reference():
    from scripts.analysis.audit_v2_turn_evidence_coverage import audit_turns

    dataset = {"cases": [_case("t3", "T3", [_route("a", "b", ("via",))])]}
    multi = {"routes": [{"case_id": "t3", "turn_index": 0,
                         "ordered_poi_ids": ["a", "b"],
                         "reference": {"graph_feasible": True,
                                       "access_uncertain": False}}]}

    with pytest.raises(ValueError, match="ordered POIs differ"):
        audit_turns(dataset, {"cases": []}, {"cases": []}, {"cases": []},
                    multi, {"cases": []})
