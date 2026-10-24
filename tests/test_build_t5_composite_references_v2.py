"""T5 authoring cases must score the entire selected route in visit order."""

import json
from pathlib import Path

from experiments.scenario_v2 import load_frozen_graph
from scripts.analysis.build_scenario_v2_authoring import build_authoring_bank


ROOT = Path(__file__).resolve().parents[1]


def test_t5_authoring_reference_covers_selection_and_both_route_legs():
    from scripts.analysis.build_t5_composite_references_v2 import audit_composites

    draft = build_authoring_bank(ROOT / "data/pois.json", diverse_anchors=True)
    graph = load_frozen_graph(ROOT / "data/whu_road_network.graphml",
                              ROOT / "data/edge_overrides.json",
                              ROOT / "data/campus_review_decisions.json")
    pois = json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]
    report = audit_composites(draft, graph, pois)
    assert report["status"] == "development_reference_only_not_agent_score"
    assert report["case_count"] == 9
    assert report["meal_selected_in_first_four_count"] == 3
    for case in report["cases"]:
        assert case["selected_in_first_four"] is True
        assert case["route_reference"]["graph_feasible"] is True
        assert len(case["route_reference"]["segments_m"]) == 2
        assert case["route_reference"]["access_uncertain"] is False
        if "sport_then_food" in case["case_id"]:
            assert case["meal_selected_in_first_four"] is True
            assert case["meal_named_in_final_followup"] is True


def test_t5_audit_rejects_first_turn_without_final_route():
    from scripts.analysis.build_t5_composite_references_v2 import audit_composites

    draft = build_authoring_bank(ROOT / "data/pois.json", diverse_anchors=True)
    target = next(case for case in draft["cases"] if case["id"] == "v2h_T5_food_before_route_1")
    target["messages"] = target["messages"][:1]
    target["expected_turns"] = target["expected_turns"][:1]
    graph = load_frozen_graph(ROOT / "data/whu_road_network.graphml",
                              ROOT / "data/edge_overrides.json",
                              ROOT / "data/campus_review_decisions.json")
    pois = json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]
    import pytest

    with pytest.raises(ValueError, match="full route"):
        audit_composites(draft, graph, pois)
