"""Visible T3–T5 drafts need independent references for every route turn."""

import json
from pathlib import Path

import pytest

from experiments.scenario_v2 import load_frozen_graph


ROOT = Path(__file__).resolve().parents[1]


def test_multiturn_audit_covers_every_ordered_route_turn():
    from scripts.analysis.audit_multiturn_route_references_v2 import audit_routes

    draft = json.loads((ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json")
                       .read_text(encoding="utf-8"))
    graph = load_frozen_graph(ROOT / "data/whu_road_network.graphml",
                              ROOT / "data/edge_overrides.json",
                              ROOT / "data/campus_review_decisions.json")
    pois = json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]

    report = audit_routes(draft, graph, pois)

    assert report["status"] == "visible_authoring_reference_not_gold"
    assert report["route_turn_count"] == 72
    assert report["by_category"] == {"T3": 3, "T4": 45, "T5": 24}
    assert len({(row["case_id"], row["turn_index"]) for row in report["routes"]}) == 72
    cases = {case["id"]: case for case in draft["cases"]}
    for row in report["routes"]:
        turn = cases[row["case_id"]]["expected_turns"][row["turn_index"]]
        assert row["ordered_poi_ids"] == [turn["start_poi_id"],
                                          *turn["via_poi_ids"], turn["end_poi_id"]]
        assert len(row["reference"]["segments_m"]) == len(turn["via_poi_ids"]) + 1


def test_multiturn_audit_rejects_stale_poi_source_before_writing(tmp_path):
    from scripts.analysis.audit_multiturn_route_references_v2 import run_audit

    poi_path = tmp_path / "pois.json"
    poi_path.write_bytes((ROOT / "data/pois.json").read_bytes() + b"\n")
    output = tmp_path / "audit.json"

    with pytest.raises(ValueError, match="poi source differs"):
        run_audit(
            ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json",
            poi_path,
            ROOT / "data/whu_road_network.graphml",
            ROOT / "data/edge_overrides.json",
            ROOT / "data/campus_review_decisions.json",
            output,
        )
    assert not output.exists()


def test_multiturn_audit_rejects_collapsed_route_endpoints():
    from scripts.analysis.audit_multiturn_route_references_v2 import audit_routes

    draft = json.loads((ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json")
                       .read_text(encoding="utf-8"))
    case = next(case for case in draft["cases"]
                if case["id"] == "v2h_T3_destination_weather_1")
    draft["cases"] = [case]
    case["expected_turns"][1]["end_poi_id"] = case["expected_turns"][1]["start_poi_id"]

    with pytest.raises(ValueError, match="identical endpoints"):
        audit_routes(draft, None, [])
