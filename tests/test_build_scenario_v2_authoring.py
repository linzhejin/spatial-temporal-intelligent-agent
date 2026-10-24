from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.scenario_split_audit import audit_scenario_splits
from experiments.scenario_v2 import validate_dataset
from scripts.analysis.build_scenario_v2_authoring import build_authoring_bank


ROOT = Path(__file__).resolve().parents[1]


def test_authoring_bank_has_120_source_linked_drafts():
    bank = build_authoring_bank(ROOT / "data/pois.json")
    validate_dataset(bank)
    assert len(bank["cases"]) == 120
    assert Counter(case["category"] for case in bank["cases"]) == {
        group: 24 for group in ("T1", "T2", "T3", "T4", "T5")
    }
    for group in ("T1", "T2", "T3", "T4", "T5"):
        families = Counter(case["template_group"] for case in bank["cases"]
                           if case["category"] == group)
        assert len(families) == 8
        assert set(families.values()) == {3}
    assert len({tuple(message["content"] for message in case["messages"])
                for case in bank["cases"]}) == 120
    assert all(case["adjudication"]["status"] == "draft" for case in bank["cases"])
    assert "graph_sha256" not in bank
    assert len(bank["poi_sha256"]) == 64
    assert all(case.get("candidate_spec", {}).get("activities")
               for case in bank["cases"]
               if any(turn["kind"] == "candidates" for turn in case["expected_turns"]))
    unresolved = {"food_before_route", "study_before_route", "sport_then_food"}
    for case in bank["cases"]:
        if case["category"] == "T5" and case["template_group"].split("v2_T5_")[1] in unresolved:
            assert case["expected_turns"][0]["kind"] == "candidates"
            assert case["expected_turns"][0]["status"] == "needs_input"
            assert "route" not in case["expected_turns"][0]["required_outputs"]


def test_authoring_bank_does_not_pass_formal_leakage_or_gold_gates():
    bank = build_authoring_bank(ROOT / "data/pois.json")
    dev = {"protocol_version": "scenario-v2", "split": "development",
           "cases": [{"id": "dev", "template_group": "dev", "od_group": "dev",
                      "messages": [{"role": "user", "content": "development"}]}]}
    report = audit_scenario_splits(dev, bank)
    assert report["formal_ready"] is False
    assert "test_adjudication_incomplete" in {item["code"] for item in report["blockers"]}


def test_diverse_authoring_bank_spreads_actual_spatial_anchors():
    bank = build_authoring_bank(ROOT / "data/pois.json", diverse_anchors=True)
    validate_dataset(bank)
    assert bank["version"] == "scenarios-test-v2-diverse-authoring-draft"
    for group in ("T1", "T2", "T3", "T4", "T5"):
        cases = [case for case in bank["cases"] if case["category"] == group]
        assert len({next((turn["start_poi_id"] for turn in case["expected_turns"]
                          if turn["start_poi_id"]), None) for case in cases}) == 24
        if group in {"T1", "T4"}:
            assert len({(case["expected_turns"][0]["start_poi_id"],
                         case["expected_turns"][0]["end_poi_id"])
                        for case in cases}) == 24
    assert all(case["adjudication"]["status"] == "draft" for case in bank["cases"])


def test_t5_venue_selection_continues_to_ordered_full_route():
    bank = build_authoring_bank(ROOT / "data/pois.json", diverse_anchors=True)
    pois = {row["id"]: row for row in json.loads((ROOT / "data/pois.json").read_text(
        encoding="utf-8"))["pois"]}
    families = {"food_before_route": {"canteen_meal", "restaurant_meal", "fast_food"},
                "study_before_route": {"study"},
                "sport_then_food": {"basketball"}}
    affected = [case for case in bank["cases"] if case["category"] == "T5"
                and case["template_group"].split("T5_")[1] in families]
    assert len(affected) == 9
    for case in affected:
        family = case["template_group"].split("T5_")[1]
        assert len(case["messages"]) == len(case["expected_turns"]) == 2
        first, final = case["expected_turns"]
        assert first["kind"] == "candidates" and first["status"] == "needs_input"
        assert final["kind"] == "route" and final["status"] == "completed"
        assert final["start_poi_id"] == first["start_poi_id"]
        assert len(final["via_poi_ids"]) == 1
        venue_id = final["via_poi_ids"][0]
        assert set(pois[venue_id]["activities"]) & families[family]
        assert venue_id in case["source_poi_ids"]
        assert final["end_poi_id"] != venue_id
        if family == "sport_then_food":
            assert set(pois[final["end_poi_id"]]["activities"]) & {
                "canteen_meal", "restaurant_meal", "fast_food"}
            assert final["end_poi_id"] in case["source_poi_ids"]
        else:
            assert final["end_poi_id"] == first["end_poi_id"]
    for label, path in (("graph", "data/whu_road_network.graphml"),
                        ("overrides", "data/edge_overrides.json"),
                        ("reviews", "data/campus_review_decisions.json")):
        assert bank[f"source_{label}_sha256"] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_t3_contextual_destination_and_t4_weather_only_followup_are_authored():
    bank = build_authoring_bank(ROOT / "data/pois.json", diverse_anchors=True)
    contextual = [case for case in bank["cases"]
                  if "T3_destination_weather_" in case["id"]]
    weather = [case for case in bank["cases"] if "T4_weather_after_" in case["id"]]
    assert len(contextual) == len(weather) == 3
    for case in contextual:
        assert len(case["messages"]) == len(case["expected_turns"]) == 2
        first, second = case["expected_turns"]
        assert first["kind"] == "clarify" and first["clarification_slot"] == "start"
        assert first["end_poi_id"] == second["end_poi_id"]
        assert second["kind"] == "route" and second["required_outputs"] == ["route", "weather"]
    for case in weather:
        first, second = case["expected_turns"]
        assert first["kind"] == "route"
        assert second["kind"] == "information"
        assert second["required_outputs"] == ["weather"]
        assert second["preserve_route_from_turn"] == 0
