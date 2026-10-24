"""Audit selected T5 venues and their entire ordered routes on independent inputs.

This examines visible authoring drafts only. It never scores an Agent response
or establishes that a campus path is physically passable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.candidate_reference import rank_nearby_activities
from experiments.route_reference import build_route_reference
from experiments.scenario_v2 import (candidate_spec_for_turn, load_frozen_graph,
                                     validate_dataset)


ROOT = Path(__file__).resolve().parents[2]


def audit_composites(draft: dict[str, Any], graph, pois: list[dict[str, Any]]) -> dict[str, Any]:
    validate_dataset(draft)
    cases = [case for case in draft["cases"] if case["category"] == "T5"
             and case["expected_turns"][0]["kind"] == "candidates"]
    if len(cases) != 9:
        raise ValueError("expected nine T5 venue-selection cases")
    poi_names = {poi["id"]: poi["name"] for poi in pois}
    results = []
    for case in cases:
        turns = case["expected_turns"]
        sequential_food = case["template_group"].endswith("sport_then_food")
        expected_count = 3 if sequential_food else 2
        if (len(case["messages"]) != expected_count or len(turns) != expected_count
                or turns[-1]["kind"] != "route" or turns[-1]["status"] != "completed"
                or len(turns[-1]["via_poi_ids"]) != 1
                or turns[0]["start_poi_id"] != turns[-1]["start_poi_id"]):
            raise ValueError(f"{case['id']}: selection must continue to an ordered full route")
        first, final = turns[0], turns[-1]
        selected = final["via_poi_ids"][0]
        spec = candidate_spec_for_turn(case, 0)
        if not isinstance(spec, dict):
            raise ValueError(f"{case['id']}: candidate spec missing")
        ranked = rank_nearby_activities(
            graph, pois, start_poi_id=first["start_poi_id"],
            activities=set(spec["activities"]), coordinates=spec["coordinates"])
        top = ranked["reachable"][:spec["max_display"]]
        meal_selected_in_first_four = None
        if sequential_food:
            meal = turns[1]
            meal_spec = candidate_spec_for_turn(case, 1)
            if (meal["kind"] != "candidates" or meal["status"] != "needs_input"
                    or meal["start_poi_id"] != selected
                    or not isinstance(meal_spec, dict)):
                raise ValueError(f"{case['id']}: meal search must start at selected sport venue")
            meal_ranked = rank_nearby_activities(
                graph, pois, start_poi_id=selected,
                activities=set(meal_spec["activities"]),
                coordinates=meal_spec["coordinates"])
            meal_selected_in_first_four = final["end_poi_id"] in {
                row["poi_id"] for row in meal_ranked["reachable"][:meal_spec["max_display"]]
            }
        reference = build_route_reference(
            graph, pois, start_poi_id=final["start_poi_id"],
            end_poi_id=final["end_poi_id"], via_poi_ids=final["via_poi_ids"],
            mode=final["travel_mode"], constraints=final["hard_constraints"],
            coordinates=spec["coordinates"])
        results.append({
            "case_id": case["id"],
            "selected_venue_id": selected,
            "selected_in_first_four": selected in {row["poi_id"] for row in top},
            "selected_venue_named_in_followup": poi_names[selected] in case["messages"][1]["content"],
            "meal_selected_in_first_four": meal_selected_in_first_four,
            "meal_named_in_final_followup": (
                poi_names[final["end_poi_id"]] in case["messages"][-1]["content"]
                if sequential_food else None),
            "first_four_poi_ids": [row["poi_id"] for row in top],
            "eligible_reachable": len(ranked["reachable"]),
            "ordered_poi_ids": [final["start_poi_id"], *final["via_poi_ids"],
                                final["end_poi_id"]],
            "route_reference": reference,
        })
    return {
        "status": "development_reference_only_not_agent_score",
        "case_count": len(results),
        "selected_in_first_four_count": sum(row["selected_in_first_four"] for row in results),
        "meal_selected_in_first_four_count": sum(row["meal_selected_in_first_four"] is True
                                                 for row in results),
        "graph_feasible_count": sum(row["route_reference"]["graph_feasible"] for row in results),
        "access_uncertain_count": sum(row["route_reference"]["access_uncertain"] for row in results),
        "cases": results,
        "limitations": [
            "These routes are authoring expectations, not observed Agent behavior.",
            "Nearest-node graph reachability does not establish physical access or field passability.",
            "The visible authoring bank is not a blinded formal test set.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--pois", type=Path, default=ROOT / "data/pois.json")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/whu_road_network.graphml")
    parser.add_argument("--overrides", type=Path, default=ROOT / "data/edge_overrides.json")
    parser.add_argument("--reviews", type=Path, default=ROOT / "data/campus_review_decisions.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = {name: path.read_bytes() for name, path in (
        ("draft", args.draft), ("poi", args.pois), ("graph", args.graph),
        ("overrides", args.overrides), ("reviews", args.reviews))}
    draft = json.loads(raw["draft"])
    if draft.get("authoring_status") is None:
        raise ValueError("only visible authoring drafts are accepted")
    for name in ("poi", "graph", "overrides", "reviews"):
        if draft.get(f"source_{name}_sha256") != hashlib.sha256(raw[name]).hexdigest():
            raise ValueError(f"{name} source differs from authoring bank")
    pois = json.loads(raw["poi"])["pois"]
    graph = load_frozen_graph(args.graph, args.overrides, args.reviews)
    report = audit_composites(draft, graph, pois)
    report["source_sha256"] = {name: hashlib.sha256(value).hexdigest()
                               for name, value in raw.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "case_count", "selected_in_first_four_count", "graph_feasible_count",
        "access_uncertain_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
