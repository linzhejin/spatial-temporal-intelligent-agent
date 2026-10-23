"""Prepare independent graph-distance POI references for T2 authoring drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.candidate_reference import build_candidate_reference, rank_nearby_activities
from experiments.scenario_v2 import load_frozen_graph


ROOT = Path(__file__).resolve().parents[2]
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path,
                        default=ROOT / "experiments/datasets/scenarios_test_v2_draft.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/experiments/T2_candidate_reference_draft.json")
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    if draft.get("version") != "scenarios-test-v2-authoring-draft":
        raise SystemExit("only the v2 authoring draft is accepted")
    paths = {
        "pois": ROOT / "data/pois.json",
        "graph": ROOT / "data/whu_road_network.graphml",
        "overrides": ROOT / "data/edge_overrides.json",
        "reviews": ROOT / "data/campus_review_decisions.json",
    }
    source_sha = {name: _sha256(path) for name, path in paths.items()}
    graph = load_frozen_graph(paths["graph"], paths["overrides"], paths["reviews"])
    pois = json.loads(paths["pois"].read_text(encoding="utf-8"))["pois"]
    references = []
    for case in draft["cases"]:
        if case.get("category") != "T2":
            continue
        family = case["template_group"].removeprefix("v2_T2_")
        start_poi_id = case["expected_turns"][0]["start_poi_id"]
        spec = case["candidate_spec"]
        result = rank_nearby_activities(graph, pois, start_poi_id=start_poi_id,
                                        activities=set(spec["activities"]),
                                        coordinates=spec["coordinates"])
        listed = result["reachable"]
        boundary_gap = (listed[4]["network_distance_m"] - listed[3]["network_distance_m"]
                        if len(listed) > 4 else None)
        references.append({"case_id": case["id"], "activity_family": family,
                           "start_poi_id": start_poi_id,
                           "top_four": listed[:4],
                           "proposed_candidate_reference": {
                               "graph_sha256": source_sha["graph"],
                               **build_candidate_reference(
                                   graph, pois, start_poi_id=start_poi_id, spec=spec),
                           },
                           "eligible_reachable": len(listed),
                           "eligible_unreachable": len(result["unreachable"]),
                           "fourth_fifth_gap_m": round(boundary_gap, 2)
                           if boundary_gap is not None else None,
                           "review_needed": any(row["access_over_30m"] for row in listed[:4])
                           or boundary_gap is not None and boundary_gap < 30})
    if len(references) != 24:
        raise SystemExit("expected 24 T2 authoring cases")
    report = {"status": "development_reference_only_not_adjudicated",
              "method": "independent_node_graph_walk_distance_plus_straight_access",
              "source_sha256": source_sha,
              "cases": references}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"cases": len(references),
                      "needs_review": sum(row["review_needed"] for row in references),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
