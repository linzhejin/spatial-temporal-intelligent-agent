"""Compute graph-only references for every route turn in visible T3–T5 drafts.

The output is an authoring audit, not independent semantic gold, field access
evidence, or an Agent performance result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.route_reference import build_route_reference
from experiments.scenario_v2 import load_frozen_graph, validate_dataset


ROOT = Path(__file__).resolve().parents[2]


def audit_routes(draft: dict[str, Any], graph, pois: list[dict[str, Any]]) -> dict[str, Any]:
    validate_dataset(draft)
    routes = []
    for case in draft["cases"]:
        if case["category"] not in {"T3", "T4", "T5"}:
            continue
        for turn_index, turn in enumerate(case["expected_turns"]):
            if turn["kind"] != "route":
                continue
            start, end = turn["start_poi_id"], turn["end_poi_id"]
            if not start or not end:
                raise ValueError(f"{case['id']} turn {turn_index}: route endpoint missing")
            if start == end:
                raise ValueError(f"{case['id']} turn {turn_index}: identical endpoints")
            via = turn["via_poi_ids"]
            if not isinstance(via, list):
                raise ValueError(f"{case['id']} turn {turn_index}: ordered via list missing")
            reference = build_route_reference(
                graph, pois, start_poi_id=start, end_poi_id=end,
                via_poi_ids=via, mode=turn["travel_mode"],
                constraints=turn["hard_constraints"],
            )
            routes.append({
                "case_id": case["id"],
                "category": case["category"],
                "turn_index": turn_index,
                "ordered_poi_ids": [start, *via, end],
                "travel_mode": turn["travel_mode"],
                "strategy": turn["strategy"],
                "hard_constraints": turn["hard_constraints"],
                "reference": reference,
            })
    return {
        "status": "visible_authoring_reference_not_gold",
        "route_turn_count": len(routes),
        "by_category": dict(sorted(Counter(row["category"] for row in routes).items())),
        "graph_feasible_count": sum(row["reference"]["graph_feasible"] for row in routes),
        "access_uncertain_count": sum(row["reference"]["access_uncertain"] for row in routes),
        "routes": routes,
    }


def run_audit(draft_path: Path, poi_path: Path, graph_path: Path,
              overrides_path: Path, reviews_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite route audit: {output}")
    paths = {
        "draft": draft_path, "poi": poi_path, "graph": graph_path,
        "overrides": overrides_path, "reviews": reviews_path,
    }
    source_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in paths.items()}
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if (draft.get("version") != "scenarios-test-v2-diverse-authoring-draft"
            or not draft.get("authoring_status")):
        raise ValueError("only the visible diverse authoring draft is accepted")
    for name in ("poi", "graph", "overrides", "reviews"):
        if draft.get(f"source_{name}_sha256") != source_sha256[name]:
            raise ValueError(f"{name} source differs from authoring bank")
    graph = load_frozen_graph(graph_path, overrides_path, reviews_path)
    pois = json.loads(poi_path.read_text(encoding="utf-8"))["pois"]
    report = audit_routes(draft, graph, pois)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != source_sha256[name]
           for name, path in paths.items()):
        raise ValueError("an authoring source changed during route audit")
    report["source_sha256"] = source_sha256
    report["limitations"] = [
        "Visible authored cases have been used for development and are not blind tests.",
        "Graph reachability and nearest-node distance do not establish field passability.",
        "Task semantics, scenic and flat preference optimality, and Agent outputs are not scored.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json")
    parser.add_argument("--pois", type=Path, default=ROOT / "data/pois.json")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/whu_road_network.graphml")
    parser.add_argument("--overrides", type=Path, default=ROOT / "data/edge_overrides.json")
    parser.add_argument("--reviews", type=Path, default=ROOT / "data/campus_review_decisions.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.draft, args.pois, args.graph, args.overrides,
                       args.reviews, args.output)
    print(json.dumps({key: report[key] for key in (
        "route_turn_count", "by_category", "graph_feasible_count",
        "access_uncertain_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
