"""Build source-hashed graph references for 24 authored T1 route questions.

The output stays an authoring report. It does not turn a visible draft into a
blind test set or verify walkability outside the frozen graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.route_reference import build_route_reference
from experiments.scenario_v2 import load_frozen_graph


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path,
                        default=ROOT / "experiments/datasets/scenarios_test_v2_draft.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/experiments/T1_route_reference_draft_routable_snap.json")
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    if draft.get("version") != "scenarios-test-v2-authoring-draft":
        raise SystemExit("only the visible v2 authoring draft is accepted")
    paths = {
        "draft": args.draft,
        "pois": ROOT / "data/pois.json",
        "graph": ROOT / "data/whu_road_network.graphml",
        "overrides": ROOT / "data/edge_overrides.json",
        "reviews": ROOT / "data/campus_review_decisions.json",
    }
    source_sha = {name: _sha256(path) for name, path in paths.items()}
    graph = load_frozen_graph(paths["graph"], paths["overrides"], paths["reviews"])
    pois = json.loads(paths["pois"].read_text(encoding="utf-8"))["pois"]
    rows = []
    for case in draft["cases"]:
        if case.get("category") != "T1":
            continue
        if len(case["expected_turns"]) != 1:
            raise ValueError(f"T1 single-turn reference expected: {case['id']}")
        turn = case["expected_turns"][0]
        ref = build_route_reference(
            graph, pois, start_poi_id=turn["start_poi_id"],
            end_poi_id=turn["end_poi_id"], via_poi_ids=turn["via_poi_ids"],
            mode=turn["travel_mode"], constraints=turn["hard_constraints"],
        )
        rows.append({"case_id": case["id"], "mode": turn["travel_mode"],
                     "constraints": turn["hard_constraints"],
                     "proposed_route_reference": ref,
                     "needs_review": ref["access_uncertain"] or not ref["graph_feasible"]})
    if len(rows) != 24:
        raise ValueError("expected 24 T1 authoring cases")
    report = {"status": "development_reference_only_not_adjudicated",
              "source_sha256": source_sha, "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != content:
            raise FileExistsError("route reference exists with different source or content")
    else:
        args.output.write_text(content, encoding="utf-8")
    print(json.dumps({"cases": len(rows),
                      "graph_feasible": sum(row["proposed_route_reference"]["graph_feasible"]
                                            for row in rows),
                      "needs_review": sum(row["needs_review"] for row in rows),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
