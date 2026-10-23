"""Build a reproducible, graph-decidable development set from visible drafts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.development_graph_reference import build_graph_development_set


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path,
                        default=ROOT / "experiments/datasets/scenarios_test_v2_draft.json")
    parser.add_argument("--route-report", type=Path,
                        default=ROOT / "output/experiments/T1_route_reference_draft_routable_snap.json")
    parser.add_argument("--candidate-report", type=Path,
                        default=ROOT / "output/experiments/T2_candidate_reference_draft.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/experiments/T1_T2_graph_development_v2.json")
    args = parser.parse_args()
    sources = {
        "draft": args.draft,
        "pois": ROOT / "data/pois.json",
        "graph": ROOT / "data/whu_road_network.graphml",
        "overrides": ROOT / "data/edge_overrides.json",
        "reviews": ROOT / "data/campus_review_decisions.json",
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in sources.items()}
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    route = json.loads(args.route_report.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    report = build_graph_development_set(draft, route, candidates, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != content:
        raise FileExistsError("development set already exists with different input hashes")
    args.output.write_text(content, encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]),
                      "excluded": len(report["excluded_cases"]),
                      "categories": {group: sum(c["category"] == group for c in report["cases"])
                                     for group in ("T1", "T2")},
                      "snapshot_id": report["snapshot_id"],
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
