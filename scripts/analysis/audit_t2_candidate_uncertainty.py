"""Separate graph-decidable T2 authoring references from uncertain access.

The output audits the independently proposed top four, not any Agent response.
It is a development artifact and never constitutes formal task gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.scenario_v2 import audit_candidate_order


ROOT = Path(__file__).resolve().parents[2]


def audit_reference(report: dict, pois: dict[str, str]) -> dict:
    rows = report.get("cases")
    if report.get("status") != "development_reference_only_not_adjudicated" or not isinstance(rows, list):
        raise ValueError("a T2 development reference report is required")
    if len(rows) != 24 or len({row.get("case_id") for row in rows}) != 24:
        raise ValueError("expected 24 distinct T2 authoring references")
    cases = []
    for row in rows:
        reference = row["proposed_candidate_reference"]
        top_four = [item["poi_id"] for item in row["top_four"]]
        verdict = audit_candidate_order(top_four, reference)
        if verdict["verdict"] not in {"pass", "unknown"}:
            raise ValueError(f"independent proposed top four failed: {row['case_id']}")
        limit = max(reference["distances_m"][poi_id] for poi_id in top_four)
        limit += reference["rank_tolerance_m"]
        affected = [poi_id for poi_id in reference["uncertain_access_ids"]
                    if reference["distances_m"][poi_id] <= limit]
        if verdict["verdict"] == "unknown" and not affected:
            raise ValueError(f"unknown ranking has no affected access: {row['case_id']}")
        cases.append({
            "case_id": row["case_id"],
            "activity_family": row["activity_family"],
            "reference_top_four_verdict": verdict["verdict"],
            "reason": verdict["reason"],
            "authoring_review_flag": row["review_needed"],
            "fourth_fifth_gap_m": row["fourth_fifth_gap_m"],
            "affected_uncertain_pois": [
                {"poi_id": poi_id, "name": pois[poi_id],
                 "estimated_distance_m": reference["distances_m"][poi_id]}
                for poi_id in affected
            ],
        })
    counts = Counter(case["reference_top_four_verdict"] for case in cases)
    unique_pois = {item["poi_id"]: item["name"] for case in cases
                   for item in case["affected_uncertain_pois"]}
    return {
        "status": "development_reference_uncertainty_only_not_agent_score",
        "case_count": 24,
        "reference_top_four_verdict_counts": dict(sorted(counts.items())),
        "affected_unique_pois": [
            {"poi_id": poi_id, "name": unique_pois[poi_id]}
            for poi_id in sorted(unique_pois)
        ],
        "cases": cases,
        "limitations": [
            "The graph-computed top four is an authoring reference, not an observed Agent answer.",
            "Pass means graph-distance consistency under the specified tolerance, not real-world access.",
            "Unknown access remains unknown until independent evidence or a different decidable case is available.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pois", type=Path, default=ROOT / "data/pois.json")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/whu_road_network.graphml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_bytes = args.reference.read_bytes()
    report = json.loads(report_bytes)
    poi_bytes = args.pois.read_bytes()
    graph_bytes = args.graph.read_bytes()
    expected = report.get("source_sha256") or {}
    if (hashlib.sha256(poi_bytes).hexdigest() != expected.get("pois")
            or hashlib.sha256(graph_bytes).hexdigest() != expected.get("graph")):
        raise ValueError("POI or graph bytes differ from reference sources")
    pois = {item["id"]: item["name"] for item in json.loads(poi_bytes)["pois"]}
    result = audit_reference(report, pois)
    result["reference_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    result["poi_sha256"] = expected["pois"]
    result["graph_sha256"] = expected["graph"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({"output": str(args.output),
                      "counts": result["reference_top_four_verdict_counts"],
                      "unique_uncertain_pois": len(result["affected_unique_pois"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
