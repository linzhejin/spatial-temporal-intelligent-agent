"""Independently audit archived visible T5 model-pilot routes against the graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.route_reference import build_route_reference
from experiments.scenario_v2 import audit_route_edges, load_frozen_graph


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    data_paths = {
        "data/pois.json": ROOT / "data/pois.json",
        "data/whu_road_network.graphml": ROOT / "data/whu_road_network.graphml",
        "data/edge_overrides.json": ROOT / "data/edge_overrides.json",
        "data/campus_review_decisions.json": ROOT / "data/campus_review_decisions.json",
    }
    graph = load_frozen_graph(
        data_paths["data/whu_road_network.graphml"],
        data_paths["data/edge_overrides.json"],
        data_paths["data/campus_review_decisions.json"],
    )
    pois = json.loads(data_paths["data/pois.json"].read_text(encoding="utf-8"))["pois"]
    rows = []
    for path in args.observation:
        observation = json.loads(path.read_text(encoding="utf-8"))
        if observation.get("status") != "development_pilot_only":
            raise ValueError(f"not a development pilot: {path}")
        for name, source in data_paths.items():
            if observation.get("data_component_sha256", {}).get(name) != _sha256(source):
                raise ValueError(f"data hash mismatch for {name}: {path}")
        if len(observation.get("turns", [])) != 1:
            raise ValueError(f"expected one completed turn: {path}")
        turn = observation["turns"][0]
        if turn.get("status") != "completed" or turn.get("kind") != "route":
            raise ValueError(f"pilot did not complete a route: {path}")
        reference = build_route_reference(
            graph, pois, start_poi_id=turn["start_poi_id"],
            end_poi_id=turn["end_poi_id"], via_poi_ids=turn["via_poi_ids"],
            mode=turn["travel_mode"], constraints=turn["hard_constraints"],
        )
        audit = audit_route_edges(
            graph, turn["route_edge_ids"], reference["start_node"],
            reference["end_node"], reference["via_nodes"],
            turn["travel_mode"], turn["hard_constraints"],
        )
        rows.append({
            "case_id": observation["case_id"],
            "observation_file": path.name,
            "observation_sha256": _sha256(path),
            "start_poi_id": turn["start_poi_id"],
            "via_poi_ids": turn["via_poi_ids"],
            "end_poi_id": turn["end_poi_id"],
            "edge_count": len(turn["route_edge_ids"]),
            "route_audit": audit,
            "access_uncertain": reference["access_uncertain"],
        })
    report = {
        "status": "visible_development_pilot_graph_audit",
        "data_sha256": {name: _sha256(path) for name, path in data_paths.items()},
        "cases": rows,
        "all_graph_routes_pass": all(row["route_audit"]["route"] == "pass" for row in rows),
        "limitations": ["graph audit is not field passability",
                        "nearest-node access can be uncertain", "not a formal holdout score"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"cases": len(rows),
                      "route_verdicts": [row["route_audit"]["route"] for row in rows],
                      "access_uncertain": [row["access_uncertain"] for row in rows]}))
    return 0 if report["all_graph_routes_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
