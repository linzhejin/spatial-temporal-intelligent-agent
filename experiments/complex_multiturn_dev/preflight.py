"""Check explicit itinerary anchors and graph feasibility without model calls."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import networkx as nx

from experiments.controlled_route_gold import derive_graph_route
from experiments.independent_source_snapshot import mode_edge_evidence


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/datasets/complex_multiturn_dev_v1"


def main():
    dataset_path = DATA / "cases_event_endpoint_revised.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    references = json.loads((DATA / "reference_route_rubric.json").read_text(encoding="utf-8"))["cases"]
    anchor_data = json.loads((DATA / "anchor_snapshot_60m.json").read_text(encoding="utf-8"))
    graph_path, poi_path = ROOT / "data/whu_road_network.graphml", ROOT / "data/pois.json"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if anchor_data["graph_sha256"] != sha(graph_path) or anchor_data["poi_sha256"] != sha(poi_path):
        raise ValueError("graph or POI snapshot changed")
    graph = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
    anchors = {row["poi_id"]: row for row in anchor_data["anchors"]}
    evidence = {mode: mode_edge_evidence(graph, mode, require_boundary=True,
                                         boundary_wkt=anchor_data["boundary_wkt"])
                for mode in ("walk", "bike", "drive")}
    rows = []
    for case in data["cases"]:
        if case["group"] not in {"itinerary", "clarification"}:
            continue
        reference = references.get(case["id"])
        if reference and reference["quoted_basis"] not in case["turns"][0]["user"]:
            raise ValueError("reference rubric does not match the question")
        state = {"origin": reference["reference_origin"]} if reference else {}
        for number, turn in enumerate(case["turns"], 1):
            checks = {c["field"]: c["value"] for c in turn["checks"]
                      if c["op"] in {"eq", "ordered_contains"} and "value" in c}
            state.update({key: checks[key] for key in ("origin", "mode", "avoid_steps") if key in checks})
            stops = checks.get("stops")
            if not isinstance(stops, list) or not stops:
                continue
            required = [state.get("origin"), *stops]
            if not state.get("mode") or any(ident not in anchors for ident in required):
                rows.append({"case": case["id"], "turn": number, "verdict": "unknown",
                             "reason": "origin_mode_or_anchor_missing"})
                continue
            result = derive_graph_route(graph, start_node=anchors[required[0]]["node_id"],
                end_node=anchors[required[-1]]["node_id"],
                via_nodes=[anchors[ident]["node_id"] for ident in required[1:-1]], mode=state["mode"],
                permitted_edge_ids=evidence[state["mode"]]["permitted"],
                unknown_edge_ids=evidence[state["mode"]]["unknown"], closed_edge_ids=set(),
                avoid_steps=bool(state.get("avoid_steps")))
            rows.append({"case": case["id"], "turn": number, "origin": required[0],
                         "stops": stops, "mode": state["mode"],
                         "avoid_steps": bool(state.get("avoid_steps")), **result})
    result = {"scope": "development_only_frozen_graph", "dataset_sha256": sha(dataset_path),
              "graph_sha256": sha(graph_path), "poi_sha256": sha(poi_path),
              "boundary": "geometric anchors only; entrances, last-meter access, candidate semantics and opening unverified",
              "checks": rows}
    path = DATA / "itinerary_preflight_v2.json"
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError("preflight already exists with different evidence")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"output": str(path), "counts": dict(Counter(row["verdict"] for row in rows))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
