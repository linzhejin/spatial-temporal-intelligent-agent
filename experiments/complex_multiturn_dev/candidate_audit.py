"""Audit delivered candidate distances without relying on product distance claims."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import networkx as nx

from experiments.complex_multiturn_dev.candidate_checks import inspect_candidate_scope, inspect_candidate_time, provider_anchor, time_distance_limit
from experiments.independent_source_snapshot import mode_edge_evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/datasets/complex_multiturn_dev_v1"


def scope_radius_for_audit(observed):
    radius = observed.get("radius_m")
    same_center = (observed.get("search_anchor") or observed.get("origin")) == observed.get("origin")
    if (type(observed.get("max_duration_s")) in (int, float) and same_center and radius == 1000
            and observed.get("search_radius_explicit") is not True):
        return None
    return radius


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--time", action="store_true", help="independently audit time from the actual origin")
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.case not in manifest.get("case_ids", []):
        raise ValueError("case not in this run")
    snapshot = json.loads((DATA / "anchor_snapshot_60m.json").read_text(encoding="utf-8"))
    graph_path, poi_path = ROOT / "data/whu_road_network.graphml", ROOT / "data/pois.json"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    checker_sha = sha(ROOT / "experiments/complex_multiturn_dev/candidate_checks.py")
    if snapshot["graph_sha256"] != sha(graph_path) or snapshot["poi_sha256"] != sha(poi_path):
        raise ValueError("frozen graph or POI changed")
    graph = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
    anchors = {row["poi_id"]: row for row in snapshot["anchors"]}
    evidence = {mode: mode_edge_evidence(graph, mode, require_boundary=True, boundary_wkt=snapshot["boundary_wkt"])
                for mode in ("walk", "bike", "drive")}
    from shapely import wkt
    from shapely.geometry import Point
    from spatial.coord_transform import gcj02_to_wgs84
    boundary = wkt.loads(snapshot["boundary_wkt"])
    edge_nodes = {node for u, v, _ in evidence["walk"]["permitted"] for node in (u, v)}
    permitted_nodes = {node for node in edge_nodes if boundary.covers(
        Point(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))) }
    provider_path = args.run_dir / "provider_evidence.jsonl"
    provider = [json.loads(line) for line in provider_path.read_text(encoding="utf-8").splitlines()] if provider_path.exists() else []
    source = args.run_dir / "restored_observations.jsonl"
    if not source.exists():
        source = args.run_dir / "turns.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()] if source.exists() else []
    extra = args.run_dir / "recovered_uncollected.jsonl"
    if extra.exists():
        rows.extend(json.loads(line) for line in extra.read_text(encoding="utf-8").splitlines())
    output = args.run_dir / (args.case + ("_candidate_time_audit.jsonl" if args.time else "_candidate_scope_audit.jsonl"))
    counts = Counter()
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            if row.get("case") != args.case:
                continue
            obs, result = row.get("observed") or {}, row.get("result") or {}
            if "search_radius_explicit" not in obs:
                request = result.get("place_request") or result.get("completed_place_request") or {}
                semantic = request.get("semantic_request") if isinstance(request, dict) else None
                if isinstance(semantic, dict) and type(semantic.get("search_radius_explicit")) is bool:
                    obs = {**obs, "search_radius_explicit": semantic["search_radius_explicit"]}
            center = obs.get("search_anchor") or obs.get("origin")
            candidates = result.get("candidates") or []
            for candidate in candidates:
                ident = candidate.get("poi_id")
                candidate_anchors = anchors
                if ident not in anchors and str(ident).startswith("amap:"):
                    raw = [poi for record in provider if record.get("case") == args.case
                           and type(record.get("turn")) is int and record["turn"] <= row["turn"]
                           for poi in record.get("pois") or []]
                    found = provider_anchor(graph, permitted_nodes, candidate, raw, to_wgs84=gcj02_to_wgs84)
                    if found:
                        candidate_anchors = {**anchors, ident: found}
                if args.time:
                    distance_limit = time_distance_limit(obs)
                    check = inspect_candidate_time(graph, candidate_anchors, evidence, obs.get("origin"), ident,
                                                   obs.get("mode"), obs.get("max_duration_s"), radius=distance_limit,
                                                   avoid_steps=bool(obs.get("avoid_steps")))
                else:
                    check = inspect_candidate_scope(graph, candidate_anchors, evidence, center, ident, obs.get("mode"),
                                                    scope_radius_for_audit(obs), avoid_steps=bool(obs.get("avoid_steps")))
                counts[check["verdict"]] += 1
                stream.write(json.dumps({"case": args.case, "turn": row["turn"], "center": center,
                    "candidate": ident, "radius_m": obs.get("radius_m"), "checker_sha256": checker_sha,
                    **check}, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "candidate_counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
