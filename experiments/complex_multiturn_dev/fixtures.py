"""Construct frozen, explicitly synthetic event cases on the campus graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx

from experiments.build_controlled_spatial_snapshot import build_spatial_snapshot
from experiments.controlled_route_gold import derive_graph_route
from experiments.independent_source_snapshot import active_closed_edges, mode_edge_evidence


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/datasets/complex_multiturn_dev_v1/event_fixtures_v2.json"
DATASET = ROOT / "experiments/datasets/complex_multiturn_dev_v1/cases_event_endpoint_revised.json"
OD = {
    "events_01": ("poi_418", "poi_004", "walk", "detour"),
    "events_02": ("poi_001", "poi_423", "walk", "blocked"),
    "events_03": ("poi_072", "poi_004", "bike", "bike_only"),
    "events_04": ("poi_140", "poi_418", "walk", "direction"),
    "events_05": ("poi_424", "poi_257", "walk", "off_path"),
    "events_06": ("poi_418", "poi_004", "walk", "lifecycle"),
    "events_07": ("poi_418", "poi_004", "bike", "bike_only"),
    "events_08": ("poi_140", "poi_418", "walk", "detour"),
}
START, END = 1792962000, 1792969200
TIMES = {"before": START - 1, "active": START, "expired": END, "revoked": END - 1}


def _pair(graph, edge):
    u, v, key = edge
    result = {tuple(edge)}
    if graph.has_edge(v, u):
        result.update((v, u, reverse_key) for reverse_key in graph[v][u])
    return result


def _derive(graph, anchors, evidence, start, end, mode, closed):
    return derive_graph_route(graph, start_node=anchors[start]["node_id"],
                              end_node=anchors[end]["node_id"], via_nodes=[], mode=mode,
                              permitted_edge_ids=evidence[mode]["permitted"],
                              unknown_edge_ids=evidence[mode]["unknown"],
                              closed_edge_ids=set(closed))


def choose_detour(graph, anchors, evidence, start, end, mode, baseline):
    """Return a real old-route edge set with an independently feasible detour."""
    for raw in baseline["route_edge_ids"]:
        closed = _pair(graph, raw)
        result = _derive(graph, anchors, evidence, start, end, mode, closed)
        if (result["verdict"] == "feasible" and
                result["length_m"] > baseline["length_m"] + 0.01):
            return closed, result
    raise ValueError("no independently confirmed detour for the chosen endpoints")


def _rows(graph, case_id, edges, mode, *, direction=False):
    from spatial.coord_transform import wgs84_to_gcj02

    rows = []
    for index, (u, v, key) in enumerate(sorted(edges), 1):
        edge = {"u": u, "v": v, "key": key}
        if direction:
            coords = [(graph_node["x"], graph_node["y"]) for graph_node in
                      (graph.nodes[u], graph.nodes[v])]
            edge["geometry_gcj"] = [list(wgs84_to_gcj02(float(x), float(y))) for x, y in coords]
        row = {"id": f"{case_id}_edge_{index}", "type": "closure", "action": "closure",
               "name": "模拟道路限制（仅隔离测试）", "affected_modes": [mode], "edge": edge,
               "start_time": START, "end_time": END,
               "source": {"kind": "research_simulated", "scope": "synthetic_development_only",
                          "campus_truth": False}}
        if direction:
            row["direction"] = "forward"
        rows.append(row)
    return rows


def build():
    from config import CAMPUS_POLYS_GCJ
    from spatial.coord_transform import gcj02_to_wgs84
    from spatial.road_conditions import _validate_condition_record

    graph_path = ROOT / "data/whu_road_network.graphml"
    poi_path = ROOT / "data/pois.json"
    dataset_bytes = DATASET.read_bytes()
    cases = {case["id"]: case for case in json.loads(dataset_bytes)["cases"]}
    graph = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
    pois = json.loads(poi_path.read_text(encoding="utf-8"))["pois"]
    boundary, snapshot = build_spatial_snapshot(graph, pois, CAMPUS_POLYS_GCJ,
                                                 to_wgs84=gcj02_to_wgs84, max_snap_m=60)
    anchors = {item["poi_id"]: item for item in snapshot["anchors"]}
    evidence = {mode: mode_edge_evidence(graph, mode, require_boundary=True, boundary_wkt=boundary)
                for mode in ("walk", "bike")}
    fixtures = []
    for ident, (start, end, mode, kind) in OD.items():
        first_turn = cases[ident]["turns"][0]
        intended_origins = {item["value"] for item in first_turn["checks"]
                            if item["field"] == "origin" and item["op"] == "eq"}
        poi = next((row for row in pois if row["id"] == end), None)
        names = [poi.get("name"), *(poi.get("aliases") or [])] if poi else []
        if intended_origins != {start} or not any(name and name in first_turn["user"] for name in names):
            raise ValueError(f"{ident}: intended endpoints do not match the question")
        if start not in anchors or end not in anchors:
            raise ValueError(f"{ident}: endpoint lacks a <=60 m graph anchor")
        baseline = _derive(graph, anchors, evidence, start, end, mode, set())
        if baseline["verdict"] != "feasible":
            raise ValueError(f"{ident}: baseline not independently feasible: {baseline['reason']}")
        if kind == "blocked":
            node = anchors[start]["node_id"]
            candidates = set(graph.out_edges(node, keys=True)) | set(graph.in_edges(node, keys=True))
            closed = set(candidates)
            changed = _derive(graph, anchors, evidence, start, end, mode, closed)
            if changed["verdict"] != "infeasible":
                raise ValueError(f"{ident}: closure is not independently infeasible")
        elif kind == "off_path":
            route_nodes = {node for edge in baseline["route_edge_ids"] for node in edge[:2]}
            center = graph.nodes[anchors[start]["node_id"]]
            candidates = [edge for edge in evidence[mode]["permitted"]
                          if edge[0] not in route_nodes and edge[1] not in route_nodes
                          and abs(float(graph.nodes[edge[0]]["x"]) - float(center["x"])) > 0.006]
            if not candidates:
                raise ValueError(f"{ident}: no sufficiently distant event edge")
            closed = _pair(graph, sorted(candidates)[0])
            changed = _derive(graph, anchors, evidence, start, end, mode, closed)
            if changed["verdict"] != "feasible" or changed["length_m"] != baseline["length_m"]:
                raise ValueError(f"{ident}: off-path event changes independently optimal length")
        else:
            closed, changed = choose_detour(graph, anchors, evidence, start, end, mode, baseline)
        if kind == "direction":
            forward = next(tuple(edge) for edge in baseline["route_edge_ids"] if tuple(edge) in closed)
            closed = {forward}
            changed = _derive(graph, anchors, evidence, start, end, mode, closed)
        rows = _rows(graph, ident, closed, mode, direction=kind == "direction")
        for row in rows:
            _validate_condition_record(row)
        stages = {}
        for stage, epoch in TIMES.items():
            stage_rows = [dict(row) for row in rows]
            if stage == "revoked":
                stage_rows = [{**row, "revoked_at": epoch} for row in stage_rows]
            state = {"as_of_epoch": epoch, "conditions": stage_rows}
            edge_set = active_closed_edges(graph, state, mode)
            expected = closed if stage == "active" else set()
            if edge_set != expected:
                raise ValueError(f"{ident}: event lifecycle does not match intended edges")
            stages[stage] = {"epoch": epoch, "rows": stage_rows,
                             "expected_closed_edges": [list(edge) for edge in sorted(edge_set)]}
        fixtures.append({"case_id": ident, "kind": kind, "start_poi_id": start, "end_poi_id": end,
                         "mode": mode, "start_anchor": anchors[start], "end_anchor": anchors[end],
                         "baseline": baseline, "active_reference": changed, "stages": stages})
        print(ident, kind, baseline["verdict"], changed["verdict"], flush=True)
    return {"scope": "synthetic_development_only", "campus_truth": False,
            "snap_limit_m": 60, "boundary": "product polygon, not field surveyed",
            "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            "poi_sha256": hashlib.sha256(poi_path.read_bytes()).hexdigest(),
            "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "fixtures": fixtures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != text:
        raise FileExistsError("event fixture already exists with different contents")
    if not args.output.exists():
        args.output.write_text(text, encoding="utf-8")
    print("Saved", len(data["fixtures"]), "synthetic event fixtures", flush=True)


if __name__ == "__main__":
    main()
