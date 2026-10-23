"""Derive fixed, unpublished road-event fixtures from simulated video links.

These cases test graph/event semantics. They do not assert that a public video
observed a campus event, or that the manager or Agent handled it successfully.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from experiments.scenario_v2 import audit_route_edges, load_frozen_graph
from experiments.vision_simulation import file_sha256


def _walk_graph(graph):
    walking = nx.MultiDiGraph()
    walking.add_nodes_from(graph.nodes(data=True))
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        if audit_route_edges(graph, [[u, v, key]], u, v, [], "walk", {})["route"] == "pass":
            walking.add_edge(str(u), str(v), key=str(key), **attrs)
    return walking


def _exact_edges(graph, nodes: list[str]) -> tuple[list[list[str]], float]:
    edges = []
    length = 0.0
    for u, v in zip(nodes, nodes[1:]):
        available = graph.get_edge_data(u, v) or {}
        if not available:
            raise ValueError("shortest path has no exact edge")
        key, attrs = min(available.items(), key=lambda item: (float(item[1]["length"]), str(item[0])))
        edges.append([str(u), str(v), str(key)])
        length += float(attrs["length"])
    return edges, round(length, 2)


def build_event_bank(graph, assignment: dict, *, active_at: int = 1500,
                     expired_at: int = 2500) -> dict:
    if assignment.get("format") != "vision-simulated-road-assignment-v1":
        raise ValueError("requires a simulated video-road assignment")
    walking = _walk_graph(graph)
    cases = []
    for video in assignment["videos"]:
        for link in video["assigned_cases"]:
            u, v, key = map(str, link["edge_id"])
            if not walking.has_edge(u, v, key):
                raise ValueError(f"{link['id']}: linked road is not walkable in frozen graph")
            without = nx.subgraph_view(
                walking, filter_edge=lambda a, b, k: (str(a), str(b), str(k))
                not in {(u, v, key), (v, u, key)})
            try:
                nodes = nx.shortest_path(without, u, v, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
                raise ValueError(f"{link['id']}: no independently walkable detour") from error
            detour, distance = _exact_edges(without, nodes)
            closed = {"|".join((u, v, key)), "|".join((v, u, key))}
            if audit_route_edges(graph, detour, u, v, [], "walk", {},
                                 closed_edges=closed)["route"] != "pass":
                raise ValueError(f"{link['id']}: independent detour legality failed")
            event = {"action": "closure", "affected_modes": ["walk"],
                     "start_time": 1000, "end_time": 2000,
                     "edge": {"u": u, "v": v, "key": key},
                     "review_status": "synthetic_fixture_only"}
            for phase, clock, expected in (
                ("active", active_at, {"direct_edge_legal": False,
                                       "detour_edge_ids": detour,
                                       "detour_distance_m": distance}),
                ("expired", expired_at, {"direct_edge_legal": True,
                                         "direct_edge_id": [u, v, key]}),
            ):
                cases.append({"id": f"D1-{link['id']}-{phase}",
                              "source_video_id": video["source_id"],
                              "source_link_id": link["id"],
                              "source_spatial_truth": "Songdo, not Wuhan",
                              "campus_road_assignment": "hypothetical_unpublished",
                              "mode": "walk", "start_node": u, "end_node": v,
                              "as_of_epoch": clock, "event": event,
                              "expected_graph_effect": expected})
    return {"format": "synthetic-road-event-bank-v1", "split": "development",
            "graph_sha256": assignment["graph_sha256"],
            "source_assignment_seed": assignment["seed"],
            "source_video_count": len(assignment["videos"]),
            "cases": cases,
            "scope": "Graph/event fixture only; no model candidate, manager approval or Agent result."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--road-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    if file_sha256(args.graph) != assignment["graph_sha256"]:
        raise ValueError("assignment and graph SHA-256 differ")
    graph = load_frozen_graph(args.graph, args.overrides, args.road_reviews)
    bank = build_event_bank(graph, assignment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(bank["cases"]),
                      "status": "development_fixture_only"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
