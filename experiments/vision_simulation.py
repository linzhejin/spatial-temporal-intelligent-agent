"""Create reproducible *hypothetical* campus road links for public UAV clips.

Songdo footage is never represented as a Wuhan observation. This authoring
fixture supports route/event plumbing tests before campus flights exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import networkx as nx

from experiments.scenario_v2 import audit_route_edges, load_frozen_graph
from spatial.coord_transform import wgs84_to_gcj02


SOURCE = "Songdo Traffic v2"
SOURCE_URL = "https://zenodo.org/records/17924857"
SEED = 20261023


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eligible_edges(graph):
    choices = []
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        try:
            length = float(attrs.get("length", 0))
            first, second = graph.nodes[u], graph.nodes[v]
            lng = (float(first["x"]) + float(second["x"])) / 2
            lat = (float(first["y"]) + float(second["y"])) / 2
        except (ValueError, TypeError, KeyError):
            continue
        highway = str(attrs.get("highway", ""))
        if (35 <= length <= 180 and 114.32 < lng < 114.43 and 30.50 < lat < 30.57
                and not any(tag in highway for tag in ("steps", "corridor", "construction"))):
            choices.append((str(u), str(v), str(key), length, lng, lat))
    return sorted(choices)


def _connected_detour(graph, u: str, v: str, key: str, direct_length: float) -> float | None:
    edges = {(u, v, key), (v, u, key)}
    without = nx.subgraph_view(
        graph, filter_edge=lambda a, b, k: (str(a), str(b), str(k)) not in edges)
    try:
        direct_best = nx.shortest_path_length(graph, u, v, weight="length")
        if direct_length > direct_best + 0.01:
            return None
        detour = nx.shortest_path_length(without, u, v, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if direct_length + 1 < detour <= 3000:
        return float(detour)
    return None


def build_manifest(graph, graph_sha256: str, video_paths: list[Path], *,
                   cases_per_video: int = 2, seed: int = SEED) -> dict:
    if not video_paths or cases_per_video < 1:
        raise ValueError("video sources and case count are required")
    if len({path.name for path in video_paths}) != len(video_paths):
        raise ValueError("source videos must have unique names")
    candidates = _eligible_edges(graph)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    for u, v, key, length, lng, lat in candidates:
        if any((u, v) == (item["edge_id"][0], item["edge_id"][1]) for item in selected):
            continue
        if audit_route_edges(graph, [[u, v, key]], u, v, [], "walk", {})["route"] != "pass":
            continue
        detour = _connected_detour(graph, u, v, key, length)
        if detour is None:
            continue
        gcj_lng, gcj_lat = wgs84_to_gcj02(lng, lat)
        selected.append({"edge_id": [u, v, key], "anchor_gcj": {
            "lng": round(gcj_lng, 7), "lat": round(gcj_lat, 7), "crs": "GCJ02"},
            "direct_length_m": round(length, 2),
            "detour_after_hypothetical_closure_m": round(detour, 2),
            "effect": {"action": "closure", "affected_modes": ["walk"],
                       "status": "hypothetical_unpublished"}})
        if len(selected) == len(video_paths) * cases_per_video:
            break
    if len(selected) != len(video_paths) * cases_per_video:
        raise ValueError("not enough valid campus edge/detour pairs for simulation")
    videos = []
    for index, path in enumerate(sorted(video_paths)):
        if not path.is_file():
            raise FileNotFoundError(path)
        videos.append({"source_id": f"songdo-video-{index + 1:02d}",
                       "filename": path.name, "size_bytes": path.stat().st_size,
                       "sha256": file_sha256(path),
                       "assigned_cases": [
                           {"id": f"V1-{index + 1:02d}-{variant + 1}",
                            **selected[index * cases_per_video + variant]}
                           for variant in range(cases_per_video)]})
    return {"format": "vision-simulated-road-assignment-v1", "source": SOURCE,
            "source_url": SOURCE_URL, "source_spatial_truth": "Songdo, not Wuhan",
            "campus_location_status": "synthetic_random_assignment_only",
            "model_inference_status": "not_run_by_this_manifest",
            "manager_review_status": "not_confirmed",
            "publication_status": "never_publish_without_review",
            "seed": seed, "graph_sha256": graph_sha256, "videos": videos}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--road-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-count", type=int, default=6)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.videos_dir.glob("*.mp4"))[:args.video_count]
    if len(paths) != args.video_count:
        raise ValueError("requested number of source videos is unavailable")
    graph = load_frozen_graph(args.graph, args.overrides, args.road_reviews)
    manifest = build_manifest(graph, file_sha256(args.graph), paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "videos": len(paths),
                      "cases": sum(len(video["assigned_cases"]) for video in manifest["videos"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
