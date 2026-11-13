"""Freeze development-only graph anchors, without claiming field access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx

from experiments.build_controlled_spatial_snapshot import build_spatial_snapshot


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/datasets/complex_multiturn_dev_v1/anchor_snapshot_60m.json"


def build():
    from config import CAMPUS_POLYS_GCJ
    from spatial.coord_transform import gcj02_to_wgs84

    graph_path = ROOT / "data/whu_road_network.graphml"
    poi_path = ROOT / "data/pois.json"
    graph = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
    pois = json.loads(poi_path.read_text(encoding="utf-8"))["pois"]
    boundary, anchors = build_spatial_snapshot(graph, pois, CAMPUS_POLYS_GCJ,
                                                to_wgs84=gcj02_to_wgs84, max_snap_m=60)
    return {"scope": "development_only", "max_snap_m": 60,
            "warning": "Geometric graph anchors only; building entrances and site access are unverified.",
            "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            "poi_sha256": hashlib.sha256(poi_path.read_bytes()).hexdigest(),
            "boundary_wkt": boundary, "anchors": anchors["anchors"], "excluded": anchors["excluded"]}


def main():
    result = build()
    content = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if OUT.exists() and OUT.read_text(encoding="utf-8") != content:
        raise FileExistsError("anchor snapshot exists with different contents")
    if not OUT.exists():
        OUT.write_text(content, encoding="utf-8")
    print("Frozen", len(result["anchors"]), "graph anchors; <=60 m, not field-verified")


if __name__ == "__main__":
    main()
