"""Freeze a conservative walk-anchor snapshot for controlled experiments.

The campus polygon comes from product configuration and is not field surveyed.
The anchor is a nearby graph node, not a verified building entrance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import networkx as nx
from shapely import wkt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from experiments.controlled_anchor_snapshot import build_anchor_candidates
from experiments.independent_source_snapshot import mode_edge_evidence


def _boundary(polygons_gcj: dict[str, list[tuple[float, float]]],
              to_wgs84: Callable[[float, float], tuple[float, float]]) -> Any:
    if not isinstance(polygons_gcj, dict) or not polygons_gcj or not callable(to_wgs84):
        raise ValueError("boundary polygons and converter required")
    polygons = []
    try:
        for name in sorted(polygons_gcj):
            points = [to_wgs84(float(lon), float(lat))
                      for lon, lat in polygons_gcj[name]]
            if len(points) < 3 or any(not all(math.isfinite(v) for v in point)
                                      for point in points):
                raise ValueError("invalid boundary polygon")
            polygon = Polygon(points)
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                raise ValueError("invalid boundary polygon")
            polygons.append(polygon)
        boundary = unary_union(polygons)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("invalid boundary polygon") from exc
    if boundary.is_empty or not boundary.is_valid:
        raise ValueError("invalid boundary polygon union")
    return boundary


def build_spatial_snapshot(
    graph: Any, pois: list[dict[str, Any]],
    polygons_gcj: dict[str, list[tuple[float, float]]], *,
    to_wgs84: Callable[[float, float], tuple[float, float]],
    max_snap_m: float = 20,
) -> tuple[str, dict[str, Any]]:
    """Derive walk anchors only from eligible graph nodes within the boundary."""
    boundary = _boundary(polygons_gcj, to_wgs84)
    boundary_wkt = wkt.dumps(boundary, rounding_precision=16, trim=True)
    if not wkt.loads(boundary_wkt).is_valid:
        raise ValueError("serialized boundary is invalid")
    evidence = mode_edge_evidence(graph, "walk", require_boundary=True,
                                  boundary_wkt=boundary_wkt)
    edge_nodes = {node for u, v, _ in evidence["permitted"] for node in (u, v)}
    permitted_nodes = set()
    for node in edge_nodes:
        attrs = graph.nodes[node]
        try:
            lon, lat = float(attrs["x"]), float(attrs["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(lon) and math.isfinite(lat) and boundary.covers(Point(lon, lat)):
            permitted_nodes.add(node)
    anchors = build_anchor_candidates(graph, pois, permitted_nodes,
                                      to_wgs84=to_wgs84, max_snap_m=max_snap_m)
    anchors["mode"] = "walk"
    anchors["boundary_scope"] = "product_derived_polygon_not_field_verified"
    anchors["permitted_node_count"] = len(permitted_nodes)
    anchors["mode_evidence_scope"] = evidence["scope"]
    return boundary_wkt, anchors


def _write_if_same_or_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"snapshot differs; select a new versioned path: {path}")
    if not path.exists():
        path.write_bytes(content)


def write_spatial_snapshot(boundary_path: Path, anchors_path: Path,
                           boundary_wkt: str, anchors: dict[str, Any]) -> None:
    """Write deterministic bytes without silently replacing a frozen snapshot."""
    boundary_bytes = (boundary_wkt + "\n").encode("utf-8")
    anchors_bytes = (json.dumps(anchors, ensure_ascii=False, sort_keys=True,
                                indent=2) + "\n").encode("utf-8")
    for path, content in ((boundary_path, boundary_bytes), (anchors_path, anchors_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"snapshot differs; select a new versioned path: {path}")
    _write_if_same_or_new(boundary_path, boundary_bytes)
    _write_if_same_or_new(anchors_path, anchors_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/whu_road_network.graphml"))
    parser.add_argument("--poi", type=Path, default=Path("data/pois.json"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("experiments/datasets/controlled_spatial_v1"))
    args = parser.parse_args()
    from config import CAMPUS_POLYS_GCJ
    from spatial.coord_transform import gcj02_to_wgs84

    graph = nx.read_graphml(args.graph, node_type=int, force_multigraph=True)
    poi_data = json.loads(args.poi.read_text(encoding="utf-8"))
    if not isinstance(poi_data, dict) or not isinstance(poi_data.get("pois"), list):
        raise ValueError("POI file must contain pois list")
    boundary_wkt, anchors = build_spatial_snapshot(
        graph, poi_data["pois"], CAMPUS_POLYS_GCJ,
        to_wgs84=gcj02_to_wgs84,
    )
    write_spatial_snapshot(args.out_dir / "boundary_wgs84.wkt",
                           args.out_dir / "walk_anchors.json", boundary_wkt, anchors)
    print(json.dumps({"anchors": len(anchors["anchors"]),
                      "excluded": len(anchors["excluded"]),
                      "permitted_nodes": anchors["permitted_node_count"],
                      "scope": anchors["boundary_scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
