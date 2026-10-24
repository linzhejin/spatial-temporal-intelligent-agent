"""Diagnose T2 final access with OSM areas; geometry never verifies entry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform

from experiments.candidate_reference import _walk_graph
from spatial.coord_transform import gcj02_to_wgs84


ROOT = Path(__file__).resolve().parents[2]
_TO_METERS = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True).transform


def audit_access_geometry(
    graph: nx.MultiDiGraph,
    pois: list[dict[str, Any]],
    areas: list[dict[str, Any]],
    affected_ids: list[str],
) -> dict[str, Any]:
    """Compare a POI pin and its source area with walk-routable graph nodes.

    A node inside an OSM area is geometric evidence, not proof of an entrance,
    public access, or a safe final segment. Every access verdict remains unknown.
    """
    by_id = {poi["id"]: poi for poi in pois}
    if len(by_id) != len(pois) or len(set(affected_ids)) != len(affected_ids):
        raise ValueError("duplicate POI or affected ID")
    if set(affected_ids) - set(by_id):
        raise ValueError("affected POI missing from master")
    walk = _walk_graph(graph, set())
    nodes = [
        (str(node), Point(_TO_METERS(float(attrs["x"]), float(attrs["y"]))))
        for node, attrs in graph.nodes(data=True)
        if walk.degree(str(node)) > 0
    ]
    if not nodes:
        raise ValueError("graph has no walk-routable nodes")

    source_areas = []
    for feature in areas:
        props = feature["properties"]
        if props.get("coordinate_system") != "WGS-84":
            raise ValueError(f"area must declare WGS-84: {props.get('osm_id')}")
        geometry = transform(_TO_METERS, shape(feature["geometry"]))
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"invalid area geometry: {props.get('osm_id')}")
        source_areas.append((props, geometry))

    rows = []
    for poi_id in affected_ids:
        poi = by_id[poi_id]
        coordinate = poi["coordinates"]
        lon, lat = gcj02_to_wgs84(
            float(coordinate["lng"]), float(coordinate["lat"])
        )
        point = Point(_TO_METERS(lon, lat))
        pin_node, pin_distance = min(
            ((node, point.distance(location)) for node, location in nodes),
            key=lambda pair: pair[1],
        )

        direct = [
            (props, geometry) for props, geometry in source_areas
            if poi_id in props.get("matched_poi_ids", [])
        ]
        containing_market = [
            (props, geometry) for props, geometry in source_areas
            if props.get("tags", {}).get("amenity") == "marketplace"
            and geometry.covers(point)
        ]
        chosen = direct[0] if len(direct) == 1 else (
            containing_market[0] if not direct and len(containing_market) == 1
            else None
        )
        role = ("matched_poi_area" if direct and chosen else
                "containing_market_area" if containing_market and chosen else None)
        row: dict[str, Any] = {
            "poi_id": poi_id,
            "poi_name": poi["name"],
            "poi_campus": poi.get("campus"),
            "poi_to_walk_node_m": round(pin_distance, 2),
            "poi_nearest_walk_node": pin_node,
            "area_osm_id": None,
            "area_role": role,
            "poi_inside_area": None,
            "area_to_walk_node_m": None,
            "area_nearest_walk_node": None,
            "walk_nodes_inside_area": None,
            "area_campus": None,
            "campus_conflict": None,
            "point_representation_warning": False,
            "access_verdict": "unknown",
        }
        if chosen:
            props, geometry = chosen
            area_node, area_distance = min(
                ((node, geometry.distance(location)) for node, location in nodes),
                key=lambda pair: pair[1],
            )
            row.update({
                "area_osm_id": props["osm_id"],
                "poi_inside_area": geometry.covers(point),
                "area_to_walk_node_m": round(area_distance, 2),
                "area_nearest_walk_node": area_node,
                "walk_nodes_inside_area": sum(
                    geometry.covers(location) for _, location in nodes
                ),
                "area_campus": props.get("campus"),
                "campus_conflict": (
                    props["campus"] != poi.get("campus")
                    if props.get("campus") else None
                ),
                "point_representation_warning": (
                    pin_distance > 30 and area_distance <= 30
                ),
            })
        rows.append(row)
    return {
        "status": "development_geometry_diagnostic_not_field_truth",
        "affected_poi_count": len(rows),
        "pois": rows,
        "limitations": [
            "Area and walk-node overlap does not establish a passable entrance.",
            "The OSM area may share lineage with the routing graph.",
            "All affected T2 ranking verdicts remain unknown pending independent access evidence.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areas", type=Path, required=True)
    parser.add_argument("--pois", type=Path, default=ROOT / "data/pois.json")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/whu_road_network.graphml")
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    poi_bytes = args.pois.read_bytes()
    graph_bytes = args.graph.read_bytes()
    area_bytes = args.areas.read_bytes()
    uncertainty_bytes = args.uncertainty.read_bytes()
    uncertainty = json.loads(uncertainty_bytes)
    if (uncertainty.get("status")
            != "development_reference_uncertainty_only_not_agent_score"
            or uncertainty.get("poi_sha256") != hashlib.sha256(poi_bytes).hexdigest()
            or uncertainty.get("graph_sha256") != hashlib.sha256(graph_bytes).hexdigest()):
        raise ValueError("T2 uncertainty report does not match current inputs")
    affected_ids = [row["poi_id"] for row in uncertainty["affected_unique_pois"]]
    areas = json.loads(area_bytes)
    if areas.get("type") != "FeatureCollection":
        raise ValueError("area source must be a FeatureCollection")
    result = audit_access_geometry(
        nx.read_graphml(args.graph),
        json.loads(poi_bytes)["pois"],
        areas["features"],
        affected_ids,
    )
    result["source_sha256"] = {
        "pois": hashlib.sha256(poi_bytes).hexdigest(),
        "graph": hashlib.sha256(graph_bytes).hexdigest(),
        "osm_areas": hashlib.sha256(area_bytes).hexdigest(),
        "t2_uncertainty": hashlib.sha256(uncertainty_bytes).hexdigest(),
    }
    result["osm_source_snapshot"] = areas.get("source_snapshot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "affected_pois": result["affected_poi_count"],
        "area_evidence": sum(row["area_osm_id"] is not None for row in result["pois"]),
        "point_warnings": sum(row["point_representation_warning"] for row in result["pois"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
