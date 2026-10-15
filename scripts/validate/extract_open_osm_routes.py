"""Extract a reusable, read-only campus road/access candidate layer from local OSM.

All original tags, way node references, complete WGS84 geometry and OSM versions
are retained. Selection does not constitute a routing permission decision.
Run with the system Python environment, which includes osmium.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import osmium
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import transform, unary_union

from config import CAMPUS_POLYS_GCJ
from spatial.coord_transform import gcj02_to_wgs84

TO_M = Transformer.from_crs(4326, 32650, always_xy=True).transform
TO_WGS = Transformer.from_crs(32650, 4326, always_xy=True).transform


def load_boundaries(path):
    """Use assembled local OSM education areas when available; otherwise config."""
    areas = []
    if path.exists():
        evidence = json.loads(path.read_text(encoding="utf-8"))
        osm_ids = {"relation/7728026", "relation/20349098", "relation/20349097"}
        if evidence.get("education_areas"):
            candidates = evidence["education_areas"]
            boundary_source = "assembled_OSM_education_areas"
        elif evidence.get("type") == "FeatureCollection":
            candidates = [
                {
                    "name": feature.get("properties", {}).get("name", ""),
                    "osm_id": feature.get("properties", {}).get("osm_id"),
                    "tags": feature.get("properties", {}).get("tags") or {},
                    "geometry": feature.get("geometry"),
                }
                for feature in evidence.get("features", [])
            ]
            boundary_source = "osm_campus_boundary_geojson"
        else:
            candidates = []
            boundary_source = ""
        for area in candidates:
            tags = area.get("tags", {})
            osm_id = area.get("osm_id")
            geometry = area.get("geometry")
            if osm_id not in osm_ids or not geometry:
                continue
            geom = shape(geometry)
            if not geom.is_empty and geom.is_valid:
                areas.append({"name": area.get("name") or tags.get("name", ""),
                              "osm_id": osm_id, "geometry": geom, "tags": tags})
    if areas:
        return areas, boundary_source
    return [{"name": name, "osm_id": None, "tags": {}, "geometry": Polygon(
        [gcj02_to_wgs84(*p) for p in points])} for name, points in CAMPUS_POLYS_GCJ.items()], "config_polygons_fallback"


def building_candidate(osm_id, tags, coords, selection, version=None, timestamp=None):
    """Keep complete OSM building footprints as crossing-audit evidence."""
    if not tags.get("building") or len(coords) < 4 or coords[0] != coords[-1]:
        return None
    west, south, east, north = selection.bounds
    xs, ys = zip(*coords)
    if max(xs) < west or min(xs) > east or max(ys) < south or min(ys) > north:
        return None
    polygon = Polygon(coords)
    if not polygon.is_valid or polygon.is_empty or not selection.intersects(polygon):
        return None
    return {"type": "Feature", "id": f"way/{osm_id}", "geometry": mapping(polygon),
            "properties": {"osm_type": "way", "osm_id": int(osm_id),
                           "osm_version": version, "osm_timestamp": str(timestamp or ""),
                           "source": "OpenStreetMap", "source_ref": f"way/{osm_id}",
                           "license": "ODbL-1.0", "verification_status": "source_only",
                           "tags": dict(tags)}}


class Extractor(osmium.SimpleHandler):
    def __init__(self, campus, selection):
        super().__init__()
        self.campus = campus
        self.selection = selection
        self.bounds = selection.bounds
        self.ways = []
        self.buildings = []
        self.nodes = []
        self.restrictions = []
        self.way_ids = set()
        self.stats = Counter()

    def common(self, item, kind):
        return {"osm_type": kind, "osm_id": item.id, "osm_version": item.version,
                "osm_timestamp": str(item.timestamp), "source": "OpenStreetMap",
                "source_ref": f"{kind}/{item.id}", "license": "ODbL-1.0",
                "verification_status": "source_only", "routing_decision": "not_evaluated",
                "tags": dict(item.tags)}

    def node(self, node):
        if not (node.tags.get("barrier") or node.tags.get("entrance") or node.tags.get("highway") == "crossing"):
            return
        if not node.location.valid():
            self.stats["invalid_access_node_location"] += 1
            return
        p = Point(node.location.lon, node.location.lat)
        if self.selection.covers(p):
            properties = self.common(node, "node")
            properties["within_campus_boundary"] = self.campus.covers(p)
            self.nodes.append({"type": "Feature", "id": f"node/{node.id}", "geometry": mapping(p), "properties": properties})

    def way(self, way):
        if not way.tags.get("highway") and not way.tags.get("building"):
            return
        is_highway = bool(way.tags.get("highway"))
        if is_highway:
            self.stats["source_highway_ways_scanned"] += 1
        coords = []
        for node in way.nodes:
            if not node.location.valid():
                self.stats["ways_with_invalid_locations"] += 1
                return
            coords.append((node.location.lon, node.location.lat))
        if len(coords) < 2:
            return
        if way.tags.get("building"):
            feature = building_candidate(way.id, dict(way.tags), coords, self.selection,
                                         way.version, way.timestamp)
            if feature:
                self.buildings.append(feature)
        if not is_highway:
            return
        west, south, east, north = self.bounds
        xs, ys = zip(*coords)
        if max(xs) < west or min(xs) > east or max(ys) < south or min(ys) > north:
            return
        line = LineString(coords)
        if not self.selection.intersects(line):
            return
        props = self.common(way, "way")
        props["node_refs"] = [n.ref for n in way.nodes]
        props["whole_way_length_m"] = round(transform(TO_M, line).length, 2)
        props["intersects_campus_boundary"] = self.campus.intersects(line)
        props["geometry_scope"] = "complete_original_way_not_clipped"
        self.ways.append({"type": "Feature", "id": f"way/{way.id}", "geometry": mapping(line), "properties": props})
        self.way_ids.add(way.id)

    def relation(self, relation):
        if relation.tags.get("type") != "restriction":
            return
        if any(m.type == "w" and m.ref in self.way_ids for m in relation.members):
            self.restrictions.append({**self.common(relation, "relation"), "members": [
                {"type": m.type, "ref": m.ref, "role": m.role} for m in relation.members]})


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def output_reference(path):
    """Keep repository outputs concise and allow research outputs on other drives."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbf", type=Path, default=ROOT / "scripts/audit_output/hubei-260924.osm.pbf")
    parser.add_argument("--boundary-evidence", type=Path, default=ROOT / "scripts/audit_output/osm_local_evidence.json")
    parser.add_argument("--buffer-m", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/audit_output")
    args = parser.parse_args()
    areas, boundary_source = load_boundaries(args.boundary_evidence)
    campus = unary_union([a["geometry"] for a in areas])
    selection = transform(TO_WGS, transform(TO_M, campus).buffer(args.buffer_m))
    handler = Extractor(campus, selection)
    print(f"Boundary source: {boundary_source}; assembling complete highway geometry", flush=True)
    handler.apply_file(str(args.pbf), locations=True, idx="flex_mem")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"source": "OpenStreetMap", "source_file": args.pbf.name,
        "license": "ODbL-1.0", "coordinate_reference": "WGS84 EPSG:4326",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "boundary_source": boundary_source,
        "buffer_m": args.buffer_m, "interpretation": "Candidate evidence only. Original tags and complete geometry retained; no access permission inferred from missing tags."}
    ways_path = args.output_dir / "osm_route_candidates.geojson"
    nodes_path = args.output_dir / "osm_access_nodes.geojson"
    boundary_path = args.output_dir / "osm_route_selection_boundaries.geojson"
    buildings_path = args.output_dir / "osm_buildings.geojson"
    write_json(ways_path, {"type": "FeatureCollection", "metadata": metadata, "features": handler.ways})
    write_json(nodes_path, {"type": "FeatureCollection", "metadata": metadata, "features": handler.nodes})
    write_json(buildings_path, {"type": "FeatureCollection", "metadata": metadata,
                                "features": handler.buildings})
    write_json(boundary_path, {"type": "FeatureCollection", "metadata": metadata, "features": [
        {"type": "Feature", "geometry": mapping(a["geometry"]), "properties": {k: v for k, v in a.items() if k != "geometry"}}
        for a in areas]})
    fields = ("access", "foot", "bicycle", "motor_vehicle", "motorcar", "vehicle", "oneway", "oneway:bicycle",
              "oneway:foot", "indoor", "level", "tunnel", "covered", "bridge", "surface", "incline", "service")
    tag_counts = {key: sum(key in f["properties"]["tags"] for f in handler.ways) for key in fields}
    report = {**metadata, "source_sha256": hashlib.sha256(args.pbf.read_bytes()).hexdigest(),
        "source_scan": dict(handler.stats), "road_way_count": len(handler.ways),
        "road_way_within_or_crossing_campus_count": sum(f["properties"]["intersects_campus_boundary"] for f in handler.ways),
        "highway_types": dict(Counter(f["properties"]["tags"]["highway"] for f in handler.ways)),
        "original_way_tag_counts": tag_counts, "access_node_count": len(handler.nodes),
        "building_footprint_count": len(handler.buildings),
        "barrier_node_count": sum("barrier" in f["properties"]["tags"] for f in handler.nodes),
        "entrance_node_count": sum("entrance" in f["properties"]["tags"] for f in handler.nodes),
        "crossing_node_count": sum(f["properties"]["tags"].get("highway") == "crossing" for f in handler.nodes),
        "restriction_relations": handler.restrictions,
        "outputs": [output_reference(p) for p in (ways_path, nodes_path, boundary_path, buildings_path)],
        "limitations": ["Source timestamp is newer than the production graph; differences include mapping updates.",
                        "Raw OSM tags need mode-specific interpretation before routing; access uncertainty remains explicit.",
                        "Complete ways can extend outside selection polygons; use intersects_campus_boundary and boundary layer when reviewing."]}
    write_json(args.output_dir / "osm_route_candidate_summary.json", report)
    print(json.dumps({"roads": len(handler.ways), "access_nodes": len(handler.nodes),
                      "buildings": len(handler.buildings), "tags": tag_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
