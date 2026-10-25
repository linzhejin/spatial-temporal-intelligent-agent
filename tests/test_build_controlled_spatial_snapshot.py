"""Controlled spatial artifacts are reproducible from explicit frozen inputs."""

import json

import networkx as nx
import pytest

from experiments.build_controlled_spatial_snapshot import (
    build_spatial_snapshot, write_spatial_snapshot,
)


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.0, y=30.0)
    graph.add_node(2, x=114.0001, y=30.0)
    graph.add_node(3, x=115.0, y=31.0)
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(2, 3, key=0, highway="footway", length=100000)
    return graph


def test_snapshot_uses_boundary_and_walk_permitted_edges():
    pois = [{"id": "a", "name": "甲", "coordinates": {"lng": 114.0, "lat": 30.0}},
            {"id": "b", "name": "乙", "coordinates": {"lng": 115.0, "lat": 31.0}}]
    boundary, anchors = build_spatial_snapshot(
        _graph(), pois, {"campus": [(113.999, 29.999), (114.001, 29.999),
                                    (114.001, 30.001), (113.999, 30.001)]},
        to_wgs84=lambda x, y: (x, y), max_snap_m=20,
    )
    assert boundary.startswith("POLYGON")
    assert [item["poi_id"] for item in anchors["anchors"]] == ["a"]
    assert anchors["excluded"] == [{"poi_id": "b", "reason": "too_far_from_permitted_graph",
                                    "nearest_distance_m": pytest.approx(146000, rel=0.02)}]
    assert anchors["mode"] == "walk"
    assert anchors["boundary_scope"] == "product_derived_polygon_not_field_verified"


def test_snapshot_rejects_invalid_polygon_and_cannot_reuse_stale_outputs(tmp_path):
    with pytest.raises(ValueError, match="boundary"):
        build_spatial_snapshot(_graph(), [], {"bad": [(114, 30), (115, 31)]},
                               to_wgs84=lambda x, y: (x, y))
    boundary, anchors = build_spatial_snapshot(
        _graph(), [{"id": "a", "name": "甲", "coordinates": {"lng": 114, "lat": 30}}],
        {"campus": [(113.999, 29.999), (114.001, 29.999),
                    (114.001, 30.001), (113.999, 30.001)]},
        to_wgs84=lambda x, y: (x, y),
    )
    boundary_path = tmp_path / "boundary.wkt"
    anchors_path = tmp_path / "anchors.json"
    write_spatial_snapshot(boundary_path, anchors_path, boundary, anchors)
    first = (boundary_path.read_bytes(), anchors_path.read_bytes())
    write_spatial_snapshot(boundary_path, anchors_path, boundary, anchors)
    assert first == (boundary_path.read_bytes(), anchors_path.read_bytes())
    assert json.loads(anchors_path.read_text(encoding="utf-8"))["anchors"][0]["poi_id"] == "a"
    with pytest.raises(ValueError, match="snapshot differs"):
        write_spatial_snapshot(boundary_path, anchors_path, boundary, {"anchors": []})
    assert first == (boundary_path.read_bytes(), anchors_path.read_bytes())


def test_product_boundary_survives_wkt_roundtrip():
    from config import CAMPUS_POLYS_GCJ
    from spatial.coord_transform import gcj02_to_wgs84
    from experiments.build_controlled_spatial_snapshot import _boundary
    from shapely import wkt

    boundary = _boundary(CAMPUS_POLYS_GCJ, gcj02_to_wgs84)
    serialized = wkt.dumps(boundary, rounding_precision=16, trim=True)
    assert wkt.loads(serialized).is_valid
