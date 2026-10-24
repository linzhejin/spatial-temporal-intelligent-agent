"""Area geometry can diagnose centre-point snap errors without asserting an entrance."""
import networkx as nx
import pytest
from shapely.geometry import mapping, Polygon

from spatial.coord_transform import gcj02_to_wgs84
from scripts.analysis.audit_t2_access_geometry import audit_access_geometry


def test_area_overlap_reduces_geometric_offset_but_never_verifies_access():
    center = gcj02_to_wgs84(114.36, 30.54)
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=center[0] + 0.0005, y=center[1])
    graph.add_node("b", x=center[0] + 0.0010, y=center[1])
    graph.add_edge("a", "b", key=0, length=48.0)
    polygon = Polygon([
        (center[0] - 0.0006, center[1] - 0.0004),
        (center[0] + 0.0006, center[1] - 0.0004),
        (center[0] + 0.0006, center[1] + 0.0004),
        (center[0] - 0.0006, center[1] + 0.0004),
    ])
    pois = [
        {"id": "court", "name": "工学部篮球场", "campus": "文理学部",
         "coordinates": {"lng": 114.36, "lat": 30.54}},
        {"id": "library", "name": "法学图书馆", "campus": "文理学部",
         "coordinates": {"lng": 114.36, "lat": 30.54}},
    ]
    areas = [
        {"type": "Feature", "properties": {
            "osm_id": "way/1", "name": "工学部篮球场",
            "campus": "工学部", "matched_poi_ids": ["court"],
            "coordinate_system": "WGS-84",
            "tags": {"leisure": "sports_centre"},
        }, "geometry": mapping(polygon)},
    ]

    result = audit_access_geometry(graph, pois, areas, ["court", "library"])
    court, library = result["pois"]
    assert court["poi_id"] == "court"
    assert court["area_role"] == "matched_poi_area"
    assert court["poi_inside_area"] is True
    assert court["poi_to_walk_node_m"] > 30
    assert court["area_to_walk_node_m"] == 0
    assert court["campus_conflict"] is True
    assert court["access_verdict"] == "unknown"
    assert library["area_osm_id"] is None
    assert library["access_verdict"] == "unknown"


def test_area_without_declared_wgs84_is_rejected():
    graph = nx.MultiDiGraph()
    graph.add_node("a", x=114.36, y=30.54)
    graph.add_node("b", x=114.361, y=30.54)
    graph.add_edge("a", "b", key=0, length=95.0)
    pois = [{"id": "p", "name": "场地", "campus": "工学部",
             "coordinates": {"lng": 114.36, "lat": 30.54}}]
    areas = [{"type": "Feature", "properties": {
        "osm_id": "way/1", "coordinate_system": "GCJ-02",
        "matched_poi_ids": ["p"],
    }, "geometry": mapping(Polygon([
        (114.359, 30.539), (114.361, 30.539),
        (114.361, 30.541), (114.359, 30.541),
    ]))}]
    with pytest.raises(ValueError, match="WGS-84"):
        audit_access_geometry(graph, pois, areas, ["p"])
