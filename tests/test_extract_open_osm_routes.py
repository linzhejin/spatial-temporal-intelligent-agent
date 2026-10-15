import json
from pathlib import Path

from scripts.validate.extract_open_osm_routes import load_boundaries, output_reference


def test_load_boundaries_accepts_current_osm_campus_geojson(tmp_path):
    path = tmp_path / "campuses.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "source": "OpenStreetMap",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[114.0, 30.0], [114.1, 30.0],
                                     [114.1, 30.1], [114.0, 30.1], [114.0, 30.0]]],
                },
                "properties": {"osm_id": "relation/7728026", "name": "武汉大学信息学部"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[115.0, 31.0], [115.1, 31.0],
                                     [115.1, 31.1], [115.0, 31.1], [115.0, 31.0]]],
                },
                "properties": {"osm_id": "way/123", "name": "neighbor"},
            },
        ],
    }, ensure_ascii=False), encoding="utf-8")

    areas, source = load_boundaries(path)

    assert source == "osm_campus_boundary_geojson"
    assert [area["osm_id"] for area in areas] == ["relation/7728026"]
    assert areas[0]["name"] == "武汉大学信息学部"
    assert areas[0]["geometry"].covers(__import__("shapely").geometry.Point(114.05, 30.05))


def test_output_reference_supports_paths_outside_repository():
    root = Path(__file__).resolve().parents[1]

    assert output_reference(root / "research" / "report.json") == "research/report.json"
    assert output_reference(Path("D:/LuojiaEnrichmentData/report.json")) == \
        "D:/LuojiaEnrichmentData/report.json"
