import json
from datetime import date

from scripts.validate import review_poi_inventory


def _write_inventory(tmp_path, rows, monkeypatch, graph_nodes=None):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "pois.json").write_text(json.dumps({
        "count": len(rows),
        "provenance_defaults": {"verification_status": "legacy_unverified"},
        "pois": rows,
    }), encoding="utf-8")
    (data_dir / "campus_review_decisions.json").write_text(
        json.dumps({"poi_decisions": [], "road_decisions": []}), encoding="utf-8")
    (data_dir / "pois_osm_candidates.json").write_text(
        json.dumps({"candidates": []}), encoding="utf-8")
    if graph_nodes is not None:
        import networkx as nx

        graph = nx.MultiDiGraph()
        for node_id, x, y in graph_nodes:
            graph.add_node(node_id, x=x, y=y)
        nx.write_graphml(graph, data_dir / "whu_road_network.graphml")
    monkeypatch.setattr(review_poi_inventory, "ROOT", tmp_path)
    return review_poi_inventory.build_report()


def test_dorm_is_a_supported_poi_type(tmp_path, monkeypatch):
    report = _write_inventory(tmp_path, [{
        "id": "poi-dorm",
        "name": "Dorm A",
        "aliases": [],
        "type": "dorm",
        "subcategory": "dormitory",
        "campus": "whu",
        "coordinates": {"lng": 114.36, "lat": 30.53},
    }], monkeypatch)

    row = report["records"][0]
    assert "type_outside_known_vocabulary" not in row["issue_codes"]


def test_audit_report_uses_the_current_local_date(tmp_path, monkeypatch):
    class FixedDate:
        @classmethod
        def today(cls):
            return date(2026, 11, 6)

    monkeypatch.setattr(review_poi_inventory, "date", FixedDate, raising=False)
    report = _write_inventory(tmp_path, [], monkeypatch)

    assert report["created_local_date"] == "2026-11-06"


def test_inventory_flags_near_duplicate_aliases_but_keeps_distant_branches(tmp_path, monkeypatch):
    rows = [
        {
            "id": "poi-a",
            "name": "Library A",
            "aliases": ["Shared campus name"],
            "type": "study",
            "campus": "whu",
            "coordinates": {"lng": 114.36, "lat": 30.53},
        },
        {
            "id": "poi-b",
            "name": "Library B",
            "aliases": ["Shared campus name"],
            "type": "study",
            "campus": "whu",
            "coordinates": {"lng": 114.36005, "lat": 30.53},
        },
        {
            "id": "poi-c",
            "name": "Library C",
            "aliases": ["Shared campus name"],
            "type": "study",
            "campus": "whu",
            "coordinates": {"lng": 114.37, "lat": 30.53},
        },
    ]

    report = _write_inventory(tmp_path, rows, monkeypatch)
    by_id = {row["poi_id"]: row for row in report["records"]}

    assert by_id["poi-a"]["near_name_or_alias_ids_within_20m"] == ["poi-b"]
    assert by_id["poi-b"]["near_name_or_alias_ids_within_20m"] == ["poi-a"]
    assert by_id["poi-c"]["near_name_or_alias_ids_within_20m"] == []
    assert "exact_name_or_alias_duplicate_needs_review" in by_id["poi-a"]["issue_codes"]


def test_repeated_normalized_aliases_on_one_record_do_not_self_collide(tmp_path, monkeypatch):
    report = _write_inventory(tmp_path, [{
        "id": "poi-one",
        "name": "Library A",
        "aliases": ["Shared alias", "shared-alias"],
        "type": "study",
        "campus": "whu",
        "coordinates": {"lng": 114.36, "lat": 30.53},
    }], monkeypatch)

    row = report["records"][0]
    assert row["near_name_or_alias_ids_within_20m"] == []
    assert "exact_name_or_alias_duplicate_needs_review" not in row["issue_codes"]


def test_inventory_flags_pois_more_than_50m_from_routed_graph(tmp_path, monkeypatch):
    from spatial.coord_transform import gcj02_to_wgs84

    lng, lat = 114.36, 30.53
    road_lng, road_lat = gcj02_to_wgs84(lng, lat)
    report = _write_inventory(tmp_path, [{
        "id": "poi-far",
        "name": "Far from routed graph",
        "aliases": [],
        "type": "study",
        "campus": "whu",
        "coordinates": {"lng": lng, "lat": lat},
    }], monkeypatch, graph_nodes=[(1, road_lng + 0.001, road_lat)])

    row = report["records"][0]
    assert row["road_network_proximity"]["snap_distance_m"] > 50
    assert "road_network_snap_over_50m_needs_review" in row["issue_codes"]
