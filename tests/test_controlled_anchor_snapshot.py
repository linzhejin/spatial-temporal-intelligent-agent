"""POI-to-graph anchors are candidates with explicit last-meter uncertainty."""

import networkx as nx
import pytest

from experiments.controlled_anchor_snapshot import build_anchor_candidates


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.0, y=30.0)
    graph.add_node(2, x=114.0001, y=30.0)
    graph.add_node(3, x=114.01, y=30.0)
    return graph


def test_anchor_uses_only_permitted_nodes_and_records_snap_uncertainty():
    pois = [{"id": "poi_a", "name": "甲楼", "coordinates": {"lng": 114.0, "lat": 30.0}},
            {"id": "poi_b", "name": "乙楼", "coordinates": {"lng": 114.01, "lat": 30.0}}]
    result = build_anchor_candidates(_graph(), pois, {2, 3},
                                     to_wgs84=lambda x, y: (x, y), max_snap_m=20)
    anchors = {item["poi_id"]: item for item in result["anchors"]}
    assert anchors["poi_a"]["node_id"] == 2
    assert 9 < anchors["poi_a"]["snap_distance_m"] < 11
    assert anchors["poi_a"]["access_status"] == "last_meter_unverified"
    assert anchors["poi_b"]["node_id"] == 3
    assert result["excluded"] == []


def test_anchor_rejects_far_missing_and_ambiguous_pois_without_guessing():
    pois = [{"id": "poi_a", "name": "甲楼", "coordinates": {"lng": 114.0, "lat": 30.0}},
            {"id": "poi_b", "name": "甲楼", "coordinates": {"lng": 114.0, "lat": 30.0}},
            {"id": "poi_c", "name": "远楼", "coordinates": {"lng": 114.01, "lat": 30.0}},
            {"id": "poi_d", "name": "无坐标", "coordinates": {}}]
    result = build_anchor_candidates(_graph(), pois, {1},
                                     to_wgs84=lambda x, y: (x, y), max_snap_m=20)
    assert result["anchors"] == []
    assert {item["reason"] for item in result["excluded"]} == {
        "ambiguous_name", "too_far_from_permitted_graph", "invalid_coordinates",
    }


def test_anchor_order_and_nearest_tie_are_deterministic():
    pois = [{"id": "poi_b", "name": "乙楼", "coordinates": {"lng": 114.00005, "lat": 30.0}},
            {"id": "poi_a", "name": "甲楼", "coordinates": {"lng": 114.0, "lat": 30.0}}]
    first = build_anchor_candidates(_graph(), pois, {2, 1},
                                    to_wgs84=lambda x, y: (x, y), max_snap_m=20)
    second = build_anchor_candidates(_graph(), list(reversed(pois)), {1, 2},
                                     to_wgs84=lambda x, y: (x, y), max_snap_m=20)
    assert first == second
    assert [item["poi_id"] for item in first["anchors"]] == ["poi_a", "poi_b"]
    assert first["anchors"][1]["node_id"] == 1


def test_anchor_requires_a_real_permitted_graph_and_valid_limit():
    with pytest.raises(ValueError, match="permitted"):
        build_anchor_candidates(_graph(), [], set(), to_wgs84=lambda x, y: (x, y))
    with pytest.raises(ValueError, match="max_snap_m"):
        build_anchor_candidates(_graph(), [], {1}, to_wgs84=lambda x, y: (x, y),
                                max_snap_m=0)


def test_duplicate_poi_ids_are_excluded_instead_of_overwriting_an_anchor():
    pois = [{"id": "same", "name": "甲楼", "coordinates": {"lng": 114.0, "lat": 30.0}},
            {"id": "same", "name": "乙楼", "coordinates": {"lng": 114.0, "lat": 30.0}}]
    result = build_anchor_candidates(_graph(), pois, {1},
                                     to_wgs84=lambda x, y: (x, y))
    assert result["anchors"] == []
    assert [item["reason"] for item in result["excluded"]] == ["duplicate_poi_id"] * 2
