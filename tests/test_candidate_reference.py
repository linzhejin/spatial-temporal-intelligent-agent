import networkx as nx
import pytest

from experiments.candidate_reference import rank_nearby_activities


def test_independent_candidate_reference_filters_activity_then_network_distance():
    graph = nx.MultiDiGraph()
    for node, x in (("start", 114.0), ("near", 114.001), ("far", 114.003)):
        graph.add_node(node, x=x, y=30.0)
    graph.add_edge("start", "near", key="a", length=100, highway="footway")
    graph.add_edge("near", "far", key="b", length=200, highway="footway")
    pois = [
        {"id": "origin", "coordinates": {"lng": 114.0, "lat": 30.0}, "activities": []},
        {"id": "near_meal", "coordinates": {"lng": 114.001, "lat": 30.0},
         "activities": ["canteen_meal"]},
        {"id": "far_meal", "coordinates": {"lng": 114.003, "lat": 30.0},
         "activities": ["restaurant_meal"]},
        {"id": "basketball", "coordinates": {"lng": 114.0, "lat": 30.0},
         "activities": ["basketball"]},
    ]
    result = rank_nearby_activities(graph, pois, start_poi_id="origin",
                                    activities={"canteen_meal", "restaurant_meal"},
                                    coordinates="wgs84")
    assert [row["poi_id"] for row in result["reachable"]] == ["near_meal", "far_meal"]
    assert [round(row["network_distance_m"]) for row in result["reachable"]] == [100, 300]
    assert result["unreachable"] == []


def test_reference_respects_blocked_modes_and_keeps_unreachable_explicit():
    graph = nx.MultiDiGraph()
    graph.add_node("s", x=114.0, y=30.0)
    graph.add_node("d", x=114.001, y=30.0)
    graph.add_edge("s", "d", key="closed", length=100,
                   highway="footway", blocked_modes=["walk"])
    pois = [{"id": "origin", "coordinates": {"lng": 114.0, "lat": 30.0}},
            {"id": "meal", "coordinates": {"lng": 114.001, "lat": 30.0},
             "activities": ["canteen_meal"]}]
    with pytest.raises(ValueError, match="no walk-routable nodes"):
        rank_nearby_activities(graph, pois, start_poi_id="origin",
                               activities={"canteen_meal"}, coordinates="wgs84")


def test_candidate_snaps_to_walk_routable_node_instead_of_nearer_drive_only_node():
    graph = nx.MultiDiGraph()
    for node, x in (("s", 114.0), ("walk", 114.001),
                    ("drive", 114.00105), ("drive_end", 114.002)):
        graph.add_node(node, x=x, y=30.0)
    graph.add_edge("s", "walk", key="walk", length=100, highway="footway")
    graph.add_edge("drive", "drive_end", key="car", length=90, highway="service",
                   foot="no")
    pois = [
        {"id": "origin", "coordinates": {"lng": 114.0, "lat": 30.0}},
        {"id": "meal", "coordinates": {"lng": 114.00105, "lat": 30.0},
         "activities": ["canteen_meal"]},
    ]
    result = rank_nearby_activities(graph, pois, start_poi_id="origin",
                                    activities={"canteen_meal"}, coordinates="wgs84")
    assert [row["poi_id"] for row in result["reachable"]] == ["meal"]
    assert result["unreachable"] == []
    assert result["reachable"][0]["destination_access_m"] > 0
