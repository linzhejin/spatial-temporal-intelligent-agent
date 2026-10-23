"""Independent T1 graph references must preserve modes, stops and access risk."""
from __future__ import annotations

import networkx as nx

from experiments.route_reference import build_route_reference


def _fixture():
    graph = nx.MultiDiGraph()
    for node, lng in (("a", 114.0), ("b", 114.0001), ("c", 114.0002)):
        graph.add_node(node, x=lng, y=30.0)
    graph.add_edge("a", "b", key="steps", length=10, highway="steps")
    graph.add_edge("b", "c", key="walk", length=10, highway="footway")
    graph.add_edge("a", "c", key="path", length=40, highway="path")
    pois = [{"id": node, "coordinates": {"lng": lng, "lat": 30.0}}
            for node, lng in (("a", 114.0), ("b", 114.0001), ("c", 114.0002))]
    return graph, pois


def test_reference_uses_ordered_stops_and_avoids_steps() -> None:
    graph, pois = _fixture()
    direct = build_route_reference(graph, pois, start_poi_id="a", end_poi_id="c",
                                   via_poi_ids=[], mode="walk",
                                   constraints={"avoid_steps": True}, coordinates="wgs84")
    assert direct["graph_feasible"] is True
    assert direct["graph_distance_m"] == 40
    assert direct["start_node"] == "a"
    assert direct["end_node"] == "c"
    assert direct["access_uncertain"] is False
    via = build_route_reference(graph, pois, start_poi_id="a", end_poi_id="c",
                                via_poi_ids=["b"], mode="walk",
                                constraints={"avoid_steps": True}, coordinates="wgs84")
    assert via["graph_feasible"] is False
    assert via["via_nodes"] == ["b"]


def test_reference_flags_large_last_meter_and_rejects_unknown_constraint() -> None:
    graph, pois = _fixture()
    pois[2]["coordinates"]["lng"] = 114.001
    result = build_route_reference(graph, pois, start_poi_id="a", end_poi_id="c",
                                   via_poi_ids=[], mode="walk", constraints={},
                                   coordinates="wgs84")
    assert result["access_uncertain"] is True
    assert result["snapped_pois"][-1]["access_over_30m"] is True
    import pytest
    with pytest.raises(ValueError, match="unsupported independent hard constraint"):
        build_route_reference(graph, pois, start_poi_id="a", end_poi_id="c",
                              via_poi_ids=[], mode="walk",
                              constraints={"max_duration_min": 5}, coordinates="wgs84")


def test_reference_snaps_driving_to_routable_nodes() -> None:
    graph = nx.MultiDiGraph()
    for node, lng in (("a", 114.0), ("road_start", 114.00005),
                      ("road_end", 114.0002)):
        graph.add_node(node, x=lng, y=30.0)
    graph.add_edge("a", "road_start", length=5, highway="footway")
    graph.add_edge("road_start", "road_end", length=18, highway="residential")
    pois = [{"id": "origin", "coordinates": {"lng": 114.0, "lat": 30.0}},
            {"id": "destination", "coordinates": {"lng": 114.0002, "lat": 30.0}}]
    result = build_route_reference(graph, pois, start_poi_id="origin",
                                   end_poi_id="destination", via_poi_ids=[],
                                   mode="drive", constraints={}, coordinates="wgs84")
    assert result["start_node"] == "road_start"
    assert result["graph_feasible"] is True
    assert result["graph_distance_m"] == 18
