import networkx as nx

from experiments.synthetic_road_events import build_event_bank


def test_synthetic_event_has_active_detour_and_expired_recovery():
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b", key="0", highway="footway", length=10)
    graph.add_edge("a", "c", key="0", highway="footway", length=8)
    graph.add_edge("c", "b", key="0", highway="footway", length=8)
    assignment = {"format": "vision-simulated-road-assignment-v1",
                  "graph_sha256": "f" * 64, "seed": 1,
                  "videos": [{"source_id": "other-city-video", "assigned_cases": [
                      {"id": "V1-01-1", "edge_id": ["a", "b", "0"]}]}]}
    bank = build_event_bank(graph, assignment)
    assert len(bank["cases"]) == 2
    active, expired = bank["cases"]
    assert active["expected_graph_effect"]["detour_edge_ids"] == [
        ["a", "c", "0"], ["c", "b", "0"]]
    assert active["expected_graph_effect"]["detour_distance_m"] == 16
    assert expired["expected_graph_effect"]["direct_edge_legal"] is True
    assert all(case["campus_road_assignment"] == "hypothetical_unpublished"
               for case in bank["cases"])
