from copy import deepcopy

import networkx as nx
import pytest

from experiments.e3_product_cases import (
    active_events, coordinate_ref, effective_snapshot, path_listener_decision,
)


def test_coordinate_endpoint_keeps_wgs84_api_contract():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=114.36, y=30.53)
    assert coordinate_ref(graph, 1) == {
        "type": "coord", "coordinates": {"lng": 114.36, "lat": 30.53}}


@pytest.mark.parametrize("tag", ["steps", ["footway", "steps"], ("service", "steps"),
                                 "['footway', 'steps']", '["service", "steps"]'])
def test_no_steps_oracle_excludes_serialized_multitag_shortcut(tag):
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3, 4):
        graph.add_node(node, x=114.36 + node * .0001, y=30.53)
    graph.add_edge(1, 2, key=0, length=1, highway=tag)
    graph.add_edge(2, 4, key=0, length=1, highway="footway")
    graph.add_edge(1, 3, key=0, length=5, highway="footway")
    graph.add_edge(3, 4, key=0, length=5, highway="service")
    unrestricted, _ = effective_snapshot(graph, [], 100, "walk", {})
    constrained, _ = effective_snapshot(graph, [], 100, "walk", {"avoid_steps": True})
    assert nx.shortest_path_length(unrestricted, 1, 4, weight="length") == 2
    assert not constrained.has_edge(1, 2, 0)
    assert constrained.has_edge(1, 3, 0)
    assert nx.shortest_path_length(constrained, 1, 4, weight="length") == 10


def test_event_clock_is_half_open_and_revocation_and_mode_are_independent():
    event = {"id": "a", "start_time": 10, "end_time": 20,
             "affected_modes": ["walk"], "action": "closure"}
    assert active_events([event], 10, "walk") == [event]
    assert active_events([event], 20, "walk") == []
    assert active_events([event], 15, "bike") == []
    assert active_events([{**event, "revoked_at": 12}], 15, "walk") == []


def test_truth_uses_frozen_binding_and_max_slowdown_without_changing_length():
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3):
        graph.add_node(node, x=114.36 + node * 0.0001, y=30.53)
    graph.add_edge(1, 2, key=0, length=10, highway="footway")
    graph.add_edge(2, 3, key=0, length=20, highway="footway")
    events = [{"id": str(i), "action": "slowdown", "cost_multiplier": factor,
               "affected_modes": ["walk"], "bindings": {"walk": [[1, 2, 0]]}}
              for i, factor in enumerate([1.5, 2.0])]
    effective, labels = effective_snapshot(graph, events, 100, "walk", {})
    assert effective[1][2][0]["length"] == 10
    assert effective[1][2][0]["audit_duration_multiplier"] == 2
    assert graph[1][2][0].get("audit_duration_multiplier") is None
    assert len(labels) == 2


def test_old_path_listener_misses_new_shortcut_but_not_used_edge_removal():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=6)
    graph.add_edge(2, 4, key=0, length=6)
    old = deepcopy(graph)
    graph.add_edge(1, 3, key=0, length=5)
    graph.add_edge(3, 4, key=0, length=5)
    path = [[1, 2, 0], [2, 4, 0]]
    assert path_listener_decision(old, graph, path) is True
    graph.remove_edge(2, 4, 0)
    assert path_listener_decision(old, graph, path) is False
