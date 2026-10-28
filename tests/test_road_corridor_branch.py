"""A diagonal road leaving a closure must remain traversable."""
import networkx as nx

from spatial.road_conditions import apply_conditions_to_graph


def test_diagonal_junction_is_not_closed_as_a_parallel_road():
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        1: (114.0, 30.0), 2: (114.001, 30.0),
        3: (114.002, 30.0), 4: (114.001, 30.001),
    }.items():
        graph.add_node(node, x=x, y=y)
    for a, b in ((1, 2), (2, 3), (1, 4), (4, 3)):
        graph.add_edge(a, b, 0, length=100, highway="residential")
        graph.add_edge(b, a, 0, length=100, highway="residential")
    event = {"id": "diagonal-test", "type": "closure", "action": "closure",
             "affected_modes": ["walk"], "edge": {"edges": [[1, 2, 0]]},
             "start_time": 0, "end_time": 0}
    updated, _, closed, _ = apply_conditions_to_graph(graph, [event], "walk", strict=True)
    assert {(1, 2, 0), (2, 1, 0)} <= closed
    assert (1, 4, 0) not in closed and (4, 1, 0) not in closed
    assert nx.has_path(updated, 1, 3)


def test_short_diagonal_spur_is_not_closed():
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        1: (114.0, 30.0), 2: (114.001, 30.0), 4: (114.0001, 30.00008),
    }.items():
        graph.add_node(node, x=x, y=y)
    for a, b in ((1, 2), (1, 4)):
        graph.add_edge(a, b, 0, length=10, highway="residential")
        graph.add_edge(b, a, 0, length=10, highway="residential")
    event = {"id": "short-diagonal", "type": "closure", "action": "closure",
             "affected_modes": ["walk"], "edge": {"edges": [[1, 2, 0]]},
             "start_time": 0, "end_time": 0}
    _, _, closed, _ = apply_conditions_to_graph(graph, [event], "walk", strict=True)
    assert (1, 4, 0) not in closed and (4, 1, 0) not in closed


def test_longer_duplicate_edge_that_covers_selected_road_is_closed():
    graph = nx.MultiDiGraph()
    for node, (x, y) in {
        1: (114.0, 30.0), 2: (114.001, 30.0),
        3: (113.9995, 30.00003), 4: (114.0015, 30.00003),
    }.items():
        graph.add_node(node, x=x, y=y)
    for a, b in ((1, 2), (3, 4)):
        graph.add_edge(a, b, 0, length=100, highway="residential")
        graph.add_edge(b, a, 0, length=100, highway="residential")
    event = {"id": "long-parallel", "type": "closure", "action": "closure",
             "affected_modes": ["walk"], "edge": {"edges": [[1, 2, 0]]},
             "start_time": 0, "end_time": 0}
    _, _, closed, _ = apply_conditions_to_graph(graph, [event], "walk", strict=True)
    assert (3, 4, 0) in closed and (4, 3, 0) in closed
