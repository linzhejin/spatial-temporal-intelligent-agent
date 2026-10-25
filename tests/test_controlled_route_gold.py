"""A separate graph oracle derives route feasibility from frozen edge evidence."""

import networkx as nx

from experiments.controlled_route_gold import derive_graph_route


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key="slow", length=12, highway="footway")
    graph.add_edge(1, 2, key="fast", length=8, highway="footway")
    graph.add_edge(2, 3, key=0, length=5, highway="footway")
    graph.add_edge(1, 3, key=0, length=10, highway="steps")
    return graph


def _solve(graph, permitted, unknown=frozenset(), closed=frozenset(), **kwargs):
    return derive_graph_route(
        graph, start_node=1, end_node=3, via_nodes=kwargs.pop("via_nodes", []),
        mode="walk", permitted_edge_ids=set(permitted),
        unknown_edge_ids=set(unknown), closed_edge_ids=set(closed), **kwargs,
    )


def test_ordered_via_uses_exact_shortest_parallel_edge():
    graph = _graph()
    result = _solve(graph, graph.edges(keys=True), via_nodes=[2])
    assert result["verdict"] == "feasible"
    assert result["route_edge_ids"] == [[1, 2, "fast"], [2, 3, 0]]
    assert result["length_m"] == 13


def test_closed_direct_edge_and_avoid_steps_do_not_enter_route():
    graph = _graph()
    result = _solve(graph, graph.edges(keys=True), closed={(1, 3, 0)},
                    avoid_steps=True)
    assert result["verdict"] == "feasible"
    assert result["route_edge_ids"] == [[1, 2, "fast"], [2, 3, 0]]


def test_unconfirmed_connector_means_unknown_not_impossible():
    graph = _graph()
    result = _solve(graph, {(1, 2, "fast")}, unknown={(2, 3, 0)})
    assert result["verdict"] == "unknown"
    assert result["reason"] == "possible_only_via_unknown_edges"


def test_disconnected_even_with_unknown_edges_is_infeasible_in_frozen_graph():
    graph = _graph()
    result = _solve(graph, {(1, 2, "fast")})
    assert result["verdict"] == "infeasible"
    assert result["reason"] == "no_path_in_frozen_eligible_graph"


def test_distance_budget_does_not_assume_unknown_edge_length():
    graph = _graph()
    result = _solve(graph, {(1, 2, "fast"), (2, 3, 0)},
                    unknown={(1, 3, 0)}, max_distance_m=11)
    assert result["verdict"] == "unknown"
    assert result["reason"] == "budget_may_use_unknown_edges"
