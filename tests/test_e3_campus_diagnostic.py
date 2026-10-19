import networkx as nx

from experiments.e3_campus_diagnostic import (
    _make_variants,
    _select_production_decision,
    _shortest_route,
    evaluate_graph_change,
)


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=5.0)
    graph.add_edge(2, 4, key=0, length=5.0)
    graph.add_edge(1, 3, key=0, length=6.0)
    graph.add_edge(3, 4, key=0, length=6.0)
    return graph


def test_off_path_removal_keeps_certified_optimum():
    old_graph = _graph()
    new_graph = old_graph.copy()
    new_graph.remove_edge(1, 3, 0)

    result = evaluate_graph_change(old_graph, new_graph, 1, 4)

    assert result["reuse_path"] is True
    assert result["chosen_cost_m"] == result["oracle_cost_m"] == 10.0
    assert result["matches_oracle"] is True


def test_opening_off_path_edge_forces_full_search_even_if_optimum_is_unchanged():
    new_graph = _graph()
    old_graph = new_graph.copy()
    old_graph.remove_edge(1, 3, 0)

    result = evaluate_graph_change(old_graph, new_graph, 1, 4)

    assert result["reuse_path"] is False
    assert result["reason"] == "new_edge_available"
    assert result["chosen_cost_m"] == result["oracle_cost_m"] == 10.0
    assert result["matches_oracle"] is True


def test_off_path_cost_decrease_forces_full_search_and_finds_new_optimum():
    old_graph = _graph()
    new_graph = old_graph.copy()
    new_graph[1][3][0]["length"] = 1.0

    result = evaluate_graph_change(old_graph, new_graph, 1, 4)

    assert result["reuse_path"] is False
    assert result["reason"] == "edge_cost_decreased"
    assert result["chosen_cost_m"] == result["oracle_cost_m"] == 7.0
    assert result["matches_oracle"] is True


def test_mixed_off_path_closure_and_slowdown_is_marked_for_full_search():
    graph = _graph()
    route = _shortest_route(graph, 1, 4)
    variants = _make_variants(graph, route)
    old_graph, changed_graph, change = variants[
        "mixed_off_path_closure_and_slowdown"
    ]

    assert change["requires_full_search"] is True
    assert change["current_closed_edge_ids"] == [[1, 3, 0]]
    assert change["changed_edge_ids"] == [[1, 3, 0], [3, 4, 0]]

    decision = _select_production_decision(
        change,
        old_edge_ids=route["edge_ids"],
        start_node=1,
        end_node=4,
    )
    oracle = evaluate_graph_change(old_graph, changed_graph, 1, 4)

    assert decision["reuse_path"] is False
    assert decision["reason"] == "road_slowdown_requires_full_search"
    assert oracle["matches_oracle"] is True
