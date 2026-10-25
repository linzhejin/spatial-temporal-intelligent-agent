"""Independent graph-level route evidence must fail closed."""

import networkx as nx

from experiments.independent_route_oracle import inspect_route_edges


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b", key=0, length=5, highway="footway")
    graph.add_edge("a", "b", key=1, length=9, highway="steps")
    graph.add_edge("b", "c", key=0, length=7, highway="footway")
    graph.add_edge("c", "d", key=0, length=3, highway="footway")
    return graph


def _inspect(edges, **overrides):
    allowed = {("a", "b", 0), ("a", "b", 1), ("b", "c", 0), ("c", "d", 0)}
    args = {"start_node": "a", "end_node": "d", "via_nodes": ["b", "c"],
            "mode": "walk", "permitted_edge_ids": allowed,
            "closed_edge_ids": set(), "avoid_steps": True}
    args.update(overrides)
    return inspect_route_edges(_graph(), edges, **args)


def test_exact_parallel_edge_and_ordered_vias_are_checked():
    valid = _inspect([["a", "b", 0], ["b", "c", 0], ["c", "d", 0]])
    assert valid["verdict"] == "pass"
    assert valid["length_m"] == 15
    assert valid["scope"] == "frozen_graph_only"
    steps = _inspect([["a", "b", 1], ["b", "c", 0], ["c", "d", 0]])
    assert steps["verdict"] == "fail"
    assert "marked_steps_used" in steps["issues"]


def test_closure_wrong_direction_budget_and_via_order_fail():
    edges = [["a", "b", 0], ["b", "c", 0], ["c", "d", 0]]
    assert "closed_edge_used" in _inspect(edges, closed_edge_ids={("b", "c", 0)})["issues"]
    assert "distance_budget_exceeded" in _inspect(edges, max_distance_m=14)["issues"]
    assert "ordered_via_mismatch" in _inspect(edges, via_nodes=["c", "b"])["issues"]
    assert "discontinuous_edges" in _inspect([["a", "b", 0], ["c", "d", 0]])["issues"]
    assert "missing_edge" in _inspect([["b", "a", 0]])["issues"]


def test_missing_mode_or_event_snapshot_is_unknown_not_a_pass():
    edges = [["a", "b", 0], ["b", "c", 0], ["c", "d", 0]]
    no_mode = _inspect(edges, permitted_edge_ids=None)
    assert no_mode["verdict"] == "unknown"
    assert "mode_evidence_missing" in no_mode["unknowns"]
    no_events = _inspect(edges, closed_edge_ids=None)
    assert no_events["verdict"] == "unknown"
    assert "event_snapshot_missing" in no_events["unknowns"]


def test_edge_with_missing_length_never_passes():
    graph = _graph()
    del graph["b"]["c"][0]["length"]
    result = inspect_route_edges(
        graph, [["a", "b", 0], ["b", "c", 0], ["c", "d", 0]],
        start_node="a", end_node="d", via_nodes=[], mode="walk",
        permitted_edge_ids={("a", "b", 0), ("b", "c", 0), ("c", "d", 0)},
        closed_edge_ids=set(), avoid_steps=False,
    )
    assert result["verdict"] == "unknown"
    assert "edge_length_missing" in result["unknowns"]
