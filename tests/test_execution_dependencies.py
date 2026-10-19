"""Safe reuse is limited to certified exact shortest paths on effective graphs."""

from itertools import product

import networkx as nx

from agents.execution_dependencies import (
    can_reuse_exact_shortest_path,
    can_reuse_exact_shortest_path_after_closure_delta,
    condition_change_set,
    condition_snapshot,
    effective_graph_delta,
)


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=5.0)
    graph.add_edge(2, 4, key=0, length=5.0)
    graph.add_edge(1, 3, key=0, length=6.0)
    graph.add_edge(3, 4, key=0, length=6.0)
    return graph


def _decision(old, new, **overrides):
    inputs = {
        "old_effective_graph": old,
        "new_effective_graph": new,
        "old_edge_ids": [(1, 2, 0), (2, 4, 0)],
        "start_node": 1,
        "end_node": 4,
        "strategy": "shortest",
        "old_exact_optimality_proven": True,
        "same_mode_constraints_and_endpoints": True,
    }
    inputs.update(overrides)
    return can_reuse_exact_shortest_path(**inputs)


def test_off_route_closure_can_reuse_certified_shortest_path():
    old = _graph()
    new = old.copy()
    new.remove_edge(1, 3, 0)

    decision = _decision(old, new)

    assert decision["reuse_path"] is True
    assert decision["requires_reverification"] is True
    assert nx.shortest_path_length(new, 1, 4, weight="length") == 10


def test_off_route_cost_increase_can_reuse_but_opening_or_decrease_cannot():
    old = _graph()
    increased = old.copy()
    increased[1][3][0]["length"] = 9.0
    assert _decision(old, increased)["reuse_path"] is True

    decreased = old.copy()
    decreased[1][3][0]["length"] = 3.0
    decreased[3][4][0]["length"] = 3.0
    assert _decision(old, decreased)["reuse_path"] is False
    assert nx.shortest_path_length(decreased, 1, 4, weight="length") == 6

    opened = old.copy()
    opened.add_edge(1, 4, key=0, length=1.0)
    assert _decision(old, opened)["reuse_path"] is False


def test_used_edge_change_or_unknown_cost_requires_recomputation():
    old = _graph()
    closed = old.copy()
    closed.remove_edge(1, 2, 0)
    assert _decision(old, closed)["reuse_path"] is False

    increased = old.copy()
    increased[1][2][0]["length"] = 8.0
    assert _decision(old, increased)["reuse_path"] is False

    unknown = old.copy()
    unknown[1][3][0]["length"] = None
    assert _decision(old, unknown)["reuse_path"] is False


def test_missing_certificate_or_changed_task_cannot_reuse():
    old = _graph()
    new = old.copy()
    new.remove_edge(1, 3, 0)
    assert _decision(old, new, old_exact_optimality_proven=False)["reuse_path"] is False
    assert _decision(old, new, strategy="scenery")["reuse_path"] is False
    assert _decision(old, new, same_mode_constraints_and_endpoints=False)["reuse_path"] is False
    assert _decision(old, new, old_edge_ids=[(1, 3, 0), (3, 4, 0)])["reuse_path"] is False


def test_effective_graph_delta_records_every_edge_change_not_only_old_path():
    old = _graph()
    new = old.copy()
    new.remove_edge(1, 3, 0)
    new[3][4][0]["length"] = 8.0
    new[1][2][0]["length"] = 4.0
    new.add_edge(4, 1, key=0, length=2.0)

    changes = effective_graph_delta(old, new)

    assert changes == {
        "known": True,
        "added_edge_ids": [[4, 1, 0]],
        "removed_edge_ids": [[1, 3, 0]],
        "cost_increases": [{"edge_id": [3, 4, 0], "before_m": 6.0, "after_m": 8.0}],
        "cost_decreases": [{"edge_id": [1, 2, 0], "before_m": 5.0, "after_m": 4.0}],
    }


def test_parallel_keys_and_reverse_edges_are_distinct_changes():
    old = _graph()
    old.add_edge(1, 2, key=1, length=7.0)
    old.add_edge(2, 1, key=0, length=7.0)
    new = old.copy()
    new[1][2][1]["length"] = 4.0
    new[2][1][0]["length"] = 3.0

    changes = effective_graph_delta(old, new)
    decision = _decision(old, new)

    assert changes["cost_decreases"] == [
        {"edge_id": [1, 2, 1], "before_m": 7.0, "after_m": 4.0},
        {"edge_id": [2, 1, 0], "before_m": 7.0, "after_m": 3.0},
    ]
    assert decision["reuse_path"] is False
    assert decision["reason"] == "edge_cost_decreased"


def test_unknown_edge_cost_makes_graph_delta_unusable():
    old = _graph()
    new = old.copy()
    new[1][3][0]["length"] = float("nan")

    assert effective_graph_delta(old, new) == {
        "known": False,
        "added_edge_ids": [],
        "removed_edge_ids": [],
        "cost_increases": [],
        "cost_decreases": [],
    }


def test_closure_delta_reuses_only_when_old_route_survives_without_opening_edges():
    path = [(1, 2, 0), (2, 4, 0)]

    off_path = can_reuse_exact_shortest_path_after_closure_delta(
        old_edge_ids=path, start_node=1, end_node=4,
        previously_closed_edge_ids=[], current_closed_edge_ids=[(1, 3, 0)],
        strategy="shortest", old_exact_optimality_proven=True,
        same_mode_constraints_and_endpoints=True,
    )
    used_path = can_reuse_exact_shortest_path_after_closure_delta(
        old_edge_ids=path, start_node=1, end_node=4,
        previously_closed_edge_ids=[], current_closed_edge_ids=[(1, 2, 0)],
        strategy="shortest", old_exact_optimality_proven=True,
        same_mode_constraints_and_endpoints=True,
    )
    opened = can_reuse_exact_shortest_path_after_closure_delta(
        old_edge_ids=path, start_node=1, end_node=4,
        previously_closed_edge_ids=[(1, 3, 0)], current_closed_edge_ids=[],
        strategy="shortest", old_exact_optimality_proven=True,
        same_mode_constraints_and_endpoints=True,
    )

    assert off_path["reuse_path"] is True
    assert off_path["change_set"]["removed_edge_ids"] == [[1, 3, 0]]
    assert used_path["reuse_path"] is False
    assert used_path["reason"] == "used_edge_changed"
    assert opened["reuse_path"] is False
    assert opened["reason"] == "new_edge_available"


def test_closure_delta_rejects_unknown_edges_or_changed_route_semantics():
    common = {
        "old_edge_ids": [(1, 2, 0), (2, 4, 0)], "start_node": 1, "end_node": 4,
        "previously_closed_edge_ids": [], "current_closed_edge_ids": [],
        "strategy": "shortest", "old_exact_optimality_proven": True,
        "same_mode_constraints_and_endpoints": True,
    }
    malformed = can_reuse_exact_shortest_path_after_closure_delta(
        **{**common, "current_closed_edge_ids": [[1, 3]]},
    )
    changed = can_reuse_exact_shortest_path_after_closure_delta(
        **{**common, "same_mode_constraints_and_endpoints": False},
    )

    assert malformed["reuse_path"] is False
    assert malformed["reason"] == "unknown_edge_change"
    assert changed["reuse_path"] is False
    assert changed["reason"] == "unsupported_or_uncertified_task"


def test_closure_delta_malformed_old_edge_ids_fail_closed_without_raising():
    common = {
        "start_node": 1,
        "end_node": 4,
        "previously_closed_edge_ids": [],
        "current_closed_edge_ids": [],
        "strategy": "shortest",
        "old_exact_optimality_proven": True,
        "same_mode_constraints_and_endpoints": True,
    }

    for path in (
        [(1, [2], 0), ([2], 4, 0)],
        [(1, 2, {}), (2, 4, 0)],
        [(True, 2, 0), (2, 4, 0)],
    ):
        decision = can_reuse_exact_shortest_path_after_closure_delta(
            old_edge_ids=path, **common,
        )
        assert decision["reuse_path"] is False
        assert decision["reason"] == "unknown_edge_change"


def test_every_reused_path_matches_full_dijkstra_across_small_graph_changes():
    old = _graph()
    path = [(1, 2, 0), (2, 4, 0)]
    actions = ("keep", "delete", "increase", "decrease")
    edge_ids = list(old.edges(keys=True))

    for choices in product(actions, repeat=len(edge_ids)):
        changed = old.copy()
        for edge, action in zip(edge_ids, choices):
            if action == "delete":
                changed.remove_edge(*edge)
            elif action == "increase":
                changed[edge[0]][edge[1]][edge[2]]["length"] *= 1.5
            elif action == "decrease":
                changed[edge[0]][edge[1]][edge[2]]["length"] *= 0.5

        decision = _decision(old, changed, old_edge_ids=path)
        if not decision["reuse_path"]:
            continue
        reused_cost = sum(changed[u][v][key]["length"] for u, v, key in path)
        full_cost = nx.shortest_path_length(changed, 1, 4, weight="length")
        assert reused_cost == full_cost, (choices, decision)

    opened = old.copy()
    opened.add_edge(1, 4, key=0, length=1.0)
    assert _decision(old, opened, old_edge_ids=path)["reuse_path"] is False


def test_condition_changeset_tracks_added_removed_and_modified_active_events():
    prior = condition_snapshot([
        {"id": "gate-a", "action": "closure", "edge": {"edges": [[1, 2, 0] ]}},
        {"id": "notice-b", "action": "notice", "edge": {"edges": [[3, 4, 0] ]}},
    ])
    current = [
        {"id": "gate-a", "action": "slowdown", "cost_multiplier": 1.5,
         "edge": {"edges": [[1, 2, 0] ]}},
        {"id": "closure-c", "action": "closure", "edge": {"edges": [[2, 4, 0] ]}},
    ]

    assert condition_change_set(prior, current) == {
        "added_event_ids": ["closure-c"],
        "removed_event_ids": ["notice-b"],
        "modified_event_ids": ["gate-a"],
        "unchanged_event_ids": [],
    }
