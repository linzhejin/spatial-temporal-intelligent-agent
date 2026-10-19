"""Conservative reuse proof for a previously certified exact shortest path.

Both graphs must already reflect the *same* travel mode, hard filters and the
respective road-condition snapshots. Only pure nonnegative edge length is the
objective. This function reuses an edge sequence, never its old ETA, event
explanation or display geometry; those must be rebuilt and verified again.
"""

from __future__ import annotations

import hashlib
import json
import math


def _length(value):
    if isinstance(value, bool):
        return None
    try:
        length = float(value)
    except (TypeError, ValueError):
        return None
    return length if math.isfinite(length) and length >= 0 else None


def _edge_lengths(graph):
    lengths = {}
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        length = _length(attrs.get("length"))
        if length is None:
            return None
        lengths[(u, v, key)] = length
    return lengths


def effective_graph_delta(old_effective_graph, new_effective_graph):
    """Describe the full effective edge-set/cost change between snapshots.

    This deliberately compares every edge, including edges outside the old
    route. A route-local scan would miss a newly opened or cheaper shortcut.
    """
    unknown = {
        "known": False,
        "added_edge_ids": [],
        "removed_edge_ids": [],
        "cost_increases": [],
        "cost_decreases": [],
    }
    old_lengths = _edge_lengths(old_effective_graph)
    new_lengths = _edge_lengths(new_effective_graph)
    if old_lengths is None or new_lengths is None:
        return unknown

    def sort_edges(edges):
        return sorted(edges, key=lambda edge: tuple(map(repr, edge)))

    added = sort_edges(new_lengths.keys() - old_lengths.keys())
    removed = sort_edges(old_lengths.keys() - new_lengths.keys())
    increases, decreases = [], []
    for edge in sort_edges(old_lengths.keys() & new_lengths.keys()):
        before, after = old_lengths[edge], new_lengths[edge]
        if after > before:
            increases.append({"edge_id": list(edge), "before_m": before, "after_m": after})
        elif after < before:
            decreases.append({"edge_id": list(edge), "before_m": before, "after_m": after})
    return {
        "known": True,
        "added_edge_ids": [list(edge) for edge in added],
        "removed_edge_ids": [list(edge) for edge in removed],
        "cost_increases": increases,
        "cost_decreases": decreases,
    }


def condition_snapshot(conditions):
    """Return a compact, stable record of the active condition inputs."""
    if not isinstance(conditions, list) or any(not isinstance(item, dict) for item in conditions):
        raise ValueError("road-condition snapshot must be a list of objects")
    rows = []
    for condition in conditions:
        encoded = json.dumps(
            condition, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()
        raw_id = condition.get("id")
        event_id = str(raw_id) if raw_id is not None else f"anonymous:{fingerprint}"
        rows.append({"event_id": event_id, "fingerprint": fingerprint})
    return sorted(rows, key=lambda row: (row["event_id"], row["fingerprint"]))


def condition_change_set(previous_snapshot, current_conditions):
    """Compare the complete active event set, including off-route changes."""
    previous = {}
    for row in previous_snapshot if isinstance(previous_snapshot, list) else []:
        if not isinstance(row, dict):
            continue
        event_id, fingerprint = row.get("event_id"), row.get("fingerprint")
        if isinstance(event_id, str) and isinstance(fingerprint, str):
            previous.setdefault(event_id, []).append(fingerprint)
    current = {}
    for row in condition_snapshot(current_conditions):
        current.setdefault(row["event_id"], []).append(row["fingerprint"])

    added, removed, modified, unchanged = [], [], [], []
    for event_id in sorted(previous.keys() | current.keys()):
        old_values, new_values = previous.get(event_id), current.get(event_id)
        if old_values is None:
            added.append(event_id)
        elif new_values is None:
            removed.append(event_id)
        elif sorted(old_values) == sorted(new_values):
            unchanged.append(event_id)
        else:
            modified.append(event_id)
    return {
        "added_event_ids": added,
        "removed_event_ids": removed,
        "modified_event_ids": modified,
        "unchanged_event_ids": unchanged,
    }


def can_reuse_exact_shortest_path(
    *, old_effective_graph, new_effective_graph, old_edge_ids,
    start_node, end_node, strategy, old_exact_optimality_proven,
    same_mode_constraints_and_endpoints,
):
    """Prove that deletion/increase away from an old optimum preserves it.

    ``old_exact_optimality_proven`` must come from a trusted server-side exact
    shortest-path computation, never a client assertion or the result verifier's
    edge-validity report. Unknown costs, an opened edge, a cost decrease, or any
    change to the old path all trigger a fresh search.
    """
    def decision(reuse, reason, change_set=None):
        return {"reuse_path": reuse, "reason": reason,
                "requires_reverification": reuse,
                "change_set": change_set}

    if (strategy != "shortest" or old_exact_optimality_proven is not True
            or same_mode_constraints_and_endpoints is not True):
        return decision(False, "unsupported_or_uncertified_task")
    if not isinstance(old_edge_ids, (list, tuple)):
        return decision(False, "invalid_old_path")
    previous = start_node
    path = []
    for edge in old_edge_ids:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            return decision(False, "invalid_old_path")
        try:
            edge = tuple(edge)
            if edge[0] != previous or not old_effective_graph.has_edge(*edge):
                return decision(False, "invalid_old_path")
        except (TypeError, ValueError):
            return decision(False, "invalid_old_path")
        path.append(edge)
        previous = edge[1]
    if previous != end_node:
        return decision(False, "invalid_old_path")

    change_set = effective_graph_delta(old_effective_graph, new_effective_graph)
    if not change_set["known"]:
        return decision(False, "unknown_edge_cost", change_set)
    if change_set["added_edge_ids"]:
        return decision(False, "new_edge_available", change_set)
    removed = {tuple(edge) for edge in change_set["removed_edge_ids"]}
    increased = {tuple(item["edge_id"]) for item in change_set["cost_increases"]}
    decreased = {tuple(item["edge_id"]) for item in change_set["cost_decreases"]}
    if any(edge in removed or edge in increased or edge in decreased for edge in path):
        return decision(False, "used_edge_changed", change_set)
    if decreased:
        return decision(False, "edge_cost_decreased", change_set)
    return decision(True, "off_path_deletion_or_increase_only", change_set)


def can_reuse_exact_shortest_path_after_closure_delta(
    *, old_edge_ids, start_node, end_node, previously_closed_edge_ids,
    current_closed_edge_ids, strategy, old_exact_optimality_proven,
    same_mode_constraints_and_endpoints,
):
    """Check closure-only changes without rebuilding/comparing the whole graph.

    The caller must establish that the graph/data version and task semantics are
    unchanged, resolve active conditions strictly against the current mode
    graph, and fall back to a full search for any slowdown penalty. Given those
    preconditions, the only graph difference represented here is edge removal:
    adding closures away from a certified route preserves its exact optimum;
    opening a closure or closing a used edge does not.
    """
    def decision(reuse, reason, change_set=None):
        return {
            "reuse_path": reuse,
            "reason": reason,
            "requires_reverification": reuse,
            "change_set": change_set,
        }

    if (strategy != "shortest" or old_exact_optimality_proven is not True
            or same_mode_constraints_and_endpoints is not True):
        return decision(False, "unsupported_or_uncertified_task")

    def normalize_edge(value):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError
        edge = tuple(value)
        if any(
            isinstance(part, bool)
            or not isinstance(part, (str, int))
            or isinstance(part, str) and (not part or "\x00" in part)
            for part in edge
        ):
            raise ValueError
        return edge

    def edge_set(values):
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise ValueError
        result = set()
        for edge in values:
            result.add(normalize_edge(edge))
        return result

    try:
        if (isinstance(start_node, bool) or not isinstance(start_node, (str, int))
                or isinstance(end_node, bool) or not isinstance(end_node, (str, int))):
            raise ValueError
        if not isinstance(old_edge_ids, (list, tuple)):
            raise ValueError
        previous = start_node
        route = []
        for raw_edge in old_edge_ids:
            edge = normalize_edge(raw_edge)
            if edge[0] != previous:
                raise ValueError
            route.append(edge)
            previous = edge[1]
        if not route or previous != end_node:
            raise ValueError
        before = edge_set(previously_closed_edge_ids)
        after = edge_set(current_closed_edge_ids)
    except (TypeError, ValueError):
        return decision(False, "unknown_edge_change")

    reopened = before - after
    newly_closed = after - before
    change_set = {
        "known": True,
        "added_edge_ids": [list(edge) for edge in sorted(reopened, key=lambda row: tuple(map(repr, row)))],
        "removed_edge_ids": [list(edge) for edge in sorted(newly_closed, key=lambda row: tuple(map(repr, row)))],
        "cost_increases": [],
        "cost_decreases": [],
    }
    if reopened:
        return decision(False, "new_edge_available", change_set)
    if set(route) & newly_closed:
        return decision(False, "used_edge_changed", change_set)
    return decision(True, "off_path_deletion_or_increase_only", change_set)
