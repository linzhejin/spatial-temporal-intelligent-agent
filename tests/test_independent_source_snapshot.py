"""Source-derived mode and event evidence stays separate from product routing."""

import json
from pathlib import Path

import networkx as nx
import pytest

from experiments.independent_source_snapshot import (
    active_closed_edges,
    mode_edge_evidence,
)


def test_mode_evidence_respects_specific_access_and_multitag_steps():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="footway", length=10)
    graph.add_edge(2, 3, key=0, highway="['service', 'steps']", length=10)
    graph.add_edge(3, 4, key=0, highway="service", access="no", foot="yes", length=10)
    graph.add_edge(4, 5, key=0, highway="service", allowed_modes="['walk']", length=10)
    graph.add_edge(5, 6, key=0, highway="motorway", length=10)

    walk = mode_edge_evidence(graph, "walk")
    bike = mode_edge_evidence(graph, "bike")
    drive = mode_edge_evidence(graph, "drive")
    assert (1, 2, 0) in walk["permitted"]
    assert (2, 3, 0) in walk["permitted"]
    assert (3, 4, 0) in walk["permitted"]  # foot=yes overrides access=no
    assert (5, 6, 0) in walk["denied"]
    assert (2, 3, 0) in bike["denied"]
    assert (3, 4, 0) in bike["denied"]
    assert (4, 5, 0) in bike["denied"]
    assert (1, 2, 0) in drive["denied"]
    assert (2, 3, 0) in drive["denied"]
    assert (5, 6, 0) in drive["permitted"]


def test_unknown_access_and_missing_highway_do_not_become_verified_open():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="path", access="unusual", length=10)
    graph.add_edge(2, 3, key=0, length=10)
    result = mode_edge_evidence(graph, "walk")
    assert result["unknown"] == {(1, 2, 0), (2, 3, 0)}
    assert not result["permitted"]


def test_active_closure_uses_mode_time_revocation_and_exact_edge():
    fixture = json.loads(Path("experiments/datasets/road_events_synthetic_v1.json")
                         .read_text(encoding="utf-8"))
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=10, highway="service")
    graph.add_edge(2, 1, key=0, length=10, highway="service")
    snapshots = {item["id"]: item for item in fixture["snapshots"]}
    assert active_closed_edges(graph, snapshots["synthetic_mode_specific_closure"], "bike") == {(1, 2, 0)}
    assert active_closed_edges(graph, snapshots["synthetic_mode_specific_closure"], "walk") == set()
    for snapshot_id in ("synthetic_walk_slowdown", "synthetic_notice_only",
                        "synthetic_expired_closure", "synthetic_revoked_closure"):
        assert active_closed_edges(graph, snapshots[snapshot_id], "walk") == set()
    bad = {"as_of_epoch": fixture["evaluation_epoch"], "conditions": [
        {"id": "missing", "action": "closure", "affected_modes": ["walk"],
         "start_time": 1, "end_time": 0,
         "edge": {"u": 9, "v": 2, "key": 0}},
    ]}
    with pytest.raises(ValueError, match="missing graph edge"):
        active_closed_edges(graph, bad, "walk")


def test_boundary_evidence_excludes_only_true_outside_edges():
    graph = nx.MultiDiGraph()
    graph.graph["routing_boundary_wkt"] = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
    for node, x, y in (("left", -1, 0.5), ("right", 2, 0.5),
                       ("far", 2, 2), ("farther", 3, 2), ("inside", 0.5, 0.5)):
        graph.add_node(node, x=x, y=y)
    graph.add_edge("left", "right", key=0, highway="footway", length=3)
    graph.add_edge("far", "farther", key=0, highway="footway", length=1)
    graph.add_edge("inside", "right", key=0, highway="footway", length=2)
    result = mode_edge_evidence(graph, "walk", require_boundary=True)
    assert ("left", "right", 0) in result["permitted"]
    assert ("inside", "right", 0) in result["permitted"]
    assert ("far", "farther", 0) in result["denied"]

    boundary_wkt = graph.graph.pop("routing_boundary_wkt")
    supplied = mode_edge_evidence(graph, "walk", require_boundary=True,
                                  boundary_wkt=boundary_wkt)
    assert supplied["permitted"] == result["permitted"]
    with pytest.raises(ValueError, match="routing boundary"):
        mode_edge_evidence(graph, "walk", require_boundary=True)


def test_nonfinite_coordinates_stay_unknown_in_boundary_evidence():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x="nan", y=0.5)
    graph.add_node(2, x=2, y=2)
    graph.add_edge(1, 2, key=0, highway="footway", length=1)
    result = mode_edge_evidence(
        graph, "walk", require_boundary=True,
        boundary_wkt="POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
    )
    assert (1, 2, 0) in result["unknown"]


def test_event_rejects_end_before_start_and_duplicate_ids():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, highway="service", length=1)
    event = {
        "id": "event-1", "action": "closure", "affected_modes": ["walk"],
        "edge": {"u": 1, "v": 2, "key": 0},
        "start_time": 20, "end_time": 10,
    }
    with pytest.raises(ValueError, match="event time"):
        active_closed_edges(graph, {"as_of_epoch": 15, "conditions": [event]}, "walk")
    event["end_time"] = 30
    with pytest.raises(ValueError, match="duplicate event"):
        active_closed_edges(graph, {"as_of_epoch": 15, "conditions": [event, event]}, "walk")
