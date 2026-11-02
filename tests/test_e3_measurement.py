"""Check that E3 observation measures real calls without replacing results."""

import json

import networkx as nx
import pytest

from experiments.e3_measurement import capture_execution_metrics
from spatial import routing
from storage import database, route_proof_repository, task_repository


def _graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, length=5.0)
    graph.add_edge(2, 4, key=0, length=5.0)
    graph.add_edge(1, 3, key=0, length=6.0)
    graph.add_edge(3, 4, key=0, length=6.0)
    return graph


def test_real_search_result_preserved_and_timing_does_not_count_expansions():
    graph = _graph()
    expected = routing._run_path(graph, 1, 4, routing._length_weight)
    original = routing._run_path
    original_dijkstra = nx.dijkstra_path
    with capture_execution_metrics() as metrics:
        assert routing._run_path(graph, 1, 4, routing._length_weight) == expected
    assert routing._run_path is original
    assert nx.dijkstra_path is original_dijkstra
    assert metrics["phases"]["path_search"]["calls"] == 1
    assert metrics["phases"]["dijkstra"]["calls"] == 1
    assert metrics["phases"]["path_search"]["errors"] == 0
    assert metrics["phases"]["path_search"]["total_ms"] >= 0
    assert metrics["searches"][0]["adjacency_expanded_nodes"] is None
    json.dumps(metrics)


def test_separate_probe_counts_real_adjacency_expansions_and_weight_evaluations():
    with capture_execution_metrics(expansion_probe=True) as metrics:
        assert routing._run_path(_graph(), 1, 4, routing._length_weight) == (
            [1, 2, 4], [(1, 2, 0), (2, 4, 0)])
    search = metrics["searches"][0]
    assert search["adjacency_expanded_nodes"] == 3  # 1, 2, 3; target 4 stops before expansion.
    assert search["weight_evaluations"] == 4
    assert search["status"] == "returned"
    assert search["source"] == 1 and search["target"] == 4


def test_unreachable_search_and_outer_exception_restore_every_patch():
    graph = _graph()
    graph.add_node(99)
    original = routing._run_path
    original_dijkstra = nx.dijkstra_path
    with pytest.raises(nx.NetworkXNoPath):
        with capture_execution_metrics(expansion_probe=True) as metrics:
            routing._run_path(graph, 1, 99, routing._length_weight)
    assert routing._run_path is original
    assert nx.dijkstra_path is original_dijkstra
    assert metrics["phases"]["path_search"]["errors"] == 1
    assert metrics["phases"]["dijkstra"]["errors"] == 1
    assert metrics["searches"][0]["status"] == "raised"
    assert metrics["searches"][0]["error_type"] == "NetworkXNoPath"
    # Node 4 has no out-neighbours, hence never evaluates an outgoing weight.
    assert metrics["searches"][0]["adjacency_expanded_nodes"] == 3
    with capture_execution_metrics() as next_metrics:
        routing._run_path(_graph(), 1, 4, routing._length_weight)
    assert next_metrics["phases"]["dijkstra"]["calls"] == 1


def test_zero_length_search_has_zero_adjacency_expansion():
    with capture_execution_metrics(expansion_probe=True) as metrics:
        assert nx.dijkstra_path(_graph(), 1, 1, weight="length") == [1]
    assert metrics["searches"][0]["adjacency_expanded_nodes"] == 0
    assert metrics["searches"][0]["weight_evaluations"] == 0


def test_parallel_edge_string_weight_uses_networkx_semantics():
    graph = _graph()
    graph.add_edge(1, 2, key=1, length=1.0)
    with capture_execution_metrics(expansion_probe=True) as metrics:
        assert nx.dijkstra_path(graph, 1, 4, weight="length") == [1, 2, 4]
    assert metrics["searches"][0]["status"] == "returned"


def test_real_compute_aliases_count_once_and_preserve_route_result():
    from agents import tools
    from api import routes

    graph = _graph()
    for node in graph:
        graph.nodes[node].update(x=114.36 + node * 0.0001, y=30.53)
    for _u, _v, attributes in graph.edges(data=True):
        attributes["highway"] = "footway"
    originals = [module.compute_route for module in (routing, tools, routes)]
    with capture_execution_metrics() as metrics:
        results = [module.compute_route(
            graph, 1, 4, mode="walk", road_conditions=[], strategy_name="shortest")
            for module in (routing, tools, routes)]
    assert [module.compute_route for module in (routing, tools, routes)] == originals
    assert all(result["recommended_edges"] == [(1, 2, 0), (2, 4, 0)] for result in results)
    assert all(result["recommended_length_m"] == 10.0 for result in results)
    assert metrics["phases"]["route_compute"]["calls"] == 3
    assert metrics["phases"]["path_search"]["calls"] == 3
    assert metrics["phases"]["dijkstra"]["calls"] == 3


def test_nested_context_is_rejected_without_disturbing_outer_capture():
    with capture_execution_metrics() as outer:
        with pytest.raises(RuntimeError, match="serial"):
            with capture_execution_metrics():
                pass
        nx.dijkstra_path(_graph(), 1, 4, weight="length")
    assert outer["phases"]["dijkstra"]["calls"] == 1


def test_actual_proof_validation_error_is_recorded_and_preserved():
    with capture_execution_metrics() as metrics:
        with pytest.raises(ValueError, match="UUID"):
            route_proof_repository.load_verified_proof(
                None, proof_id="invalid", owner_id="invalid", semantic_digest="a" * 64,
                graph_version="graph", event_version="event")
    assert metrics["phases"]["proof_read"]["calls"] == 1
    assert metrics["phases"]["proof_read"]["errors"] == 1


def test_real_database_proof_roundtrip_is_not_replaced(isolated_test_database):
    url = isolated_test_database
    database.initialize(url)
    owner, _token, _conversation = task_repository.create_conversation(url)
    binding = {
        "owner_id": owner, "semantic_digest": "a" * 64,
        "graph_version": "e3-measurement-test-graph",
        "event_version": "e3-measurement-test-event",
    }
    with capture_execution_metrics() as metrics:
        proof = route_proof_repository.create_verified_proof(
            url, **binding, selected_edges=[[1, 2, 0]], closed_edge_ids=[],
            ttl_seconds=60, event_snapshot=[])
        result = route_proof_repository.load_verified_proof(url, proof_id=proof, **binding)
    assert result["proof_id"] == proof
    assert result["owner_id"] == owner
    assert result["selected_edges"] == [[1, 2, 0]]
    with database.connect(url) as conn:
        stored = conn.execute("SELECT proof_id FROM route_proof WHERE proof_id=%s", (proof,)).fetchone()
    assert str(stored["proof_id"]) == proof
    assert metrics["phases"]["proof_issue"]["calls"] == 1
    assert metrics["phases"]["proof_read"]["calls"] == 1
    assert metrics["phases"]["proof_issue"]["errors"] == 0
