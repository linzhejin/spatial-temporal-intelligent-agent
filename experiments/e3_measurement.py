"""Observe actual product calls in a serial, isolated E3 experiment process.

The wrappers call the original functions with their original inputs and return
their results unchanged. Phase durations are inclusive and must not be summed:
for example, route_compute contains path_search. Request wall time belongs to
the caller and includes work outside these instrumented functions.

Timing runs leave the search weight callback untouched. A separate expansion
probe wraps the callback used by NetworkX and counts actual outgoing-adjacency
expansions: distinct source nodes for which a neighbour weight is evaluated in
one search. A target that causes early termination, and a settled node with no
outgoing neighbours, do not expand adjacency and are not counted. This is not
a count of all discovered/settled nodes, nor a substitute for runtime timing.
The probe's callback overhead means its elapsed times must not enter the main
timing summary. No node or call counts are inferred from product diagnostics.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
from threading import Lock
import time
from unittest.mock import patch


_CAPTURE_LOCK = Lock()
_PHASE_NAMES = (
    "proof_read", "proof_issue", "reuse_assessment", "route_compute",
    "verification", "path_search", "dijkstra",
)


def _node_label(value):
    """Keep metrics JSON-safe without storing graph data or request secrets."""
    return value if value is None or type(value) in (str, int, float, bool) else repr(value)


@contextmanager
def capture_execution_metrics(*, expansion_probe: bool = False):
    """Yield a JSON-compatible metrics dict while recording original calls.

    Global patches require a serial request loop in an isolated process. A
    second or nested capture raises before modifying any instrumentation.
    All patched functions are restored even if the request or setup raises.
    """
    if not isinstance(expansion_probe, bool):
        raise TypeError("expansion_probe must be a bool")
    if not _CAPTURE_LOCK.acquire(blocking=False):
        raise RuntimeError("execution capture requires serial, non-nested use")
    metrics = {
        "measurement_version": "e3-execution-metrics-v1",
        "expansion_probe": expansion_probe,
        "timing_eligible": not expansion_probe,
        "phase_times_are_inclusive": True,
        "expanded_node_definition": "distinct sources with evaluated outgoing neighbour weights per search",
        "phases": {name: {"calls": 0, "errors": 0, "total_ms": 0.0} for name in _PHASE_NAMES},
        "searches": [],
    }

    def observe(name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            phase = metrics["phases"][name]
            phase["calls"] += 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            except BaseException:
                phase["errors"] += 1
                raise
            finally:
                phase["total_ms"] += (time.perf_counter() - started) * 1000
        return measured

    try:
        import networkx as nx
        from networkx.algorithms.shortest_paths.weighted import _weight_function

        from agents import route_reuse, tools
        from api import routes
        from spatial import routing
        from storage import route_proof_repository

        with ExitStack() as stack:
            for target, attribute, name in (
                (route_proof_repository, "load_verified_proof", "proof_read"),
                (route_proof_repository, "create_verified_proof", "proof_issue"),
                (route_reuse, "assess_shortest_path_reuse", "reuse_assessment"),
                (tools, "_audit_tool_route", "verification"),
                (routing, "_run_path", "path_search"),
            ):
                stack.enter_context(patch.object(target, attribute, observe(name, getattr(target, attribute))))

            # The module imports share this function by value. Each alias gets
            # one wrapper around its current callable; the wrappers never call
            # one another, so a computation is counted once.
            for module in (routing, tools, routes):
                stack.enter_context(patch.object(
                    module, "compute_route", observe("route_compute", module.compute_route)))

            original_dijkstra = nx.dijkstra_path

            @wraps(original_dijkstra)
            def measured_dijkstra(graph, source, target, weight="weight", **kwargs):
                row = {
                    "source": _node_label(source), "target": _node_label(target),
                    "status": "running", "error_type": None,
                    "adjacency_expanded_nodes": None,
                    "weight_evaluations": None, "elapsed_ms": None,
                }
                metrics["searches"].append(row)
                phase = metrics["phases"]["dijkstra"]
                phase["calls"] += 1
                expanded = set()
                evaluations = 0
                actual_weight = weight
                if expansion_probe:
                    callback = _weight_function(graph, weight)

                    def observed_weight(u, v, attributes):
                        nonlocal evaluations
                        expanded.add(u)
                        evaluations += 1
                        return callback(u, v, attributes)

                    actual_weight = observed_weight
                started = time.perf_counter()
                try:
                    result = original_dijkstra(graph, source, target, weight=actual_weight, **kwargs)
                    row["status"] = "returned"
                    return result
                except BaseException as error:
                    row["status"] = "raised"
                    row["error_type"] = type(error).__name__
                    phase["errors"] += 1
                    raise
                finally:
                    elapsed = (time.perf_counter() - started) * 1000
                    phase["total_ms"] += elapsed
                    row["elapsed_ms"] = round(elapsed, 6)
                    if expansion_probe:
                        row["adjacency_expanded_nodes"] = len(expanded)
                        row["weight_evaluations"] = evaluations

            stack.enter_context(patch.object(nx, "dijkstra_path", measured_dijkstra))
            yield metrics
    finally:
        for phase in metrics["phases"].values():
            phase["total_ms"] = round(phase["total_ms"], 6)
        _CAPTURE_LOCK.release()
