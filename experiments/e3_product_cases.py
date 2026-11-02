"""Frozen development cases for the existing P3/E3 route-update mechanism.

Road bindings and static mode admissibility are shared inputs, frozen before
either update policy executes. Active-time filtering, event application and
path evaluation below are separate from the product event evaluator.
"""
from copy import deepcopy
import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
CLOCK = 1793577600.0
VARIANTS = (
    "unchanged", "off_path_closure", "on_path_closure", "reopen",
    "expiry", "revocation", "future_event", "expired_event", "other_mode",
    "on_path_slowdown", "off_path_slowdown", "mixed", "notice",
    "mode_change", "start_change", "end_change", "constraint_change",
    "joint_change", "graph_cost_decrease", "unreachable", "recovery",
)


def active_events(events, clock, mode):
    return [e for e in events if not e.get("revoked_at")
            and (not e.get("start_time") or e["start_time"] <= clock)
            and (not e.get("end_time") or clock < e["end_time"])
            and mode in e.get("affected_modes", [])]


def effective_snapshot(raw_graph, events, clock, mode, constraints):
    from experiments.e3_independent_audit import _tags
    from spatial.routing_index import get_routing_index

    graph = get_routing_index(raw_graph).for_mode(mode).graph.copy()
    excluded = {tuple(e) for e in constraints.get("excluded_edge_ids", [])}
    for u, v, k, attrs in list(graph.edges(keys=True, data=True)):
        # GraphML stores multi-valued tags as strings; use the independent
        # scorer's reader so the oracle cannot retain a serialized steps edge.
        steps = "steps" in _tags(attrs.get("highway", ""))
        if ((u, v, k) in excluded
                or constraints.get("avoid_steps") and steps
                or constraints.get("slope") == "avoid" and attrs.get("slope_level") == 5):
            graph.remove_edge(u, v, k)
    labels, closed = [], set()
    for event in active_events(events, clock, mode):
        edges = [tuple(e) for e in event["bindings"][mode]]
        labels.append({**{k: v for k, v in event.items() if k != "bindings"},
                       "edge_ids": [list(e) for e in edges]})
        if event["action"] == "closure":
            closed.update(edges)
        elif event["action"] == "slowdown":
            for edge in edges:
                if graph.has_edge(*edge):
                    a = graph[edge[0]][edge[1]][edge[2]]
                    a["audit_duration_multiplier"] = max(
                        a.get("audit_duration_multiplier", 1), event["cost_multiplier"])
    graph.remove_edges_from(closed)
    return graph, labels


def path_listener_decision(old_graph, new_graph, edges):
    """Deliberately weak diagnostic: inspect only edges on the old route."""
    return bool(edges) and all(
        old_graph.has_edge(*e) and new_graph.has_edge(*e)
        and old_graph[e[0]][e[1]][e[2]]["length"] == new_graph[e[0]][e[1]][e[2]]["length"]
        for e in edges)


def coordinate_ref(graph, node):
    a = graph.nodes[node]
    # The product coord endpoint contract is WGS84; POI records are GCJ02.
    return {"type": "coord", "coordinates": {"lng": float(a["x"]), "lat": float(a["y"])}}


def _bind_event(graph, edge, identifier, action="closure", modes=None):
    from spatial.road_conditions import resolve_condition_effects
    from spatial.routing_index import get_routing_index

    event = {"id": identifier, "type": "event", "name": identifier,
             "description": "E3受控道路事件", "action": action,
             "affected_modes": modes or ["walk"], "start_time": CLOCK - 100,
             "end_time": CLOCK + 100, "edge": {"edges": [list(edge)]}}
    if action == "slowdown":
        event["cost_multiplier"] = 1.8
    bindings = {}
    for mode in event["affected_modes"]:
        _, _, _, details = resolve_condition_effects(
            get_routing_index(graph).for_mode(mode).graph, [event], mode,
            strict=True, include_event_details=True)
        bindings[mode] = [list(e) for e in sorted(details[0][1])]
    return {**event, "bindings": bindings}


def _path(graph, start, end):
    nodes = nx.dijkstra_path(graph, start, end, weight="length")
    edges = [[u, v, min(graph[u][v], key=lambda k: (graph[u][v][k]["length"], str(k)))]
             for u, v in zip(nodes, nodes[1:])]
    return nodes, edges


def build_campus_cases(graph):
    """Keep all six declared ODs; freeze all variations before policy runs."""
    from spatial.coord_transform import gcj02_to_wgs84
    from spatial.network import get_nearest_node
    from spatial.poi import get_poi
    from spatial.routing_index import get_routing_index

    mode_graph = get_routing_index(graph).for_mode("walk").graph
    specs = json.loads((ROOT / "data/route_reference_samples.json").read_text(encoding="utf-8"))["samples"]
    cases = []
    for spec in specs:
        nodes = []
        for key in ("start", "end"):
            poi = get_poi(spec[key], fuzzy=False)
            coord = poi["coordinates"]
            nodes.append(get_nearest_node(mode_graph, *gcj02_to_wgs84(coord["lng"], coord["lat"])))
        path_nodes, path_edges = _path(mode_graph, *nodes)
        route_set = {tuple(e) for e in path_edges}
        on = _bind_event(graph, path_edges[len(path_edges) // 2], "on")
        off = None
        for edge in sorted(mode_graph.edges(keys=True)):
            if edge in route_set:
                continue
            candidate = _bind_event(graph, edge, "off")
            if not route_set.intersection(map(tuple, candidate["bindings"]["walk"])):
                off = candidate
                break
        if off is None:
            raise ValueError(f"no off-path event for {spec['id']}")
        old_closed = on
        closed_graph, _ = effective_snapshot(graph, [on], CLOCK, "walk", {})
        if not nx.has_path(closed_graph, *nodes):
            old_closed = off
        # Freeze an explicit outgoing cut for the unreachable/recovery pair.
        cut = deepcopy(on)
        cut["id"] = cut["name"] = "origin-cut"
        cut["edge"]["edges"] = [list(e) for e in mode_graph.out_edges(nodes[0], keys=True)]
        from spatial.road_conditions import resolve_condition_effects
        _, _, _, details = resolve_condition_effects(mode_graph, [
            {k: v for k, v in cut.items() if k != "bindings"}], "walk", strict=True,
            include_event_details=True)
        cut["bindings"] = {"walk": [list(e) for e in sorted(details[0][1])]}
        body = {"start": coordinate_ref(graph, nodes[0]),
                "end": coordinate_ref(graph, nodes[1]), "travel_mode": "walk",
                "strategy": "shortest", "constraints": {}, "query": ""}
        for variant in VARIANTS:
            old, new, clock = [], [], CLOCK
            request_body, change = deepcopy(body), {"refresh_road_conditions": True}
            entry, edit = "replan", None
            if variant in {"off_path_closure", "on_path_closure"}:
                new = [deepcopy(off if variant.startswith("off") else on)]
            elif variant in {"reopen", "expiry", "revocation"}:
                old = [deepcopy(old_closed)]
                if variant == "expiry":
                    new, clock = deepcopy(old), CLOCK + 101
                elif variant == "revocation":
                    new = [{**deepcopy(old[0]), "revoked_at": CLOCK}]
            elif variant in {"future_event", "expired_event"}:
                new = [deepcopy(on)]
                new[0].update({"start_time": CLOCK + 1, "end_time": CLOCK + 100}
                              if variant == "future_event" else {"end_time": CLOCK})
            elif variant == "other_mode":
                new = [{**deepcopy(on), "affected_modes": ["drive"],
                        "bindings": {"drive": []}}]
            elif variant in {"on_path_slowdown", "off_path_slowdown", "mixed", "notice"}:
                event = deepcopy(off if variant == "off_path_slowdown" else on)
                event.update(action="notice" if variant == "notice" else "slowdown")
                if event["action"] == "slowdown":
                    event["cost_multiplier"] = 1.8
                new = [event] if variant != "mixed" else [deepcopy(off), event]
            elif variant in {"mode_change", "joint_change"}:
                request_body["travel_mode"] = "bike"
                change = {"travel_mode": "bike"}
                if variant == "joint_change":
                    bike_graph = get_routing_index(graph).for_mode("bike").graph
                    bike_nodes = [get_nearest_node(bike_graph, graph.nodes[n]["x"],
                                                  graph.nodes[n]["y"]) for n in nodes]
                    try:
                        _, bike_edges = _path(bike_graph, *bike_nodes)
                        bike_edge = bike_edges[len(bike_edges) // 2]
                    except nx.NetworkXNoPath:
                        bike_edge = sorted(bike_graph.edges(keys=True))[0]
                    new = [_bind_event(graph, bike_edge, "joint-bike", modes=["bike"])]
            elif variant in {"start_change", "end_change", "constraint_change"}:
                entry = "new_task"
                if variant == "constraint_change":
                    request_body["constraints"] = {"avoid_steps": True}
                else:
                    field = "start" if variant == "start_change" else "end"
                    node = path_nodes[max(1, min(len(path_nodes) - 2, len(path_nodes) // 2))]
                    request_body[field] = coordinate_ref(graph, node)
            elif variant == "graph_cost_decrease":
                edge = off["edge"]["edges"][0]
                edit = {"edge": edge, "length": float(graph[edge[0]][edge[1]][edge[2]]["length"]) * 0.5}
            elif variant == "unreachable":
                new = [deepcopy(cut)]
            elif variant == "recovery":
                # The pre-update state is a saved reachable route; then show the
                # temporary failure and restore it as an explicitly ordered run.
                old, new = [deepcopy(cut)], []
                entry = "recovery"
            cases.append({"id": f"{spec['id']}__{variant}", "family": spec["id"],
                          "variant": variant, "initial_body": deepcopy(body),
                          "updated_body": request_body, "change": change, "entry": entry,
                          "old_events": old, "new_events": new,
                          "initial_clock": CLOCK, "updated_clock": clock,
                          "graph_edit": edit, "baseline_edge_ids": path_edges,
                          "source_names": [spec["start"], spec["end"]]})
    return cases
