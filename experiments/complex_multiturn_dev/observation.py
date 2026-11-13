"""Extract observable accepted state without inventing missing evidence."""
from __future__ import annotations
import re


def _id(value, lookup):
    if isinstance(value, str):
        return value if value in lookup["ids"] else lookup["names"].get(value)
    if not isinstance(value, dict):
        return None
    identifier = value.get("poi_id") or value.get("id")
    if identifier in lookup["ids"]:
        return identifier
    if isinstance(identifier, str) and re.fullmatch(r"amap:[A-Za-z0-9_-]{3,80}", identifier):
        return identifier
    name = value.get("name")
    return lookup["names"].get(name) if isinstance(name, str) else None


def _refs(via):
    if not isinstance(via, dict):
        return []
    if via.get("type") == "multi":
        return list(via["points"]) if isinstance(via.get("points"), list) else []
    return [via] if via.get("name") or via.get("poi_id") else []


def project_observation(run, task, lookup, *, selected_first=None):
    result = run.get("result") if isinstance(run.get("result"), dict) else {}
    task = task if isinstance(task, dict) else {}
    state = task.get("state") if isinstance(task.get("state"), dict) else {}
    if run.get("task_id") is None and state.get("place_request"):
        # An unbound informational reply must not replace the accepted task
        # with its own empty/default temporary request.
        request = state["place_request"]
    else:
        request = (result.get("place_request") or result.get("completed_place_request")
                   or state.get("place_request") or {})
    if not isinstance(request, dict):
        request = {}
    sem = request.get("semantic_request") if isinstance(request.get("semantic_request"), dict) else {}
    route = result.get("route_state") or task.get("route_state") or sem.get("route_to_modify") or {}
    if not isinstance(route, dict):
        route = {}
    spec = state.get("route_spec") if isinstance(state.get("route_spec"), dict) else {}
    out = {"status": run.get("status"), "response_kind": result.get("response_kind") or result.get("task_type")}

    origin = _id(request.get("start") or route.get("start") or spec.get("start"), lookup)
    anchor = _id(request.get("search_anchor"), lookup)
    if origin:
        out["origin"] = origin
    if anchor:
        out["search_anchor"] = anchor
    destination = _id(route.get("end") or spec.get("end"), lookup)
    if destination:
        out["destination"] = destination
    mode = request.get("mode") or route.get("travel_mode") or spec.get("travel_mode")
    if mode:
        out["mode"] = mode
    constraints = request.get("constraints")
    if not isinstance(constraints, dict):
        constraints = route.get("hard_constraints") or spec.get("hard_constraints") or {}
    if "avoid_steps" in constraints and type(constraints["avoid_steps"]) is bool:
        out["avoid_steps"] = constraints["avoid_steps"]
    if "max_duration_s" in constraints and (constraints["max_duration_s"] is None
                                            or type(constraints["max_duration_s"]) in (int, float)):
        out["max_duration_s"] = constraints["max_duration_s"]
    if "max_distance_m" in constraints and (constraints["max_distance_m"] is None
                                             or type(constraints["max_distance_m"]) in (int, float)):
        out["route_max_distance_m"] = constraints["max_distance_m"]
    if type(request.get("search_radius_m")) is int:
        out["radius_m"] = request["search_radius_m"]
    if type(sem.get("search_radius_explicit")) is bool:
        out["search_radius_explicit"] = sem["search_radius_explicit"]
    if isinstance(request.get("pending_slot"), str):
        out["clarification_slot"] = request["pending_slot"]
    clarification = result.get("clarify") if isinstance(result.get("clarify"), dict) else {}
    if result.get("response_kind") == "clarify" and isinstance(clarification.get("slot"), str):
        out["clarification_slot"] = clarification["slot"]
    activities = request.get("activities") or sem.get("activities")
    if isinstance(activities, list):
        out["activities"] = activities

    tour = route.get("tour") if isinstance(route.get("tour"), dict) else {}
    raw_stops = tour.get("pois") if isinstance(tour.get("pois"), list) else []
    if not raw_stops:
        raw_stops = _refs(route.get("via"))
        if route.get("end"):
            raw_stops.append(route["end"])
    if not raw_stops and isinstance(sem.get("tour_stops"), list):
        raw_stops = sem["tour_stops"]
    resolved = [_id(item, lookup) for item in raw_stops]
    if raw_stops and all(resolved):
        out["stops"] = [identifier for identifier in resolved if identifier != origin]

    candidates = result.get("candidates")
    if isinstance(candidates, list):
        out["candidate_ids"] = [ident for item in candidates if (ident := _id(item, lookup))]
        out["candidate_count"] = len(candidates)
    if selected_first:
        selected = request.get("selected_poi_id") or _id(route.get("end"), lookup)
        if selected:
            out["selected_matches_first"] = selected == selected_first
    return out
