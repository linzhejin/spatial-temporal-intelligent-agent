"""Validated server route state for deterministic, non-LLM replanning."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4


SCHEMA_VERSION = 1
ROUTE_KINDS = {"direct", "via", "itinerary", "tour", "multimodal"}
TRAVEL_MODES = {"walk", "bike", "drive"}
STRATEGIES = {"shortest", "recommended", "scenery", "flat", "custom"}
STATE_FIELDS = {
    "schema_version",
    "route_id",
    "route_kind",
    "original_query",
    "start",
    "end",
    "via",
    "itinerary",
    "tour",
    "legs",
    "travel_mode",
    "hard_constraints",
    "strategy",
    "data_version",
    "road_condition_version",
    "reuse_certificate",
}
STRATEGY_FIELDS = {"name", "source", "task_class", "weights", "detour_cap"}


class RouteStateVersionConflict(ValueError):
    """Raised when route data changed after the state was issued."""


def current_data_version(graph=None) -> str:
    """Hash the effective graph and local source bytes used by replanning."""
    if graph is None:
        from spatial.network import get_network
        graph = get_network()
    digest = hashlib.sha256()

    def add(value) -> None:
        digest.update(json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")

    if graph is not None:
        add(("graph", graph.graph))
        for node, attrs in sorted(graph.nodes(data=True), key=lambda row: repr(row[0])):
            add(("node", repr(node), attrs))
        for u, v, key, attrs in sorted(
            graph.edges(keys=True, data=True),
            key=lambda row: (repr(row[0]), repr(row[1]), repr(row[2])),
        ):
            add(("edge", repr(u), repr(v), repr(key), attrs))
    else:
        add(("graph", None))
    import config

    for filename in (
        "whu_road_network.graphml", "pois.json", "road_annotations.json",
        "edge_overrides.json", "edge_attribute_master.json",
        "campus_review_decisions.json",
    ):
        path = Path(config.DATA_DIR) / filename
        add(("file", filename, path.is_file()))
        if path.is_file():
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\n")
    return "data-" + digest.hexdigest()[:16]


def current_road_condition_version(conditions=None) -> str:
    """Version either the caller's frozen event snapshot or current active events."""
    try:
        if conditions is None:
            from spatial.road_conditions import list_conditions
            conditions = list_conditions()
        payload = json.dumps(conditions, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        payload = "unavailable"
    return "roads-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _route_id() -> str:
    return f"route-{uuid4().hex}"


def _sanitize_strategy(raw) -> dict:
    if not isinstance(raw, dict) or raw.get("name") not in STRATEGIES:
        raise ValueError("invalid strategy")
    strategy = {
        key: deepcopy(raw[key]) for key in STRATEGY_FIELDS if key in raw
    }
    if "source" not in strategy:
        strategy["source"] = "commute_default"
    return strategy


def build_route_state(**values) -> dict:
    """Build a complete state and then pass it through the same validator."""
    state = {
        "schema_version": SCHEMA_VERSION,
        "route_id": values.pop("route_id", _route_id()),
        "via": None,
        "tour": None,
        "legs": [],
    }
    state.update(values)
    return validate_route_state(state)


def validate_route_state(raw) -> dict:
    """Return a deep-copied whitelist of route semantics from client input."""
    if not isinstance(raw, dict):
        raise ValueError("route_state must be an object")
    state = {key: deepcopy(raw.get(key)) for key in STATE_FIELDS if key in raw}
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported route_state schema_version")
    if state.get("route_kind") not in ROUTE_KINDS:
        raise ValueError("invalid route_kind")
    if state.get("travel_mode") not in TRAVEL_MODES:
        raise ValueError("invalid travel_mode")
    if not isinstance(state.get("start"), dict) or not isinstance(state.get("end"), dict):
        raise ValueError("route endpoints are required")
    if not isinstance(state.get("route_id"), str) or not state["route_id"]:
        raise ValueError("route_id is required")
    if not isinstance(state.get("original_query"), str):
        raise ValueError("original_query must be a string")
    if not isinstance(state.get("hard_constraints"), dict):
        raise ValueError("hard_constraints must be an object")
    if not isinstance(state.get("legs"), list):
        raise ValueError("legs must be a list")
    if ("reuse_certificate" in state
            and not isinstance(state["reuse_certificate"], dict)):
        raise ValueError("reuse_certificate must be an object")
    if state["route_kind"] in {"via", "itinerary"}:
        via = state.get("via")
        if not isinstance(via, dict):
            raise ValueError(f"{state['route_kind']} route requires a via point")
        if via.get("type") == "multi":
            points = via.get("points")
            if not isinstance(points, list) or not 1 <= len(points) <= 10:
                raise ValueError("multi-via route requires 1 to 10 ordered points")
            if any(not isinstance(point, dict) for point in points):
                raise ValueError("multi-via points must be objects")
    if state["route_kind"] == "itinerary":
        itinerary = state.get("itinerary")
        if not isinstance(itinerary, dict):
            raise ValueError("itinerary route requires budget settings")
        raw_budget = itinerary.get("time_budget_min")
        raw_dwell = itinerary.get("stop_duration_min")
        if isinstance(raw_budget, bool) or isinstance(raw_dwell, bool):
            raise ValueError("itinerary budget settings are invalid")
        try:
            budget = float(raw_budget)
            dwell = float(raw_dwell)
        except (TypeError, ValueError):
            raise ValueError("itinerary budget settings are invalid") from None
        if (not math.isfinite(budget) or not 0 < budget <= 1440
                or not math.isfinite(dwell) or not 0 <= dwell <= 240):
            raise ValueError("itinerary budget settings are invalid")
        state["itinerary"] = {
            "time_budget_min": budget,
            "stop_duration_min": dwell,
        }
    if state["route_kind"] == "tour" and not isinstance(state.get("tour"), dict):
        raise ValueError("tour route requires tour settings")
    if state["route_kind"] == "multimodal" and not state["legs"]:
        raise ValueError("multimodal route requires legs")
    state["strategy"] = _sanitize_strategy(state.get("strategy"))
    return state


def apply_change(state, change) -> dict:
    """Apply exactly one user-visible change while preserving route semantics."""
    if not isinstance(change, dict) or set(change) not in (
        {"travel_mode"},
        {"strategy"},
        {"refresh_road_conditions"},
    ):
        raise ValueError("change must contain exactly one supported field")
    out = deepcopy(validate_route_state(state))
    if "refresh_road_conditions" in change:
        if change["refresh_road_conditions"] is not True:
            raise ValueError("refresh_road_conditions must be true")
    elif "travel_mode" in change:
        if change["travel_mode"] not in TRAVEL_MODES:
            raise ValueError("invalid travel_mode")
        out["travel_mode"] = change["travel_mode"]
        if out["route_kind"] == "multimodal":
            # The global mode button changes every travel leg while preserving
            # their ordered stops. Otherwise the top-level mode lies about the
            # route that is actually recomputed.
            out["legs"] = [
                {**leg, "travel_mode": change["travel_mode"]}
                for leg in out["legs"]
            ]
    else:
        if change["strategy"] not in STRATEGIES:
            raise ValueError("invalid strategy")
        out["strategy"] = {"name": change["strategy"], "source": "button"}
    out["route_id"] = _route_id()
    return out


def route_state_to_context(raw) -> dict:
    """Project a validated route state into the legacy parser context shape."""
    state = validate_route_state(raw)
    strategy = state["strategy"]
    previous_intent = {
        "task_type": "path_planning",
        "start": deepcopy(state["start"]),
        "end": deepcopy(state["end"]),
        "constraints": deepcopy(state["hard_constraints"]),
        "weights": deepcopy(strategy.get("weights")),
        "strategy_hint": strategy["name"],
        "mode": state["travel_mode"],
        "ambiguity": None,
    }
    return {
        "start": deepcopy(state["start"]),
        "end": deepcopy(state["end"]),
        "constraints": deepcopy(state["hard_constraints"]),
        "weights": deepcopy(strategy.get("weights")),
        "previous_intent": previous_intent,
        "previous_route_state": state,
    }
