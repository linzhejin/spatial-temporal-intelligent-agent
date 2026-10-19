"""Shared validation for exact-edge exclusions and route-level budgets."""

import json
import math
from collections.abc import Mapping


def excluded_edge_ids(constraints) -> set[tuple]:
    """Decode directed ``(u, v, key)`` IDs without guessing identifier types.

    Durable task state stores each ID as a compact JSON array string. Internal
    callers may pass the three-element array directly. Reverse directions are
    separate edges; excluding one does not silently exclude its reverse.
    """
    if not isinstance(constraints, Mapping):
        raise ValueError("constraints must be an object")
    raw_ids = constraints.get("excluded_edge_ids")
    if raw_ids is None:
        return set()
    if not isinstance(raw_ids, list):
        raise ValueError("excluded_edge_ids must be a list")
    selected = set()
    for raw in raw_ids:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("excluded_edge_ids must contain JSON edge triples") from exc
        if (not isinstance(raw, (list, tuple)) or len(raw) != 3
                or any(isinstance(part, bool) or not isinstance(part, (int, str))
                       for part in raw)):
            raise ValueError("excluded_edge_ids must contain exact (u,v,key) triples")
        selected.add(tuple(raw))
    return selected


def positive_budget(constraints, key: str) -> float | None:
    """Return a finite positive hard limit or reject malformed values."""
    if not isinstance(constraints, Mapping):
        raise ValueError("constraints must be an object")
    if constraints.get(key) is None:
        return None
    raw = constraints[key]
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a finite positive number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite positive number")
    return value
