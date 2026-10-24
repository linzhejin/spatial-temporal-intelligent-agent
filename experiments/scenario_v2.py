"""Versioned, source-aware task gold and an independent graph route audit.

V1 remains available for historical replays. This module never accepts a
product-supplied route review as evidence for the revised experiment.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from experiments.scenario_evaluator import REQUIRED_GOLD, evaluate_scenario


VERSION = "scenario-v2"
GROUPS = ("T1", "T2", "T3", "T4", "T5")
METHODS = {"graph_computed", "official_reference", "manual_independent"}


def _route_reference_for_turn(case: dict[str, Any], index: int) -> dict[str, Any] | None:
    references = case.get("route_references")
    if isinstance(references, dict):
        return references.get(str(index))
    if len(case["expected_turns"]) == 1 and index == 0:
        return case.get("route_reference")
    return None


def validate_dataset(dataset: dict[str, Any], *, formal: bool = False) -> None:
    """Require 120 source-backed cases for a formal T1–T5 comparison."""
    if dataset.get("protocol_version") != VERSION:
        raise ValueError("unsupported protocol_version")
    if dataset.get("split") not in {"development", "test"}:
        raise ValueError("split must be development or test")
    if formal and dataset["split"] != "test":
        raise ValueError("formal scenario-v2 requires the test split")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset requires nonempty cases")
    repeats = dataset.get("repeats", 1)
    if type(repeats) is not int or repeats < 1 or (formal and repeats != 3):
        raise ValueError("formal scenario-v2 requires three repeats")
    digest = dataset.get("graph_sha256")
    if formal and (not isinstance(digest, str) or len(digest) != 64
                   or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("formal scenario-v2 requires graph_sha256")
    if formal and len(cases) != 120:
        raise ValueError("formal scenario-v2 requires 120 cases")
    if formal and any(
            any(turn.get("kind") == "candidates" for turn in case.get("expected_turns", []))
            or ((case.get("adjudication") or {}).get("method") == "graph_computed"
                and any(turn.get("kind") == "route"
                        for turn in case.get("expected_turns", [])))
            for case in cases):
        poi_digest = dataset.get("poi_sha256")
        if (not isinstance(poi_digest, str) or len(poi_digest) != 64
                or any(c not in "0123456789abcdef" for c in poi_digest)):
            raise ValueError("formal candidate gold requires poi_sha256")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case IDs must be unique and nonempty")
        ids.add(case_id)
        if case.get("category") not in GROUPS:
            raise ValueError(f"{case_id}: invalid T1–T5 category")
        for field in ("template_group", "od_group", "snapshot_id"):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ValueError(f"{case_id}: missing {field}")
        if formal and (len(case["snapshot_id"]) != 64
                       or any(c not in "0123456789abcdef" for c in case["snapshot_id"])):
            raise ValueError(f"{case_id}: snapshot must be a frozen SHA-256 digest")
        messages, turns = case.get("messages"), case.get("expected_turns")
        if (not isinstance(messages, list) or not messages or not isinstance(turns, list)
                or len(messages) != len(turns)):
            raise ValueError(f"{case_id}: one expected turn per user message is required")
        if any(not isinstance(msg, dict) or msg.get("role") != "user"
               or not isinstance(msg.get("content"), str) or not msg["content"].strip()
               for msg in messages):
            raise ValueError(f"{case_id}: invalid user message")
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict) or REQUIRED_GOLD - turn.keys():
                raise ValueError(f"{case_id}: incomplete expected turn")
            preserve_from = turn.get("preserve_route_from_turn")
            if "preserve_route_from_turn" in turn and (
                    type(preserve_from) is not int or not 0 <= preserve_from < turn_index
                    or turns[preserve_from].get("kind") != "route"):
                raise ValueError(f"{case_id}: invalid route preservation reference")
            if formal and turn["kind"] == "candidates" and "candidates" not in turn["required_outputs"]:
                raise ValueError(f"{case_id}: candidate output must be scored")
            if (turn["kind"] in {"route", "no_route"} and formal
                    and turn["feasibility"] not in {"feasible", "infeasible"}):
                raise ValueError(f"{case_id}: spatial feasibility must be decided")
        adjudication = case.get("adjudication")
        if not isinstance(adjudication, dict) or adjudication.get("status") not in {"draft", "adjudicated"}:
            raise ValueError(f"{case_id}: adjudication status required")
        if formal and adjudication["status"] != "adjudicated":
            raise ValueError(f"{case_id}: formal gold must be adjudicated")
        if adjudication["status"] == "adjudicated":
            method = adjudication.get("method")
            if method not in METHODS or not isinstance(adjudication.get("evidence"), list) or not adjudication["evidence"]:
                raise ValueError(f"{case_id}: adjudicated gold needs a supported evidence source")
            if method == "manual_independent" and not adjudication.get("reviewer"):
                raise ValueError(f"{case_id}: independent manual reviewer required")
            if method == "official_reference" and not adjudication.get("source_id"):
                raise ValueError(f"{case_id}: official source ID required")
            route_indices = [index for index, turn in enumerate(turns)
                             if turn["kind"] == "route"]
            if method == "graph_computed" and route_indices:
                references = case.get("route_references")
                if (references is not None
                        and (not isinstance(references, dict)
                             or set(references) != {str(i) for i in route_indices})):
                    raise ValueError(f"{case_id}: per-turn route references are stale")
                if len(turns) > 1 and references is None:
                    raise ValueError(f"{case_id}: complete per-turn route references required")
                for index in route_indices:
                    ref = _route_reference_for_turn(case, index)
                    if (not isinstance(ref, dict) or ref.get("start_node") is None
                            or ref.get("end_node") is None
                            or not isinstance(ref.get("via_nodes"), list)
                            or len(ref["via_nodes"]) != len(turns[index]["via_poi_ids"])):
                        raise ValueError(f"{case_id}: per-turn route reference is incomplete")
            if formal and any(t["kind"] == "candidates" for t in turns):
                if method == "graph_computed":
                    spec = case.get("candidate_spec")
                    candidate_turns = [turn for turn in turns if turn["kind"] == "candidates"]
                    if (len(candidate_turns) != 1 or not isinstance(spec, dict)
                            or not isinstance(spec.get("activities"), list)
                            or not spec["activities"]
                            or any(not isinstance(tag, str) or not tag for tag in spec["activities"])
                            or len(set(spec["activities"])) != len(spec["activities"])
                            or spec.get("coordinates") not in {"gcj02", "wgs84"}
                            or type(spec.get("max_display")) is not int
                            or not 3 <= spec["max_display"] <= 4
                            or type(spec.get("rank_tolerance_m")) not in {int, float}
                            or not math.isfinite(spec["rank_tolerance_m"])
                            or not 0 <= spec["rank_tolerance_m"] <= 30
                            or not candidate_turns[0].get("start_poi_id")):
                        raise ValueError(f"{case_id}: reproducible candidate spec is missing")
                elif any(not t["candidate_poi_ids"] for t in turns if t["kind"] == "candidates"):
                    raise ValueError(f"{case_id}: candidate gold IDs are missing")
    if formal:
        counts = Counter(c["category"] for c in cases)
        if any(counts[g] != 24 for g in GROUPS):
            raise ValueError("formal scenario-v2 requires 24 cases per T1–T5 group")
        for group in GROUPS:
            families = Counter(c["template_group"] for c in cases if c["category"] == group)
            if len(families) != 8 or any(count != 3 for count in families.values()):
                raise ValueError(f"{group}: expected eight scenario families with three variants each")


def _edge_tuple(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, str):
        parts = value.split("|")
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        return None
    return tuple(str(p) for p in parts) if len(parts) == 3 else None


def _blocked_modes(value: Any) -> set[str] | None:
    if value is None or value == "":
        return set()
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if isinstance(value, (list, tuple, set)) and all(isinstance(v, str) for v in value):
        return set(value)
    return None


def _edge_lookup(graph) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(str(u), str(v), str(k)): data
            for u, v, k, data in graph.edges(keys=True, data=True)}


def load_frozen_graph(graph_path: Path, overrides_path: Path,
                      reviews_path: Path):
    """Materialize separately frozen route inputs without invoking product routing."""
    import networkx as nx

    graph = nx.read_graphml(graph_path, force_multigraph=True)
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    for label, layer in (("overrides", overrides), ("reviews", reviews)):
        if not isinstance(layer, dict):
            raise ValueError(f"invalid {label} layer")
        if (graph.graph.get("annotation_policy") == "source_scoped_only"
                and layer.get("graph_source_sha256") != graph.graph.get("source_sha256")):
            raise ValueError(f"{label} layer does not match frozen graph source")
    edges = _edge_lookup(graph)
    for item in overrides.get("edges", []):
        identity = _edge_tuple(item.get("edge_id"))
        if identity is None:
            identity = (str(item.get("u")), str(item.get("v")), str(item.get("k", 0)))
        if identity not in edges:
            raise ValueError(f"override edge does not match frozen graph: {identity}")
        if item.get("blocked_modes"):
            modes = _blocked_modes(item["blocked_modes"])
            if modes is None:
                raise ValueError("invalid override blocked_modes")
            edges[identity]["blocked_modes"] = sorted(modes)
    for item in reviews.get("road_decisions", []):
        if item.get("verification_status") not in {"field_verified", "institution_verified"}:
            continue
        if not item.get("evidence") or not item.get("verified_at"):
            continue
        identity = _edge_tuple(item.get("edge_id"))
        if identity not in edges:
            raise ValueError(f"verified road edge does not match frozen graph: {identity}")
        blocked = _blocked_modes(item.get("blocked_modes"))
        passable = _blocked_modes(item.get("passable_modes"))
        if blocked is None or passable is None or blocked & passable:
            raise ValueError("invalid verified road decision")
        old = _blocked_modes(edges[identity].get("blocked_modes"))
        if old is None:
            raise ValueError("invalid graph blocked_modes")
        if "all" in old:
            old = {"walk", "bike", "drive"}
        edges[identity]["blocked_modes"] = sorted((old - passable) | blocked)
    return graph


def verify_route_reference_sources(case: dict[str, Any], graph,
                                   pois: list[dict[str, Any]]) -> int:
    """Reject gold route nodes that disagree with independently snapped POIs."""
    from experiments.route_reference import build_route_reference

    coordinates = case.get("route_coordinates", "gcj02")
    if coordinates not in {"gcj02", "wgs84"}:
        raise ValueError(f"{case['id']}: invalid route coordinates")
    checked = 0
    for index, turn in enumerate(case["expected_turns"]):
        if turn["kind"] != "route":
            continue
        stored = _route_reference_for_turn(case, index)
        if not isinstance(stored, dict):
            raise ValueError(f"{case['id']} turn {index}: per-turn route reference missing")
        source = build_route_reference(
            graph, pois, start_poi_id=turn["start_poi_id"],
            end_poi_id=turn["end_poi_id"], via_poi_ids=turn["via_poi_ids"],
            mode=turn["travel_mode"], constraints=turn["hard_constraints"],
            coordinates=coordinates,
        )
        anchors = ("start_node", "via_nodes", "end_node")
        if any(stored.get(field) != source[field] for field in anchors):
            raise ValueError(f"{case['id']} turn {index}: route reference differs from frozen sources")
        if turn["feasibility"] == "feasible" and not source["graph_feasible"]:
            raise ValueError(f"{case['id']} turn {index}: graph route is not feasible")
        checked += 1
    return checked


def closed_edges_at(graph, conditions: list[dict[str, Any]], mode: str,
                    as_of_epoch: float) -> set[str]:
    """Resolve exact, complete edge bindings at a fixed experiment clock."""
    graph_edges = _edge_lookup(graph)
    closed = set()
    for condition in conditions:
        if condition.get("revoked_at"):
            continue
        start, end = condition.get("start_time") or 0, condition.get("end_time") or 0
        if (start and as_of_epoch < float(start)) or (end and as_of_epoch >= float(end)):
            continue
        if "action" in condition:
            if condition["action"] != "closure" or mode not in condition.get("affected_modes", []):
                continue
        elif mode not in condition.get("blocked_modes", ["walk", "bike", "drive"]):
            continue
        binding = condition.get("edge") or {}
        chain = binding.get("edges") or ([
            [binding["u"], binding["v"], binding.get("key", 0)]
        ] if "u" in binding and "v" in binding else [])
        if not chain:
            raise ValueError("active closure requires exact edge IDs for independent audit")
        for raw in chain:
            identity = _edge_tuple(raw)
            if identity is None:
                raise ValueError("invalid closure edge ID")
            forward = identity in graph_edges
            reverse = (identity[1], identity[0], identity[2]) in graph_edges
            if not forward and not reverse:
                raise ValueError("active closure cannot bind frozen graph")
            if forward:
                closed.add("|".join(identity))
            if reverse:
                closed.add("|".join((identity[1], identity[0], identity[2])))
    return closed


def audit_candidate_order(ids: list[str], reference: dict[str, Any], *,
                          validate_only: bool = False) -> dict[str, Any]:
    """Audit one top-k list against all eligible frozen-graph distances.

    Small distance differences are treated as ties. A flagged, unverified
    terminal connection makes ranking undecidable rather than correct.
    """
    distances = reference.get("distances_m")
    tolerance = reference.get("rank_tolerance_m")
    cap = reference.get("max_display")
    uncertain = reference.get("uncertain_access_ids")
    if (not isinstance(distances, dict) or not distances
            or any(not isinstance(key, str) or type(value) not in {int, float}
                   or not math.isfinite(value) or value < 0
                   for key, value in distances.items())
            or type(tolerance) not in {int, float} or not math.isfinite(tolerance)
            or tolerance < 0 or type(cap) is not int or cap < 1
            or not isinstance(uncertain, list)
            or any(not isinstance(key, str) or key not in distances
                   for key in uncertain)):
        raise ValueError("invalid independent candidate reference")
    if validate_only:
        return {"verdict": "unknown", "reason": "reference_valid"}
    if (not isinstance(ids, list) or not ids or len(ids) > cap
            or any(not isinstance(key, str) for key in ids)
            or len(set(ids)) != len(ids) or any(key not in distances for key in ids)):
        return {"verdict": "fail", "reason": "invalid_or_ineligible_candidates"}
    minimum_count = min(3, cap, len(distances))
    if len(ids) < minimum_count:
        return {"verdict": "fail", "reason": "too_few_candidates"}
    farthest = max(distances[key] for key in ids)
    if (any(key in uncertain for key in ids)
            or any(key in uncertain and value <= farthest + tolerance
                   for key, value in distances.items() if key not in ids)):
        return {"verdict": "unknown", "reason": "last_meter_access_unverified"}
    if any(distances[left] > distances[right] + tolerance
           for left, right in zip(ids, ids[1:])):
        return {"verdict": "fail", "reason": "wrong_distance_order"}
    if any(value + tolerance < farthest for key, value in distances.items() if key not in ids):
        return {"verdict": "fail", "reason": "closer_eligible_omitted"}
    return {"verdict": "pass", "reason": "eligible_and_distance_consistent"}


def _tag_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if isinstance(value, (list, tuple, set)):
        return {str(item).lower() for item in value}
    return {str(value).strip().lower()}


def _access_denied(attrs: dict[str, Any], mode: str, *, node: bool = False) -> bool:
    keys = {"walk": ("foot", "access"), "bike": ("bicycle", "vehicle", "access"),
            "drive": ("motorcar", "motor_vehicle", "vehicle", "access")}[mode]
    explicit = None
    for key in keys:
        values = _tag_values(attrs.get(key))
        if values:
            if values & {"no", "private", "permit", "destination", "customers", "delivery",
                         "agricultural", "forestry"}:
                return True
            explicit = key
            break
    if node:
        barriers = _tag_values(attrs.get("barrier"))
        specific = explicit is not None and explicit != "access"
        if not specific and barriers & {"wall", "fence", "hedge", "retaining_wall"}:
            return True
        if not specific and _tag_values(attrs.get("entrance")) & {"private", "emergency"}:
            return True
        if not specific and mode == "bike" and barriers & {"stile", "turnstile", "kissing_gate"}:
            return True
        if not specific and mode == "drive" and barriers & {
                "bollard", "block", "stile", "turnstile", "kissing_gate", "cycle_barrier"}:
            return True
        return False
    highway = _tag_values(attrs.get("highway"))
    if highway & {"construction", "proposed", "abandoned", "razed"}:
        return True
    if mode == "walk":
        return bool((highway & {"motorway", "motorway_link"}
                     or _tag_values(attrs.get("indoor")) & {"yes", "room"})
                    and explicit != "foot")
    if mode == "bike":
        return bool(highway & {"steps", "elevator", "escalator"}
                    or ("corridor" in highway and explicit != "bicycle")
                    or (highway & {"motorway", "motorway_link"} and explicit != "bicycle"))
    motor_highway = {"motorway", "motorway_link", "trunk", "trunk_link", "primary",
                     "primary_link", "secondary", "secondary_link", "tertiary",
                     "tertiary_link", "unclassified", "residential", "service",
                     "living_street", "road", "busway"}
    return bool(highway & {"steps", "elevator", "escalator", "corridor", "platform"}
                or (highway and not highway & motor_highway
                    and explicit not in {"motorcar", "motor_vehicle", "vehicle"}))


def audit_route_edges(graph, edge_ids: list[Any], start_node: Any, end_node: Any,
                      via_nodes: list[Any], mode: str, constraints: dict[str, Any],
                      *, closed_edges: set[str] | None = None) -> dict[str, Any]:
    """Check exact directed edges against a frozen graph, separately from routing code."""
    issues = []
    if mode not in {"walk", "bike", "drive"}:
        issues.append("unsupported_mode")
    if not isinstance(edge_ids, list) or not edge_ids:
        issues.append("empty_edge_sequence")
        edge_ids = []
    parsed = [_edge_tuple(raw) for raw in edge_ids]
    if any(edge is None for edge in parsed):
        issues.append("invalid_edge_id")
    edges = [edge for edge in parsed if edge is not None]
    if edges:
        if edges[0][0] != str(start_node) or edges[-1][1] != str(end_node):
            issues.append("endpoint_mismatch")
        if any(left[1] != right[0] for left, right in zip(edges, edges[1:])):
            issues.append("discontinuous_edges")
    visited = [edges[0][0]] + [edge[1] for edge in edges] if edges else []
    position = 0
    for node in via_nodes:
        try:
            position = visited.index(str(node), position) + 1
        except ValueError:
            issues.append("missing_ordered_via")
            break
    closed = {e for raw in (closed_edges or set()) if (e := _edge_tuple(raw)) is not None}
    nodes = {str(node): node for node in graph.nodes}
    for u, v, key in edges:
        if u not in nodes or v not in nodes:
            issues.append("missing_edge")
            continue
        attrs_by_key = graph.get_edge_data(nodes[u], nodes[v]) or {}
        attrs = next((d for k, d in attrs_by_key.items() if str(k) == key), None)
        if attrs is None:
            issues.append("missing_edge")
            continue
        if (u, v, key) in closed:
            issues.append("closed_edge_used")
        if mode in {"walk", "bike", "drive"}:
            if (_access_denied(graph.nodes[nodes[u]], mode, node=True)
                    or _access_denied(graph.nodes[nodes[v]], mode, node=True)
                    or _access_denied(attrs, mode)):
                issues.append("access_denied")
            if attrs.get("allowed_modes") is not None:
                allowed = _blocked_modes(attrs["allowed_modes"])
                if allowed is None:
                    issues.append("edge_access_unknown")
                elif mode not in allowed:
                    issues.append("mode_not_allowed")
        blocked = _blocked_modes(attrs.get("blocked_modes"))
        if blocked is None:
            issues.append("edge_access_unknown")
        elif mode in blocked or "all" in blocked:
            issues.append("blocked_mode_used")
        if constraints.get("avoid_steps") and "steps" in str(attrs.get("highway", "")):
            issues.append("steps_used")
    supported = {"avoid_steps"}
    if any(key not in supported and value for key, value in constraints.items()):
        issues.append("constraint_not_independently_checked")
    unknown = {"edge_access_unknown", "constraint_not_independently_checked"}
    decisive = set(issues) - unknown
    return {"route": "fail" if decisive else "unknown" if issues else "pass",
            "issues": sorted(set(issues)),
            "hard_constraints": {"avoid_steps": "fail" if "steps_used" in issues else "pass"}
            if constraints.get("avoid_steps") else {}}


def score_case(case: dict[str, Any], observed: dict[str, Any] | None, *, graph=None,
               candidate_reference: dict[str, Any] | None = None,
               closed_edges: set[str] | None = None,
               conditions: list[dict[str, Any]] | None = None,
               as_of_epoch: float | None = None,
               independent_reviews: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Evaluate a task after discarding every review embedded in model output."""
    sanitized = deepcopy(observed)
    turns = sanitized.get("turns") if isinstance(sanitized, dict) else None
    if isinstance(turns, list):
        for index, actual in enumerate(turns):
            if not isinstance(actual, dict):
                continue
            actual.pop("independent_review", None)
            method = (case.get("adjudication") or {}).get("method")
            expected = case["expected_turns"][index] if index < len(case["expected_turns"]) else {}
            ref = _route_reference_for_turn(case, index) if expected.get("kind") == "route" else None
            if (method == "graph_computed" and graph is not None
                    and isinstance(ref, dict) and ref.get("start_node") is not None
                    and ref.get("end_node") is not None
                    and isinstance(ref.get("via_nodes"), list)):
                mode = expected.get("travel_mode")
                timed_closures = closed_edges
                if conditions is not None:
                    clock = case.get("as_of_epoch", as_of_epoch)
                    if clock is None:
                        raise ValueError("road condition audit requires a fixed clock")
                    timed_closures = closed_edges_at(graph, conditions, mode, float(clock))
                audit = audit_route_edges(
                    graph, actual.get("route_edge_ids"), ref["start_node"], ref["end_node"],
                    ref["via_nodes"], mode, expected.get("hard_constraints") or {},
                    closed_edges=timed_closures,
                )
                actual["independent_review"] = {
                    "source": "independent_program", "reviewer": "scenario_v2.route_audit",
                    "evidence": list(case["adjudication"]["evidence"]),
                    "route": audit["route"], "hard_constraints": audit["hard_constraints"],
                    "outputs": {},
                }
            elif method == "graph_computed" and expected.get("kind") == "candidates":
                ref = candidate_reference
                if isinstance(ref, dict):
                    audit = audit_candidate_order(actual.get("candidate_poi_ids"), ref)
                    actual["independent_review"] = {
                        "source": "independent_program",
                        "reviewer": "scenario_v2.candidate_audit",
                        "evidence": list(case["adjudication"]["evidence"]),
                        "candidate_order": audit,
                        "outputs": {"candidates": audit["verdict"]},
                    }
            elif (independent_reviews and index < len(independent_reviews)
                  and (method != "graph_computed"
                       or expected.get("kind") not in {"route", "candidates"})):
                actual["independent_review"] = deepcopy(independent_reviews[index])
    return evaluate_scenario(case, sanitized, _trusted_program_review=True)
