"""Build an evidence-bounded audit of development scenario drafts.

The audit checks catalog identity and base-graph topology only. It does not
promote draft expectations into research gold or infer field accessibility.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata
from typing import Any

import networkx as nx

from experiments.scenario_evaluator import validate_dataset
from spatial.coord_transform import gcj02_to_wgs84


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "experiments" / "datasets" / "scenarios_dev_v1.json"
DEFAULT_POIS = ROOT / "data" / "pois.json"
DEFAULT_GRAPH = ROOT / "data" / "whu_road_network.graphml"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "experiments"
_OUTPUT_NAME = re.compile(r"scenario_draft_audit_[A-Za-z0-9_.-]+\.json\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_CUE = re.compile(r"路况|管制|封路|封闭|缓行|事件快照|道路事件")
_NUMBER = r"(\d+(?:\.\d+)?|两|二|一)"
_DISTANCE_LIMIT = re.compile(
    rf"(?:距离(?:上限|限制)?(?:收紧)?(?:为)?|总距离(?:上限|限制)?(?:收紧)?(?:为)?|"
    rf"上限(?:收紧)?(?:为)?|限制(?:收紧)?(?:为)?|不超过|不能超过|不得超过|不超){_NUMBER}"
    r"(公里|千米|km|米|m)"
)
_DURATION_LIMIT = re.compile(
    rf"(?:不超过|不能超过|不得超过|不超|上限(?:收紧)?(?:为)?|"
    rf"限制(?:收紧)?(?:为)?|在){_NUMBER}(分钟|分|小时|h|min)(?:以内|内)?"
)


def _normalize(value: Any) -> str:
    text = str(value or "").casefold()
    return "".join(char for char in text
                   if not char.isspace() and not unicodedata.category(char).startswith("P"))


def _poi_index(pois: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    by_id: dict[str, dict[str, Any]] = {}
    alias_ids: dict[str, set[str]] = defaultdict(set)
    for poi in pois:
        poi_id = poi.get("id")
        if not isinstance(poi_id, str) or not poi_id or poi_id in by_id:
            raise ValueError("POI catalog IDs must be unique nonempty strings")
        by_id[poi_id] = poi
        labels = [poi.get("name"), *(poi.get("aliases") or [])]
        for label in labels:
            alias = _normalize(label)
            if len(alias) >= 2:
                alias_ids[alias].add(poi_id)
    return by_id, alias_ids


def _message_mentions(messages: list[dict[str, Any]], alias_ids: dict[str, set[str]]) -> dict[str, Any]:
    text = _normalize("\n".join(message["content"] for message in messages))
    matches: list[dict[str, Any]] = []
    ambiguous: list[str] = []
    occupied_spans: list[tuple[int, int]] = []
    # Longer labels are more discriminative. Keep generic colliding aliases
    # visible, but do not treat them as a stable identity.
    for alias in sorted(alias_ids, key=lambda item: (-len(item), item)):
        owners = sorted(alias_ids[alias])
        start = 0
        while (position := text.find(alias, start)) >= 0:
            end = position + len(alias)
            start = position + 1
            if any(position < span_end and span_start < end
                   for span_start, span_end in occupied_spans):
                continue
            occupied_spans.append((position, end))
            if len(owners) == 1:
                matches.append({"alias": alias, "poi_id": owners[0],
                                "position": position})
            else:
                ambiguous.append(alias)
    matches.sort(key=lambda item: item["position"])
    return {
        "unique_matches": matches,
        "unique_poi_ids": sorted({item["poi_id"] for item in matches}),
        "ordered_unique_poi_ids": [item["poi_id"] for item in matches],
        "ambiguous_aliases": ambiguous,
    }


def _distance_m(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    radius_m = 6_371_008.8
    phi_a, phi_b = radians(lat_a), radians(lat_b)
    d_phi = phi_b - phi_a
    d_lon = radians(lon_b - lon_a)
    value = sin(d_phi / 2) ** 2 + cos(phi_a) * cos(phi_b) * sin(d_lon / 2) ** 2
    return 2 * radius_m * asin(sqrt(min(1.0, value)))


def _poi_wgs84(poi: dict[str, Any], default_coordinate_system: str) -> tuple[float, float] | None:
    coords = poi.get("coordinates") or {}
    try:
        lng, lat = float(coords["lng"]), float(coords["lat"])
    except (KeyError, TypeError, ValueError):
        return None
    system = str(poi.get("coordinate_system") or default_coordinate_system).upper().replace("_", "-")
    if system in {"GCJ02", "GCJ-02"}:
        return gcj02_to_wgs84(lng, lat)
    if system in {"WGS84", "WGS-84"}:
        return lng, lat
    return None


def _nearest_graph_node(graph: nx.MultiDiGraph, poi: dict[str, Any], default_coordinate_system: str):
    point = _poi_wgs84(poi, default_coordinate_system)
    if point is None or graph.number_of_nodes() == 0:
        return None
    lng, lat = point
    best_node, best_distance = None, float("inf")
    for node_id, attrs in graph.nodes(data=True):
        try:
            node_lng, node_lat = float(attrs["x"]), float(attrs["y"])
        except (KeyError, TypeError, ValueError):
            continue
        distance = _distance_m(lng, lat, node_lng, node_lat)
        if distance < best_distance:
            best_node, best_distance = node_id, distance
    if best_node is None:
        return None
    return {"node_id": str(best_node), "distance_m": round(best_distance, 2)}


def _entity_check(turn: dict[str, Any], current_mentions: dict[str, Any],
                  history_mentions: dict[str, Any]) -> dict[str, Any]:
    expected = {poi_id for poi_id in [turn.get("start_poi_id"), turn.get("end_poi_id"),
                                      *(turn.get("via_poi_ids") or [])]
                if poi_id is not None}
    current_ids = set(current_mentions["unique_poi_ids"])
    history_ids = set(history_mentions["unique_poi_ids"])
    missing = sorted(expected - history_ids)
    unexpected = sorted(current_ids - expected) if turn.get("kind") in {"route", "no_route", "candidates"} else []
    ambiguous = sorted(set(current_mentions["ambiguous_aliases"])
                       | set(history_mentions["ambiguous_aliases"]))
    expected_order = ([turn.get("start_poi_id"), *(turn.get("via_poi_ids") or []),
                       turn.get("end_poi_id")] if turn.get("kind") in {"route", "no_route"} else [])
    expected_order = [poi_id for poi_id in expected_order if poi_id is not None]
    current_order = [poi_id for poi_id in current_mentions["ordered_unique_poi_ids"]
                     if poi_id in set(expected_order)]
    if not expected_order:
        order_status = "not_applicable"
    elif all(current_order.count(poi_id) >= expected_order.count(poi_id)
             for poi_id in set(expected_order)):
        order_status = ("literal_reference_order_matches_expected" if current_order == expected_order
                        else "literal_reference_order_differs_semantics_unresolved")
    else:
        order_status = "not_assessable_from_current_message"
    if missing and unexpected:
        status = "possible_role_mismatch_needs_review"
    elif ambiguous or missing:
        status = "manual_review_required"
    else:
        status = "explicit_catalog_names_match" if expected else "no_named_place_to_check"
    return {
        "status": status,
        "expected_route_poi_ids": sorted(expected),
        "explicit_unique_poi_ids": sorted(current_ids),
        "prior_context_poi_ids": sorted(history_ids - current_ids),
        "expected_ids_not_explicitly_mentioned": missing,
        "other_explicit_poi_ids_in_current_message": unexpected,
        "matched_aliases": [item for item in history_mentions["unique_matches"]
                            if item["poi_id"] in expected],
        "ambiguous_aliases": ambiguous,
        "route_order_check": {
            "status": order_status,
            "expected_poi_id_order": expected_order,
            "current_message_poi_id_order_filtered_to_expected": current_order,
            "scope": "literal_name_occurrence_order_is_not_route_order_evidence",
        },
        "scope": "literal_catalog_name_or_alias_match_only",
    }


def _number_value(raw: str) -> float:
    chinese_numbers = {"一": 1.0, "二": 2.0, "两": 2.0}
    return chinese_numbers[raw] if raw in chinese_numbers else float(raw)


def _extract_text_constraints(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract only explicit, narrow constraint wording; this is not semantic gold."""
    detected: dict[str, Any] = {}
    evidence: dict[str, int] = {}
    avoid_steps = None
    for index, message in enumerate(messages, 1):
        text = _normalize(message["content"])
        cancels_steps = bool(re.search(
            r"(?<!不)允许(?:走|使用)台阶|(?:台阶).{0,6}(?:取消|不再限制)|"
            r"(?:取消|去掉).{0,8}(?:不走|避开|台阶)|不再避开台阶|不用避开台阶",
            text,
        ))
        forbids_steps = bool(re.search(
            r"(?:不要走|不允许走|别走|不走|避开|避免|绕开|不经过)"
            r"(?:已标记的?|已标注的?)?台阶", text,
        ))
        if cancels_steps:
            avoid_steps = False
            evidence["avoid_steps"] = index
        elif forbids_steps:
            avoid_steps = True
            evidence["avoid_steps"] = index

        for match in _DISTANCE_LIMIT.finditer(text):
            raw_value, unit = match.groups()
            value = _number_value(raw_value)
            if unit in {"公里", "千米", "km"}:
                value *= 1000
            detected["max_distance_m"] = int(value) if value.is_integer() else value
            evidence["max_distance_m"] = index
        for match in _DURATION_LIMIT.finditer(text):
            raw_value, unit = match.groups()
            value = _number_value(raw_value)
            if unit in {"小时", "h"}:
                value *= 3600
            else:
                value *= 60
            detected["max_duration_s"] = int(value) if value.is_integer() else value
            evidence["max_duration_s"] = index

        if ("無障碍" in text or "无障碍" in text or "轮椅" in text) and re.search(
                r"必须|要求|核实|证据|輪椅|轮椅", text):
            detected["require_verified_accessibility"] = True
            evidence["require_verified_accessibility"] = index
        closure_mentioned = "封闭" in text or "封路" in text
        closure_expired = bool(re.search(r"已到期|已经到期|已失效|已经失效|过期", text))
        closure_scope = bool(re.search(r"只对.{0,8}(?:封闭|封路)|适用范围.{0,8}(?:封闭|封路)", text))
        closure_avoidance = bool(re.search(
            r"避开|不能.{0,12}穿越|不允许.{0,12}穿越|不准.{0,12}穿越|不得.{0,12}穿越|绕开",
            text,
        ))
        if closure_mentioned and not closure_expired and (closure_scope or closure_avoidance):
            detected["avoid_closed_edges"] = True
            evidence["avoid_closed_edges"] = index

    if avoid_steps is not None:
        detected["avoid_steps"] = avoid_steps
    return {"detected_constraints": detected, "evidence_message_indices": evidence}


def _constraint_check(turn: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = _extract_text_constraints(messages)
    detected = parsed["detected_constraints"]
    expected = turn.get("hard_constraints") or {}
    compared_keys = set(expected) | set(detected)
    comparisons = {}
    mismatches = []
    for key in sorted(compared_keys):
        expected_value = expected.get(key, False if key in {"avoid_steps", "avoid_closed_edges"} else None)
        detected_value = detected.get(key)
        if detected_value is None:
            status = "not_found_in_literal_text"
        elif detected_value != expected_value:
            status = "possible_mismatch"
            mismatches.append(key)
        else:
            status = "textually_supported"
        comparisons[key] = {"expected": expected_value, "detected": detected_value,
                            "status": status,
                            "evidence_message_index": parsed["evidence_message_indices"].get(key)}
    if mismatches:
        status = "possible_mismatch"
    elif comparisons and all(item["status"] == "textually_supported" for item in comparisons.values()):
        status = "textually_supported"
    elif not expected and not detected:
        status = "no_explicit_constraint_detected"
    else:
        status = "not_enough_literal_evidence"
    return {"status": status, "comparisons": comparisons,
            "detected_constraints": detected,
            "evidence_message_indices": parsed["evidence_message_indices"],
            "scope": "literal_constraint_cues_only_not_route_compliance_evidence"}


def _graph_check(turn: dict[str, Any], graph: nx.MultiDiGraph | None,
                 poi_by_id: dict[str, dict[str, Any]], default_coordinate_system: str) -> dict[str, Any]:
    result = {
        "scope": "base_mode_graph_only",
        "status": "not_applicable",
        "travel_mode": turn.get("travel_mode"),
        "legs": [],
        "limitations": [
            "does_not_apply_route_hard_constraints_or_active_road_events",
            "nearest_node_snap_is_diagnostic_and_not_a_verified_entrance_link",
            "graph_connectivity_does_not_prove_field_accessibility_or_legal_access",
        ],
    }
    if turn.get("kind") not in {"route", "no_route"}:
        return result
    if graph is None:
        result["status"] = "mode_graph_unavailable"
        return result
    poi_ids = [turn.get("start_poi_id"), *(turn.get("via_poi_ids") or []),
               turn.get("end_poi_id")]
    if len(poi_ids) < 2 or any(poi_id not in poi_by_id for poi_id in poi_ids):
        result["status"] = "endpoint_reference_missing"
        return result
    snaps = {poi_id: _nearest_graph_node(graph, poi_by_id[poi_id], default_coordinate_system)
             for poi_id in set(poi_ids)}
    for source_id, target_id in zip(poi_ids, poi_ids[1:]):
        source_snap, target_snap = snaps[source_id], snaps[target_id]
        leg = {"from_poi_id": source_id, "to_poi_id": target_id,
               "from_snap_m": source_snap["distance_m"] if source_snap else None,
               "to_snap_m": target_snap["distance_m"] if target_snap else None,
               "connected": None, "base_graph_distance_m": None}
        if source_snap is not None and target_snap is not None:
            try:
                leg["connected"] = bool(nx.has_path(
                    graph, source_snap["node_id"], target_snap["node_id"]))
                if leg["connected"]:
                    distance = nx.shortest_path_length(
                        graph, source_snap["node_id"], target_snap["node_id"], weight="length")
                    leg["base_graph_distance_m"] = round(float(distance), 2)
            except (nx.NetworkXError, TypeError, ValueError):
                leg["connected"] = None
        result["legs"].append(leg)
    connectivity = [leg["connected"] for leg in result["legs"]]
    if any(value is False for value in connectivity):
        result["status"] = "not_connected_in_base_mode_graph"
    elif connectivity and all(value is True for value in connectivity):
        result["status"] = "connected_in_base_mode_graph"
    else:
        result["status"] = "connectivity_unknown"
    return result


def _case_review_gaps(case: dict[str, Any]) -> list[dict[str, str]]:
    gaps = []
    adjudication = case.get("adjudication") or {}
    if (adjudication.get("status") != "adjudicated"
            or adjudication.get("method") != "manual_independent"
            or not adjudication.get("reviewer")
            or not adjudication.get("evidence")):
        gaps.append({"code": "gold_not_independently_adjudicated", "scope": "case"})
    if not isinstance(case.get("snapshot_id"), str) or not _SHA256.fullmatch(case["snapshot_id"]):
        gaps.append({"code": "snapshot_not_frozen", "scope": "case"})
    messages = case.get("messages") or []
    has_event_cue = any(
        isinstance(message.get("content"), str)
        and _EVENT_CUE.search(message["content"])
        for message in messages
    )
    event_snapshot_id = case.get("event_snapshot_id")
    if has_event_cue and not (
        isinstance(event_snapshot_id, str) and _SHA256.fullmatch(event_snapshot_id)
    ):
        gaps.append({"code": "event_snapshot_not_defined", "scope": "case"})
    elif case.get("category") == "event_infeasible" and not has_event_cue:
        gaps.append({"code": "event_category_without_event_cue", "scope": "case"})
    return gaps


def _turn_review_gaps(turn: dict[str, Any]) -> list[dict[str, str]]:
    gaps = []
    if not turn.get("intent_label"):
        gaps.append({"code": "intent_label_missing", "scope": "turn"})
    if not turn.get("expected_tools"):
        gaps.append({"code": "tool_use_expectation_missing", "scope": "turn"})
    if turn.get("kind") in {"route", "no_route"}:
        if turn.get("feasibility") == "unknown":
            gaps.append({"code": "route_feasibility_undetermined", "scope": "turn"})
        if not turn.get("feasibility_evidence"):
            gaps.append({"code": "route_feasibility_evidence_missing", "scope": "turn"})
    if turn.get("kind") == "candidates" and not turn.get("candidate_poi_ids"):
        gaps.append({"code": "candidate_set_unadjudicated", "scope": "turn"})
    return gaps


def audit_scenario_drafts(dataset: dict[str, Any], pois: list[dict[str, Any]],
                          mode_graphs: dict[str, nx.MultiDiGraph], *,
                          poi_coordinate_system: str = "GCJ-02") -> dict[str, Any]:
    """Check literal POI identity and graph-only connectivity without scoring gold."""
    validate_dataset(dataset)
    poi_by_id, alias_ids = _poi_index(pois)
    cases = []
    status_counts: Counter[str] = Counter()
    review_gap_counts: Counter[str] = Counter()
    for case in dataset["cases"]:
        case_turns = []
        history: list[dict[str, Any]] = []
        case_gaps = _case_review_gaps(case)
        review_gap_counts.update(gap["code"] for gap in case_gaps)
        for index, (message, expected) in enumerate(zip(case["messages"], case["expected_turns"]), 1):
            history.append(message)
            current_mentions = _message_mentions([message], alias_ids)
            history_mentions = _message_mentions(history, alias_ids)
            mode = expected.get("travel_mode")
            entity = _entity_check(expected, current_mentions, history_mentions)
            constraint = _constraint_check(expected, history)
            graph = mode_graphs.get(mode) if isinstance(mode, str) else None
            graph_result = _graph_check(expected, graph, poi_by_id, poi_coordinate_system)
            review_gaps = _turn_review_gaps(expected)
            review_gap_counts.update(gap["code"] for gap in review_gaps)
            item = {
                "turn": index,
                "message": message["content"],
                "context_messages": [entry["content"] for entry in history],
                "provisional_expectation": {
                    "kind": expected["kind"],
                    "status": expected["status"],
                    "feasibility": expected["feasibility"],
                },
                "draft_expectation": {
                    field: expected.get(field)
                    for field in (
                        "start_poi_id", "end_poi_id", "via_poi_ids", "travel_mode",
                        "strategy", "hard_constraints", "required_outputs",
                        "candidate_poi_ids", "clarification_slot", "allowed_unknowns",
                    )
                },
                "entity_check": entity,
                "constraint_check": constraint,
                "graph_check": graph_result,
                "candidate_gold": "unadjudicated",
                "review_gaps": review_gaps,
                "research_gold_status": "still_unadjudicated",
            }
            status_counts[entity["status"]] += 1
            status_counts[f"route_order:{entity['route_order_check']['status']}"] += 1
            status_counts[f"constraint:{constraint['status']}"] += 1
            status_counts[f"graph:{graph_result['status']}"] += 1
            case_turns.append(item)
        cases.append({"scenario_id": case["id"], "category": case["category"],
                      "adjudication_status": (case.get("adjudication") or {}).get("status"),
                      "review_gaps": case_gaps,
                      "turns": case_turns})
    return {
        "audit_version": "scenario-draft-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dataset_version": dataset.get("version"),
        "split": dataset.get("split"),
        "summary": {
            "scenario_count": len(cases),
            "turn_count": sum(len(case["turns"]) for case in cases),
            "research_gold_adjudicated": False,
            "status_counts": dict(sorted(status_counts.items())),
            "review_gap_counts": dict(sorted(review_gap_counts.items())),
        },
        "method": {
            "catalog_check": "unique literal current POI canonical names and aliases",
            "graph_check": "nearest graph node per referenced POI then shortest base-mode graph path for each ordered leg",
            "poi_coordinate_system_default": poi_coordinate_system,
            "graph_coordinate_system": "WGS-84",
            "not_checked": [
                "whether draft intent and response kind are correct",
                "whether expected tool use and candidate sets are correct",
                "candidate completeness or ranking quality",
                "whether a route is physically open, accessible, safe, or feasible under constraints and events",
                "whether route evidence, event snapshots, and hard constraints support the draft",
                "whether the nearest-node straight-line connector is traversable",
            ],
            "review_gap_check": "draft evidence completeness only; gaps do not establish expected answers or system performance",
        },
        "cases": cases,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_audit(dataset_path: Path, pois_path: Path, graph_path: Path,
              output_path: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    poi_doc = json.loads(pois_path.read_text(encoding="utf-8"))
    base_graph = nx.read_graphml(graph_path, force_multigraph=True)
    from spatial.routing import filter_graph_for_mode

    modes = {turn.get("travel_mode") for case in dataset.get("cases", [])
             for turn in case.get("expected_turns", []) if turn.get("travel_mode")}
    mode_graphs = {mode: filter_graph_for_mode(base_graph, mode)[0] for mode in sorted(modes)}
    report = audit_scenario_drafts(dataset, poi_doc["pois"], mode_graphs,
                                   poi_coordinate_system=poi_doc.get("coordinate_system", "GCJ-02"))
    report["git_head"] = _git_head()
    report["inputs"] = {
        "dataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path)},
        "poi_catalog": {"path": str(pois_path), "sha256": _sha256(pois_path)},
        "base_graph": {"path": str(graph_path), "sha256": _sha256(graph_path)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pois", type=Path, default=DEFAULT_POIS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or DEFAULT_OUTPUT_ROOT / f"scenario_draft_audit_{timestamp}.json"
    if not _OUTPUT_NAME.fullmatch(output.name):
        parser.error("output filename must match scenario_draft_audit_<id>.json")
    path = run_audit(args.dataset, args.pois, args.graph, output)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"report": str(path), "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
