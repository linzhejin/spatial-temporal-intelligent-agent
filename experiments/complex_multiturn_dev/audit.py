"""Independently inspect route edges in saved development observations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import networkx as nx

from experiments.complex_multiturn_dev.contracts import assess_turn, read_dataset
from experiments.complex_multiturn_dev.event_runtime import load_event_bundle
from experiments.complex_multiturn_dev.graph_checks import (
    inspect_delivery_route, inspect_expected_route_fields, inspect_radius_transition, inspect_reference_endpoints,
    timeout_evidence,
)
from experiments.complex_multiturn_dev.observation import _id
from experiments.complex_multiturn_dev.semantic_checks import (
    inspect_candidate_semantics, inspect_destination_clarification, inspect_empty_search,
    inspect_expected_no_path, inspect_route_distance_requirement, inspect_stale_center_requirement,
    inspect_provider_type_denial, inspect_sport_clarification, inspect_start_clarification, inspect_time_only_scope,
    inspect_unconfirmed_position, inspect_saved_stop_orders,
)
from experiments.independent_source_snapshot import active_closed_edges, mode_edge_evidence
from experiments.scenario_live_adapter import build_poi_lookup


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/datasets/complex_multiturn_dev_v1"


def audit(run_dir: Path, dataset: Path, output_name: str = "independent_route_audit.jsonl"):
    run_dir, dataset = Path(run_dir), Path(dataset)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("run and dataset differ")
    graph_path, poi_path = ROOT / "data/whu_road_network.graphml", ROOT / "data/pois.json"
    anchors_data = json.loads((DATA / "anchor_snapshot_60m.json").read_text(encoding="utf-8"))
    if (anchors_data["graph_sha256"] != hashlib.sha256(graph_path.read_bytes()).hexdigest()
            or anchors_data["poi_sha256"] != hashlib.sha256(poi_path.read_bytes()).hexdigest()):
        raise ValueError("frozen graph or POI snapshot changed")
    graph = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
    anchors = {row["poi_id"]: row for row in anchors_data["anchors"]}
    evidence = {mode: mode_edge_evidence(graph, mode, require_boundary=True,
                                         boundary_wkt=anchors_data["boundary_wkt"])
                for mode in ("walk", "bike", "drive")}
    poi_rows = json.loads(poi_path.read_text(encoding="utf-8"))["pois"]
    lookup = build_poi_lookup(poi_rows)
    local_pois = {row["id"]: row for row in poi_rows if row.get("id")}
    cases = {case["id"]: case for case in read_dataset(dataset)["cases"]}
    reference_rubric = json.loads((DATA / "reference_route_rubric.json").read_text(encoding="utf-8"))["cases"]
    semantic_rubric = json.loads((DATA / "semantic_rubric_v4.json").read_text(encoding="utf-8"))["cases"]
    frozen_cases = {case["id"]: case for case in read_dataset(DATA / "cases_event_endpoint_revised.json")["cases"]}
    for ident, rules in semantic_rubric.items():
        if ident in cases and (len(rules) != len(cases[ident]["turns"]) or any(
                left["user"] != right["user"] for left, right in zip(cases[ident]["turns"], frozen_cases[ident]["turns"]))):
            raise ValueError("semantic rubric does not match the case wording")
    event_cases = any(case_id.startswith("events_") for case_id in manifest["case_ids"])
    fixtures = load_event_bundle(DATA / "event_fixtures_v2.json", dataset, graph_path, poi_path) if event_cases else {}
    source = run_dir / "restored_observations.jsonl"
    if not source.exists():
        source = run_dir / "turns.jsonl"
    if Path(output_name).name != output_name:
        raise ValueError("output must be a filename within the run directory")
    output = run_dir / output_name
    if output.exists():
        raise FileExistsError(output)
    summary = Counter()
    previous_radius = {}
    previously_shown, shown_rows, tool_history, previous_centers = {}, {}, {}, {}
    previous_routes, prior_user_inputs = {}, {}
    provider_path = run_dir / "provider_evidence.jsonl"
    provider_records = [json.loads(line) for line in provider_path.read_text(encoding="utf-8").splitlines()] if provider_path.exists() else []
    candidate_scope = {}
    for path in run_dir.glob("*_candidate_scope_audit.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            candidate_scope.setdefault((row["case"], row["turn"]), []).append(row)
    candidate_time = {}
    for path in run_dir.glob("*_candidate_time_audit.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            candidate_time.setdefault((row["case"], row["turn"]), []).append(row)
    time_checker_hash = hashlib.sha256((ROOT / "experiments/complex_multiturn_dev/candidate_checks.py").read_bytes()).hexdigest()
    items = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()] if source.exists() else []
    recovered = run_dir / "recovered_uncollected.jsonl"
    if recovered.exists():
        extra = [json.loads(line) for line in recovered.read_text(encoding="utf-8").splitlines()]
        if {row.get("run_id") for row in items} & {row.get("run_id") for row in extra}:
            raise ValueError("recovered observations duplicate an existing run")
        items.extend(extra)
        order = {ident: index for index, ident in enumerate(manifest["case_ids"])}
        items.sort(key=lambda row: (order.get(row.get("case"), len(order)), row.get("turn", 0)))
    with output.open("x", encoding="utf-8", newline="\n") as dest:
        for item in items:
            if item.get("budget_stop_reason"):
                dest.write(json.dumps({"case": item.get("case"), "turn": item.get("turn"),
                                       "verdict": "not_run", "reason": "budget_interrupted"}, ensure_ascii=False) + "\n")
                summary["not_run"] += 1
                continue
            if not item.get("run_id"):
                dest.write(json.dumps({"case": item.get("case"), "turn": item.get("turn"),
                                       "verdict": "not_run", "reason": item.get("reason")}, ensure_ascii=False) + "\n")
                summary["not_run"] += 1
                continue
            case = cases[item["case"]]
            turn = case["turns"][item["turn"] - 1]
            obs, result = dict(item.get("observed") or {}), item.get("result") or {}
            saved_request = result.get("place_request") or result.get("completed_place_request") or {}
            saved_semantic = saved_request.get("semantic_request") if isinstance(saved_request, dict) else None
            requirements = saved_semantic.get("requirements") if isinstance(saved_semantic, dict) else None
            if not isinstance(requirements, list):
                requirements = []
            if item.get("task_id"):
                saved_constraints = saved_request.get("constraints") if isinstance(saved_request, dict) else None
                if isinstance(saved_constraints, dict):
                    for source_key, observed_key in (("max_duration_s", "max_duration_s"),
                                                     ("max_distance_m", "route_max_distance_m")):
                        value = saved_constraints.get(source_key)
                        if (observed_key not in obs and source_key in saved_constraints
                                and (value is None or type(value) in (int, float))):
                            obs[observed_key] = value
            clarification = result.get("clarify") if isinstance(result.get("clarify"), dict) else {}
            if result.get("response_kind") == "clarify" and isinstance(clarification.get("slot"), str):
                obs["clarification_slot"] = clarification["slot"]
            if not obs.get("destination") and isinstance(result.get("route_state"), dict):
                destination = _id(result["route_state"].get("end"), lookup)
                if destination:
                    obs["destination"] = destination
            if any(check.get("field") == "time_only_scope_integrity" for check in turn.get("checks") or []):
                verified_scope = inspect_time_only_scope(obs, item.get("tool_observations"), lookup)
                if verified_scope is not None:
                    obs["time_only_scope_integrity"] = verified_scope
            base = assess_turn(turn, obs)
            expected_stops = next((check.get("value") for check in turn.get("checks") or []
                                   if check.get("field") == "stops" and check.get("op") == "ordered_contains"), None)
            base["failures"].extend(inspect_saved_stop_orders(requirements, expected_stops, lookup))
            user_text = item.get("actual_user_input") or turn["user"]
            user_history = prior_user_inputs.setdefault(item["case"], [])
            base["failures"].extend(inspect_unconfirmed_position(
                " ".join([*user_history, user_text]), result.get("assistant_text") or result.get("message"),
                previous_routes.get(item["case"]), lookup))
            user_history.append(user_text)
            if isinstance(result.get("route_state"), dict):
                previous_routes[item["case"]] = result["route_state"]
            prior_raw = [poi for record in provider_records if record.get("case") == item["case"]
                         and type(record.get("turn")) is int and record["turn"] <= item["turn"]
                         for poi in record.get("pois") or []]
            saved_pool = saved_request.get("candidate_pool") if isinstance(saved_request, dict) else []
            base["failures"].extend(inspect_provider_type_denial(
                result.get("assistant_text") or result.get("message"), saved_pool, prior_raw))
            previous_center, current_center = previous_centers.get(item["case"]), obs.get("search_anchor")
            if item.get("task_id") and previous_center and current_center and previous_center != current_center:
                utterance = item.get("actual_user_input") or turn["user"]
                previous_names = [name for name, ident in lookup["names"].items() if ident == previous_center and len(name) >= 2]
                if not any(name in utterance for name in previous_names):
                    base["failures"].extend(inspect_stale_center_requirement(previous_center, current_center, requirements, lookup))
            if current_center:
                previous_centers[item["case"]] = current_center
            timeout = timeout_evidence(result)
            if timeout:
                base["failures"].append(timeout)
            for check in candidate_scope.get((item["case"], item["turn"]), []):
                if check["verdict"] == "fail":
                    base["failures"].append("candidate_scope:" + str(check["candidate"]) + ":" + check["reason"])
                elif check["verdict"] == "unknown":
                    base["unknowns"].append("candidate_scope_unverified:" + str(check["candidate"]))
            rules = semantic_rubric.get(item["case"]) or []
            rule = rules[item["turn"] - 1] if item["turn"] <= len(rules) else {}
            distance_check = inspect_route_distance_requirement(rule, obs)
            base["failures"].extend(distance_check["failures"])
            base["unknowns"].extend(distance_check["unknowns"])
            candidates = result.get("candidates") if isinstance(result.get("candidates"), list) else []
            visible_candidates = bool(candidates and (result.get("response_kind") == "candidates" or
                result.get("response_kind") == "route" and result.get("needs_input") is True))
            expects_time = any(check.get("field") == "max_duration_s" and check.get("op") in {"eq", "exists"}
                               for check in turn.get("checks") or [])
            if expects_time and visible_candidates:
                checked = candidate_time.get((item["case"], item["turn"]), [])
                ids = {row.get("poi_id") for row in candidates if isinstance(row, dict)}
                checked_ids = {row.get("candidate") for row in checked}
                if (ids != checked_ids or len(checked) != len(candidates)
                        or any(row.get("checker_sha256") != time_checker_hash for row in checked)):
                    base["unknowns"].append("candidate_time_evidence_missing_or_stale")
                else:
                    for row in checked:
                        if row.get("verdict") == "fail":
                            base["failures"].append("candidate_time:" + str(row.get("candidate")) + ":" + str(row.get("reason")))
                        elif row.get("verdict") != "pass":
                            base["unknowns"].append("candidate_time_unverified:" + str(row.get("candidate")))
            shown = previously_shown.setdefault(item["case"], set())
            prior_candidates = shown_rows.setdefault(item["case"], [])
            tool_history.setdefault(item["case"], []).extend(item.get("tool_observations") or [])
            semantic_ok = False
            if rule.get("activity"):
                if candidates and not visible_candidates:
                    base["failures"].append("candidate_cards_not_visible")
                if visible_candidates:
                    raw = [poi for record in provider_records if record.get("case") == item["case"]
                           and type(record.get("turn")) is int and record["turn"] <= item["turn"]
                           for poi in record.get("pois") or []]
                    checked = inspect_candidate_semantics(rule, candidates, obs.get("activities"), local_pois, raw, shown,
                                                          requirements=requirements, previously_shown_rows=prior_candidates)
                    base["failures"].extend(checked["failures"])
                    base["unknowns"].extend(checked["unknowns"])
                    expected_ids = {row.get("poi_id") for row in candidates if isinstance(row, dict)}
                    scope_rows = candidate_scope.get((item["case"], item["turn"]), [])
                    scoped_ids = {row.get("candidate") for row in scope_rows}
                    if expected_ids != scoped_ids or len(scope_rows) != len(candidates):
                        base["unknowns"].append("candidate_scope_evidence_missing")
                    semantic_ok = (not checked["failures"] and not checked["unknowns"]
                                   and expected_ids == scoped_ids and len(scope_rows) == len(candidates)
                                   and all(row.get("verdict") == "pass" for row in scope_rows))
                else:
                    checked = inspect_candidate_semantics(rule, [], obs.get("activities"), local_pois, [], shown,
                                                          requirements=requirements)
                    base["failures"].extend(checked["failures"])
                    base["unknowns"].extend(checked["unknowns"])
                    empty = inspect_empty_search(rule, obs, result, tool_history[item["case"]], lookup)
                    semantic_ok = empty["verdict"] == "pass" and not checked["failures"] and not checked["unknowns"]
                    if empty["verdict"] == "fail":
                        base["failures"].append(empty["reason"])
                    elif empty["verdict"] != "pass":
                        base["unknowns"].append(empty["reason"])
            if visible_candidates:
                shown.update(row.get("poi_id") for row in candidates if isinstance(row, dict) and row.get("poi_id"))
                prior_candidates.extend(row for row in candidates if isinstance(row, dict))
            if case["group"] == "search" and type(obs.get("radius_m")) is int:
                prior = previous_radius.get(case["id"], 1000)
                issue = inspect_radius_transition(prior, obs["radius_m"], item.get("actual_user_input") or turn["user"])
                if issue:
                    base["failures"].append(issue)
                previous_radius[case["id"]] = obs["radius_m"]
            event = fixtures.get(item["case"])
            closed = set()
            if event:
                stage = turn["event_stage"]
                snapshot = event["stages"][stage]
                mode = obs.get("mode") or event["mode"]
                closed = active_closed_edges(graph, {"as_of_epoch": snapshot["epoch"],
                                                     "conditions": snapshot["rows"]}, mode)
            is_route = result.get("response_kind") == "route" or bool(result.get("recommended_edge_ids"))
            expects_start = any(check.get("field") == "clarification_slot" and check.get("value") == "start"
                                for check in turn.get("checks") or [])
            expects_destination = rule.get("clarification") == "ambiguous_destination"
            expects_sport = rule.get("clarification") == "ambiguous_sport"
            if expects_start or expects_destination or expects_sport:
                checked = (inspect_start_clarification(turn, result) if expects_start else
                           inspect_destination_clarification(result) if expects_destination else
                           inspect_sport_clarification(result))
                if checked["verdict"] == "fail":
                    base["failures"].append(checked["reason"])
                elif checked["verdict"] == "unknown":
                    base["unknowns"].append(checked["reason"])
            route = None
            if is_route:
                delivered = inspect_expected_route_fields(turn, result, lookup)
                base["failures"].extend(delivered["failures"])
                base["unknowns"].extend(delivered["unknowns"])
                reference = reference_rubric.get(item["case"])
                if reference:
                    if reference["quoted_basis"] not in case["turns"][0]["user"]:
                        raise ValueError("reference rubric does not match this dataset")
                    checked = inspect_reference_endpoints(reference, result, lookup)
                    base["failures"].extend(checked["failures"])
                    base["unknowns"].extend(checked["unknowns"])
                expected_steps = next((check["value"] for check in turn["checks"]
                                       if check["field"] == "avoid_steps" and check["op"] == "eq"), None)
                route = inspect_delivery_route(graph, anchors, evidence, lookup, result, closed,
                                               avoid_steps=expected_steps)
                if route["verdict"] == "fail":
                    base["failures"].extend("route:" + issue for issue in route["issues"])
                elif route["verdict"] == "unknown":
                    base["unknowns"].extend("route:" + issue for issue in route["unknowns"])
                if event:
                    expected_end = event["end_poi_id"]
                    if item["case"] == "events_04" and item["turn"] == 3:
                        expected_end = event["start_poi_id"]
                    if obs.get("destination") is None:
                        base["unknowns"].append("event_destination_unresolved")
                    elif obs["destination"] != expected_end:
                        base["failures"].append("event_destination_mismatch")
            else:
                if expects_start or expects_destination or expects_sport:
                    pass
                elif event and event["kind"] == "blocked" and turn.get("event_stage") == "active":
                    checked = inspect_expected_no_path(event, item, result, lookup)
                    if result.get("failure_code") == "route_not_found":
                        base["failures"] = [failure for failure in base["failures"] if failure != "run:failed"]
                    if checked["verdict"] != "pass":
                        base["unknowns"].append(checked["reason"])
                elif semantic_ok:
                    pass
                else:
                    base["unknowns"].append("non_route_delivery_requires_semantic_review")
            verdict = "fail" if base["failures"] else "unknown" if base["unknowns"] else "pass"
            summary[verdict] += 1
            dest.write(json.dumps({"case": item["case"], "turn": item["turn"], "verdict": verdict,
                                   "failures": base["failures"], "unknowns": base["unknowns"],
                                   "route_check": route,
                                   "boundary": "frozen graph only; site access and provider freshness unverified"},
                                  ensure_ascii=False, default=str) + "\n")
    return {"output": str(output), "turn_counts": dict(summary), "scope": "development_graph_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-name", default="independent_route_audit.jsonl")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.dataset, args.output_name), ensure_ascii=False))


if __name__ == "__main__":
    main()
