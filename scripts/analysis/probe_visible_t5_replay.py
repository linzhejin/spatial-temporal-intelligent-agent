"""Replay all visible T5 drafts in isolation; never produce formal test scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def selected_followup_poi(current_turn: dict, next_turn: dict) -> str | None:
    """Identify the venue selected after a candidate turn, not an older stop."""
    if next_turn.get("kind") == "candidates":
        return next_turn.get("start_poi_id")
    if next_turn.get("kind") == "route":
        previous_start = current_turn.get("start_poi_id")
        for poi_id in [*next_turn.get("via_poi_ids", []), next_turn.get("end_poi_id")]:
            if poi_id and poi_id != previous_start:
                return poi_id
    return None


def compare_fields(expected: dict, actual: dict) -> dict[str, bool]:
    fields = ("status", "kind", "start_poi_id", "via_poi_ids", "travel_mode",
              "hard_constraints")
    if expected.get("kind") != "candidates":
        fields += ("end_poi_id", "strategy")
    matches = {field: actual.get(field) == expected.get(field) for field in fields}
    def active(value):
        return {key: flag for key, flag in (value or {}).items()
                if flag is not False and flag is not None and flag != [] and flag != {}}
    matches["hard_constraints"] = (
        active(actual.get("hard_constraints")) ==
        active(expected.get("hard_constraints")))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # Import the application only after the isolated database has been checked.
    from tests.db_safety import verify_isolated_test_database

    url = os.environ["TEST_DATABASE_URL"]
    verify_isolated_test_database(url, business_urls=(os.environ.get("DATABASE_URL"),))
    os.environ.update(DATABASE_URL=url, AGENT_REPLAY_ENABLE_LOCAL="1",
                      FLASK_ENV="development", DEEPSEEK_API_KEY="development-model-blocked")

    import config

    config.DEEPSEEK_API_KEY = "development-model-blocked"

    from langgraph.checkpoint.postgres import PostgresSaver

    from agents import planner, workflow
    from agents.workflow import build_agent_workflow
    from app import create_app
    from experiments.candidate_reference import build_candidate_reference
    from experiments.route_reference import build_route_reference
    from experiments.scenario_live_adapter import build_poi_lookup, run_case
    from experiments.scenario_v2 import (audit_candidate_order, audit_route_edges,
                                         candidate_spec_for_turn, load_frozen_graph)
    from scripts.analysis.probe_product_live_case import _LocalSession
    from storage import database

    paths = {
        "draft": ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json",
        "pois": ROOT / "data/pois.json",
        "graph": ROOT / "data/whu_road_network.graphml",
        "overrides": ROOT / "data/edge_overrides.json",
        "reviews": ROOT / "data/campus_review_decisions.json",
    }
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for name, path in paths.items()}
    dataset = json.loads(paths["draft"].read_text(encoding="utf-8"))
    if (dataset.get("version") != "scenarios-test-v2-diverse-authoring-draft"
            or dataset.get("authoring_status") is None):
        raise ValueError("only visible development authoring drafts are accepted")
    for name in ("pois", "graph", "overrides", "reviews"):
        field = "poi" if name == "pois" else name
        if dataset.get(f"source_{field}_sha256") != source_hashes[name]:
            raise ValueError(f"{name} source differs from the authoring bank")
    cases = [case for case in dataset["cases"] if case["category"] == "T5"]
    if len(cases) != 24:
        raise ValueError("expected 24 visible T5 authoring cases")
    if args.case_id:
        chosen = set(args.case_id)
        cases = [case for case in cases if case["id"] in chosen]
        if len(cases) != len(chosen):
            raise ValueError("unknown or duplicate T5 case ID")
    pois = json.loads(paths["pois"].read_text(encoding="utf-8"))["pois"]
    lookup = build_poi_lookup(pois)
    graph = load_frozen_graph(paths["graph"], paths["overrides"], paths["reviews"])
    workflow._run_weather_tool = lambda: {"weather": "晴", "temperature": 22,
                                          "windpower": "2级"}
    database.initialize(url)
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="visible-t5-development", DATABASE_URL=url)

    model_attempts = [0]

    def forbid_model(**_kwargs):
        model_attempts[0] += 1
        raise RuntimeError("model call forbidden in visible T5 diagnostic")

    rows = []
    with PostgresSaver.from_conn_string(url) as saver:
        saver.setup()
        agent = build_agent_workflow(saver, database_url=url)
        with patch.object(planner, "_make_client", forbid_model):
            session = _LocalSession(app.test_client(), agent, url)
            for case in cases:
                model_attempts[0] = 0
                try:
                    turns, elapsed = run_case(case, session, "http://127.0.0.1:55440",
                                              lookup, poll_interval_s=0)
                    details = []
                    checks = [len(turns) == len(case["expected_turns"])]
                    for index, (expected, actual) in enumerate(zip(case["expected_turns"], turns)):
                        matches = compare_fields(expected, actual)
                        output_match = set(expected["required_outputs"]) <= set(actual["outputs"])
                        detail = {
                            "turn_index": index, "expected_kind": expected["kind"],
                            "actual_kind": actual["kind"], "field_matches": matches,
                            "field_mismatches": {
                                field: {"expected": expected.get(field),
                                        "actual": actual.get(field)}
                                for field, matches_expected in matches.items()
                                if not matches_expected
                            },
                            "required_outputs_present": output_match,
                            "actual_unknowns": actual["unknowns"],
                        }
                        checks.extend([*matches.values(), output_match])
                        if expected["kind"] == actual["kind"] == "candidates":
                            spec = candidate_spec_for_turn(case, index)
                            reference = build_candidate_reference(
                                graph, pois, start_poi_id=expected["start_poi_id"], spec=spec)
                            rank = audit_candidate_order(actual["candidate_poi_ids"], reference)
                            detail["candidate_poi_ids"] = actual["candidate_poi_ids"]
                            detail["candidate_rank_audit"] = rank
                            checks.append(rank["verdict"] != "fail")
                            if index + 1 < len(case["expected_turns"]):
                                selected = selected_followup_poi(
                                    expected, case["expected_turns"][index + 1])
                                detail["selected_followup_poi"] = selected
                                detail["selection_in_candidates"] = (
                                    selected in actual["candidate_poi_ids"])
                                checks.append(detail["selection_in_candidates"])
                        if expected["kind"] == actual["kind"] == "route":
                            reference = build_route_reference(
                                graph, pois, start_poi_id=expected["start_poi_id"],
                                end_poi_id=expected["end_poi_id"],
                                via_poi_ids=expected["via_poi_ids"],
                                mode=expected["travel_mode"],
                                constraints=expected["hard_constraints"],
                            )
                            detail["route_edge_ids"] = actual["route_edge_ids"]
                            detail["route_audit"] = audit_route_edges(
                                graph, actual["route_edge_ids"], reference["start_node"],
                                reference["end_node"], reference["via_nodes"],
                                expected["travel_mode"], expected["hard_constraints"])
                            detail["access_uncertain"] = reference["access_uncertain"]
                            checks.extend([detail["route_audit"]["route"] == "pass",
                                           not reference["access_uncertain"]])
                        details.append(detail)
                    status = ("model_required" if model_attempts[0] else
                              "pass" if all(checks) else "fail")
                    rows.append({"case_id": case["id"], "elapsed_s": round(elapsed, 3),
                                 "expected_turn_count": len(case["expected_turns"]),
                                 "actual_turn_count": len(turns),
                                 "model_call_attempts_blocked": model_attempts[0],
                                 "case_status": status,
                                 "development_checks_pass": all(checks) if not model_attempts[0] else None,
                                 "turns": details})
                except Exception as exc:
                    rows.append({"case_id": case["id"], "case_status": "fail",
                                 "development_checks_pass": False,
                                 "error_type": type(exc).__name__})

    report = {"status": "visible_development_diagnostic_no_model_not_formal",
              "source_sha256": source_hashes, "case_count": len(rows),
              "development_checks_pass_count": sum(row["case_status"] == "pass" for row in rows),
              "model_required_count": sum(row["case_status"] == "model_required" for row in rows),
              "unexpected_failure_count": sum(row["case_status"] == "fail" for row in rows),
              "candidate_order_unknown_turns": sum(
                  turn.get("candidate_rank_audit", {}).get("verdict") == "unknown"
                  for row in rows for turn in row.get("turns", [])),
              "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"cases": len(rows),
                      "development_checks_pass": report["development_checks_pass_count"],
                      "model_required": report["model_required_count"],
                      "unexpected_failures": report["unexpected_failure_count"],
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["unexpected_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
