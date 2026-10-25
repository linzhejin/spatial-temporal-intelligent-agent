"""Local development diagnostic for prior destinations and ordered full routes."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.postgres import PostgresSaver

from agents import planner, workflow
from agents.workflow import build_agent_workflow
from app import create_app
from experiments.route_reference import build_route_reference
from experiments.scenario_live_adapter import build_poi_lookup, run_case
from experiments.scenario_v2 import audit_route_edges, load_frozen_graph
from scripts.analysis.probe_product_live_case import _LocalSession
from storage import database
from tests.db_safety import verify_isolated_test_database

ROOT = Path(__file__).resolve().parents[2]
url = os.environ["TEST_DATABASE_URL"]
verify_isolated_test_database(url, business_urls=(os.environ.get("DATABASE_URL"),))
os.environ.update(DATABASE_URL=url, AGENT_REPLAY_ENABLE_LOCAL="1", FLASK_ENV="development")
workflow._run_weather_tool = lambda: {"weather": "晴", "temperature": 22, "windpower": "2级"}
database.initialize(url)
app = create_app()
app.config.update(TESTING=True, SECRET_KEY="development-only-t3-t5", DATABASE_URL=url)
dataset_path = ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json"
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
case_ids = [f"v2h_T3_destination_weather_{i}" for i in (1, 2, 3)] + [
    f"v2h_T5_food_before_route_{i}" for i in (1, 2, 3)
]
cases = {case["id"]: case for case in dataset["cases"]}
pois = json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]
lookup = build_poi_lookup(pois)
graph = load_frozen_graph(ROOT / "data/whu_road_network.graphml",
                          ROOT / "data/edge_overrides.json",
                          ROOT / "data/campus_review_decisions.json")

def forbid_model(**_kwargs):
    raise RuntimeError("model call forbidden in deterministic development probe")

rows = []
with PostgresSaver.from_conn_string(url) as saver:
    saver.setup()
    graph_agent = build_agent_workflow(saver, database_url=url)
    with patch.object(planner, "_make_client", forbid_model):
        session = _LocalSession(app.test_client(), graph_agent, url)
        for case_id in case_ids:
            case = cases[case_id]
            turns, elapsed = run_case(case, session, "http://127.0.0.1:55440", lookup,
                                      poll_interval_s=0)
            details = []
            for index, (expected, actual) in enumerate(zip(case["expected_turns"], turns)):
                detail = {
                    "turn_index": index,
                    "expected_status": expected["status"], "actual_status": actual["status"],
                    "expected_kind": expected["kind"], "actual_kind": actual["kind"],
                    "expected_start": expected["start_poi_id"],
                    "actual_start": actual["start_poi_id"],
                    "expected_end": expected["end_poi_id"],
                    "actual_end": actual["end_poi_id"],
                    "expected_via": expected["via_poi_ids"],
                    "actual_via": actual["via_poi_ids"],
                    "expected_outputs": expected["required_outputs"],
                    "actual_outputs": actual["outputs"],
                    "actual_candidate_poi_ids": actual["candidate_poi_ids"],
                }
                if expected["kind"] == "route" and actual["kind"] == "route":
                    reference = build_route_reference(
                        graph, pois, start_poi_id=expected["start_poi_id"],
                        end_poi_id=expected["end_poi_id"],
                        via_poi_ids=expected["via_poi_ids"],
                        mode=expected["travel_mode"] or "walk",
                        constraints=expected["hard_constraints"],
                    )
                    edge_ids = actual["route_edge_ids"]
                    detail["edge_count"] = len(edge_ids)
                    detail["route_edge_ids"] = edge_ids
                    detail["independent_route_audit"] = audit_route_edges(
                        graph, edge_ids, reference["start_node"], reference["end_node"],
                        reference["via_nodes"], expected["travel_mode"] or "walk",
                        expected["hard_constraints"],
                    )
                    detail["endpoint_access_uncertain"] = reference["access_uncertain"]
                if expected["kind"] == "candidates" and index + 1 < len(case["expected_turns"]):
                    next_via = case["expected_turns"][index + 1]["via_poi_ids"]
                    if next_via:
                        detail["selected_stop_in_candidates"] = (
                            next_via[0] in actual["candidate_poi_ids"]
                        )
                details.append(detail)
            checks = [len(turns) == len(case["expected_turns"])]
            for detail in details:
                checks.extend([
                    detail["expected_status"] == detail["actual_status"],
                    detail["expected_kind"] == detail["actual_kind"],
                    detail["expected_start"] == detail["actual_start"],
                    set(detail["expected_outputs"]) <= set(detail["actual_outputs"]),
                ])
                if detail["expected_kind"] in {"clarify", "route"}:
                    checks.append(detail["expected_end"] == detail["actual_end"])
                if detail["expected_kind"] == "route":
                    checks.extend([
                        detail["expected_via"] == detail["actual_via"],
                        detail.get("independent_route_audit", {}).get("route") == "pass",
                    ])
                if "selected_stop_in_candidates" in detail:
                    checks.append(detail["selected_stop_in_candidates"])
            rows.append({"case_id": case_id, "expected_turn_count": len(case["expected_turns"]),
                         "actual_turn_count": len(turns), "elapsed_s": round(elapsed, 3),
                         "development_checks_pass": all(checks), "turns": details})

report = {"status": "visible_development_diagnostic_no_model",
          "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
          "data_component_sha256": {
              name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ("data/pois.json", "data/whu_road_network.graphml",
                           "data/edge_overrides.json", "data/campus_review_decisions.json")
          },
          "case_count": len(rows),
          "development_checks_pass_count": sum(row["development_checks_pass"] for row in rows),
          "cases": rows}
target = ROOT / "output/experiments/t3_t5_ordered_replay_20261025.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"cases": len(rows), "output": str(target)}, ensure_ascii=True))
if report["development_checks_pass_count"] != report["case_count"]:
    raise SystemExit("T3/T5 development replay checks failed; inspect the saved report")
