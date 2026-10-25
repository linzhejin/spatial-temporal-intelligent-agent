import json
import os
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.postgres import PostgresSaver

from agents import planner, workflow
from agents.workflow import build_agent_workflow
from app import create_app
from experiments.scenario_live_adapter import build_poi_lookup, run_case
from scripts.analysis.probe_product_live_case import _LocalSession
from storage import database
from tests.db_safety import verify_isolated_test_database

ROOT = Path(__file__).resolve().parents[2]
url = os.environ["TEST_DATABASE_URL"]
verify_isolated_test_database(url, business_urls=(os.environ.get("DATABASE_URL"),))
os.environ.update(DATABASE_URL=url, AGENT_REPLAY_ENABLE_LOCAL="1", FLASK_ENV="development")
database.initialize(url)
workflow._run_weather_tool = lambda: {"weather": "晴", "temperature": 22, "windpower": "2级"}
app = create_app()
app.config.update(TESTING=True, SECRET_KEY="visible-t3-only", DATABASE_URL=url)
dataset = json.loads((ROOT / "experiments/datasets/scenarios_test_v2_diverse_draft.json").read_text(encoding="utf-8"))
cases = [case for case in dataset["cases"] if case["category"] == "T3"]
pois = json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]
lookup = build_poi_lookup(pois)
rows = []

def forbidden_model(**_kwargs):
    raise RuntimeError("external model call forbidden by visible T3 diagnostic")

with PostgresSaver.from_conn_string(url) as saver:
    saver.setup()
    graph = build_agent_workflow(saver, database_url=url)
    with patch.object(planner, "_make_client", forbidden_model):
        session = _LocalSession(app.test_client(), graph, url)
        for case in cases:
            try:
                turns, elapsed = run_case(case, session, "http://127.0.0.1:55440", lookup,
                                          poll_interval_s=0)
                rows.append({"case_id": case["id"], "status": [t["status"] for t in turns],
                             "kind": [t["kind"] for t in turns],
                             "slot": [t["clarification_slot"] for t in turns],
                             "start": [t["start_poi_id"] for t in turns],
                             "end": [t["end_poi_id"] for t in turns],
                             "outputs": [t["outputs"] for t in turns],
                             "elapsed_s": round(elapsed, 3)})
            except Exception as exc:
                rows.append({"case_id": case["id"], "error_type": type(exc).__name__})

target = ROOT / "output/experiments/visible_t3_replay_after_fix_20261025.json"
target.write_text(json.dumps({"status": "visible_development_diagnostic_no_model",
                              "cases": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"total": len(rows), "rows": rows}, ensure_ascii=False))
