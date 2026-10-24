"""Replay one visible development case through the real API, worker and Agent.

This writes only an isolated test database and produces diagnostic projections,
not a formal experiment score. Raw prompts, credentials and response geometry
are omitted from the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from dotenv import dotenv_values

from experiments.scenario_live_adapter import TERMINAL, build_poi_lookup, run_case


ROOT = Path(__file__).resolve().parents[2]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


class _Response:
    def __init__(self, response):
        self.response = response

    def raise_for_status(self):
        if self.response.status_code >= 400:
            raise RuntimeError(f"local API returned HTTP {self.response.status_code}")

    def json(self):
        return self.response.get_json()


class _LocalSession:
    """Drive the same Flask API and database worker without opening a port."""

    def __init__(self, client, graph, database_url):
        self.client = client
        self.graph = graph
        self.database_url = database_url
        self.terminal_runs: dict[str, dict] = {}

    def post(self, url, json, timeout):
        from jobs.worker import process_next_run

        path = urlsplit(url).path
        response = self.client.post(path, json=json)
        if path.endswith("/messages") and response.status_code == 202:
            process_next_run(self.database_url, "development-replay-worker", self.graph)
        return _Response(response)

    def get(self, url, params=None, timeout=10):
        path = urlsplit(url).path
        response = self.client.get(path, query_string=params)
        data = (response.get_json() or {}).get("data") or {}
        if path.startswith("/api/runs/") and data.get("status") in TERMINAL:
            events = data.get("events") if isinstance(data.get("events"), list) else []
            model_requests = []
            for event in events:
                if isinstance(event, dict) and event.get("event_type") == "agent_model_request":
                    payload = event.get("payload")
                    if isinstance(payload, dict):
                        model_requests.append({key: payload.get(key) for key in (
                            "requested_model_id", "resolved_model_id", "tool_schema_sha256",
                            "temperature", "max_tokens", "input_tokens", "output_tokens",
                            "cache_hit_input_tokens", "cache_miss_input_tokens",
                            "outcome", "error_type",
                        )})
            run_id = path.rsplit("/", 1)[-1]
            self.terminal_runs[run_id] = {
                "run_id": run_id, "status": data["status"],
                "model_requests": model_requests,
            }
        return _Response(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--test-database-url", required=True)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "experiments/datasets/scenarios_dev_v2.json")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--weather-fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-model-call", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_call:
        raise SystemExit("explicit --allow-model-call is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("split") != "development":
        raise SystemExit("only visible development cases may be replayed")
    case = next((item for item in dataset["cases"] if item.get("id") == args.case_id), None)
    if case is None:
        raise SystemExit("case ID missing from development dataset")
    needs_weather = any("weather" in (turn.get("required_outputs") or [])
                        for turn in case.get("expected_turns", []))
    if needs_weather and args.weather_fixture is None:
        raise SystemExit("weather case requires a fixed development fixture")

    env = dotenv_values(args.env_file)
    from tests.db_safety import verify_isolated_test_database

    verify_isolated_test_database(args.test_database_url,
                                  business_urls=(env.get("DATABASE_URL"),
                                                 os.environ.get("DATABASE_URL")))
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DeepSeek credential unavailable in selected local config")
    os.environ.update({
        "DATABASE_URL": args.test_database_url,
        "DEEPSEEK_API_KEY": key,
        "OPENAI_BASE_URL": env.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1",
        "LLM_MODEL": env.get("LLM_MODEL") or "deepseek-chat",
        "AGENT_REPLAY_ENABLE_LOCAL": "1",
        "AGENT_TOOL_TRACE_ENABLED": "1",
        "FLASK_ENV": "development",
    })

    weather_provenance = None
    if args.weather_fixture:
        from experiments.controlled_weather import load_development_weather
        from agents import workflow

        weather_snapshot, weather_provenance = load_development_weather(args.weather_fixture)
        workflow._run_weather_tool = lambda: dict(weather_snapshot["live"])

    from agents.workflow import build_agent_workflow
    from agents.tools import TOOL_SCHEMAS
    from app import create_app
    from langgraph.checkpoint.postgres import PostgresSaver
    from storage import database

    database.initialize(args.test_database_url)
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY=secrets.token_hex(32),
                      DATABASE_URL=args.test_database_url)
    lookup = build_poi_lookup(json.loads((ROOT / "data/pois.json").read_text(
        encoding="utf-8"))["pois"])
    with PostgresSaver.from_conn_string(args.test_database_url) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, database_url=args.test_database_url)
        session = _LocalSession(app.test_client(), graph, args.test_database_url)
        turns, elapsed_s = run_case(case, session, "http://127.0.0.1:55440",
                                    lookup, poll_interval_s=0)

    report = {
        "status": "development_pilot_only", "condition": "campus_agent",
        "case_id": case["id"], "category": case["category"],
        "development_dataset_sha256": _file_hash(args.dataset),
        "case_messages_sha256": _digest(case["messages"]),
        "tool_schema_sha256": _digest(TOOL_SCHEMAS),
        "data_component_sha256": {name: _file_hash(ROOT / name) for name in (
            "data/pois.json", "data/whu_road_network.graphml",
            "data/edge_overrides.json", "data/campus_review_decisions.json")},
        "weather_fixture": weather_provenance,
        "elapsed_s": round(elapsed_s, 3),
        "turns": turns,
        "run_evidence": list(session.terminal_runs.values()),
        "limitations": ["visible development task", "not formal gold",
                        "graph verification is not field truth"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"case_id": case["id"],
                      "turn_statuses": [turn["status"] for turn in turns],
                      "turn_kinds": [turn["kind"] for turn in turns],
                      "model_requests": sum(len(row["model_requests"])
                                            for row in session.terminal_runs.values())},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
