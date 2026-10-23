"""Run one single-turn development case through the product Agent in isolation.

This is a diagnostic pilot, not an experiment score. It never loads the
business database or writes the output into a paper results table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--test-database-url", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--weather-fixture", type=Path,
                        help="Labelled synthetic weather for isolated development replay")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "experiments/datasets/scenarios_dev_v2.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/experiments/product_agent_pilot.json")
    parser.add_argument("--allow-model-call", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_call:
        raise SystemExit("explicit --allow-model-call is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("split") != "development":
        raise SystemExit("only development cases are accepted")
    case = next((row for row in dataset["cases"] if row.get("id") == args.case_id), None)
    if case is None or len(case.get("messages", [])) != 1:
        raise SystemExit("case must exist and contain exactly one turn")
    weather_required = "weather" in (case["expected_turns"][0].get("required_outputs") or [])
    if weather_required and args.weather_fixture is None:
        raise SystemExit("weather case requires --weather-fixture for comparable replay")

    env = dotenv_values(args.env_file)
    from tests.db_safety import verify_isolated_test_database
    verify_isolated_test_database(args.test_database_url,
                                  business_urls=(env.get("DATABASE_URL"),))
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DeepSeek credential missing from selected local config")
    os.environ["DATABASE_URL"] = args.test_database_url
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
    os.environ["LLM_MODEL"] = env.get("LLM_MODEL") or "deepseek-chat"

    from agents.planner import run_agent
    from agents.tools import TOOL_SCHEMAS
    from experiments.controlled_weather import load_development_weather
    snapshot, weather_provenance = (load_development_weather(args.weather_fixture)
                                    if args.weather_fixture else (None, None))
    tool_trace: list[dict] = []
    model_trace: list[dict] = []
    started = time.monotonic()
    response = run_agent(case["messages"][0]["content"],
                         weather_snapshot=snapshot,
                         tool_trace_callback=tool_trace.append,
                         model_trace_callback=model_trace.append)
    report = {
        "status": "development_pilot_only", "condition": "campus_agent",
        "case_id": args.case_id,
        "development_dataset_sha256": _digest(args.dataset),
        "tool_schema_sha256": hashlib.sha256(json.dumps(
            TOOL_SCHEMAS, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest(),
        "case_messages_sha256": hashlib.sha256(json.dumps(
            case["messages"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest(),
        "data_component_sha256": {name: _digest(ROOT / name) for name in (
            "data/pois.json", "data/whu_road_network.graphml",
            "data/edge_overrides.json", "data/campus_review_decisions.json")},
        "weather_fixture": weather_provenance,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        "response_kind": response.get("response_kind"),
        "message": response.get("message"),
        "route": response.get("route"),
        "weather": response.get("weather"),
        "clarify": response.get("clarify"),
        "candidates": response.get("candidates"),
        "tool_trace": tool_trace,
        "model_trace": model_trace,
        "limitation": "single visible development case; synthetic weather; no independent gold or formal score",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"case_id": args.case_id, "response_kind": report["response_kind"],
                      "tool_calls": len(tool_trace), "model_requests": len(model_trace)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
