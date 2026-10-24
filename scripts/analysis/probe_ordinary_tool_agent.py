"""One development-only real-model pilot for the transcript tool baseline.

No score is produced. The command requires an explicitly isolated test DB and
never loads the business .env into the application process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]


def _make_audited_client(*, key, base_url, audit, transport=None):
    import httpx2
    from openai import OpenAI

    http_client = httpx2.Client(event_hooks=audit.event_hooks, timeout=25,
                                transport=transport)
    client = OpenAI(api_key=key, base_url=base_url,
                    default_headers={"Accept-Encoding": "gzip, deflate"},
                    http_client=http_client, timeout=25)
    return client, http_client


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                        default=ROOT / "output/experiments/ordinary_agent_pilot.json")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("split") != "development":
        raise SystemExit("pilot only accepts development cases")
    case = next((row for row in dataset["cases"] if row.get("id") == args.case_id), None)
    if case is None:
        raise SystemExit("case_id missing from development dataset")
    weather_required = any("weather" in (turn.get("required_outputs") or [])
                           for turn in case.get("expected_turns", []))
    if weather_required and args.weather_fixture is None:
        raise SystemExit("weather case requires --weather-fixture for comparable replay")

    env = dotenv_values(args.env_file)
    from tests.db_safety import verify_isolated_test_database
    verify_isolated_test_database(
        args.test_database_url,
        business_urls=(env.get("DATABASE_URL"),),
    )
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY unavailable in the selected local env file")
    os.environ["DATABASE_URL"] = args.test_database_url
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
    os.environ["LLM_MODEL"] = env.get("LLM_MODEL") or "deepseek-chat"

    from agents.tools import TOOL_SCHEMAS
    from experiments.baselines.openai_transport_audit import OpenAITransportAudit
    from experiments.baselines.ordinary_tool_agent import SYSTEM_PROMPT, run_conversation
    from experiments.controlled_weather import load_development_weather

    weather_snapshot, weather_provenance = (load_development_weather(args.weather_fixture)
                                            if args.weather_fixture else (None, None))

    transport_audit = OpenAITransportAudit()
    client, http_client = _make_audited_client(
        key=key, base_url=os.environ["OPENAI_BASE_URL"], audit=transport_audit,
    )
    turns = [message["content"] for message in case["messages"]]
    try:
        result = run_conversation(turns, client=client, model=os.environ["LLM_MODEL"],
                                  context=({"_route_weather_snapshot": weather_snapshot}
                                           if weather_provenance else None),
                                  max_rounds_per_turn=6, time_budget_s=40,
                                  max_tokens=800)
    finally:
        http_client.close()
    tool_schema_sha256 = hashlib.sha256(json.dumps(
        TOOL_SCHEMAS, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    data_hashes = {name: _sha256(ROOT / name) for name in (
        "data/pois.json", "data/whu_road_network.graphml",
        "data/edge_overrides.json", "data/campus_review_decisions.json",
    )}
    report = {"status": "development_pilot_only", "case_id": args.case_id,
              "development_dataset_sha256": _sha256(args.dataset),
              "case_messages_sha256": hashlib.sha256(json.dumps(
                  case["messages"], ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"),
              ).encode("utf-8")).hexdigest(),
              "baseline_system_prompt_sha256": hashlib.sha256(json.dumps(
                  SYSTEM_PROMPT, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"),
              ).encode("utf-8")).hexdigest(),
              "resolved_model_id": result["resolved_model_id"],
              "requested_model": result["model"],
              "tool_schema_sha256": tool_schema_sha256,
              "data_component_sha256": data_hashes,
              "event_snapshot_sha256": None,
              "weather_fixture": weather_provenance,
              "tool_calls": result["tool_calls"],
              "measured_usage": result["measured_usage"],
              **transport_audit.summary(logical_requests=result["measured_usage"]["requests"]),
              "turns": result["turns"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output.with_name(args.output.stem + "_run_evidence.json")
    evidence_path.write_text(json.dumps(result["run_evidence"], ensure_ascii=False,
                                        indent=2) + "\n", encoding="utf-8")
    report["run_evidence_path"] = str(evidence_path.resolve())
    report["run_evidence_sha256"] = _sha256(evidence_path)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"case_id": args.case_id,
                      "turn_statuses": [turn["status"] for turn in result["turns"]],
                      "resolved_model_id": result["resolved_model_id"],
                      "tool_calls": result["tool_calls"],
                      "usage": result["measured_usage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
