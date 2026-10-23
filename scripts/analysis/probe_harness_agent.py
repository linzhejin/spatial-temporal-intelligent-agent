"""Run one development case through the real pinned DeepSeek Harness SDK.

The 14 campus tools execute through the same Python registry as the ordinary
baseline. A local middleware limits six logical model streams per turn. This
remains development-only until the provider response identity and actual HTTP
request budget are verified independently.
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


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _request_evidence(model_name: str, usage: dict, schema_sha256: str) -> dict:
    """Keep the requested model separate from the unobserved provider response."""
    return {
        "kind": "request", "requested_model_id": model_name,
        "resolved_model_id": None,
        "tool_schema_sha256": schema_sha256,
        "temperature": 0, "max_tokens": 800,
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "outcome": "success",
    }


def _attribute_transport_models(requests: list[dict], attempts: list[dict]) -> tuple[bool, str | None]:
    """Attribute response IDs only for a complete one-to-one HTTP observation."""
    successful = [attempt for attempt in attempts
                  if attempt["http_status"] == 200 and attempt["stream_complete"]]
    complete = (bool(requests) and len(attempts) == len(requests)
                and len(successful) == len(requests)
                and all(attempt["response_model_id"] for attempt in successful))
    if not complete:
        return False, None
    for record, attempt in zip(requests, successful):
        record["resolved_model_id"] = attempt["response_model_id"]
    models = {record["resolved_model_id"] for record in requests}
    return True, next(iter(models)) if len(models) == 1 else None


def _patch(plugin: Path) -> list[dict]:
    return [
        {"id": "persistent-bash", "disabled": True},
        {"id": "persistent-pwsh", "disabled": True},
        {"id": "tool-fs", "disabled": True},
        {"id": "tool-working-directory", "disabled": True},
        {"id": "session-title", "disabled": True},
        {"id": "session-log-deepseek", "config": {"enabled": False}},
        {"insert": [{"id": "campus-tools", "name": plugin.as_uri()}]},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--test-database-url", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--weather-fixture", type=Path,
                        help="Labelled synthetic weather for isolated development replay")
    parser.add_argument("--model", default="deepseek-flash",
                        help="Concrete model ID in the pinned official Harness catalog")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "experiments/datasets/scenarios_dev_v2.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/experiments/harness_agent_pilot.json")
    parser.add_argument("--allow-model-call", action="store_true",
                        help="Required: this pilot makes billable DeepSeek API calls")
    args = parser.parse_args()
    if not args.allow_model_call:
        raise SystemExit("explicit --allow-model-call is required for the development pilot")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("split") != "development":
        raise SystemExit("Harness pilot accepts development cases only")
    case = next((item for item in dataset["cases"] if item.get("id") == args.case_id), None)
    if case is None:
        raise SystemExit("case ID missing from development dataset")
    weather_required = any("weather" in (turn.get("required_outputs") or [])
                           for turn in case.get("expected_turns", []))
    if weather_required and args.weather_fixture is None:
        raise SystemExit("weather case requires --weather-fixture for comparable replay")
    env_file = dotenv_values(args.env_file)
    from tests.db_safety import verify_isolated_test_database
    verify_isolated_test_database(args.test_database_url,
                                  business_urls=(env_file.get("DATABASE_URL"),))
    key = env_file.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DeepSeek credential missing from selected local config")
    os.environ["DATABASE_URL"] = args.test_database_url

    from agents import tools as campus_tools
    from deepseek_harness import DeepSeekHarness
    from experiments.baselines.harness_messages_audit import open_messages_audit
    from experiments.baselines.harness_tool_bridge import open_tool_bridge
    from experiments.baselines.ordinary_tool_agent import SYSTEM_PROMPT
    from experiments.controlled_weather import load_development_weather

    weather_snapshot, weather_provenance = (load_development_weather(args.weather_fixture)
                                            if args.weather_fixture else (None, None))

    upstream = args.upstream.resolve()
    expected_revision = "687f282e8625c6b8d11ff58211a90524be2e4317"
    import subprocess
    observed_revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True,
        text=True, check=True).stdout.strip()
    if observed_revision != expected_revision:
        raise SystemExit("Harness source is not the pinned revision")
    tracked_changes = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True).stdout.strip()
    if tracked_changes:
        raise SystemExit("Harness source has tracked modifications")
    runtime = args.runtime.resolve()
    if not runtime.is_file():
        raise FileNotFoundError(runtime)
    run_dir = args.output.resolve().with_suffix("")
    evidence_path = args.output.resolve().with_name(args.output.stem + "_run_evidence.json")
    if args.output.exists() or evidence_path.exists() or run_dir.exists():
        raise FileExistsError("pilot output already exists; use a new output path")
    run_dir.mkdir(parents=True, exist_ok=True)
    schemas_path = run_dir / "campus_tool_schemas.json"
    patch_path = run_dir / "campus.patch.json"
    if schemas_path.exists() or patch_path.exists():
        raise FileExistsError("pilot run directory already used")
    _write_json(schemas_path, campus_tools.TOOL_SCHEMAS)
    plugin = ROOT / "experiments/baselines/harness_campus_tools.mjs"
    _write_json(patch_path, _patch(plugin))
    model_name = args.model
    # The pinned Harness uses the official Messages transport, while the
    # ordinary Python client uses DeepSeek's OpenAI-compatible /v1 endpoint.
    tool_names = sorted(row["function"]["name"] for row in campus_tools.TOOL_SCHEMAS)
    records: list[dict] = [{
        "kind": "run", "condition": "deepseek_harness",
        "data_snapshot_id": case.get("snapshot_id"),
        "event_snapshot_sha256": None, "event_clock_epoch": None,
        "enabled_tool_names": tool_names, "extra_tool_names": [],
        "max_model_rounds_per_turn": 6,
        "time_budget_s": 40.0,
    }]
    turns = []
    observed_models: set[str] = set()
    with open_tool_bridge() as bridge, open_messages_audit(key) as messages_audit:
        child_env = {
            "CAMPUS_TOOL_SCHEMA_PATH": str(schemas_path),
            "CAMPUS_TOOL_BRIDGE_URL": bridge.url,
            "CAMPUS_TOOL_BRIDGE_TOKEN": bridge.token,
            "DSH_SYSTEM_PROMPT": SYSTEM_PROMPT,
            "DSH_PRIMARY_RUNTIME": "",
            "DSH_TELEMETRY_DISABLED": "1",
            "CAMPUS_MAX_MODEL_REQUESTS_PER_TURN": "6",
            "DATABASE_URL": args.test_database_url,
        }
        with DeepSeekHarness(
                provider="deepseek-official", model=model_name, max_tokens=800,
                cwd=str(run_dir), dsh_home=str(run_dir / "dsh_home"),
                dsh_bin=str(runtime), profile="sdk-minimal",
                patches=(str(patch_path),), env=child_env, api_key=key,
                base_url=messages_audit.url, request_timeout_seconds=40) as harness:
            session = harness.start_session("campus-development-pilot")
            for message in case["messages"]:
                text = message["content"]
                bridge.context = {"query": text}
                if weather_provenance:
                    bridge.context["_route_weather_snapshot"] = weather_snapshot
                before = len(bridge.trace)
                before_artifacts = len(bridge.artifacts)
                started = time.monotonic()
                result = session.run(text)
                trace = bridge.trace[before:]
                artifacts = bridge.artifacts[before_artifacts:]
                selected = next((item for item in reversed(bridge.selection_context[before:])
                                 if item), {})
                weather_observation = next((item for item in reversed(
                    bridge.weather_observations[before:]) if item), None)
                for event in result.events:
                    if event.get("type") == "request/context":
                        observed = event.get("data", {}).get("model")
                        if isinstance(observed, str) and observed:
                            observed_models.add(observed)
                    if event.get("type") != "assistant/message":
                        continue
                    usage = event.get("data", {}).get("usage") or {}
                    records.append(_request_evidence(
                        model_name, usage, _digest(campus_tools.TOOL_SCHEMAS)))
                records.extend({"kind": "tool", "name": row["tool_name"],
                                "outcome": row["outcome"]} for row in trace)
                turns.append({
                    "status": result.finish_reason or "completed",
                    "message": result.final_response,
                    "tool_trace": trace, "artifacts": artifacts,
                    "selected_start_name": selected.get("start_name"),
                    "selected_mode": selected.get("mode"),
                    "weather_observation": weather_observation,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                })
    requests = [record for record in records if record["kind"] == "request"]
    observed_model = next(iter(observed_models)) if len(observed_models) == 1 else None
    attempts = list(messages_audit.attempts)
    complete_transport, resolved_model = _attribute_transport_models(requests, attempts)
    report = {
        "status": "development_pilot_only",
        "case_id": args.case_id, "condition": "deepseek_harness",
        "development_dataset_sha256": _sha256_file(args.dataset),
        "case_messages_sha256": _digest(case["messages"]),
        "baseline_system_prompt_sha256": _digest(SYSTEM_PROMPT),
        "upstream_revision": expected_revision,
        "runtime_sha256": _sha256_file(runtime),
        "plugin_sha256": _sha256_file(plugin),
        "requested_model": model_name,
        "observed_request_model_id": observed_model,
        "resolved_model_id": resolved_model,
        "provider_http_attempts": attempts,
        "provider_http_attempt_count": len(attempts),
        "provider_transport_complete": complete_transport,
        "weather_fixture": weather_provenance,
        "tool_schema_sha256": _digest(campus_tools.TOOL_SCHEMAS),
        "data_component_sha256": {name: _sha256_file(ROOT / name) for name in (
            "data/pois.json", "data/whu_road_network.graphml",
            "data/edge_overrides.json", "data/campus_review_decisions.json")},
        "measured_usage": {"requests": len(requests),
                           "input_tokens": sum(r["input_tokens"] for r in requests),
                           "output_tokens": sum(r["output_tokens"] for r in requests)},
        "tool_calls": len(bridge.trace), "turns": turns,
        "limitation": "transport audit captures actual HTTP attempts and message_start model without recording request content; the development case still lacks independent gold",
    }
    _write_json(evidence_path, records)
    report["run_evidence_path"] = str(evidence_path)
    report["run_evidence_sha256"] = _sha256_file(evidence_path)
    _write_json(args.output.resolve(), report)
    print(json.dumps({"case_id": args.case_id, "status": report["status"],
                      "requests": report["measured_usage"]["requests"],
                      "tool_calls": report["tool_calls"],
                      "turn_statuses": [turn["status"] for turn in turns]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
