"""Run one development case through the real pinned DeepSeek Harness SDK.

The 14 campus tools execute through the same Python registry as the ordinary
baseline. This is a development probe, not a formal matched-budget result:
Harness has not yet been given an enforced per-turn model-round cap.
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
    from experiments.baselines.harness_tool_bridge import open_tool_bridge
    from experiments.baselines.ordinary_tool_agent import SYSTEM_PROMPT

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
    model_name = env_file.get("LLM_MODEL") or "deepseek-chat"
    # The pinned Harness uses the official Messages transport, while the
    # ordinary Python client uses DeepSeek's OpenAI-compatible /v1 endpoint.
    base_url = "https://api.deepseek.com/anthropic"
    tool_names = sorted(row["function"]["name"] for row in campus_tools.TOOL_SCHEMAS)
    records: list[dict] = [{
        "kind": "run", "condition": "deepseek_harness",
        "data_snapshot_id": case.get("snapshot_id"),
        "event_snapshot_sha256": None, "event_clock_epoch": None,
        "enabled_tool_names": tool_names, "extra_tool_names": [],
        "max_model_rounds_per_turn": None,  # Not yet enforced by the pinned SDK.
        "time_budget_s": 40.0,
    }]
    turns = []
    observed_models: set[str] = set()
    with open_tool_bridge() as bridge:
        child_env = {
            "CAMPUS_TOOL_SCHEMA_PATH": str(schemas_path),
            "CAMPUS_TOOL_BRIDGE_URL": bridge.url,
            "CAMPUS_TOOL_BRIDGE_TOKEN": bridge.token,
            "DSH_SYSTEM_PROMPT": SYSTEM_PROMPT,
            "DSH_PRIMARY_RUNTIME": "",
            "DSH_TELEMETRY_DISABLED": "1",
            "DATABASE_URL": args.test_database_url,
        }
        with DeepSeekHarness(
                provider="deepseek-official", model=model_name, max_tokens=800,
                cwd=str(run_dir), dsh_home=str(run_dir / "dsh_home"),
                dsh_bin=str(runtime), profile="sdk-minimal",
                patches=(str(patch_path),), env=child_env, api_key=key,
                base_url=base_url, request_timeout_seconds=40) as harness:
            session = harness.start_session("campus-development-pilot")
            for message in case["messages"]:
                text = message["content"]
                bridge.context = {"query": text}
                before = len(bridge.trace)
                before_artifacts = len(bridge.artifacts)
                started = time.monotonic()
                result = session.run(text)
                trace = bridge.trace[before:]
                artifacts = bridge.artifacts[before_artifacts:]
                for event in result.events:
                    if event.get("type") == "request/context":
                        observed = event.get("data", {}).get("model")
                        if isinstance(observed, str) and observed:
                            observed_models.add(observed)
                    if event.get("type") != "assistant/message":
                        continue
                    usage = event.get("data", {}).get("usage") or {}
                    records.append({
                        "kind": "request", "requested_model_id": model_name,
                        "resolved_model_id": None,  # Filled after event inspection below.
                        "tool_schema_sha256": _digest(campus_tools.TOOL_SCHEMAS),
                        "temperature": None, "max_tokens": 800,
                        "input_tokens": usage.get("inputTokens", 0),
                        "output_tokens": usage.get("outputTokens", 0),
                        "outcome": "success",
                    })
                records.extend({"kind": "tool", "name": row["tool_name"],
                                "outcome": row["outcome"]} for row in trace)
                turns.append({
                    "status": result.finish_reason or "completed",
                    "message": result.final_response,
                    "tool_trace": trace, "artifacts": artifacts,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                })
    requests = [record for record in records if record["kind"] == "request"]
    observed_model = next(iter(observed_models)) if len(observed_models) == 1 else None
    for record in requests:
        record["resolved_model_id"] = observed_model
    report = {
        "status": "development_pilot_only_unmatched_round_budget",
        "case_id": args.case_id, "condition": "deepseek_harness",
        "upstream_revision": expected_revision,
        "runtime_sha256": _sha256_file(runtime),
        "plugin_sha256": _sha256_file(plugin),
        "requested_model": model_name,
        "observed_request_model_id": observed_model,
        "resolved_model_id": None,
        "tool_schema_sha256": _digest(campus_tools.TOOL_SCHEMAS),
        "data_component_sha256": {name: _sha256_file(ROOT / name) for name in (
            "data/pois.json", "data/whu_road_network.graphml",
            "data/edge_overrides.json", "data/campus_review_decisions.json")},
        "measured_usage": {"requests": len(requests),
                           "input_tokens": sum(r["input_tokens"] for r in requests),
                           "output_tokens": sum(r["output_tokens"] for r in requests)},
        "tool_calls": len(bridge.trace), "turns": turns,
        "limitation": "provider response model ID, model-round cap and temperature equality are not yet independently proven",
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
