"""Keyless check of the pinned Harness campus-tool roster and schema parity.

This uses the upstream mock model, never a paid endpoint. It does not claim
task success. The full Harness comparator additionally needs real model run
evidence, matched limits, and independent task scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

from agents import tools as campus_tools
from experiments.baselines.harness_tool_bridge import open_tool_bridge


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True,
                        help="Pinned deepseek-harness checkout")
    parser.add_argument("--runtime", type=Path, required=True,
                        help="Locally built Windows SDK runtime executable")
    parser.add_argument("--exercise-tool", action="store_true",
                        help="Also call local resolve_poi through the real Python executor")
    parser.add_argument("--exercise-budget", action="store_true",
                        help="Force repeated keyless tool calls and check the model-stream cap")
    args = parser.parse_args()
    if args.exercise_tool and args.exercise_budget:
        parser.error("choose one mock exercise")
    upstream = args.upstream.resolve()
    runtime = args.runtime.resolve()
    observed_revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True,
        text=True, check=True).stdout.strip()
    if observed_revision != "687f282e8625c6b8d11ff58211a90524be2e4317":
        raise ValueError("Harness source is not the pinned revision")
    if subprocess.run(["git", "-C", str(upstream), "status", "--porcelain",
                       "--untracked-files=all"], capture_output=True, text=True,
                      check=True).stdout.strip():
        raise ValueError("Harness source checkout is not clean")
    if not runtime.is_file():
        raise FileNotFoundError(runtime)
    source = upstream / "scripts" / "smoke-python-runtime.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    mocks = runpy.run_path(str(source))
    from deepseek_harness import DeepSeekHarness  # Installed only in the separate SDK env.

    plugin = (Path(__file__).resolve().parents[2] / "experiments" / "baselines"
              / "harness_campus_tools.mjs")
    with tempfile.TemporaryDirectory(prefix="campus-harness-roster-") as temp:
        root = Path(temp)
        schemas = root / "schemas.json"
        schemas.write_text(json.dumps(campus_tools.TOOL_SCHEMAS, ensure_ascii=False), encoding="utf-8")
        patch = root / "campus.patch.json"
        patch.write_text(json.dumps([
            {"id": "persistent-bash", "disabled": True},
            {"id": "persistent-pwsh", "disabled": True},
            {"id": "tool-fs", "disabled": True},
            {"id": "tool-working-directory", "disabled": True},
            {"id": "session-title", "disabled": True},
            {"id": "session-log-deepseek", "config": {"enabled": False}},
            {"insert": [{"id": "campus-tools", "name": plugin.as_uri()}]},
        ]), encoding="utf-8")
        with open_tool_bridge() as bridge, mocks["MockModel"]() as model:
            expected_reply = mocks["EXPECTED_TEXT"]
            if args.exercise_budget:
                attempt = 0

                def repeat_tool(_body: dict) -> list[dict]:
                    nonlocal attempt
                    attempt += 1
                    return mocks["tool_call_chunks"](
                        f"campus-budget-{attempt}", "resolve_poi", {"name": "卓尔体育馆"})

                mocks["MockModelHandler"].do_POST.__globals__["completion_chunks"] = repeat_tool
            if args.exercise_tool:
                expected_reply = "CAMPUS_BRIDGE_OK"

                def scripted_completion(_body: dict) -> list[dict]:
                    if bridge.trace:
                        return mocks["text_chunks"](expected_reply)
                    return mocks["tool_call_chunks"](
                        "campus-resolve-smoke", "resolve_poi", {"name": "卓尔体育馆"})

                mocks["MockModelHandler"].do_POST.__globals__["completion_chunks"] = scripted_completion
            env = {
                "CAMPUS_TOOL_SCHEMA_PATH": str(schemas),
                "CAMPUS_TOOL_BRIDGE_URL": bridge.url,
                "CAMPUS_TOOL_BRIDGE_TOKEN": bridge.token,
                "CAMPUS_MAX_MODEL_REQUESTS_PER_TURN": "6",
                "DSH_TELEMETRY_DISABLED": "1",
                "DSH_PRIMARY_RUNTIME": "",
            }
            with DeepSeekHarness(
                    provider="deepseek-official", model="smoke-model",
                    cwd=str(root), dsh_bin=str(runtime), dsh_home=str(root / "home"),
                    profile="sdk-minimal", patches=(str(patch),), env=env,
                    api_key="keyless-campus-roster-smoke", base_url=model.url,
                    request_timeout_seconds=60) as harness:
                result = harness.run("Keyless campus tool schema probe.")
            requests = mocks["MockModelHandler"].requests
            if any(request.get("temperature") != 0 for request in requests):
                raise RuntimeError("Harness model wire requests did not fix temperature=0")
            if args.exercise_budget:
                errors = [event.get("data", {}).get("reason", {}).get("error", {}).get("message")
                          for event in result.events
                          if event.get("type") == "turn/end"]
                if len(requests) != 6 or result.finish_reason != "error" or not any(
                        "campus_model_request_budget_exceeded" in str(error) for error in errors):
                    raise RuntimeError(
                        "keyless budget boundary failed: "
                        f"requests={len(requests)} finish_reason={result.finish_reason!r} "
                        f"turn_errors={errors!r}")
                print(json.dumps({"keyless_budget_smoke": "passed",
                                  "logical_model_requests": len(requests),
                                  "wire_temperature": 0,
                                  "seventh_request_blocked": True,
                                  "tool_calls": len(bridge.trace)}))
                return
            if not requests or result.final_response != expected_reply:
                errors = [event.get("data", {}).get("reason", {}).get("error", {}).get("message")
                          for event in result.events
                          if event.get("type") == "turn/end"]
                raise RuntimeError(
                    "mock Harness request failed: "
                    f"requests={len(requests)} finish_reason={result.finish_reason!r} "
                    f"tool_calls={len(bridge.trace)} "
                    f"event_types={[event.get('type') for event in result.events]!r} "
                    f"turn_errors={errors!r}"
                )
            if args.exercise_tool and (len(bridge.trace) != 1 or
                                       bridge.trace[0]["tool_name"] != "resolve_poi" or
                                       bridge.trace[0]["outcome"] != "success"):
                raise RuntimeError("campus tool did not execute successfully")
            advertised = requests[0].get("tools")
            if not isinstance(advertised, list):
                raise ValueError("Harness did not advertise a tool list")
            expected_names = sorted(row["function"]["name"] for row in campus_tools.TOOL_SCHEMAS)
            observed_names = sorted(row.get("name") for row in advertised)
            normalized = [{"type": "function", "function": {
                "name": row.get("name"), "description": row.get("description"),
                "parameters": row.get("input_schema"),
            }} for row in advertised]
            product_by_name = {row["function"]["name"]: row for row in campus_tools.TOOL_SCHEMAS}
            normalized_by_name = {row["function"]["name"]: row for row in normalized}
            schema_mismatches = [name for name in expected_names
                                 if product_by_name[name] != normalized_by_name.get(name)]
            ordered_visible = [normalized_by_name[row["function"]["name"]]
                               for row in campus_tools.TOOL_SCHEMAS]
            print(json.dumps({"keyless_smoke": "passed", "expected": expected_names,
                              "observed": observed_names,
                              "names_match": expected_names == observed_names,
                              "normalized_schema_mismatches": schema_mismatches,
                              "product_schema_sha256": _digest(campus_tools.TOOL_SCHEMAS),
                              "model_visible_schema_sha256": _digest(advertised),
                              "normalized_visible_schema_sha256": _digest(ordered_visible),
                              "request_count": len(requests),
                              "wire_temperature": 0,
                              "event_types": [event.get("type") for event in result.events],
                              "event_fields": {event.get("type"): sorted(event)
                                               for event in result.events if event.get("type") in
                                               {"request/context", "assistant/message", "step/end", "tool/call"}},
                              "event_data_fields": {event.get("type"): sorted(event.get("data", {}))
                                                    for event in result.events if event.get("type") in
                                                    {"request/context", "assistant/message", "step/end", "tool/call"}},
                              "usage_fields": [event.get("data", {}).get("usage")
                                               for event in result.events
                                               if event.get("type") == "assistant/message"],
                              "executed_campus_tools": [row["tool_name"] for row in bridge.trace]},
                             ensure_ascii=False))
            if expected_names != observed_names or schema_mismatches:
                raise ValueError("Harness tool schemas differ from product")


if __name__ == "__main__":
    main()
