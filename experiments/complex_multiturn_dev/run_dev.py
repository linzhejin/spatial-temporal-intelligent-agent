"""Run exposed development cases through the real local conversation worker.

The model and optional web search use the configured provider. POI lookup uses
the local catalogue and the database is a newly created localhost test database.
No production event is created. Event cases require verified fixtures.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values

from experiments.complex_multiturn_dev.budget import BudgetLedger, BudgetUnavailable
from experiments.complex_multiturn_dev.contracts import assess_turn, read_dataset, validate_dataset, validate_supplement
from experiments.complex_multiturn_dev.event_runtime import activate_event, deactivate_event, load_event_bundle
from experiments.complex_multiturn_dev.observation import project_observation
from experiments.scenario_live_adapter import build_poi_lookup


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "experiments/datasets/complex_multiturn_dev_v1/cases.json"
OUTPUT = ROOT / "output/experiments/complex_multiturn_dev_v1"
EXPECTED_POSTGRES = Path(r"C:\Users\HUAWEI\.codex\test-postgres-runtime\Library\bin\postgres.exe")
EXPECTED_DATA = Path(r"C:\Users\HUAWEI\.codex\test-postgres-data")
MODEL = "deepseek-flash"


def _append(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _safe_provider_evidence(payload):
    """Keep only noncredential POI evidence, never request URLs or keys."""
    if not isinstance(payload, dict):
        return {"status": "invalid_response", "pois": []}
    allowed = ("id", "name", "type", "typecode", "location")
    rows = []
    for row in (payload.get("pois") or [])[:100]:
        if isinstance(row, dict):
            rows.append({key: str(row[key])[:300] for key in allowed if isinstance(row.get(key), str)})
    return {"status": str(payload.get("status") or "")[:10],
            "count": str(payload.get("count") or "")[:20], "pois": rows}


@contextmanager
def _single_process(path):
    """Prevent concurrent workers racing the shared cumulative budget."""
    import msvcrt
    with path.open("a+b") as lock:
        lock.seek(0)
        if lock.read(1) == b"":
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("another development run holds the budget lock") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _create_test_database(business_url):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    check = ("$p = Get-NetTCPConnection -LocalPort 55439 -State Listen | "
             "Where-Object LocalAddress -eq '127.0.0.1' | Select-Object -First 1; "
             "if (-not $p) { exit 2 }; (Get-Process -Id $p.OwningProcess).Path")
    process_path = subprocess.check_output(["powershell", "-NoProfile", "-Command", check], text=True).strip()
    if Path(process_path).resolve() != EXPECTED_POSTGRES.resolve():
        raise RuntimeError("unexpected local PostgreSQL process")
    name = "luojia_multiturn_test_" + uuid4().hex[:16]
    if business_url and conninfo_to_dict(business_url).get("dbname", "").casefold() == name.casefold():
        raise RuntimeError("test and business database names coincide")
    params = dict(host="127.0.0.1", port=55439, dbname="postgres", user="postgres", password="", connect_timeout=5)
    with psycopg.connect(**params, autocommit=True) as conn:
        row = conn.execute("SELECT inet_server_addr()::text, inet_server_port(), current_setting('data_directory')").fetchone()
        if row[0].split("/")[0] != "127.0.0.1" or row[1] != 55439 or Path(row[2]).resolve() != EXPECTED_DATA.resolve():
            raise RuntimeError("unexpected local PostgreSQL cluster")
        if conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,)).fetchone():
            raise RuntimeError("new test database unexpectedly exists")
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    params["dbname"] = name
    url = make_conninfo(**params)
    from tests.db_safety import verify_isolated_test_database
    verify_isolated_test_database(url, business_urls=(business_url,))
    return name, url


def _select_task(tasks, task_id):
    return next((task for task in tasks if task.get("task_id") == task_id), None)


def _guard_client(factory, ledger, journal, budget_state=None):
    def wrapped_factory():
        client = factory()
        if urlsplit(str(client.base_url)).hostname != "api.deepseek.com":
            raise BudgetUnavailable("provider host differs from checked pricing")
        original = client.chat.completions.create

        def guarded_create(**kwargs):
            if kwargs.get("model") != MODEL:
                raise BudgetUnavailable("model differs from checked Flash pricing")
            maximum = kwargs.get("max_tokens")
            request = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            try:
                ident = ledger.reserve(request_bytes=len(request), max_output_tokens=maximum)
            except BudgetUnavailable as exc:
                if budget_state is not None:
                    budget_state["stop_reason"] = str(exc)
                raise
            _append(journal, {"event": "started", "reservation": ident,
                              "request_sha256": hashlib.sha256(request).hexdigest(),
                              "model": MODEL, "request_bytes": len(request), "max_tokens": maximum})
            try:
                response = original(**kwargs)
            except Exception as exc:
                _append(journal, {"event": "provider_error", "reservation": ident,
                                  "error_type": type(exc).__name__, "usage": "unknown"})
                raise
            usage = getattr(response, "usage", None)
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
            cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
            ledger.settle(ident, prompt_tokens=prompt, completion_tokens=completion,
                          cache_hit_tokens=cache_hit, cache_miss_tokens=cache_miss)
            _append(journal, {"event": "returned", "reservation": ident,
                              "actual_model": getattr(response, "model", None),
                              "prompt_tokens": prompt, "completion_tokens": completion,
                              "cache_hit_tokens": cache_hit, "cache_miss_tokens": cache_miss})
            return response

        client.chat.completions.create = guarded_create
        return client
    return wrapped_factory


def _selection_payload(previous, observed):
    candidates = observed.get("candidate_ids") or []
    task_id = previous.get("task_id")
    revision = previous.get("current_task_revision")
    result = previous.get("result") if isinstance(previous.get("result"), dict) else {}
    if not (result.get("response_kind") == "candidates" or
            result.get("response_kind") == "route" and result.get("needs_input") is True):
        raise ValueError("the previous candidate cards were not visible for selection")
    if not candidates or not isinstance(task_id, str) or type(revision) is not int:
        raise ValueError("the previous run did not expose a selectable candidate and task revision")
    chosen = candidates[0]
    return {"continuation_task_id": task_id, "base_revision": revision,
            "selected_poi_id": chosen, "ui_action": {"type": "candidate_selection",
            "task_id": task_id, "task_revision": revision, "poi_id": chosen}}, chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--phase", choices=["core", "variant"], default="core")
    parser.add_argument("--group", choices=["search", "itinerary", "clarification", "events"])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--browser-case", action="append", default=[],
                        help="drive these selected cases through Chromium instead of direct API input")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--budget-limit-cny", type=Decimal, default=Decimal("20"),
                        help="cumulative conservative cap; up to the currently authorized 40 CNY")
    args = parser.parse_args(argv)
    data = read_dataset(args.dataset)
    if data.get("scope") == "development_supplement":
        validate_supplement(data)
    else:
        validate_dataset(data)
    credentials = dotenv_values(args.env_file)
    if not credentials.get("DEEPSEEK_API_KEY"):
        raise SystemExit("configured model key is unavailable")
    provider = credentials.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
    if urlsplit(provider).hostname != "api.deepseek.com":
        raise SystemExit("configured provider is not the verified DeepSeek host")
    wanted = [c for c in data["cases"] if c["phase"] == args.phase
              and (not args.group or c["group"] == args.group)
              and (not args.case or c["id"] in args.case)]
    if not wanted:
        raise SystemExit("no matching cases")
    fixtures = {}
    if any(c["group"] == "events" for c in wanted):
        fixtures = load_event_bundle(ROOT / "experiments/datasets/complex_multiturn_dev_v1/event_fixtures_v2.json",
                                     args.dataset, ROOT / "data/whu_road_network.graphml", ROOT / "data/pois.json")
        if any(c["id"] not in fixtures for c in wanted if c["group"] == "events"):
            raise SystemExit("one or more events lack a verified fixture")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with _single_process(OUTPUT / "budget.lock"):
        return _run(wanted, credentials, provider, args.dataset, fixtures, set(args.browser_case),
                    args.budget_limit_cny)


def _run(cases, credentials, provider, dataset_path, fixtures, browser_cases=frozenset(), budget_limit=Decimal("20")):
    ledger = BudgetLedger(OUTPUT / "budget.jsonl", limit_cny=budget_limit)
    # Fail before creating a database if a previous request has unknown usage.
    _, pending = ledger._read()
    if pending:
        raise SystemExit("previous model request has unconfirmed usage; no new calls allowed")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    out = OUTPUT / run_name
    out.mkdir(parents=True, exist_ok=False)
    db_name, db_url = _create_test_database(credentials.get("DATABASE_URL"))
    condition_file = out / "empty_road_conditions.json"
    condition_file.write_text("[]\n", encoding="utf-8")
    for key in ("DEEPSEEK_API_KEY", "AMAP_KEY", "AMAP_WEB_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        if credentials.get(key):
            os.environ[key] = credentials[key]
    os.environ.update(DATABASE_URL=db_url, TEST_DATABASE_URL=db_url, AGENT_DIALOGUE_MODE="model",
                      OPENAI_BASE_URL=provider, LLM_MODEL=MODEL, AGENT_TOOL_TRACE_ENABLED="1",
                      AMAP_POI_LOOKUP_ENABLED="false", ROAD_CONDITIONS_FILE=str(condition_file))
    from storage import database
    database.initialize(db_url)
    from app import create_app
    from agents import planner, tools, web_evidence
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run
    from langgraph.checkpoint.postgres import PostgresSaver
    from spatial import network, amap_poi

    # No weather-dependent case is in this suite. Freeze absent weather and do
    # not let an unavailable outside service change the tested road graph.
    tools._weather_snapshot = lambda: None
    network._download_network = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("graph download disabled"))
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=db_url)
    pois = json.loads((ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]
    lookup = build_poi_lookup(pois)
    journal = out / "model_requests.jsonl"
    provider_journal = out / "provider_evidence.jsonl"
    original_factory = planner._make_client
    original_web_request = web_evidence._request
    original_map_get = amap_poi._provider_get
    original_conditions = tools.list_conditions
    original_execute_tool = tools.execute_tool
    consumed_conditions = []
    observed_tool_calls = []
    budget_state = {"stop_reason": None}
    current_turn = {"case": None, "turn": None}

    def observed_execute_tool(name, args, context):
        result, artifact = original_execute_tool(name, args, context)
        if name in {"search_poi_candidates", "find_reachable_places", "plan_route", "plan_via_route",
                    "plan_itinerary", "plan_tour", "list_road_conditions", "resolve_poi", "get_place_details"}:
            safe_args = {key: args.get(key) for key in
                         ("name", "poi_id", "start", "end", "via_name", "via_points", "poi_names", "search_anchor", "mode",
                          "activities", "subcategory", "subcategories", "candidate_names", "area", "keyword", "poi_type", "limit",
                          "max_distance_m", "max_time_min", "constraints")
                         if key in args}
            candidates = (result or {}).get("candidates") or []
            observed_tool_calls.append({"tool": name, "args": safe_args,
                "result": {key: (result or {}).get(key) for key in ("error", "reason_code", "status", "count", "online_status", "has_more", "found")},
                "trusted_search_scope": {key: context[key] for key in
                    ("_search_scope_distance_m", "_search_scope_duration_s")
                    if key in context and type(context[key]) in (int, float)},
                "candidates": [{key: row.get(key) for key in ("poi_id", "name", "subcategory", "activities",
                                "network_distance_m", "distance_reference_name", "lat", "lng")}
                               for row in candidates[:20] if isinstance(row, dict)]})
        return result, artifact

    tools.execute_tool = observed_execute_tool

    def observed_conditions(*args, **kwargs):
        rows = original_conditions(*args, **kwargs)
        consumed_conditions.append(sorted(row.get("id") for row in rows if row.get("id")))
        return rows

    tools.list_conditions = observed_conditions
    planner._make_client = _guard_client(original_factory, ledger, journal, budget_state)

    def guarded_web_request(endpoint, payload):
        purpose = {"/search": "search", "/fetch": "fetch"}.get(endpoint)
        if purpose is None:
            raise BudgetUnavailable("unrecognized paid provider endpoint")
        try:
            ident = ledger.reserve_fixed(amount_cny=Decimal("0.005"), purpose=purpose)
        except BudgetUnavailable as exc:
            budget_state["stop_reason"] = str(exc)
            raise
        _append(journal, {"event": "web_started", "reservation": ident, "purpose": purpose})
        try:
            return original_web_request(endpoint, payload)
        finally:
            ledger.settle_fixed(ident)
            _append(journal, {"event": "web_reserved_maximum", "reservation": ident, "purpose": purpose})

    web_evidence._request = guarded_web_request

    def guarded_map_get(client, url, **kwargs):
        parsed = urlsplit(url)
        if parsed.hostname != "restapi.amap.com" or parsed.path not in {
                "/v3/place/around", "/v3/place/text", "/v3/place/detail"}:
            raise BudgetUnavailable("unrecognized map provider endpoint")
        try:
            ident = ledger.reserve_fixed(amount_cny=Decimal("0.003"), purpose="amap_search")
        except BudgetUnavailable as exc:
            budget_state["stop_reason"] = str(exc)
            raise
        _append(journal, {"event": "map_started", "reservation": ident, "path": parsed.path})
        try:
            response = original_map_get(client, url, **kwargs)
            try:
                evidence = _safe_provider_evidence(response.json())
            except (ValueError, TypeError, AttributeError):
                evidence = {"status": "unreadable_response", "pois": []}
            _append(provider_journal, {**current_turn, "path": parsed.path,
                                       "http_status": getattr(response, "status_code", None), **evidence})
            return response
        finally:
            ledger.settle_fixed(ident)
            _append(journal, {"event": "map_reserved_maximum", "reservation": ident})

    amap_poi._provider_get = guarded_map_get
    tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", "agents", "tests"], cwd=ROOT)
    (out / "tracked_source_changes.patch").write_bytes(tracked_diff)
    source_paths = sorted((ROOT / "experiments/complex_multiturn_dev").glob("*.py"))
    source_paths.extend([ROOT / "agents/dialogue.py", ROOT / "agents/dialogue_loop.py",
                         ROOT / "agents/prompts/dialogue_system.txt", ROOT / "static/js/app.js"])
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    manifest = {"scope": read_dataset(dataset_path)["scope"], "formal_test": False, "model": MODEL,
                "production_configured_model": "deepseek-chat", "model_changed_with_user_approval": True,
                "provider_host": "api.deepseek.com", "max_cny": str(budget_limit),
                "pricing_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
                "pricing_checked": "2026-11-13", "price_policy": "peak rates; provider-reported cache hits; unknown cache treated as misses",
                "database_name": db_name, "database_isolated": True,
                "poi_source": "local_catalogue_and_live_amap_search",
                "map_pricing_source": "https://lbs.amap.com/upgrade",
                "weather": "no weather input (suite excludes weather tasks)",
                "case_ids": [case["id"] for case in cases], "browser_cases": sorted(browser_cases),
                "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "graph_sha256": hashlib.sha256((ROOT / "data/whu_road_network.graphml").read_bytes()).hexdigest(),
                "working_tree_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
                "source_sha256": source_hashes,
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = out / "turns.jsonl"
    completed = []
    browser = None
    try:
        with PostgresSaver.from_conn_string(db_url) as saver:
            saver.setup()
            workflow = build_agent_workflow(saver, database_url=db_url)
            if browser_cases:
                from experiments.complex_multiturn_dev.browser import BrowserCases
                browser = BrowserCases(ROOT, db_url, workflow)
            for case in cases:
                deactivate_event(condition_file)
                client = app.test_client()
                ui = browser.new_case(client) if case["id"] in browser_cases else None
                conversation = None
                if not ui:
                    response = client.post("/api/conversations", json={})
                    if response.status_code != 201:
                        raise RuntimeError("could not create isolated conversation")
                    conversation = response.get_json()["data"]["conversation_id"]
                previous, observed, last_bound_task = None, {}, None
                for index, turn in enumerate(case["turns"], 1):
                    current_turn.update(case=case["id"], turn=index)
                    consumed_conditions.clear()
                    observed_tool_calls.clear()
                    event_proof = None
                    if case["group"] == "events":
                        event_proof = activate_event(fixtures[case["id"]], turn["event_stage"],
                                                     run_id=run_name, turn_number=index)
                    selected = None
                    body = {"query": turn["user"], "request_id": uuid4().hex}
                    if turn.get("action") == "select_first_candidate":
                        try:
                            extra, selected = _selection_payload(previous or {}, observed)
                        except ValueError as error:
                            _append(records, {"case": case["id"], "turn": index, "status": "not_run",
                                              "reason": str(error), "user": turn["user"]})
                            break
                        body.update(extra)
                    browser_evidence, actual_body = None, body
                    if ui:
                        previous, actual_body, browser_evidence = ui.turn(turn["user"], selected=selected)
                        conversation = ui.conversation
                        run_id = previous["run_id"]
                    else:
                        response = client.post(f"/api/conversations/{conversation}/messages", json=body)
                        if response.status_code != 202:
                            _append(records, {"case": case["id"], "turn": index, "status": "transport_error",
                                              "http_status": response.status_code, "user": turn["user"]})
                            break
                        run_id = response.get_json()["data"]["run_id"]
                        outcome = process_next_run(db_url, "complex-dev-" + uuid4().hex[:8], workflow, lease_seconds=75)
                        response = client.get(f"/api/runs/{run_id}?conversation_id={conversation}")
                        if response.status_code != 200 or not outcome or outcome.get("run_id") != run_id:
                            raise RuntimeError("run could not be collected")
                        previous = response.get_json()["data"]
                    restored = client.get(f"/api/conversations/{conversation}")
                    if restored.status_code != 200:
                        raise RuntimeError("conversation could not be restored")
                    last_bound_task = previous.get("task_id") or last_bound_task
                    task = _select_task(restored.get_json()["data"].get("tasks", []), last_bound_task)
                    observed = project_observation(previous, task, lookup, selected_first=selected)
                    if (event_proof and consumed_conditions
                            and all(row == event_proof["active_ids"] for row in consumed_conditions)):
                        observed["event_state"] = event_proof["stage"]
                    score = assess_turn(turn, observed)
                    # Route edges and candidate semantics require independent
                    # evidence; field matches alone cannot certify delivery.
                    if score["verdict"] == "pass":
                        score["verdict"] = "unknown"
                        score["unknowns"].append("delivery_independent_verification_pending")
                    _append(records, {"case": case["id"], "turn": index, "user": turn["user"],
                                      "run_id": run_id, "task_id": previous.get("task_id"),
                                      "observed_task_id": last_bound_task,
                                      "task_revision": previous.get("current_task_revision"),
                                      "observed": observed, "score": score,
                                      "event_proof": event_proof,
                                      "consumed_event_ids": list(consumed_conditions),
                                      "tool_observations": list(observed_tool_calls),
                                      "browser_evidence": browser_evidence,
                                      "actual_user_input": actual_body.get("query"),
                                      "budget_stop_reason": budget_state["stop_reason"],
                                      "result": previous.get("result"), "events": previous.get("events", [])})
                    print(json.dumps({"case": case["id"], "turn": index, "status": previous.get("status"),
                                      "verdict": score["verdict"], "failures": score["failures"],
                                      "reserved_or_spent_cny": str(ledger.spent_or_reserved())},
                                     ensure_ascii=False), flush=True)
                    if ledger._read()[1] or budget_state["stop_reason"]:
                        print("STOP: budget unavailable or provider usage unconfirmed; remaining requests not sent", flush=True)
                        return 2
                if ui:
                    ui.close(out / (case["id"] + "_browser.png"))
                completed.append(case["id"])
    finally:
        if browser:
            browser.close()
        planner._make_client = original_factory
        web_evidence._request = original_web_request
        amap_poi._provider_get = original_map_get
        tools.list_conditions = original_conditions
        tools.execute_tool = original_execute_tool
    print(json.dumps({"output": str(out), "cases_attempted": completed,
                      "reserved_or_spent_cny": str(ledger.spent_or_reserved())}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
