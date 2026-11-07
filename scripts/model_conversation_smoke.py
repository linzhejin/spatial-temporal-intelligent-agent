"""Real-provider conversation acceptance through the browser and isolated worker.

Synthetic user turns go through the public conversation API from a real mobile
browser, run through the normal worker, and are checked in both persisted chat
and rendered UI. Request journals omit credentials and model reasoning content;
never point this script at a business database.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
from uuid import uuid4
from urllib.parse import urlsplit

from dotenv import dotenv_values

EXPECTED_START_POI_ID = "poi_240"
EXPECTED_SEARCH_ANCHOR_POI_ID = "poi_015"

_PROVIDER_ENV = dotenv_values(os.environ.get("LUOJIA_PROVIDER_ENV", "/nonexistent"))
_BUSINESS_DATABASE_URL = os.environ.get("DATABASE_URL") or _PROVIDER_ENV.get("DATABASE_URL")
for name, value in _PROVIDER_ENV.items():
    if name in {"DEEPSEEK_API_KEY", "AMAP_KEY", "AMAP_WEB_KEY", "LLM_MODEL", "OPENAI_BASE_URL",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"} and value:
        os.environ[name] = value
os.environ["AGENT_DIALOGUE_MODE"] = "model"

import httpx
from langgraph.checkpoint.postgres import PostgresSaver
from playwright.sync_api import expect, sync_playwright
from agents import planner, workflow
from agents.dialogue import Dialogue
from jobs.worker import process_next_run
from storage import database
from storage.task_repository import get_run_input
from tests.db_safety import verify_isolated_test_database


def _safe_origin(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.hostname or ''}" + (f":{parsed.port}" if parsed.port else "")


def _safe_proxy_origin(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value if "://" in value else "http://" + value)
    return f"{parsed.scheme}://{parsed.hostname or ''}" + (f":{parsed.port}" if parsed.port else "")


def _safe_request_id(error) -> str | None:
    request_id = getattr(error, "request_id", None) or getattr(error, "_request_id", None)
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not request_id and headers is not None:
        request_id = headers.get("x-request-id") or headers.get("request-id")
    request_id = str(request_id) if request_id is not None else None
    return request_id if request_id and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id) else None


def _endpoint_record(ref, dialogue, local_places):
    if not isinstance(ref, dict):
        return {}
    identifier = ref.get("poi_id") or ref.get("id")
    row = dialogue.known.get(identifier) if identifier else None
    if row is None and identifier:
        row = local_places.get(identifier)
    if row is None and isinstance(ref.get("name"), str):
        match, alternatives = planner.agent_tools.find_poi_ambiguous(ref["name"])
        if match is not None and not alternatives:
            row = match
            identifier = match.get("poi_id") or match.get("id")
    if not identifier and isinstance(row, dict):
        identifier = row.get("poi_id") or row.get("id")
    coordinate = (row or {}).get("coordinates") if isinstance(row, dict) else None
    if not isinstance(coordinate, dict) and isinstance(row, dict):
        coordinate = {"lng": row.get("lng", row.get("lon")), "lat": row.get("lat")}
    point = {}
    if isinstance(coordinate, dict):
        lng = coordinate.get("lng", coordinate.get("longitude"))
        lat = coordinate.get("lat", coordinate.get("latitude"))
        if isinstance(lng, (int, float)) and isinstance(lat, (int, float)):
            point = {"lng": float(lng), "lat": float(lat),
                     "crs": coordinate.get("crs") or (row or {}).get("coordinate_system")}
    return {"poi_id": identifier, "name": (row or {}).get("name") or ref.get("name"),
            "coordinates": point}


def _same_endpoint(first, other, *, tolerance_m=25):
    if not first or not other or not first.get("poi_id") or first["poi_id"] != other.get("poi_id"):
        return False
    a, b = first.get("coordinates") or {}, other.get("coordinates") or {}
    if not all(key in a and key in b for key in ("lng", "lat")):
        return False
    dy = math.radians(float(a["lat"]) - float(b["lat"]))
    dx = math.radians(float(a["lng"]) - float(b["lng"])) * math.cos(math.radians(float(a["lat"])))
    return math.hypot(dx, dy) * 6_371_000 <= tolerance_m


def _point_record(row):
    if not isinstance(row, dict):
        return {}
    coordinate = row.get("coordinates")
    coordinate = coordinate if isinstance(coordinate, dict) else row
    lng = coordinate.get("lng", coordinate.get("longitude", coordinate.get("lon")))
    lat = coordinate.get("lat", coordinate.get("latitude"))
    if (not isinstance(lng, (int, float)) or isinstance(lng, bool)
            or not isinstance(lat, (int, float)) or isinstance(lat, bool)):
        return {}
    return {"lng": float(lng), "lat": float(lat),
            "crs": coordinate.get("crs") or row.get("coordinate_system")}


def _accepted_model_messages(accepted_history, current_query):
    return [
        {"role": item["role"], "content": item["content"]}
        for item in [*(accepted_history or []), {"role": "user", "content": current_query}]
        if (isinstance(item, dict) and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str))
    ]


def _failure_frames(error):
    return [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)[-8:]]


def _has_restaurant_intent(place_request):
    activities = set(place_request.get("activities") or [])
    subcategories = set(place_request.get("subcategories") or [])
    return bool(activities & {"restaurant_meal", "fast_food"}
                or subcategories & {"restaurant", "fastfood"})


def _accepted_history_present(model_messages, accepted_history, current_query):
    expected = _accepted_model_messages(accepted_history, current_query)
    actual = [{"role": item.get("role"), "content": item.get("content")}
              for item in model_messages or [] if isinstance(item, dict)
              and item.get("role") in {"user", "assistant"}
              and isinstance(item.get("content"), str)]
    position = 0
    for message in expected:
        match = next((index for index in range(position, len(actual)) if actual[index] == message), None)
        if match is None:
            return False
        position = match + 1
    return bool(expected)


def _check_user_copy(record, *, cards=()):
    text = record.get("assistant_text") or ""
    internal = ("来源与位置", "记录来源", "查询时间", "候选池", "search_poi_candidates",
                "model_dialogue", "路网和路线约束", "未实地确认道路当前可通行",
                "来源编号", "参考来源", "参考链接", "目录匹配", "处理阶段",
                "**", "http://", "https://")
    markdown = re.search(r"(?:^|\n)\s*(?:#{1,6}\s|>\s|[-+*]\s|\d+[.)]\s)|`|\[[^\]]+\]\(", text)
    return (len(text) <= 100 and not re.search(r"\bS[0-9a-f]{8}\b", text, re.IGNORECASE)
            and not markdown and not any(marker in text for marker in internal)
            and not any(card.get("name") and card["name"] in text for card in cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        parser.error("TEST_DATABASE_URL must point to a dedicated isolated test database")
    verify_isolated_test_database(url, business_urls=(_BUSINESS_DATABASE_URL,))
    os.environ["DATABASE_URL"] = url
    database.initialize(url)

    from app import create_app

    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=url)
    resources = ExitStack()
    saver = resources.enter_context(PostgresSaver.from_conn_string(url))
    saver.setup()
    graph = workflow.build_agent_workflow(saver, database_url=url)
    factory = planner._make_client
    original_present, original_spatial, original_endpoint = Dialogue.present, Dialogue.spatial, Dialogue._endpoint
    local_places = {str(row.get("id") or row.get("poi_id")): row for row in planner.agent_tools.load_pois()
                    if row.get("id") or row.get("poi_id")}
    output = Path(os.environ.get("LUOJIA_SMOKE_OUTPUT_DIR") or
                  Path(__file__).resolve().parents[1] / "output/experiments/model_conversation_acceptance")
    report_dir = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") / uuid4().hex
    report_dir.mkdir(parents=True, exist_ok=False)
    request_journal = report_dir / "model_requests.jsonl"
    report_path = report_dir / "acceptance.json"
    global_record = {"scope": "real_provider_api_worker_acceptance", "database_isolated": True,
                     "provider": {"model": planner.config.LLM_MODEL,
                                  "base_url_origin": _safe_origin(planner.config.OPENAI_BASE_URL),
                                  "openai_sdk_version": importlib.metadata.version("openai"),
                                  "httpx_version": httpx.__version__, "max_retries": 0,
                                  "proxy_origins": {name: _safe_proxy_origin(os.environ.get(name))
                                                    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")
                                                    if os.environ.get(name)},
                                  "call_budget_per_session": 12},
                     "sessions": []}
    tracker = {"active_tool": {}, "endpoint_observations": [], "search_snapshots": [],
               "presentation_errors": [], "call_count": 0, "last_failure_fingerprint": None,
               "same_request_failures": 0, "map_resource_responses": []}
    static = Path(__file__).resolve().parents[1] / "static"

    class QuietStaticHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    static_server = ThreadingHTTPServer(("127.0.0.1", 0),
                                        partial(QuietStaticHandler, directory=str(static)))
    static_thread = threading.Thread(target=static_server.serve_forever, daemon=True)
    static_thread.start()

    def monitored_present(self, args):
        try:
            return original_present(self, args)
        except ValueError as error:
            tracker["presentation_errors"].append(str(error).split(":", 1)[0])
            raise

    def monitored_spatial(self, name, args):
        tracker["active_tool"][id(self)] = name
        before = len(tracker["endpoint_observations"])
        try:
            result = original_spatial(self, name, args)
        finally:
            tracker["active_tool"].pop(id(self), None)
        if name in {"search_poi_candidates", "find_reachable_places"}:
            tracker["search_snapshots"].append({"tool": name,
                "radius_m": (self.pending or {}).get("search_radius_m"),
                "candidate_ids": [row.get("poi_id") for row in result.get("candidates") or []],
                "error": result.get("error")})
        return result

    def monitored_endpoint(self, ref, *, allow_pending=True):
        resolved = original_endpoint(self, ref, allow_pending=allow_pending)
        name = tracker["active_tool"].get(id(self))
        if name:
            pending_start = self.pending.get("start")
            pending_anchor = self.pending.get("search_anchor")
            selected_id = self.pending.get("selected_poi_id")
            resolved_id = _endpoint_record(resolved, self, local_places).get("poi_id")
            start_id = _endpoint_record(pending_start, self, local_places).get("poi_id")
            anchor_id = _endpoint_record(pending_anchor, self, local_places).get("poi_id")
            if name in {"search_poi_candidates", "find_reachable_places"}:
                role = ("search_anchor" if anchor_id and resolved_id == anchor_id else
                        "start" if start_id and resolved_id == start_id else "endpoint")
            elif name in {"plan_route", "plan_via_route", "plan_itinerary", "plan_multimodal_route"}:
                role = ("start" if start_id and resolved_id == start_id else
                        "selected_destination" if resolved_id and resolved_id == selected_id else "endpoint")
            else:
                role = "endpoint"
            if role != "endpoint":
                tracker["endpoint_observations"].append({"tool": name, "role": role,
                    **_endpoint_record(resolved, self, local_places)})
        return resolved

    Dialogue.present, Dialogue.spatial, Dialogue._endpoint = monitored_present, monitored_spatial, monitored_endpoint

    try:
        scenarios = [
            {"id": "original", "queries": [
                "我从信图出发，在总图附近找餐馆，别推荐食堂档口",
                "再换几家看看", "范围放宽些，其他要求都保持", "去这家",
                "刚才这条路线改成尽量平坦，还是从原来的地方出发",
                "先不改这个行程，帮我看看今天的天气。",
                "回到刚才去那家店的路线，走最短路，起点不要动。",
            ]},
            {"id": "paraphrase", "queries": [
                "我人现在在信图，午饭想在总馆周围吃口热乎的，别给我食堂窗口。",
                "再挑几间不重样的", "把搜索圈往外扩一圈，别的要求照旧", "走到卡片这家",
                "这条路线能不能尽量走平坦些，出发点仍是信图",
                "武汉今天啥天气？",
                "回来继续刚才去店里的路线，我想走最短的，还是从信图出发。",
            ]},
        ]

        def run_session(scenario, browser):
            tracker.update({"endpoint_observations": [], "search_snapshots": [], "presentation_errors": [],
                            "call_count": 0, "last_failure_fingerprint": None, "same_request_failures": 0,
                            "conversation_id": None, "browser_errors": [], "map_resource_responses": [],
                            "map_resource_requests": [], "last_turn_summary": None})
            client = app.test_client()
            captured = []
            submissions = []

            def monitored_client():
                client_instance = factory()
                original_create = client_instance.chat.completions.create

                def create(**kwargs):
                    request_bytes = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
                    fingerprint = sha256(request_bytes).hexdigest()
                    if tracker["call_count"] >= 12:
                        from agents.dialogue_loop import DiagnosticCallBudgetExceeded
                        raise DiagnosticCallBudgetExceeded("diagnostic call budget exhausted")
                    if (tracker["last_failure_fingerprint"] == fingerprint
                            and tracker["same_request_failures"] >= 3):
                        from agents.dialogue_loop import DiagnosticCallBudgetExceeded
                        raise DiagnosticCallBudgetExceeded("same provider request failed three times")
                    tracker["call_count"] += 1
                    index = tracker["call_count"]
                    started_at = time.monotonic()
                    safe_messages = [{key: deepcopy(value) for key, value in item.items()
                                      if key != "reasoning_content"}
                                     for item in kwargs.get("messages", []) if isinstance(item, dict)]
                    captured.append({"messages": deepcopy(kwargs.get("messages") or []), "tools": kwargs.get("tools"),
                                     "model": kwargs.get("model"), "temperature": kwargs.get("temperature"),
                                     "max_tokens": kwargs.get("max_tokens"), "timeout": kwargs.get("timeout")})
                    entry = {"session": scenario["id"], "call": index, "status": "started",
                             "request_sha256": fingerprint, "model": kwargs.get("model"),
                             "base_url_origin": _safe_origin(str(client_instance.base_url)),
                             "temperature": kwargs.get("temperature"), "max_tokens": kwargs.get("max_tokens"),
                             "timeout_seconds": kwargs.get("timeout"), "message_count": len(safe_messages),
                             "role_counts": {role: sum(item.get("role") == role for item in safe_messages)
                                             for role in ("system", "user", "assistant", "tool")},
                             "tools": deepcopy(kwargs.get("tools") or []), "messages": safe_messages}
                    with request_journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                    try:
                        response = original_create(**kwargs)
                    except Exception as error:
                        status = getattr(error, "status_code", None) or getattr(
                            getattr(error, "response", None), "status_code", None)
                        same = tracker["last_failure_fingerprint"] == fingerprint
                        tracker["same_request_failures"] = tracker["same_request_failures"] + 1 if same else 1
                        tracker["last_failure_fingerprint"] = fingerprint
                        failure = {"session": scenario["id"], "call": index, "status": "failed",
                                   "http_status": status if isinstance(status, int) else None,
                                   "request_id": _safe_request_id(error),
                                   "duration_ms": round((time.monotonic() - started_at) * 1000),
                                   "error_code": (str(getattr(error, "code", ""))
                                                  if re.fullmatch(r"[A-Za-z0-9_-]{1,80}",
                                                                  str(getattr(error, "code", "")))
                                                  else type(error).__name__),
                                   "same_request_failure_count": tracker["same_request_failures"]}
                        with request_journal.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(failure, ensure_ascii=True) + "\n")
                        raise
                    tracker["last_failure_fingerprint"] = None
                    tracker["same_request_failures"] = 0
                    usage = getattr(response, "usage", None)
                    choice = (getattr(response, "choices", None) or [None])[0]
                    message = getattr(choice, "message", None)
                    tool_call_names = [call.function.name for call in
                                       (getattr(message, "tool_calls", None) or [])
                                       if getattr(call, "function", None)]
                    terminal_decisions = []
                    for call in getattr(message, "tool_calls", None) or []:
                        function = getattr(call, "function", None)
                        if not function or function.name not in {"get_weather", "plan_route"}:
                            continue
                        try:
                            arguments = json.loads(function.arguments or "{}")
                        except (TypeError, ValueError):
                            arguments = {}
                        terminal_decisions.append({"tool": function.name,
                            "end_turn_if_success": arguments.get("end_turn_if_success")
                            if type(arguments.get("end_turn_if_success")) is bool else None})
                    captured[-1]["terminal_decisions"] = terminal_decisions
                    returned = {"session": scenario["id"], "call": index, "status": "returned",
                                "duration_ms": round((time.monotonic() - started_at) * 1000),
                                 "request_id": _safe_request_id(response),
                                "provider_model": getattr(response, "model", None),
                                "tool_call_names": tool_call_names,
                                 "terminal_decisions": terminal_decisions,
                                "input_tokens": getattr(usage, "prompt_tokens", None),
                                "output_tokens": getattr(usage, "completion_tokens", None)}
                    with request_journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(returned, ensure_ascii=True) + "\n")
                    return response

                client_instance.chat.completions.create = create
                return client_instance

            planner._make_client = monitored_client

            def browser_api_bridge(route):
                request = route.request
                parsed = urlsplit(request.url)
                if parsed.hostname not in {"127.0.0.1", "localhost"}:
                    host = (parsed.hostname or "").lower()
                    if host == "webapi.amap.com":
                        tracker["map_resource_requests"].append("amap_sdk")
                        route.continue_()
                    elif host.endswith(".is.autonavi.com"):
                        tracker["map_resource_requests"].append("map_tile")
                        route.continue_()
                    else:
                        route.abort()
                    return
                if not parsed.path.startswith("/api/"):
                    route.continue_()
                    return
                try:
                    response = client.open(parsed.path + ("?" + parsed.query if parsed.query else ""),
                                           method=request.method, data=request.post_data,
                                           content_type=request.headers.get("content-type", "application/json"))
                    payload = response.get_json(silent=True) or {}
                    if parsed.path == "/api/conversations" and request.method == "POST" and response.status_code == 201:
                        tracker["conversation_id"] = (payload.get("data") or {}).get("conversation_id")
                    if parsed.path.endswith("/messages") and request.method == "POST" and response.status_code == 202:
                        accepted = payload.get("data") or {}
                        run_id = accepted.get("run_id")
                        if not run_id:
                            raise RuntimeError("message API accepted without a run id")
                        body = request.post_data_json or {}
                        input_snapshot = get_run_input(url, run_id)
                        submission = {"run_id": run_id, "query": body.get("query"),
                                      "history": deepcopy(input_snapshot.get("history") or []),
                                      "request_body_keys": sorted(body)}
                        submissions.append(submission)
                        outcome = process_next_run(url, "acceptance-" + uuid4().hex, graph, lease_seconds=75)
                        status_response = client.get(
                            f"/api/runs/{run_id}?conversation_id={tracker['conversation_id']}")
                        if status_response.status_code != 200:
                            raise RuntimeError("worker run could not be fetched through the conversation API")
                        run = status_response.get_json()["data"]
                        transcript_response = client.get(f"/api/conversations/{tracker['conversation_id']}")
                        transcript = ((transcript_response.get_json() or {}).get("data") or {}).get("messages", [])
                        final_assistant = next((item for item in reversed(transcript)
                                                if item.get("role") == "assistant"), {})
                        submission.update({"worker_status": outcome and outcome.get("status"),
                                           "run": deepcopy(run),
                                           "persisted_assistant_text": final_assistant.get("content")})
                    headers = {key.lower(): value for key, value in response.headers.items()
                               if key.lower() in {"content-type", "set-cookie", "cache-control"}}
                    route.fulfill(status=response.status_code, headers=headers, body=response.get_data())
                except Exception as error:
                    tracker["browser_errors"].append(type(error).__name__)
                    route.fulfill(status=500, content_type="application/json",
                                  body=json.dumps({"error": "acceptance_bridge_failed"}))

            context = browser.new_context(viewport={"width": 390, "height": 844},
                                          is_mobile=True, has_touch=True, service_workers="block")
            context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
            page = context.new_page()
            page.on("pageerror", lambda error: tracker["browser_errors"].append("page_error"))
            def observe_map_response(response):
                host = (urlsplit(response.url).hostname or "").lower()
                if host == "webapi.amap.com":
                    kind = "amap_sdk"
                elif host.endswith(".is.autonavi.com"):
                    kind = "map_tile"
                else:
                    return
                tracker["map_resource_responses"].append({"kind": kind, "status": response.status,
                                                          "ok": response.ok})
            tracker["map_resource_requests"] = []
            page.on("response", observe_map_response)
            page.route("**/*", browser_api_bridge)
            page.goto(f"http://127.0.0.1:{static_server.server_port}/index.html",
                      wait_until="domcontentloaded")

            def turn(query, *, select_poi_id=None, expected_kind=None):
                started_at = time.monotonic()
                submission_count = len(submissions)
                call_start = len(captured)
                endpoint_start = len(tracker["endpoint_observations"])
                search_start = len(tracker["search_snapshots"])
                with page.expect_response(
                        lambda item: item.request.method == "POST" and item.url.endswith("/messages"),
                        timeout=90000) as accepted_response_info:
                    if select_poi_id:
                        group = page.locator(".candidate-cards").last
                        selector = ".candidate-card[data-poi-id=" + json.dumps(select_poi_id) + "]"
                        card = group.locator(selector)
                        expect(card).to_be_visible(timeout=10000)
                        card.click()
                    else:
                        page.locator("#nl-input").fill(query)
                        page.locator("#submit-btn").click()
                accepted_response = accepted_response_info.value
                if accepted_response.status != 202:
                    raise RuntimeError("browser message API rejected the acceptance turn")
                if len(submissions) != submission_count + 1:
                    raise RuntimeError("browser message submission was not observed by the worker bridge")
                submission = submissions[-1]
                run = submission["run"]
                result = run.get("result") or {}
                turn_calls = captured[call_start:]
                terminal_decisions = [decision for call in turn_calls
                                      for decision in call.get("terminal_decisions", [])]
                actual_query = submission.get("query") or query
                all_accepted_messages_received = bool(turn_calls) and all(
                    _accepted_history_present(call["messages"], submission["history"], actual_query)
                    for call in turn_calls
                )
                pending = result.get("place_request") or result.get("completed_place_request") or {}
                events = [event.get("payload") or {} for event in run.get("events", [])
                          if event.get("event_type") == "agent_tool_call"
                          and (event.get("payload") or {}).get("phase") == "completed"]
                assistant_text = result.get("assistant_text") or result.get("message") or result.get("reply") or ""
                cards = result.get("candidates") or []
                tracker["last_turn_summary"] = {
                    "run_status": run.get("status"),
                    "worker_status": submission.get("worker_status"),
                    "response_kind": result.get("response_kind"),
                    "failure_code": result.get("failure_code"),
                    "candidate_count": len(cards),
                    "rendered_card_count": page.locator(".candidate-card").count(),
                    "assistant_text_length": len(assistant_text),
                    "history_count": len(_accepted_model_messages(submission["history"], actual_query)),
                }
                if assistant_text:
                    expect(page.locator("#chat-content")).to_contain_text(assistant_text, timeout=15000)
                browser_candidate_ids = []
                browser_candidate_text = ""
                candidate_map_identities = []
                candidate_map_matches_cards = expected_kind != "candidates"
                if expected_kind == "candidates":
                    latest_group = page.locator(".candidate-cards").last
                    expect(latest_group.locator(".candidate-card").first).to_be_visible(timeout=15000)
                    if len(submissions) == 1:
                        page.wait_for_function(
                            "Array.from(document.querySelectorAll('#map-container img.leaflet-tile'))"
                            ".some(tile => tile.complete && tile.naturalWidth > 0)", timeout=20000)
                    browser_candidate_ids = latest_group.locator(".candidate-card").evaluate_all(
                        "cards => cards.map(card => card.dataset.poiId).filter(Boolean)")
                    browser_candidate_text = latest_group.inner_text()
                    candidate_map_identities = page.locator(
                        ".leaflet-marker-icon[data-poi-identities]"
                    ).evaluate_all("markers => markers.flatMap(marker => "
                                   "JSON.parse(marker.getAttribute('data-poi-identities') || '[]'))")
                elif expected_kind == "route":
                    expect(page.locator("#results-section")).to_be_visible(timeout=15000)
                route_map_visible = page.locator(
                    "#map-container .leaflet-overlay-pane path.leaflet-interactive"
                ).count() > 0
                cards_by_id = {item.get("poi_id"): item for item in cards if item.get("poi_id")}
                if expected_kind == "candidates":
                    map_by_id = {item.get("poi_id"): item for item in candidate_map_identities
                                 if item.get("poi_id")}
                    candidate_map_matches_cards = bool(cards_by_id) and set(map_by_id) == set(cards_by_id)
                    for identifier, card_data in cards_by_id.items():
                        point = _point_record(card_data)
                        if not point:
                            candidate_map_matches_cards = False
                            break
                        mapped = map_by_id.get(identifier) or {}
                        if (abs(float(mapped.get("lng", float("inf"))) - point["lng"]) > 1e-6
                                or abs(float(mapped.get("lat", float("inf"))) - point["lat"]) > 1e-6):
                            candidate_map_matches_cards = False
                            break
                loaded_map_tiles = page.locator("#map-container img.leaflet-tile").evaluate_all(
                    "tiles => tiles.filter(tile => tile.complete && tile.naturalWidth > 0).length")
                candidate_card_presentation_ok = expected_kind != "candidates" or (
                    bool(browser_candidate_ids)
                    and set(browser_candidate_ids) == set(cards_by_id)
                    and len(browser_candidate_ids) == len(cards_by_id)
                            and all(isinstance(cards_by_id[identifier].get("distance_m"), (int, float))
                            and not isinstance(cards_by_id[identifier].get("distance_m"), bool)
                            and bool(cards_by_id[identifier].get("distance_reference_name"))
                            and ("距" + cards_by_id[identifier]["distance_reference_name"])
                            in browser_candidate_text
                            and (("约 " + str(math.floor(cards_by_id[identifier]["distance_m"] + 0.5)) + " 米")
                                 if cards_by_id[identifier]["distance_m"] < 1000
                                 else ("约 " + format(cards_by_id[identifier]["distance_m"] / 1000, ".1f")
                                       + " 公里")) in browser_candidate_text
                            for identifier in browser_candidate_ids)
                    and not any(marker in browser_candidate_text
                                for marker in ("来源与位置", "查询时间", "目录匹配")))
                latest_bubble = page.locator("#chat-content .chat-bubble-text").last
                browser_bubble = latest_bubble.inner_text() if latest_bubble.count() else ""
                browser_chat = page.locator("#chat-content").inner_text()
                record = {"session": scenario["id"], "query": actual_query, "run_id": run["run_id"],
                          "run_status": run.get("status"), "worker_status": submission["worker_status"],
                          "response_kind": result.get("response_kind"), "failure_code": result.get("failure_code"),
                          "model_calls_this_turn": len(turn_calls), "model_calls_session": tracker["call_count"],
                          "terminal_decisions": terminal_decisions,
                          "history_count": len(_accepted_model_messages(submission["history"], actual_query)),
                          "all_accepted_messages_received": all_accepted_messages_received,
                          "assistant_text": assistant_text, "assistant_text_length": len(assistant_text),
                          "assistant_text_matches_persisted": submission.get("persisted_assistant_text") == assistant_text,
                          "assistant_text_rendered_in_browser": assistant_text in browser_chat,
                          "latest_bubble_matches_result": assistant_text in browser_bubble if assistant_text else False,
                          "browser_candidate_card_ids": browser_candidate_ids,
                          "browser_candidate_card_text": browser_candidate_text,
                          "candidate_card_presentation_matches_result": candidate_card_presentation_ok,
                          "candidate_map_poi_ids": [item.get("poi_id") for item in candidate_map_identities],
                          "candidate_map_coordinates_match_cards": candidate_map_matches_cards,
                          "route_line_visible_on_map": route_map_visible,
                          "loaded_real_map_tile_count": loaded_map_tiles,
                          "map_resource_responses": deepcopy(tracker["map_resource_responses"]),
                          "short_clean_user_copy": _check_user_copy({"assistant_text": assistant_text}, cards=cards),
                          "task_id": run.get("task_id"), "task_revision": run.get("current_task_revision"),
                          "start": deepcopy(pending.get("start")), "search_anchor": deepcopy(pending.get("search_anchor")),
                          "search_radius_m": pending.get("search_radius_m"), "activities": pending.get("activities") or [],
                          "subcategories": pending.get("subcategories") or [],
                          "food_intent_present": _has_restaurant_intent(pending),
                          "selected_poi_id": pending.get("selected_poi_id"),
                          "end": deepcopy(result.get("end")), "route_state": deepcopy(result.get("route_state")),
                          "candidate_cards": [{"poi_id": item.get("poi_id"), "name": item.get("name"),
                              "distance_m": item.get("distance_m"),
                              "coordinates": _point_record(item),
                              "distance_reference_name": item.get("distance_reference_name"),
                              "distance_basis": item.get("distance_basis"),
                              "recommendation_reason": item.get("recommendation_reason")}
                              for item in cards],
                          "route_verification": (result.get("route_verification") or {}).get("status"),
                          "tool_names": list(dict.fromkeys(item.get("tool_name") for item in events
                                                            if isinstance(item.get("tool_name"), str))),
                          "endpoint_observations": deepcopy(tracker["endpoint_observations"][endpoint_start:]),
                          "search_snapshots": deepcopy(tracker["search_snapshots"][search_start:]),
                          "presentation_errors": list(tracker["presentation_errors"]),
                          "browser_errors": list(tracker["browser_errors"]),
                          "seconds": round(time.monotonic() - started_at, 2)}
                print(json.dumps(record, ensure_ascii=False), flush=True)
                return run, result, record

            turns = []
            queries = scenario["queries"]
            first_run, first_result, first = turn(queries[0], expected_kind="candidates")
            turns.append(first)
            initial_ids = [item.get("poi_id") for item in first.get("candidate_cards") or []]
            first_start_records = [item for item in first["endpoint_observations"]
                                   if item.get("role") == "start"]
            first_anchor_records = [item for item in first["endpoint_observations"]
                                    if item.get("role") == "search_anchor"]
            if not any(item.get("poi_id") == EXPECTED_START_POI_ID for item in first_start_records):
                return {"session": scenario["id"], "status": "failed", "failed_at": 1,
                        "failure_code": "start_identity_mismatch", "turns": turns}
            if not any(item.get("poi_id") == EXPECTED_SEARCH_ANCHOR_POI_ID for item in first_anchor_records):
                return {"session": scenario["id"], "status": "failed", "failed_at": 1,
                        "failure_code": "search_anchor_identity_mismatch", "turns": turns}
            if (not initial_ids or first.get("search_radius_m") != 1000 or not first.get("food_intent_present")
                    or not first["all_accepted_messages_received"]
                    or not first["candidate_map_coordinates_match_cards"]
                    or first["loaded_real_map_tile_count"] < 1
                    or not any(item.get("kind") == "map_tile" and item.get("ok")
                               for item in first["map_resource_responses"])):
                return {"session": scenario["id"], "status": "failed", "failed_at": 1, "turns": turns}

            _, _, second = turn(queries[1], expected_kind="candidates")
            turns.append(second)
            if (not second.get("candidate_cards") or second.get("search_radius_m") != 1000
                    or second["task_id"] != first["task_id"]
                    or not second.get("food_intent_present")
                    or set(initial_ids) & {row["poi_id"] for row in second["candidate_cards"]}
                    or not second["all_accepted_messages_received"]):
                return {"session": scenario["id"], "status": "failed", "failed_at": 2, "turns": turns}

            _, _, third = turn(queries[2], expected_kind="candidates")
            turns.append(third)
            expanded_ids = {row["poi_id"] for row in third.get("candidate_cards") or []}
            if (third.get("search_radius_m") != 2000 or not expanded_ids
                    or not third.get("food_intent_present")
                    or expanded_ids & (set(initial_ids) | {row["poi_id"] for row in second["candidate_cards"]})
                    or third["task_id"] != first["task_id"] or not third["all_accepted_messages_received"]):
                return {"session": scenario["id"], "status": "failed", "failed_at": 3, "turns": turns}

            selected = next((item for item in third.get("browser_candidate_card_ids") or []
                             if item in expanded_ids), None)
            if not selected:
                return {"session": scenario["id"], "status": "failed", "failed_at": 3,
                        "failure_code": "candidate_not_rendered_in_browser", "turns": turns}
            _, _, fourth = turn(queries[3], select_poi_id=selected, expected_kind="route")
            turns.append(fourth)
            selected_route_state = fourth.get("route_state") or {}
            if (fourth.get("response_kind") != "route" or fourth.get("route_verification") != "pass"
                    or fourth.get("selected_poi_id") != selected or not fourth["all_accepted_messages_received"]
                    or fourth.get("task_id") != first.get("task_id")
                    or not fourth.get("route_line_visible_on_map")
                    or not any(item == {"tool": "plan_route", "end_turn_if_success": True}
                               for item in fourth.get("terminal_decisions", []))
                    or (selected_route_state.get("end") or {}).get("poi_id") != selected):
                return {"session": scenario["id"], "status": "failed", "failed_at": 4, "turns": turns}

            _, _, fifth = turn(queries[4], expected_kind="route")
            turns.append(fifth)
            if (fifth.get("response_kind") != "route" or fifth.get("route_verification") != "pass"
                    or fifth.get("selected_poi_id") != selected or not fifth["all_accepted_messages_received"]
                    or fifth.get("task_id") != first.get("task_id")
                    or not fifth.get("route_line_visible_on_map")
                    or not any(item == {"tool": "plan_route", "end_turn_if_success": True}
                               for item in fifth.get("terminal_decisions", []))
                    or (fifth.get("route_state") or {}).get("route_id") != selected_route_state.get("route_id")):
                return {"session": scenario["id"], "status": "failed", "failed_at": 5, "turns": turns}

            _, _, sixth = turn(queries[5], expected_kind="chat")
            turns.append(sixth)
            if ("get_weather" not in sixth["tool_names"] or not sixth["all_accepted_messages_received"]
                    or not any(item == {"tool": "get_weather", "end_turn_if_success": True}
                               for item in sixth.get("terminal_decisions", []))
                    or sixth["history_count"] <= 8):
                return {"session": scenario["id"], "status": "failed", "failed_at": 6, "turns": turns}

            _, _, seventh = turn(queries[6], expected_kind="route")
            turns.append(seventh)
            if (seventh.get("response_kind") != "route" or seventh.get("route_verification") != "pass"
                    or seventh.get("selected_poi_id") != selected or not seventh["all_accepted_messages_received"]
                    or seventh.get("task_id") != first.get("task_id")
                    or not seventh.get("route_line_visible_on_map")
                    or not any(item == {"tool": "plan_route", "end_turn_if_success": True}
                               for item in seventh.get("terminal_decisions", []))
                    or (seventh.get("route_state") or {}).get("route_id") != selected_route_state.get("route_id")
                    or seventh["history_count"] <= 8):
                return {"session": scenario["id"], "status": "failed", "failed_at": 7, "turns": turns}

            start_records = [item for row in turns for item in row["endpoint_observations"]
                             if item.get("role") == "start"]
            anchor_records = [item for row in turns for item in row["endpoint_observations"]
                              if item.get("role") == "search_anchor"]
            selected_records = [item for row in turns for item in row["endpoint_observations"]
                                if item.get("role") == "selected_destination"]
            anchor_identity = anchor_records[0] if anchor_records else None
            card_records = [row for record in turns for row in record["candidate_cards"]]
            same_anchor = bool(anchor_identity) and all(_same_endpoint(anchor_identity, item)
                                                        for item in anchor_records[1:])
            same_anchor = same_anchor and all(item.get("poi_id") == EXPECTED_SEARCH_ANCHOR_POI_ID
                                              for item in anchor_records)
            same_start = bool(start_records) and all(_same_endpoint(start_records[0], item)
                                                     for item in start_records[1:])
            same_start = same_start and all(item.get("poi_id") == EXPECTED_START_POI_ID
                                            for item in start_records)
            selected_card = next((row for row in card_records if row.get("poi_id") == selected), None)
            same_selected = bool(selected_records and selected_card) and all(
                _same_endpoint({"poi_id": selected, "coordinates": selected_card.get("coordinates")}, item)
                for item in selected_records)
            distance_cards = [row for row in card_records
                              if row.get("distance_basis") == "network_path_plus_unverified_straight_line_access_estimate"]
            correct_distance_reference = bool(anchor_identity and card_records) and (
                len(distance_cards) == len(card_records) and all(
                row.get("distance_reference_name") == anchor_identity.get("name") for row in distance_cards)
            )
            food_intent_persisted = bool(first.get("food_intent_present")) and all(
                row.get("food_intent_present") for row in turns[1:5])
            copy_ok = all(row["short_clean_user_copy"] and row["candidate_card_presentation_matches_result"]
                          and row["candidate_map_coordinates_match_cards"]
                          for row in turns[:3])
            browser_delivery_ok = all(row["assistant_text_matches_persisted"]
                                      and row["assistant_text_rendered_in_browser"]
                                      and row["latest_bubble_matches_result"] for row in turns)
            browser_errors_clear = not tracker["browser_errors"]
            complete = (same_anchor and same_start and same_selected and correct_distance_reference
                        and food_intent_persisted and copy_ok and browser_delivery_ok and browser_errors_clear)
            screenshot = report_dir / (scenario["id"] + "_mobile-browser.png")
            page.screenshot(path=str(screenshot), full_page=True)
            screenshot_hash = sha256(screenshot.read_bytes()).hexdigest()
            context.close()
            return {"session": scenario["id"], "status": "passed" if complete else "failed",
                    "conversation_id": tracker["conversation_id"], "task_id": first["task_id"],
                    "model_calls": tracker["call_count"], "start_identity": start_records[0] if start_records else None,
                    "search_anchor_identity": anchor_identity, "same_start_id_and_coordinates": same_start,
                    "same_anchor_id_and_coordinates": same_anchor, "selected_place_id_and_coordinates": same_selected,
                    "card_distance_reference_matches_anchor": correct_distance_reference,
                    "food_intent_persists_through_selection": food_intent_persisted,
                    "all_recommendation_copy_short_and_clean": copy_ok,
                    "assistant_delivery_matches_persisted_chat": browser_delivery_ok,
                    "browser_errors_clear": browser_errors_clear,
                    "screenshot_sha256": screenshot_hash, "turns": turns}

        original_factory = planner._make_client
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    for scenario in scenarios:
                        try:
                            planner._make_client = original_factory
                            session_record = run_session(scenario, browser)
                        except Exception as error:
                            session_record = {"session": scenario["id"], "status": "failed",
                                              "failure_type": type(error).__name__,
                                              "failure_code": str(getattr(error, "code", "") or "")[:80],
                                              "provider_status": getattr(error, "status_code", None),
                                              "conversation_id": tracker.get("conversation_id"),
                                              "model_calls": tracker.get("call_count", 0),
                                              "last_turn_summary": deepcopy(tracker.get("last_turn_summary")),
                                              "failure_frames": _failure_frames(error)}
                        global_record["sessions"].append(session_record)
                finally:
                    browser.close()
        finally:
            planner._make_client = original_factory
            Dialogue.present, Dialogue.spatial, Dialogue._endpoint = original_present, original_spatial, original_endpoint
    finally:
        planner._make_client = factory
        Dialogue.present, Dialogue.spatial, Dialogue._endpoint = original_present, original_spatial, original_endpoint
        static_server.shutdown()
        static_server.server_close()
        static_thread.join(timeout=5)
        resources.close()

    global_record["passed"] = all(item.get("status") == "passed" for item in global_record["sessions"])
    global_record["request_journal"] = request_journal.name
    global_record["request_journal_sha256"] = sha256(request_journal.read_bytes()).hexdigest() if request_journal.exists() else None
    global_record["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()
    report_path.write_text(json.dumps(global_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "passed": global_record["passed"],
                      "sessions": [{"session": item.get("session"), "status": item.get("status"),
                                    "failure_code": item.get("failure_code"),
                                    "failed_at": item.get("failed_at"), "model_calls": item.get("model_calls")}
                                   for item in global_record["sessions"]]}, ensure_ascii=False), flush=True)
    return 0 if global_record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
