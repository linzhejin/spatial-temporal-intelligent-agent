"""Opt-in, bounded real-model smoke through a local browser and isolated worker.

This is development verification, not a field or formal experiment. It never
loads the business database, publishes an event, or logs the model credential.
"""

import json
import os
import re
import shutil
import threading
from hashlib import sha256
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from dotenv import dotenv_values
from openai import OpenAI

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None

from storage import database


DATABASE_URL = None
pytestmark = [
    pytest.mark.usefixtures("isolated_test_database"),
    pytest.mark.skipif(os.environ.get("RUN_REAL_MODEL_BROWSER_E2E") != "1",
                       reason="explicit real-model development opt-in required"),
    pytest.mark.skipif(sync_playwright is None, reason="Playwright unavailable"),
]


@pytest.mark.parametrize("mobile", [False, True], ids=["pc", "mobile-web"])
def test_real_model_route_through_isolated_browser_and_worker(monkeypatch, tmp_path, mobile):
    from app import create_app
    from agents import planner, tools
    from agents.workflow import build_agent_workflow
    from experiments.event_replay_guard import assert_replay_database_target
    from jobs.worker import process_next_run
    from langgraph.checkpoint.postgres import PostgresSaver
    from spatial import network
    from tests.db_safety import verify_isolated_test_database

    env_path = os.environ.get("LIVE_DEEPSEEK_ENV")
    if not env_path:
        pytest.skip("explicit model credential file path required")
    credentials = dotenv_values(env_path)
    key = credentials.get("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("model credential unavailable")
    assert_replay_database_target(DATABASE_URL)
    # The module fixture already points DATABASE_URL at the test DB. Hide that
    # temporary value while comparing the separately supplied credential file;
    # the fixture has already checked the original business configuration.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    verify_isolated_test_database(DATABASE_URL, business_urls=(credentials.get("DATABASE_URL"),))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    database.initialize(DATABASE_URL)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    events = tmp_path / "conditions.json"
    events.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(events))
    cache = tmp_path / "graph.graphml"
    shutil.copyfile(Path(__file__).resolve().parents[2] / "data/whu_road_network.graphml", cache)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(network, "_cache_path", lambda: str(cache))
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("map network requested"))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {})
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", key)
    monkeypatch.setattr(planner.config, "LLM_MODEL", "deepseek-chat")
    network.load_or_download_network()

    http = httpx.Client(trust_env=False, timeout=60)
    real_client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1",
                         max_retries=0, timeout=60, http_client=http)
    model_calls = []
    report_dir = (Path(__file__).resolve().parents[2] / "output/experiments/product_live_browser_development"
                  / uuid4().hex)
    report_dir.mkdir(parents=True, exist_ok=False)
    journal = report_dir / "model_calls.jsonl"

    def record_call(row):
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def create(**kwargs):
        if len(model_calls) >= 2:
            raise RuntimeError("development model call limit exceeded")
        number = len(model_calls) + 1
        model_calls.append(number)
        record_call({"call": number, "status": "started"})
        try:
            response = real_client.chat.completions.create(**kwargs)
        except Exception as error:
            record_call({"call": number, "status": "failed", "error_type": type(error).__name__})
            raise
        usage = getattr(response, "usage", None)
        record_call({"call": number, "status": "returned",
                     "input_tokens": getattr(usage, "prompt_tokens", None),
                     "output_tokens": getattr(usage, "completion_tokens", None),
                     "model_alias": getattr(response, "model", None)})
        return response

    monkeypatch.setattr(planner, "_make_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="isolated-live-browser", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    completed, terminal = [], []

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    static = Path(__file__).resolve().parents[2] / "static"
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(static)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def forward(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return
        if parsed.path == "/api/weather":
            route.fulfill(content_type="application/json", body=json.dumps({"data": {"weather": "晴"}}))
            return
        target = parsed.path + ("?" + parsed.query if parsed.query else "")
        response = client.open(target, method=request.method, data=request.post_data,
                               content_type=request.headers.get("content-type", "application/json"))
        if parsed.path.endswith("/messages") and request.method == "POST" and response.status_code == 202:
            expected_id = response.get_json()["data"]["run_id"]
            with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
                saver.setup()
                workflow = build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)
                outcome = process_next_run(DATABASE_URL, "isolated-live-browser-worker", workflow)
                assert outcome and outcome["run_id"] == expected_id
                completed.append(outcome)
        if parsed.path.startswith("/api/runs/") and response.status_code == 200:
            record = response.get_json()["data"]
            if record.get("status") == "completed":
                terminal.append(record)
        route.fulfill(status=response.status_code, content_type=response.content_type or "application/json",
                      body=response.get_data())

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 390, "height": 844} if mobile
                                           else {"width": 1280, "height": 900},
                                          is_mobile=mobile, has_touch=mobile)
            context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
            page = context.new_page()
            page.on("pageerror", lambda error: pytest.fail(f"browser page error: {error}"))
            page.route("**/*", forward)
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="domcontentloaded")
            page.locator("#nl-input").fill("从桂园食堂到樱顶，途中顺便买咖啡，告诉我绕行多少距离。")
            page.locator("#submit-btn").click()
            expect(page.locator("#results-section")).to_be_visible(timeout=90000)
            assert completed and completed[-1]["status"] == "completed"
            assert terminal and isinstance(terminal[-1].get("result"), dict)
            result = terminal[-1]["result"]
            assert result.get("recommended") and result.get("direct_reference_verified") is True
            delta = result["recommended_length_m"] - result["direct_reference_length_m"]
            assert delta >= 0.5, "this smoke case must have a measurable detour"
            expected_distance = f"{delta / 1000:.1f} 公里" if delta >= 1000 else f"{round(delta)} 米"
            expected_pattern = re.escape(expected_distance).replace(r"\ ", r"\s*")
            answer = page.locator("#chat-content .chat-bubble-text").filter(
                has_text=re.compile(r"多走约\s*" + expected_pattern)).last
            expect(answer).to_be_visible(timeout=90000)
            answer.scroll_into_view_if_needed()
            assert 1 <= len(model_calls) <= 2
            screenshot = report_dir / "browser.png"
            page.screenshot(path=str(screenshot), full_page=True)
            source_root = Path(__file__).resolve().parents[2]
            source_files = ("static/index.html", "static/js/app.js", "static/sw.js",
                            "agents/planner.py", "api/routes.py", "data/pois.json",
                            "data/whu_road_network.graphml")
            report = {"scope": "isolated_browser_live_model_development_not_field_or_formal",
                      "viewport": "mobile-web" if mobile else "pc",
                      "worker_status": completed[-1]["status"], "model_call_count": len(model_calls),
                      "model_call_limit": 2, "provider_model_revision_verified": False,
                      "route_length_m": result.get("recommended_length_m"),
                      "direct_reference_length_m": result.get("direct_reference_length_m"),
                      "direct_reference_verified": result.get("direct_reference_verified"),
                      "screenshot_sha256": sha256(screenshot.read_bytes()).hexdigest(),
                      "model_journal_sha256": sha256(journal.read_bytes()).hexdigest(),
                      "test_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                      "source_sha256": {name: sha256((source_root / name).read_bytes()).hexdigest()
                                        for name in source_files},
                      "limitations": ["mobile viewport emulation, not a physical phone",
                                      "synthetic empty road conditions", "third-party map tiles blocked",
                                      "not a field route check"]}
            with (report_dir / "report.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        real_client.close()
