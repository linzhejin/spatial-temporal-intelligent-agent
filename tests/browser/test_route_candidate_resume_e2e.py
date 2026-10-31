"""Real browser, Flask, worker and graph; fake LLM, isolated database and events."""
import json
import shutil
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None

from tests.test_vision_condition_feedback import _call, _message

DATABASE_URL = None
pytestmark = [pytest.mark.usefixtures("isolated_test_database"),
              pytest.mark.skipif(sync_playwright is None, reason="Playwright unavailable")]


@pytest.mark.parametrize("mobile", [False, True], ids=["pc", "mobile-web"])
def test_route_and_candidates_resume_to_full_trip(monkeypatch, tmp_path, mobile):
    from app import create_app
    from agents import planner, tools
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run
    from langgraph.checkpoint.postgres import PostgresSaver
    from spatial import network
    from storage import database

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "fixture-only")
    events = tmp_path / "conditions.json"
    events.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(events))
    cache = tmp_path / "graph.graphml"
    shutil.copyfile(Path(__file__).resolve().parents[2] / "data/whu_road_network.graphml", cache)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(network, "_cache_path", lambda: str(cache))
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("network requested"))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {})
    network.load_or_download_network()
    calls = [
        _call("plan_route", {"start": {"type": "poi", "name": "总图"},
                             "end": {"type": "poi", "name": "樱顶"}, "mode": "walk",
                             "constraints": {"avoid_steps": True}, "strategy": "shortest"}),
        _call("search_poi_candidates", {"start": {"type": "poi", "name": "总图"},
                                       "activities": ["canteen_meal"], "mode": "walk"}),
    ]
    replies = [_message(calls=calls), _message(content="首段和附近候选。")]
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: replies.pop(0))))
    monkeypatch.setattr(planner, "_make_client", lambda: fake_client)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        assert conn.execute("SELECT count(*) AS n FROM run WHERE status='queued'").fetchone()["n"] == 0
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="isolated-browser-test", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    completed, submissions = [], []

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
            submissions.append(json.loads(request.post_data))
            expected = response.get_json()["data"]["run_id"]
            with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
                saver.setup()
                workflow = build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)
                outcome = process_next_run(DATABASE_URL, "candidate-browser-isolated", workflow)
                assert outcome and outcome["run_id"] == expected
                completed.append(outcome)
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
            page.locator("#nl-input").fill("从总图到樱顶，然后去附近食堂，不走台阶")
            page.locator("#submit-btn").click()
            card = page.locator(".candidate-cards").last.locator(".candidate-card").first
            expect(card).to_be_visible(timeout=90000)
            card.click()
            expect(page.locator("#chat-content")).to_contain_text("经", timeout=90000)
            assert [outcome["status"] for outcome in completed] == ["needs_input", "completed"]
            assert len(submissions) == 2 and submissions[1].get("selected_poi_id")
            assert isinstance(submissions[1].get("base_revision"), int)
            with database.connect(DATABASE_URL) as conn:
                row = conn.execute("SELECT result FROM run WHERE run_id=%s", (completed[-1]["run_id"],)).fetchone()
            state = row["result"]["route_state"]
            assert state["start"]["name"] in {"总图", "武汉大学图书馆(总馆)"}
            assert state["end"].get("poi_id") == submissions[1]["selected_poi_id"]
            assert state["hard_constraints"]["avoid_steps"] is True
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
