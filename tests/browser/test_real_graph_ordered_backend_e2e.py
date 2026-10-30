"""Browser to real Flask/worker and real road graph; model and network disabled."""

import json
import shutil
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None

from storage import database

DATABASE_URL = None
pytestmark = [
    pytest.mark.usefixtures("isolated_test_database"),
    pytest.mark.skipif(sync_playwright is None, reason="Playwright unavailable"),
]


@pytest.mark.parametrize("mobile", [False, True], ids=["pc", "mobile-web"])
@pytest.mark.parametrize("compound", [False, True], ids=["ordered", "compound"])
def test_trip_through_real_api_worker_and_graph(monkeypatch, tmp_path, mobile, compound):
    from app import create_app
    from agents import planner, tools
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run
    from langgraph.checkpoint.postgres import PostgresSaver
    from spatial import network

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
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("network requested"))
    monkeypatch.setattr(planner, "_make_client", lambda: pytest.fail("model requested"))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {})
    network.load_or_download_network()

    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="isolated-real-graph", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    completed = []

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
            expected_run_id = response.get_json()["data"]["run_id"]
            with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
                saver.setup()
                workflow = build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)
                outcome = process_next_run(DATABASE_URL, "isolated-graph-worker", workflow)
                assert outcome and outcome["run_id"] == expected_run_id, (
                    "worker claimed an unrelated test run; check test queue isolation"
                )
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
            query = ("我在总图，先找一个球场然后去吃饭，不走台阶" if compound
                     else "从玉兰2门先到卓尔体育馆，再到桂园食堂，最后樱顶。")
            page.locator("#nl-input").fill(query)
            page.locator("#submit-btn").click()
            if compound:
                basketball = page.locator(".clarify-chip").filter(has_text="篮球")
                expect(basketball).to_be_visible(timeout=90000)
                basketball.click()
                sport = page.locator(".candidate-cards").last.locator(".candidate-card").first
                expect(sport).to_be_visible(timeout=90000)
                sport.click()
                expect(page.locator(".candidate-cards")).to_have_count(2, timeout=90000)
                meal = page.locator(".candidate-cards").last.locator(".candidate-card").first
                expect(meal).to_be_visible(timeout=90000)
                meal.click()
                expect(page.locator("#chat-content")).to_contain_text("经武汉大学篮球场", timeout=90000)
                assert [row["status"] for row in completed] == [
                    "needs_input", "needs_input", "needs_input", "completed",
                ]
            else:
                expect(page.locator("#chat-content")).to_contain_text("卓尔体育馆", timeout=90000)
                expect(page.locator("#chat-content")).to_contain_text("桂园食堂", timeout=90000)
                assert len(completed) == 1 and completed[0]["status"] == "completed"
            expect(page.locator("#results-section")).to_be_visible(timeout=90000)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
