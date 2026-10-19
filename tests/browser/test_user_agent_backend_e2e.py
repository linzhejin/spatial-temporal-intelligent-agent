"""Browser to Flask/worker conversation flow on an isolated PostgreSQL database."""

from __future__ import annotations

import json
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


DATABASE_URL = None  # assigned only after isolated_test_database validates its target
pytestmark = [
    pytest.mark.usefixtures("isolated_test_database"),
    pytest.mark.skipif(
        sync_playwright is None,
        reason="install development Playwright and Chromium to run browser acceptance",
    ),
]


@pytest.fixture(scope="module")
def static_site():
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    static_dir = Path(__file__).resolve().parents[2] / "static"
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(static_dir)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_browser_conversation_persists_and_restores_through_real_api_and_worker(
    static_site, monkeypatch, tmp_path,
):
    from app import create_app
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run
    from langgraph.checkpoint.postgres import PostgresSaver

    database.initialize(DATABASE_URL)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(tmp_path / "road_conditions.json"))
    (tmp_path / "road_conditions.json").write_text("[]\n", encoding="utf-8")

    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="isolated-browser-e2e", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    created_conversations = []
    worker_calls = []

    def fake_agent(query, *, context, **_kwargs):
        assert context is not None
        return {"response_kind": "chat", "message": f"隔离后端已处理：{query}", "turns": 1}

    def forward_to_local_flask(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return

        # Weather is peripheral to this conversation acceptance and must not
        # reach any external provider.
        if parsed.path == "/api/weather":
            route.fulfill(
                content_type="application/json; charset=utf-8",
                body=json.dumps({"data": {"weather": "晴", "temperature": 20}}, ensure_ascii=False),
            )
            return

        if parsed.path == "/api/conversations" and request.method == "POST":
            response = client.open(parsed.path, method=request.method, json=request.post_data_json)
            payload = response.get_json()
            if response.status_code == 201:
                created_conversations.append(payload["data"]["conversation_id"])
        else:
            target = parsed.path + ("?" + parsed.query if parsed.query else "")
            if request.post_data is not None:
                response = client.open(
                    target,
                    method=request.method,
                    data=request.post_data,
                    content_type=request.headers.get("content-type", "application/json"),
                )
            else:
                response = client.open(target, method=request.method)

        if parsed.path.endswith("/messages") and request.method == "POST" and response.status_code == 202:
            with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
                saver.setup()
                graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)
                worker_calls.append(process_next_run(DATABASE_URL, "isolated-browser-worker", graph))

        route.fulfill(
            status=response.status_code,
            content_type=response.content_type or "application/json; charset=utf-8",
            body=response.get_data(),
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.on("pageerror", lambda error: pytest.fail(f"browser page error: {error}"))
        page.route("**/*", forward_to_local_flask)

        try:
            page.goto(static_site, wait_until="domcontentloaded")
            query = "\u4ecb\u7ecd\u4e00\u4e0b\u6a31\u9876"
            input_box = page.locator("#nl-input")
            input_box.fill(query)
            page.locator("#submit-btn").click()
            expect(page.locator("#chat-content")).to_contain_text(
                f"\u9694\u79bb\u540e\u7aef\u5df2\u5904\u7406\uff1a{query}", timeout=15000,
            )
            assert len(created_conversations) == 1
            assert len(worker_calls) == 1
            assert worker_calls[0]["status"] == "completed"

            with database.connect(DATABASE_URL) as conn:
                stored = conn.execute(
                    "SELECT role, content FROM conversation_message WHERE conversation_id=%s ORDER BY seq",
                    (created_conversations[0],),
                ).fetchall()
            assert [(item["role"], item["content"]) for item in stored] == [
                ("user", query),
                ("assistant", f"\u9694\u79bb\u540e\u7aef\u5df2\u5904\u7406\uff1a{query}"),
            ]

            page.reload(wait_until="domcontentloaded")
            expect(page.locator("#chat-content")).to_contain_text(query, timeout=15000)
            expect(page.locator("#chat-content")).to_contain_text(
                f"\u9694\u79bb\u540e\u7aef\u5df2\u5904\u7406\uff1a{query}", timeout=15000,
            )

            page.locator("#reset-btn").click()
            expect(page.locator("#chat-content .chat-bubble-row")).to_have_count(0)
            new_query = "\u65b0\u4f1a\u8bdd\u6d4b\u8bd5"
            input_box.fill(new_query)
            page.locator("#submit-btn").click()
            expect(page.locator("#chat-content")).to_contain_text(
                f"\u9694\u79bb\u540e\u7aef\u5df2\u5904\u7406\uff1a{new_query}", timeout=15000,
            )
            assert len(created_conversations) == 2
            assert created_conversations[0] != created_conversations[1]
            assert len(worker_calls) == 2
            assert [call["status"] for call in worker_calls] == ["completed", "completed"]
            assert query not in page.locator("#chat-content").inner_text()
            with database.connect(DATABASE_URL) as conn:
                old_count = conn.execute(
                    "SELECT count(*) AS n FROM conversation_message WHERE conversation_id=%s",
                    (created_conversations[0],),
                ).fetchone()["n"]
                new_messages = conn.execute(
                    "SELECT role, content FROM conversation_message WHERE conversation_id=%s ORDER BY seq",
                    (created_conversations[1],),
                ).fetchall()
            assert old_count == 2
            assert [(item["role"], item["content"]) for item in new_messages] == [
                ("user", new_query),
                ("assistant", f"\u9694\u79bb\u540e\u7aef\u5df2\u5904\u7406\uff1a{new_query}"),
            ]
        finally:
            browser.close()
