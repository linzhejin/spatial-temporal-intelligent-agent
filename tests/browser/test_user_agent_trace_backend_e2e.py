"""Browser-to-worker tool trace persistence on an isolated PostgreSQL database."""

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


def test_browser_agent_tool_trace_is_persisted_and_redacted(
    static_site, monkeypatch, tmp_path,
):
    from app import create_app
    from agents.workflow import build_agent_workflow
    from experiments.scenario_live_adapter import _project_tool_trace
    from jobs.worker import process_next_run
    from langgraph.checkpoint.postgres import PostgresSaver

    database.initialize(DATABASE_URL)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("AGENT_TOOL_TRACE_ENABLED", "1")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(tmp_path / "road_conditions.json"))
    (tmp_path / "road_conditions.json").write_text("[]\n", encoding="utf-8")

    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="isolated-trace-browser", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    created_conversations = []
    accepted_runs = []
    worker_calls = []

    def fake_agent(query, *, tool_trace_callback=None, **_kwargs):
        assert callable(tool_trace_callback)
        tool_trace_callback({
            "tool_name": "get_place_details",
            "argument_keys": ["poi_id"],
            "outcome": "success",
            "artifact_kind": "candidates",
            "raw_arguments": {"poi_id": "PRIVATE_ARGUMENT_SENTINEL"},
            "raw_result": {"message": "PRIVATE_RESULT_SENTINEL"},
        })
        return {
            "response_kind": "chat",
            "message": f"隔离追踪已处理：{query}",
            "turns": 1,
        }

    def forward_to_local_flask(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return

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

        payload = response.get_json(silent=True) or {}
        if parsed.path == "/api/conversations" and request.method == "POST":
            if response.status_code == 201:
                created_conversations.append(payload["data"]["conversation_id"])
        if parsed.path.endswith("/messages") and request.method == "POST" and response.status_code == 202:
            accepted_runs.append(payload["data"]["run_id"])
            with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
                saver.setup()
                graph = build_agent_workflow(
                    saver, agent=fake_agent, database_url=DATABASE_URL,
                )
                worker_calls.append(process_next_run(
                    DATABASE_URL, "isolated-trace-browser-worker", graph,
                ))

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
            query = "查询珞珈山详情"
            page.locator("#nl-input").fill(query)
            page.locator("#submit-btn").click()
            expect(page.locator("#chat-content")).to_contain_text(
                f"隔离追踪已处理：{query}", timeout=15000,
            )

            assert len(created_conversations) == 1
            assert len(accepted_runs) == 1
            assert len(worker_calls) == 1
            assert worker_calls[0]["status"] == "completed"

            conversation_id = created_conversations[0]
            run_id = accepted_runs[0]
            run_response = client.get(
                f"/api/runs/{run_id}?conversation_id={conversation_id}",
            )
            assert run_response.status_code == 200
            stored_run = run_response.get_json()["data"]

            trace_events = [
                event for event in stored_run["events"]
                if event["event_type"].startswith("agent_tool_")
            ]
            assert [event["event_type"] for event in trace_events] == [
                "agent_tool_trace_started", "agent_tool_call",
            ]
            assert all(event["node_id"] == "agent_tool" for event in trace_events)
            assert trace_events[0]["payload"] == {"source": "agent_execution"}
            assert trace_events[1]["payload"] == {
                "tool_name": "get_place_details",
                "argument_keys": ["poi_id"],
                "outcome": "success",
                "artifact_kind": "candidates",
            }
            assert "PRIVATE_ARGUMENT_SENTINEL" not in json.dumps(trace_events)
            assert "PRIVATE_RESULT_SENTINEL" not in json.dumps(trace_events)
            assert _project_tool_trace(stored_run) == [{
                "tool_name": "get_place_details",
                "argument_keys": ["poi_id"],
                "outcome": "success",
                "artifact_kind": "candidates",
            }]

            with database.connect(DATABASE_URL) as conn:
                persisted = conn.execute(
                    "SELECT event_type, node_id, payload FROM run_event "
                    "WHERE run_id=%s AND event_type LIKE %s ORDER BY seq",
                    (run_id, "agent_tool_%"),
                ).fetchall()
            assert [row["event_type"] for row in persisted] == [
                "agent_tool_trace_started", "agent_tool_call",
            ]
            assert [row["payload"] for row in persisted] == [
                {"source": "agent_execution"},
                {
                    "tool_name": "get_place_details",
                    "argument_keys": ["poi_id"],
                    "outcome": "success",
                    "artifact_kind": "candidates",
                },
            ]
        finally:
            browser.close()
