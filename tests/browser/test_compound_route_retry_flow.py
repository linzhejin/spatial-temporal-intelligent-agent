"""Browser-level retry interaction with a simulated task API; not field QA."""
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

pytestmark = pytest.mark.skipif(sync_playwright is None, reason="Playwright not installed")


@pytest.mark.parametrize("mobile", [False, True], ids=["pc", "mobile-web"])
def test_retry_button_continues_same_task_without_selecting_poi(mobile):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    static_dir = Path(__file__).resolve().parents[2] / "static"
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(static_dir)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    messages = []

    def respond(route, obj):
        route.fulfill(status=200, content_type="application/json; charset=utf-8",
                      body=json.dumps(obj, ensure_ascii=False))

    def handle(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
        elif parsed.path == "/api/conversations" and request.method == "POST":
            respond(route, {"data": {"conversation_id": "retry-browser-test"}})
        elif parsed.path.endswith("/messages") and request.method == "POST":
            messages.append(request.post_data_json)
            respond(route, {"data": {"run_id": f"retry-{len(messages)}"}})
        elif parsed.path.startswith("/api/runs/"):
            index = int(parsed.path.rsplit("-", 1)[-1])
            result = ({"task_type": "path_planning", "response_kind": "clarify",
                       "message": "已保留你的行程，可以点击‘重试’。",
                       "clarify": {"question": "已保留你的行程，可以点击‘重试’。", "options": ["重试"]}}
                      if index == 1 else
                      {"task_type": "chat", "response_kind": "chat", "reply": "已继续原来的行程。"})
            respond(route, {"data": {"run_id": f"retry-{index}", "task_id": "retry-task",
                                     "current_task_revision": index, "status": "completed",
                                     "events": [], "result": result}})
        elif parsed.path.startswith("/api/"):
            respond(route, {"data": {}})
        else:
            route.continue_()

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            options = {"viewport": {"width": 390, "height": 844} if mobile else {"width": 1440, "height": 1000},
                       "is_mobile": mobile, "has_touch": mobile, "permissions": []}
            if mobile:
                options["user_agent"] = ("Mozilla/5.0 (Linux; Android 14; Pixel 7) "
                                         "AppleWebKit/537.36 Chrome/129.0.0.0 Mobile Safari/537.36")
            context = browser.new_context(**options)
            context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
            page = context.new_page()
            page.route("**/*", handle)
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="domcontentloaded")
            page.locator("#nl-input").fill("从总图先打球再吃饭")
            page.locator("#submit-btn").click()
            retry = page.locator(".clarify-chip", has_text="重试")
            expect(retry).to_be_visible(timeout=10000)
            retry.click()
            expect(page.get_by_text("已继续原来的行程。")).to_be_visible(timeout=10000)
            assert len(messages) == 2
            assert messages[1]["query"] == "重试"
            assert messages[1]["continuation_task_id"] == "retry-task"
            assert messages[1]["base_revision"] == 1
            assert not messages[1].get("selected_poi_id")
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
