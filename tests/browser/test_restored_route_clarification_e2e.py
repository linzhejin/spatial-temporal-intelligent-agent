"""Browser regression for restoring an unfinished route with a clarification.

All APIs are mocked and all non-local network requests are blocked. The mobile
case emulates a viewport and user agent; it is not a real-device test.
"""

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


pytestmark = pytest.mark.skipif(sync_playwright is None, reason="Playwright unavailable")


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


@pytest.mark.parametrize("viewport,mobile", [
    ({"width": 1440, "height": 1000}, False),
    ({"width": 390, "height": 844}, True),
], ids=["pc", "mobile-web"])
def test_restored_route_clarification_preserves_question_and_task(static_site, viewport, mobile):
    task_id, revision = "restored-route-task", 7
    conversation = "local-restore-conversation"
    question = "下一站想去哪里？"
    submissions = []
    route_result = {
        "task_type": "path_planning", "response_kind": "route", "needs_input": True,
        "reply": "首段路线已生成。", "clarify": {"question": question, "options": ["食堂", "超市"]},
        "candidates": [], "route_kind": "direct", "mode": "walk",
        "recommended": [{"lng": 114.361, "lat": 30.535}, {"lng": 114.362, "lat": 30.536}],
        "shortest": [{"lng": 114.361, "lat": 30.535}, {"lng": 114.362, "lat": 30.536}],
        "recommended_length_m": 255, "shortest_length_m": 255, "duration_min": 3,
        "start": {"name": "总图"}, "end": {"name": "樱顶"},
    }

    def respond(route, body):
        route.fulfill(status=200, content_type="application/json; charset=utf-8",
                      body=json.dumps({"data": body}, ensure_ascii=False))

    def handle(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
        elif parsed.path == f"/api/conversations/{conversation}" and request.method == "GET":
            respond(route, {
                "messages": [{"role": "user", "content": "从总图到樱顶，再去附近吃饭",
                              "task_id": task_id, "seq": 1}],
                "tasks": [], "active_runs": [{"run_id": "restored-run", "task_id": task_id,
                                                "query": "从总图到樱顶，再去附近吃饭", "message_seq": 1}],
            })
        elif parsed.path == "/api/runs/restored-run":
            respond(route, {"run_id": "restored-run", "task_id": task_id,
                            "current_task_revision": revision, "status": "needs_input",
                            "events": [], "result": route_result})
        elif parsed.path == f"/api/conversations/{conversation}/messages" and request.method == "POST":
            submissions.append(request.post_data_json)
            respond(route, {"run_id": "continued-run"})
        elif parsed.path == "/api/runs/continued-run":
            respond(route, {"run_id": "continued-run", "task_id": task_id,
                            "current_task_revision": revision + 1, "status": "completed", "events": [],
                            "result": {"task_type": "chat", "response_kind": "chat", "reply": "已收到"}})
        elif parsed.path.startswith("/api/"):
            respond(route, {"conditions": [], "count": 0})
        else:
            route.continue_()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        args = {"viewport": viewport, "is_mobile": mobile, "has_touch": mobile, "permissions": []}
        if mobile:
            args["user_agent"] = ("Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36")
        context = browser.new_context(**args)
        context.add_init_script("""localStorage.setItem('whu_welcome_seen', 'true');
            localStorage.setItem('whu_walker:active_session', 'sess_restore_test');
            localStorage.setItem('whu_walker:server_conversation:sess_restore_test', 'local-restore-conversation');""")
        page = context.new_page()
        page.route("**/*", handle)
        page.goto(static_site, wait_until="domcontentloaded")
        option = page.locator(".clarify-chip", has_text="食堂")
        expect(option).to_be_visible(timeout=10000)
        question_visible = question in page.locator("#chat-content").inner_text()
        with page.expect_response(lambda response: response.url.endswith('/messages')):
            option.click()
        assert len(submissions) == 1
        assert submissions[0]["query"] == "食堂"
        assert submissions[0]["continuation_task_id"] == task_id
        assert submissions[0]["base_revision"] == revision
        assert question_visible, "恢复后必须显示明确的待答问题，不能只显示选项"
        browser.close()
