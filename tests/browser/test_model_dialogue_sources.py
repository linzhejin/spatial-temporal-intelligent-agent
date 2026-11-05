"""Browser acceptance for model-selected place citations and card selection."""
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


pytestmark = pytest.mark.skipif(sync_playwright is None, reason="Playwright browser is unavailable")


@pytest.fixture
def local_site():
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


def test_place_card_shows_model_cited_source_and_keeps_selected_id(local_site):
    submissions = []
    source = {"source_id": "Sabc123", "title": "武大梅园门店资料",
              "site_name": "示例资料站", "url": "https://example.org/meiyuan"}
    candidate_result = {
        "task_type": "candidates", "response_kind": "candidates",
        "message": "按你说的区域找到一个可核对的候选。",
        "candidates": [{
            "poi_id": "amap:A123", "name": "梅园在线餐厅", "subcategory": "restaurant",
            "source": "amap_online", "description": "梅园商圈附近", "lng": 114.36, "lat": 30.54,
            "source_ids": ["Sabc123"], "recommendation_reason": "网页资料与地图地点名称相符。",
        }],
        "sources": [source],
        "place_request": {"request_id": "req-1", "original_query": "我在信图，想去总图附近干饭",
                          "pending_slot": "selection", "candidate_poi_ids": ["amap:A123"]},
    }
    final_result = {"task_type": "chat", "response_kind": "chat", "reply": "路线已交给路线规划工具。"}

    def fulfill(route, payload, status=200):
        route.fulfill(status=status, content_type="application/json; charset=utf-8",
                      body=json.dumps(payload, ensure_ascii=False))

    def handle_request(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return
        if parsed.path == "/api/conversations" and request.method == "POST":
            fulfill(route, {"data": {"conversation_id": "citation-flow"}})
            return
        if parsed.path.endswith("/messages") and request.method == "POST":
            submissions.append(request.post_data_json)
            fulfill(route, {"data": {"run_id": f"citation-run-{len(submissions)}"}})
            return
        if parsed.path.startswith("/api/runs/") and request.method == "GET":
            index = int(parsed.path.rsplit("-", 1)[-1])
            fulfill(route, {"data": {
                "run_id": f"citation-run-{index}", "task_id": "citation-task",
                "current_task_revision": index, "status": "completed", "events": [],
                "result": candidate_result if index == 1 else final_result,
            }})
            return
        if parsed.path.startswith("/api/"):
            fulfill(route, {"data": {}})
            return
        route.continue_()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.on("pageerror", lambda error: pytest.fail(f"browser page error: {error}"))
        page.route("**/*", handle_request)
        page.goto(local_site, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=15000)

        page.locator("#nl-input").fill("我在信图，想去总图附近干饭")
        page.locator("#submit-btn").click()
        card = page.locator(".candidate-card").first
        expect(card).to_be_visible(timeout=10000)
        expect(card).to_contain_text("梅园在线餐厅")
        details = page.locator(".candidate-details").first
        details.locator("summary").click()
        citation = details.locator("a.candidate-source-link")
        expect(citation).to_have_attribute("href", source["url"])
        expect(citation).to_have_attribute("rel", "noopener noreferrer")
        expect(citation).to_have_text(source["title"])

        with page.expect_response(lambda response: "citation-run-2" in response.url, timeout=10000):
            card.click()
        assert submissions[0]["query"] == "我在信图，想去总图附近干饭"
        assert submissions[1]["selected_poi_id"] == "amap:A123"
        assert submissions[1]["continuation_task_id"] == "citation-task"

        context.close()
        browser.close()
