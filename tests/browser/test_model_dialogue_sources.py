"""Browser acceptance for concise place cards and stable selection."""
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


def test_place_card_keeps_distance_and_reason_without_showing_source_details(local_site):
    submissions = []
    poll_counts = {}
    source = {"source_id": "Sabc123", "title": "武大梅园门店资料",
              "site_name": "示例资料站", "url": "https://example.org/meiyuan"}
    candidate_result = {
        "task_type": "candidates", "response_kind": "candidates",
        "message": "**总图附近可以看看这家。**",
        "candidates": [{
            "poi_id": "amap:A123", "name": "梅园在线餐厅", "subcategory": "restaurant",
            "subcategory_label": "餐厅",
            "source": "amap_online", "description": "梅园商圈附近", "lng": 114.36, "lat": 30.54,
            "distance_m": 328, "distance_reference_name": "总图",
            "source_ids": ["Sabc123"], "recommendation_reason": "适合坐下来慢慢吃饭。",
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
            poll_counts[index] = poll_counts.get(index, 0) + 1
            if index == 1 and poll_counts[index] == 1:
                fulfill(route, {"data": {
                    "run_id": "citation-run-1", "status": "running",
                    "events": [{"seq": 1, "event_type": "node_started",
                                "payload": {"label": "地点检索 search_poi_candidates"}}],
                }})
                return
            fulfill(route, {"data": {
                "run_id": f"citation-run-{index}", "task_id": "citation-task",
                "current_task_revision": index, "status": "completed",
                "events": [{"seq": 1, "event_type": "node_started",
                            "payload": {"label": "地点检索 search_poi_candidates"}}],
                "result": candidate_result if index == 1 else final_result,
            }})
            return
        if parsed.path.startswith("/api/"):
            fulfill(route, {"data": {}})
            return
        route.continue_()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True,
                                      service_workers="block")
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.on("pageerror", lambda error: pytest.fail(f"browser page error: {error}"))
        page.route("**/*", handle_request)
        page.goto(local_site, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=15000)

        page.locator("#nl-input").fill("我在信图，想去总图附近干饭")
        page.locator("#submit-btn").click()
        status = page.locator("#chat-run-statuses .chat-run-status-label")
        expect(status).to_be_visible(timeout=5000)
        expect(status).to_have_text("正在处理…")
        expect(page.locator("#chat-content")).not_to_contain_text("我看一下")
        expect(page.get_by_role("button", name="取消本次处理")).to_be_visible()
        expect(page.locator("#chat-content")).not_to_contain_text("search_poi_candidates")
        card = page.locator(".candidate-card").first
        expect(card).to_be_visible(timeout=10000)
        expect(card).to_contain_text("梅园在线餐厅")
        expect(card).to_contain_text("适合坐下来慢慢吃饭。")
        expect(card.locator(".candidate-dist")).to_contain_text("总图")
        expect(card.locator(".candidate-dist")).to_contain_text("328")
        expect(page.locator(".candidate-details")).to_have_count(0)
        expect(page.locator("#route-strategy-hint")).to_have_count(0)
        expect(page.locator("#chat-content")).to_contain_text("总图附近可以看看这家。")
        expect(page.locator("#chat-content")).not_to_contain_text("**")
        expect(page.locator("#chat-content")).not_to_contain_text("search_poi_candidates")
        expect(page.locator("#chat-run-statuses .chat-run-status-row")).to_have_count(0)
        for internal_detail in ("来源与位置", source["source_id"], source["url"], "记录来源", "查询时间"):
            expect(page.locator("#chat-content")).not_to_contain_text(internal_detail)

        with page.expect_response(lambda response: "citation-run-2" in response.url, timeout=10000):
            card.click()
        assert submissions[0]["query"] == "我在信图，想去总图附近干饭"
        assert submissions[1]["selected_poi_id"] == "amap:A123"
        assert submissions[1]["continuation_task_id"] == "citation-task"

        context.close()
        browser.close()


def test_mobile_wait_cancel_empty_retry_and_conversation_restore(local_site):
    submissions = []
    polls = {}
    cancelled = set()
    place = {
        "poi_id": "amap:RESTORE123", "name": "梅园在线餐厅", "subcategory": "restaurant",
        "subcategory_label": "餐厅", "distance_m": 328, "distance_reference_name": "总图",
        "recommendation_reason": "适合坐下来吃饭。", "source_ids": ["Sabc123"],
    }
    place_request = {"request_id": "req-restore", "original_query": "附近帮我找吃饭的地方",
                     "pending_slot": "selection", "candidate_poi_ids": [place["poi_id"]],
                     "candidate_places": [place]}
    candidate_result = {"task_type": "candidates", "response_kind": "candidates",
                        "message": "总图附近可以看看这家。", "candidates": [place],
                        "place_request": place_request}

    def fulfill(route, payload, status=200):
        route.fulfill(status=status, content_type="application/json; charset=utf-8",
                      body=json.dumps(payload, ensure_ascii=False))

    def handler(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return
        if parsed.path == "/api/conversations" and request.method == "POST":
            fulfill(route, {"data": {"conversation_id": "mobile-session"}})
            return
        if parsed.path == "/api/conversations/mobile-session" and request.method == "GET":
            fulfill(route, {"data": {
                "conversation_id": "mobile-session",
                "messages": [
                    {"role": "user", "content": "附近帮我找吃饭的地方", "task_id": "task-retry", "seq": 1},
                    {"role": "assistant", "content": candidate_result["message"], "task_id": "task-retry", "seq": 2,
                     "metadata": {"task_type": "candidates", "response_kind": "candidates",
                                  "status": "needs_input", "source_message_seq": 1}},
                ],
                "tasks": [{"task_id": "task-retry", "revision": 4, "status": "needs_input",
                           "state": {"pending_questions": ["请选择一个地点继续规划路线。"],
                                     "place_request": place_request}}],
                "active_runs": [],
            }})
            return
        if parsed.path.endswith("/messages") and request.method == "POST":
            body = request.post_data_json
            submissions.append(body)
            fulfill(route, {"data": {"run_id": f"mobile-run-{len(submissions)}", "status": "queued"}})
            return
        if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/cancel"):
            run_id = parsed.path.split("/")[-2]
            cancelled.add(run_id)
            fulfill(route, {"data": {"status": "cancelled"}})
            return
        if parsed.path.startswith("/api/runs/") and request.method == "GET":
            run_id = parsed.path.rsplit("/", 1)[-1]
            index = int(run_id.rsplit("-", 1)[-1])
            polls[run_id] = polls.get(run_id, 0) + 1
            base = {"run_id": run_id, "task_id": f"task-{index}", "current_task_revision": index}
            if run_id in cancelled:
                fulfill(route, {"data": {**base, "status": "cancelled", "events": []}})
            elif index == 1:
                fulfill(route, {"data": {**base, "status": "running", "events": [{
                    "seq": 1, "event_type": "node_started",
                    "payload": {"label": "正在调用地点检索工具"},
                }]}})
            elif index == 2:
                fulfill(route, {"data": {**base, "status": "completed", "events": [], "result": {
                    "task_type": "candidates", "response_kind": "candidates",
                    "message": "附近没找到合适的地点。", "candidates": [],
                }}})
            elif index == 3:
                fulfill(route, {"data": {**base, "status": "failed", "events": [], "result": {
                    "message": "服务暂时不可用，请重试。",
                }}})
            elif index == 4:
                fulfill(route, {"data": {**base, "status": "completed", "events": [],
                                         "result": candidate_result}})
            else:
                fulfill(route, {"data": {**base, "status": "completed", "events": [],
                                         "result": {"task_type": "chat", "response_kind": "chat",
                                                    "message": "已选好地点。"}}})
            return
        if parsed.path.startswith("/api/"):
            fulfill(route, {"data": {}})
            return
        route.continue_()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True,
                                      has_touch=True, service_workers="block")
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.on("pageerror", lambda error: pytest.fail(f"browser page error: {error}"))
        page.route("**/*", handler)
        page.goto(local_site, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle", timeout=15000)

        page.locator("#nl-input").fill("先试试等待，再取消")
        page.locator("#submit-btn").click()
        status = page.locator("#chat-run-statuses .chat-run-status-label")
        expect(status).to_be_visible(timeout=5000)
        expect(status).to_have_text("正在处理…")
        expect(page.locator("#chat-content")).not_to_contain_text("我看一下")
        expect(page.get_by_role("button", name="取消本次处理")).to_be_visible()
        expect(page.locator("#chat-content")).not_to_contain_text("地点检索工具")
        page.get_by_role("button", name="取消本次处理").click()
        expect(page.locator("#chat-content")).to_contain_text("已取消，原有条件已保留。", timeout=10000)
        expect(page.locator("#chat-run-statuses .chat-run-status-row")).to_have_count(0)

        page.locator("#nl-input").fill("查一下附近的吃饭地点")
        page.locator("#submit-btn").click()
        expect(page.locator("#chat-content")).to_contain_text("附近没找到合适的地点。", timeout=10000)
        expect(page.locator(".candidate-card")).to_have_count(0)

        page.locator("#nl-input").fill("附近帮我找吃饭的地方")
        page.locator("#submit-btn").click()
        expect(page.locator("#chat-content")).to_contain_text("服务暂时不可用，请重试。", timeout=10000)
        expect(page.get_by_role("button", name="重试")).to_be_visible()
        page.get_by_role("button", name="重试").click()
        card = page.locator(".candidate-card").last
        expect(card).to_be_visible(timeout=10000)
        expect(card).to_contain_text("总图")

        page.reload(wait_until="domcontentloaded")
        restored = page.locator(".candidate-card").last
        expect(restored).to_be_visible(timeout=10000)
        expect(restored).to_contain_text("梅园在线餐厅")
        with page.expect_response(lambda response: response.request.method == "POST"
                                                   and response.url.endswith("/messages"), timeout=10000):
            restored.click()
        assert submissions[-1]["selected_poi_id"] == place["poi_id"]
        assert submissions[-1]["continuation_task_id"] == "task-retry"

        context.close()
        browser.close()
