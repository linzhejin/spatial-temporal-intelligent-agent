"""Desktop and mobile-size UI checks with synthetic candidate responses."""
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import expect, sync_playwright


@pytest.fixture(scope="module")
def static_site():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(
        Quiet, directory=str(Path(__file__).resolve().parents[2] / "static")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mobile", [False, True], ids=["desktop", "mobile-simulation"])
def test_strategy_is_visible_before_route_and_food_candidates_can_be_paged(static_site, mobile):
    messages = []
    candidates = [{"poi_id": f"poi_{i}", "name": f"测试餐饮{i}",
                   "subcategory": "restaurant", "distance_m": 100 * i} for i in range(1, 10)]

    def json_response(route, data, status=200):
        route.fulfill(status=status, content_type="application/json; charset=utf-8",
                      body=json.dumps({"data": data}, ensure_ascii=False))

    def handle(route):
        request = route.request
        url = urlsplit(request.url)
        if url.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
        elif url.path == "/api/conversations" and request.method == "POST":
            json_response(route, {"conversation_id": "local-test-conversation"})
        elif url.path.endswith("/messages") and request.method == "POST":
            messages.append(request.post_data_json)
            json_response(route, {"run_id": f"local-test-run-{len(messages)}"})
        elif url.path.startswith("/api/runs/"):
            index = int(url.path.rsplit("-", 1)[-1])
            result = ({"task_type": "path_planning", "response_kind": "candidates", "message": "附近找到候选。",
                       "candidates": candidates} if index <= 2 else
                      {"task_type": "chat", "response_kind": "chat", "reply": "已收到地点选择。"})
            json_response(route, {"run_id": f"local-test-run-{index}", "task_id": f"local-meal-task-{index}",
                                  "current_task_revision": index, "status": "completed", "events": [],
                                  "result": result})
        elif url.path == "/api/road-conditions":
            json_response(route, {"count": 0, "conditions": [], "active_event_version": "test"})
        elif url.path.startswith("/api/"):
            json_response(route, {})
        else:
            route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 390, "height": 844} if mobile
                                       else {"width": 1280, "height": 900},
                                       is_mobile=mobile, has_touch=mobile, service_workers="block")
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.route("**/*", handle)
        page.goto(static_site, wait_until="domcontentloaded")
        if mobile:
            assert page.locator("#chat-content").bounding_box()["height"] >= 200
        scenery = page.locator('.route-strategy-btn[data-route-strategy="scenery"]')
        expect(scenery).to_be_visible()
        scenery.click()
        expect(page.locator("#nl-input")).to_have_value("全程风景优先，")
        expect(page.locator("#route-strategy-hint")).to_contain_text("发送")
        assert not messages
        page.locator('.route-strategy-btn[data-route-strategy="shortest"]').click()
        expect(page.locator("#nl-input")).to_have_value("全程最短路径，")
        page.locator("#nl-input").fill("我想去总图附近干饭")
        page.locator("#submit-btn").click()
        cards = page.locator(".candidate-cards").last
        expect(cards.locator(".candidate-card")).to_have_count(4, timeout=15000)
        expect(cards.locator(".candidate-name").first).to_have_text("测试餐饮1")
        cards.get_by_role("button", name="换一批").click()
        expect(cards.locator(".candidate-name").first).to_have_text("测试餐饮5")
        assert len(messages) == 1, "browsing the received shortlist must not create a new task"
        cards.get_by_role("button", name="上一批").click()
        expect(cards.locator(".candidate-name").first).to_have_text("测试餐饮1")
        old_cards = page.locator(".candidate-cards").first
        page.locator("#nl-input").fill("查询另一个新的地点需求")
        page.locator("#submit-btn").click()
        expect(page.locator(".candidate-cards")).to_have_count(2, timeout=15000)
        old_cards.get_by_role("button", name="换一批").click()
        expect(old_cards.locator(".candidate-page-status")).to_contain_text("已过期")
        expect(old_cards.locator(".candidate-card").first).to_be_disabled()
        assert len(messages) == 2, "an old candidate page must not submit a request"
        current_cards = page.locator(".candidate-cards").last
        current_cards.get_by_role("button", name="换一批").click()
        current_cards.locator(".candidate-card").first.click()
        expect(page.locator(".chat-bubble-reply .chat-bubble-text").last).to_contain_text("已收到地点选择")
        assert messages[-1]["selected_poi_id"] == "poi_5"
        assert messages[-1]["continuation_task_id"] == "local-meal-task-2"
        assert messages[-1]["base_revision"] == 2
        browser.close()


def test_failed_run_shows_retry_and_cancel_without_exposing_internal_error(static_site):
    messages = []

    def json_response(route, data, status=200):
        route.fulfill(status=status, content_type="application/json; charset=utf-8",
                      body=json.dumps({"data": data}, ensure_ascii=False))

    def handle(route):
        url = urlsplit(route.request.url)
        if url.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
        elif url.path == "/api/conversations" and route.request.method == "POST":
            json_response(route, {"conversation_id": "recovery-conversation"})
        elif url.path.endswith("/messages") and route.request.method == "POST":
            messages.append(route.request.post_data_json)
            json_response(route, {"run_id": f"recovery-run-{len(messages)}"})
        elif url.path == "/api/runs/recovery-run-1":
            json_response(route, {
                "status": "failed", "task_id": "recovery-task", "current_task_revision": 2,
                "error": {"code": "provider_internal", "message": "private provider traceback"},
            })
        elif url.path == "/api/runs/recovery-run-2":
            json_response(route, {
                "status": "completed", "task_id": "recovery-task", "current_task_revision": 3,
                "result": {"task_type": "chat", "response_kind": "chat", "message": "已按原条件继续。"},
            })
        elif url.path == "/api/road-conditions":
            json_response(route, {"count": 0, "conditions": [], "active_event_version": "test"})
        elif url.path.startswith("/api/"):
            json_response(route, {})
        else:
            route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 900}, service_workers="block")
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.route("**/*", handle)
        page.goto(static_site, wait_until="domcontentloaded")
        page.locator("#nl-input").fill("我在信图，想去总图附近干饭")
        page.locator("#submit-btn").click()

        expect(page.get_by_text("本轮处理未完成，请稍后重试。", exact=False)).to_be_visible(timeout=15000)
        expect(page.get_by_text("private provider traceback", exact=False)).to_have_count(0)
        expect(page.get_by_role("button", name="取消", exact=True)).to_be_visible()
        page.get_by_role("button", name="重试", exact=True).click()
        expect(page.get_by_text("已按原条件继续。", exact=False)).to_be_visible(timeout=15000)
        assert messages[1]["continuation_task_id"] == "recovery-task"
        assert messages[1]["base_revision"] == 2
        assert messages[1]["ui_action"]["type"] == "retry"
        browser.close()


def test_required_viewport_widths_fit_without_horizontal_page_scroll(static_site):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for mobile, width, height in [
            (True, 360, 780), (True, 390, 844), (True, 430, 932),
            (False, 1280, 900), (False, 1440, 1000),
        ]:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                is_mobile=mobile, has_touch=mobile, service_workers="block",
            )
            context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
            page = context.new_page()

            def local_only(route):
                url = urlsplit(route.request.url)
                if url.hostname not in {"127.0.0.1", "localhost"}:
                    route.abort()
                elif url.path == "/api/road-conditions":
                    route.fulfill(status=200, content_type="application/json; charset=utf-8",
                                  body=json.dumps({"data": {"count": 0, "conditions": []}}))
                elif url.path.startswith("/api/"):
                    route.fulfill(status=200, content_type="application/json; charset=utf-8",
                                  body=json.dumps({"data": {}}))
                else:
                    route.continue_()

            page.route("**/*", local_only)
            page.goto(static_site, wait_until="domcontentloaded")
            page.locator("#nl-input").wait_for(state="visible")
            actual = page.evaluate("document.documentElement.scrollWidth")
            assert actual <= width, f"viewport {width}px has document width {actual}px"
            context.close()
        browser.close()



@pytest.mark.parametrize("subcategory", ["restaurant", "field"],
                         ids=["food", "unrelated-sport"])
def test_free_text_revision_does_not_preselect_a_task_in_the_browser(static_site, subcategory):
    messages = []
    def handle(route):
        request = route.request
        url = urlsplit(request.url)
        if url.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return
        def respond(data):
            route.fulfill(status=200, content_type="application/json; charset=utf-8",
                          body=json.dumps({"data": data}, ensure_ascii=False))
        if url.path == "/api/conversations" and request.method == "POST":
            respond({"conversation_id": "food-scope-test"})
        elif url.path.endswith("/messages") and request.method == "POST":
            messages.append(request.post_data_json)
            respond({"run_id": f"food-scope-{len(messages)}"})
        elif url.path.startswith("/api/runs/"):
            number = int(url.path.rsplit("-", 1)[-1])
            result = ({"task_type": "path_planning", "response_kind": "candidates", "message": "请选择地点",
                       "candidates": [{"poi_id": "p1", "name": "测试地点", "subcategory": subcategory}]}
                      if number == 1 else {"task_type": "chat", "response_kind": "chat", "reply": "已收到"})
            respond({"run_id": f"food-scope-{number}", "task_id": "first-task",
                     "current_task_revision": 1, "status": "completed", "events": [], "result": result})
        elif url.path == "/api/road-conditions":
            respond({"count": 0, "conditions": [], "active_event_version": "test"})
        elif url.path.startswith("/api/"):
            respond({})
        else:
            route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(service_workers="block")
        page.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page.route("**/*", handle)
        page.goto(static_site, wait_until="domcontentloaded")
        page.locator("#nl-input").fill("先给我地点候选")
        page.locator("#submit-btn").click()
        expect(page.locator(".candidate-card")).to_have_count(1, timeout=15000)
        page.locator("#nl-input").fill("不想吃食堂，想要吃餐厅")
        page.locator("#submit-btn").click()
        expect(page.locator(".chat-bubble-reply .chat-bubble-text").last).to_contain_text("已收到", timeout=15000)
        assert len(messages) == 2
        assert not messages[-1].get("continuation_task_id")
        assert not messages[-1].get("base_revision")
        browser.close()
