"""PC/mobile browser flow for clarification, mode changes, and POI selection.

Run with ``python -m pytest -q tests/browser/test_user_agent_web_flow.py``.
The page is served locally and only the task API is mocked; no model, database,
production service, location permission, or external tile server is used.
"""

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


pytestmark = pytest.mark.skipif(
    sync_playwright is None,
    reason="install development Playwright and Chromium to run browser acceptance",
)


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


@pytest.mark.parametrize(
    ("viewport", "user_agent", "is_mobile", "expected_mode"),
    [
        ({"width": 1440, "height": 1000}, None, False, "is-planning-mode"),
        (
            {"width": 390, "height": 844},
            "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/129.0.0.0 Mobile Safari/537.36",
            True,
            "is-navigation-mode",
        ),
    ],
    ids=["pc", "mobile-web"],
)
def test_user_keeps_clarification_and_candidate_task_when_switching_mode(
    static_site, viewport, user_agent, is_mobile, expected_mode,
):
    messages = []
    runs = {}
    run_number = 0
    clarify_result = {
        "task_type": "path_planning",
        "response_kind": "clarify",
        "clarify": {
            "question": "\u60f3\u6253\u4ec0\u4e48\u7403\uff1f",
            "options": [
                "\u7bee\u7403", "\u7fbd\u6bdb\u7403", "\u4e52\u4e53\u7403",
                "\u8db3\u7403", "\u5176\u4ed6\u7403\u7c7b",
            ],
        },
    }
    candidate_result = {
        "task_type": "path_planning",
        "response_kind": "candidates",
        "message": "\u627e\u5230\u9644\u8fd1\u7684\u7bee\u7403\u573a\uff0c\u70b9\u4e00\u4e2a\u7ee7\u7eed\u89c4\u5212\u3002",
        "candidates": [
            {
                "poi_id": "whu-gongxue-basketball-court",
                "name": "\u5de5\u5b66\u90e8\u7bee\u7403\u573a",
                "activity_labels": ["\u7bee\u7403"],
                "campus": "\u5de5\u5b66\u90e8",
                "distance_m": 255,
                "lng": 114.362,
                "lat": 30.536,
            },
        ],
    }
    route_result = {
        "task_type": "path_planning",
        "route_kind": "direct",
        "recommended": [
            {"lng": 114.361, "lat": 30.535},
            {"lng": 114.362, "lat": 30.536},
        ],
        "shortest": [
            {"lng": 114.361, "lat": 30.535},
            {"lng": 114.362, "lat": 30.536},
        ],
        "recommended_length_m": 255.0,
        "shortest_length_m": 255.0,
        "duration_min": 3.0,
        "shortest_duration_min": 3.0,
        "overlap_rate": 1.0,
        "mode": "bike",
        "start": {"name": "\u4fe1\u606f\u5b66\u90e8"},
        "end": {
            "poi_id": "whu-gongxue-basketball-court",
            "name": "\u5de5\u5b66\u90e8\u7bee\u7403\u573a",
        },
        "strategy": {"name": "shortest", "weights": {"distance": 1, "slope": 0, "scenery": 0}},
        "route_verification": {"status": "pass", "field_truth_verified": False},
        "explanation": "\u8def\u7ebf\u5df2\u89c4\u5212\uff0c\u672b\u7aef\u63a5\u9a73\u4ecd\u9700\u73b0\u573a\u786e\u8ba4\u3002",
    }
    custom_ball_candidate_result = {
        "task_type": "path_planning",
        "response_kind": "candidates",
        "message": "\u8fd9\u662f\u6d4f\u89c8\u5668\u9a8c\u6536\u7528\u7684\u5408\u6210\u5019\u9009\u70b9\u3002",
        "candidates": [{
            "poi_id": "local-test-rugby-place",
            "name": "\u6d4b\u8bd5\u6a44\u6984\u7403\u5019\u9009\u70b9",
            "activity_labels": ["\u6a44\u6984\u7403"],
            "campus": "\u6d4b\u8bd5\u6821\u533a",
            "distance_m": 310,
            "lng": 114.363,
            "lat": 30.537,
        }],
    }
    route_comparison_result = {
        "task_type": "chat",
        "response_kind": "route_comparison",
        "reply": "已比较两种可用方案，时间为估算值；比较不会改变当前地图路线。",
        "route_comparison": {
            "status": "partial",
            "start_name": "信息学部",
            "end_name": "卓尔体育馆",
            "mode": "bike",
            "alternatives": [
                {"strategy": "shortest", "label": "最短路径", "distance_m": 500,
                 "duration_min": 7.0, "distance_delta_from_shortest_m": 0,
                 "slope_level_avg": 2.5, "scenery_level_avg": 3.0},
                {"strategy": "flat", "label": "平坦优先", "distance_m": 540,
                 "duration_min": 7.8, "distance_delta_from_shortest_m": 40,
                 "slope_level_avg": 1.7, "scenery_level_avg": 2.4},
            ],
            "unavailable": [
                {"strategy": "scenery", "label": "风景优先", "reason": "当前无可行路线"},
            ],
        },
    }
    no_route_result = {
        "task_type": "chat", "response_kind": "no_route",
        "reply": "当前路网下没有可通行的完整路线。",
        "route_state": {
            "start": {"name": "梅园综合楼"},
            "end": {"name": "武汉大学图书馆"},
            "travel_mode": "walk", "hard_constraints": {},
        },
    }
    results = [clarify_result, candidate_result, route_result, clarify_result,
               custom_ball_candidate_result, route_comparison_result,
               no_route_result]

    def json_response(route, payload, status=200):
        route.fulfill(
            status=status,
            content_type="application/json; charset=utf-8",
            body=json.dumps(payload, ensure_ascii=False),
        )

    def handle_request(route):
        nonlocal run_number
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return

        if parsed.path == "/api/conversations" and request.method == "POST":
            json_response(route, {"data": {"conversation_id": "local-web-flow-conversation"}})
            return
        if parsed.path.endswith("/messages") and request.method == "POST":
            body = request.post_data_json
            messages.append(body)
            run_number += 1
            run_id = f"local-run-{run_number}"
            runs[run_id] = results[run_number - 1]
            json_response(route, {"data": {"run_id": run_id}})
            return
        if parsed.path.startswith("/api/runs/") and request.method == "GET":
            run_id = parsed.path.rsplit("/", 1)[-1]
            index = int(run_id.rsplit("-", 1)[-1])
            json_response(route, {"data": {
                "run_id": run_id,
                "task_id": "local-basketball-task" if index <= 3 else "local-other-ball-task",
                "current_task_revision": index if index <= 3 else index - 3,
                "status": "completed",
                "events": [],
                "result": runs[run_id],
            }})
            return
        if parsed.path == "/api/weather":
            json_response(route, {"data": {"weather": "\u6674", "temperature": 20}})
            return
        if parsed.path == "/api/road-conditions":
            json_response(route, {"data": {
                "count": 0,
                "conditions": [],
                "active_event_version": "local-static-test",
            }})
            return
        if parsed.path.startswith("/api/"):
            json_response(route, {"data": {}})
            return
        route.continue_()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context_args = {
            "viewport": viewport,
            "is_mobile": is_mobile,
            "has_touch": is_mobile,
            "permissions": [],
        }
        if user_agent:
            context_args["user_agent"] = user_agent
        context = browser.new_context(**context_args)
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.on("pageerror", lambda error: pytest.fail(f"browser page error: {error}"))
        page.route("**/*", handle_request)
        page.goto(static_site, wait_until="domcontentloaded")

        body_classes = (page.locator("body").get_attribute("class") or "").split()
        assert expected_mode in body_classes
        if is_mobile:
            resizer = page.locator("#mobile-panel-resizer")
            initial_ratio = int(resizer.get_attribute("aria-valuenow"))
            bounds = resizer.bounding_box()
            assert bounds is not None
            drag_x = bounds["x"] + bounds["width"] / 2
            drag_y = bounds["y"] + bounds["height"] / 2
            page.mouse.move(drag_x, drag_y)
            page.mouse.down()
            page.mouse.move(drag_x, drag_y + 64, steps=4)
            page.mouse.up()
            assert int(resizer.get_attribute("aria-valuenow")) > initial_ratio
            expect(page.locator("body")).not_to_have_class("is-resizing-mobile-panels")

            # Chromium CDP touch input exercises the pointer-event path used by
            # a touchscreen. This remains browser emulation, not Android-device QA.
            touch_ratio = int(resizer.get_attribute("aria-valuenow"))
            touch_bounds = resizer.bounding_box()
            assert touch_bounds is not None
            touch_x = round(touch_bounds["x"] + touch_bounds["width"] / 2)
            touch_y = round(touch_bounds["y"] + touch_bounds["height"] / 2)
            cdp = context.new_cdp_session(page)
            cdp.send("Input.dispatchTouchEvent", {
                "type": "touchStart",
                "touchPoints": [{"x": touch_x, "y": touch_y, "id": 1, "force": 1}],
            })
            cdp.send("Input.dispatchTouchEvent", {
                "type": "touchMove",
                "touchPoints": [{
                    "x": touch_x, "y": touch_y + 48, "id": 1, "force": 1,
                }],
            })
            cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
            assert int(resizer.get_attribute("aria-valuenow")) > touch_ratio
            expect(page.locator("body")).not_to_have_class("is-resizing-mobile-panels")
            cdp.detach()
        input_box = page.locator("#nl-input")
        input_box.fill("\u53bb\u6253\u7403")
        page.locator("#submit-btn").click()
        basketball = page.locator(".clarify-chip").nth(0)
        expect(basketball).to_be_visible(timeout=10000)
        expect(page.locator(".clarify-chip")).to_have_count(5)
        assert len(messages) == 1
        assert messages[0]["query"] == "\u53bb\u6253\u7403"

        page.locator('.travel-mode-btn[data-travel-mode="bike"]').click()
        expect(page.locator('.travel-mode-btn[data-travel-mode="bike"]')).to_have_attribute(
            "aria-pressed", "true",
        )
        assert len(messages) == 1, "changing mode must not submit a new conversation turn"

        basketball.click()
        candidate = page.locator(".candidate-card").nth(0)
        expect(candidate).to_be_visible(timeout=10000)
        expect(candidate).to_contain_text("255 m")
        assert len(messages) == 2
        assert messages[1]["query"] == "\u7bee\u7403"
        assert messages[1]["travel_mode"] == "bike"
        assert messages[1]["continuation_task_id"] == "local-basketball-task"
        assert messages[1]["base_revision"] == 1

        candidate.click()
        expect(page.locator("#results-section")).to_be_visible(timeout=10000)
        assert len(messages) == 3
        assert messages[2]["query"] == "\u53bb\u5de5\u5b66\u90e8\u7bee\u7403\u573a"
        assert messages[2]["travel_mode"] == "bike"
        assert messages[2]["continuation_task_id"] == "local-basketball-task"
        assert messages[2]["base_revision"] == 2
        assert messages[2]["selected_poi_id"] == "whu-gongxue-basketball-court"
        assert page.locator("#estimated-duration").inner_text().strip()
        expect(page.locator("#chat-content")).to_contain_text(
            "\u672a\u5b9e\u5730\u786e\u8ba4\u9053\u8def\u5f53\u524d\u53ef\u901a\u884c",
        )
        route_distance_before_comparison = page.locator("#recommended-distance").inner_text().strip()
        route_duration_before_comparison = page.locator("#estimated-duration").inner_text().strip()

        input_box.fill("\u53bb\u6253\u7403")
        with page.expect_response(lambda response: "/api/runs/local-run-4?" in response.url):
            page.locator("#submit-btn").click()
        expect(page.locator(".clarify-chips")).to_have_count(2, timeout=10000)
        latest_clarification = page.locator(".clarify-chips").last
        expect(latest_clarification.locator(".clarify-chip")).to_have_count(5, timeout=10000)
        assert len(messages) == 4
        assert messages[3]["query"] == "\u53bb\u6253\u7403"
        assert "continuation_task_id" not in messages[3]

        latest_clarification.locator(".clarify-chip").nth(4).click()
        expect(input_box).to_be_focused()
        assert "\u8865\u5145\u7403\u7c7b" in input_box.get_attribute("placeholder")
        assert len(messages) == 4, "the catch-all option should ask for free text, not submit a task"
        input_box.fill("\u6a44\u6984\u7403")
        with page.expect_response(lambda response: "/api/runs/local-run-5?" in response.url):
            page.locator("#submit-btn").click()
        expect(page.locator(".candidate-cards")).to_have_count(2, timeout=10000)
        candidate = page.locator(".candidate-cards").last.locator(".candidate-card").nth(0)
        expect(candidate).to_be_visible(timeout=10000)
        expect(candidate).to_contain_text("\u6d4b\u8bd5\u6a44\u6984\u7403\u5019\u9009\u70b9")
        assert len(messages) == 5
        assert messages[4]["query"] == "\u6a44\u6984\u7403"
        assert messages[4]["continuation_task_id"] == "local-other-ball-task"
        assert messages[4]["base_revision"] == 1

        input_box.fill("比较三种走法")
        with page.expect_response(lambda response: "/api/runs/local-run-6?" in response.url):
            page.locator("#submit-btn").click()
        comparison = page.locator(".route-comparison-card")
        expect(comparison).to_be_visible(timeout=10000)
        expect(comparison).to_contain_text("信息学部 → 卓尔体育馆")
        expect(comparison).to_contain_text("最短路径")
        expect(comparison).to_contain_text("500 m")
        expect(comparison).to_contain_text("比最短多 40 m")
        expect(comparison).to_contain_text("风景优先")
        expect(comparison).to_contain_text("当前无可行路线")
        # Route comparison is read-only: retain the existing route and its metrics.
        expect(page.locator("#results-section")).to_be_visible()
        assert page.locator("#recommended-distance").inner_text().strip() == route_distance_before_comparison
        assert page.locator("#estimated-duration").inner_text().strip() == route_duration_before_comparison
        assert len(messages) == 6

        input_box.fill("从梅园综合楼到武汉大学图书馆，现在能走吗")
        with page.expect_response(lambda response: "/api/runs/local-run-7?" in response.url):
            page.locator("#submit-btn").click()
        expect(page.locator("#chat-content")).to_contain_text("当前路网下没有可通行的完整路线。")
        expect(page.locator("#results-section")).to_be_hidden()
        assert len(messages) == 7

        context.close()
        browser.close()


def test_user_can_retry_submission_and_refresh_cancels_old_task(static_site):
    """Exercise actual error, manual retry, cancel, and clean-session behavior."""
    submissions = []
    created_conversations = []
    cancelled_runs = set()
    fail_once = {"value": True}

    def json_response(route, payload, status=200):
        route.fulfill(
            status=status,
            content_type="application/json; charset=utf-8",
            body=json.dumps(payload, ensure_ascii=False),
        )

    def handle_request(route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return

        if parsed.path == "/api/conversations" and request.method == "POST":
            conversation_id = f"local-conversation-{len(created_conversations) + 1}"
            created_conversations.append(conversation_id)
            json_response(route, {"data": {"conversation_id": conversation_id}})
            return
        if parsed.path.endswith("/messages") and request.method == "POST":
            body = request.post_data_json
            submissions.append((parsed.path.split("/")[3], body["query"]))
            if fail_once["value"]:
                fail_once["value"] = False
                json_response(route, {
                    "error": "internal_error",
                    "message": "temporary test failure",
                }, status=503)
                return
            if body["query"] == "\u5904\u7406\u4e2d":
                run_id = "local-active-run"
            elif body["query"] == "\u65b0\u4f1a\u8bdd":
                run_id = "local-fresh-run"
            else:
                run_id = "local-retry-run"
            json_response(route, {"data": {"run_id": run_id}})
            return
        if parsed.path.endswith("/cancel") and request.method == "POST":
            cancelled_runs.add(parsed.path.split("/")[-2])
            json_response(route, {"data": {"status": "cancelled"}})
            return
        if parsed.path == "/api/runs/local-active-run":
            status = "cancelled" if "local-active-run" in cancelled_runs else "running"
            json_response(route, {"data": {
                "run_id": "local-active-run",
                "task_id": "old-task",
                "current_task_revision": 1,
                "status": status,
                "events": [],
                "error": {"message": "cancelled"} if status == "cancelled" else None,
            }})
            return
        if parsed.path == "/api/runs/local-fresh-run":
            json_response(route, {"data": {
                "run_id": "local-fresh-run",
                "task_id": "fresh-task",
                "current_task_revision": 1,
                "status": "completed",
                "events": [],
                "result": {"task_type": "chat", "reply": "\u65b0\u4f1a\u8bdd\u5df2\u6b63\u5e38\u54cd\u5e94"},
            }})
            return
        if parsed.path == "/api/runs/local-retry-run":
            json_response(route, {"data": {
                "run_id": "local-retry-run",
                "task_id": "retry-task",
                "current_task_revision": 1,
                "status": "completed",
                "events": [],
                "result": {"task_type": "chat", "reply": "\u91cd\u8bd5\u5df2\u6210\u529f"},
            }})
            return
        if parsed.path == "/api/weather":
            json_response(route, {"data": {"weather": "\u6674", "temperature": 20}})
            return
        if parsed.path == "/api/road-conditions":
            json_response(route, {"data": {"count": 0, "conditions": []}})
            return
        if parsed.path.startswith("/api/"):
            json_response(route, {"data": {}})
            return
        route.continue_()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.on("pageerror", lambda error: pytest.fail(f"browser page error: {error}"))
        page.route("**/*", handle_request)
        page.goto(static_site, wait_until="domcontentloaded")
        input_box = page.locator("#nl-input")
        submit = page.locator("#submit-btn")

        input_box.fill("\u7b2c\u4e00\u6b21\u63d0\u4ea4")
        with page.expect_response(lambda response: response.url.endswith("/messages") and response.request.method == "POST"):
            submit.click()
        expect(page.locator("#chat-content")).to_contain_text(
            "\u51fa\u4e86\u70b9\u5c0f\u95ee\u9898\uff0c\u7a0d\u7b49\u4e00\u4e0b\u518d\u8bd5\u5c31\u597d",
            timeout=10000,
        )
        assert input_box.input_value() == ""

        input_box.fill("\u7b2c\u4e00\u6b21\u63d0\u4ea4")
        with page.expect_response(lambda response: response.url.endswith("/messages") and response.request.method == "POST"):
            submit.click()
        expect(page.locator("#chat-content")).to_contain_text(
            "\u91cd\u8bd5\u5df2\u6210\u529f",
            timeout=10000,
        )
        assert len(submissions) == 2
        assert submissions[0] == submissions[1]
        assert submissions[0][0] == "local-conversation-1"

        input_box.fill("\u5904\u7406\u4e2d")
        with page.expect_response(lambda response: "/api/runs/local-active-run?" in response.url):
            submit.click()
        with page.expect_response(lambda response: response.url.endswith("/api/runs/local-active-run/cancel")) as cancelled:
            page.locator("#reset-btn").click()
        assert cancelled.value.status == 200
        assert "local-active-run" in cancelled_runs
        expect(page.locator("#chat-content .chat-bubble-row")).to_have_count(0)

        input_box.fill("\u65b0\u4f1a\u8bdd")
        with page.expect_response(lambda response: response.url.endswith("/messages") and response.request.method == "POST"):
            submit.click()
        expect(page.locator("#chat-content")).to_contain_text(
            "\u65b0\u4f1a\u8bdd\u5df2\u6b63\u5e38\u54cd\u5e94",
            timeout=10000,
        )
        assert len(submissions) == 4
        assert submissions[2] == ("local-conversation-1", "\u5904\u7406\u4e2d")
        assert submissions[-1] == ("local-conversation-2", "\u65b0\u4f1a\u8bdd")
        assert "\u7b2c\u4e00\u6b21\u63d0\u4ea4" not in page.locator("#chat-content").inner_text()

        context.close()
        browser.close()
