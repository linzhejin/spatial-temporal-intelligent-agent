"""Candidate names and explanations remain readable and clickable at all sizes."""
import json
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import expect, sync_playwright

from tests.browser.test_strategy_and_food_paging import static_site  # noqa: F401


@pytest.mark.parametrize("width", [1280, 390, 320], ids=["desktop", "mobile", "narrow-mobile"])
def test_candidate_text_layout_and_selection(static_site, width, tmp_path):  # noqa: F811
    submissions = []
    candidates = [
        {"poi_id": "poi_1", "name": "赛百味（武汉大学店）", "subcategory": "restaurant",
         "recommendation_reason": "位于工学部三食堂2层"},
        {"poi_id": "poi_2", "name": "朴大叔拌饭（武汉大学店）", "subcategory": "restaurant",
         "recommendation_reason": "位于学生三食堂2楼", "distance_m": 358,
         "distance_reference_name": "工学部"},
        {"poi_id": "poi_3", "name": "武汉大学工学部学生一食堂", "subcategory": "canteen",
         "distance_m": 550, "distance_reference_name": "工学部"},
        {"poi_id": "poi_4", "name": "武汉大学工学部特别长的餐饮地点名称",
         "subcategory": "restaurant", "recommendation_reason": "位于食堂二楼靠近入口的位置，提供多种餐饮选择。",
         "distance_m": 1200, "distance_reference_name": "工学部校区东门附近"},
    ]

    def respond(route, data):
        route.fulfill(status=200, content_type="application/json; charset=utf-8",
                      body=json.dumps({"data": data}, ensure_ascii=False))

    def handle(route):
        request = route.request
        url = urlsplit(request.url)
        if url.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
        elif url.path == "/api/conversations" and request.method == "POST":
            respond(route, {"conversation_id": "layout-test"})
        elif url.path.endswith("/messages") and request.method == "POST":
            submissions.append(request.post_data_json)
            respond(route, {"run_id": f"layout-run-{len(submissions)}"})
        elif url.path.startswith("/api/runs/"):
            result = ({"task_type": "chat", "response_kind": "chat", "reply": "已选择地点。"}
                      if submissions[-1].get("selected_poi_id") else
                      {"task_type": "path_planning", "response_kind": "candidates",
                       "message": "请选择地点。", "candidates": candidates})
            respond(route, {"status": "completed", "events": [], "task_id": "layout-task",
                            "current_task_revision": 1, "result": result})
        elif url.path.startswith("/api/"):
            respond(route, {})
        else:
            route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        mobile = width < 600
        context = browser.new_context(viewport={"width": width, "height": 900 if not mobile else 844},
                                      is_mobile=mobile, has_touch=mobile, service_workers="block")
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.route("**/*", handle)
        page.goto(static_site, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        page.locator("#nl-input").fill("我想在工学部找吃饭的地方")
        page.locator("#submit-btn").click()
        cards = page.locator(".candidate-cards").last.locator(".candidate-card")
        expect(cards).to_have_count(4)

        for index in range(4):
            card = cards.nth(index)
            name = card.locator(".candidate-name")
            expect(name).to_have_text(candidates[index]["name"])
            card_box, name_box = card.bounding_box(), name.bounding_box()
            assert name_box["width"] >= min(100, card_box["width"] * 0.35), (width, index, name_box)
            assert name_box["height"] < 85, (width, index, name_box)
            for selector in (".candidate-name", ".candidate-reason", ".candidate-meta"):
                part = card.locator(selector)
                if not part.count():
                    continue
                rect = part.bounding_box()
                assert rect["x"] >= card_box["x"] - 1 and rect["x"] + rect["width"] <= card_box["x"] + card_box["width"] + 1
                assert part.evaluate("el => el.scrollWidth <= el.clientWidth + 1"), (width, index, selector)
            reason = card.locator(".candidate-reason")
            if reason.count():
                assert reason.evaluate("""el => {
                    const reason = el.getBoundingClientRect();
                    const card = el.closest('.candidate-card');
                    const name = card.querySelector('.candidate-name').getBoundingClientRect();
                    const meta = card.querySelector('.candidate-meta');
                    return reason.top >= name.bottom - 1 &&
                        (!meta || reason.top >= meta.getBoundingClientRect().bottom - 1);
                }"""), (width, index)

        expect(cards.nth(2).locator(".candidate-dist")).to_contain_text("550")
        cards.nth(1).screenshot(path=str(tmp_path / f"candidate-layout-{width}.png"))
        cards.nth(1).click()
        expect(page.get_by_text("已选择地点。", exact=False)).to_be_visible()
        assert submissions[-1]["selected_poi_id"] == "poi_2"
        assert submissions[-1]["continuation_task_id"] == "layout-task"
        context.close()
        browser.close()
