"""Opt-in browser review: notice clue → review → adopt → event → route detour.

Run with ``RUN_MANAGER_CLUE_BROWSER_E2E=1 python -m pytest -q
tests/browser/test_manager_clue_review.py``. The road network is a small
synthetic fixture, not the real campus graph; a passing test only proves the
manager workflow and routing consumption work, not that a real road condition
was verified on site.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from urllib.parse import urlsplit

import networkx as nx
import pytest
from werkzeug.serving import make_server

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None


RUN_E2E = os.getenv("RUN_MANAGER_CLUE_BROWSER_E2E") == "1" and sync_playwright is not None


def _haversine_m(lng1, lat1, lng2, lat2):
    radius = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _make_graph():
    graph = nx.MultiDiGraph()
    points = {
        0: (114.36000, 30.53000), 1: (114.36042, 30.53000),
        2: (114.36125, 30.53000), 3: (114.36167, 30.53000),
        4: (114.36000, 30.53040), 5: (114.36042, 30.53040),
        6: (114.36125, 30.53040), 7: (114.36167, 30.53040),
        8: (114.36042, 30.52991), 9: (114.36125, 30.52991),
    }
    for node, (x, y) in points.items():
        graph.add_node(node, x=x, y=y)
    for u, v, name in (
        (0, 1, "自强大道"), (1, 2, "自强大道"), (2, 3, "自强大道"),
        (4, 5, "北环路"), (5, 6, "北环路"), (6, 7, "北环路"),
        (0, 4, ""), (3, 7, ""), (1, 8, ""), (2, 9, ""),
    ):
        length = _haversine_m(points[u][0], points[u][1], points[v][0], points[v][1])
        attrs = {"length": length, "highway": "residential"}
        if name:
            attrs["name"] = name
        graph.add_edge(u, v, 0, **attrs)
        graph.add_edge(v, u, 0, **attrs)
    return graph


@pytest.mark.skipif(not RUN_E2E, reason="set RUN_MANAGER_CLUE_BROWSER_E2E=1 and install Playwright Chromium")
def test_manager_clue_review_publishes_event_and_route_detours(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(tmp_path / "road_conditions.json"))

    import config
    import api.routes as routes
    from app import create_app
    from spatial import road_clues
    from spatial import road_conditions as rc

    graph = _make_graph()
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "local-clue-e2e-only")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    conditions = tmp_path / "road_conditions.json"
    conditions.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    monkeypatch.setattr(rc, "_cache_error", None)
    monkeypatch.setattr(road_clues, "_CLUES_FILE", str(tmp_path / "road_clues.json"))
    road_clues.reset_collect_attempts()
    monkeypatch.setattr(routes, "_ensure_network", lambda: (graph, None))
    monkeypatch.setattr(road_clues, "extract_notice_draft", lambda _text: {
        "summary": "自强大道施工封闭",
        "confidence": 0.92,
        "draft": {
            "event_type": "construction",
            "road_hint": "自强大道",
            "description": "通知明确该路段施工期间禁止通行。",
            "action": "closure",
            "affected_modes": ["walk"],
        },
    })

    app = create_app()
    app.config.update(TESTING=True)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    user = app.test_client()

    def endpoint(node_id):
        return {"type": "coord", "coordinates": {
            "lng": float(graph.nodes[node_id]["x"]),
            "lat": float(graph.nodes[node_id]["y"]),
        }}

    baseline = user.post("/api/route", json={
        "start": endpoint(0), "end": endpoint(3),
        "travel_mode": "walk", "strategy": "shortest",
    })
    assert baseline.status_code == 200, baseline.get_json()
    baseline_edges = {tuple(edge) for edge in baseline.get_json()["data"]["shortest_edge_ids"]}

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            browser_errors = []
            page.on("pageerror", lambda error: browser_errors.append(str(error)))

            def local_backend_only(route_request):
                if urlsplit(route_request.request.url).hostname in {"127.0.0.1", "localhost"}:
                    route_request.continue_()
                else:
                    route_request.abort()

            page.route("**/*", local_backend_only)
            page.goto(f"{base_url}/manager", wait_until="domcontentloaded")
            expect(page.locator("#login-panel")).to_be_visible()
            page.locator("#password").fill("local-clue-e2e-only")
            page.locator("#login-form button[type='submit']").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)

            page.locator('.workflow-nav a[href="#clue-workspace"]').click()
            expect(page.locator("#clue-list")).to_contain_text(
                "目前没有线索记录", timeout=30000
            )
            page.locator("#clue-create > summary").click()
            page.locator("#clue-create details.vision-note-collapse summary").click()
            expect(page.locator("#clue-source-catalog")).to_contain_text("尚未接入")
            expect(page.locator("#clue-source-catalog")).to_contain_text("气象预警")
            page.locator("#clue-notice-media").fill("https://example.org/notice.png")
            page.locator("#clue-notice-submit").click()
            expect(page.locator("#clue-create-message")).to_contain_text(
                "不能自动抽取截图"
            )
            page.locator("#clue-notice-media").fill("")
            page.locator("#clue-notice-text").fill(
                "通知：自强大道自今日起进行路面施工，施工期间禁止通行，请师生绕行。"
            )
            page.locator("#clue-notice-submit").click()

            expect(page.locator("#clue-detail .clue-detail-head h3")).to_contain_text(
                "自强大道施工封闭", timeout=30000
            )
            expect(page.locator("#clue-candidate-list .clue-candidate")).to_have_count(
                3, timeout=30000
            )
            expect(page.locator("#clue-adopt")).to_be_disabled()
            page.locator("#clue-start-review").click()
            expect(page.locator("#clue-detail .clue-detail-head .event-status")).to_have_text(
                "复核中", timeout=30000
            )

            page.locator("#clue-candidate-list .clue-candidate").nth(1).locator("input").check()
            expect(page.locator("#clue-candidate-list .clue-candidate").nth(1)).to_have_class(
                re.compile(r"is-selected")
            )
            expect(page.locator("#clue-action")).to_have_value("closure")
            expect(page.locator("#clue-mode-walk")).to_be_checked()
            expect(page.locator("#clue-review-message")).to_contain_text(
                "道路绑定已核对", timeout=30000
            )
            page.locator("#clue-review-note").fill(
                "已核对官方通知原文与自强大道中间路段，确认施工时段和禁止通行影响。"
            )
            expect(page.locator("#clue-adopt")).to_be_enabled(timeout=30000)

            expect(page.locator("#clue-list .clue-row").first).to_contain_text("发布")
            clue_id = page.locator("#clue-detail-id").get_attribute("data-clue-id")
            assert clue_id
            page.locator("#clue-search").fill(clue_id)
            expect(page.locator("#clue-list .clue-row")).to_have_count(1, timeout=30000)
            page.locator("#clue-search").fill("")
            note_text = page.locator("#clue-review-note").input_value()
            page.locator("#refresh-clues").click()
            expect(page.locator("#clue-review-note")).to_have_value(note_text)

            page.locator("#clue-add-candidate").click()
            expect(page.locator("#clue-add-candidate")).to_have_text("取消地图选路")
            page.locator("#clue-add-candidate").click()
            expect(page.locator("#clue-add-candidate")).to_have_text("在地图补充候选路段")

            adopt_attempts = {"count": 0}

            def fail_first_adopt(route_request):
                if route_request.request.method == "POST" and adopt_attempts["count"] == 0:
                    adopt_attempts["count"] = 1
                    route_request.fulfill(
                        status=503,
                        content_type="application/json",
                        body=json.dumps({
                            "error": "temporary",
                            "message": "模拟采纳失败，请重试。",
                        }, ensure_ascii=False),
                    )
                else:
                    route_request.continue_()

            page.route("**/api/manager/clues/*/adopt", fail_first_adopt)
            page.locator("#clue-adopt").click()
            expect(page.locator("#clue-review-message")).to_contain_text(
                "模拟采纳失败", timeout=30000
            )
            expect(page.locator("#clue-detail .clue-detail-head .event-status")).to_have_text("复核中")
            expect(page.locator("#clue-adopt")).to_be_enabled(timeout=30000)

            page.locator("#clue-adopt").evaluate("(button) => { button.click(); button.click(); }")
            expect(page.locator("#clue-review-message")).to_contain_text(
                "已采纳为事件", timeout=30000
            )
            assert len(rc.list_conditions(include_inactive=True, strict=True)) == 1

            page.locator('.workflow-nav a[href="#event-history"]').click()
            event_row = page.locator("#event-list .event-row").filter(has_text="自强大道施工封闭")
            expect(event_row).to_have_count(1, timeout=30000)
            expect(event_row.locator(".event-status")).to_have_text("生效中", timeout=30000)

            stored = rc.list_conditions(include_inactive=True, strict=True)
            assert len(stored) == 1
            condition = stored[0]
            assert condition["source"]["kind"] == "clue"
            assert condition["source"]["clue_ids"]
            assert condition["action"] == "closure"

            closed = {tuple(edge) for edge in condition["edge"]["edges"]}
            assert closed & baseline_edges, "baseline route should use the road later closed"
            after = user.post("/api/route", json={
                "start": endpoint(0), "end": endpoint(3),
                "travel_mode": "walk", "strategy": "shortest",
            })
            assert after.status_code == 200, after.get_json()
            after_edges = {tuple(edge) for edge in after.get_json()["data"]["shortest_edge_ids"]}
            assert not (closed & after_edges), "route must avoid the adopted closure"

            assert browser_errors == []
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
