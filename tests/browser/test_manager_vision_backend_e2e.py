"""Opt-in browser-to-PostgreSQL integration for reviewed vision events.

Run with ``RUN_MANAGER_VISION_BACKEND_E2E=1`` and an explicitly isolated
``TEST_DATABASE_URL``. The vision job uses synthetic evidence; this verifies
the review/publication plumbing, not model accuracy or a real road incident.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from psycopg.types.json import Jsonb
from werkzeug.serving import make_server

from storage import database, vision_repository

DATABASE_URL = None  # assigned only by the isolated database guard fixture
pytestmark = pytest.mark.usefixtures("isolated_test_database")

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None


RUN_E2E = os.getenv("RUN_MANAGER_VISION_BACKEND_E2E") == "1" and sync_playwright is not None


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set RUN_MANAGER_VISION_BACKEND_E2E=1 and install Playwright Chromium",
)
def test_manager_reviews_persisted_candidate_and_publishes_user_route_event(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(tmp_path / "road_conditions.json"))
    monkeypatch.setenv("VISION_UPLOAD_DIR", str(tmp_path / "vision_uploads"))

    import config
    import api.routes as routes
    from app import create_app
    from agents.tools import execute_tool
    from spatial import network, road_conditions

    database.initialize(DATABASE_URL)
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "local-vision-e2e-only")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(tmp_path / "vision_uploads"))
    Path(config.VISION_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(tmp_path / "road_conditions.json"))
    (tmp_path / "road_conditions.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    monkeypatch.setattr(routes, "_network_initialized", False)
    monkeypatch.setattr(network, "_G", None)

    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    job_id = None
    try:
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8Cq3sAAAAASUVORK5CYII="
        )
        media_path = Path(config.VISION_UPLOAD_DIR) / "synthetic-campus-evidence.png"
        media_path.write_bytes(png)
        job = vision_repository.create_job(
            DATABASE_URL,
            created_by="local-e2e",
            original_name=media_path.name,
            media_kind="image",
            media_path=media_path.name,
            sha256="f" * 64,
            anchor_gcj={"lng": 114.363, "lat": 30.5365},
            captured_at=datetime.now(timezone.utc),
            capture_time_source="manual",
        )
        job_id = job["job_id"]
        candidate = {
            "kind": "vehicle_cluster_review",
            "review_required": True,
            "auto_publish": False,
            "confidence": 0.74,
            "reason": "合成测试候选：持续低速车辆线索，需管理员复核。",
            "evidence": {
                "segments": [{"start_seconds": 2.0, "end_seconds": 5.0}],
                "summary": {"peak_vehicle_count": 3, "private_media_path": "must-not-leak"},
            },
        }
        rejected_candidate = {
            "kind": "possible_crowding",
            "review_required": True,
            "auto_publish": False,
            "confidence": 0.68,
            "reason": "合成测试候选：另一处人群聚集线索，单独核对是否影响道路。",
            "evidence": {
                "segments": [{"start_seconds": 8.0, "end_seconds": 11.0}],
                "summary": {"peak_pedestrian_count": 5},
            },
        }
        with database.connect(DATABASE_URL) as conn:
            conn.execute(
                "UPDATE manager_vision_job SET status='needs_review', result=%s,"
                " progress=%s, captured_at=to_timestamp(%s) WHERE job_id=%s",
                (
                    Jsonb({"candidates": [candidate, rejected_candidate], "safety": {"automatically_changes_routing": False}}),
                    Jsonb({"phase": "complete", "percent": 100}),
                    time.time(),
                    job_id,
                ),
            )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            browser_errors = []
            snapped_roads = []
            published_posts = []
            page.on("pageerror", lambda error: browser_errors.append(str(error)))

            def remember_publication(request):
                if urlsplit(request.url).path == "/api/road-conditions" and request.method == "POST":
                    published_posts.append({
                        "body": request.post_data_json,
                        "csrf": request.headers.get("x-csrf-token", ""),
                    })

            page.on("request", remember_publication)

            def remember_snap(response):
                if "/api/road-conditions/snap" in response.url and response.status == 200:
                    payload = response.json()
                    if payload.get("data", {}).get("snap"):
                        snapped_roads.append(payload["data"]["snap"])

            page.on("response", remember_snap)

            def local_backend_only(route_request):
                if urlsplit(route_request.request.url).hostname in {"127.0.0.1", "localhost"}:
                    route_request.continue_()
                else:
                    route_request.abort()

            page.route("**/*", local_backend_only)
            page.goto(f"{base_url}/manager", wait_until="domcontentloaded")
            expect(page.locator("#login-panel")).to_be_visible()
            page.locator("#password").fill("local-vision-e2e-only")
            page.locator("#login-form button[type='submit']").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)

            job_card = page.locator(".vision-job").filter(has_text=media_path.name)
            expect(job_card).to_have_count(1, timeout=30000)
            candidate_cards = job_card.locator(".candidate-card")
            expect(candidate_cards).to_have_count(2)
            review_note = job_card.locator("textarea.vision-review-note")
            review_note.fill("合成验收：第二项画面线索不构成道路影响，单独排除。")
            candidate_cards.nth(1).get_by_role("button", name="排除此候选").click()
            expect(candidate_cards.nth(1)).to_contain_text("已排除", timeout=30000)
            expect(candidate_cards.nth(1).get_by_role("button", name="将此候选转入道路事件")).to_have_count(0)

            review_note.fill("合成验收：第一项车辆线索已单独核对，可进入道路事件草稿。")
            candidate_cards.nth(0).get_by_role("button", name="确认此候选").click()
            expect(candidate_cards.nth(0)).to_contain_text("已确认", timeout=30000)

            candidate_cards.nth(0).get_by_role("button", name="将此候选转入道路事件").click()
            expect(page.locator("#vision-source-banner")).to_be_visible()
            expect(page.locator("#field-confirmation-wrap")).to_be_visible()
            page.locator("#event-type").select_option("event")
            page.locator("#event-action").select_option("closure")
            page.locator("#event-mode-walk").check()
            page.locator("#event-name").fill("合成影像复核封路验收")
            page.locator("#field-confirmation").fill("合成验收记录：管理员确认事件来源和所选道路，仅用于隔离环境联动测试。")

            map_element = page.locator("#manager-map")
            map_box = map_element.bounding_box()
            assert map_box is not None
            sample_positions = [
                (0.50, 0.50), (0.36, 0.50), (0.64, 0.50),
                (0.50, 0.34), (0.50, 0.66), (0.38, 0.36), (0.62, 0.64),
            ]
            selected_snap = None
            selected_position = None
            for index, (x_ratio, y_ratio) in enumerate(sample_positions):
                if index:
                    page.locator("#pick-road").click()
                selected_position = {
                    "x": map_box["width"] * x_ratio,
                    "y": map_box["height"] * y_ratio,
                }
                with page.expect_response(
                    lambda response: "/api/road-conditions/snap" in response.url
                    and response.status == 200,
                    timeout=30000,
                ) as snap_response:
                    map_element.click(position=selected_position)
                expect(page.locator("#selected-road")).to_contain_text("吸附距离", timeout=30000)
                selected_snap = snap_response.value.json()["data"]["snap"]
                if any(int(edge[0]) != int(edge[1]) for edge in selected_snap["edges"]):
                    break
            assert selected_snap and any(
                int(edge[0]) != int(edge[1]) for edge in selected_snap["edges"]
            ), "browser map picks should resolve to a traversable road edge"

            graph = network.get_network()
            start_node, end_node = selected_snap["u"], selected_snap["v"]

            def endpoint(node_id):
                return {
                    "type": "coord",
                    "coordinates": {
                        "lng": float(graph.nodes[node_id]["x"]),
                        "lat": float(graph.nodes[node_id]["y"]),
                    },
                }

            user = app.test_client()
            baseline = user.post(
                "/api/route",
                json={
                    "start": endpoint(start_node),
                    "end": endpoint(end_node),
                    "travel_mode": "walk",
                    "strategy": "shortest",
                },
            )
            assert baseline.status_code == 200, baseline.get_json()
            baseline_data = baseline.get_json()["data"]
            target_edges = {tuple(edge) for edge in selected_snap["edges"]}
            assert target_edges.intersection(
                tuple(edge) for edge in baseline_data["shortest_edge_ids"]
            )

            page.locator("#clear-picked-road").click()
            expect(page.locator("#selected-road")).to_contain_text("尚未选择道路")
            expect(page.locator("#publish-event")).to_be_disabled()
            page.locator("#pick-road").click()
            with page.expect_response(
                lambda response: "/api/road-conditions/snap" in response.url
                and response.status == 200,
                timeout=30000,
            ) as repick_response:
                map_element.click(position=selected_position)
            selected_snap = repick_response.value.json()["data"]["snap"]
            target_edges = {tuple(edge) for edge in selected_snap["edges"]}
            assert target_edges.intersection(
                tuple(edge) for edge in baseline_data["shortest_edge_ids"]
            )
            impact_preview = page.locator("#impact-preview")
            expect(impact_preview).to_be_visible()
            expect(impact_preview).to_contain_text("发布影响预览")
            expect(impact_preview).to_contain_text("步行：该方式在所选路段禁行")
            expect(impact_preview).to_contain_text("骑行：不受该事件影响")
            expect(impact_preview).to_contain_text("驾车：不受该事件影响")

            page.locator("#confirm-picked-road").click()
            expect(page.locator("#publish-event")).to_be_enabled()
            page.locator("#publish-event").click()
            expect(page.locator("#form-message")).to_contain_text("事件已发布", timeout=30000)
            assert len(published_posts) == 1

            persisted_job = vision_repository.get_job(DATABASE_URL, job_id)
            reviews = persisted_job["candidate_reviews"]
            assert reviews["0"]["status"] == "confirmed"
            assert reviews["0"]["reviewed_by"] == "web"
            assert "可进入道路事件草稿" in reviews["0"]["note"]
            assert reviews["1"]["status"] == "dismissed"
            assert reviews["1"]["reviewed_by"] == "web"
            assert "单独排除" in reviews["1"]["note"]

            conditions = road_conditions.list_conditions(strict=True)
            assert len(conditions) == 1
            event = conditions[0]
            assert event["action"] == "closure"
            assert event["source"]["kind"] == "vision_job"
            assert event["source"]["job_id"] == job_id
            assert event["source"]["candidate_index"] == 0
            assert event["source"]["reviewed_by"] == "web"
            assert event["source"]["evidence_reference"]["segments"] == [
                {"start_seconds": 2.0, "end_seconds": 5.0}
            ]
            assert "media_path" not in event["source"]

            duplicate = page.evaluate(
                """async ({body, csrf}) => {
                    const response = await fetch('/api/road-conditions', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf},
                        body: JSON.stringify(body),
                    });
                    return {status: response.status, payload: await response.json()};
                }""",
                published_posts[0],
            )
            assert duplicate["status"] == 200
            assert duplicate["payload"]["data"]["already_published"] is True
            assert duplicate["payload"]["data"]["condition"]["id"] == event["id"]
            assert len(road_conditions.list_conditions(strict=True)) == 1

            event_row = page.locator("#event-list .event-row").filter(has_text=event["name"])
            expect(event_row).to_have_count(1, timeout=30000)
            event_row.get_by_role("button", name="修改事件").click()
            event_row.screenshot(path=str(tmp_path / "manager-event-edit.png"))
            edit_name = "管理员修订后的影像封路验收"
            event_row.locator(".event-edit-name").fill(edit_name)
            event_row.locator(".event-edit-description").fill("管理员补充了核对后的事件说明。")
            new_end = page.evaluate(
                "() => new Date(Date.now() + 2 * 3600000 + 8 * 3600000).toISOString().slice(0, 19)"
            )
            invalid_start = page.evaluate(
                "() => new Date(Date.now() + 4 * 3600000 + 8 * 3600000).toISOString().slice(0, 19)"
            )
            event_row.locator(".event-edit-start").fill(invalid_start)
            event_row.locator(".event-edit-end").fill(new_end)
            event_row.locator(".event-edit-save").click()
            expect(event_row.locator(".event-edit-message")).to_contain_text("结束时间必须晚于开始时间")
            assert road_conditions.list_conditions(strict=True)[0]["name"] == event["name"]
            valid_start = page.evaluate(
                "() => new Date(Date.now() - 60000 + 8 * 3600000).toISOString().slice(0, 19)"
            )
            event_row.locator(".event-edit-start").fill(valid_start)
            event_row.locator(".event-edit-save").click()
            event_row = page.locator("#event-list .event-row").filter(has_text=edit_name)
            expect(event_row.locator(".event-name")).to_contain_text(edit_name, timeout=30000)
            persisted_edit = road_conditions.list_conditions(strict=True)[0]
            assert persisted_edit["name"] == edit_name
            assert persisted_edit["description"] == "管理员补充了核对后的事件说明。"
            assert persisted_edit["end_time"] > event["end_time"]
            assert persisted_edit["audit"][-1]["action"] == "updated"

            snapshot = user.get("/api/road-conditions")
            assert snapshot.status_code == 200
            assert snapshot.get_json()["data"]["count"] == 1
            assert snapshot.get_json()["data"]["conditions"][0]["name"] == edit_name
            tool_result, _ = execute_tool("list_road_conditions", {}, {})
            assert tool_result["count"] == 1
            assert tool_result["conditions"][0]["source_kind"] == "vision_job"
            assert tool_result["conditions"][0]["vision_evidence"]["job_id"] == job_id
            assert tool_result["conditions"][0]["route_effect"]["action"] == "closure"
            replanned = user.post(
                "/api/route/replan",
                json={
                    "route_state": baseline_data["route_state"],
                    "change": {"refresh_road_conditions": True},
                },
            )
            assert replanned.status_code == 200, replanned.get_json()
            closed_ids = {tuple(edge) for edge in event["edge"]["edges"]}
            route_ids = {tuple(edge) for edge in replanned.get_json()["data"]["shortest_edge_ids"]}
            assert not closed_ids.intersection(route_ids)

            monkeypatch.setattr(road_conditions, "_cache", None)
            monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
            restarted_app = create_app()
            restarted = restarted_app.test_client().get("/api/road-conditions")
            assert restarted.status_code == 200
            assert restarted.get_json()["data"]["conditions"][0]["id"] == event["id"]
            assert restarted.get_json()["data"]["conditions"][0]["name"] == edit_name

            event_row = page.locator("#event-list .event-row").filter(has_text=edit_name)
            page.once("dialog", lambda dialog: dialog.accept())
            event_row.get_by_role("button", name="永久删除").click()
            expect(page.locator("#event-list .event-row")).to_have_count(0, timeout=30000)
            deleted_snapshot = user.get("/api/road-conditions")
            assert deleted_snapshot.status_code == 200
            assert deleted_snapshot.get_json()["data"]["count"] == 0
            cleared_tool_result, _ = execute_tool("list_road_conditions", {}, {})
            assert cleared_tool_result["count"] == 0
            after_delete = user.post(
                "/api/route/replan",
                json={
                    "route_state": baseline_data["route_state"],
                    "change": {"refresh_road_conditions": True},
                },
            )
            assert after_delete.status_code == 200, after_delete.get_json()
            after_delete_edges = {
                tuple(edge) for edge in after_delete.get_json()["data"]["shortest_edge_ids"]
            }
            assert closed_ids.intersection(after_delete_edges)
            assert road_conditions.list_conditions(include_inactive=True, strict=True) == []
            assert browser_errors == []
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        if job_id:
            with database.connect(DATABASE_URL) as conn:
                conn.execute("DELETE FROM manager_vision_job WHERE job_id=%s", (job_id,))
