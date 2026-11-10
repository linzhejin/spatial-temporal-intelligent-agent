"""Actual upload, worker, browser, database and route flow with synthetic analysis."""
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from werkzeug.serving import make_server

from storage import database, vision_repository
from tests.browser.test_manager_vision_backend_e2e import _real_network_anchor

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


@pytest.mark.skipif(os.getenv("RUN_MANAGER_VISION_BACKEND_E2E") != "1" or sync_playwright is None,
                    reason="requires isolated PostgreSQL and Playwright")
def test_video_upload_worker_candidate_suggested_edge_and_route(monkeypatch, tmp_path):
    import api.routes as routes
    import config
    from app import create_app
    from spatial import network, road_conditions
    from vision import worker

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "isolated-upload-test")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(tmp_path / "media"))
    monkeypatch.setattr(config, "VISION_CHUNK_BYTES", 4)
    monkeypatch.setattr(config, "VISION_MAX_VIDEO_BYTES", 1024)
    monkeypatch.setattr(routes, "_vision_inference_readiness", lambda *_: (True, True, True, True))
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(tmp_path / "conditions.json"))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(routes, "_network_initialized", False)
    Path(config.VISION_UPLOAD_DIR).mkdir()
    Path(road_conditions._CONDITIONS_FILE).write_text("[]", encoding="utf-8")
    database.initialize(DATABASE_URL)
    anchor = _real_network_anchor()

    observed_claim = []
    real_claim = vision_repository.claim_next_job
    def capture_claim(*args, **kwargs):
        result = real_claim(*args, **kwargs)
        if result:
            observed_claim.append(result)
        return result
    monkeypatch.setattr(vision_repository, "claim_next_job", capture_claim)
    candidates = [
        {"kind": "possible_crowding", "review_required": True, "auto_publish": False,
         "confidence": .6, "reason": "隔离测试的第一个候选", "evidence": {"segments": [], "summary": {}}},
        {"kind": "vehicle_cluster_review", "review_required": True, "auto_publish": False,
         "confidence": .7, "reason": "隔离测试的第二个候选", "evidence": {"segments": [], "summary": {}}},
    ]
    monkeypatch.setattr(worker, "analyze_media", lambda *args, **kwargs: {
        "media": {"kind": "video", "width": 640, "height": 480},
        "metrics": {"frames_analyzed": 1}, "candidates": candidates,
        "safety": {"automatically_changes_routing": False},
    })
    monkeypatch.setattr(worker, "attach_evidence_clips", lambda *args, **kwargs: {"clips_created": 0})
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="isolated-upload-test")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    job_id = None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True,
                                    has_touch=True, service_workers="block")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", lambda route: route.continue_()
                       if urlsplit(route.request.url).hostname in {"127.0.0.1", "localhost"}
                       else route.abort())
            page.goto(base + "/manager", wait_until="domcontentloaded")
            page.locator("#password").fill("isolated-upload-test")
            page.locator("#login-form button[type=submit]").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            csrf = page.request.get(base + "/api/admin/status").json()["data"]["csrf_token"]
            headers = {"X-CSRF-Token": csrf}
            pose = {"crs": "WGS84", "parsed": True, "trajectory": [{
                "lng": anchor["wgs"]["lng"], "lat": anchor["wgs"]["lat"],
                "seconds": 0, "rel_alt_m": 80, "gb_pitch_deg": -45,
                "gb_yaw_deg": 0, "focal_len_35mm": 24,
            }]}
            started = page.request.post(base + "/api/manager/vision-uploads", headers=headers, data={
                "filename": "synthetic-upload.mp4", "size": 12, "captured_at": time.time(),
                "lng": anchor["lng"], "lat": anchor["lat"], "camera_pose": json.dumps(pose),
                "capture_time_source": "dji_srt", "capture_timezone": "Asia/Shanghai",
            })
            assert started.status == 201, started.text()
            upload_id = started.json()["data"]["upload_id"]
            for offset, chunk in ((0, b"\x00\x00\x00\x18"), (4, b"ftyp"), (8, b"isom")):
                response = page.request.put(base + f"/api/manager/vision-uploads/{upload_id}/chunks",
                    headers={**headers, "Upload-Offset": str(offset), "Content-Type": "application/octet-stream"},
                    data=chunk)
                assert response.status == 200, response.text()
            finished = page.request.post(base + f"/api/manager/vision-uploads/{upload_id}/complete",
                                         headers=headers, data={})
            assert finished.status == 202, finished.text()
            job_id = finished.json()["data"]["job"]["job_id"]
            page.reload(wait_until="domcontentloaded")
            card = page.locator(".vision-job").filter(has_text="synthetic-upload.mp4")
            expect(card.locator(".vision-job-state")).to_have_text("排队中", timeout=30000)
            page.once("dialog", lambda dialog: dialog.accept())
            card.get_by_role("button", name="取消分析").click()
            expect(card.locator(".vision-job-state")).to_have_text("已取消", timeout=30000)
            page.once("dialog", lambda dialog: dialog.accept())
            card.get_by_role("button", name="从断点重试").click()
            expect(card.locator(".vision-job-state")).to_have_text("排队中", timeout=30000)
            outcome = worker.process_next_job(DATABASE_URL, "isolated-upload-worker", lease_seconds=30)
            assert outcome == {"job_id": job_id, "status": "needs_review"}
            assert observed_claim[0]["camera_pose"]["trajectory"][0]["rel_alt_m"] == 80
            assert vision_repository.get_job(DATABASE_URL, job_id)["camera_pose"] == observed_claim[0]["camera_pose"]

            fail_suggestion = {"once": True}
            def fail_first_suggestion(route):
                if fail_suggestion["once"]:
                    fail_suggestion["once"] = False
                    route.fulfill(status=503, json={"error": "road_suggestions_unavailable",
                                                    "message": "邻近道路暂时不可用，请重试。"})
                else:
                    route.fallback()
            page.route("**/suggested-edges", fail_first_suggestion)
            page.reload(wait_until="domcontentloaded")
            card = page.locator(".vision-job").filter(has_text="synthetic-upload.mp4")
            expect(card).to_have_count(1, timeout=30000)
            retry = card.get_by_role("button", name="重试道路查询")
            expect(retry).to_be_visible(timeout=30000)
            retry.click()
            candidate = card.locator(".candidate-list:not(.suggested-edges) .candidate-card").nth(1)
            card.locator("textarea.vision-review-note").fill("隔离测试：第二项影像候选已经单独核对。")
            candidate.get_by_role("button", name="确认此候选").click()
            expect(candidate).to_contain_text("已确认", timeout=30000)
            candidate.get_by_role("button", name="将此候选转入道路事件").click()
            suggestion = card.locator(".suggested-edges .candidate-card").first
            expect(suggestion).to_be_visible(timeout=30000)
            suggestions = page.request.get(base + f"/api/manager/vision-jobs/{job_id}/suggested-edges").json()["data"]
            assert suggestions["basis"] == "radius_only"
            expected = suggestions["suggested_edges"][0]
            with page.expect_response(lambda r: "/api/road-conditions/snap?" in r.url and r.status == 200,
                                      timeout=30000) as snapped:
                suggestion.get_by_role("button", name="在地图定位并预填发布落点").click()
            snap = snapped.value.json()["data"]["snap"]
            assert (snap["u"], snap["v"], snap["key"]) == (expected["u"], expected["v"], expected["key"])
            graph = network.get_network()
            user = app.test_client()
            def endpoint(node):
                return {"type": "coord", "coordinates": {
                    "lng": float(graph.nodes[node]["x"]), "lat": float(graph.nodes[node]["y"]),
                }}
            baseline = user.post("/api/route", json={
                "start": endpoint(snap["u"]), "end": endpoint(snap["v"]),
                "travel_mode": "walk", "strategy": "shortest",
            })
            assert baseline.status_code == 200, baseline.get_json()
            route_state = baseline.get_json()["data"]["route_state"]
            snap_edges = set(map(tuple, snap["edges"]))
            assert snap_edges.intersection(map(tuple, baseline.get_json()["data"]["shortest_edge_ids"]))
            page.locator("#event-type").select_option("event")
            page.locator("#event-action").select_option("closure")
            page.locator("#event-mode-walk").check()
            page.locator("#event-name").fill("隔离影像建议道路验收")
            page.locator("#field-confirmation").fill("隔离测试：管理员核对了画面候选和道路。")
            page.locator("#confirm-picked-road").click()
            expect(page.locator("#publish-event")).to_be_enabled(timeout=30000)
            with page.expect_response(lambda r: r.url.endswith("/api/road-conditions") and
                                      r.request.method == "POST", timeout=30000) as publication:
                page.locator("#publish-event").click()
            response = publication.value
            assert response.status == 201, response.text()
            body = response.request.post_data_json
            assert body["source_vision_candidate_index"] == 1
            assert body["edge_ref"] == {key: expected[key] for key in ("u", "v", "key")}
            condition = response.json()["data"]["condition"]
            assert condition["edge"]["edges"] == snap["edges"]
            assert condition["source"]["candidate_index"] == 1
            closed = road_conditions.resolve_condition_effects(graph, [condition], "walk", strict=True)[1]
            assert set(map(tuple, snap["edges"])) & closed
            replanned = user.post("/api/route/replan", json={
                "route_state": route_state, "change": {"refresh_road_conditions": True},
            })
            assert replanned.status_code == 200, replanned.get_json()
            assert not snap_edges.intersection(map(tuple, replanned.get_json()["data"]["shortest_edge_ids"]))
            assert page.evaluate("document.documentElement.scrollWidth") <= 390
            page.screenshot(path=str(tmp_path / "vision-upload-flow-mobile.png"), full_page=True)
            row = page.locator("#event-list .event-row").filter(has_text="隔离影像建议道路验收")
            expect(row).to_have_count(1, timeout=30000)
            page.once("dialog", lambda dialog: dialog.accept())
            row.get_by_role("button", name="永久删除").click()
            expect(row).to_have_count(0, timeout=30000)
            assert not road_conditions.list_conditions(strict=True)
            restored = user.post("/api/route/replan", json={
                "route_state": route_state, "change": {"refresh_road_conditions": True},
            })
            assert restored.status_code == 200, restored.get_json()
            assert snap_edges.intersection(map(tuple, restored.get_json()["data"]["shortest_edge_ids"]))
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if job_id:
            with database.connect(DATABASE_URL) as conn:
                conn.execute("DELETE FROM manager_vision_job WHERE job_id=%s", (job_id,))
