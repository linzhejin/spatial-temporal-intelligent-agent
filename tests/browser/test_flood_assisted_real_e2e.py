"""Real Doodleverse image, PostgreSQL, browser review, edge binding and publication."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from werkzeug.serving import make_server

from storage import database, vision_repository

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


@pytest.mark.skipif(os.getenv("RUN_FLOOD_ASSISTED_REAL_E2E") != "1" or sync_playwright is None,
                    reason="requires explicit isolated PostgreSQL, real model weights and Playwright")
@pytest.mark.parametrize("impact", ["notice", "slowdown", "closure"])
def test_real_flood_image_upload_review_road_and_event(monkeypatch, tmp_path, impact):
    import api.routes as routes
    import config
    from app import create_app
    from spatial import network, road_conditions
    from vision import worker
    from vision.specialized import DoodleverseFloodSegmenter

    evidence = Path(r"D:\LuoJiaValidation")
    image = evidence / "model_comparison_20261109/flood/road_inundation_delta_ian_2024/samples_v1_2/test/HURRICANE-IAN_000352_image.png"
    flood_path = evidence / "doodleverse_floodnet_v7_fp32.onnx"
    runtime = evidence / "flood_runtime_release_20261110/Scripts/python.exe"
    traffic_path = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    assert all(path.is_file() for path in (image, flood_path, runtime, traffic_path))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    conditions = tmp_path / "road_conditions.json"
    conditions.write_text("[]", encoding="utf-8")
    model_hash = hashlib.sha256(flood_path.read_bytes()).hexdigest()
    values = {
        "ROAD_CONDITION_ADMIN_PASSWORD": "isolated-flood-browser-only",
        "ROAD_CONDITION_ADMIN_TOKEN": "", "VISION_UPLOAD_DIR": str(uploads),
        "VISION_MODEL_PATH": str(traffic_path), "VISION_TRIAL_ENABLED": True,
        "VISION_FLOOD_ENABLED": True, "VISION_FLOOD_RELEASE_MODE": "assisted_review",
        "VISION_FLOOD_MODEL_PATH": str(flood_path),
        "VISION_FLOOD_REVISION": "doodleverse-floodnet-resunet-v7",
    }
    for key, value in values.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(conditions))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(routes, "_network_initialized", False)
    # Only the test DB receives this injected capability; no release evidence is fabricated.
    monkeypatch.setattr(routes, "_vision_flood_assistance_ready", lambda caps: bool(
        caps.get("flood_assistance_supported") and caps.get("flood_assistance_model_sha256") == model_hash
    ))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    capability = {
        "flood_assistance_supported": True,
        "flood_assistance_release_status": "test_injected",
        "flood_assistance_model_id": "doodleverse-floodnet-resunet-v7",
        "flood_assistance_model_version": config.VISION_FLOOD_REVISION,
        "flood_assistance_model_sha256": model_hash,
        "trial_flood_supported": True, "trial_flood_media": "image",
    }
    heartbeat_stop = threading.Event()

    def keep_heartbeats():
        while not heartbeat_stop.is_set():
            vision_repository.heartbeat_worker(DATABASE_URL, "flood-browser-operational", ready=True, role="operational")
            vision_repository.heartbeat_worker(DATABASE_URL, "flood-browser-trial", ready=True, role="trial",
                                               status_detail=json.dumps(capability))
            heartbeat_stop.wait(5)

    heartbeat = threading.Thread(target=keep_heartbeats, daemon=True)
    heartbeat.start()
    model = DoodleverseFloodSegmenter(str(flood_path), model_version=config.VISION_FLOOD_REVISION,
                                     runtime_python=str(runtime))
    model.model_sha256 = model_hash
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="isolated-flood-browser")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 390, "height": 844},
                                          is_mobile=True, has_touch=True, service_workers="block")
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", lambda route: route.continue_()
                       if urlsplit(route.request.url).hostname in {"127.0.0.1", "localhost"}
                       else route.abort())
            page.goto(base + "/manager", wait_until="networkidle")
            page.locator("#password").fill("isolated-flood-browser-only")
            page.locator("#login-form button[type=submit]").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            page.get_by_text("新建影像分析", exact=True).click()
            expect(page.locator("#vision-flood-assistance-option")).to_be_enabled(timeout=30000)
            page.locator("#vision-purpose").select_option("flood_assisted")
            page.locator("#media-file").set_input_files(str(image))
            page.locator("#captured-at").fill(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
            page.locator("#pick-anchor").click()
            map_element = page.locator("#manager-map")
            map_element.click(position={"x": 100, "y": 100})
            expect(page.locator("#selected-anchor")).not_to_contain_text("尚未标注", timeout=30000)
            page.wait_for_function("document.querySelector('#observation-image').naturalWidth > 0")
            page.locator("#observation-kind").select_option("road_surface")
            page.locator("#start-region").click()
            canvas = page.locator("#observation-canvas")
            bounds = canvas.bounding_box()
            assert bounds
            for x, y in ((.02, .02), (.98, .02), (.98, .98), (.02, .98)):
                canvas.click(position={"x": bounds["width"] * x, "y": bounds["height"] * y})
            page.locator("#finish-region").click()
            with page.expect_response(lambda response: response.url.endswith("/api/manager/vision-jobs")
                                      and response.request.method == "POST", timeout=30000) as submitted:
                page.locator("#submit-vision").click()
            assert submitted.value.status == 202, submitted.value.text()
            job_id = submitted.value.json()["data"]["job"]["job_id"]
            outcome = worker.process_next_job(DATABASE_URL, "flood-browser-actual-worker",
                                              trial_flood_segmenter=model, trial_only=True,
                                              assisted_flood_supported=True)
            assert outcome == {"job_id": job_id, "status": "needs_review"}
            stored = vision_repository.get_job(DATABASE_URL, job_id)
            assert stored["result"]["flood_assistance"]["model_sha256"] == model_hash
            flood_indices = [index for index, item in enumerate(stored["result"]["candidates"])
                             if item.get("kind") == "possible_flooding"]
            assert len(flood_indices) == 1
            page.reload(wait_until="networkidle")
            card = page.locator(".vision-job").filter(has_text=image.name)
            expect(card).to_be_visible(timeout=30000)
            candidate = card.locator(".candidate-list:not(.suggested-edges) .candidate-card").nth(flood_indices[0])
            expect(candidate).to_contain_text("积水")
            card.locator("textarea.vision-review-note").fill("隔离验收：已核对图片中的积水候选。")
            candidate.get_by_role("button", name="确认此候选").click()
            expect(candidate).to_contain_text("已确认", timeout=30000)
            expect(card.locator(".vision-job-state")).to_have_text("候选已复核", timeout=30000)
            candidate.get_by_role("button", name="将此候选转入道路事件").click()
            assert page.locator("#event-type").input_value() == "flooding"
            assert page.locator("#event-action").input_value() == "notice"
            suggestion = card.locator(".suggested-edges .candidate-card").first
            expect(suggestion).to_be_visible(timeout=30000)
            suggestions = page.request.get(base + f"/api/manager/vision-jobs/{job_id}/suggested-edges").json()["data"]
            expected_edge = suggestions["suggested_edges"][0]
            with page.expect_response(lambda r: "/api/road-conditions/snap?" in r.url and r.status == 200,
                                      timeout=30000) as snapped:
                suggestion.get_by_role("button", name="在地图定位并预填发布落点").click()
            snap = snapped.value.json()["data"]["snap"]
            assert (snap["u"], snap["v"], snap["key"]) == tuple(expected_edge[k] for k in ("u", "v", "key"))
            page.locator("#event-action").select_option(impact)
            page.locator("#event-mode-walk").check()
            page.locator("#field-confirmation").fill("隔离验收：测试人员已核对候选和所选道路，仅供测试。")
            page.locator("#confirm-picked-road").click()
            expect(page.locator("#publish-event")).to_be_enabled(timeout=30000)
            with page.expect_response(lambda r: r.url.endswith("/api/road-conditions")
                                      and r.request.method == "POST", timeout=30000) as publication:
                page.locator("#publish-event").click()
            assert publication.value.status == 201, publication.value.text()
            body = publication.value.request.post_data_json
            assert body["action"] == impact and body["source_vision_candidate_index"] == flood_indices[0]
            assert body["edge_ref"] == {key: expected_edge[key] for key in ("u", "v", "key")}
            condition = publication.value.json()["data"]["condition"]
            assert condition["source"]["candidate_index"] == flood_indices[0]
            penalties, closed, _ = road_conditions.resolve_condition_effects(network.get_network(), [condition], "walk", strict=True)
            if impact == "notice":
                assert not penalties and not closed
            elif impact == "slowdown":
                assert penalties and not closed
            else:
                assert not penalties and closed
            row = page.locator("#event-list .event-row").filter(has_text="疑似积水")
            expect(row).to_have_count(1, timeout=30000)
            page.once("dialog", lambda dialog: dialog.accept())
            row.get_by_role("button", name="永久删除").click()
            expect(row).to_have_count(0, timeout=30000)
            assert not road_conditions.list_conditions(strict=True)
            for width in (360, 390, 430, 1365):
                page.set_viewport_size({"width": width, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth") <= width
                page.screenshot(path=str(tmp_path / f"flood-assisted-{width}.png"), full_page=True)
            assert not errors
            context.close()
            browser.close()
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
