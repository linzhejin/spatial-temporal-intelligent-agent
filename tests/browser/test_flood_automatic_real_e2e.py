"""Public image upload through real model and isolated review, without a drawn road."""
from __future__ import annotations

import hashlib
import json
import os
import threading
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


@pytest.mark.skipif(os.getenv("RUN_FLOOD_AUTOMATIC_REAL_E2E") != "1" or sync_playwright is None,
                    reason="requires isolated PostgreSQL and the real flood model")
def test_public_image_generates_reviewable_candidates_without_drawing(monkeypatch, tmp_path):
    import api.routes as routes
    import config
    from app import create_app
    from vision import worker
    from vision.specialized import DoodleverseFloodSegmenter

    root = Path(r"D:\LuoJiaValidation")
    image = (root / "model_comparison_20261109/flood/road_inundation_delta_ian_2024/"
             "samples_v1_2/test/HURRICANE-IAN_000491_image.png")
    model_path = root / "doodleverse_floodnet_v7_fp32.onnx"
    runtime = Path(os.environ["VISION_FLOOD_TEST_RUNTIME"])
    traffic_path = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    assert all(path.is_file() for path in (image, model_path, runtime, traffic_path))
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    values = {
        "ROAD_CONDITION_ADMIN_PASSWORD": "isolated-auto-flood-test",
        "ROAD_CONDITION_ADMIN_TOKEN": "", "VISION_UPLOAD_DIR": str(upload_dir),
        "VISION_MODEL_PATH": str(traffic_path), "VISION_TRIAL_ENABLED": True,
        "VISION_FLOOD_ENABLED": True, "VISION_FLOOD_RELEASE_MODE": "assisted_review",
        "VISION_FLOOD_MODEL_PATH": str(model_path),
        "VISION_FLOOD_REVISION": "doodleverse-floodnet-resunet-v7",
    }
    for key, value in values.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(routes, "_vision_flood_assistance_ready", lambda caps: bool(
        caps.get("flood_assistance_supported") and caps.get("flood_assistance_model_sha256") == model_hash
    ))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    capability = {
        "flood_assistance_supported": True, "flood_assistance_automatic_supported": True,
        "flood_assistance_release_status": "test_injected",
        "flood_assistance_model_id": "doodleverse-floodnet-resunet-v7",
        "flood_assistance_model_version": config.VISION_FLOOD_REVISION,
        "flood_assistance_model_sha256": model_hash, "trial_flood_supported": True,
        "trial_flood_media": "image", "trial_flood_automatic_supported": True,
    }
    vision_repository.heartbeat_worker(DATABASE_URL, "auto-browser-operational",
                                       ready=True, role="operational")
    vision_repository.heartbeat_worker(DATABASE_URL, "auto-browser-trial", ready=True, role="trial",
                                       status_detail=json.dumps(capability))
    model = DoodleverseFloodSegmenter(str(model_path), model_version=config.VISION_FLOOD_REVISION,
                                      runtime_python=str(runtime))
    model.model_sha256 = model_hash
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="isolated-auto-flood")
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
            page.locator("#password").fill("isolated-auto-flood-test")
            page.locator("#login-form button[type=submit]").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            page.get_by_text("新建影像分析", exact=True).click()
            expect(page.locator("#vision-trial-option")).to_be_enabled(timeout=30000)
            page.locator("#vision-purpose").select_option("trial")
            page.locator("#media-file").set_input_files(str(image))
            expect(page.locator("#vision-purpose-note")).to_contain_text("自动")
            with page.expect_response(lambda response: response.url.endswith("/api/manager/vision-jobs")
                                      and response.request.method == "POST", timeout=30000) as submitted:
                page.locator("#submit-vision").click()
            assert submitted.value.status == 202, submitted.value.text()
            job_id = submitted.value.json()["data"]["job"]["job_id"]
            stored = vision_repository.get_job(DATABASE_URL, job_id)
            assert stored["observation_regions"] == []
            assert stored["purpose"] == "trial"
            outcome = worker.process_next_job(DATABASE_URL, "auto-browser-real-worker",
                                              trial_flood_segmenter=model, trial_only=True,
                                              assisted_flood_supported=True, automatic_flood_supported=True)
            assert outcome == {"job_id": job_id, "status": "needs_review"}
            stored = vision_repository.get_job(DATABASE_URL, job_id)
            assert stored["result"]["trial_coverage"]["flood"] == "checked"
            candidate = stored["result"]["candidates"][0]
            assert candidate["kind"] == "possible_flooding"
            assert candidate["evidence"]["summary"]["region_origin"] == "automatic_model"
            page.reload(wait_until="networkidle")
            card = page.locator(".vision-job").filter(has_text=image.name)
            expect(card).to_be_visible(timeout=30000)
            outlines = card.locator(".vision-image-stage svg[aria-label*='积水'] > g")
            expect(outlines.first).to_be_visible()
            assert outlines.count() > 5
            expect(outlines.nth(1)).to_be_hidden()
            first = card.locator(".candidate-list:not(.suggested-edges) .candidate-card").first
            expect(first).to_contain_text("候选 1")
            more = card.locator("details.candidate-more")
            expect(more.locator("summary")).to_contain_text("查看其余")
            more.locator("summary").click()
            sixth = more.locator(".candidate-card").first
            expect(sixth).to_contain_text("候选 6")
            sixth.get_by_role("button", name="查看位置").click()
            expect(outlines.first).to_be_hidden()
            expect(outlines.nth(5)).to_be_visible()
            first.get_by_role("button", name="查看位置").click()
            expect(outlines.first).to_be_visible()
            card.locator("textarea.vision-review-note").fill("隔离测试：已查看图片中的疑似积水轮廓。")
            first.get_by_role("button", name="确认此候选").click()
            expect(first).to_contain_text("已确认", timeout=30000)
            assert card.get_by_role("button", name="将此候选转入道路事件").count() == 0
            page.screenshot(path=str(tmp_path / "public-auto-flood-390.png"), full_page=True)
            assert not errors
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
