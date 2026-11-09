"""Opt-in real-model trial: browser, upload, PostgreSQL and actual worker."""
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
RUN_E2E = os.getenv("RUN_VISION_TRIAL_REAL_E2E") == "1" and sync_playwright is not None


@pytest.mark.skipif(not RUN_E2E, reason="requires explicit isolated DB, real models and browser")
@pytest.mark.parametrize("flood_variant", ["baseline", "segformer_b0"])
def test_public_image_real_inference_review_and_missed_case(monkeypatch, tmp_path, flood_variant):
    import config
    from app import create_app
    from vision import worker
    from vision.engine import OnnxDetector

    base = Path(r"D:\LuoJiaExperimentData\WHUWalker_vision_research")
    image = Path(r"D:\LuoJiaValidation\public_vision_20261109\fwisd\images\00324.png")
    detector_path = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    flood_path = base / "Floodwater/visible_water_run_20261011/visible-water-mobilenetv3-unet.onnx"
    if flood_variant == "segformer_b0":
        flood_path = Path(r"D:\LuoJiaValidation\model_trials_20261109\segformer_b0_fwisd_head\segformer-b0-fwisd.onnx")
        image = image.with_name("00270.png")
    accident_path = base / "AIDER_artifacts/mobilenetv3_run_20261010_seed20261010/aider-mobilenetv3-small.onnx"
    assert all(path.is_file() for path in (image, detector_path, flood_path, accident_path))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "trial-browser-secret")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "VISION_MODEL_PATH", str(detector_path))
    monkeypatch.setattr(config, "VISION_TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_MODEL_PATH", str(accident_path))
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_MODEL_PATH", str(flood_path))
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_PROBABILITY", 0.5 if flood_variant == "segformer_b0" else 0.65429688)
    if flood_variant == "segformer_b0":
        monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_MODEL_ID", "segformer-b0-fwisd-head-trial")
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    models, capabilities = worker.load_trial_models()
    assert capabilities == {"trial_accident_supported": True, "trial_flood_supported": True}
    detector = OnnxDetector(str(detector_path))
    vision_repository.heartbeat_worker(DATABASE_URL, "trial-browser-worker", ready=True,
                                       status_detail=json.dumps(capabilities))
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="trial-browser-secret")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
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
            page.goto(base_url + "/manager", wait_until="networkidle")
            page.locator("#password").fill("trial-browser-secret")
            page.locator("#login-form button[type=submit]").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            page.get_by_text("新建影像分析", exact=True).click()
            expect(page.locator("#vision-trial-option")).to_be_enabled(timeout=30000)
            page.locator("#vision-purpose").select_option("trial")
            assert not page.locator("#captured-at").evaluate("element => element.required")
            page.locator("#media-file").set_input_files(str(image))
            page.wait_for_function("document.querySelector('#observation-image').naturalWidth > 0")
            page.locator("#observation-kind").select_option("road_surface")
            page.locator("#start-region").click()
            canvas = page.locator("#observation-canvas")
            bounds = canvas.bounding_box()
            assert bounds and bounds["width"] > 0
            # Select a labeled road area to exercise review; this is not an accuracy evaluation.
            points = (((.01, .543), (.99, .543), (.99, .945), (.01, .945))
                      if flood_variant == "segformer_b0" else
                      ((.77, .005), (.868, .2766), (.997, .6207),
                       (.997, .3245), (.98, .2678), (.87, .005)))
            for x, y in points:
                canvas.click(position={"x": bounds["width"] * x, "y": bounds["height"] * y})
            page.locator("#finish-region").click()
            with page.expect_response(lambda r: r.url.endswith("/api/manager/vision-jobs")
                                      and r.request.method == "POST", timeout=30000) as submitted:
                page.locator("#submit-vision").click()
            assert submitted.value.status == 202, submitted.value.text()
            job_id = submitted.value.json()["data"]["job"]["job_id"]
            result = worker.process_next_job(DATABASE_URL, "trial-browser-actual", detector=detector,
                trial_accident_model=models["accident"], trial_flood_segmenter=models["flood"])
            assert result["job_id"] == job_id and result["status"] == "needs_review"
            if flood_variant == "segformer_b0":
                stored = vision_repository.get_job(DATABASE_URL, job_id)
                assert stored["result"]["model"]["flood_model"]["id"] == "segformer-b0-fwisd-head-trial"
            page.reload(wait_until="networkidle")
            card = page.locator(".vision-job").filter(has_text=image.name)
            expect(card).to_be_visible(timeout=30000)
            expect(card).to_contain_text("试用结果可能误报或漏报")
            expect(card).to_contain_text("积水：已检查")
            candidate = card.locator(".candidate-card").filter(has_text="积水").first
            expect(candidate).to_be_visible()
            card.locator("textarea.vision-review-note").fill("对照公开影像标注后确认模型候选")
            candidate.get_by_role("button", name="确认此候选").click()
            expect(candidate).to_contain_text("已确认", timeout=30000)
            assert card.get_by_role("button", name="将此候选转入道路事件").count() == 0
            card.get_by_label("漏检类型").select_option("possible_accident")
            card.get_by_label("漏检说明").fill("此处用于检查漏检记录是否可以保存")
            card.get_by_role("button", name="保存漏检").click()
            expect(card).to_contain_text("此处用于检查漏检记录是否可以保存", timeout=30000)
            page.reload(wait_until="networkidle")
            card = page.locator(".vision-job").filter(has_text=image.name)
            expect(card).to_contain_text("此处用于检查漏检记录是否可以保存")
            for width in (360, 390, 430):
                page.set_viewport_size({"width": width, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth") <= width
                page.screenshot(path=str(tmp_path / f"vision-trial-{width}.png"), full_page=True)
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(not RUN_E2E, reason="requires explicit isolated DB, real models and browser")
def test_public_video_resume_cancel_retry_and_real_inference(monkeypatch, tmp_path):
    import config
    from app import create_app
    from vision import worker
    from vision.engine import OnnxDetector

    video = Path(r"D:\LuoJiaValidation\public_vision_20261109\accident\2wUrP2wgXhs_00.mp4")
    detector_path = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    accident_path = Path(r"D:\LuoJiaExperimentData\WHUWalker_vision_research\AIDER_artifacts\mobilenetv3_run_20261010_seed20261010\aider-mobilenetv3-small.onnx")
    assert all(path.is_file() for path in (video, detector_path, accident_path))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "trial-browser-secret")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "VISION_MODEL_PATH", str(detector_path))
    monkeypatch.setattr(config, "VISION_CHUNK_BYTES", 512 * 1024)
    monkeypatch.setattr(config, "VISION_TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_MODEL_PATH", str(accident_path))
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_MODEL_PATH", "")
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    models, capabilities = worker.load_trial_models()
    assert capabilities == {"trial_accident_supported": True, "trial_flood_supported": False}
    detector = OnnxDetector(str(detector_path))
    vision_repository.heartbeat_worker(DATABASE_URL, "trial-video-status", ready=True,
                                       status_detail=json.dumps(capabilities))
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="trial-browser-secret")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
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
            page.goto(base_url + "/manager", wait_until="networkidle")
            page.locator("#password").fill("trial-browser-secret")
            page.locator("#login-form button[type=submit]").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            page.get_by_text("新建影像分析", exact=True).click()
            expect(page.locator("#vision-trial-option")).to_be_enabled(timeout=30000)
            page.locator("#vision-purpose").select_option("trial")
            page.locator("#media-file").set_input_files(str(video))
            page.wait_for_function("document.querySelector('#observation-video').videoWidth > 0", timeout=30000)
            page.locator("#observation-kind").select_option("vehicle_lane")
            page.locator("#start-region").click()
            canvas = page.locator("#observation-canvas")
            bounds = canvas.bounding_box()
            assert bounds and bounds["width"] > 0
            for x, y in ((.02, .02), (.98, .02), (.98, .98), (.02, .98)):
                canvas.click(position={"x": bounds["width"] * x, "y": bounds["height"] * y})
            page.locator("#finish-region").click()
            with page.expect_response(lambda r: r.url.endswith("/complete")
                                      and r.request.method == "POST", timeout=60000) as submitted:
                page.locator("#submit-vision").click()
            assert submitted.value.status == 202, submitted.value.text()
            job_id = submitted.value.json()["data"]["job"]["job_id"]
            card = page.locator(".vision-job").filter(has_text=video.name)
            expect(card.locator(".vision-job-state")).to_have_text("排队中", timeout=30000)
            page.once("dialog", lambda dialog: dialog.accept())
            card.get_by_role("button", name="取消分析").click()
            expect(card.locator(".vision-job-state")).to_have_text("已取消", timeout=30000)
            page.once("dialog", lambda dialog: dialog.accept())
            card.get_by_role("button", name="从断点重试").click()
            expect(card.locator(".vision-job-state")).to_have_text("排队中", timeout=30000)
            outcomes = []
            work_thread = threading.Thread(target=lambda: outcomes.append(worker.process_next_job(
                DATABASE_URL, "trial-video-actual", detector=detector,
                trial_accident_model=models["accident"])), daemon=True)
            work_thread.start()
            expect(card.locator(".vision-job-state")).to_have_text(
                __import__("re").compile("完成|待复核"), timeout=300000)
            work_thread.join(timeout=5)
            assert outcomes and outcomes[0]["job_id"] == job_id
            stored = vision_repository.get_job(DATABASE_URL, job_id)
            assert stored["purpose"] == "trial" and stored["captured_at"] is None
            assert stored["result"]["trial_coverage"] == {"accident": "checked", "flood": "not_configured"}
            assert "research_replay" in stored["result"]
            assert card.get_by_role("button", name="转入道路事件").count() == 0
            card.get_by_label("漏检类型").select_option("possible_accident")
            card.get_by_label("漏检说明").fill("公开标注指出这段视频中出现交通事故")
            card.get_by_label("漏检时间（秒，可选）").fill("7.8")
            card.get_by_role("button", name="保存漏检").click()
            expect(card).to_contain_text("公开标注指出这段视频中出现交通事故", timeout=30000)
            page.reload(wait_until="networkidle")
            expect(page.locator(".vision-job").filter(has_text=video.name)).to_contain_text("7.8 秒")
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


@pytest.mark.skipif(not RUN_E2E, reason="requires explicit isolated DB, real models and browser")
def test_temporal_video_trial_uses_whole_window_and_cannot_publish(monkeypatch, tmp_path):
    import config
    from app import create_app
    from vision import worker
    from vision.engine import OnnxDetector

    video = Path(r"D:\LuoJiaValidation\public_vision_20261109\accident\7QawsvO1F1M_00.mp4")
    detector_path = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    temporal_path = Path(r"D:\LuoJiaValidation\model_trials_20261109\temporal-crash-trial.onnx")
    assert all(path.is_file() for path in (video, detector_path, temporal_path))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "trial-browser-secret")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "VISION_MODEL_PATH", str(detector_path))
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 1)
    monkeypatch.setattr(config, "VISION_TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_KIND", "temporal")
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_THRESHOLD", .5)
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_MODEL_PATH", str(temporal_path))
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_MODEL_PATH", "")
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    models, capabilities = worker.load_trial_models()
    assert capabilities["trial_accident_media"] == "video"
    detector = OnnxDetector(str(detector_path))
    vision_repository.heartbeat_worker(DATABASE_URL, "trial-temporal-status", ready=True,
                                       status_detail=json.dumps(capabilities))
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="trial-browser-secret")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
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
            page.goto(f"http://127.0.0.1:{server.server_port}/manager", wait_until="networkidle")
            page.locator("#password").fill("trial-browser-secret")
            page.locator("#login-form button[type=submit]").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            page.get_by_text("新建影像分析", exact=True).click()
            expect(page.locator("#vision-trial-option")).to_be_enabled(timeout=30000)
            page.locator("#vision-purpose").select_option("trial")
            expect(page.locator("#vision-purpose-note")).to_contain_text("事故按视频抽帧筛查")
            page.locator("#media-file").set_input_files(str(video))
            page.wait_for_function("document.querySelector('#observation-video').videoWidth > 0", timeout=30000)
            with page.expect_response(lambda response: response.url.endswith("/complete")
                                      and response.request.method == "POST", timeout=60000) as submitted:
                page.locator("#submit-vision").click()
            assert submitted.value.status == 202, submitted.value.text()
            job_id = submitted.value.json()["data"]["job"]["job_id"]
            outcome = worker.process_next_job(DATABASE_URL, "trial-temporal-actual", detector=detector,
                                              trial_accident_model=models["accident"])
            assert outcome["status"] == "needs_review"
            stored = vision_repository.get_job(DATABASE_URL, job_id)
            result = stored["result"]
            assert result["trial_coverage"] == {"accident": "checked", "flood": "not_configured"}
            candidate = next(c for c in result["candidates"] if c["kind"] == "possible_accident")
            assert candidate["evidence"]["summary"]["model_scope"] == "temporal_window"
            assert candidate["evidence"]["clips"] == [] and candidate["evidence"]["clip_status"] == "window_only"
            page.reload(wait_until="networkidle")
            card = page.locator(".vision-job").filter(has_text=video.name)
            expect(card).to_contain_text("事故：已检查", timeout=30000)
            expect(card).to_contain_text("每段抽取 16 帧筛查，不能定位事故时刻或位置")
            expect(card).to_contain_text("没有统计人车数量")
            assert card.get_by_role("button", name="转入道路事件").count() == 0
            card.screenshot(path=str(tmp_path / "temporal-trial-card.png"))
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


@pytest.mark.skipif(not RUN_E2E, reason="requires explicit isolated DB, real models and browser")
def test_trial_video_can_run_flood_and_temporal_models_together(monkeypatch, tmp_path):
    import config
    from app import create_app
    from vision import worker

    video = Path(r"D:\LuoJiaValidation\public_vision_20261109\accident\2wUrP2wgXhs_00.mp4")
    detector_path = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    temporal_path = Path(r"D:\LuoJiaValidation\model_trials_20261109\temporal-crash-trial.onnx")
    flood_path = Path(r"D:\LuoJiaValidation\model_trials_20261109\segformer_b0_fwisd_head\segformer-b0-fwisd.onnx")
    assert all(path.is_file() for path in (video, detector_path, temporal_path, flood_path))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "trial-browser-secret")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(config, "VISION_MODEL_PATH", str(detector_path))
    monkeypatch.setattr(config, "VISION_TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_KIND", "temporal")
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_THRESHOLD", .5)
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_MODEL_PATH", str(temporal_path))
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_MODEL_PATH", str(flood_path))
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_MODEL_ID", "segformer-b0-trial")
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_PROBABILITY", .5)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    models, capabilities = worker.load_trial_models()
    vision_repository.heartbeat_worker(DATABASE_URL, "trial-video-flood-status", ready=True,
                                       status_detail=json.dumps(capabilities))
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="trial-browser-secret")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
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
            page.goto(f"http://127.0.0.1:{server.server_port}/manager", wait_until="networkidle")
            page.locator("#password").fill("trial-browser-secret")
            page.locator("#login-form button[type=submit]").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)
            page.get_by_text("新建影像分析", exact=True).click()
            expect(page.locator("#vision-trial-option")).to_be_enabled(timeout=30000)
            page.locator("#vision-purpose").select_option("trial")
            page.locator("#media-file").set_input_files(str(video))
            page.wait_for_function("document.querySelector('#observation-video').videoWidth > 0", timeout=30000)
            page.locator("#camera-stabilized").check()
            page.locator("#observation-kind").select_option("road_surface")
            page.locator("#start-region").click()
            canvas = page.locator("#observation-canvas")
            bounds = canvas.bounding_box()
            for x, y in ((.02, .02), (.98, .02), (.98, .98), (.02, .98)):
                canvas.click(position={"x": bounds["width"] * x, "y": bounds["height"] * y})
            page.locator("#finish-region").click()
            with page.expect_response(lambda response: response.url.endswith("/complete")
                                      and response.request.method == "POST", timeout=60000) as submitted:
                page.locator("#submit-vision").click()
            assert submitted.value.status == 202, submitted.value.text()
            job_id = submitted.value.json()["data"]["job"]["job_id"]
            outcome = worker.process_next_job(DATABASE_URL, "trial-video-flood-actual",
                trial_accident_model=models["accident"], trial_flood_segmenter=models["flood"], trial_only=True)
            assert outcome["status"] in {"completed", "needs_review"}, outcome
            stored = vision_repository.get_job(DATABASE_URL, job_id)
            result = stored["result"]
            assert result["trial_coverage"] == {"accident": "checked", "flood": "checked"}
            assert result["video_coverage"]["sample_interval_seconds"] == 1.0
            assert result["trial_specialized_only"] is True
            page.reload(wait_until="networkidle")
            card = page.locator(".vision-job").filter(has_text=video.name)
            expect(card).to_contain_text("事故：已检查 · 积水：已检查", timeout=30000)
            expect(card).to_contain_text("没有统计人车数量")
            assert card.get_by_role("button", name="转入道路事件").count() == 0
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
