"""Real manager DOM + Leaflet regression checks; all API writes are intercepted.

Run unittest discovery in tests/browser. An isolated local static server is started.
Set RUN_MANAGER_BROWSER_TESTS=1 to enable. Requires development-only Playwright
and Chromium; normal backend test runs skip this optional browser suite.
"""

import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None


@unittest.skipUnless(os.getenv("RUN_MANAGER_BROWSER_TESTS") == "1" and sync_playwright,
                     "Enable RUN_MANAGER_BROWSER_TESTS=1 and install Playwright/Chromium for browser QA")
class ManagerEventControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *_args):
                pass

        static_dir = Path(__file__).resolve().parents[2] / "static"
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(static_dir)))
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/manager.html"
        cls.app_url = f"http://127.0.0.1:{cls.server.server_port}/index.html"
        cls.runtime = sync_playwright().start()
        cls.browser = cls.runtime.chromium.launch()
        cls.qa_dir = Path(tempfile.mkdtemp(prefix="manager-event-qa-"))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.runtime.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()
        print(f"Manager screenshots: {cls.qa_dir}")

    def setUp(self):
        mobile = self._testMethodName == "test_mobile_controls_fit_and_custom_time_can_be_typed"
        self.context = self.browser.new_context(
            timezone_id="America/New_York", is_mobile=mobile, has_touch=mobile,
            viewport={"width": 390, "height": 844} if mobile else {"width": 1280, "height": 1000},
        )
        self.page = self.context.new_page()
        self.errors = []
        self.posts = []
        self.snap_requests = []
        self.road_conditions = []
        self.pending_posts = []
        self.hold_posts = False
        self.fail_posts = False
        self.vision_jobs = []
        self.vision_preview_data = None
        self.vision_preview_requests = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.route("**/api/**", self.api)
        self.page.route("**/*", self.local_only)

    def local_only(self, route):
        if urlsplit(route.request.url).hostname in {"127.0.0.1", "localhost"}:
            route.fallback()
        else:
            route.abort()

    def api(self, route):
        path = urlsplit(route.request.url).path
        data = {}
        if path == "/api/admin/status":
            data = {"is_admin": True, "csrf_token": "test-csrf"}
        elif path == "/api/road-conditions" and route.request.method == "POST":
            self.posts.append(route.request.post_data_json)
            if self.hold_posts:
                self.pending_posts.append(route)
                return
            if self.fail_posts:
                route.fulfill(status=503, json={"message": "发布失败，请重试"})
                return
        elif path == "/api/road-conditions":
            data = {"conditions": self.road_conditions}
        elif path == "/api/road-conditions/snap":
            self.snap_requests.append(route.request.url)
            data = {"snap": {
                "u": 1, "v": 2, "key": 0, "road_name": "测试路",
                "dist_m": 2, "chain_length_m": 100,
                "snap_lng_gcj": 114.363, "snap_lat_gcj": 30.5365,
                "geometry_gcj": [[114.363, 30.5365], [114.364, 30.5365]],
            }, "impact_preview": {
                "road_name": "测试路", "affected_road_segments": 1,
                "affected_length_m": 100, "active_event_count": 0,
                "effects_by_mode": {}, "sample_routes": {},
            }}
        elif path == "/api/manager/vision-jobs":
            data = {"jobs": self.vision_jobs}
        elif path.endswith("/preview-frames"):
            self.vision_preview_requests.append(path)
            data = self.vision_preview_data or {
                "media": {}, "preview_frame_interval_seconds": 1, "preview_frames": [],
            }
        elif path == "/api/manager/vision-scenes":
            data = {"scenes": []}
        elif path == "/api/manager/vision-status":
            data = {"inference_ready": False, "notice": "测试环境未启用影像模型"}
        route.fulfill(json={"data": data})

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [], "manager must not raise browser errors")

    def open_manager(self):
        self.page.goto(self.url, wait_until="networkidle")
        expect(self.page.locator("#workspace")).to_be_visible()

    def test_legacy_road_report_requires_selected_effect_before_publishing(self):
        self.page.add_init_script("try { localStorage.setItem('whu_welcome_seen', 'true'); } catch (_) {}")
        self.page.goto(self.app_url, wait_until="domcontentloaded")
        expect(self.page.locator(".leaflet-container")).to_be_visible(timeout=10000)

        self.page.locator("#road-report-btn").click()
        expect(self.page.locator("#road-report-section")).to_be_visible()
        self.page.locator("#rr-name").fill("浏览器验收封路")
        self.page.locator("#rr-action").select_option("closure")
        self.page.locator("#rr-pick-btn").click()
        self.page.locator("#map-container").click(position={"x": 260, "y": 260})
        expect(self.page.locator("#rr-location")).to_have_value(
            "测试路（吸附偏移 2 米）", timeout=5000,
        )

        self.page.locator("#rr-submit-btn").click()
        expect(self.page.locator("#road-report-hint")).to_contain_text("至少选择一种受影响的出行方式")
        self.assertEqual(self.posts, [], "an event without an affected travel mode must not publish")

        self.page.locator("#rr-mode-walk").check()
        self.page.locator("#rr-submit-btn").click()
        expect(self.page.locator("#road-report-hint")).to_contain_text("已发布：封路并绕行")
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]["action"], "closure")
        self.assertEqual(self.posts[0]["affected_modes"], ["walk"])
        self.assertFalse(self.errors, "the public route-report flow must not raise browser errors")

    def test_route_condition_map_markers_follow_current_mode_effect(self):
        self.page.add_init_script("try { localStorage.setItem('whu_welcome_seen', 'true'); } catch (_) {}")
        self.road_conditions = [
            {"id": "notice", "type": "closure", "action": "notice",
             "affected_modes": ["walk"], "name": "仅提示道路",
             "edge": {"snap": {"lng": 114.361, "lat": 30.531},
                      "geometry_gcj": [[114.361, 30.531], [114.362, 30.531]]}},
            {"id": "closure", "type": "event", "action": "closure",
             "affected_modes": ["walk"], "name": "步行封闭道路",
             "edge": {"snap": {"lng": 114.363, "lat": 30.532},
                      "geometry_gcj": [[114.363, 30.532], [114.364, 30.532]]}},
            {"id": "slowdown", "type": "event", "action": "slowdown",
             "affected_modes": ["walk"], "cost_multiplier": 1.4, "name": "步行缓行道路",
             "edge": {"snap": {"lng": 114.364, "lat": 30.5325},
                      "geometry_gcj": [[114.364, 30.5325], [114.365, 30.5325]]}},
            {"id": "unaffected", "type": "flooding", "action": "closure",
             "affected_modes": ["drive"], "name": "驾车封闭道路",
             "edge": {"snap": {"lng": 114.365, "lat": 30.533},
                      "geometry_gcj": [[114.365, 30.533], [114.366, 30.533]]}},
            {"id": "unknown", "type": "event", "action": "closure",
             "name": "影响未配置道路",
             "edge": {"snap": {"lng": 114.367, "lat": 30.534},
                      "geometry_gcj": [[114.367, 30.534], [114.368, 30.534]]}},
        ]
        self.page.goto(self.app_url, wait_until="networkidle")
        expect(self.page.locator(".leaflet-container")).to_be_visible(timeout=10000)
        expect(self.page.locator("#legend-closure-item")).to_be_visible()
        expect(self.page.locator("#legend-slow-item")).to_be_visible()
        expect(self.page.locator("#legend-info-item")).to_be_visible()
        expect(self.page.locator("#legend-unknown-item")).to_be_visible()

        pills = self.page.locator(".whu-map-pill")
        expect(pills).to_have_count(5)
        rendered = {
            text: self.page.locator(".whu-map-pill").filter(has_text=text).first
            for text in ("仅提示道路", "步行封闭道路", "步行缓行道路", "驾车封闭道路", "影响未配置道路")
        }
        expect(rendered["仅提示道路"]).to_contain_text("ℹ️")
        expect(rendered["仅提示道路"]).to_have_css("background-color", "rgb(40, 120, 168)")
        expect(rendered["步行封闭道路"]).to_contain_text("🚫")
        expect(rendered["步行封闭道路"]).to_have_css("background-color", "rgb(192, 57, 43)")
        expect(rendered["步行缓行道路"]).to_contain_text("⚠️")
        expect(rendered["步行缓行道路"]).to_have_css("background-color", "rgb(217, 119, 6)")
        expect(rendered["驾车封闭道路"]).to_contain_text("ℹ️")
        expect(rendered["驾车封闭道路"]).to_have_css("background-color", "rgb(40, 120, 168)")
        expect(rendered["影响未配置道路"]).to_contain_text("⚠️")
        expect(rendered["影响未配置道路"]).to_have_css("background-color", "rgb(107, 114, 128)")
        self.assertIn("rgb(40, 120, 168)", self.page.locator("#legend-info-item .legend-line").evaluate(
            "element => getComputedStyle(element).backgroundImage"))
        self.assertIn("rgb(217, 119, 6)", self.page.locator("#legend-slow-item .legend-line").evaluate(
            "element => getComputedStyle(element).backgroundImage"))

    def test_video_detection_overlays_follow_source_and_evidence_time(self):
        job_id = "12345678-1234-4234-8234-123456789abc"
        self.vision_preview_data = {
            "media": {"width": 1000, "height": 800},
            "preview_frame_interval_seconds": 1,
            "preview_frames": [
                {"time_seconds": 0, "detections": [{
                    "box": [0.1, 0.2, 0.4, 0.5], "label": "car",
                    "confidence": 0.91, "track_id": "7",
                }]},
                {"time_seconds": 11, "detections": [{
                    "box": [0.2, 0.25, 0.5, 0.55], "label": "pedestrian",
                    "confidence": 0.82, "track_id": "9",
                }]},
            ],
        }
        self.vision_jobs = [{
            "job_id": job_id, "status": "needs_review", "media_kind": "video",
            "original_name": "巡查视频.mp4", "media_url": f"/api/manager/vision-jobs/{job_id}/media",
            "result": {
                "media": {"width": 1000, "height": 800},
                "candidates": [{
                    "kind": "possible_congestion", "confidence": 0.74,
                    "evidence": {
                        "segments": [{"start_seconds": 10, "end_seconds": 12}],
                        "clips": [{
                            "media_url": f"/api/manager/vision-jobs/{job_id}/media?clip=0:0",
                            "start_seconds": 10, "end_seconds": 20,
                        }],
                    },
                }],
            },
        }]
        self.open_manager()

        main_overlay = self.page.locator(".vision-job .video-detection-overlay").first
        self.assertEqual(self.vision_preview_requests, [], "the 5-second task-list poll must not fetch frame overlays")
        self.page.locator(".vision-job video.vision-preview").evaluate("video => video.dispatchEvent(new Event('seeking'))")
        expect(main_overlay.locator("text")).to_contain_text("汽车 91% #7")
        self.assertEqual(len(self.vision_preview_requests), 1)
        evidence_video = self.page.locator("video.candidate-evidence-clip")
        expect(evidence_video).to_have_count(1)
        evidence_video.evaluate("video => { Object.defineProperty(video, 'currentTime', {configurable: true, value: 1}); video.dispatchEvent(new Event('timeupdate')); }")
        evidence_overlay = self.page.locator(".candidate-evidence .video-detection-overlay")
        expect(evidence_overlay.locator("text")).to_contain_text("行人 82% #9")
        expect(self.page.locator(".video-overlay-note")).to_contain_text("不是地面坐标")
        self.page.locator(".candidate-evidence").evaluate("details => { details.open = true; }")
        self.page.screenshot(path=str(self.qa_dir / "video-detection-overlays.png"), full_page=True)

    def prepare_event(self):
        self.page.locator("#event-type").select_option("construction")
        self.page.locator("#event-name").fill("测试路临时施工")
        self.page.locator("#event-action").select_option("closure")
        self.page.get_by_role("checkbox", name="步行", exact=True).check()
        self.page.locator("#pick-road").click()
        self.page.locator("#manager-map").click(position={"x": 180, "y": 180})
        self.page.locator("#confirm-picked-road").click()

    def test_empty_state_and_independent_touch_friendly_modes(self):
        self.open_manager()
        expect(self.page.locator("#publish-event")).to_be_disabled()
        expect(self.page.locator("#publish-guidance")).to_contain_text("请选择事件类型")
        self.assertEqual(self.page.locator("select[multiple]").count(), 0)
        self.page.get_by_role("checkbox", name="步行", exact=True).check()
        self.page.get_by_role("checkbox", name="骑行", exact=True).check()
        self.page.get_by_role("checkbox", name="步行", exact=True).uncheck()
        expect(self.page.get_by_role("checkbox", name="骑行", exact=True)).to_be_checked()
        expect(self.page.get_by_role("checkbox", name="驾车", exact=True)).not_to_be_checked()

    def test_immediate_30_minutes_publishes_and_resets(self):
        self.open_manager()
        self.prepare_event()
        expect(self.page.locator("#event-time-summary")).to_contain_text("30 分钟")
        expect(self.page.locator("#publish-event")).to_be_enabled()
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#form-message")).to_contain_text("事件已发布")
        body = self.posts[0]
        self.assertEqual(body["affected_modes"], ["walk"])
        start = datetime.fromisoformat(body["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(body["end_time"].replace("Z", "+00:00"))
        self.assertAlmostEqual((end - start).total_seconds(), 1800, delta=1)
        self.assertLess(abs((datetime.now(timezone.utc) - start).total_seconds()), 10)
        expect(self.page.locator("#event-name")).to_have_value("")
        expect(self.page.get_by_role("checkbox", name="步行", exact=True)).not_to_be_checked()
        expect(self.page.locator("#publish-event")).to_be_disabled()

    def test_scheduled_custom_time_uses_beijing_and_rejects_reverse_range(self):
        self.open_manager()
        self.prepare_event()
        self.page.locator("#event-start-mode").select_option("scheduled")
        self.page.locator("#event-duration").select_option("custom")
        self.page.locator("#event-start-date").fill("2099-01-02")
        self.page.locator("#event-start-clock").fill("09:15")
        self.page.locator("#event-end-date").fill("2099-01-02")
        self.page.locator("#event-end-clock").fill("08:15")
        expect(self.page.locator("#publish-event")).to_be_disabled()
        expect(self.page.locator("#event-time-summary")).to_contain_text("结束时间必须晚于开始时间")
        self.page.locator("#event-end-clock").fill("10:15")
        expect(self.page.locator("#publish-event")).to_be_enabled()
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#form-message")).to_contain_text("事件已发布")
        self.assertEqual(self.posts[0]["start_time"], "2099-01-02T01:15:00.000Z")
        self.assertEqual(self.posts[0]["end_time"], "2099-01-02T02:15:00.000Z")

    def test_error_retains_draft_and_busy_state_prevents_duplicate_posts(self):
        self.open_manager()
        self.prepare_event()
        self.fail_posts = True
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#form-message")).to_contain_text("发布失败")
        expect(self.page.locator("#event-name")).to_have_value("测试路临时施工")
        expect(self.page.locator("#publish-event")).to_be_enabled()
        self.fail_posts = False
        self.hold_posts = True
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#publish-event")).to_have_text("正在发布…")
        self.page.locator("#event-name").dispatch_event("input")
        expect(self.page.locator("#publish-event")).to_be_disabled()
        self.page.locator("#condition-form").evaluate("form => form.requestSubmit()")
        self.assertEqual(len(self.posts), 2)
        self.pending_posts.pop().fulfill(json={"data": {"id": "test-event"}})
        expect(self.page.locator("#form-message")).to_contain_text("事件已发布")

    def test_end_today_and_explicit_manual_ongoing(self):
        self.open_manager()
        self.prepare_event()
        self.page.locator("#event-duration").select_option("day")
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#form-message")).to_contain_text("事件已发布")
        self.assertTrue(self.posts[0]["end_time"].endswith("T16:00:00.000Z"))
        self.prepare_event()
        self.page.locator("#event-duration").select_option("ongoing")
        expect(self.page.locator("#event-time-summary")).to_contain_text("手动结束")
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#form-message")).to_contain_text("事件已发布")
        self.assertNotIn("end_time", self.posts[1])

    def test_dji_aircraft_track_is_local_and_only_shown_after_click(self):
        self.open_manager()
        self.page.locator("#media-file").set_input_files({
            "name": "DJI_flight.mp4", "mimeType": "video/mp4", "buffer": b"test video",
        })
        self.page.locator("#dji-subtitle-file").set_input_files({
            "name": "DJI_flight.SRT", "mimeType": "text/plain",
            "buffer": (
                b"1\n00:00:00,000 --> 00:00:00,200\n"
                b"[latitude: 30.536000] [longitude: 114.363000]\n\n"
                b"2\n00:00:01,000 --> 00:00:01,200\n"
                b"[latitude: 30.537000] [longitude: 114.364000]"
            ),
        })
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("读到 2 个机身 GPS 位置点")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("字幕含视频时间码")
        expect(self.page.locator("#show-dji-track")).to_be_visible()
        self.assertEqual(self.snap_requests, [], "reading the local SRT must not request a road snap")

        self.page.evaluate("""() => {
          const video = document.getElementById('observation-video');
          let current = 0;
          Object.defineProperty(video, 'currentTime', { configurable: true, get: () => current });
          window.__setObservationVideoTime = value => { current = value; };
          const createCircleMarker = L.circleMarker;
          window.__djiPlaybackMarker = null;
          L.circleMarker = (coordinates, options) => {
            const layer = createCircleMarker(coordinates, options);
            if (options && options.className === 'dji-flight-current-position') {
              window.__djiPlaybackMarker = layer;
            }
            return layer;
          };
        }""")
        self.page.locator("#show-dji-track").click()
        expect(self.page.locator("#show-dji-track")).to_have_text("隐藏无人机航迹")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("不能表示镜头拍到的地面位置")
        positions = self.page.evaluate("""() => {
          const video = document.getElementById('observation-video');
          const marker = window.__djiPlaybackMarker;
          video.dispatchEvent(new Event('timeupdate'));
          const readPoint = () => marker ? [marker.getLatLng().lat, marker.getLatLng().lng] : null;
          const first = readPoint();
          window.__setObservationVideoTime(1.05);
          video.dispatchEvent(new Event('timeupdate'));
          const second = readPoint();
          window.__setObservationVideoTime(10);
          video.dispatchEvent(new Event('timeupdate'));
          return { first, second, hidden: !marker || marker._map === null };
        }""")
        self.assertNotEqual(positions["first"], positions["second"], "video playback time must update the map hint")
        self.assertTrue(positions["hidden"], "a time far from the subtitle must not retain a stale GPS marker")
        self.page.set_viewport_size({"width": 320, "height": 844})
        self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), 320)
        self.page.set_viewport_size({"width": 1280, "height": 1000})
        self.page.screenshot(path=str(self.qa_dir / "dji-track-preview.png"), full_page=True)
        self.assertEqual(self.snap_requests, [], "showing aircraft position must not bind it to a road")

        self.page.locator("#show-dji-track").click()
        expect(self.page.locator("#show-dji-track")).to_have_text("在地图上查看无人机航迹")

    def test_dji_subtitle_outside_video_duration_warns_but_keeps_static_track_preview(self):
        self.open_manager()
        self.page.evaluate("""() => {
          const video = document.getElementById('observation-video');
          Object.defineProperty(video, 'duration', { configurable: true, get: () => 2 });
        }""")
        self.page.locator("#media-file").set_input_files({
            "name": "DJI_offset.mp4", "mimeType": "video/mp4", "buffer": b"test video",
        })
        self.page.locator("#dji-subtitle-file").set_input_files({
            "name": "DJI_offset.SRT", "mimeType": "text/plain",
            "buffer": (
                b"1\n00:00:08,000 --> 00:00:08,200\n"
                b"[latitude: 30.536000] [longitude: 114.363000]\n\n"
                b"2\n00:00:09,000 --> 00:00:09,200\n"
                b"[latitude: 30.537000] [longitude: 114.364000]"
            ),
        })
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("字幕时间范围与视频时长不重叠")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("请核对视频和字幕是否配套")
        self.page.locator("#show-dji-track").click()
        expect(self.page.locator("#show-dji-track")).to_have_text("隐藏无人机航迹")
        self.page.locator("#media-file").set_input_files({
            "name": "different-flight.mp4", "mimeType": "video/mp4", "buffer": b"another video",
        })
        expect(self.page.locator("#show-dji-track")).to_be_hidden()
        self.assertEqual(self.snap_requests, [])

    def test_mobile_controls_fit_and_custom_time_can_be_typed(self):
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.open_manager()
        self.prepare_event()
        self.page.locator("label:has(#event-mode-bike)").tap()
        expect(self.page.get_by_role("checkbox", name="骑行", exact=True)).to_be_checked()
        expect(self.page.get_by_role("checkbox", name="步行", exact=True)).to_be_checked()
        self.page.locator("#event-start-mode").select_option("scheduled")
        self.page.locator("#event-duration").select_option("custom")
        self.page.locator("#event-start-date").fill("2099-01-02")
        self.page.locator("#event-start-clock").fill("09:15")
        self.page.locator("#event-end-date").fill("2099-01-02")
        self.page.locator("#event-end-clock").fill("10:15")
        expect(self.page.locator("#publish-event")).to_be_enabled()
        self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), 390)
        self.page.set_viewport_size({"width": 320, "height": 844})
        self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), 320)
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.page.locator(".form-card").screenshot(path=str(self.qa_dir / "mobile-form.png"))
        self.page.set_viewport_size({"width": 1440, "height": 1100})
        self.page.locator(".form-card").screenshot(path=str(self.qa_dir / "desktop-form.png"))

    def test_vision_source_resets_to_finite_duration_and_requires_evidence(self):
        self.vision_jobs = [{
            "job_id": "a6f6c6b1-d237-43fa-8ea2-b01ae48a3e8e", "status": "completed",
            "media_kind": "image", "original_name": "航拍.png",
            "result": {"metrics": {"peak_vehicle_count": 3}, "candidates": []},
        }]
        self.open_manager()
        self.page.locator("#event-duration").select_option("ongoing")
        self.page.get_by_role("button", name="依据原片手工登记路况").click()
        expect(self.page.locator("#event-duration")).to_have_value("30")
        expect(self.page.locator("#event-ongoing-option")).to_have_js_property("disabled", True)
        self.page.locator("#event-type").select_option("event")
        self.page.locator("#event-name").fill("测试路人流提示")
        self.page.locator("#event-action").select_option("notice")
        self.page.get_by_role("checkbox", name="步行", exact=True).check()
        self.page.locator("#manager-map").click(position={"x": 180, "y": 180})
        self.page.locator("#confirm-picked-road").click()
        expect(self.page.locator("#publish-guidance")).to_contain_text("现场核实依据")
        expect(self.page.locator("#publish-event")).to_be_disabled()
        self.page.locator("#field-confirmation").fill("现场管理员已确认该路段存在人流聚集。")
        expect(self.page.locator("#publish-event")).to_be_enabled()
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#form-message")).to_contain_text("事件已发布")
        body = self.posts[0]
        self.assertEqual(body["source_vision_job_id"], self.vision_jobs[0]["job_id"])
        self.assertIsNone(body["source_vision_candidate_index"])
        self.assertIn("end_time", body)
        expect(self.page.locator("#event-ongoing-option")).to_have_js_property("disabled", False)

    def test_expired_incomplete_time_and_slowdown_cost_show_blocking_reason(self):
        self.open_manager()
        self.prepare_event()
        self.page.locator("#event-start-mode").select_option("scheduled")
        self.page.locator("#event-duration").select_option("custom")
        self.page.locator("#event-start-date").fill("2000-01-02")
        self.page.locator("#event-start-clock").fill("09:15")
        self.page.locator("#event-end-date").fill("2000-01-02")
        self.page.locator("#event-end-clock").fill("10:15")
        expect(self.page.locator("#publish-guidance")).to_contain_text("结束时间已过去")
        expect(self.page.locator("#publish-event")).to_be_disabled()
        self.page.locator("#event-end-date").fill("")
        expect(self.page.locator("#event-time-summary")).to_contain_text("请填写完整")
        self.page.locator("#event-start-mode").select_option("now")
        self.page.locator("#event-duration").select_option("60")
        self.page.locator("#event-action").select_option("slowdown")
        self.page.locator("#event-cost").fill("0")
        expect(self.page.locator("#publish-guidance")).to_contain_text("缓行成本倍数")
        expect(self.page.locator("#publish-event")).to_be_disabled()
        self.page.locator("#event-cost").fill("2.5")
        expect(self.page.locator("#publish-event")).to_be_enabled()
        self.page.locator("#publish-event").click()
        expect(self.page.locator("#form-message")).to_contain_text("事件已发布")
        self.assertEqual(self.posts[0]["cost_multiplier"], 2.5)


    def test_matching_dji_subtitle_fills_time_without_using_gps_for_road_binding(self):
        self.open_manager()
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), 390)
        self.page.locator("#media-file").set_input_files({
            "name": "DJI_1081.mp4", "mimeType": "video/mp4", "buffer": b"test-video",
        })
        with self.page.expect_file_chooser() as chooser_info:
            self.page.locator("#pick-dji-subtitle").click()
        chooser_info.value.set_files({
            "name": "DJI_1081.SRT", "mimeType": "text/plain",
            "buffer": b"2025-06-15 15:47:50.149 [latitude:30.536000] [longitude:114.363000]\n"
                      b"2025-06-15 15:47:50.182 [latitude:30.537000] [longitude:114.364000]",
        })
        expect(self.page.locator("#captured-at")).to_have_value("2025-06-15T15:47:50")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("读到 2 个机身 GPS 位置点")
        expect(self.page.locator("#show-dji-track")).to_be_visible()
        self.page.locator("#show-dji-track").click()
        expect(self.page.locator("#show-dji-track")).to_have_text("隐藏无人机航迹")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("不能表示镜头拍到的地面位置")
        self.assertEqual(self.snap_requests, [])
        self.page.locator("#captured-at").fill("2026-10-12T08:00")
        self.page.locator("#dji-subtitle-file").set_input_files({
            "name": "other.SRT", "mimeType": "text/plain",
            "buffer": b"2025-06-15 16:00:00.149 [latitude:x] [longitude:y]",
        })
        expect(self.page.locator("#captured-at")).to_have_value("2026-10-12T08:00")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("文件名与视频不一致")
        expect(self.page.locator("#show-dji-track")).to_be_hidden()
        self.assertEqual(self.snap_requests, [])
        self.page.set_viewport_size({"width": 320, "height": 844})
        self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), 320)


if __name__ == "__main__":
    unittest.main()
