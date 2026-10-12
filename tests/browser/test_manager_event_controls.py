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
        self.pending_posts = []
        self.hold_posts = False
        self.fail_posts = False
        self.vision_jobs = []
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
            data = {"conditions": []}
        elif path == "/api/road-conditions/snap":
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
            "buffer": b"2025-06-15 15:47:50.149 [latitude:x] [longitude:y]\n"
                      b"2025-06-15 15:47:50.182 [latitude:x] [longitude:y]",
        })
        expect(self.page.locator("#captured-at")).to_have_value("2025-06-15T15:47:50")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("GPS 记录 2 条")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("不用于道路定位")
        self.page.locator("#captured-at").fill("2026-10-12T08:00")
        self.page.locator("#dji-subtitle-file").set_input_files({
            "name": "other.SRT", "mimeType": "text/plain",
            "buffer": b"2025-06-15 16:00:00.149 [latitude:x] [longitude:y]",
        })
        expect(self.page.locator("#captured-at")).to_have_value("2026-10-12T08:00")
        expect(self.page.locator("#dji-subtitle-status")).to_contain_text("文件名与视频不一致")
        self.page.set_viewport_size({"width": 320, "height": 844})
        self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), 320)


if __name__ == "__main__":
    unittest.main()
