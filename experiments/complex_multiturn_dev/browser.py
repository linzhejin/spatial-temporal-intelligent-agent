"""Drive the actual browser UI against the same isolated Flask/worker stack."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class BrowserCases:
    def __init__(self, root, database_url, workflow):
        from playwright.sync_api import sync_playwright
        self.database_url, self.workflow = database_url, workflow
        self.server = ThreadingHTTPServer(("127.0.0.1", 0),
            partial(_QuietHandler, directory=str(root / "static")))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)

    def new_case(self, client):
        return BrowserCase(self, client)

    def close(self):
        self.browser.close()
        self.playwright.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class BrowserCase:
    def __init__(self, host, client):
        self.host, self.client = host, client
        self.conversation = None
        self.last_run = None
        self.last_body = None
        self.last_run_id = None
        self.jobs = {}
        self.errors = []
        self.context = host.browser.new_context(viewport={"width": 1280, "height": 900},
                                                service_workers="block")
        self.context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        self.page = self.context.new_page()
        self.page.on("pageerror", lambda error: self.errors.append(str(error)[:300]))
        self.page.route("**/*", self._bridge)
        self.page.goto(f"http://127.0.0.1:{host.server.server_port}/index.html", wait_until="networkidle")

    def _bridge(self, route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            route.abort()
            return
        if not parsed.path.startswith("/api/"):
            route.continue_()
            return
        target = parsed.path + ("?" + parsed.query if parsed.query else "")
        response = self.client.open(target, method=request.method, data=request.post_data,
                                    content_type=request.headers.get("content-type", "application/json"))
        payload = response.get_json(silent=True) or {}
        if parsed.path == "/api/conversations" and request.method == "POST" and response.status_code == 201:
            self.conversation = payload["data"]["conversation_id"]
        if parsed.path.endswith("/messages") and request.method == "POST" and response.status_code == 202:
            self.last_body = request.post_data_json or {}
            run_id = payload["data"]["run_id"]
            self.last_run_id = run_id
            if run_id not in self.jobs:
                job = {"outcome": None, "error": None}
                self.jobs[run_id] = job

                def process():
                    from jobs.worker import process_next_run
                    try:
                        job["outcome"] = process_next_run(self.host.database_url,
                                                          "complex-ui-" + uuid4().hex[:8],
                                                          self.host.workflow, lease_seconds=75)
                    except Exception as error:
                        job["error"] = error

                job["thread"] = threading.Thread(target=process, daemon=True)
                job["thread"].start()
        headers = {key.lower(): value for key, value in response.headers.items()
                   if key.lower() in {"content-type", "set-cookie", "cache-control"}}
        route.fulfill(status=response.status_code, headers=headers, body=response.get_data())

    def turn(self, query, *, selected=None):
        from playwright.sync_api import expect
        self.last_run, self.last_body, self.last_run_id = None, None, None
        with self.page.expect_response(lambda response: response.request.method == "POST"
                                       and response.url.endswith("/messages"), timeout=120000) as accepted:
            if selected:
                card = self.page.locator(".candidate-cards").last.locator(".candidate-card").first
                expect(card).to_be_visible(timeout=10000)
                if card.get_attribute("data-poi-id") != selected:
                    raise RuntimeError("first visible card differs from selected candidate")
                card.click()
            else:
                self.page.locator("#nl-input").fill(query)
                self.page.locator("#submit-btn").click()
        if accepted.value.status != 202 or not self.last_run_id:
            raise RuntimeError("browser message was not accepted and processed")
        job = self.jobs[self.last_run_id]
        deadline = time.monotonic() + 100
        while job["thread"].is_alive() and time.monotonic() < deadline:
            self.page.wait_for_timeout(200)
        if job["thread"].is_alive():
            raise TimeoutError("browser worker did not finish")
        if job["error"]:
            raise RuntimeError("browser worker run could not be collected") from job["error"]
        outcome = job["outcome"]
        collected = self.client.get(f"/api/runs/{self.last_run_id}?conversation_id={self.conversation}")
        if collected.status_code != 200 or not outcome or outcome.get("run_id") != self.last_run_id:
            raise RuntimeError("browser worker run could not be collected")
        self.last_run = collected.get_json()["data"]
        result = self.last_run.get("result") or {}
        evidence = {"page_errors": list(self.errors), "chat_visible": False, "cards_match": None,
                    "route_panel_visible": None, "route_map_path": None, "selected_matches": None}
        text = result.get("assistant_text") or result.get("message") or result.get("reply") or ""
        try:
            if text:
                expect(self.page.locator("#chat-content")).to_contain_text(text, timeout=15000)
                evidence["chat_visible"] = True
            candidates = result.get("candidates") or []
            if candidates:
                group = self.page.locator(".candidate-cards").last
                expect(group.locator(".candidate-card").first).to_be_visible(timeout=15000)
                visible = group.locator(".candidate-card").evaluate_all(
                    "cards => cards.map(card => card.dataset.poiId).filter(Boolean)")
                evidence["cards_match"] = visible == [row.get("poi_id") for row in candidates]
            if result.get("response_kind") == "route" and result.get("route_state"):
                expect(self.page.locator("#results-section")).to_be_visible(timeout=15000)
                evidence["route_panel_visible"] = True
                evidence["route_map_path"] = self.page.locator(
                    "#map-container .leaflet-overlay-pane path.leaflet-interactive").count() > 0
            if selected:
                evidence["selected_matches"] = self.last_body.get("selected_poi_id") == selected
        except Exception as error:
            evidence["ui_error"] = type(error).__name__
        return self.last_run, self.last_body, evidence

    def close(self, screenshot):
        try:
            self.page.screenshot(path=str(screenshot), full_page=True)
        finally:
            self.context.close()
