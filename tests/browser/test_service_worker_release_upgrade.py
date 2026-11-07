"""Install a production shell, then verify the real browser activates this release."""
from __future__ import annotations

import io
from pathlib import Path
import re
import subprocess
import tarfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


pytestmark = pytest.mark.skipif(sync_playwright is None, reason="Playwright browser is unavailable")


def test_production_service_worker_cache_upgrades_to_candidate(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    baseline = "bbfeafc942758b29e802d7e0f406fd4d1fd372bf"
    archive = subprocess.check_output(["git", "archive", baseline, "static"], cwd=repo)
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(baseline_root, filter="data")
    old_static = baseline_root / "static"
    new_static = repo / "static"
    old_sw = (old_static / "sw.js").read_text(encoding="utf-8")
    new_sw = (new_static / "sw.js").read_text(encoding="utf-8")
    old_cache = re.search(r"var CACHE_NAME = '([^']+)'", old_sw).group(1)
    new_cache = re.search(r"var CACHE_NAME = '([^']+)'", new_sw).group(1)
    assert old_cache != new_cache

    class DynamicHandler(SimpleHTTPRequestHandler):
        root = old_static

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(type(self).root), **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), DynamicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True,
                                          has_touch=True, service_workers="allow")
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="domcontentloaded")
            page.evaluate("navigator.serviceWorker.register('/sw.js')")
            page.wait_for_function("navigator.serviceWorker.controller !== null", timeout=15000)
            page.wait_for_function("async oldName => (await caches.keys()).includes(oldName)", arg=old_cache,
                                   timeout=15000)
            before = page.evaluate("async () => await caches.keys()")
            assert old_cache in before

            DynamicHandler.root = new_static
            page.evaluate("() => { window.__controllerChanged = new Promise(resolve => "
                          "navigator.serviceWorker.addEventListener('controllerchange', resolve, {once: true})); return true; }")
            page.evaluate("async () => { const reg = await navigator.serviceWorker.getRegistration(); await reg.update(); }")
            page.wait_for_function(
                "async args => { const keys = await caches.keys(); return keys.includes(args.newName) && !keys.includes(args.oldName); }",
                arg={"oldName": old_cache, "newName": new_cache}, timeout=30000)
            page.evaluate("() => window.__controllerChanged")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("async () => navigator.serviceWorker.controller !== null", timeout=10000)

            result = page.evaluate("""async newName => {
                const cache = await caches.open(newName);
                const script = await cache.match('/js/app.js?v=20261107-concise-dialogue-v2');
                return {keys: await caches.keys(), app: script ? await script.text() : ''};
            }""", new_cache)
            assert result["keys"] == [new_cache]
            assert "appendRouteStepNotice" in result["app"]
            assert "appendRouteFieldTruthNotice" not in result["app"]
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
