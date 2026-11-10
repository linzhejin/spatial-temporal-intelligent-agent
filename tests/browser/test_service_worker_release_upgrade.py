"""Install a production shell, then verify the real browser activates this release."""
from __future__ import annotations

import io
import os
from pathlib import Path
import re
import subprocess
import tarfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


pytestmark = pytest.mark.skipif(sync_playwright is None, reason="Playwright browser is unavailable")


def test_production_service_worker_cache_upgrades_and_rolls_back(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    baseline = os.getenv("VISION_RELEASE_CACHE_BASELINE", "873f576d3b138847dc794a33354807e448f444ed")
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
    app_asset = lambda root: re.search(r'src="(/js/app\.js\?[^\"]+)"',
                                       (root / "index.html").read_text(encoding="utf-8")).group(1)
    old_asset, new_asset = app_asset(old_static), app_asset(new_static)
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

    def wait_for_cache(page, present, absent):
        # wait_for_function treats a Promise object as truthy; poll the awaited
        # cache keys inside the browser instead of accepting a pending Promise.
        page.evaluate("""async names => {
            const until = Date.now() + 30000;
            while (Date.now() < until) {
                const keys = await caches.keys();
                if (keys.includes(names.present) && !keys.includes(names.absent)) return;
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            throw new Error('cache transition timed out: ' + (await caches.keys()).join(','));
        }""", {"present": present, "absent": absent})

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
            wait_for_cache(page, old_cache, new_cache)
            before = page.evaluate("async () => await caches.keys()")
            assert old_cache in before

            DynamicHandler.root = new_static
            page.evaluate("() => { window.__controllerChanged = new Promise(resolve => "
                          "navigator.serviceWorker.addEventListener('controllerchange', resolve, {once: true})); return true; }")
            page.evaluate("async () => { const reg = await navigator.serviceWorker.getRegistration(); await reg.update(); }")
            wait_for_cache(page, new_cache, old_cache)
            page.evaluate("""() => Promise.race([
                window.__controllerChanged,
                new Promise((_, reject) => setTimeout(() => reject(new Error('controller change timed out')), 15000))
            ])""")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("() => navigator.serviceWorker.controller !== null", timeout=10000)

            result = page.evaluate("""async ({name, asset}) => {
                const cache = await caches.open(name);
                const script = await cache.match(asset);
                return {keys: await caches.keys(), app: script ? await script.text() : ''};
            }""", {"name": new_cache, "asset": new_asset})
            assert result["keys"] == [new_cache]
            assert result["app"].replace("\r\n", "\n") == (new_static / "js/app.js").read_text(encoding="utf-8")

            # A rollback must also replace the newly installed shell, not leave
            # a newer cached manager/client bundle paired with the old backend.
            page.wait_for_load_state("load")
            page.evaluate("""async () => {
                const until = Date.now() + 15000;
                while (Date.now() < until) {
                    const reg = await navigator.serviceWorker.getRegistration();
                    if (reg && reg.active === navigator.serviceWorker.controller && !reg.installing && !reg.waiting) return;
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
                throw new Error('service worker did not settle');
            }""")
            page.evaluate("() => { window.__previousController = navigator.serviceWorker.controller; }")
            # Git checkout updates changed files' mtimes when rolling back.
            # The extracted baseline otherwise has an older mtime than the
            # candidate and SimpleHTTPRequestHandler incorrectly returns 304.
            rollback_mtime = max(time.time(), (new_static / "sw.js").stat().st_mtime) + 2
            for name in ("sw.js", "index.html", "js/app.js", "css/style.css"):
                os.utime(old_static / name, (rollback_mtime, rollback_mtime))
            DynamicHandler.root = old_static
            page.evaluate("async () => { const reg = await navigator.serviceWorker.getRegistration(); await reg.update(); }")
            page.wait_for_function("() => navigator.serviceWorker.controller !== window.__previousController", timeout=30000)
            wait_for_cache(page, old_cache, new_cache)
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("() => navigator.serviceWorker.controller !== null", timeout=10000)
            rollback = page.evaluate("""async ({name, asset}) => {
                const cache = await caches.open(name);
                const script = await cache.match(asset);
                return {keys: await caches.keys(), app: script ? await script.text() : ''};
            }""", {"name": old_cache, "asset": old_asset})
            assert rollback["keys"] == [old_cache]
            assert rollback["app"].replace("\r\n", "\n") == (old_static / "js/app.js").read_text(encoding="utf-8")
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
