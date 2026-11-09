"""Real service-worker upgrade and rollback for the trial manager assets."""
import io
import os
import re
import subprocess
import tarfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@pytest.mark.skipif(sync_playwright is None, reason="Playwright unavailable")
def test_manager_assets_survive_real_cache_upgrade_and_rollback(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    baseline = os.getenv("VISION_TRIAL_CACHE_BASELINE", "505a5d49c5bfa25a850814c60638f5d96b283405")
    archive = subprocess.check_output(["git", "archive", baseline, "static"], cwd=repo)
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(baseline_root, filter="data")
    old, new = baseline_root / "static", repo / "static"
    cache_name = lambda root: re.search(r"var CACHE_NAME = '([^']+)'",
                                        (root / "sw.js").read_text(encoding="utf8")).group(1)
    old_cache, new_cache = cache_name(old), cache_name(new)
    assert old_cache != new_cache
    # A deployed checkout gives the incoming files newer mtimes. Git archives
    # instead carry commit timestamps, which may be newer than our local edits.
    initial_mtime = min(time.time() - 30, (new / "sw.js").stat().st_mtime - 30)
    for name in ("sw.js", "index.html", "js/app.js", "css/style.css", "manager.html"):
        os.utime(old / name, (initial_mtime, initial_mtime))

    class Handler(SimpleHTTPRequestHandler):
        root = old
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(type(self).root), **kwargs)
        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()
        def log_message(self, *_args):
            pass

    def wait_for_cache(page, expected, absent):
        page.evaluate("""async names => {
            const until = Date.now() + 30000;
            while (Date.now() < until) {
                const keys = await caches.keys();
                if (keys.includes(names.expected) && !keys.includes(names.absent)) return;
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            throw new Error('cache transition timed out');
        }""", {"expected": expected, "absent": absent})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 390, "height": 844}, service_workers="allow")
            page = context.new_page()
            base = f"http://127.0.0.1:{server.server_port}"
            page.goto(base + "/index.html", wait_until="domcontentloaded")
            page.evaluate("navigator.serviceWorker.register('/sw.js')")
            page.wait_for_function("navigator.serviceWorker.controller !== null", timeout=30000)
            wait_for_cache(page, old_cache, new_cache)
            page.goto(base + "/manager.html", wait_until="domcontentloaded")
            assert "vision-safety-v1" in page.locator("script[src*='manager.js']").get_attribute("src")

            Handler.root = new
            page.evaluate("() => { window.__previousController = navigator.serviceWorker.controller; return true; }")
            page.evaluate("async () => { const reg = await navigator.serviceWorker.getRegistration(); await reg.update(); }")
            page.wait_for_function("navigator.serviceWorker.controller !== window.__previousController", timeout=30000)
            wait_for_cache(page, new_cache, old_cache)
            page.goto(base + "/manager.html", wait_until="domcontentloaded")
            asset = page.locator("script[src*='manager.js']").get_attribute("src")
            assert "vision-height-v1" in asset
            assert "appendTrialFeedback" in page.evaluate("async url => (await fetch(url)).text()", asset)
            assert page.locator("#vision-purpose").count() == 1

            page.evaluate("() => { window.__previousController = navigator.serviceWorker.controller; return true; }")
            rollback_mtime = max(time.time(), (new / "sw.js").stat().st_mtime) + 2
            for name in ("sw.js", "index.html", "js/app.js", "css/style.css", "manager.html"):
                os.utime(old / name, (rollback_mtime, rollback_mtime))
            Handler.root = old
            page.evaluate("async () => { const reg = await navigator.serviceWorker.getRegistration(); await reg.update(); }")
            page.wait_for_function("navigator.serviceWorker.controller !== window.__previousController", timeout=30000)
            wait_for_cache(page, old_cache, new_cache)
            page.goto(base + "/manager.html", wait_until="domcontentloaded")
            assert "vision-safety-v1" in page.locator("script[src*='manager.js']").get_attribute("src")
            context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
