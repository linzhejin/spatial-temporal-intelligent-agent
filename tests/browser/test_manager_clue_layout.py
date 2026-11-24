"""Browser layout regression checks for the manager clue workspace."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@pytest.mark.skipif(sync_playwright is None, reason="install Playwright to run layout checks")
def test_clue_workspace_fits_desktop_and_mobile_viewports():
    root = Path(__file__).resolve().parents[2]
    html = (root / "static" / "manager.html").read_text(encoding="utf-8")
    css = (root / "static" / "css" / "manager.css").read_text(encoding="utf-8")
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<link\s+rel="stylesheet"[^>]*>', "", html, flags=re.IGNORECASE)
    html = html.replace('id="workspace" class="workspace" hidden', 'id="workspace" class="workspace"')
    html = html.replace('id="login-panel" class="login-panel"', 'id="login-panel" class="login-panel" hidden')
    html = html.replace("</head>", f"<style>{css}</style></head>", 1)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_content(html, wait_until="load")
            assert page.locator("#clue-create").get_attribute("open") is None

            page.locator("#clue-create > summary").click()
            assert page.locator("#clue-create").get_attribute("open") is not None

            for width in (1440, 1200, 1101, 1100, 900, 680, 390, 320):
                page.set_viewport_size({"width": width, "height": 1000})
                result = page.evaluate("""() => {
                  const box = (selector) => {
                    const rect = document.querySelector(selector).getBoundingClientRect();
                    return {left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom};
                  };
                  const edges = (el) => {
                    const rect = el.getBoundingClientRect();
                    return {left: rect.left, right: rect.right};
                  };
                  return {
                    filters: box('.clue-filters'),
                    panel: box('.clue-list-panel'),
                    detail: box('.clue-detail'),
                    filterChildren: [...document.querySelector('.clue-filters').children].map(edges),
                    filterControls: [...document.querySelectorAll('.clue-filters input, .clue-filters select, .clue-filters button')].map(edges),
                    registrationControls: [...document.querySelectorAll('.clue-create-card input, .clue-create-card select, .clue-create-card textarea')]
                      .filter((el) => el.getClientRects().length)
                      .map((el) => ({control: edges(el), card: edges(el.closest('.clue-create-card'))})),
                    documentOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
                    eventsDivider: getComputedStyle(document.querySelector('#event-history')).borderTopWidth,
                  };
                }""")

                assert result["documentOverflow"] <= 0, f"horizontal page overflow at {width}px"
                assert result["filters"]["left"] >= result["panel"]["left"] - 1, f"filter starts outside list panel at {width}px"
                assert result["filters"]["right"] <= result["panel"]["right"] + 1, f"filter exceeds list panel at {width}px"
                assert all(rect["left"] >= result["filters"]["left"] - 1 and rect["right"] <= result["filters"]["right"] + 1 for rect in result["filterChildren"]), f"filter row overflows at {width}px"
                assert all(rect["left"] >= result["filters"]["left"] - 1 and rect["right"] <= result["filters"]["right"] + 1 for rect in result["filterControls"]), f"filter control overflows at {width}px"
                assert all(control["control"]["left"] >= control["card"]["left"] - 1 and control["control"]["right"] <= control["card"]["right"] + 1 for control in result["registrationControls"]), f"expanded registration control overflows at {width}px"
                assert result["eventsDivider"] == "1px", "event history should be separated from clue review"
                if width > 1100:
                    assert result["detail"]["left"] >= result["panel"]["right"] - 1, f"detail overlaps list at {width}px"
                else:
                    assert result["detail"]["top"] >= result["panel"]["bottom"] - 1, f"detail does not stack below list at {width}px"

            page.set_viewport_size({"width": 320, "height": 1000})
            page.locator("#clue-create > summary").click()
            assert page.locator("#clue-create").get_attribute("open") is None
            page.locator("#clue-create > summary").click()
            assert page.locator("#clue-create").get_attribute("open") is not None

            page.evaluate("""() => {
              const list = document.querySelector('#clue-list');
              const row = document.createElement('article');
              row.className = 'clue-row';
              row.innerHTML = '<div class="clue-row-head"><span class="clue-kind">官方通知</span><span class="clue-id">clue_' + 'x'.repeat(100) + '</span></div><div class="clue-summary">施工线索及较长的道路摘要内容</div><div class="clue-meta">等待人工核对原始证据</div>';
              list.replaceChildren(row);
              document.querySelector('#clue-detail').innerHTML = '<div class="clue-detail-head"><h3>施工线索详情标题</h3><span class="event-status">待复核</span></div><p class="clue-id-line">线索编号：' + 'clue_' + 'x'.repeat(100) + '</p><section class="clue-evidence"><h4>原始证据</h4><pre class="clue-raw-text">' + '长证据文本'.repeat(100) + '</pre></section>';
            }""")
            populated = page.evaluate("""() => {
              const boxes = [...document.querySelectorAll('.clue-row, .clue-detail, .clue-id, .clue-id-line, .clue-raw-text')].map((el) => {
                const rect = el.getBoundingClientRect();
                return {left: rect.left, right: rect.right, scrollWidth: el.scrollWidth, clientWidth: el.clientWidth};
              });
              return {boxes, overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth};
            }""")
            assert populated["overflow"] <= 0, "populated clue details cause horizontal page overflow at 320px"
            assert all(box["right"] <= 321 and box["left"] >= 0 for box in populated["boxes"]), "populated clue content exceeds the mobile viewport"
            assert all(box["scrollWidth"] <= box["clientWidth"] + 1 for box in populated["boxes"]), "long clue text or IDs overflow their cards"

            page.evaluate("window.scrollTo(0, 0)")
            page.locator('.workflow-nav a[href="#clue-workspace"]').click()
            heading = page.locator("#clue-workspace .events-heading").bounding_box()
            nav = page.locator(".workflow-nav").bounding_box()
            assert heading and nav and heading["y"] >= nav["y"] + nav["height"] - 1, "clue heading is hidden behind the sticky navigation"
        finally:
            browser.close()
