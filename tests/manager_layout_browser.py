"""Browser regression for the manager map and sticky navigation stacking."""
from playwright.sync_api import sync_playwright


def check(page, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.route("**/js/manager.js*", lambda route: route.abort())
    page.goto("http://127.0.0.1:8765/manager.html", wait_until="networkidle")
    page.evaluate("""() => {
      document.querySelector('#login-panel').hidden = true;
      document.querySelector('#workspace').hidden = false;
      const map = L.map('manager-map').setView([30.54, 114.36], 16);
      L.rectangle([[30.535, 114.35], [30.545, 114.37]]).addTo(map);
    }""")
    page.wait_for_timeout(500)
    page.evaluate("window.scrollTo(0, 650)")
    page.wait_for_timeout(200)
    result = page.evaluate("""() => {
      const nav = document.querySelector('.workflow-nav');
      const rect = nav.getBoundingClientRect();
      const x = rect.left + rect.width / 2, y = rect.top + rect.height / 2;
      return {nav: nav.contains(document.elementFromPoint(x, y)),
        top: document.elementFromPoint(x, y).outerHTML.slice(0, 200)};
    }""")
    assert result['nav'], f"navigation covered at {width}px: {result['top']}"
    page.locator('a[href="#vision-workspace"]').click()
    page.wait_for_timeout(600)
    assert page.evaluate("""() => document.querySelector('#vision-workspace').getBoundingClientRect().top >= document.querySelector('.workflow-nav').getBoundingClientRect().bottom - 1"""), 'anchor hidden beneath navigation'


if __name__ == '__main__':
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for size in [(1440, 900), (390, 844)]:
            page = browser.new_page()
            check(page, *size)
            page.close()
        browser.close()
    print('manager layout browser checks passed')
