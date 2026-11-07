"""Nearby candidate controls remain compact and coincident map locations selectable."""
import json
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import expect, sync_playwright

from tests.browser.test_strategy_and_food_paging import static_site  # noqa: F401


@pytest.mark.parametrize('mobile', [False, True], ids=['desktop', 'mobile-size'])
def test_overlapping_map_places_can_be_chosen_individually(static_site, mobile, tmp_path):  # noqa: F811
    submissions = []
    candidates = [{'poi_id': f'poi_{i}', 'name': f'梅园餐饮地点{i}', 'subcategory': 'restaurant',
                   'campus': '文理学部', 'distance_m': 100 + i,
                   'distance_reference_name': '总图',
                   'distance_basis': 'network_path_plus_unverified_straight_line_access_estimate',
                   'lng': 114.36, 'lat': 30.54} for i in range(1, 5)]

    def respond(route, data):
        route.fulfill(status=200, content_type='application/json; charset=utf-8',
                      body=json.dumps({'data': data}, ensure_ascii=False))

    def handle(route):
        req, parsed = route.request, urlsplit(route.request.url)
        if parsed.hostname not in {'127.0.0.1', 'localhost'}:
            route.abort()
        elif parsed.path == '/api/conversations' and req.method == 'POST':
            respond(route, {'conversation_id': 'map-overlap-session'})
        elif parsed.path.endswith('/messages') and req.method == 'POST':
            submissions.append(req.post_data_json)
            respond(route, {'run_id': f'map-overlap-{len(submissions)}'})
        elif parsed.path.startswith('/api/runs/'):
            result = ({'task_type': 'chat', 'response_kind': 'chat', 'reply': '已选择地点。'}
                      if submissions[-1].get('selected_poi_id') else
                      {'task_type': 'path_planning', 'response_kind': 'candidates',
                       'message': '请选择地点。', 'candidates': candidates})
            respond(route, {'status': 'completed', 'events': [], 'task_id': 'map-task',
                            'current_task_revision': 1, 'result': result})
        elif parsed.path.startswith('/api/'):
            respond(route, {})
        else:
            route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={'width': 390, 'height': 844} if mobile else
                                       {'width': 1280, 'height': 900},
                                       is_mobile=mobile, has_touch=mobile, service_workers='block')
        context.add_init_script("localStorage.setItem('whu_welcome_seen', 'true');")
        page = context.new_page()
        page.route('**/*', handle)
        page.goto(static_site, wait_until='domcontentloaded')
        page.locator('#nl-input').fill('我在信图，想去总图附近干饭')
        page.locator('#submit-btn').click()
        cards = page.locator('.candidate-cards').last
        expect(cards.locator('.candidate-card')).to_have_count(4)
        distance_text = cards.locator('.candidate-dist').first.inner_text()
        assert '总图' in distance_text and '101' in distance_text
        assert '路网距' not in distance_text
        assert cards.locator('.candidate-card').first.bounding_box()['height'] <= 70
        marker = page.locator('.candidate-map-pin')
        expect(marker).to_have_count(1)
        identities = marker.evaluate(
            "el => JSON.parse(el.closest('.leaflet-marker-icon').dataset.poiIdentities)")
        assert [item['poi_id'] for item in identities] == [f'poi_{i}' for i in range(1, 5)]
        assert all(item['lng'] == 114.36 and item['lat'] == 30.54 for item in identities)
        marker.click()
        choices = page.locator('.candidate-map-choice')
        expect(choices).to_have_count(4)
        expect(choices.first).to_be_in_viewport()
        assert choices.first.evaluate("el => { const r=el.getBoundingClientRect(), m=document.querySelector('#map-container').getBoundingClientRect(); return r.top >= m.top && r.bottom <= m.bottom; }")
        choices.last.scroll_into_view_if_needed()
        expect(choices.last).to_be_in_viewport()
        page.screenshot(path=str(tmp_path / 'candidate-map-overlap.png'), full_page=True)
        choices.filter(has_text='梅园餐饮地点3').click()
        expect(page.get_by_text('已选择地点。')).to_be_visible()
        assert submissions[-1]['selected_poi_id'] == 'poi_3'
        expect(page.locator('.candidate-map-pin')).to_have_count(0)
        context.close()
        browser.close()
