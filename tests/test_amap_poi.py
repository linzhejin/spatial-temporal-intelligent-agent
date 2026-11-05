"""Provider data must establish identity before it can move a navigation endpoint."""
import pytest

from spatial.amap_poi import normalize_poi, choose_match, lookup
from scripts.fetch.fetch_pois_amap_full import keep_poi


def test_campus_names_are_not_rejected_by_ambiguous_substrings():
    assert keep_poi('茶港门', '武汉大学', '190301')
    assert keep_poi('武汉大学学生服务中心', '', '141200')
    assert keep_poi('武汉大学健身中心', '', '080100')
    assert not keep_poi('茶港门外卖柜', '', '070000')


def test_empty_provider_arrays_are_not_coordinates_or_aliases():
    row = normalize_poi({'name': '卓尔体育馆', 'location': '114.36,30.54',
                         'alias': [], 'entr_location': [], 'parent': []})
    assert row['aliases'] == []
    assert row['navigation_coordinates'] is None
    assert row['parent_id'] == ''


def test_wrong_gate_is_never_an_exact_match():
    rows = [normalize_poi({'name': '凌波门', 'location': '114.36,30.54'})]
    assert choose_match('澄波门', rows) is None


def test_parent_building_and_locker_do_not_match_gate():
    rows = [normalize_poi({'name': name, 'location': '114.36,30.54'})
            for name in ['武汉大学', '茶港门外卖柜']]
    assert choose_match('茶港门', rows) is None


def test_exact_identity_preserves_entrance_and_alias_without_moving_center():
    row = normalize_poi({'name': '武汉大学卓尔体育馆', 'location': '114.36,30.54',
                         'entr_location': '114.3601,30.5401', 'alias': '卓尔馆|卓尔体育馆'})
    found = choose_match('卓尔体育馆', [row], {'lon': 114.36, 'lat': 30.54})
    assert found['coordinates'] == {'lng': 114.36, 'lat': 30.54}
    assert found['navigation_coordinates'] == {'lng': 114.3601, 'lat': 30.5401}
    assert found['aliases'] == ['卓尔馆', '卓尔体育馆']


def test_choose_match_accepts_flat_candidate_lng_lat_coordinates():
    row = normalize_poi({'name': '椒大喜铜炉火锅鸡(武大店)', 'location': '114.356807,30.532098'})
    found = choose_match(row['name'], [row], {
        'poi_id': 'amap:B0K3M1Q398', 'name': row['name'], 'lng': 114.356807, 'lat': 30.532098,
    })
    assert found is row


def test_same_name_far_away_or_ambiguous_is_not_selected():
    row = normalize_poi({'name': '体育馆', 'location': '114.38,30.54'})
    assert choose_match('体育馆', [row], {'lon': 114.35, 'lat': 30.54}) is None
    other = normalize_poi({'name': '体育馆', 'location': '114.36,30.54'})
    assert choose_match('体育馆', [row, other]) is None


def test_remote_entrance_does_not_override_local_destination():
    row = normalize_poi({'name': '总馆', 'location': '114.36,30.54',
                         'entr_location': '114.38,30.54'})
    assert row['navigation_coordinates'] is None


def test_online_lookup_searches_all_wuhan_districts(monkeypatch):
    monkeypatch.setenv('AMAP_WEB_KEY', 'test-key')

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'status': '1', 'pois': [
                {'id': '1', 'name': '卓尔体育馆', 'location': '114.36,30.54'}]}

    class Client:
        def __init__(self):
            self.params = None

        def get(self, url, params):
            self.params = params
            return Response()

    client = Client()
    assert lookup('卓尔体育馆', client=client)['name'] == '卓尔体育馆'
    assert client.params['city'] == '420100'
    assert client.params['citylimit'] == 'false'


def test_nearby_search_is_callable_outside_flask_and_returns_live_provider_identity(monkeypatch):
    from spatial.amap_poi import search_nearby

    monkeypatch.setenv('AMAP_WEB_KEY', 'test-key')

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'status': '1', 'infocode': '10000', 'count': '1', 'pois': [{
                'id': 'B0MEIYUAN01', 'name': '肯德基(武大校园店)',
                'type': '餐饮服务;快餐厅', 'typecode': '050301',
                'location': '114.365149,30.536069', 'distance': '224',
                'address': '梅园食堂二层', 'alias': '肯德基',
            }]}

    class Client:
        def __init__(self):
            self.calls = []

        def get(self, url, params):
            self.calls.append((url, params))
            return Response()

    client = Client()
    result = search_nearby(
        {'lng': 114.3648, 'lat': 30.5362}, keywords='餐饮', radius_m=1000, client=client,
    )

    assert result['status'] == 'ok'
    assert len(result['candidates']) == 1
    candidate = result['candidates'][0]
    assert candidate['id'] == 'amap:B0MEIYUAN01'
    assert candidate['source'] == 'amap_online'
    assert candidate['coordinate_system'] == 'GCJ-02'
    assert candidate['distance_m'] == 224
    assert candidate['coordinates'] == {'lng': 114.365149, 'lat': 30.536069}
    assert client.calls[0][1]['radius'] == 1000
    assert client.calls[0][1]['location'] == '114.3648,30.5362'
    assert client.calls[0][1]['key'] == 'test-key'


def test_nearby_search_distinguishes_missing_key_from_empty_result(monkeypatch):
    from spatial.amap_poi import search_nearby

    monkeypatch.delenv('AMAP_WEB_KEY', raising=False)
    monkeypatch.delenv('AMAP_KEY', raising=False)

    result = search_nearby({'lng': 114.36, 'lat': 30.53}, keywords='餐饮')

    assert result == {'status': 'missing_key', 'candidates': []}


def test_nearby_search_returns_provider_errors_without_echoing_credentials(monkeypatch):
    from spatial.amap_poi import search_nearby

    monkeypatch.setenv('AMAP_WEB_KEY', 'never-print-this-key')

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'status': '0', 'infocode': '10003', 'info': 'INVALID_USER_KEY', 'pois': []}

    class Client:
        def get(self, url, params):
            assert params['key'] == 'never-print-this-key'
            return Response()

    result = search_nearby({'lng': 114.36, 'lat': 30.53}, keywords='餐饮', client=Client())

    assert result == {'status': 'provider_error', 'provider_code': '10003', 'candidates': []}


def test_text_search_supports_offline_recommendations_without_a_start(monkeypatch):
    from spatial.amap_poi import search_text
    monkeypatch.setenv('AMAP_WEB_KEY', 'test-key')

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'status': '1', 'count': '1', 'pois': [{
                'id': 'B0TEXT001', 'name': '周麻婆(梅园店)', 'type': '餐饮服务;中餐厅',
                'typecode': '050100', 'location': '114.365,30.536', 'address': '梅园CBD',
            }]}

    class Client:
        def __init__(self):
            self.calls = []

        def get(self, url, params):
            self.calls.append((url, params))
            return Response()

    client = Client()
    result = search_text('梅园 火锅', client=client)

    assert result['status'] == 'ok'
    assert result['candidates'][0]['id'] == 'amap:B0TEXT001'
    assert result['candidates'][0]['subcategory'] == 'restaurant'
    assert client.calls[0][0].endswith('/v3/place/text')
    assert client.calls[0][1]['keywords'] == '梅园 火锅'
    assert client.calls[0][1]['key'] == 'test-key'


def test_text_search_distinguishes_missing_key_from_provider_empty(monkeypatch):
    from spatial.amap_poi import search_text
    monkeypatch.delenv('AMAP_WEB_KEY', raising=False)
    monkeypatch.delenv('AMAP_KEY', raising=False)
    assert search_text('梅园 火锅') == {'status': 'missing_key', 'candidates': []}


@pytest.mark.parametrize(('type_code', 'expected'), [
    ('050101', 'restaurant'),
    ('050301', 'fastfood'),
    ('050500', 'coffee'),
    ('050600', 'tea_drink'),
])
def test_nearby_search_preserves_specific_provider_dining_category(monkeypatch, type_code, expected):
    from spatial.amap_poi import search_nearby

    monkeypatch.setenv('AMAP_WEB_KEY', 'test-key')

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'status': '1', 'count': '1', 'pois': [{
                'id': 'B0CATEGORY', 'name': '测试地点', 'type': '餐饮服务',
                'typecode': type_code, 'location': '114.36,30.53', 'distance': '20',
            }]}

    class Client:
        def get(self, _url, params):
            assert params['page'] == 1
            return Response()

    result = search_nearby({'lng': 114.36, 'lat': 30.53}, client=Client())

    assert result['candidates'][0]['subcategory'] == expected


def test_lookup_by_id_refreshes_only_requested_provider_identity(monkeypatch):
    from spatial.amap_poi import lookup_by_id

    monkeypatch.setenv('AMAP_WEB_KEY', 'test-key')

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'status': '1', 'pois': [{
                'id': 'B0MEIYUAN01', 'name': '肯德基(武大校园店)',
                'type': '餐饮服务;快餐厅', 'typecode': '050301',
                'location': '114.365149,30.536069',
            }]}

    class Client:
        def get(self, url, params):
            assert url.endswith('/v3/place/detail')
            assert params['id'] == 'B0MEIYUAN01'
            return Response()

    candidate = lookup_by_id('amap:B0MEIYUAN01', client=Client())

    assert candidate['id'] == 'amap:B0MEIYUAN01'
    assert candidate['name'] == '肯德基(武大校园店)'
