import json
from pathlib import Path

from spatial import poi


def test_user_reported_missing_shops_are_preserved_for_review_but_not_recommended():
    raw = json.loads((Path(__file__).resolve().parents[1] / 'data/pois.json').read_bytes())['pois']
    disputed = {'poi_346', 'poi_348', 'poi_351'}
    entries = [row for row in raw if row['id'] in disputed]
    assert len(entries) == 3
    assert all(row.get('recommendation_status') == 'reported_absent_pending_review' for row in entries)
    assert all(row.get('recommendation_review', {}).get('source') == 'user_report' for row in entries)
    assert not disputed.intersection(row['id'] for row in poi.reload_pois())
    assert not disputed.intersection(row['id'] for row in poi.search_by_category(poi_type='dining', limit=2000))
    for row in entries:
        found, alternatives = poi.find_poi_ambiguous(row['name'])
        assert found is None and not alternatives


def test_reported_existing_meiyuan_shops_are_not_removed():
    ids = {row['id'] for row in poi.load_pois()}
    assert {'poi_340', 'poi_347'} <= ids


def test_legacy_canteen_requires_a_record_specific_source_match():
    rows = {row['id']: row for row in poi.load_pois()}

    assert poi.has_source_backed_record(rows['poi_071'])
    assert not poi.has_source_backed_record(rows['poi_361'])


def test_reported_absent_online_shop_is_blocked_by_identity_and_location_only():
    raw = json.loads((Path(__file__).resolve().parents[1] / 'data/pois.json').read_bytes())['pois']
    absent = next(row for row in raw if row['id'] == 'poi_348')
    center = absent['coordinates']
    candidates = [
        {'id': 'amap:reported', 'name': absent['name'], 'coordinates': center},
        {'id': 'amap:other-branch', 'name': '麦当劳(光谷店)',
         'coordinates': {'lng': center['lng'] + 0.00001, 'lat': center['lat']}},
        {'id': 'amap:same-name-far-away', 'name': absent['name'],
         'coordinates': {'lng': center['lng'] + 0.01, 'lat': center['lat']}},
    ]

    remaining = poi.filter_reported_absent_provider_results(candidates)

    assert [row['id'] for row in remaining] == ['amap:other-branch', 'amap:same-name-far-away']


def test_feedback_data_change_invalidates_a_previous_candidate_snapshot(tmp_path, monkeypatch):
    import networkx as nx
    import config
    from agents import result_maintenance, route_state, tools

    raw = json.loads((Path(__file__).resolve().parents[1] / 'data/pois.json').read_bytes())
    legacy = json.loads(json.dumps(raw, ensure_ascii=False))
    for row in legacy['pois']:
        if row['id'] in {'poi_346', 'poi_348', 'poi_351'}:
            row.pop('recommendation_status', None)
            row.pop('recommendation_review', None)
    file = tmp_path / 'pois.json'
    monkeypatch.setattr(config, 'DATA_DIR', str(tmp_path))
    file.write_text(json.dumps(legacy, ensure_ascii=False), encoding='utf-8')
    graph = nx.MultiDiGraph()
    before = route_state.current_data_version(graph)
    file.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    after = route_state.current_data_version(graph)
    assert before != after
    events = route_state.current_road_condition_version([])
    monkeypatch.setattr(route_state, 'current_data_version', lambda: after)
    monkeypatch.setattr(tools, 'list_conditions', lambda **_kwargs: [])
    old = {'candidate_snapshot': {'data_version': before, 'road_condition_version': events}}
    new = {'candidate_snapshot': {'data_version': after, 'road_condition_version': events}}
    assert not result_maintenance.candidate_snapshot_is_current(old)
    assert result_maintenance.candidate_snapshot_is_current(new)
