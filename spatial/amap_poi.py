"""Bounded online AMap lookup; provider data lives only in the current request.

Open master data is maintained separately. Exact name/alias and local proximity
must agree before a provider entrance can be used for a known destination.
"""
import math
import os
import re
from datetime import datetime, timezone

import httpx

def _text(value):
    return value.strip() if isinstance(value, str) else ''


def _coord(value):
    try:
        lng, lat = map(float, _text(value).split(','))
        if math.isfinite(lng) and math.isfinite(lat) and -180 <= lng <= 180 and -90 <= lat <= 90:
            return {'lng': lng, 'lat': lat}
    except (ValueError, TypeError):
        pass
    return None


def _distance(a, b):
    lat = math.radians((a['lat'] + b['lat']) / 2)
    return math.hypot((a['lng'] - b['lng']) * 111320 * math.cos(lat),
                      (a['lat'] - b['lat']) * 110540)


def _label(value):
    from spatial.poi import _normalize_num
    value = re.sub(r'^(武汉大学|武大)[\s\-—]*', '', _text(value))
    return re.sub(r'[\s\-—（）()]', '', _normalize_num(value))


def _brand_label(value):
    text = re.sub(r'^(武汉大学|武大)[\s\-—]*', '', _text(value))
    return _label(re.split(r'[·・/／|（(]', text, maxsplit=1)[0])


def normalize_poi(raw):
    center = _coord(raw.get('location'))
    if not center or not _text(raw.get('name')):
        return None
    entrance = _coord(raw.get('entr_location'))
    if entrance and _distance(center, entrance) > 150:
        entrance = None
    aliases = list(dict.fromkeys(x.strip() for x in re.split(r'[|;；]', _text(raw.get('alias'))) if x.strip()))
    return {'id': 'amap:' + _text(raw.get('id')), 'name': _text(raw.get('name')),
            'aliases': aliases, 'coordinates': center, 'lon': center['lng'], 'lat': center['lat'],
            'navigation_coordinates': entrance, 'coordinate_system': 'GCJ-02',
            'parent_id': _text(raw.get('parent')), 'navi_poiid': _text(raw.get('navi_poiid')),
            'type_code': _text(raw.get('typecode')), 'type_name': _text(raw.get('type')),
            'description': _text(raw.get('address')), 'source': 'amap_online',
            'verification_status': 'provider_only'}


def _dining_subcategory(raw):
    name = _text(raw.get('name'))
    type_code = _text(raw.get('typecode'))
    type_name = _text(raw.get('type'))
    if '食堂' in name or '食堂' in type_name:
        return 'canteen'
    if type_code.startswith('0503') or any(token in type_name for token in ('快餐', '小吃')):
        return 'fastfood'
    if type_code.startswith('0505') or '咖啡' in type_name:
        return 'coffee'
    if type_code.startswith('0506') or any(token in type_name for token in ('茶饮', '奶茶')):
        return 'tea_drink'
    if type_code.startswith(('0501', '0502', '0504')) or any(
            token in type_name for token in ('中餐', '外国餐厅', '休闲餐饮')):
        return 'restaurant'
    return 'other'


def _nearby_pages(client, params, pages):
    response = client.get('https://restapi.amap.com/v3/place/around',
                          params={**params, 'page': 1})
    response.raise_for_status()
    body = response.json()
    if str(body.get('status')) != '1':
        return None, int(body.get('count') or 0), str(body.get('infocode') or '')[:16]
    reported_count = int(body.get('count') or 0)
    raw_rows = list(body.get('pois') or [])
    for page in range(2, pages + 1):
        if len(raw_rows) < 25 or len(raw_rows) >= reported_count:
            break
        response = client.get('https://restapi.amap.com/v3/place/around',
                              params={**params, 'page': page})
        response.raise_for_status()
        body = response.json()
        if str(body.get('status')) != '1':
            return None, reported_count, str(body.get('infocode') or '')[:16]
        raw_rows.extend(body.get('pois') or [])
    return raw_rows, reported_count, None


def search_text(keywords, *, city="420100", pages=3, client=None):
    """Search live provider POIs by a model-selected name/area keyword."""
    query = _text(keywords)[:80]
    try:
        pages = int(pages)
    except (ValueError, TypeError, OverflowError):
        return {"status": "invalid_query", "candidates": []}
    if not query or not 1 <= pages <= 3:
        return {"status": "invalid_query", "candidates": []}
    key = os.getenv("AMAP_WEB_KEY") or os.getenv("AMAP_KEY")
    if not key:
        return {"status": "missing_key", "candidates": []}
    params = {"key": key, "keywords": query, "city": _text(city) or "420100",
              "citylimit": "false", "offset": 25, "extensions": "all"}
    try:
        if client is None:
            with httpx.Client(timeout=httpx.Timeout(3, connect=2)) as request_client:
                rows, reported_count, error = _text_pages(request_client, params, pages)
        else:
            rows, reported_count, error = _text_pages(client, params, pages)
        if error is not None:
            return {"status": "provider_error", "provider_code": error, "candidates": []}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return {"status": "network_error", "candidates": []}
    observed_at = datetime.now(timezone.utc).isoformat()
    candidates, seen = [], set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        poi = normalize_poi(raw)
        if not poi or poi["id"] == "amap:" or poi["id"] in seen:
            continue
        seen.add(poi["id"])
        poi.update({"type": "dining", "category": "dining",
                    "subcategory": _dining_subcategory(raw), "campus": "",
                    "source": "amap_online", "observed_at": observed_at,
                    "business_area": _text(raw.get("adname") or raw.get("address")),
                    "provider_rating": _text((raw.get("biz_ext") or {}).get("rating")),
                    "provider_cost": _text((raw.get("biz_ext") or {}).get("cost")),
                    "verification_status": "provider_only"})
        candidates.append(poi)
    return {"status": "ok", "candidates": candidates[:75],
            "count": len(candidates), "provider_count": max(len(candidates), reported_count),
            "has_more": max(len(candidates), reported_count) > len(candidates),
            "observed_at": observed_at, "query": query}


def _text_pages(client, params, pages):
    first = client.get("https://restapi.amap.com/v3/place/text", params={**params, "page": 1})
    first.raise_for_status()
    body = first.json()
    if str(body.get("status")) != "1":
        return None, int(body.get("count") or 0), str(body.get("infocode") or "")[:16]
    total, rows = int(body.get("count") or 0), list(body.get("pois") or [])
    for page in range(2, pages + 1):
        if len(rows) < 25 or len(rows) >= total:
            break
        response = client.get("https://restapi.amap.com/v3/place/text", params={**params, "page": page})
        response.raise_for_status()
        body = response.json()
        if str(body.get("status")) != "1":
            return None, total, str(body.get("infocode") or "")[:16]
        rows.extend(body.get("pois") or [])
    return rows, total, None


def search_nearby(center, *, keywords=None, radius_m=1000, pages=2, client=None):
    """Search nearby dining from a worker or request without retaining provider state."""
    location = _coord(f"{center.get('lng')},{center.get('lat')}") if isinstance(center, dict) else None
    if location is None:
        return {'status': 'invalid_center', 'candidates': []}
    try:
        radius_m = int(radius_m)
        pages = int(pages)
    except (ValueError, TypeError, OverflowError):
        return {'status': 'invalid_query', 'candidates': []}
    if not 100 <= radius_m <= 5000 or not 1 <= pages <= 3:
        return {'status': 'invalid_query', 'candidates': []}
    key = os.getenv('AMAP_WEB_KEY') or os.getenv('AMAP_KEY')
    if not key:
        return {'status': 'missing_key', 'candidates': []}

    params = {
        'key': key,
        'location': f"{location['lng']},{location['lat']}",
        'radius': radius_m,
        'types': '050000',
        'offset': 25,
        'extensions': 'all',
    }
    if _text(keywords):
        params['keywords'] = _text(keywords)[:80]

    candidates = []
    reported_count = 0
    try:
        if client is None:
            with httpx.Client(timeout=httpx.Timeout(3, connect=2)) as request_client:
                raw_rows, reported_count, provider_error = _nearby_pages(
                    request_client, params, pages,
                )
        else:
            raw_rows, reported_count, provider_error = _nearby_pages(client, params, pages)
        if provider_error is not None:
            return {'status': 'provider_error', 'provider_code': provider_error,
                    'candidates': []}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        # Never include exception text: provider request URLs can contain the key.
        return {'status': 'network_error', 'candidates': []}

    observed_at = datetime.now(timezone.utc).isoformat()
    seen = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        poi = normalize_poi(raw)
        if not poi or poi['id'] == 'amap:' or poi['id'] in seen:
            continue
        try:
            distance = float(raw.get('distance'))
        except (TypeError, ValueError):
            distance = _distance(location, poi['coordinates'])
        if not math.isfinite(distance) or distance < 0 or distance > radius_m:
            continue
        seen.add(poi['id'])
        poi.update({
            'type': 'dining',
            'category': 'dining',
            'subcategory': _dining_subcategory(raw),
            'campus': '',
            'distance_m': round(distance),
            'distance_basis': 'amap_provider_distance_from_search_center',
            'search_center': location,
            'search_radius_m': radius_m,
            'source': 'amap_online',
            'observed_at': observed_at,
        })
        candidates.append(poi)
    candidates.sort(key=lambda row: (row['distance_m'], row['name']))
    return {
        'status': 'ok',
        'candidates': candidates,
        'count': len(candidates),
        'provider_count': max(reported_count, len(candidates)),
        'has_more': max(reported_count, len(candidates)) > len(candidates),
        'observed_at': observed_at,
        'search_center': location,
        'search_radius_m': radius_m,
    }


def lookup_by_id(poi_id, client=None):
    """Refresh one provider POI by its stable AMap ID before routing to it."""
    value = _text(poi_id)
    if value.startswith('amap:'):
        value = value[5:]
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,80}', value):
        return None
    key = os.getenv('AMAP_WEB_KEY') or os.getenv('AMAP_KEY')
    if not key:
        return None
    params = {'key': key, 'id': value, 'extensions': 'all'}
    try:
        if client is None:
            with httpx.Client(timeout=httpx.Timeout(3, connect=2)) as request_client:
                response = request_client.get('https://restapi.amap.com/v3/place/detail',
                                              params=params)
        else:
            response = client.get('https://restapi.amap.com/v3/place/detail', params=params)
        response.raise_for_status()
        body = response.json()
        if str(body.get('status')) != '1':
            return None
        matches = [row for row in body.get('pois') or []
                   if isinstance(row, dict) and _text(row.get('id')) == value]
        return normalize_poi(matches[0]) if len(matches) == 1 else None
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return None


def choose_match(name, rows, known=None):
    labels = {_label(name)}
    brand_labels = {_brand_label(name)}
    if known:
        known_labels = [known.get('name', ''), *known.get('aliases', [])]
        labels.update(_label(x) for x in known_labels)
        brand_labels.update(_brand_label(x) for x in known_labels)
    labels.discard('')
    brand_labels.discard('')
    exact_matches = []
    brand_matches = []
    for row in rows:
        if not row:
            continue
        row_names = [row['name'], *row['aliases']]
        if labels.intersection(_label(x) for x in row_names):
            exact_matches.append(row)
        elif brand_labels.intersection(_brand_label(x) for x in row_names):
            brand_matches.append(row)
    matches = exact_matches or brand_matches
    if known:
        matches_in_range = []
        for row in matches:
            center = known.get('coordinates')
            if not isinstance(center, dict):
                center = {'lng': known.get('lon', known.get('lng')), 'lat': known.get('lat')}
            try:
                center = {'lng': float(center['lng']), 'lat': float(center['lat'])}
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if not all(math.isfinite(center[key]) for key in ('lng', 'lat')):
                return None
            if _distance(center, row['coordinates']) <= 100:
                matches_in_range.append(row)
        matches = matches_in_range
    return matches[0] if len(matches) == 1 else None


def lookup(name, known=None, client=None):
    """One query, no pagination harvesting, disk cache or raw-response logging."""
    key = os.getenv('AMAP_WEB_KEY') or os.getenv('AMAP_KEY')
    if not key or not name:
        return None
    params = {'key': key, 'keywords': name, 'city': '420100', 'citylimit': 'false',
              'extensions': 'all', 'offset': 20, 'page': 1}
    try:
        if client is None:
            with httpx.Client(timeout=httpx.Timeout(3, connect=2)) as request_client:
                response = request_client.get('https://restapi.amap.com/v3/place/text', params=params)
        else:
            response = client.get('https://restapi.amap.com/v3/place/text', params=params)
        response.raise_for_status()
        body = response.json()
        if str(body.get('status')) != '1':
            return None
        rows = []
        for raw in body.get('pois') or []:
            if not isinstance(raw, dict):
                continue
            row = normalize_poi(raw)
            if not row:
                continue
            row.update({'type': 'dining', 'category': 'dining',
                        'subcategory': _dining_subcategory(raw), 'campus': '',
                        'source': 'amap_online',
                        'observed_at': datetime.now(timezone.utc).isoformat()})
            rows.append(row)
        return choose_match(name, rows, known)
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        # HTTP exception strings can contain the key-bearing request URL.
        return None


def online_lookup(name, known=None):
    """Request-scoped lookup; offline algorithms and automated tests stay offline."""
    from flask import current_app, g, has_request_context
    if not has_request_context() or current_app.testing:
        return None
    if os.getenv('AMAP_POI_LOOKUP_ENABLED', 'true').lower() in ('false', '0', 'no'):
        return None
    cache = g.setdefault('_amap_poi_request', {})
    token = (name, known.get('id') if known else None)
    if token not in cache:
        if len(cache) >= 4:
            return None
        cache[token] = lookup(name, known)
    return cache[token]


def navigation_wgs(poi, mode='walk'):
    """Keep destination display coordinates intact; select a separate arrival point."""
    from spatial.coord_transform import gcj02_to_wgs84
    center = poi.get('coordinates') or {'lng': poi['lon'], 'lat': poi['lat']}
    # Gate permissions belong to the graph. Never move a restricted gate to a
    # different provider entrance, potentially bypassing its access control.
    if mode == 'walk' and poi.get('type') not in ('gate', 'area') and poi.get('name'):
        source = online_lookup(poi['name'], poi)
        if source and source.get('navigation_coordinates'):
            center = source['navigation_coordinates']
    return gcj02_to_wgs84(center['lng'], center['lat'])
