"""Model-directed place retrieval over reviewed campus data and live AMap POIs."""
from __future__ import annotations

from copy import deepcopy
import re

import config
from spatial import amap_poi
from spatial.coord_transform import wgs84_to_gcj02
from spatial.poi import filter_reported_absent_provider_results, has_source_backed_record


def _within_campus(poi):
    coords = poi.get("coordinates") or {}
    try:
        lng, lat = float(coords.get("lng", poi.get("lon", poi.get("lng")))), float(coords.get("lat", poi.get("lat")))
        return config.WHU_BBOX["west"] <= lng <= config.WHU_BBOX["east"] and config.WHU_BBOX["south"] <= lat <= config.WHU_BBOX["north"]
    except (TypeError, ValueError, KeyError):
        return False


def _inside_requested_area(poi, area):
    area = re.sub(r"[\s\-—_()（）·]", "", str(area or "")).casefold()
    if not area or area in {"武汉大学", "武大", "校园", "校内"}:
        return True
    text = " ".join(str(poi.get(key) or "") for key in ("name", "description", "address", "campus", "business_area", "aliases"))
    normalized = re.sub(r"[\s\-—_()（）·]", "", text).casefold()
    return area in normalized


def _matches_exclusions(poi, terms):
    text = " ".join(str(poi.get(key) or "") for key in ("name", "description", "address", "subcategory", "type_name", "aliases"))
    normalized = re.sub(r"\s+", "", text).casefold()
    return any(str(term or "").strip() and str(term).strip().casefold() in normalized for term in (terms or []))


_GENERIC_DINING_TERMS = {"吃饭", "干饭", "用餐", "餐饮", "餐厅", "食堂", "快餐", "美食", "饮食", "吃点东西"}


def _is_generic_meal_request(keyword, keywords, activities):
    activity_terms = {str(item or "").strip().casefold() for item in activities or []}
    if activity_terms & {"eat", "eating", "food", "meal", "dining"}:
        return True
    def normalize(value):
        return re.sub(r"[\s，,、。!！?？·]+", "", str(value or "")).casefold()

    terms = [normalize(item) for item in (keywords or []) if normalize(item)]
    terms.extend(normalize(item) for item in re.split(r"[\s，,、。!！?？·]+", str(keyword or ""))
                 if normalize(item))
    return bool(terms) and set(terms) <= _GENERIC_DINING_TERMS


def _provider_keyword(keyword, keywords, activities, candidate_names):
    normalized = re.sub(r"[\s，,、。!！?？·]+", "", str(keyword or "")).casefold()
    generic_meal = _is_generic_meal_request(keyword, keywords, activities)
    if candidate_names or generic_meal:
        return "餐饮"
    return str(keyword or "").strip()


def search_places_for_dialogue(args: dict, ctx: dict):
    """Search all requested providers; route-distance calculation stays in tools.py."""
    from agents import tools

    query_state = ctx.get("_semantic_request") or {}
    keyword = str(args.get("keyword") or " ".join(query_state.get("keywords") or "")).strip()[:80]
    area = str(args.get("area") or query_state.get("area") or "武汉大学")[:80]
    subcategories = args.get("subcategories") or ctx.get("_semantic_request", {}).get("subcategories") or []
    activities = args.get("activities") or ctx.get("_semantic_request", {}).get("activities") or []
    exclusions = args.get("exclude_terms") or query_state.get("exclude_terms") or []
    raw_candidate_names = args.get("candidate_names") or []
    if not isinstance(raw_candidate_names, list):
        return {"error": "invalid_candidate_names", "message": "地图核对名称必须是列表。"}, None
    candidate_names = [name.strip()[:160] for name in raw_candidate_names
                       if isinstance(name, str) and name.strip()][:5]
    anchor = args.get("start")
    generic_areas = {"武汉大学", "武大", "校园", "校内"}
    search_center = args.get("search_anchor") or (anchor if area in generic_areas else None)
    # A nearby search is ranked against the place the user asked about; the
    # actual trip origin remains separate for a later route request.
    distance_origin = search_center or anchor
    local = tools._search_poi_categories(subcategories or None, args.get("poi_type"), keyword,
                                         args.get("season"), True, activities=activities)
    local = [deepcopy(item) for item in local if has_source_backed_record(item)
             and item.get("subcategory") not in tools._ONLINE_DINING_SUBCATEGORIES
             and _inside_requested_area(item, area) and not _matches_exclusions(item, exclusions)]

    online_status = "not_requested"
    online = []
    search_center_gcj = None
    generic_meal = _is_generic_meal_request(keyword, query_state.get("keywords"), activities)
    wants_online = bool(candidate_names) or generic_meal or bool(
        set(subcategories or []) & tools._ONLINE_DINING_SUBCATEGORIES) or (
        not subcategories and bool(keyword))
    map_keyword = _provider_keyword(keyword, query_state.get("keywords"), activities, candidate_names)
    provider_categories = set(subcategories or [])
    if generic_meal:
        provider_categories.update({"restaurant", "fastfood"})
    if wants_online:
        if not map_keyword:
            online_status = "query_required"
        elif search_center:
            from agents.tools import _endpoint_target_wgs
            lng, lat, _, error = _endpoint_target_wgs(search_center, args.get("mode") or "walk")
            if error or lng is None:
                online_status = "center_unavailable"
            else:
                lng, lat = wgs84_to_gcj02(float(lng), float(lat))
                search_center_gcj = {"lng": lng, "lat": lat}
                response = amap_poi.search_nearby({"lng": lng, "lat": lat}, keywords=map_keyword,
                                                  radius_m=min(5000, max(100, int(args.get("max_distance_m") or 1000))),
                                                  pages=3)
                online_status = response.get("status", "provider_error")
                online = list(response.get("candidates") or [])
        else:
            # Keep the venue type as the provider query and apply the requested
            # campus area separately; combining both can overconstrain retrieval.
            response = amap_poi.search_text(map_keyword, city="420100")
            online_status = response.get("status", "provider_error")
            online = list(response.get("candidates") or [])
            if area in {"武汉大学", "武大", "校园", "校内"}:
                online = [item for item in online if _within_campus(item)]
        if not search_center:
            online = [item for item in online if _inside_requested_area(item, area)]
        elif area in generic_areas:
            online = [item for item in online if _within_campus(item)]
        online = [item for item in online if not _matches_exclusions(item, exclusions)]
        online = filter_reported_absent_provider_results(online)
        if provider_categories:
            online = [item for item in online if item.get("subcategory") in provider_categories]

    resolved_online = []
    for name in candidate_names:
        candidate = amap_poi.lookup(name)
        if not candidate and search_center_gcj:
            response = amap_poi.search_nearby(
                search_center_gcj, keywords=name,
                radius_m=min(5000, max(100, int(args.get("max_distance_m") or 1000))), pages=2,
            )
            candidate = amap_poi.choose_match(name, response.get("candidates") or [])
        if not candidate:
            continue
        candidate = deepcopy(candidate)
        identifier = candidate.get("poi_id") or candidate.get("id")
        if not identifier or _matches_exclusions(candidate, exclusions):
            continue
        if not search_center and area in generic_areas and not _within_campus(candidate):
            continue
        if not search_center and area not in generic_areas and not _inside_requested_area(candidate, area):
            continue
        candidate.update({"poi_id": identifier, "source": "amap_online"})
        resolved_online.append(candidate)
    resolved_online = filter_reported_absent_provider_results(resolved_online)
    if provider_categories:
        resolved_online = [item for item in resolved_online
                           if item.get("subcategory") in provider_categories]
    if resolved_online:
        online_status = "ok"

    by_id = {}
    for poi in [*local, *online, *resolved_online]:
        identifier = poi.get("id") or poi.get("poi_id")
        if identifier:
            poi["poi_id"] = identifier
            by_id[identifier] = poi
    pool = list(by_id.values())
    if not pool:
        return {"status": "ok" if online_status in {"ok", "not_requested"} else online_status,
                "candidates": [], "count": 0, "online_status": online_status,
                "message": ("没有检索到匹配的商户或校园地点；这不代表现实中不存在。"
                            if online_status in {"ok", "not_requested"} else "地点服务暂时不可用，请换关键词或稍后重试。")}, {"candidates": []}

    # The same accepted route tool computes path distance and applies closure,
    # mode and accessibility filters to this exact server-fetched pool.
    if distance_origin:
        distance_args = {"start": distance_origin, "mode": args.get("mode") or "walk",
                         "subcategories": list(subcategories or []), "activities": list(activities or []),
                         "max_distance_m": min(10000, max(100, int(args.get("max_distance_m") or
                                                                      (1000 if search_center else 10000)))),
                         "limit": 20, "constraints": args.get("constraints") or {}}
        route_context = {key: deepcopy(ctx[key]) for key in (
            "_route_graph_snapshot", "_route_conditions_snapshot", "_route_weather_snapshot",
        ) if key in ctx}
        route_context.update({"query": "", "_candidate_pool": pool, "_model_dialogue": True,
                              "_excluded_candidate_ids": list(query_state.get("shown_ids") or [])})
        result, artifact = tools._tool_find_reachable_places(distance_args, route_context)
        if result.get("error"):
            return result, None
        shown_ids = set(query_state.get("shown_ids") or [])
        candidates = [item for item in result.get("candidates") or []
                      if item.get("poi_id") not in shown_ids]
        for item in candidates:
            item["distance_reference_name"] = (result.get("start_name")
                or (distance_origin.get("name") if isinstance(distance_origin, dict) else None)
                or "搜索中心")
        result.update({"candidates": candidates, "count": len(candidates),
                       "online_status": online_status, "search_area": area,
                       "search_keyword": map_keyword})
    else:
        candidates = []
        shown_ids = set(query_state.get("shown_ids") or [])
        for poi in pool:
            if (poi.get("poi_id") or poi.get("id")) in shown_ids:
                continue
            item = tools._poi_public(poi, with_coords=True, with_provenance=True)
            item["poi_id"] = poi.get("poi_id") or poi.get("id")
            candidates.append(item)
        candidates = candidates[:20]
        result = {"status": "ok", "candidates": candidates, "count": len(candidates),
                  "online_status": online_status, "search_area": area, "search_keyword": map_keyword}
        artifact = {"candidates": candidates}
    return result, artifact
