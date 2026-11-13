"""Model-directed place retrieval over reviewed campus data and live AMap POIs."""
from __future__ import annotations

from copy import deepcopy
import re

import config
from spatial import amap_poi
from spatial.coord_transform import wgs84_to_gcj02
from spatial.poi import filter_reported_absent_provider_results, has_source_backed_record, same_display_place
from spatial.poi_taxonomy import ACTIVITY_LABELS, ACTIVITY_SUBCATEGORIES, activities_for_poi, resolve_activity_query


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
    generic_meal = _is_generic_meal_request(keyword, keywords, activities)
    if generic_meal:
        return "餐饮"
    return str(keyword or "").strip() or (candidate_names[0] if candidate_names else "")


def _provider_supports_activities(poi, activities):
    """Require provider type evidence before using an online POI for an activity."""
    requested = {str(item) for item in activities or [] if item}
    if not requested:
        return True
    if requested & {"eat", "eating", "food", "dining"}:
        requested.add("meal")
    supported = activities_for_poi(poi)
    subcategory = poi.get("subcategory")
    # Dining subcategories are derived from the provider's specific type; a
    # shared sports category such as "court" is not proof of a particular game.
    for activity in ("meal", "canteen_meal", "restaurant_meal", "fast_food", "coffee", "tea_drink"):
        if subcategory in ACTIVITY_SUBCATEGORIES[activity]:
            supported.add(activity)
    type_name = str(poi.get("type_name") or "")
    type_code = str(poi.get("type_code") or "")
    if type_name and "餐饮" not in type_name and not type_code.startswith("05"):
        evidence = resolve_activity_query(type_name) or {}
        supported.update(evidence.get("activities") or [])
        if "体育" in type_name or "运动场馆" in type_name:
            evidence = resolve_activity_query(str(poi.get("name") or "")) or {}
            supported.update(evidence.get("activities") or [])
    return bool(supported & requested)


def search_places_for_dialogue(args: dict, ctx: dict):
    """Search all requested providers; route-distance calculation stays in tools.py."""
    from agents import tools

    constraints = args.get("constraints") or {}
    if not isinstance(constraints, dict):
        return {"error": "invalid_constraints", "message": "通行约束格式不正确。"}, None
    try:
        origin_duration = tools.positive_budget(constraints, "max_duration_s")
        origin_distance = tools.positive_budget(constraints, "max_distance_m")
    except ValueError:
        return {"error": "invalid_constraints", "message": "路线距离或时间上限无效。"}, None
    if (origin_duration is not None or origin_distance is not None) and not args.get("start"):
        return {"error": "origin_required_for_route_budget",
                "message": "请先明确实际出发点，才能核验从出发点到候选地点的时间或距离。"}, None
    query_state = ctx.get("_semantic_request") or {}
    keyword = str(args.get("keyword") or " ".join(query_state.get("keywords") or "")).strip()[:80]
    area = str(args.get("area") or query_state.get("area") or "武汉大学")[:80]
    subcategories = (args.get("subcategories") or ([args["subcategory"]] if args.get("subcategory") else None)
                     or ctx.get("_semantic_request", {}).get("subcategories") or [])
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
    sports = [item for item in activities if item in {
        "basketball", "badminton", "table_tennis", "football", "volleyball", "tennis",
        "running", "fitness", "swimming"}]
    wants_online = bool(candidate_names) or generic_meal or bool(sports) or bool(
        set(subcategories or []) & tools._ONLINE_DINING_SUBCATEGORIES) or (
        not subcategories and bool(keyword))
    map_keyword = _provider_keyword(keyword, query_state.get("keywords"), activities, candidate_names)
    if not map_keyword and sports:
        map_keyword = ACTIVITY_LABELS[sports[0]]
    if not map_keyword and set(subcategories) & tools._ONLINE_DINING_SUBCATEGORIES:
        # A structured category is enough to query the provider; no sentence
        # classification or fabricated candidate name is needed.
        map_keyword = "餐饮"
    provider_categories = set(subcategories or [])
    if generic_meal:
        provider_categories.update({"canteen", "restaurant", "fastfood"})
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
                                                  pages=3, **({"poi_type": "sports"} if sports else {}))
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
        online = [item for item in online if _provider_supports_activities(item, activities)]
        if provider_categories:
            online = [item for item in online if item.get("subcategory") in provider_categories]

    resolved_online = []
    for name in candidate_names:
        candidate = amap_poi.lookup(name)
        if not candidate and search_center_gcj:
            response = amap_poi.search_nearby(
                search_center_gcj, keywords=name,
                radius_m=min(5000, max(100, int(args.get("max_distance_m") or 1000))), pages=2,
                **({"poi_type": "sports"} if sports else {}),
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
    resolved_online = [item for item in resolved_online if _provider_supports_activities(item, activities)]
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
            if identifier not in by_id and any(same_display_place(poi, existing) for existing in by_id.values()):
                continue
            by_id[identifier] = poi
    pool = list(by_id.values())
    if not pool:
        return {"status": "ok" if online_status in {"ok", "not_requested"} else online_status,
                "candidates": [], "count": 0, "online_status": online_status,
                "message": ("没有检索到匹配的商户或校园地点；这不代表现实中不存在。"
                            if online_status in {"ok", "not_requested"} else
                            "缺少检索词，请提供具体地点类型或关键词。" if online_status == "query_required" else
                            "搜索中心未能解析，请先核对中心地点。" if online_status == "center_unavailable" else
                            "地点服务暂时不可用，请稍后重试。")}, {"candidates": []}

    # The same accepted route tool computes path distance and applies closure,
    # mode and accessibility filters to this exact server-fetched pool.
    if distance_origin:
        from agents.dialogue import _same_accepted_location
        separate_origin = bool(search_center and anchor and
                               not _same_accepted_location(search_center, anchor, tools))
        scope_constraints = dict(constraints)
        if separate_origin:
            scope_constraints.pop("max_duration_s", None)
            scope_constraints.pop("max_distance_m", None)
        distance_args = {"start": distance_origin, "mode": args.get("mode") or "walk",
                         "subcategories": list(subcategories or []), "activities": list(activities or []),
                         "max_distance_m": min(10000, max(100, int(args.get("max_distance_m") or
                                                                      (1000 if search_center else 10000)))),
                         "limit": 20, "constraints": scope_constraints}
        route_context = {key: deepcopy(ctx[key]) for key in (
            "_route_graph_snapshot", "_route_conditions_snapshot", "_route_weather_snapshot",
        ) if key in ctx}
        route_context.update({"query": "", "_candidate_pool": pool, "_model_dialogue": True,
                              "_excluded_candidate_ids": list(query_state.get("shown_ids") or [])})
        if origin_duration is not None and not separate_origin:
            route_context["_search_scope_duration_s"] = origin_duration
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
        if separate_origin and (origin_duration is not None or origin_distance is not None) and candidates:
            scope_rows = {row.get("poi_id"): row for row in candidates}
            source_rows = [row for row in pool if (row.get("poi_id") or row.get("id")) in scope_rows]
            origin_args = {"start": anchor, "mode": args.get("mode") or "walk",
                           "subcategories": list(subcategories or []), "activities": list(activities or []),
                           "limit": 20, "constraints": constraints}
            if origin_duration is not None:
                if origin_duration > 180 * 60:
                    return {"error": "invalid_reachability_budget", "message": "时间上限超过可核验范围。"}, None
                origin_args["max_time_min"] = max(1.0, origin_duration / 60.0)
            else:
                if origin_distance > 10000:
                    return {"error": "invalid_reachability_budget", "message": "路线距离上限超过可核验范围。"}, None
                origin_args["max_distance_m"] = origin_distance
            origin_context = {**route_context, "_candidate_pool": source_rows}
            if origin_duration is not None:
                origin_context["_search_scope_duration_s"] = origin_duration
            from_origin, _ = tools._tool_find_reachable_places(origin_args, origin_context)
            if from_origin.get("error"):
                return from_origin, None
            reachable = []
            for row in from_origin.get("candidates") or []:
                scoped = scope_rows.get(row.get("poi_id"))
                if scoped:
                    reachable.append({**scoped, "origin_distance_m": row.get("network_distance_m"),
                                      "estimated_duration_min": row.get("estimated_duration_min"),
                                      "duration_reference_name": from_origin.get("start_name")})
            if not reachable and result.get("has_more"):
                return {"error": "reachability_incomplete",
                        "message": "候选较多，暂无法完整核验从出发点可达的地点，请缩小范围。"}, None
            candidates = reachable
            result["total_count"] = len(candidates) if not result.get("has_more") else None
            artifact = {"candidates": candidates}
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
