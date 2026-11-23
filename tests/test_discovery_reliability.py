"""Regression coverage for scoped nearest routes and complete place searches."""

from types import SimpleNamespace

import pytest

from agents import discovery, tools
from agents.dialogue import Dialogue, tool_schemas
from agents.dialogue_loop import _bounded_empty_reply, _failure, _has_verified_empty_search
from scripts.tag_subcategory import tag_poi
from spatial import amap_poi
from spatial.poi_taxonomy import ACTIVITY_SUBCATEGORIES, matches_activities, resolve_activity_query
from spatial.poi import load_pois


def _dialogue(query, execute_tool, *, context=None, coord_start=None):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: None)))
    return Dialogue(query, context=context, coord_start=coord_start, execute_tool=execute_tool,
                    client_factory=lambda: client, trace=lambda _item: None)


def test_nearest_route_contract_is_exposed_as_a_model_selected_tool():
    schemas = {item["function"]["name"]: item["function"] for item in tool_schemas()}

    assert schemas["update_task"]["parameters"]["properties"]["destination_selection"]["enum"] == [
        "nearest_route",
    ]
    assert "route_to_nearest_place" in schemas


def test_nearest_route_searches_first_then_routes_to_the_nearest_verified_candidate(monkeypatch):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    candidates = [
        {"poi_id": "poi_068", "id": "poi_068", "name": "桂园一食堂", "network_distance_m": 410,
         "coordinates": {"lng": 114.366, "lat": 30.537}},
        {"poi_id": "poi_076", "id": "poi_076", "name": "老樱园食堂", "network_distance_m": 220,
         "coordinates": {"lng": 114.3655, "lat": 30.5367}},
    ]
    calls = []
    route = {"route_verification": {"status": "pass"}, "route_state": {"start": start}}

    def execute(name, args, context):
        calls.append((name, args, context))
        if name == "search_poi_candidates":
            return ({"candidates": candidates, "count": 2, "has_more": False,
                     "online_status": "not_requested", "data_version": "pois-v1",
                     "road_condition_version": "roads-v1"}, {"candidates": candidates})
        if name == "plan_route":
            return ({"status": "ok"}, {"route": route, "route_kind": "direct"})
        raise AssertionError(f"unexpected tool: {name}")

    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({"_route_graph_snapshot": {}}, None))
    dialogue = _dialogue("从老斋舍去最近的食堂", execute, coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    result = dialogue.spatial("route_to_nearest_place", {})

    assert [call[0] for call in calls] == ["search_poi_candidates", "plan_route"]
    assert calls[0][2]["_distance_first"] is True
    assert calls[1][1]["end"]["poi_id"] == "poi_076"
    assert dialogue.route is route
    assert not result.get("error")


def test_nearest_route_tries_next_reachable_candidate_if_first_route_fails(monkeypatch):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    candidates = [
        {"poi_id": "poi_076", "id": "poi_076", "name": "老樱园食堂", "network_distance_m": 220},
        {"poi_id": "poi_068", "id": "poi_068", "name": "桂园一食堂", "network_distance_m": 280},
    ]
    routed = []

    def execute(name, args, _context):
        if name == "search_poi_candidates":
            return ({"candidates": candidates, "count": 2, "has_more": False,
                     "online_status": "not_requested"}, {"candidates": candidates})
        if name == "plan_route":
            routed.append(args["end"]["poi_id"])
            if len(routed) == 1:
                return ({"error": "route_not_found", "reason_code": "graph_no_path", "message": "无路网连接"}, None)
            return ({"status": "ok"}, {"route": {"route_verification": {"status": "pass"},
                                               "route_state": {"start": start}}})
        raise AssertionError(f"unexpected tool: {name}")

    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue = _dialogue("从老斋舍去最近的食堂", execute, coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    result = dialogue.spatial("route_to_nearest_place", {})

    assert routed == ["poi_076", "poi_068"]
    assert dialogue.pending["selected_poi_id"] == "poi_068"
    assert not result.get("error")


def test_nearest_route_stops_if_closest_candidate_route_cannot_be_verified(monkeypatch):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    candidates = [
        {"poi_id": "poi_076", "name": "老樱园食堂", "network_distance_m": 220},
        {"poi_id": "poi_068", "name": "桂园一食堂", "network_distance_m": 280},
    ]
    routed = []

    def execute(name, args, _context):
        if name == "search_poi_candidates":
            return ({"candidates": candidates, "count": 2, "online_status": "not_requested"},
                    {"candidates": candidates})
        routed.append(args["end"]["poi_id"])
        if len(routed) == 1:
            return ({"status": "ok"}, {"route": {"route_verification": {"status": "fail"}}})
        return ({"status": "ok"}, {"route": {"route_verification": {"status": "pass"}}})

    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue = _dialogue("从老斋舍去最近的食堂", execute, coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    result = dialogue.spatial("route_to_nearest_place", {})

    assert routed == ["poi_076"]
    assert result["status"] == "route_unavailable"
    assert dialogue.route is None


def test_failed_nearest_route_does_not_save_an_unrouted_candidate(monkeypatch):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    candidates = [{"poi_id": "poi_076", "name": "老樱园食堂", "network_distance_m": 220}]

    def execute(name, _args, _context):
        if name == "search_poi_candidates":
            return ({"candidates": candidates, "count": 1, "online_status": "not_requested"},
                    {"candidates": candidates})
        return ({"error": "graph_no_path", "message": "无路网连接"}, None)

    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue = _dialogue("从老斋舍去最近的食堂", execute, coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    result = dialogue.spatial("route_to_nearest_place", {})

    assert result["status"] == "route_unavailable"
    assert "selected_poi_id" not in dialogue.pending
    assert "selected_place" not in dialogue.pending
    assert "place_request.selected_poi_id" not in dialogue.modified_fields


@pytest.mark.parametrize("tool_name", ["plan_route", "plan_itinerary"])
def test_nearest_route_task_cannot_bypass_server_candidate_search(monkeypatch, tool_name):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    calls = []
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue = _dialogue("从老斋舍去最近的食堂", lambda name, args, context: (
        calls.append((name, args, context)) or ({"status": "ok"}, {})
    ), coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    result = dialogue.spatial(tool_name, {})

    assert result["error"] == "nearest_search_required"
    assert calls == []


def test_nearest_route_does_not_claim_nearest_from_incomplete_provider_results(monkeypatch):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    route_calls = []
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    def execute(name, _args, _context):
        if name == "search_poi_candidates":
            rows = [{"poi_id": "poi_076", "name": "老樱园食堂", "network_distance_m": 220}]
            return ({"candidates": rows, "count": 1, "online_status": "incomplete",
                     "online_has_more": True}, {"candidates": rows})
        route_calls.append(name)
        return ({"status": "ok"}, {})

    dialogue = _dialogue("从老斋舍去最近的食堂", execute, coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    dialogue.spatial("route_to_nearest_place", {})

    assert route_calls == []
    assert dialogue.nearest_route_outcome["status"] == "incomplete"


def test_nearest_route_requires_network_distance_evidence(monkeypatch):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    route_calls = []
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    def execute(name, _args, _context):
        if name == "search_poi_candidates":
            rows = [{"poi_id": "poi_076", "name": "老樱园食堂", "network_distance_m": 80},
                    {"poi_id": "poi_068", "name": "桂园一食堂", "distance_m": 40}]
            return ({"candidates": rows, "count": 2, "online_status": "not_requested"},
                    {"candidates": rows})
        route_calls.append(name)
        return ({"status": "ok"}, {})

    dialogue = _dialogue("从老斋舍去最近的食堂", execute, coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    dialogue.spatial("route_to_nearest_place", {})

    assert route_calls == []
    assert dialogue.nearest_route_outcome["status"] == "incomplete"


def test_nearest_route_records_temporary_provider_failure_as_retryable(monkeypatch):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue = _dialogue("从老斋舍去最近的食堂", lambda *_args: (
        {"candidates": [], "count": 0, "online_status": "network_error"}, {"candidates": []}
    ), coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["canteen_meal"], "subcategories": ["canteen"],
                     "constraints": {}})

    result = dialogue.spatial("route_to_nearest_place", {})

    assert result["error"] == "place_service_unavailable"
    assert dialogue.nearest_route_outcome["failure_code"] == "place_service_unavailable"


@pytest.mark.parametrize("provider_status", ["network_error", "provider_error", "missing_key"])
def test_nearest_route_does_not_claim_nearest_when_provider_failed_with_local_candidates(monkeypatch, provider_status):
    start = {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365}
    routes = []
    rows = [{"poi_id": "poi_231", "name": "珞珈自强超市", "network_distance_m": 220}]

    def execute(name, _args, _context):
        if name == "search_poi_candidates":
            return ({"candidates": rows, "count": 1, "online_status": provider_status},
                    {"candidates": rows})
        routes.append(name)
        return ({"status": "ok"}, {"route": {"route_verification": {"status": "pass"}}})

    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))
    dialogue = _dialogue("从老斋舍去最近的超市", execute, coord_start=start)
    dialogue.update({"relation": "new", "operation": "route", "destination_selection": "nearest_route",
                     "start": start, "activities": ["shopping"], "subcategories": ["supermarket"],
                     "constraints": {}})

    result = dialogue.spatial("route_to_nearest_place", {})

    assert routes == []
    assert result["error"] == "place_service_unavailable"
    assert dialogue.nearest_route_outcome["failure_code"] == "place_service_unavailable"


@pytest.mark.parametrize(
    ("phrase", "activity", "subcategory"),
    [("附近卫生间", "restroom", "toilet"),
     ("附近便利店", "convenience_store", "convenience_store"),
     ("附近集贸市场", "market", "market")],
)
def test_facility_and_retail_queries_resolve_as_distinct_provider_categories(phrase, activity, subcategory):
    intent = resolve_activity_query(phrase)

    assert intent["activities"] == [activity]
    assert intent["subcategories"] == [subcategory]
    assert subcategory in ACTIVITY_SUBCATEGORIES[activity]


def test_supermarket_activity_does_not_include_a_market():
    assert ACTIVITY_SUBCATEGORIES["shopping"] == ("supermarket",)
    assert not matches_activities({"subcategory": "market", "activities": ["market"]}, ["shopping"])


def test_retagging_distinguishes_market_supermarket_and_food_stall_inside_canteen():
    assert tag_poi({"name": "武汉大学工学部集贸市场", "type": "service"}) == ("market", False, True)
    assert tag_poi({"name": "珞珈自强超市", "type": "service"}) == ("supermarket", False, True)
    assert tag_poi({"name": "匠心卤(枫园食堂店)", "type": "dining"}) == ("fastfood", True, True)


def test_catalog_separates_engineering_market_from_supermarket_and_canteen():
    pois = {row["id"]: row for row in load_pois()}

    assert pois["poi_415"]["subcategory"] == "market"
    assert pois["poi_231"]["subcategory"] == "supermarket"
    assert pois["poi_361"]["subcategory"] == "fastfood"
    assert "shopping" not in pois["poi_415"]["activities"]


@pytest.mark.parametrize(
    ("subcategory", "provider_subcategory", "activity", "keyword"),
    [("toilet", "toilet", "restroom", "卫生间"),
     ("restroom", "toilet", "restroom", "卫生间"),
     (None, "toilet", "restroom", "卫生间"),
     ("supermarket", "supermarket", "shopping", "超市")],
)
def test_structured_amenity_search_uses_unfiltered_nearby_provider(subcategory, provider_subcategory, activity, keyword, monkeypatch):
    candidate = {"poi_id": "amap:PLACE1", "id": "amap:PLACE1", "name": keyword,
                 "type_name": keyword, "subcategory": provider_subcategory, "source": "amap_online",
                 "coordinate_system": "GCJ-02", "coordinates": {"lng": 114.365, "lat": 30.536}}
    calls = []
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "_endpoint_target_wgs", lambda *_args: (114.365, 30.536, "老斋舍", None))
    monkeypatch.setattr(discovery, "wgs84_to_gcj02", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(amap_poi, "search_nearby", lambda center, **kwargs: (
        calls.append((center, kwargs)) or {"status": "ok", "candidates": [candidate], "has_more": False}
    ))
    monkeypatch.setattr(tools, "_tool_find_reachable_places", lambda args, context: (
        {"start_name": "老斋舍", "candidates": [candidate], "count": 1, "total_count": 1,
         "has_more": False}, {"candidates": [candidate]}
    ))

    result, _artifact = discovery.search_places_for_dialogue(
        {"start": {"type": "poi", "poi_id": "poi_161", "name": "老斋舍"},
         "subcategories": [subcategory] if subcategory else [], "activities": [activity], "max_distance_m": 1000},
        {"_semantic_request": {"keywords": [keyword]}},
    )

    assert result["count"] == 1
    assert calls[0][1]["poi_type"] == "all"
    assert calls[0][1]["keywords"] == keyword


def test_task_normalizes_restroom_activity_used_as_a_subcategory():
    dialogue = _dialogue("老斋舍附近的卫生间", lambda *_args: ({}, {}))

    dialogue.update({"relation": "new", "operation": "place_search", "constraints": {},
                     "activities": ["restroom"], "subcategories": ["restroom"]})

    assert dialogue.pending["subcategories"] == ["toilet"]


def test_incomplete_provider_pages_are_not_marked_as_empty_or_complete(monkeypatch):
    from agents.dialogue_loop import _has_verified_empty_search

    pending = {"start": {"type": "coord", "name": "老斋舍", "lng": 114.3653, "lat": 30.5365},
               "search_radius_m": 1000, "semantic_request": {"operation": "place_search"}}
    dialogue = _dialogue("老斋舍附近找卫生间", lambda *_args: (
        {"count": 0, "candidates": [], "online_status": "incomplete", "online_has_more": True}, None
    ), context={"place_request": pending}, coord_start=pending["start"])
    monkeypatch.setattr(tools, "_prepare_route_context", lambda _context: ({}, None))

    dialogue.spatial("search_poi_candidates", {"subcategories": ["toilet"], "max_distance_m": 1000})

    assert not dialogue._map_search_completed
    assert dialogue.place_service_error is None
    assert not dialogue._empty_search_keys
    assert not _has_verified_empty_search(dialogue)
    assert "未完成" in _bounded_empty_reply(dialogue, "当前附近没有卫生间。")


def test_empty_incomplete_amenity_search_does_not_call_provider_failure(monkeypatch):
    monkeypatch.setattr(tools, "_search_poi_categories", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tools, "_endpoint_target_wgs", lambda *_args: (114.365, 30.536, "老斋舍", None))
    monkeypatch.setattr(discovery, "wgs84_to_gcj02", lambda lng, lat: (lng, lat))
    monkeypatch.setattr(amap_poi, "search_nearby", lambda *_args, **_kwargs: {
        "status": "ok", "candidates": [], "has_more": True,
    })

    result, _artifact = discovery.search_places_for_dialogue(
        {"start": {"type": "poi", "poi_id": "poi_161", "name": "老斋舍"},
         "subcategories": ["toilet"], "activities": ["restroom"], "max_distance_m": 1000},
        {"_semantic_request": {"keywords": ["卫生间"]}},
    )

    assert result["online_status"] == "incomplete"
    assert result["online_has_more"] is True
    assert "尚未完成" in result["message"]
    assert "不可用" not in result["message"]


@pytest.mark.parametrize(
    ("code", "retryable"),
    [("deadline_exceeded", True), ("place_service_unavailable", True),
     ("invalid_tool_request", False), ("provider_internal", False)],
)
def test_retry_action_is_limited_to_known_transient_failure_codes(code, retryable):
    class FailedDialogue:
        def result(self, message, *, failed=False):
            assert failed is True
            return {"message": message, "execution_status": "failed"}

    result = _failure(FailedDialogue(), "暂时失败", code)

    assert result["retryable"] is retryable
