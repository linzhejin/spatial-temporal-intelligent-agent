"""Audited running venues and explicitly named choices during a recommendation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from agents import planner
from spatial.poi import _exact_master_poi_matches, load_pois
from tests.test_real_graph_composite_route import offline_graph  # noqa: F401


@pytest.fixture(autouse=True)
def use_legacy_dialogue_for_rule_path_regressions(monkeypatch):
    """These tests cover the explicit legacy fallback, not model-led dialogue."""
    monkeypatch.setenv("AGENT_DIALOGUE_MODE", "legacy")


def test_running_tags_are_evidenced_and_meiyuan_is_a_square():
    pois = {poi["id"]: poi for poi in load_pois()}
    meiyuan, guiyuan = pois["poi_082"], pois["poi_085"]
    assert meiyuan["subcategory"] == "square"
    assert "running" not in meiyuan["activities"]
    assert "用户" in meiyuan["activity_basis"]
    assert set(guiyuan["activities"]) == {"running", "football"}
    for poi_id in ("poi_080", "poi_085", "poi_205", "poi_260"):
        assert "running" in pois[poi_id]["activities"]
        assert "https://" in pois[poi_id]["activity_basis"]
    assert all("https://" in poi["activity_basis"] for poi in pois.values()
               if "running" in poi.get("activities", []))


@pytest.mark.parametrize("alias", ["奥场", "奥林匹克操场", "奥林匹克运动场", "奥林匹克广场"])
def test_aolin_aliases_resolve_to_912_instead_of_another_ground_or_building(alias):
    assert [poi["id"] for poi in _exact_master_poi_matches(alias, load_pois())] == ["poi_080"]


def test_activity_rebuild_preserves_reviewed_corrections():
    from scripts.validate.rebuild_poi_activity_tags import rebuild

    document = json.loads(Path("data/pois.json").read_text(encoding="utf-8"))
    assert rebuild(deepcopy(document)) == document


def test_recommendations_for_running_from_the_library_use_audited_venues(offline_graph):  # noqa: F811
    result = planner.run_agent("我在总图，想找个地方跑步")
    assert result["response_kind"] == "candidates"
    ids = [candidate["poi_id"] for candidate in result["candidates"]]
    assert ids[0] == "poi_085"
    assert "poi_082" not in ids
    assert all("running" in candidate["activities"] for candidate in result["candidates"])
    distances = [candidate["distance_m"] for candidate in result["candidates"]]
    assert distances == sorted(distances)


@pytest.mark.parametrize("query,expected", [
    ("去桂园操场", "poi_085"),
    ("就去桂操", "poi_085"),
    ("去信息学部操场", "poi_260"),
])
def test_named_choice_outside_shown_candidates_keeps_start_and_routes(
        offline_graph, query, expected):  # noqa: F811
    first = planner.run_agent("我在总图，想找个地方跑步")
    pending = deepcopy(first["place_request"])
    pending["candidate_poi_ids"] = [value for value in pending["candidate_poi_ids"] if value != expected]
    pending["candidate_places"] = [value for value in pending["candidate_places"]
                                   if value.get("poi_id") != expected]
    # An old shortlist must not override a new, server-resolved place name.
    pending["result_dependencies"]["candidate_snapshot"]["data_version"] = "old-data"
    chosen = planner.run_agent(query, context={"place_request": pending})
    assert chosen["response_kind"] == "route", chosen.get("message")
    assert chosen["route"]["route_state"]["start"]["name"] == "武汉大学图书馆(总馆)"
    assert chosen["route"]["route_state"]["end"]["poi_id"] == expected
    assert chosen["route"]["route_verification"]["status"] == "pass"


def test_named_square_is_not_silently_claimed_to_support_running(offline_graph):  # noqa: F811
    first = planner.run_agent("我在总图，想找个地方跑步")
    result = planner.run_agent("去梅操", context={"place_request": first["place_request"]})
    assert result["response_kind"] == "clarify"
    assert "跑步" in result["message"]
    assert result.get("route") is None
    assert result.get("candidates") is None


def test_named_choice_can_add_a_route_constraint(offline_graph, monkeypatch):  # noqa: F811
    first = planner.run_agent("我在总图，想找个地方跑步")
    pending = deepcopy(first["place_request"])
    pending["candidate_poi_ids"] = [item for item in pending["candidate_poi_ids"] if item != "poi_085"]
    calls = []
    def unavailable(name, args, _context):
        calls.append((name, args))
        return {"error": "route_unavailable", "message": "路线暂不可用"}, None
    monkeypatch.setattr(planner.agent_tools, "execute_tool", unavailable)
    result = planner.run_agent("去桂园操场，不走台阶", context={"place_request": pending})
    assert result["response_kind"] == "clarify"
    assert calls[0][0] == "plan_route"
    assert calls[0][1]["end"]["poi_id"] == "poi_085"
    assert calls[0][1]["constraints"]["avoid_steps"] is True


def test_unlisted_client_id_is_not_accepted_as_a_text_selection(offline_graph):  # noqa: F811
    first = planner.run_agent("我在总图，想找个地方跑步")
    pending = deepcopy(first["place_request"])
    pending["candidate_poi_ids"] = [item for item in pending["candidate_poi_ids"] if item != "poi_085"]
    result = planner.run_agent("去这个地点", context={"place_request": pending, "selected_poi_id": "poi_085"})
    assert result["response_kind"] == "clarify"
    assert result.get("route") is None


def test_named_choice_and_failed_route_fit_a_full_durable_shortlist(monkeypatch):
    from agents.state_models import PlaceRequestState

    ids = [f"poi_{number:03d}" for number in range(1, 21)]
    pending = planner._place_request_state(
        "我在总图，想找个地方跑步", activities=["running"], subcategories=["field"],
        pending_slot="selection", start={"type": "poi", "name": "总图"},
        candidate_poi_ids=ids,
        candidate_places=[{"poi_id": value, "name": value} for value in ids],
    )
    monkeypatch.setattr(planner.agent_tools, "execute_tool", lambda *_: (
        {"error": "route_unavailable", "message": "路线暂不可用"}, None,
    ))
    result = planner.run_agent("去桂园操场", context={"place_request": pending})
    saved = PlaceRequestState.model_validate(result["place_request"])
    assert saved.selected_poi_id == "poi_085"
    assert len(saved.candidate_poi_ids) == 20
    assert all(item["poi_id"] in saved.candidate_poi_ids for item in saved.candidate_places)


def test_online_listed_name_keeps_its_online_identity_when_a_local_name_matches(monkeypatch):
    online = {"id": "amap:B012345", "poi_id": "amap:B012345", "name": "湖边咖啡"}
    local = {"id": "poi_900", "name": "湖边咖啡", "aliases": [], "activities": ["coffee"]}
    pending = planner._place_request_state(
        "我在总图，想喝咖啡", activities=["coffee"], subcategories=["coffee"],
        pending_slot="selection", start={"type": "poi", "name": "总图"},
        candidate_poi_ids=[online["id"]], candidate_places=[online],
    )
    monkeypatch.setattr(planner.agent_tools, "load_pois", lambda: [local])
    monkeypatch.setattr(planner.agent_tools.amap_poi, "lookup_by_id", lambda _id: online)
    monkeypatch.setattr(planner.agent_tools.amap_poi, "choose_match", lambda *_a, **_kw: online)
    monkeypatch.setattr("agents.result_maintenance.candidate_snapshot_is_current", lambda *_: True)
    calls = []
    def unavailable(name, args, _context):
        calls.append((name, args))
        return {"error": "route_unavailable", "message": "路线暂不可用"}, None
    monkeypatch.setattr(planner.agent_tools, "execute_tool", unavailable)
    planner.run_agent("去湖边咖啡", context={"place_request": pending})
    assert calls[0][1]["end"]["poi_id"] == online["id"]


def test_unknown_named_choice_does_not_repeat_previous_recommendations(offline_graph):  # noqa: F811
    first = planner.run_agent("我在总图，想找个地方跑步")
    result = planner.run_agent("去不存在的跑步新场地", context={"place_request": first["place_request"]})
    assert result["response_kind"] == "clarify"
    assert result.get("route") is None
    assert result.get("candidates") is None
    assert "无法确认" in result["message"]
