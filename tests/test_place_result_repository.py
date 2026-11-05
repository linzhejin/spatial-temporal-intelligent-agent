"""PostgreSQL integration checks for completed-trip continuity."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from storage import database
from storage.task_repository import (
    TaskContinuationConflict, cancel_run, claim_next_run, create_conversation,
    create_task_run, finish_run, get_run, get_run_input,
)


DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def _submit(owner, conversation_id, text):
    return create_task_run(DATABASE_URL, owner, conversation_id, text, str(uuid4()),
                           datetime.now(timezone.utc) + timedelta(minutes=2))


def _current_route_environment(monkeypatch):
    from agents import route_state, tools
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-test")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    return route_state.current_road_condition_version([])


def test_only_the_latest_owned_completed_delivery_supplies_activity_context():
    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    trip = {"request_id": "trip", "pending_slot": "selection", "completed_legs": [{"edge": [1, 2, 0]}]}
    first = _submit(owner, conversation, "先打球再吃饭")
    assert claim_next_run(DATABASE_URL, "trip-test", 30)["run_id"] == first["run_id"]
    assert finish_run(DATABASE_URL, first["run_id"], "trip-test", status="completed", result={
        "response_kind": "route", "task_type": "path_planning", "explanation": "完整行程已生成。",
        "completed_place_request": trip,
    })

    revised = _submit(owner, conversation, "把用餐改成餐厅")
    assert get_run_input(DATABASE_URL, revised["run_id"])["previous_completed_place_request"] == trip
    assert claim_next_run(DATABASE_URL, "trip-test", 30)["run_id"] == revised["run_id"]
    assert finish_run(DATABASE_URL, revised["run_id"], "trip-test", status="completed", result={
        "task_type": "chat", "reply": "其他问题已回答。",
    })

    later = _submit(owner, conversation, "把用餐改成食堂")
    assert get_run_input(DATABASE_URL, later["run_id"])["previous_completed_place_request"] is None

    other_owner, _, other_conversation = create_conversation(DATABASE_URL)
    other = _submit(other_owner, other_conversation, "把用餐改成餐厅")
    assert get_run_input(DATABASE_URL, other["run_id"])["previous_completed_place_request"] is None
    cancel_run(DATABASE_URL, owner, later["run_id"])
    cancel_run(DATABASE_URL, other_owner, other["run_id"])


def test_a_later_task_supersedes_candidate_buttons_from_an_older_task():
    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    first = _submit(owner, conversation, "改成食堂")
    assert claim_next_run(DATABASE_URL, "stale-test", 30)["run_id"] == first["run_id"]
    assert finish_run(DATABASE_URL, first["run_id"], "stale-test", status="needs_input", result={
        "task_type": "candidates", "message": "请选择食堂。",
        "place_request": {"request_id": "first", "original_query": "食堂", "pending_slot": "selection",
                          "candidate_poi_ids": ["poi_310"]},
    })
    second = _submit(owner, conversation, "改成餐厅")
    assert claim_next_run(DATABASE_URL, "stale-test", 30)["run_id"] == second["run_id"]
    assert finish_run(DATABASE_URL, second["run_id"], "stale-test", status="needs_input", result={
        "task_type": "candidates", "message": "请选择餐厅。",
        "place_request": {"request_id": "second", "original_query": "餐厅", "pending_slot": "selection",
                          "candidate_poi_ids": ["poi_311"]},
    })

    with pytest.raises(TaskContinuationConflict):
        create_task_run(DATABASE_URL, owner, conversation, "选食堂", str(uuid4()),
                        datetime.now(timezone.utc) + timedelta(minutes=2),
                        continuation_task_id=first["task_id"], base_revision=1, selected_poi_id="poi_310")
    accepted = create_task_run(DATABASE_URL, owner, conversation, "选餐厅", str(uuid4()),
                               datetime.now(timezone.utc) + timedelta(minutes=2),
                               continuation_task_id=second["task_id"], base_revision=1,
                               selected_poi_id="poi_311")
    cancel_run(DATABASE_URL, owner, accepted["run_id"])


def test_late_completed_result_does_not_replace_the_newer_users_trip():
    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    older = _submit(owner, conversation, "旧行程")
    assert claim_next_run(DATABASE_URL, "older-worker", 30)["run_id"] == older["run_id"]
    newer = _submit(owner, conversation, "新行程")
    assert claim_next_run(DATABASE_URL, "newer-worker", 30)["run_id"] == newer["run_id"]
    old_trip = {"request_id": "old", "pending_slot": "selection", "completed_legs": [{"edge": [1, 2, 0]}]}
    new_trip = {"request_id": "new", "pending_slot": "selection", "completed_legs": [{"edge": [3, 4, 0]}]}
    for item, worker, trip in ((newer, "newer-worker", new_trip), (older, "older-worker", old_trip)):
        assert finish_run(DATABASE_URL, item["run_id"], worker, status="completed", result={
            "response_kind": "route", "task_type": "path_planning", "explanation": "行程已生成。",
            "completed_place_request": trip,
        })

    followup = _submit(owner, conversation, "把用餐改成食堂")
    assert get_run_input(DATABASE_URL, followup["run_id"])["previous_completed_place_request"] == new_trip
    cancel_run(DATABASE_URL, owner, followup["run_id"])


def test_late_route_answer_does_not_replace_newer_route_for_followup(monkeypatch):
    from agents.route_state import build_route_state

    database.initialize(DATABASE_URL)
    event_version = _current_route_environment(monkeypatch)
    owner, _, conversation = create_conversation(DATABASE_URL)
    older = _submit(owner, conversation, "旧路线")
    assert claim_next_run(DATABASE_URL, "older-route-worker", 30)["run_id"] == older["run_id"]
    newer = _submit(owner, conversation, "新路线")
    assert claim_next_run(DATABASE_URL, "newer-route-worker", 30)["run_id"] == newer["run_id"]

    def route(identifier):
        return build_route_state(route_id=identifier, route_kind="direct", original_query="路线",
                                 start={"type": "poi", "name": "总图"}, end={"type": "poi", "name": "樱顶"},
                                 travel_mode="walk", hard_constraints={}, strategy={"name": "shortest"},
                                 data_version="data-test", road_condition_version=event_version)

    for item, worker, identifier in ((newer, "newer-route-worker", "new-route"),
                                     (older, "older-route-worker", "old-route")):
        assert finish_run(DATABASE_URL, item["run_id"], worker, status="completed", result={
            "response_kind": "route", "task_type": "path_planning", "explanation": "路线已生成。",
            "route_state": route(identifier),
        })
    followup = _submit(owner, conversation, "改成骑行")
    try:
        assert get_run_input(DATABASE_URL, followup["run_id"])["previous_route_state"]["route_id"] == "new-route"
    finally:
        cancel_run(DATABASE_URL, owner, followup["run_id"])


@pytest.mark.parametrize("newer_state", ["failed", "queued"])
def test_failed_new_route_blocks_followup_from_inheriting_older_route(newer_state, monkeypatch):
    from agents.route_state import build_route_state

    database.initialize(DATABASE_URL)
    event_version = _current_route_environment(monkeypatch)
    owner, _, conversation = create_conversation(DATABASE_URL)
    older = _submit(owner, conversation, "从总图到樱顶")
    assert claim_next_run(DATABASE_URL, "old-route", 30)["run_id"] == older["run_id"]
    old_state = build_route_state(route_id="old-route", route_kind="direct", original_query="从总图到樱顶",
                                  start={"type": "poi", "name": "总图"}, end={"type": "poi", "name": "樱顶"},
                                  travel_mode="walk", hard_constraints={}, strategy={"name": "shortest"},
                                  data_version="data-test", road_condition_version=event_version)
    assert finish_run(DATABASE_URL, older["run_id"], "old-route", status="completed", result={
        "task_type": "path_planning", "response_kind": "route", "explanation": "旧路线已完成。",
        "route_state": old_state,
    })
    assert get_run(DATABASE_URL, owner, older["run_id"])["status"] == "completed"
    newer = _submit(owner, conversation, "从桂园食堂到梅园操场")
    if newer_state == "failed":
        assert claim_next_run(DATABASE_URL, "failed-route", 30)["run_id"] == newer["run_id"]
        assert finish_run(DATABASE_URL, newer["run_id"], "failed-route", status="failed", result={
            "task_type": "path_planning", "response_kind": "no_route", "message": "新路线不可达。",
        })
    followup = _submit(owner, conversation, "改成骑行")
    try:
        data = get_run_input(DATABASE_URL, followup["run_id"])
        assert data["previous_route_state"] is None
        assert data["route_context_uncertain"] is True
    finally:
        cancel_run(DATABASE_URL, owner, followup["run_id"])
        if newer_state == "queued":
            cancel_run(DATABASE_URL, owner, newer["run_id"])


def test_unrelated_weather_question_does_not_block_latest_route_context(monkeypatch):
    from agents.route_state import build_route_state

    database.initialize(DATABASE_URL)
    event_version = _current_route_environment(monkeypatch)
    owner, _, conversation = create_conversation(DATABASE_URL)
    original = _submit(owner, conversation, "从总图到樱顶")
    assert claim_next_run(DATABASE_URL, "route-before-weather", 30)["run_id"] == original["run_id"]
    route = build_route_state(route_id="previous-route", route_kind="direct", original_query="从总图到樱顶",
                              start={"type": "poi", "name": "总图"}, end={"type": "poi", "name": "樱顶"},
                              travel_mode="walk", hard_constraints={}, strategy={"name": "shortest"},
                              data_version="data-test", road_condition_version=event_version)
    assert finish_run(DATABASE_URL, original["run_id"], "route-before-weather", status="completed", result={
        "task_type": "path_planning", "response_kind": "route", "explanation": "路线已完成。", "route_state": route})
    weather = _submit(owner, conversation, "今天会下雨吗")
    assert claim_next_run(DATABASE_URL, "weather-only", 30)["run_id"] == weather["run_id"]
    assert finish_run(DATABASE_URL, weather["run_id"], "weather-only", status="completed", result={
        "task_type": "chat", "reply": "天气信息。"})
    followup = _submit(owner, conversation, "改成骑行")
    try:
        data = get_run_input(DATABASE_URL, followup["run_id"])
        assert data["previous_route_state"]["route_id"] == "previous-route"
        assert data["route_context_uncertain"] is False
    finally:
        cancel_run(DATABASE_URL, owner, followup["run_id"])


def test_candidate_id_is_rejected_when_no_current_selection_list_exists():
    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    first = _submit(owner, conversation, "我要素食")
    assert claim_next_run(DATABASE_URL, "no-candidate-worker", 30)["run_id"] == first["run_id"]
    assert finish_run(DATABASE_URL, first["run_id"], "no-candidate-worker", status="needs_input", result={
        "task_type": "unknown", "message": "无法确认素食条件。",
        "place_request": {"request_id": "unresolved", "original_query": "素食", "pending_slot": "activity",
                          "candidate_poi_ids": []},
    })
    with pytest.raises(TaskContinuationConflict):
        create_task_run(DATABASE_URL, owner, conversation, "就去这里", str(uuid4()),
                        datetime.now(timezone.utc) + timedelta(minutes=2),
                        continuation_task_id=first["task_id"], base_revision=1, selected_poi_id="poi_310")


@pytest.mark.parametrize("status", ["completed", "partial", "needs_input"])
def test_environment_change_after_computation_blocks_final_route_publication(monkeypatch, status):
    from agents import route_state, tools

    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    queued = _submit(owner, conversation, "选择餐厅")
    assert claim_next_run(DATABASE_URL, "publish-guard", 30)["run_id"] == queued["run_id"]
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-new")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    previous = {
        "request_id": "trip", "original_query": "餐厅", "pending_slot": "selection",
        "activities": ["restaurant_meal"], "subcategories": ["restaurant"],
        "candidate_poi_ids": [], "completed_legs": [],
    }
    stale = {
        "task_type": "path_planning", "response_kind": "route", "explanation": "旧路线",
        "recommended": [[114.35, 30.54]],
        "route_state": {"data_version": "data-old", "road_condition_version": "roads-old"},
        "completed_place_request": previous,
        "result_maintenance": {"prefix_action": "retained"},
    }
    assert finish_run(DATABASE_URL, queued["run_id"], "publish-guard", status=status, result=stale)
    actual = get_run(DATABASE_URL, owner, queued["run_id"])
    assert actual["status"] == "needs_input"
    assert actual["result"]["response_kind"] == "clarify"
    assert actual["result"].get("recommended") is None
    assert "变化" in actual["result"]["message"]


def test_worker_reports_the_committed_status_for_a_stale_updated_route(monkeypatch):
    from agents import route_state, tools
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    queued = _submit(owner, conversation, "选择餐厅")
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-new")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])

    class Graph:
        def invoke(self, *_args, **_kwargs):
            return {"result": {
                "task_type": "path_planning", "response_kind": "route", "explanation": "旧路线",
                "recommended": [[114.35, 30.54]],
                "route_state": {"data_version": "data-old", "road_condition_version": "roads-old"},
                "completed_place_request": {"request_id": "trip", "original_query": "餐厅",
                                            "pending_slot": "selection", "candidate_poi_ids": []},
                "result_maintenance": {"prefix_action": "retained"},
            }}

    outcome = process_next_run(DATABASE_URL, "worker-publish-guard", Graph())
    assert outcome["run_id"] == queued["run_id"]
    assert outcome["status"] == "needs_input"
    assert get_run(DATABASE_URL, owner, queued["run_id"])["status"] == "needs_input"


def test_delivery_guard_checks_graph_and_events_independently(monkeypatch):
    from agents import route_state, tools
    from storage.task_repository import _maintenance_snapshot_is_current

    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-current")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    result = {"route_state": {"data_version": "data-current",
                              "road_condition_version": route_state.current_road_condition_version([])}}
    assert _maintenance_snapshot_is_current(result)
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [{"id": "new-event"}])
    assert not _maintenance_snapshot_is_current(result)
    monkeypatch.setattr(tools, "list_conditions", lambda **_: (_ for _ in ()).throw(OSError("unavailable")))
    assert not _maintenance_snapshot_is_current(result)


def test_delivery_guard_also_rejects_outdated_candidate_snapshot(monkeypatch):
    from agents import route_state, tools
    from storage.task_repository import _maintenance_snapshot_is_current

    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-current")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    current_event = route_state.current_road_condition_version([])
    result = {"route_state": {"data_version": "data-current", "road_condition_version": current_event},
              "result_maintenance": {"dependencies": {"candidate_snapshot": {
                  "data_version": "data-current", "road_condition_version": "roads-old"}}}}
    assert not _maintenance_snapshot_is_current(result)


@pytest.mark.parametrize("snapshot", [None, {"data_version": "data-old", "road_condition_version": "roads-old"}])
@pytest.mark.parametrize("has_prefix", [False, True])
@pytest.mark.parametrize("anchor_only", [False, True])
def test_stale_candidate_only_delivery_keeps_task_but_clears_candidates(monkeypatch, snapshot, has_prefix, anchor_only):
    from agents import route_state, tools

    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    queued = _submit(owner, conversation, "选择餐厅")
    assert claim_next_run(DATABASE_URL, "candidate-publish-guard", 30)["run_id"] == queued["run_id"]
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-new")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    pending = {"request_id": "trip", "original_query": "餐厅", "pending_slot": "selection",
               "activities": ["restaurant_meal"], "subcategories": ["restaurant"],
               "start": {"type": "poi", "name": "工学部"}, "candidate_poi_ids": ["poi_311"],
               "completed_legs": [{"edge": [1, 2, 0]}] if has_prefix else [],
               "result_dependencies": {"candidate_snapshot": snapshot}}
    if anchor_only:
        pending["search_anchor"] = {"type": "poi", "name": "总图"}
        pending["start"] = None
    stale = {"task_type": "candidates", "response_kind": "candidates",
             "candidates": [{"poi_id": "poi_311", "name": "旧候选"}], "place_request": pending}
    assert finish_run(DATABASE_URL, queued["run_id"], "candidate-publish-guard",
                      status="needs_input", result=stale)
    actual = get_run(DATABASE_URL, owner, queued["run_id"])
    assert actual["status"] == "needs_input"
    assert actual["result"]["response_kind"] == "clarify"
    assert actual["result"].get("candidates") is None
    assert actual["result"]["place_request"]["completed_legs"] == pending["completed_legs"]
    assert actual["result"]["place_request"]["candidate_poi_ids"] == []
    if anchor_only:
        assert actual["result"]["place_request"]["search_anchor"] == pending["search_anchor"]
        assert actual["result"]["place_request"]["start"] is None


def test_current_nearby_candidates_without_origin_can_be_published(monkeypatch):
    database.initialize(DATABASE_URL)
    current_event = _current_route_environment(monkeypatch)
    owner, _, conversation = create_conversation(DATABASE_URL)
    queued = _submit(owner, conversation, "总图附近干饭")
    assert claim_next_run(DATABASE_URL, "nearby-publish", 30)["run_id"] == queued["run_id"]
    pending = {"request_id": "nearby", "original_query": "总图附近干饭", "pending_slot": "selection",
               "activities": ["meal"], "subcategories": ["restaurant"], "start": None,
               "search_anchor": {"type": "poi", "name": "总图"}, "candidate_poi_ids": ["poi_311"],
               "result_dependencies": {"candidate_snapshot": {
                   "data_version": "data-test", "road_condition_version": current_event}}}
    result = {"task_type": "candidates", "response_kind": "candidates",
              "candidates": [{"poi_id": "poi_311", "name": "餐厅"}], "place_request": pending}
    assert finish_run(DATABASE_URL, queued["run_id"], "nearby-publish", status="needs_input", result=result)
    actual = get_run(DATABASE_URL, owner, queued["run_id"])
    assert actual["status"] == "needs_input"
    assert actual["result"]["response_kind"] == "candidates"
    assert actual["result"]["candidates"][0]["poi_id"] == "poi_311"


def test_stale_partial_route_without_candidates_preserves_destination_question(monkeypatch):
    from agents import route_state, tools

    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    queued = _submit(owner, conversation, "先吃饭再回宿舍")
    assert claim_next_run(DATABASE_URL, "partial-route-guard", 30)["run_id"] == queued["run_id"]
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-new")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    pending = {"request_id": "trip", "original_query": "回宿舍", "pending_slot": "destination",
               "candidate_poi_ids": [], "completed_legs": [{"edge": [1, 2, 0]}]}
    stale = {"task_type": "path_planning", "response_kind": "route", "recommended": [[114.35, 30.54]],
             "route_state": {"data_version": "data-old", "road_condition_version": "roads-old"},
             "place_request": pending, "candidates": None, "needs_input": True}
    assert finish_run(DATABASE_URL, queued["run_id"], "partial-route-guard", status="needs_input", result=stale)
    actual = get_run(DATABASE_URL, owner, queued["run_id"])
    assert actual["status"] == "needs_input"
    assert actual["result"]["response_kind"] == "clarify"
    assert actual["result"].get("recommended") is None
    assert actual["result"]["place_request"]["pending_slot"] == "destination"
    assert actual["result"]["place_request"]["original_query"] == "回宿舍"


@pytest.mark.parametrize("response_kind", ["route", "no_route"])
def test_stale_single_route_or_unreachable_result_is_not_published(monkeypatch, response_kind):
    from agents import route_state, tools

    database.initialize(DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    queued = _submit(owner, conversation, "从总图到樱顶")
    assert claim_next_run(DATABASE_URL, "direct-route-guard", 30)["run_id"] == queued["run_id"]
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-new")
    monkeypatch.setattr(tools, "list_conditions", lambda **_: [])
    stale = {"task_type": "path_planning", "response_kind": response_kind,
             "message": "旧环境下的结果", "recommended": [[114.35, 30.54]],
             "route_state": {"data_version": "data-old", "road_condition_version": "roads-old"}}
    assert finish_run(DATABASE_URL, queued["run_id"], "direct-route-guard", status="completed", result=stale)
    actual = get_run(DATABASE_URL, owner, queued["run_id"])
    assert actual["status"] == "failed"
    assert actual["result"].get("recommended") is None
    assert "变化" in actual["result"]["message"]
