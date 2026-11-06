"""Integration contracts for unbound, ordered conversation runs."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from psycopg.types.json import Jsonb

from storage import database
from storage.task_repository import (
    bind_model_task, cancel_run, claim_next_run, create_conversation, create_task_run,
    finish_run, get_run, get_run_input, get_task, list_active_runs, read_conversation_result, RunConflict, TaskContinuationConflict,
)

pytestmark = pytest.mark.usefixtures("isolated_test_database")
DATABASE_URL = None


def submit(owner, conversation, text, key=None):
    return create_task_run(DATABASE_URL, owner, conversation, text, key or str(uuid4()),
                           datetime.now(timezone.utc) + timedelta(seconds=75))


@pytest.fixture(autouse=True)
def model_mode(monkeypatch):
    monkeypatch.setenv("AGENT_DIALOGUE_MODE", "model")
    database.initialize(DATABASE_URL)


def test_message_is_queued_unbound_and_all_previous_raw_messages_are_loaded():
    owner, _, conversation = create_conversation(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        for index in range(1, 19):
            conn.execute("INSERT INTO conversation_message(conversation_id,seq,message_id,role,content,metadata)"
                         " VALUES (%s,%s,%s,%s,%s,%s)",
                         (conversation, index, str(uuid4()), "user" if index % 2 else "assistant",
                          f"original-{index}-" + "x" * 5001, Jsonb({"index": index})))
        conn.execute("UPDATE conversation SET revision=18 WHERE conversation_id=%s", (conversation,))
    accepted = submit(owner, conversation, "继续最初的要求", "full-history-" + str(uuid4()))
    assert accepted["task_id"] is None
    assert accepted["task_revision"] is None
    active = list_active_runs(DATABASE_URL, owner, conversation)
    assert active[0]["run_id"] == accepted["run_id"]
    data = get_run_input(DATABASE_URL, accepted["run_id"])
    assert len(data["history"]) == 18
    assert data["history"][0]["content"] == "original-1-" + "x" * 5001
    assert data["history"][-1]["content"] == "original-18-" + "x" * 5001
    assert cancel_run(DATABASE_URL, owner, accepted["run_id"])["status"] == "cancelled"


def test_unbound_runs_serialize_per_conversation_and_cancel_by_owner():
    owner, _, conversation = create_conversation(DATABASE_URL)
    first = submit(owner, conversation, "先查天气")
    second = submit(owner, conversation, "再看路线")
    assert claim_next_run(DATABASE_URL, "worker-one")["run_id"] == first["run_id"]
    assert claim_next_run(DATABASE_URL, "worker-two") is None
    assert cancel_run(DATABASE_URL, str(uuid4()), first["run_id"]) is None
    assert cancel_run(DATABASE_URL, owner, first["run_id"])["status"] == "cancelled"
    assert claim_next_run(DATABASE_URL, "worker-two")["run_id"] == second["run_id"]
    assert cancel_run(DATABASE_URL, owner, second["run_id"])["status"] == "cancelled"


def test_model_binding_checks_version_and_late_run_cannot_publish():
    owner, _, conversation = create_conversation(DATABASE_URL)
    first = submit(owner, conversation, "从信图出发找总图附近吃饭")
    assert claim_next_run(DATABASE_URL, "binding-worker")["run_id"] == first["run_id"]
    pending = {"request_id": str(uuid4()), "original_query": "找吃饭地点", "mode": "walk",
               "start": {"type": "poi", "name": "信图"},
               "search_anchor": {"type": "poi", "name": "总图"},
               "semantic_request": {"operation": "place_search"}}
    bound = bind_model_task(DATABASE_URL, first["run_id"], "binding-worker", relation="new",
                            place_request=pending)
    assert bound["task_id"] and bound["task_revision"] == 0
    newer = submit(owner, conversation, "先换个话题")
    assert finish_run(DATABASE_URL, first["run_id"], "binding-worker", status="completed",
                      result={"message": "过期结果"}, return_status=True) == "superseded"
    assert get_run(DATABASE_URL, owner, first["run_id"])["status"] == "superseded"
    assert cancel_run(DATABASE_URL, owner, newer["run_id"])["status"] == "cancelled"


def test_newer_input_prevents_late_binding_and_duplicate_request_is_idempotent():
    owner, _, conversation = create_conversation(DATABASE_URL)
    key = "idempotency-" + str(uuid4())
    first = submit(owner, conversation, "  原文  ", key)
    repeated = submit(owner, conversation, "  原文  ", key)
    assert repeated["run_id"] == first["run_id"] and repeated["created"] is False
    assert get_run_input(DATABASE_URL, first["run_id"])["query"] == "  原文  "
    assert claim_next_run(DATABASE_URL, "late-bind-worker")["run_id"] == first["run_id"]
    newer = submit(owner, conversation, "换个问题")
    with pytest.raises(RunConflict):
        bind_model_task(DATABASE_URL, first["run_id"], "late-bind-worker", relation="new",
                        place_request={"request_id": str(uuid4()), "original_query": "原文"})
    assert finish_run(DATABASE_URL, first["run_id"], "late-bind-worker", status="completed",
                      result={"message": "迟到"}, return_status=True) == "superseded"
    assert cancel_run(DATABASE_URL, owner, newer["run_id"])["status"] == "cancelled"


def test_a_candidate_card_is_rejected_after_a_newer_message(monkeypatch):
    monkeypatch.setattr("storage.task_repository._maintenance_snapshot_is_current", lambda _result: True)
    owner, _, conversation = create_conversation(DATABASE_URL)
    first = submit(owner, conversation, "推荐餐厅")
    assert claim_next_run(DATABASE_URL, "card-worker")["run_id"] == first["run_id"]
    candidate = {"poi_id": "amap:A123", "name": "候选甲"}
    pending = {"request_id": str(uuid4()), "original_query": "推荐餐厅", "pending_slot": "selection",
               "candidate_poi_ids": ["amap:A123"], "candidate_places": [candidate], "candidate_pool": [candidate]}
    bound = bind_model_task(DATABASE_URL, first["run_id"], "card-worker", relation="new", place_request=pending)
    assert finish_run(DATABASE_URL, first["run_id"], "card-worker", status="needs_input",
                      result={"message": "请选择", "response_kind": "candidates", "candidates": [candidate],
                              "place_request": pending})
    revision = get_run(DATABASE_URL, owner, first["run_id"])["current_task_revision"]
    newer = submit(owner, conversation, "先查天气")
    with pytest.raises(TaskContinuationConflict):
        create_task_run(DATABASE_URL, owner, conversation, "去这家", str(uuid4()),
                        datetime.now(timezone.utc) + timedelta(seconds=75),
                        continuation_task_id=bound["task_id"], base_revision=revision, selected_poi_id="amap:A123")
    assert cancel_run(DATABASE_URL, owner, newer["run_id"])["status"] == "cancelled"


def test_unbound_timeout_is_visible_after_conversation_restore():
    from storage.task_repository import get_conversation_messages
    owner, _, conversation = create_conversation(DATABASE_URL)
    run = create_task_run(DATABASE_URL, owner, conversation, "查一下", str(uuid4()),
                          datetime.now(timezone.utc) - timedelta(seconds=1))
    terminal = get_run(DATABASE_URL, owner, run["run_id"])
    assert terminal["status"] == "failed"
    assert terminal["error"]["code"] == "deadline_exceeded"
    messages = get_conversation_messages(DATABASE_URL, owner, conversation)
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert "超时" in messages[-1]["content"]
    assert messages[-1]["metadata"]["source_message_seq"] == messages[0]["seq"]


def test_cancelled_run_records_the_visible_cancel_action_once():
    from storage.task_repository import get_conversation_messages
    owner, _, conversation = create_conversation(DATABASE_URL)
    queued = submit(owner, conversation, "先帮我查一下")
    assert cancel_run(DATABASE_URL, owner, queued["run_id"])["status"] == "cancelled"
    assert cancel_run(DATABASE_URL, owner, queued["run_id"])["status"] == "cancelled"
    messages = get_conversation_messages(DATABASE_URL, owner, conversation)
    assert [row["role"] for row in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "这项任务已停止，你之前确认的条件仍保留。需要继续时告诉我下一步。"
    assert messages[-1]["metadata"]["ui_action"]["type"] == "cancel"
    assert messages[-1]["metadata"]["source_message_seq"] == messages[0]["seq"]


def test_historical_result_pages_are_bounded_and_conversation_scoped():
    owner, _, conversation = create_conversation(DATABASE_URL)
    run = submit(owner, conversation, "旧路线")
    assert claim_next_run(DATABASE_URL, "result-page-worker")["run_id"] == run["run_id"]
    assert finish_run(DATABASE_URL, run["run_id"], "result-page-worker", status="completed",
                      result={"message": "路线", "recommended": [[1, 2], [3, 4], [5, 6]]})
    page = read_conversation_result(DATABASE_URL, conversation, run_id=run["run_id"],
                                    section="route_geometry", offset=1, limit=1)
    assert page["items"] == [[3, 4]] and page["next_offset"] == 2 and not page["exhausted"]
    _, _, other = create_conversation(DATABASE_URL)
    assert read_conversation_result(DATABASE_URL, other, run_id=run["run_id"],
                                    section="route_geometry")["error"] == "result_reference_unavailable"


def test_candidate_delivery_uses_snapshot_from_an_earlier_nonempty_search(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from langgraph.checkpoint.memory import InMemorySaver
    from agents import planner, route_state, workflow
    from jobs.worker import process_next_run
    from tests.test_model_dialogue import place, reply

    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "test-only")
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda _context: ({}, None))
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-current")
    monkeypatch.setattr(route_state, "current_road_condition_version", lambda _conditions: "roads-current")
    monkeypatch.setattr(planner.agent_tools, "list_conditions", lambda **_kwargs: [])
    candidate = place("amap:A123", "候选餐厅")
    searches = Mock(side_effect=[
        ({"status": "ok", "candidates": [candidate], "data_version": "data-current",
          "road_condition_version": "roads-current"}, {"candidates": [candidate]}),
        ({"status": "ok", "candidates": [], "count": 0}, {"candidates": []}),
    ])
    monkeypatch.setattr(planner.agent_tools, "execute_tool", searches)
    model = Mock(side_effect=[
        reply(update_task={"relation": "new", "operation": "place_search", "subcategories": ["restaurant"],
                           "start": {"type": "poi", "name": "信图"},
                           "search_anchor": {"type": "poi", "name": "总图"}}),
        reply(search_poi_candidates={"keyword": "餐馆"}),
        reply(search_poi_candidates={"keyword": "另一关键词"}),
        reply(present_places={"introduction": "候选", "places": [
            {"poi_id": candidate["poi_id"], "reason": "目录候选"}]}),
    ])
    monkeypatch.setattr(planner, "_make_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=model))))
    owner, _, conversation = create_conversation(DATABASE_URL)
    run = submit(owner, conversation, "信图出发，总图附近找餐馆")
    graph = workflow.build_agent_workflow(InMemorySaver(), database_url=DATABASE_URL)
    assert process_next_run(DATABASE_URL, "snapshot-model-worker", graph, lease_seconds=60)["status"] == "needs_input"
    delivery = get_run(DATABASE_URL, owner, run["run_id"])["result"]
    assert delivery["response_kind"] == "candidates"
    assert delivery["candidates"][0]["poi_id"] == candidate["poi_id"]


def test_worker_model_continues_same_task_and_preserves_distinct_anchors(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from langgraph.checkpoint.memory import InMemorySaver
    from agents import planner, workflow
    from jobs.worker import process_next_run
    from tests.test_model_dialogue import place, reply

    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "test-only")
    monkeypatch.setattr("storage.task_repository._maintenance_snapshot_is_current", lambda _result: True)
    monkeypatch.setattr(planner.agent_tools, "_prepare_route_context", lambda _context: ({}, None))
    rows = [place("amap:A123", "甲餐厅"), place("amap:B123", "乙餐厅")]
    observed = []
    def spatial(name, args, _context):
        observed.append((name, args))
        if name == "get_weather":
            return {"weather": "晴", "temperature": 22}, {}
        if args.get("max_distance_m") == 2000:
            extra = place("amap:C123", "丙餐厅")
            return {"status": "ok", "count": 1, "candidates": [extra]}, {"candidates": [extra]}
        return {"status": "ok", "count": 2, "candidates": rows}, {"candidates": rows}
    monkeypatch.setattr(planner.agent_tools, "execute_tool", spatial)
    model = Mock()
    monkeypatch.setattr(planner, "_make_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=model))))
    graph = workflow.build_agent_workflow(InMemorySaver(), database_url=DATABASE_URL)
    owner, _, conversation = create_conversation(DATABASE_URL)
    first = submit(owner, conversation, "信图出发，总图附近吃饭")
    model.side_effect = [
        reply(update_task={"relation": "new", "operation": "place_search", "activities": ["meal"],
                           "start": {"type": "poi", "name": "信图"},
                           "search_anchor": {"type": "poi", "name": "总图"}},
              search_poi_candidates={"keyword": "餐饮"}),
        reply(present_places={"introduction": "先看甲。", "places": [
            {"poi_id": "amap:A123", "reason": "目录候选"}]}),
    ]
    assert process_next_run(DATABASE_URL, "first-model-worker", graph, lease_seconds=60)["status"] == "needs_input"
    completed = get_run(DATABASE_URL, owner, first["run_id"])
    task_id, revision = completed["task_id"], completed["current_task_revision"]
    second = submit(owner, conversation, "再给我瞧瞧其他吃饭的地方")
    model.side_effect = [
        reply(update_task={"relation": "continue", "operation": "place_search", "task_id": task_id,
                           "task_revision": revision, "candidate_action": "next_batch"}),
        reply(present_places={"introduction": "再看乙。", "places": [
            {"poi_id": "amap:B123", "reason": "目录候选"}]}),
    ]
    assert process_next_run(DATABASE_URL, "second-model-worker", graph, lease_seconds=60)["status"] == "needs_input"
    latest = get_run(DATABASE_URL, owner, second["run_id"])
    assert latest["task_id"] == task_id
    assert [item["poi_id"] for item in latest["result"]["candidates"]] == ["amap:B123"]
    pending = latest["result"]["place_request"]
    assert pending["start"]["name"] == "信图"
    assert pending["search_anchor"]["name"] == "总图"
    assert pending["activities"] == ["meal"]
    assert observed[0][1]["max_distance_m"] == 1000
    assert "信图出发，总图附近吃饭" in str(model.call_args_list[2].kwargs["messages"])
    last_revision = latest["current_task_revision"]
    for index in range(3):
        submit(owner, conversation, f"换个话题 {index}")
        model.side_effect = [reply(f"无关话题的回答 {index}")]
        assert process_next_run(DATABASE_URL, f"interlude-{index}", graph, lease_seconds=60)["status"] == "completed"
    resumed = submit(owner, conversation, "回到之前吃饭的事，范围加大，其他照旧")
    model.side_effect = [
        reply(update_task={"relation": "continue", "operation": "place_search", "task_id": task_id,
                           "task_revision": last_revision, "expand_search_radius": True},
              search_poi_candidates={"keyword": "餐饮"}),
        reply(present_places={"introduction": "扩大到两公里。", "places": [
            {"poi_id": "amap:C123", "reason": "目录候选"}]}),
    ]
    assert process_next_run(DATABASE_URL, "resumed-model-worker", graph, lease_seconds=60)["status"] == "needs_input"
    restored = get_run(DATABASE_URL, owner, resumed["run_id"])
    restored_pending = restored["result"]["place_request"]
    assert restored["task_id"] == task_id
    assert restored_pending["search_radius_m"] == 2000
    assert restored_pending["start"]["name"] == "信图"
    assert restored_pending["search_anchor"]["name"] == "总图"
    assert restored_pending["activities"] == ["meal"]
    resumed_input = get_run_input(DATABASE_URL, resumed["run_id"])
    assert len(resumed_input["history"]) >= 10
    stored = get_task(DATABASE_URL, owner, task_id)
    assert stored["state"]["field_provenance"]["place_request.search_radius_m"]["source_message_id"] == resumed_input["message_id"]
