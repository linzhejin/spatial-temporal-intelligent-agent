import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from storage import database
from storage.task_repository import (
    RevisionConflict,
    apply_task_update,
    create_conversation,
    create_task,
    get_conversation,
    get_task,
    apply_task_patch,
    append_run_event,
    cancel_run,
    claim_next_run,
    enqueue_run,
    finish_run,
    get_run,
    heartbeat_run,
    create_task_run,
    get_run_input,
    RunConflict,
    list_active_runs,
)


DATABASE_URL = None  # assigned only after isolated_test_database validates the target
pytestmark = pytest.mark.usefixtures("isolated_test_database")


@pytest.fixture(autouse=True)
def clean_tables():
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
            conn.execute("TRUNCATE run_event, run, task_history, task, profile, conversation CASCADE")


def task_payload(conversation_id, query):
    from agents.state_models import TaskState
    return TaskState(task_id="temporary", conversation_id=conversation_id).model_dump(mode="json")


def sample_route_state(start="玉兰2门", end="星湖园食堂"):
    return {
        "schema_version": 1, "route_id": "route-sample", "route_kind": "direct",
        "original_query": f"从{start}到{end}",
        "start": {"name": start, "type": "poi"},
        "end": {"name": end, "type": "poi"}, "via": None, "tour": None, "legs": [],
        "travel_mode": "walk", "hard_constraints": {"slope": "normal"},
        "strategy": {"name": "shortest", "source": "commute_default", "task_class": "commute",
                     "weights": {"distance": 1.0, "slope": 0.0, "scenery": 0.0}, "detour_cap": 1.0},
        "data_version": "data-v1", "road_condition_version": "roads-v1",
    }


def test_conversation_credential_is_required_to_restore_state():
    owner, token, conversation_id = create_conversation(DATABASE_URL)
    assert get_conversation(DATABASE_URL, conversation_id, token) == {
        "conversation_id": conversation_id,
        "owner_id": owner,
        "revision": 0,
    }
    assert get_conversation(DATABASE_URL, conversation_id, "wrong-token") is None


def test_task_update_uses_compare_and_swap_and_preserves_history():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    task_id = create_task(DATABASE_URL, owner, conversation_id, task_payload(conversation_id, "去樱顶"))
    updated = apply_task_update(DATABASE_URL, owner, task_id, 0, {"status": "running"})
    assert updated["revision"] == 1
    assert updated["status"] == "running"
    with pytest.raises(RevisionConflict):
        apply_task_update(DATABASE_URL, owner, task_id, 0, {"status": "failed"})
    assert get_task(DATABASE_URL, owner, task_id)["status"] == "running"


def test_only_one_concurrent_update_can_commit_for_one_revision():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    task_id = create_task(DATABASE_URL, owner, conversation_id, task_payload(conversation_id, "查路线"))
    barrier = threading.Barrier(2)

    def update(status):
        barrier.wait()
        try:
            apply_task_update(DATABASE_URL, owner, task_id, 0, {"status": status})
            return "committed"
        except RevisionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, ["running", "failed"]))
    assert sorted(results) == ["committed", "conflict"]
    assert get_task(DATABASE_URL, owner, task_id)["revision"] == 1


def test_another_owner_cannot_read_or_update_task():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    task_id = create_task(DATABASE_URL, owner, conversation_id, task_payload(conversation_id, "私人路线"))
    other_owner = str(uuid.uuid4())
    assert get_task(DATABASE_URL, other_owner, task_id) is None
    assert apply_task_update(DATABASE_URL, other_owner, task_id, 0, {"status": "failed"}) is None


def test_user_patch_is_validated_and_committed_with_provenance():
    from agents.state_models import TaskState

    owner, _, conversation_id = create_conversation(DATABASE_URL)
    state = TaskState.model_validate({
        "task_id": "temporary", "conversation_id": conversation_id,
        "route_spec": {
            "start": {"poi_id": "gate-yulan-2"},
            "end": {"poi_id": "poi-library"},
            "hard_constraints": {"avoid_slope": True},
        },
    })
    task_id = create_task(DATABASE_URL, owner, conversation_id, state.model_dump(mode="json"))
    state.task_id = task_id
    patch = {
        "task_id": task_id, "base_revision": 0, "source_message_id": "message-1",
        "operations": [{"op": "remove", "path": "route_spec.hard_constraints.avoid_slope"}],
    }
    updated = apply_task_patch(DATABASE_URL, owner, patch)
    assert updated["revision"] == 1
    assert updated["state"]["route_spec"]["hard_constraints"]["avoid_slope"] is False
    assert updated["state"]["field_provenance"]["route_spec.hard_constraints.avoid_slope"]["source_message_id"] == "message-1"


def test_invalid_user_patch_rolls_back_without_revision_change():
    from agents.state_models import TaskState

    owner, _, conversation_id = create_conversation(DATABASE_URL)
    state = TaskState(task_id="temporary", conversation_id=conversation_id)
    task_id = create_task(DATABASE_URL, owner, conversation_id, state.model_dump(mode="json"))
    with pytest.raises(ValueError):
        apply_task_patch(DATABASE_URL, owner, {
            "task_id": task_id, "base_revision": 0, "source_message_id": "message-2",
            "operations": [{"op": "set", "path": "status", "value": "completed"}],
        })
    assert get_task(DATABASE_URL, owner, task_id)["revision"] == 0


def test_run_enqueue_is_idempotent_and_events_have_ordered_sequences():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    task_id = create_task(DATABASE_URL, owner, conversation_id, task_payload(conversation_id, "路线"))
    deadline = datetime.now(timezone.utc) + timedelta(minutes=2)
    first = enqueue_run(DATABASE_URL, owner, task_id, 0, "request-1", deadline)
    retry = enqueue_run(DATABASE_URL, owner, task_id, 0, "request-1", deadline)
    assert retry["run_id"] == first["run_id"]
    assert retry["created"] is False

    claimed = claim_next_run(DATABASE_URL, "worker-a", 30)
    assert claimed["run_id"] == first["run_id"]
    assert heartbeat_run(DATABASE_URL, first["run_id"], "worker-a", 30)
    one = append_run_event(DATABASE_URL, first["run_id"], "worker-a", "node_started", {"node": "route"})
    two = append_run_event(DATABASE_URL, first["run_id"], "worker-a", "node_completed", {"node": "route"})
    assert (one["seq"], two["seq"]) == (1, 2)
    assert finish_run(DATABASE_URL, first["run_id"], "worker-a", status="completed", result={"ok": True})
    restored = get_run(DATABASE_URL, owner, first["run_id"], after_seq=1)
    assert restored["status"] == "completed"
    assert [event["seq"] for event in restored["events"]] == [2, 3]
    assert restored["events"][-1]["event_type"] == "run_completed"


def test_two_workers_cannot_claim_the_same_run():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    task_id = create_task(DATABASE_URL, owner, conversation_id, task_payload(conversation_id, "路线"))
    enqueue_run(DATABASE_URL, owner, task_id, 0, "claim-race", datetime.now(timezone.utc) + timedelta(minutes=2))
    barrier = threading.Barrier(2)

    def claim(worker):
        barrier.wait()
        return claim_next_run(DATABASE_URL, worker, 30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["worker-a", "worker-b"]))
    claims = [item for item in results if item]
    assert len(claims) == 1


def test_expired_lease_recovers_and_rejects_old_worker_writes():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    task_id = create_task(DATABASE_URL, owner, conversation_id, task_payload(conversation_id, "路线"))
    queued = enqueue_run(DATABASE_URL, owner, task_id, 0, "lease-recovery", datetime.now(timezone.utc) + timedelta(minutes=2))
    first = claim_next_run(DATABASE_URL, "worker-a", 30)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("UPDATE run SET lease_until=now()-interval '1 second' WHERE run_id=%s", (queued["run_id"],))
    second = claim_next_run(DATABASE_URL, "worker-b", 30)
    assert first["run_id"] == second["run_id"]
    assert second["attempt"] == 2
    assert not append_run_event(DATABASE_URL, queued["run_id"], "worker-a", "late", {})
    assert not finish_run(DATABASE_URL, queued["run_id"], "worker-a", status="completed")
    assert heartbeat_run(DATABASE_URL, queued["run_id"], "worker-b", 30)


def test_deadline_expiry_closes_run_and_task_instead_of_leaving_it_queued():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "过期任务", "deadline-expired",
        datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert claim_next_run(DATABASE_URL, "worker", 30) is None
    task = get_task(DATABASE_URL, owner, queued["task_id"])
    run = get_run(DATABASE_URL, owner, queued["run_id"])
    assert task["status"] == "failed"
    assert run["status"] == "failed"
    assert run["events"][-1]["event_type"] == "run_completed"


def test_poll_closes_expired_run_even_without_an_available_worker():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "等待中的任务", "poll-deadline-expired",
        datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    polled = get_run(DATABASE_URL, owner, queued["run_id"])

    assert polled["status"] == "failed"
    assert polled["error"]["code"] == "deadline_exceeded"
    assert polled["events"][-1]["event_type"] == "run_completed"
    assert get_task(DATABASE_URL, owner, queued["task_id"])["status"] == "failed"


def test_cancel_is_owner_scoped_and_idempotent():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    task_id = create_task(DATABASE_URL, owner, conversation_id, task_payload(conversation_id, "路线"))
    queued = enqueue_run(DATABASE_URL, owner, task_id, 0, "cancel-me", datetime.now(timezone.utc) + timedelta(minutes=2))
    assert cancel_run(DATABASE_URL, str(uuid.uuid4()), queued["run_id"]) is None
    cancelled = cancel_run(DATABASE_URL, owner, queued["run_id"])
    assert cancelled["status"] == "cancelled"
    assert cancel_run(DATABASE_URL, owner, queued["run_id"])["cancel_requested"] is True
    assert claim_next_run(DATABASE_URL, "worker", 30) is None


def test_message_task_and_run_are_created_atomically_and_idempotently():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=2)
    first = create_task_run(
        DATABASE_URL, owner, conversation_id, "去星湖园食堂，别走楼梯", "submit-1", deadline
    )
    retry = create_task_run(
        DATABASE_URL, owner, conversation_id, "去星湖园食堂，别走楼梯", "submit-1", deadline
    )
    assert retry["task_id"] == first["task_id"]
    assert retry["run_id"] == first["run_id"]
    assert retry["created"] is False
    payload = get_run_input(DATABASE_URL, first["run_id"])
    assert payload["query"] == "去星湖园食堂，别走楼梯"
    assert payload["task"]["status"] == "queued"
    assert payload["task"]["route_spec"]["travel_mode"] == "walk"
    with database.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversation_message WHERE task_id=%s",
            (first["task_id"],),
        ).fetchall()
    assert [(row["role"], row["content"]) for row in rows] == [("user", "去星湖园食堂，别走楼梯")]


def test_fresh_task_does_not_receive_unfinished_earlier_task_as_history():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=2)
    create_task_run(DATABASE_URL, owner, conversation_id, "第一个未完成任务", "queued-one", deadline)
    second = create_task_run(DATABASE_URL, owner, conversation_id, "第二个独立任务", "queued-two", deadline)

    second_input = get_run_input(DATABASE_URL, second["run_id"])
    assert second_input["query"] == "第二个独立任务"
    assert all("第一个未完成任务" not in entry["content"] for entry in second_input["history"])


def test_run_input_preserves_selected_mode_points_and_previous_route_context():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    first = create_task_run(
        DATABASE_URL, owner, conversation_id, "从玉兰2门去星湖园食堂", "context-first",
        datetime.now(timezone.utc) + timedelta(minutes=2),
        route_spec={
            "travel_mode": "bike",
            "start": {"name": "玉兰2门", "coordinates": {
                "longitude": 114.3601, "latitude": 30.5299, "crs": "WGS84",
            }},
        },
    )
    claim_next_run(DATABASE_URL, "worker-a", 30)
    assert finish_run(DATABASE_URL, first["run_id"], "worker-a", status="completed", result={
        "task_type": "path_planning", "explanation": "已规划到食堂。",
        "route_state": sample_route_state(),
    })

    second = create_task_run(
        DATABASE_URL, owner, conversation_id, "换一条平坦的", "context-second",
        datetime.now(timezone.utc) + timedelta(minutes=2),
        route_spec={"travel_mode": "walk"},
    )
    payload = get_run_input(DATABASE_URL, second["run_id"])
    assert payload["task"]["route_spec"]["travel_mode"] == "walk"
    assert payload["history"] == [
        {"role": "user", "content": "从玉兰2门去星湖园食堂"},
        {"role": "assistant", "content": "已规划到食堂。"},
    ]
    assert payload["previous_route_state"]["end"]["name"] == "星湖园食堂"


def test_clarification_answer_resumes_same_task_with_revision_and_route_fields():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    first = create_task_run(
        DATABASE_URL, owner, conversation_id, "带我去图书馆", "clarification-first",
        datetime.now(timezone.utc) + timedelta(minutes=2), route_spec={"travel_mode": "walk"},
    )
    claim_next_run(DATABASE_URL, "worker-a", 30)
    assert finish_run(DATABASE_URL, first["run_id"], "worker-a", status="needs_input", result={
        "task_type": "unknown", "message": "你想从哪里出发？",
        "clarify": {"question": "你想从哪里出发？", "options": ["从樱顶出发"]},
    })
    awaiting = get_task(DATABASE_URL, owner, first["task_id"])
    assert awaiting["revision"] == 1
    assert awaiting["state"]["pending_questions"] == ["你想从哪里出发？"]

    resumed = create_task_run(
        DATABASE_URL, owner, conversation_id, "从樱顶出发", "clarification-answer",
        datetime.now(timezone.utc) + timedelta(minutes=2),
        route_spec={"travel_mode": "bike"}, continuation_task_id=first["task_id"], base_revision=1,
    )
    assert resumed["task_id"] == first["task_id"]
    assert resumed["task_revision"] == 2
    state = get_task(DATABASE_URL, owner, first["task_id"])["state"]
    assert state["status"] == "queued"
    assert state["pending_questions"] == []
    assert state["route_spec"]["travel_mode"] == "bike"
    payload = get_run_input(DATABASE_URL, resumed["run_id"])
    assert [item["content"] for item in payload["history"]] == [
        "带我去图书馆", "你想从哪里出发？",
    ]
    assert payload["query"] == "从樱顶出发"
    assert payload["task_origin_query"] == "带我去图书馆"
    retry = create_task_run(
        DATABASE_URL, owner, conversation_id, "从樱顶出发", "clarification-answer",
        datetime.now(timezone.utc) + timedelta(minutes=2),
        route_spec={"travel_mode": "bike"}, continuation_task_id=first["task_id"], base_revision=1,
    )
    assert retry["run_id"] == resumed["run_id"]
    with pytest.raises(RunConflict):
        create_task_run(
            DATABASE_URL, owner, conversation_id, "另一个回答", "stale-answer",
            datetime.now(timezone.utc) + timedelta(minutes=2),
            continuation_task_id=first["task_id"], base_revision=1,
        )


def test_message_creation_rejects_foreign_conversation_without_writes():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    with pytest.raises(PermissionError):
        create_task_run(
            DATABASE_URL, str(uuid.uuid4()), conversation_id, "private", "foreign-1",
            datetime.now(timezone.utc) + timedelta(minutes=2),
        )
    with database.connect(DATABASE_URL) as conn:
        assert conn.execute("SELECT count(*) AS n FROM task").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM run").fetchone()["n"] == 0


def test_successful_run_appends_assistant_message_and_revisions_task():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "去图书馆", "reply-1",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    claim_next_run(DATABASE_URL, "worker-a", 30)
    assert finish_run(DATABASE_URL, queued["run_id"], "worker-a", status="completed",
                      result={
                          "task_type": "path_planning", "explanation": "图书馆在文理学部。",
                          "route_state": sample_route_state("樱顶", "图书馆"),
                      })
    task = get_task(DATABASE_URL, owner, queued["task_id"])
    assert task["revision"] == 1
    assert task["status"] == "completed"
    assert task["state"]["route_spec"]["start"]["name"] == "樱顶"
    assert task["state"]["route_spec"]["end"]["name"] == "图书馆"
    assert task["state"]["route_spec"]["data_version"] == "data-v1"
    with database.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT seq, role, content, metadata FROM conversation_message WHERE task_id=%s ORDER BY seq",
            (queued["task_id"],),
        ).fetchall()
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "去图书馆"), ("assistant", "图书馆在文理学部。"),
    ]
    assert rows[-1]["metadata"]["source_message_seq"] == rows[0]["seq"]


def test_active_run_exposes_original_user_message_sequence():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "从玉兰二门去图书馆", "active-sequence",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )

    active = list_active_runs(DATABASE_URL, owner, conversation_id)

    assert active is not None
    run = next(item for item in active if item["run_id"] == queued["run_id"])
    assert run["query"] == "从玉兰二门去图书馆"
    assert run["message_seq"] == 1


def test_cancelled_run_drops_malformed_optional_route_proof():
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "从樱顶去图书馆", "cancelled-bad-proof",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    claim_next_run(DATABASE_URL, "worker-cancelled-proof", 30)
    route_state = sample_route_state("樱顶", "图书馆")
    route_state["reuse_certificate"] = "malformed-handle"

    assert finish_run(
        DATABASE_URL, queued["run_id"], "worker-cancelled-proof", status="cancelled",
        result={"route_state": route_state},
    )

    restored = get_run(DATABASE_URL, owner, queued["run_id"])
    assert restored["status"] == "cancelled"
    assert "reuse_certificate" not in restored["result"]["route_state"]
