"""A rollback keeps online-candidate history readable and refuses unsafe continuation."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from agents.state_models import PlaceRequestState, TaskState
from storage import task_repository


def test_online_candidate_snapshot_round_trips_without_losing_provider_identity():
    places = [{"id": "amap:release_probe", "name": "test location"}]
    state = PlaceRequestState.model_validate({
        "request_id": "rollback-probe", "original_query": "dining probe",
        "candidate_places": places,
    })
    assert state.model_dump()["candidate_places"] == places


def test_online_candidate_continuation_is_rejected_before_any_write(monkeypatch):
    state = TaskState(task_id="task-probe", conversation_id="conversation-probe",
                      revision=1, status="needs_input").model_dump(mode="json")
    state["place_request"] = {
        "request_id": "rollback-probe", "original_query": "dining probe",
        "candidate_places": [{"id": "amap:release_probe"}],
    }

    class Cursor:
        def __init__(self, value):
            self.value = value

        def fetchone(self):
            return self.value

    class Connection:
        def execute(self, sql, params):
            if sql.startswith("SELECT pg_advisory"):
                return Cursor(None)
            if sql.startswith("SELECT r.run_id"):
                return Cursor(None)
            if sql.startswith("SELECT 1 FROM conversation"):
                return Cursor((1,))
            if sql.startswith("SELECT t.revision"):
                return Cursor({"revision": 1, "status": "needs_input", "state": state,
                               "task_kind": "route"})
            raise AssertionError("Rollback must not modify a newer online-candidate task")

    @contextmanager
    def connect(_url):
        yield Connection()

    monkeypatch.setattr(task_repository.database, "connect", connect)
    with pytest.raises(task_repository.TaskContinuationConflict, match="新建对话"):
        task_repository.create_task_run(
            None, "owner-probe", "conversation-probe", "choose this location", "probe-id",
            datetime.now(timezone.utc) + timedelta(minutes=1),
            continuation_task_id="task-probe", base_revision=1,
        )
