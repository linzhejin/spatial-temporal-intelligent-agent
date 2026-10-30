from datetime import datetime, timedelta, timezone

import pytest

from storage import database


DATABASE_URL = None  # assigned only after isolated_test_database validates the target
pytestmark = pytest.mark.usefixtures("isolated_test_database")


@pytest.fixture
def client(monkeypatch):
    # app.py creates an app at import time and immediately migrates DATABASE_URL.
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    import app as app_module

    database.initialize(DATABASE_URL)
    test_app = app_module.create_app()
    test_app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, profile, conversation CASCADE")
    return test_app.test_client()


def test_conversation_cookie_restores_task_after_client_restart(client):
    created = client.post("/api/conversations", json={}).get_json()["data"]
    conversation_id = created["conversation_id"]
    assert "access_token" not in created
    task = client.post(f"/api/conversations/{conversation_id}/tasks", json={
        "task_kind": "route",
        "route_spec": {
            "start": {"poi_id": "gate-yulan-2", "name": "玉兰2门"},
            "end": {"poi_id": "poi-xinghu-canteen", "name": "星湖园食堂"},
            "strategy": {"name": "shortest", "source": "commute_default",
                         "task_class": "commute", "weights": {"distance": 1, "slope": 0, "scenery": 0}},
        },
    })
    assert task.status_code == 201
    task_id = task.get_json()["data"]["task_id"]

    # A fresh client with only the persisted cookie can restore the task.
    fresh = client.application.test_client()
    fresh.set_cookie("whu_conversation_token", client.get_cookie("whu_conversation_token").value)
    restored = fresh.get(f"/api/conversations/{conversation_id}")
    assert restored.status_code == 200
    assert restored.get_json()["data"]["tasks"][0]["task_id"] == task_id


def test_conversation_cookie_respects_production_secure_setting(client):
    client.application.config["SESSION_COOKIE_SECURE"] = True
    response = client.post("/api/conversations", json={})
    assert response.status_code == 201
    cookie = client.get_cookie("whu_conversation_token")
    assert cookie is not None and cookie.secure is True


def test_foreign_or_missing_credential_cannot_restore_or_patch(client):
    created = client.post("/api/conversations", json={}).get_json()["data"]
    conversation_id = created["conversation_id"]
    task = client.post(f"/api/conversations/{conversation_id}/tasks", json={}).get_json()["data"]
    task_id = task["task_id"]

    foreign = client.application.test_client()
    assert foreign.get(f"/api/conversations/{conversation_id}").status_code == 404

    patch = {
        "task_id": task_id, "base_revision": 0, "source_message_id": "m1",
        "operations": [{"op": "set", "path": "route_spec.travel_mode", "value": "bike"}],
    }
    denied = foreign.patch(f"/api/conversations/{conversation_id}/tasks/{task_id}", json=patch)
    assert denied.status_code == 404


def test_task_patch_returns_conflict_when_base_revision_is_stale(client):
    created = client.post("/api/conversations", json={}).get_json()["data"]
    conversation_id = created["conversation_id"]
    task_id = client.post(f"/api/conversations/{conversation_id}/tasks", json={}).get_json()["data"]["task_id"]
    patch = {
        "task_id": task_id, "base_revision": 0, "source_message_id": "m1",
        "operations": [{"op": "set", "path": "route_spec.travel_mode", "value": "bike"}],
    }
    assert client.patch(f"/api/conversations/{conversation_id}/tasks/{task_id}", json=patch).status_code == 200
    assert client.patch(f"/api/conversations/{conversation_id}/tasks/{task_id}", json=patch).status_code == 409


def test_restore_returns_server_transcript_after_browser_state_is_lost(client):
    from api.conversations import COOKIE_NAME
    from storage.task_repository import claim_next_run, create_task_run, finish_run, get_conversation

    conversation_id = client.post("/api/conversations", json={}).get_json()["data"]["conversation_id"]
    token = client.get_cookie(COOKIE_NAME).value
    owner = get_conversation(DATABASE_URL, conversation_id, token)["owner_id"]
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "介绍樱顶", "restore-transcript",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    claim_next_run(DATABASE_URL, "worker", 30)
    assert finish_run(DATABASE_URL, queued["run_id"], "worker", status="completed",
                      result={"task_type": "chat", "reply": "樱顶在珞珈山。"})
    restored = client.get(f"/api/conversations/{conversation_id}")
    assert restored.status_code == 200
    assert [(m["role"], m["content"]) for m in restored.get_json()["data"]["messages"]] == [
        ("user", "介绍樱顶"), ("assistant", "樱顶在珞珈山。"),
    ]


def test_restore_includes_active_runs_for_client_resume(client):
    from api.conversations import COOKIE_NAME
    from storage.task_repository import cancel_run, create_task_run, get_conversation

    conversation_id = client.post("/api/conversations", json={}).get_json()["data"]["conversation_id"]
    owner = get_conversation(DATABASE_URL, conversation_id,
                             client.get_cookie(COOKIE_NAME).value)["owner_id"]
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "恢复中的路线", "restore-active-run",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )

    try:
        restored = client.get(f"/api/conversations/{conversation_id}")
        assert restored.status_code == 200
        active = restored.get_json()["data"]["active_runs"]
        assert len(active) == 1
        assert active[0]["run_id"] == queued["run_id"]
        assert active[0]["task_id"] == queued["task_id"]
        assert active[0]["status"] == "queued"
        assert active[0]["task_revision"] == queued["task_revision"]
        assert active[0]["query"] == "恢复中的路线"
        assert active[0]["deadline_at"]
    finally:
        # This deliberately queued run must not leak into subsequent workers.
        cancelled = cancel_run(DATABASE_URL, owner, queued["run_id"])
    assert cancelled is not None and cancelled["status"] == "cancelled"
