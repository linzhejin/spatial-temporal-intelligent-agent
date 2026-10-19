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
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    return test_app.test_client()


def make_conversation(client):
    data = client.post("/api/conversations", json={}).get_json()["data"]
    return data["conversation_id"]


def test_message_endpoint_returns_independent_idempotent_run_and_poll_status(client):
    conversation_id = make_conversation(client)
    body = {"query": "从玉兰2门去星湖园食堂，别走楼梯", "request_id": "message-1"}
    first = client.post(f"/api/conversations/{conversation_id}/messages", json=body)
    assert first.status_code == 202
    payload = first.get_json()["data"]
    assert payload["status"] == "queued"
    assert payload["task_id"] and payload["run_id"]

    retry = client.post(f"/api/conversations/{conversation_id}/messages", json=body)
    assert retry.status_code == 202
    assert retry.get_json()["data"]["run_id"] == payload["run_id"]

    polled = client.get(f"/api/runs/{payload['run_id']}?conversation_id={conversation_id}")
    assert polled.status_code == 200
    assert polled.get_json()["data"]["events"] == []


def test_run_status_and_cancel_are_owner_scoped(client):
    conversation_id = make_conversation(client)
    response = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "查校园天气", "request_id": "message-2",
    })
    run_id = response.get_json()["data"]["run_id"]
    foreign = client.application.test_client()
    assert foreign.get(f"/api/runs/{run_id}?conversation_id={conversation_id}").status_code == 404
    cancelled = client.post(f"/api/runs/{run_id}/cancel", json={"conversation_id": conversation_id})
    assert cancelled.status_code == 200
    assert cancelled.get_json()["data"]["status"] == "cancelled"
    assert client.get(f"/api/runs/{run_id}?conversation_id={conversation_id}").get_json()["data"]["status"] == "cancelled"


def test_retry_key_cannot_be_reused_for_different_user_message(client):
    conversation_id = make_conversation(client)
    client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "查天气", "request_id": "message-3",
    })
    conflict = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "规划路线", "request_id": "message-3",
    })
    assert conflict.status_code == 409


def test_message_endpoint_persists_travel_mode_and_map_coordinates(client):
    from storage.task_repository import get_run_input

    conversation_id = make_conversation(client)
    response = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "从地图起点去地图终点", "request_id": "map-points-1", "travel_mode": "bike",
        "coord_start": {"lng": 114.3601, "lat": 30.5299, "name": "地图起点"},
        "coord_end": {"lng": 114.3612, "lat": 30.5302, "name": "地图终点"},
        "coord_waypoints": [{"lng": 114.3607, "lat": 30.5300}],
    })
    assert response.status_code == 202
    run_id = response.get_json()["data"]["run_id"]
    payload = get_run_input(DATABASE_URL, run_id)
    spec = payload["task"]["route_spec"]
    assert spec["travel_mode"] == "bike"
    assert spec["route_kind"] == "via"
    assert spec["start"]["coordinates"]["longitude"] == pytest.approx(114.3601)
    assert spec["end"]["coordinates"]["latitude"] == pytest.approx(30.5302)
    assert spec["stops"][0]["coordinates"]["crs"] == "WGS84"


def test_message_endpoint_rejects_unknown_or_invalid_request_fields(client):
    conversation_id = make_conversation(client)
    bad_mode = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "路线", "request_id": "bad-mode", "travel_mode": "teleport",
    })
    assert bad_mode.status_code == 422
    bad_crs = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "路线", "request_id": "bad-crs", "coord_start": {"lng": 114, "lat": 30, "crs": "unknown"},
    })
    assert bad_crs.status_code == 422


def test_message_endpoint_resumes_only_the_owned_pending_clarification(client):
    from storage.task_repository import claim_next_run, finish_run

    conversation_id = make_conversation(client)
    first = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "带我去图书馆", "request_id": "clarify-run-1",
    })
    first_data = first.get_json()["data"]
    claim_next_run(DATABASE_URL, "worker", 30)
    assert finish_run(DATABASE_URL, first_data["run_id"], "worker", status="needs_input", result={
        "task_type": "unknown", "message": "从哪里出发？",
        "clarify": {"question": "从哪里出发？", "options": []},
    })

    continued = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "从樱顶出发", "request_id": "clarify-run-2",
        "continuation_task_id": first_data["task_id"], "base_revision": 1,
    })
    assert continued.status_code == 202
    assert continued.get_json()["data"]["task_id"] == first_data["task_id"]
    assert continued.get_json()["data"]["task_revision"] == 2

    stale = client.post(f"/api/conversations/{conversation_id}/messages", json={
        "query": "另一个出发地", "request_id": "clarify-run-stale",
        "continuation_task_id": first_data["task_id"], "base_revision": 1,
    })
    assert stale.status_code == 409
    assert stale.get_json()["error"] == "task_revision_conflict"
