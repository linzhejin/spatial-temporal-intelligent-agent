from agents.state_models import TaskState


def test_rollback_preserves_model_release_place_fields():
    state = TaskState.model_validate({
        "task_id": "task-1", "conversation_id": "conversation-1",
        "place_request": {"request_id": "request-1", "original_query": "去吃饭",
                          "search_radius_m": 2000, "selected_poi_id": "amap:TEST",
                          "selected_place": {"poi_id": "amap:TEST", "name": "候选"}},
    })
    restored = state.model_dump(mode="json")["place_request"]
    assert restored["search_radius_m"] == 2000
    assert restored["selected_place"]["poi_id"] == restored["selected_poi_id"]


def test_rollback_run_lifecycle_on_migrated_schema(isolated_test_database):
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4
    import pytest
    from storage import database
    from storage.task_repository import (claim_next_run, create_conversation, create_task_run,
                                         finish_run, get_run, get_run_input, list_active_runs)

    url = isolated_test_database
    with database.connect(url) as conn:
        migrated = conn.execute("SELECT 1 FROM information_schema.columns WHERE table_name='run'"
                                " AND column_name='model_led'").fetchone()
    if not migrated:
        pytest.skip("requires a test database already migrated by the model-led release")
    owner, _, conversation = create_conversation(url)
    run = create_task_run(url, owner, conversation, "查询天气", str(uuid4()),
                          datetime.now(timezone.utc) + timedelta(seconds=75))
    assert get_run_input(url, run["run_id"])["query"] == "查询天气"
    assert list_active_runs(url, owner, conversation)[0]["run_id"] == run["run_id"]
    assert get_run(url, owner, run["run_id"])["status"] == "queued"
    assert claim_next_run(url, "rollback-contract")["run_id"] == run["run_id"]
    assert finish_run(url, run["run_id"], "rollback-contract", status="completed", result={"message": "完成"})
    assert get_run(url, owner, run["run_id"])["status"] == "completed"
