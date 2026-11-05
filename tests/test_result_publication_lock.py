"""A route and a concurrent road event must have a definite commit order."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from uuid import uuid4

import pytest

from agents import route_state
from agents.route_state import build_route_state
from spatial import road_conditions
from storage import database
from storage.task_repository import claim_next_run, create_conversation, create_task_run, finish_run, get_run


pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_road_event_update_cannot_interleave_between_snapshot_and_route_commit(tmp_path, monkeypatch):
    event_file = tmp_path / "road_conditions.json"
    event_file.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(event_file))
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(event_file))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    monkeypatch.setattr(route_state, "current_data_version", lambda: "data-test")

    database.initialize(None)
    owner, _, conversation = create_conversation(None)
    queued = create_task_run(None, owner, conversation, "从总图到樱顶", str(uuid4()),
                             datetime.now(timezone.utc) + timedelta(minutes=2))
    assert claim_next_run(None, "publication-test", 30)["run_id"] == queued["run_id"]
    state = build_route_state(
        route_id="route-before-event", route_kind="direct", original_query="从总图到樱顶",
        start={"type": "poi", "name": "总图"}, end={"type": "poi", "name": "樱顶"},
        travel_mode="walk", hard_constraints={}, strategy={"name": "shortest"},
        data_version="data-test", road_condition_version=route_state.current_road_condition_version([]),
    )
    result = {"task_type": "path_planning", "response_kind": "route",
              "explanation": "路线已生成。", "route_state": state}

    before_commit, allow_commit, writer_started, writer_finished = (Event() for _ in range(4))
    real_connect = database.connect
    errors, outcomes = [], []

    @contextmanager
    def delayed_connect(url):
        with real_connect(url) as conn:
            yield conn
            before_commit.set()
            if not allow_commit.wait(10):
                raise TimeoutError("test did not release the route transaction")

    monkeypatch.setattr(database, "connect", delayed_connect)

    def publish_route():
        try:
            outcomes.append(finish_run(None, queued["run_id"], "publication-test", status="completed", result=result))
        except Exception as exc:
            errors.append(exc)

    def update_event():
        try:
            writer_started.set()
            road_conditions.add_condition("closure", "并发测试封闭", {
                "u": 1, "v": 2, "key": 0, "snap_lng_gcj": 114.35, "snap_lat_gcj": 30.54,
            })
        except Exception as exc:
            errors.append(exc)
        finally:
            writer_finished.set()

    publish = Thread(target=publish_route, daemon=True)
    writer = Thread(target=update_event, daemon=True)
    publish.start()
    try:
        assert before_commit.wait(10), "route publication did not reach the pending commit"
        writer.start()
        assert writer_started.wait(5)
        event_changed_before_commit = writer_finished.wait(0.5)
    finally:
        allow_commit.set()
        publish.join(10)
        if writer.ident is not None:
            writer.join(10)

    assert not publish.is_alive() and not writer.is_alive()
    assert not errors
    assert not event_changed_before_commit, "road event changed before the checked route committed"
    assert outcomes == [True]
    monkeypatch.setattr(database, "connect", real_connect)
    assert get_run(None, owner, queued["run_id"])["status"] == "completed"
    assert len(road_conditions.list_conditions(strict=True)) == 1


def test_publication_lock_blocks_a_road_event_writer_in_another_process(tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time

    event_file = tmp_path / "road_conditions.json"
    event_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(event_file))
    environment = os.environ.copy()
    environment["ROAD_CONDITIONS_FILE"] = str(event_file)
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    code = """
from spatial.road_conditions import add_condition
print('ready', flush=True)
add_condition('closure', 'cross-process-test', {
    'u': 1, 'v': 2, 'key': 0, 'snap_lng_gcj': 114.35, 'snap_lat_gcj': 30.54,
})
"""
    child = None
    try:
        with road_conditions.condition_publication_lock():
            child = subprocess.Popen(
                [sys.executable, "-u", "-c", code], cwd=Path(__file__).resolve().parents[1],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            assert child.stdout.readline().strip() == "ready"
            time.sleep(0.5)
            assert child.poll() is None
            assert json.loads(event_file.read_text(encoding="utf-8")) == []
    finally:
        if child is not None:
            try:
                _, stderr = child.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=5)
                raise
    assert child.returncode == 0, stderr
    assert len(json.loads(event_file.read_text(encoding="utf-8"))) == 1
