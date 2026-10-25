"""Exercise the replay adapter through the real API and isolated task store."""

import hashlib
import json
from urllib.parse import urlsplit

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

from experiments.scenario_live_adapter import build_poi_lookup, run_case
from storage import database


DATABASE_URL = None  # assigned after the isolated-test fixture verifies its target
pytestmark = pytest.mark.usefixtures("isolated_test_database")


class _Response:
    def __init__(self, flask_response):
        self.response = flask_response

    def raise_for_status(self):
        if self.response.status_code >= 400:
            raise RuntimeError(self.response.get_json())

    def json(self):
        return self.response.get_json()


def test_local_replay_survives_real_api_persistence_and_worker_continuation(monkeypatch):
    from agents.workflow import build_agent_workflow
    from app import create_app
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="test-secret", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    seen = []

    def fake_agent(query, *, context, **_kwargs):
        seen.append((query, context.get("place_request")))
        if query == "我想打球":
            return {"response_kind": "clarify", "message": "想打什么球？",
                    "clarify": {"question": "想打什么球？", "options": ["篮球"]},
                    "place_request": {"request_id": "sport-1", "original_query": query,
                                      "pending_slot": "activity"}}
        return {"response_kind": "candidates", "message": "找到一个地点",
                "candidates": [{"poi_id": "poi_002", "name": "桂园操场"}],
                "place_request": {"request_id": "sport-1", "original_query": "我想打球",
                                  "activities": ["basketball"], "pending_slot": "selection"}}

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)

        class LocalSession:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def post(self, url, json, timeout):
                path = urlsplit(url).path
                response = client.post(path, json=json)
                if path.endswith("/messages") and response.status_code == 202:
                    process_next_run(DATABASE_URL, "replay-worker", graph)
                return _Response(response)

            def get(self, url, params=None, timeout=10):
                return _Response(client.get(urlsplit(url).path, query_string=params))

        turns, _ = run_case({"messages": [
            {"role": "user", "content": "我想打球"},
            {"role": "user", "content": "篮球"},
        ]}, LocalSession(), "http://127.0.0.1:55440",
            build_poi_lookup([{"id": "poi_002", "name": "桂园操场"}]),
            poll_interval_s=0)
    assert [turn["kind"] for turn in turns] == ["clarify", "candidates"]
    assert turns[0]["clarification_slot"] == "activity"
    assert turns[1]["candidate_poi_ids"] == ["poi_002"]
    assert seen[1][1]["request_id"] == "sport-1"


def test_local_replay_readiness_only_certifies_an_opted_in_isolated_database(monkeypatch):
    from app import create_app

    database.initialize(DATABASE_URL)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)

    response = app.test_client().get("/api/research/replay-readiness")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ready", "service": "local_development",
        "database_isolated": True,
    }


def test_local_replay_readiness_does_not_disclose_or_certify_business_database(monkeypatch):
    from app import create_app

    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL="postgresql://localhost/business")

    response = app.test_client().get("/api/research/replay-readiness")

    assert response.status_code == 503
    assert response.get_json() == {"status": "blocked", "database_isolated": False}


def test_local_replay_readiness_is_disabled_in_production(monkeypatch):
    from app import create_app

    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "x" * 48)
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)

    response = app.test_client().get("/api/research/replay-readiness")

    assert response.status_code == 404


def test_event_replay_readiness_certifies_db_file_and_frozen_time(tmp_path, monkeypatch):
    from app import create_app
    from experiments import event_replay_guard as guard
    from spatial import road_conditions as rc

    database.initialize(DATABASE_URL)
    root = tmp_path / "event_runs"
    root.mkdir()
    fixture = root / "road_conditions.json"
    fixture.write_text(json.dumps([{
        "id": "simulated_1", "type": "closure", "action": "closure",
        "affected_modes": ["walk"], "edge": {"u": 1, "v": 2, "key": 0},
        "start_time": 100, "end_time": 200,
        "source": {"kind": "research_simulated",
                   "scope": "synthetic_development_only", "campus_truth": False},
    }]), encoding="utf-8")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    monkeypatch.setattr(guard, "REPLAY_ROOT", root)
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(fixture))
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(fixture))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("AGENT_TOOL_TRACE_ENABLED", "1")
    monkeypatch.setenv("AGENT_REPLAY_EVENT_EPOCH", "150")
    monkeypatch.setenv("AGENT_REPLAY_EVENT_SHA256", digest)
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)

    response = app.test_client().get("/api/research/event-replay-readiness")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ready", "service": "local_development",
        "database_isolated": True, "road_conditions_isolated": True,
        "as_of_epoch": 150, "fixture_sha256": digest,
    }


def test_event_replay_readiness_rejects_default_road_condition_file(monkeypatch):
    from app import create_app

    database.initialize(DATABASE_URL)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.delenv("AGENT_REPLAY_EVENT_EPOCH", raising=False)
    monkeypatch.delenv("AGENT_REPLAY_EVENT_SHA256", raising=False)
    monkeypatch.delenv("ROAD_CONDITIONS_FILE", raising=False)
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)

    response = app.test_client().get("/api/research/event-replay-readiness")

    assert response.status_code == 503
    assert response.get_json() == {
        "status": "blocked", "database_isolated": True,
        "road_conditions_isolated": False,
    }
