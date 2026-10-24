"""Exercise the replay adapter through the real API and isolated task store."""

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


class _LocalSession:
    def __init__(self, client, graph):
        self.client = client
        self.graph = graph

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, url, json, timeout):
        from jobs.worker import process_next_run

        path = urlsplit(url).path
        response = self.client.post(path, json=json)
        if path.endswith("/messages") and response.status_code == 202:
            process_next_run(DATABASE_URL, "replay-worker", self.graph)
        return _Response(response)

    def get(self, url, params=None, timeout=10):
        return _Response(self.client.get(urlsplit(url).path, query_string=params))


def test_local_replay_survives_real_api_persistence_and_worker_continuation(monkeypatch):
    from agents.workflow import build_agent_workflow
    from app import create_app

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

        turns, _ = run_case({"messages": [
            {"role": "user", "content": "我想打球"},
            {"role": "user", "content": "篮球"},
        ]}, _LocalSession(client, graph), "http://127.0.0.1:55440",
            build_poi_lookup([{"id": "poi_002", "name": "桂园操场"}]),
            poll_interval_s=0)
    assert [turn["kind"] for turn in turns] == ["clarify", "candidates"]
    assert turns[0]["clarification_slot"] == "activity"
    assert turns[1]["candidate_poi_ids"] == ["poi_002"]
    assert seen[1][1]["request_id"] == "sport-1"


def test_local_replay_keeps_destination_when_origin_is_supplied_later(monkeypatch):
    from agents.workflow import build_agent_workflow
    from app import create_app

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr("agents.workflow._run_weather_tool",
                        lambda: {"weather": "晴", "temperature": 22, "windpower": "2级"})
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="test-secret", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    seen = []

    def fake_agent(query, *, context, **_kwargs):
        seen.append((query, context.get("place_request")))
        if len(seen) == 1:
            return {"response_kind": "clarify", "message": "你从哪里出发？",
                    "clarify": {"question": "你从哪里出发？"},
                    "place_request": {"request_id": "origin-1", "original_query": query,
                                      "selected_poi_id": "poi_002", "pending_slot": "start",
                                      "mode": "walk"}}
        return {"response_kind": "route", "message": "从玉兰二门到桂园操场。",
                "route": {"start_name": "玉兰二门", "end_name": "桂园操场",
                          "recommended": [[114.35, 30.53], [114.36, 30.54]],
                          "recommended_edge_ids": [[1, 2, 0]],
                          "route_verification": {"status": "pass"},
                          "route_state": {"start": {"name": "玉兰二门"},
                                          "end": {"name": "桂园操场"},
                                          "travel_mode": "walk", "strategy": {"name": "shortest"},
                                          "hard_constraints": {}}}}

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)

        turns, _ = run_case({"messages": [
            {"role": "user", "content": "带我去桂园操场，起点等我补充。"},
            {"role": "user", "content": "我在玉兰二门，查天气后再带我过去。"},
        ]}, _LocalSession(client, graph), "http://127.0.0.1:55440",
            build_poi_lookup([{"id": "poi_001", "name": "玉兰2门", "aliases": ["玉兰二门"]},
                              {"id": "poi_002", "name": "桂园操场"}]),
            poll_interval_s=0)
    assert [turn["kind"] for turn in turns] == ["clarify", "route"]
    assert turns[0]["end_poi_id"] == "poi_002"
    assert turns[0]["clarification_slot"] == "start"
    assert seen[1][1]["selected_poi_id"] == "poi_002"
    assert turns[1]["start_poi_id"] == "poi_001"
    assert turns[1]["end_poi_id"] == "poi_002"
    assert turns[1]["route_edge_ids"] == [[1, 2, 0]]
    assert turns[1]["outputs"] == ["route", "weather"]


def test_local_replay_weather_followup_does_not_replan_existing_route(monkeypatch):
    from agents.workflow import build_agent_workflow
    from app import create_app

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr("agents.workflow._run_weather_tool",
                        lambda: {"weather": "晴", "temperature": 22, "windpower": "2级"})
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="test-secret", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    agent_calls = []

    def fake_agent(query, **_kwargs):
        agent_calls.append(query)
        return {"response_kind": "route", "message": "从玉兰二门到桂园操场。",
                "route": {"start_name": "玉兰二门", "end_name": "桂园操场",
                          "recommended": [[114.35, 30.53], [114.36, 30.54]],
                          "recommended_edge_ids": [[1, 2, 0]],
                          "route_verification": {"status": "pass"},
                          "route_state": {"start": {"name": "玉兰二门"},
                                          "end": {"name": "桂园操场"},
                                          "travel_mode": "walk", "strategy": {"name": "shortest"},
                                          "hard_constraints": {}}}}

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)

        turns, _ = run_case({"messages": [
            {"role": "user", "content": "从玉兰二门到桂园操场，走最短路。"},
            {"role": "user", "content": "再告诉我天气，不要改路线。"},
        ]}, _LocalSession(client, graph), "http://127.0.0.1:55440",
            build_poi_lookup([{"id": "poi_001", "name": "玉兰2门", "aliases": ["玉兰二门"]},
                              {"id": "poi_002", "name": "桂园操场"}]),
            poll_interval_s=0)
    assert agent_calls == ["从玉兰二门到桂园操场，走最短路。"]
    assert turns[0]["kind"] == "route"
    assert turns[0]["route_edge_ids"] == [[1, 2, 0]]
    assert turns[1]["kind"] == "information"
    assert turns[1]["outputs"] == ["weather"]
    assert turns[1]["route_edge_ids"] == []


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
