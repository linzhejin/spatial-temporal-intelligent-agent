"""Isolated API/worker check for a route with a selectable next destination."""
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

from tests.test_vision_condition_feedback import _call, _message

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_route_candidate_selection_preserves_complete_itinerary(monkeypatch, tmp_path):
    from app import create_app
    from agents import planner, tools
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run
    from spatial import network
    from storage import database

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", "fixture-only")
    events = tmp_path / "conditions.json"
    events.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(events))
    cache = tmp_path / "graph.graphml"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "data/whu_road_network.graphml", cache)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(network, "_cache_path", lambda: str(cache))
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("network requested"))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {})
    network.load_or_download_network()
    calls = [
        _call("plan_route", {"start": {"type": "poi", "name": "总图"},
                             "end": {"type": "poi", "name": "樱顶"}, "mode": "walk",
                             "constraints": {"avoid_steps": True}, "strategy": "shortest"}),
        _call("search_poi_candidates", {"start": {"type": "poi", "name": "总图"},
                                       "activities": ["canteen_meal"], "mode": "walk"}),
    ]
    replies = [_message(calls=calls), _message(content="首段和附近候选。")]
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: replies.pop(0))))
    monkeypatch.setattr(planner, "_make_client", lambda: fake_client)

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        assert conn.execute("SELECT count(*) AS n FROM run WHERE status = 'queued'").fetchone()["n"] == 0
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    conversation = client.post("/api/conversations", json={}).get_json()["data"]["conversation_id"]
    first = client.post(f"/api/conversations/{conversation}/messages", json={
        "query": "从总图到樱顶，然后去附近食堂，不走台阶", "request_id": str(uuid.uuid4()),
    })
    assert first.status_code == 202
    initial = first.get_json()["data"]
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        workflow = build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)
        outcome = process_next_run(DATABASE_URL, "candidate-api-isolated", workflow)
        assert outcome["run_id"] == initial["run_id"] and outcome["status"] == "needs_input"
        restored = client.get(f"/api/runs/{initial['run_id']}?conversation_id={conversation}").get_json()["data"]
        result = restored["result"]
        assert result["place_request"]["start"]["poi_id"] == "poi_161"
        assert result["place_request"]["constraints"]["avoid_steps"] is True
        assert result["candidates"] and result["clarify"]
        choice = result["candidates"][0]
        second = client.post(f"/api/conversations/{conversation}/messages", json={
            "query": choice["name"], "request_id": str(uuid.uuid4()), "selected_poi_id": choice["poi_id"],
            "continuation_task_id": initial["task_id"], "base_revision": restored["current_task_revision"],
        })
        assert second.status_code == 202, second.get_json()
        continuation = second.get_json()["data"]
        final = process_next_run(DATABASE_URL, "candidate-api-isolated", workflow)
        assert final["run_id"] == continuation["run_id"] and final["status"] == "completed"
        result = client.get(f"/api/runs/{continuation['run_id']}?conversation_id={conversation}").get_json()["data"]["result"]
        assert result["route_state"]["start"]["name"] in {"总图", "武汉大学图书馆(总馆)"}
        assert result["route_state"]["end"]["name"] == choice["name"]
        assert result["route_state"]["hard_constraints"]["avoid_steps"] is True
