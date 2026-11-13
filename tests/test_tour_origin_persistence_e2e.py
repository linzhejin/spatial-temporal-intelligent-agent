"""Optional real-model check of two turns through an isolated durable worker."""

import os
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import dotenv_values

DATABASE_URL = None
pytestmark = [
    pytest.mark.usefixtures("isolated_test_database"),
    pytest.mark.skipif(os.environ.get("RUN_TOUR_REAL_MODEL_E2E") != "1",
                       reason="explicit real-model opt-in required"),
]


def test_real_model_tour_origin_survives_task_persistence(monkeypatch, tmp_path):
    credential_file = os.environ.get("LIVE_DEEPSEEK_ENV")
    if not credential_file:
        pytest.skip("explicit model credential file required")
    credentials = dotenv_values(credential_file)
    key = credentials.get("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("model credential unavailable")

    from tests.db_safety import verify_isolated_test_database

    monkeypatch.delenv("DATABASE_URL", raising=False)
    verify_isolated_test_database(DATABASE_URL, business_urls=(credentials.get("DATABASE_URL"),))
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AGENT_DIALOGUE_MODE", "model")
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    conditions = tmp_path / "conditions.json"
    conditions.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(conditions))

    from app import create_app
    from agents import planner, tools
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run
    from langgraph.checkpoint.postgres import PostgresSaver
    from spatial import network
    from storage import database

    monkeypatch.setattr(planner.config, "DEEPSEEK_API_KEY", key)
    monkeypatch.setattr(planner.config, "LLM_MODEL", "deepseek-chat")
    monkeypatch.setattr(tools, "list_conditions", lambda **_kwargs: [])
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: None)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(network, "_cache_path", lambda: str(
        Path(__file__).resolve().parents[1] / "data/whu_road_network.graphml"))
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("unexpected map download"))
    database.initialize(DATABASE_URL)
    network.load_or_download_network()
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    created = client.post("/api/conversations", json={})
    assert created.status_code == 201
    conversation = created.get_json()["data"]["conversation_id"]

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        workflow = build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)

        def turn(query):
            response = client.post(f"/api/conversations/{conversation}/messages", json={
                "query": query, "request_id": str(uuid4()), "travel_mode": "walk",
            })
            assert response.status_code == 202
            run_id = response.get_json()["data"]["run_id"]
            outcome = process_next_run(DATABASE_URL, "isolated-tour-origin-worker", workflow)
            assert outcome and outcome["run_id"] == run_id
            record = client.get(f"/api/runs/{run_id}?conversation_id={conversation}")
            assert record.status_code == 200
            return record.get_json()["data"]

        first = turn("带朋友去老图书馆和凌波门逛逛")
        assert first["status"] == "needs_input", first.get("result")
        result = first["result"]
        assert result.get("response_kind") == "clarify"
        assert not result.get("recommended")
        pending = result["place_request"]
        assert pending["pending_slot"] == "start"
        assert not pending.get("start")
        assert set(pending["semantic_request"]["tour_stops"]) == {"老图书馆", "凌波门"}

        # Restore through the API before sending an ordinary follow-up: this
        # exercises database state and model-selected continuation together.
        restored = client.get(f"/api/conversations/{conversation}")
        assert restored.status_code == 200
        assert any(task["task_id"] == first["task_id"] for task in restored.get_json()["data"]["tasks"])
        second = turn("从信息学部图书馆出发")
        assert second["status"] == "completed", second.get("result")
        assert second["task_id"] == first["task_id"]
        result = second["result"]
        assert result.get("response_kind") == "route"
        state = result.get("route_state") or {}
        assert "信息学部" in (state.get("start") or {}).get("name", "")
        via = state.get("via") or {}
        locations = [state.get("end") or {}, *(via.get("points") or []), via]
        locations.extend((state.get("tour") or {}).get("pois") or [])
        names = " ".join(str(place.get("name") or "") for place in locations)
        assert "图书馆" in names and "凌波门" in names
