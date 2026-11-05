"""Route destination corrections through the API, worker and real campus graph."""
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

from tests.test_vision_condition_feedback import _call, _message

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_route_followup_keeps_current_origin_through_real_task_persistence(monkeypatch, tmp_path):
    from app import create_app
    from agents import planner, tools, workflow as workflow_module
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
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("unexpected network download"))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {})
    monkeypatch.setattr(workflow_module, "_run_weather_tool", lambda: {"weather": "多云", "temperature": 20})
    network.load_or_download_network()
    replies = [_message(calls=[_call("plan_route", {
        "start": {"type": "poi", "name": "武汉大学图书馆(总馆)"},
        "end": {"type": "poi", "name": "武汉大学梅园操场"}, "mode": "walk",
    })]), _message(content="首条路线已规划。")]
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: replies.pop(0))))
    monkeypatch.setattr(planner, "_make_client", lambda: fake_client)
    database.initialize(DATABASE_URL)
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    conversation = client.post("/api/conversations", json={}).get_json()["data"]["conversation_id"]

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = workflow_module.build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)

        def submit(query):
            response = client.post(f"/api/conversations/{conversation}/messages", json={
                "query": query, "request_id": str(uuid.uuid4()), "travel_mode": "walk",
            })
            assert response.status_code == 202, response.get_json()
            run = response.get_json()["data"]
            outcome = process_next_run(DATABASE_URL, "route-followup-isolated", graph)
            assert outcome and outcome["run_id"] == run["run_id"]
            return client.get(f"/api/runs/{run['run_id']}?conversation_id={conversation}").get_json()["data"]

        first = submit("从武汉大学图书馆(总馆)到武汉大学梅园操场")
        assert first["status"] == "completed", first.get("result")
        assert first["result"]["route_verification"]["status"] == "pass"
        second = submit("桂园操场不行吗")
        assert second["status"] == "completed", second.get("result")
        route = second["result"]
        assert route["response_kind"] == "route"
        assert route["route_state"]["start"]["name"] == "武汉大学图书馆(总馆)"
        assert route["route_state"]["end"]["name"] == "武汉大学桂园操场"
        assert route["route_verification"]["status"] == "pass"
        assert route["recommended_edge_ids"]

        third = submit("换成樱顶吧")
        assert third["status"] == "completed", third.get("result")
        assert third["result"]["route_state"]["start"]["name"] == "武汉大学图书馆(总馆)"
        assert third["result"]["route_state"]["end"]["name"] == "武汉大学老斋舍"

        weather = submit("天气怎么样")
        assert weather["status"] == "completed", weather.get("result")
        stale = submit("桂园操场不行吗")
        assert stale["status"] == "needs_input", stale.get("result")
        assert stale["result"]["response_kind"] == "clarify"
        assert not stale["result"].get("route_state")
