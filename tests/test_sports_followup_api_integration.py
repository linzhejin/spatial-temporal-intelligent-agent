"""Running recommendations and named choices through API, worker, DB and road graph."""
import shutil
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_running_recommendation_and_named_choices(monkeypatch, tmp_path):
    from app import create_app
    from agents import planner, tools
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run
    from spatial import network
    from storage import database

    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    events = tmp_path / "conditions.json"
    events.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(events))
    cache = tmp_path / "graph.graphml"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "data/whu_road_network.graphml", cache)
    monkeypatch.setattr(network, "_G", None)
    monkeypatch.setattr(network, "_cache_path", lambda: str(cache))
    monkeypatch.setattr(network, "_download_network", lambda *_: pytest.fail("unexpected download"))
    monkeypatch.setattr(planner, "_make_client", lambda: pytest.fail("model not needed for this flow"))
    monkeypatch.setattr(tools, "_weather_snapshot", lambda: {})
    network.load_or_download_network()
    database.initialize(DATABASE_URL)
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    client = app.test_client()

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)

        def conversation():
            return client.post("/api/conversations", json={}).get_json()["data"]["conversation_id"]

        def submit(conv, query, previous=None):
            body = {"query": query, "request_id": str(uuid.uuid4()), "travel_mode": "walk"}
            if previous:
                body.update({"continuation_task_id": previous["task_id"],
                             "base_revision": previous["current_task_revision"]})
            response = client.post(f"/api/conversations/{conv}/messages", json=body)
            assert response.status_code == 202, response.get_json()
            run = response.get_json()["data"]
            outcome = process_next_run(DATABASE_URL, "sports-followup-isolated", graph)
            assert outcome and outcome["run_id"] == run["run_id"]
            return client.get(f"/api/runs/{run['run_id']}?conversation_id={conv}").get_json()["data"]

        first_conv = conversation()
        first = submit(first_conv, "我在总图，想找个地方跑步")
        assert first["status"] == "needs_input", first.get("result")
        candidates = first["result"]["candidates"]
        assert candidates[0]["poi_id"] == "poi_085"
        assert all(row["poi_id"] != "poi_082" for row in candidates)
        chosen = submit(first_conv, "去桂园操场", first)
        assert chosen["status"] == "completed", chosen.get("result")
        assert chosen["result"]["route_state"]["start"]["name"] == "武汉大学图书馆(总馆)"
        assert chosen["result"]["route_state"]["end"]["poi_id"] == "poi_085"
        assert chosen["result"]["route_verification"]["status"] == "pass"

        other_conv = conversation()
        shortlist = submit(other_conv, "我在总图，推荐1个跑步场地")
        assert shortlist["status"] == "needs_input", shortlist.get("result")
        assert [item["poi_id"] for item in shortlist["result"]["candidates"]] == ["poi_085"]
        other = submit(other_conv, "去信息学部操场", shortlist)
        assert other["status"] == "completed", other.get("result")
        assert other["result"]["route_state"]["start"]["name"] == "武汉大学图书馆(总馆)"
        assert other["result"]["route_state"]["end"]["poi_id"] == "poi_260"
