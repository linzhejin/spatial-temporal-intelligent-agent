"""Three-turn mixed itinerary with real API, worker, database and road graph."""
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from tests.test_vision_condition_feedback import _call, _message

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_mixed_named_and_fuzzy_stops_survive_api_and_worker(monkeypatch, tmp_path):
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
    replies = [_message(calls=[_call("plan_route", {
        "start": {"type": "poi", "name": "总图"}, "end": {"type": "poi", "name": "樱顶"},
        "mode": "walk", "constraints": {"avoid_steps": True}, "strategy": "shortest",
    })]), _message(content="首段路线已生成。")]
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: replies.pop(0))))
    monkeypatch.setattr(planner, "_make_client", lambda: fake_client)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        assert conn.execute("SELECT count(*) AS n FROM run WHERE status='queued'").fetchone()["n"] == 0
    app = create_app()
    app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    conversation = client.post("/api/conversations", json={}).get_json()["data"]["conversation_id"]

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        workflow = build_agent_workflow(saver, agent=planner.run_agent, database_url=DATABASE_URL)

        def submit(query, continuation=None, selected=None):
            body = {"query": query, "request_id": str(uuid.uuid4())}
            if continuation:
                body.update({"continuation_task_id": continuation["task_id"],
                             "base_revision": continuation["current_task_revision"],
                             "selected_poi_id": selected})
            response = client.post(f"/api/conversations/{conversation}/messages", json=body)
            assert response.status_code == 202, response.get_json()
            run = response.get_json()["data"]
            outcome = process_next_run(DATABASE_URL, "mixed-api-isolated", workflow)
            assert outcome and outcome["run_id"] == run["run_id"]
            restored = client.get(f"/api/runs/{run['run_id']}?conversation_id={conversation}").get_json()["data"]
            return restored

        first = submit("从总图到樱顶，然后去附近食堂，再去梅园操场，最后去超市，不走台阶")
        assert first["status"] == "needs_input", first.get("result")
        meal = first["result"]["candidates"][0]
        assert first["result"]["place_request"]["follow_up_requests"][0]["destination_poi_id"] == "poi_082"
        real_execute = tools.execute_tool

        def failed_route(name, args, context):
            if name == "plan_route":
                return {"error": "route_unavailable", "message": "路线服务暂不可用"}, None
            return real_execute(name, args, context)

        monkeypatch.setattr(tools, "execute_tool", failed_route)
        failed = submit(meal["name"], first, meal["poi_id"])
        assert failed["status"] == "needs_input", failed.get("result")
        assert failed["result"]["place_request"]["selected_poi_id"] == meal["poi_id"]
        assert failed["result"]["place_request"]["follow_up_requests"][0]["destination_poi_id"] == "poi_082"
        monkeypatch.setattr(tools, "execute_tool", real_execute)
        second = submit("重试", failed)
        assert second["status"] == "needs_input", second.get("result")
        assert second["result"]["place_request"]["start"]["poi_id"] == "poi_082"
        shop = second["result"]["candidates"][0]
        final = submit(shop["name"], second, shop["poi_id"])
        assert final["status"] == "completed", final.get("result")
        route = final["result"]
        assert [leg["end_name"] for leg in route["legs"]] == [
            "武汉大学老斋舍", meal["name"], "武汉大学梅园操场", shop["name"],
        ]
        assert route["route_state"]["hard_constraints"]["avoid_steps"] is True
