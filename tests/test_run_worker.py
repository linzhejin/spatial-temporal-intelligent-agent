import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.postgres import PostgresSaver

from storage import database
from storage.task_repository import create_conversation, create_task_run, get_run, get_task


DATABASE_URL = None  # assigned only after isolated_test_database validates the target
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def test_worker_executes_langgraph_and_persists_result_and_progress():
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "介绍一下樱顶", "worker-test",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )

    def fake_agent(query, **kwargs):
        assert query == "介绍一下樱顶"
        return {"response_kind": "chat", "message": "樱顶在珞珈山上。", "turns": 1}

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)
        result = process_next_run(DATABASE_URL, "worker-test", graph, lease_seconds=30)

    assert result["run_id"] == queued["run_id"]
    assert result["status"] == "completed"
    restored = get_run(DATABASE_URL, owner, queued["run_id"])
    assert restored["result"]["reply"] == "樱顶在珞珈山上。"
    assert [event["event_type"] for event in restored["events"]][-1] == "run_completed"


def test_worker_persists_needs_input_as_a_resumable_task_state():
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "帶我去圖書館", "clarify-worker-test",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )

    def fake_agent(query, **kwargs):
        return {
            "response_kind": "clarify", "message": "你想從哪裡出發？",
            "clarify": {"question": "你想從哪裡出發？", "options": []},
        }

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)
        result = process_next_run(DATABASE_URL, "worker-clarify", graph, lease_seconds=30)

    assert result["status"] == "needs_input"
    task = get_task(DATABASE_URL, owner, queued["task_id"])
    run = get_run(DATABASE_URL, owner, queued["run_id"])
    assert task["status"] == "needs_input"
    assert task["state"]["pending_questions"] == [run["result"]["message"]]


def test_composite_route_and_weather_requirements_execute_independently(monkeypatch):
    from agents import workflow
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    query = "从玉兰2门去图书馆，顺便查天气"
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, query, "composite-worker-test",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    rendezvous = threading.Barrier(2)

    def weather():
        rendezvous.wait(timeout=2)
        return {"weather": "多云", "temperature": 23, "windpower": "2级", "advice": "适合步行"}

    def fake_agent(query, **kwargs):
        assert query == "从玉兰2门去图书馆，顺便查天气"
        rendezvous.wait(timeout=2)
        return {"response_kind": "chat", "message": "路线需求已识别。"}

    monkeypatch.setattr(workflow, "_run_weather_tool", weather)
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)
        result = process_next_run(DATABASE_URL, "worker-composite", graph, lease_seconds=30)

    assert result["status"] == "partial"
    run = get_run(DATABASE_URL, owner, queued["run_id"])
    task = get_task(DATABASE_URL, owner, queued["task_id"])
    assert run["result"]["weather"]["temperature"] == 23
    assert "路线暂时无法规划" in run["result"]["reply"]
    assert "多云" in run["result"]["reply"]
    assert {r["requirement_id"]: r["status"] for r in task["state"]["requirements"]} == {
        "route": "failed", "weather": "satisfied",
    }


@pytest.mark.parametrize("agent_response", [
    {"response_kind": "chat", "message": "路线核验没有通过，请调整条件。"},
    {"response_kind": "route", "message": "已规划", "route": {
        "recommended": [[114.0, 30.0], [114.1, 30.1]],
        "recommended_edge_ids": [[1, 2, 0]],
    }},
])
def test_unproven_route_reply_does_not_satisfy_route_requirement(agent_response):
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "从玉兰2门到樱顶，走平坦路线", "route-chat-failure",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )

    def fake_agent(_query, **_kwargs):
        return agent_response

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)
        outcome = process_next_run(DATABASE_URL, "worker-route-chat", graph)

    run = get_run(DATABASE_URL, owner, queued["run_id"])
    assert outcome["status"] == "failed"
    assert run["result"]["requirement_results"]["route"] == "failed"
    assert run["result"]["response_kind"] == "chat"
    assert run["result"]["task_type"] == "chat"


def test_weather_only_run_skips_route_agent(monkeypatch):
    from agents import workflow
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, "查一下武大今天的天气", "weather-worker-test",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )

    def unexpected_agent(*_args, **_kwargs):
        raise AssertionError("weather-only request must not call the route agent")

    monkeypatch.setattr(workflow, "_run_weather_tool", lambda: {
        "weather": "晴", "temperature": 25, "windpower": "1级", "advice": "注意防晒",
    })
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=unexpected_agent, database_url=DATABASE_URL)
        result = process_next_run(DATABASE_URL, "worker-weather", graph, lease_seconds=30)

    run = get_run(DATABASE_URL, owner, queued["run_id"])
    task = get_task(DATABASE_URL, owner, queued["task_id"])
    assert result["status"] == "completed"
    assert run["result"]["reply"].startswith("当前天气：晴")
    assert task["task_kind"] == "information"
    assert task["state"]["requirements"][0]["status"] == "satisfied"


def test_weather_plus_poi_request_reaches_full_tool_agent(monkeypatch):
    from agents import workflow
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    query = "今天下雨吗，顺便推荐附近的食堂"
    queued = create_task_run(
        DATABASE_URL, owner, conversation_id, query, "weather-poi-worker-test",
        datetime.now(timezone.utc) + timedelta(minutes=2),
    )
    seen = []

    def fake_agent(agent_query, **_kwargs):
        seen.append(agent_query)
        return {"response_kind": "chat", "message": "已同时处理天气和食堂推荐。"}

    def unexpected_weather_shortcut():
        raise AssertionError("mixed information asks must not be swallowed by the weather shortcut")

    monkeypatch.setattr(workflow, "_run_weather_tool", unexpected_weather_shortcut)
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)
        result = process_next_run(DATABASE_URL, "worker-weather-poi", graph, lease_seconds=30)

    assert result["status"] == "completed"
    assert seen == [query]
    run = get_run(DATABASE_URL, owner, queued["run_id"])
    assert run["result"]["reply"] == "已同时处理天气和食堂推荐。"
    task = get_task(DATABASE_URL, owner, queued["task_id"])
    assert task["state"]["requirements"][0]["status"] == "satisfied"


def test_two_agent_runs_can_finish_concurrently_without_crossing_results():
    from agents.workflow import build_agent_workflow
    from jobs.worker import process_next_run

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE run_event, run, task_history, task, conversation_message, profile, conversation CASCADE")
    owner, _, conversation_id = create_conversation(DATABASE_URL)
    queued = [
        create_task_run(
            DATABASE_URL, owner, conversation_id, query, "parallel-" + str(index),
            datetime.now(timezone.utc) + timedelta(minutes=2),
        )
        for index, query in enumerate(("介绍樱顶", "介绍卓尔体育馆"))
    ]
    rendezvous = threading.Barrier(2)

    def fake_agent(query, **kwargs):
        rendezvous.wait(timeout=3)
        return {"response_kind": "chat", "message": "答复：" + query}

    def process(index):
        with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
            saver.setup()
            graph = build_agent_workflow(saver, agent=fake_agent, database_url=DATABASE_URL)
            return process_next_run(DATABASE_URL, "parallel-worker-" + str(index), graph, lease_seconds=30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(process, range(2)))
    assert {item["status"] for item in outcomes} == {"completed"}
    by_run = {item["run_id"]: get_run(DATABASE_URL, owner, item["run_id"]) for item in queued}
    assert by_run[queued[0]["run_id"]]["result"]["reply"] == "答复：介绍樱顶"
    assert by_run[queued[1]["run_id"]]["result"]["reply"] == "答复：介绍卓尔体育馆"
