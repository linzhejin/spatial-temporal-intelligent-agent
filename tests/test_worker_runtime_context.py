from concurrent.futures import ThreadPoolExecutor
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from langgraph.checkpoint.memory import InMemorySaver


def _c1_runtime_case(case_id):
    project_root = Path(__file__).resolve().parents[1]
    replay = json.loads(
        (project_root / "experiments/datasets/c1_production_context_replay_v1.json")
        .read_text(encoding="utf-8")
    )
    return next(case for case in replay["runtime_cases"] if case["id"] == case_id)


def _verified_route_response(message):
    """Small deterministic route artifact for workflow status replay tests."""
    return {
        "response_kind": "route",
        "message": message,
        "route": {
            "start_name": "起点",
            "end_name": "终点",
            "recommended": [[114.36, 30.53], [114.3601, 30.53]],
            "recommended_edge_ids": [[1, 2, 0]],
            "route_verification": {"status": "pass"},
        },
    }


def test_c1_runtime_case_manifest_matches_context_and_cancellation_regressions():
    project_root = Path(__file__).resolve().parents[1]
    replay = json.loads(
        (project_root / "experiments/datasets/c1_production_context_replay_v1.json")
        .read_text(encoding="utf-8")
    )
    runtime_cases = replay["runtime_cases"]

    assert {case["id"] for case in runtime_cases} == {
        "parallel_runs_keep_contexts_isolated",
        "lease_loss_stops_before_agent",
        "late_result_after_invalidation_is_not_published",
    }
    for case in runtime_cases:
        assert case["implemented_by"] in globals()
        assert callable(globals()[case["implemented_by"]])


def test_resumed_graph_emits_events_with_the_current_lease_worker(monkeypatch):
    from agents import workflow

    events = []
    monkeypatch.setattr(workflow, "append_run_event", lambda _url, _run, worker, *_args, **_kwargs: (
        events.append(worker) or {"stored": True}
    ))
    monkeypatch.setattr(workflow, "get_run_input", lambda _url, _run: {
        "query": "介绍樱顶", "history": [], "task": {},
    })
    graph = workflow.build_agent_workflow(
        InMemorySaver(),
        agent=lambda *_args, **_kwargs: {"response_kind": "chat", "message": "樱顶在珞珈山上。"},
        interrupt_after=["load_task"],
    )
    config = {"configurable": {"thread_id": "lease-reclaim-test"}}

    graph.invoke(
        {"run_id": "lease-reclaim-test", "worker_id": "worker-old"},
        config,
        context={"worker_id": "worker-old"},
    )
    checkpoint = graph.get_state(config)
    assert checkpoint.values["worker_id"] == "worker-old"
    assert checkpoint.next == ("execute_agent",)

    resumed = graph.invoke(None, config, context={"worker_id": "worker-new"})

    assert resumed["result"]["reply"] == "樱顶在珞珈山上。"
    assert events[:2] == ["worker-old", "worker-old"]
    assert events[2:] and set(events[2:]) == {"worker-new"}


@pytest.mark.parametrize("outcome,artifact_kind,expected_outcome,expected_artifact", [
    ("success", None, "success", None),
    ("success", "route_comparison", "success", "route_comparison"),
    ("skipped_after_route_verification_retry", None,
     "skipped_after_route_verification_retry", None),
    ("skipped_after_route_verification_failure", None,
     "skipped_after_route_verification_failure", None),
    ("unknown_outcome", None, "exception", None),
    ("success", "unknown_artifact", "success", None),
])
def test_opt_in_agent_tool_trace_records_safe_metadata_as_run_events(
    monkeypatch, outcome, artifact_kind, expected_outcome, expected_artifact,
):
    from agents import workflow

    monkeypatch.setenv("AGENT_TOOL_TRACE_ENABLED", "1")
    events = []

    def append(_url, run_id, worker_id, event_type, payload, node_id=None):
        events.append({"run_id": run_id, "worker_id": worker_id,
                       "event_type": event_type, "payload": payload,
                       "node_id": node_id})
        return {"stored": True}

    monkeypatch.setattr(workflow, "append_run_event", append)
    monkeypatch.setattr(workflow, "get_run_input", lambda _url, _run: {
        "query": "介绍樱顶", "history": [], "task": {},
    })

    def fake_agent(_query, *, tool_trace_callback=None, **_kwargs):
        assert callable(tool_trace_callback)
        tool_trace_callback({
            "tool_name": "get_place_details", "argument_keys": ["name"],
            "outcome": outcome, "artifact_kind": artifact_kind,
            "raw_arguments": {"name": "private argument"},
            "raw_result": {"message": "private result"},
        })
        return {"response_kind": "chat", "message": "樱顶在珞珈山上。"}

    result = workflow.build_agent_workflow(InMemorySaver(), agent=fake_agent).invoke(
        {"run_id": "trace-run"},
        {"configurable": {"thread_id": "trace-run"}},
        context={"worker_id": "trace-worker"},
    )

    trace_events = [event for event in events if event["event_type"].startswith("agent_tool_")]
    assert [event["event_type"] for event in trace_events] == [
        "agent_tool_trace_started", "agent_tool_call",
    ]
    assert trace_events[1]["worker_id"] == "trace-worker"
    assert trace_events[1]["payload"] == {
        "tool_name": "get_place_details", "argument_keys": ["name"],
        "outcome": expected_outcome, "artifact_kind": expected_artifact,
    }
    assert result["result"]["reply"] == "樱顶在珞珈山上。"


def test_opt_in_agent_tool_trace_records_direct_weather_execution(monkeypatch):
    from agents import workflow

    monkeypatch.setenv("AGENT_TOOL_TRACE_ENABLED", "1")
    events = []

    def append(_url, run_id, worker_id, event_type, payload, node_id=None):
        events.append({"run_id": run_id, "worker_id": worker_id,
                       "event_type": event_type, "payload": payload,
                       "node_id": node_id})
        return {"stored": True}

    monkeypatch.setattr(workflow, "append_run_event", append)
    monkeypatch.setattr(workflow, "get_run_input", lambda _url, _run: {
        "query": "查一下天气", "history": [], "task": {},
    })
    monkeypatch.setattr(workflow, "_run_weather_tool", lambda: {
        "weather": "晴", "temperature": 22,
    })

    result = workflow.build_agent_workflow(
        InMemorySaver(), agent=lambda *_args, **_kwargs: pytest.fail("weather-only request called Agent"),
    ).invoke(
        {"run_id": "weather-trace-run"},
        {"configurable": {"thread_id": "weather-trace-run"}},
        context={"worker_id": "weather-trace-worker"},
    )

    trace_events = [event for event in events if event["event_type"].startswith("agent_tool_")]
    assert [event["event_type"] for event in trace_events] == [
        "agent_tool_trace_started", "agent_tool_call",
    ]
    assert trace_events[0]["payload"] == {"source": "agent_execution"}
    assert trace_events[1]["payload"] == {
        "tool_name": "get_weather", "argument_keys": [],
        "outcome": "success", "artifact_kind": None,
    }
    assert trace_events[1]["worker_id"] == "weather-trace-worker"
    assert result["result"]["weather"]["temperature"] == 22


def test_clarification_answer_keeps_original_multitool_request(monkeypatch):
    from agents import workflow

    original = "骑车从玉兰2门到樱顶，经过卓尔体育馆，避开台阶，顺便查天气"
    observed = {}
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: {"stored": True})
    monkeypatch.setattr(workflow, "get_run_input", lambda _url, _run: {
        "query": "从信息学部出发", "task_origin_query": original,
        "task_revision": 2,
        "history": [
            {"role": "user", "content": original},
            {"role": "assistant", "content": "你想从哪里出发？"},
        ],
        "task": {"route_spec": {"travel_mode": "bike"}},
    })
    monkeypatch.setattr(workflow, "_run_weather_tool", lambda: {"weather": "晴", "temperature": 22})

    def fake_agent(query, *, context, travel_mode, **_kwargs):
        observed.update(query=query, context=context, travel_mode=travel_mode)
        return _verified_route_response("路线已规划")

    graph = workflow.build_agent_workflow(InMemorySaver(), agent=fake_agent)
    result = graph.invoke(
        {"run_id": "clarification-multitool"},
        {"configurable": {"thread_id": "clarification-multitool"}},
        context={"worker_id": "worker-a"},
    )["result"]

    assert observed["query"] == "从信息学部出发"
    assert observed["context"]["active_task_request"] == original
    assert observed["context"]["task_revision"] == 2
    assert observed["travel_mode"] == "bike"
    assert result["execution_status"] == "completed"
    assert result["requirement_results"] == {"route": "satisfied", "weather": "satisfied"}
    assert result["requirement_errors"] == {}
    assert "路线已规划" in result["explanation"]
    assert "当前天气：晴" in result["explanation"]
    assert result["weather"]["temperature"] == 22


def test_clarification_of_new_trip_does_not_import_older_completed_route(monkeypatch):
    from agents import workflow
    from agents.route_state import build_route_state

    observed = {}
    prior = json.loads((Path(__file__).resolve().parents[1]
                        / "experiments/datasets/c1_production_context_replay_v1.json")
                       .read_text(encoding="utf-8"))["prior_route_state"]
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: {"stored": True})
    monkeypatch.setattr(workflow, "get_run_input", lambda _url, _run: {
        "query": "从信息学部出发", "task_origin_query": "去卓尔体育馆，避开台阶",
        "task_id": "current-task", "task_revision": 2,
        "history": [{"role": "user", "content": "去卓尔体育馆，避开台阶"},
                    {"role": "assistant", "content": "你从哪里出发？"}],
        "task": {"route_spec": {}},
        "previous_route_state": build_route_state(**prior),
        "previous_route_state_task_id": "older-task",
    })

    def fake_agent(_query, *, context, **_kwargs):
        observed.update(context)
        return {"response_kind": "chat", "message": "继续当前任务"}

    workflow.build_agent_workflow(InMemorySaver(), agent=fake_agent).invoke(
        {"run_id": "no-cross-task-route"},
        {"configurable": {"thread_id": "no-cross-task-route"}},
        context={"worker_id": "worker-a"},
    )
    assert "previous_route_state" not in observed
    assert observed["active_task_request"] == "去卓尔体育馆，避开台阶"


def test_production_workflow_keeps_concurrent_task_contexts_isolated(monkeypatch):
    from agents import workflow
    from agents.route_state import build_route_state

    case = _c1_runtime_case("parallel_runs_keep_contexts_isolated")
    inputs = {}
    for run in case["runs"]:
        inputs[run["run_id"]] = {
            "query": run["query"],
            "history": [],
            "task_revision": run["task_revision"],
            "task": {"route_spec": {}},
            "previous_route_state": build_route_state(**run["previous_route_state"]),
        }
    monkeypatch.setattr(workflow, "get_run_input", lambda _url, run_id: inputs[run_id])
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: {"stored": True})

    rendezvous = threading.Barrier(2)
    observed = {}

    def fake_agent(query, *, context, **_kwargs):
        rendezvous.wait(timeout=3)
        observed[query] = context
        return {"response_kind": "chat", "message": f"完成：{query}"}

    graph = workflow.build_agent_workflow(InMemorySaver(), agent=fake_agent)

    def invoke(run_id):
        return graph.invoke(
            {"run_id": run_id},
            {"configurable": {"thread_id": run_id}},
            context={"worker_id": f"worker-{run_id}"},
        )

    run_ids = [run["run_id"] for run in case["runs"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, run_ids))

    for query, expected_route_id in case["expected_route_id_by_query"].items():
        context = observed[query]
        if expected_route_id is None:
            assert "previous_route_state" not in context
            assert "previous_intent" not in context
        else:
            assert context["previous_route_state"]["route_id"] == expected_route_id
    assert [result["result"]["reply"] for result in results] == [
        f"完成：{run['query']}" for run in case["runs"]
    ]


def test_production_workflow_stops_before_agent_when_run_lease_is_lost(monkeypatch):
    import pytest
    from agents import workflow

    case = _c1_runtime_case("lease_loss_stops_before_agent")
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: None)
    agent_calls = []
    graph = workflow.build_agent_workflow(
        InMemorySaver(),
        agent=lambda *_args, **_kwargs: agent_calls.append("called"),
    )

    with pytest.raises(RuntimeError, match=case["expected_error"]):
        graph.invoke(
            {"run_id": case["run_id"]},
            {"configurable": {"thread_id": case["run_id"]}},
            context={"worker_id": case["worker_id"]},
        )

    assert len(agent_calls) == case["expected_agent_calls"]


def test_worker_does_not_publish_result_after_cancellation_or_lease_loss(monkeypatch):
    from jobs import worker

    case = _c1_runtime_case("late_result_after_invalidation_is_not_published")
    heartbeat_failed = threading.Event()
    finish_calls = []
    monkeypatch.setattr(worker, "claim_next_run", lambda *_args: {
        "run_id": case["run_id"], "attempt": 1,
    })

    def lose_lease(*_args):
        heartbeat_failed.set()
        return False

    monkeypatch.setattr(worker, "heartbeat_run", lose_lease)
    monkeypatch.setattr(worker, "finish_run", lambda *_args, **_kwargs: finish_calls.append(True))

    class SlowGraph:
        def invoke(self, _value, _config, *, context):
            assert context["worker_id"] == case["worker_id"]
            assert heartbeat_failed.wait(timeout=2)
            # Simulate a route/LLM call that returns after its run was cancelled.
            return {"result": {"task_type": "path_planning", "reply": case["late_reply"]}}

    result = worker.process_next_run(
        "postgres://test", case["worker_id"], SlowGraph(), lease_seconds=3,
    )

    assert result == {"run_id": case["run_id"], "status": case["expected_status"]}
    assert len(finish_calls) == case["expected_finish_calls"]


def test_c1_production_context_replay_cases_follow_route_inheritance_policy(monkeypatch):
    from agents import workflow

    project_root = Path(__file__).resolve().parents[1]
    replay = json.loads(
        (project_root / "experiments/datasets/c1_production_context_replay_v1.json")
        .read_text(encoding="utf-8")
    )
    route_state = replay["prior_route_state"]
    inputs = {
        f"c1-{case['id']}": {
            "query": case["query"],
            "history": replay["history"],
            "task_revision": case["task_revision"],
            "task": {"route_spec": {}},
            "previous_route_state": route_state,
        }
        for case in replay["cases"]
    }
    monkeypatch.setattr(workflow, "get_run_input", lambda _url, run_id: inputs[run_id])
    monkeypatch.setattr(workflow, "append_run_event", lambda *_args, **_kwargs: {"stored": True})
    active_case = {"fail_weather": False, "fail_route": False, "query": ""}

    def weather_tool():
        if active_case["fail_weather"]:
            raise RuntimeError("simulated weather outage")
        return {"weather": "多云", "temperature": 23, "windpower": "2级", "advice": "适合步行"}

    monkeypatch.setattr(workflow, "_run_weather_tool", weather_tool)
    observed = {}

    def fake_agent(query, *, context, **_kwargs):
        observed[query] = context
        if active_case["fail_route"] and query == active_case["query"]:
            raise RuntimeError("simulated route agent failure")
        return _verified_route_response(f"AGENT:{query}")

    graph = workflow.build_agent_workflow(InMemorySaver(), agent=fake_agent)
    results = {}
    for case in replay["cases"]:
        active_case.update(
            fail_weather=case.get("fail_weather", False),
            fail_route=case.get("fail_route", False),
            query=case["query"],
        )
        run_id = f"c1-{case['id']}"
        results[case["id"]] = graph.invoke(
            {"run_id": run_id},
            {"configurable": {"thread_id": run_id}},
            context={"worker_id": f"worker-{run_id}"},
        )

    for case in replay["cases"]:
        context = observed[case["query"]]
        has_previous_route = "previous_route_state" in context
        assert has_previous_route is case["inherit_route_state"], case["id"]
        if has_previous_route:
            inherited = context["previous_route_state"]
            assert inherited["route_id"] == route_state["route_id"]
            assert inherited["route_kind"] == "via"
            assert inherited["via"] == route_state["via"]
            assert inherited["travel_mode"] == "walk"
            assert inherited["strategy"] == route_state["strategy"]
            assert inherited["hard_constraints"] == route_state["hard_constraints"]

    for case in replay["cases"]:
        if case.get("execution_kind") != "composite":
            continue
        result = results[case["id"]]["result"]
        assert result["execution_status"] == case.get("expected_status", "completed"), case["id"]
        assert result["requirement_results"]["route"] == (
            "failed" if case.get("fail_route") else "satisfied"
        )
        assert result["requirement_results"]["weather"] == (
            "failed" if case.get("fail_weather") else "satisfied"
        )
        if case.get("fail_weather"):
            assert result["weather"] is None
            assert result["requirement_errors"]["weather"] == "RuntimeError"
            assert f"AGENT:{case['query']}" in result["explanation"]
        else:
            assert result["weather"]["temperature"] == 23
            response_text = result["reply"] if case.get("fail_route") else result["explanation"]
            assert workflow._weather_message(result["weather"]) in response_text
        if case.get("fail_route"):
            assert result["requirement_errors"]["route"] == "RuntimeError"
            assert "路线暂时无法规划" in result["reply"]
        else:
            assert "route" not in result["requirement_errors"]
            assert f"AGENT:{case['query']}" in result["explanation"]


def test_worker_passes_current_identity_when_resuming_a_reclaimed_run(monkeypatch):
    from jobs import worker

    observed = {}
    monkeypatch.setattr(worker, "claim_next_run", lambda *_args: {
        "run_id": "reclaimed-run", "attempt": 2,
    })
    monkeypatch.setattr(worker, "heartbeat_run", lambda *_args: True)
    monkeypatch.setattr(worker, "finish_run", lambda *_args, **_kwargs: True)

    class CheckpointedGraph:
        def get_state(self, _config):
            return SimpleNamespace(values={"run_id": "reclaimed-run", "worker_id": "worker-old"},
                                   next=("execute_agent",))

        def invoke(self, value, _config, *, context):
            observed.update(input=value, context=context)
            return {"result": {"task_type": "chat", "reply": "已完成"}}

    outcome = worker.process_next_run("postgres://test", "worker-new", CheckpointedGraph())

    assert outcome["status"] == "completed"
    assert observed == {"input": None, "context": {"worker_id": "worker-new"}}


def test_systemd_agent_worker_loads_dotenv_before_connecting_to_postgres(monkeypatch):
    from jobs import worker

    calls = []
    monkeypatch.setattr(worker, "load_dotenv", lambda *_args: calls.append("dotenv"), raising=False)
    monkeypatch.setattr(worker, "run_forever", lambda: calls.append("worker"))
    monkeypatch.setattr(worker.signal, "signal", lambda *_args: None)

    worker.main()

    assert calls == ["dotenv", "worker"]


def test_agent_worker_supervisor_replaces_a_slot_that_dies_during_startup(monkeypatch):
    from jobs import worker

    stop = worker.threading.Event()
    replacement_ready = worker.threading.Event()
    attempts = {0: 0, 1: 0}

    def target(_database_url, _idle_seconds, slot):
        attempts[slot] += 1
        if slot == 0:
            replacement_ready.wait(timeout=1)
        elif attempts[slot] > 1:
            replacement_ready.set()
            stop.set()
        else:
            raise RuntimeError("startup connection failed")

    worker._supervise_workers(
        "postgres://test", 0.001, 2, stop_event=stop,
        worker_target=target, poll_seconds=0.001, restart_delay_seconds=0.001,
    )

    assert attempts == {0: 1, 1: 2}


def test_vision_readiness_requires_a_live_ready_worker(monkeypatch, tmp_path):
    import importlib.util
    from api import routes
    from storage import database, vision_repository

    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    monkeypatch.setattr(routes.config, "VISION_MODEL_PATH", str(model))
    monkeypatch.setattr(database, "initialize", lambda *_args: None)
    monkeypatch.setattr(vision_repository, "has_ready_worker", lambda *_args, **_kwargs: False)

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "PIL" else object())
    assert routes._vision_inference_readiness("postgres://test") == (False, True, False, False)

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    assert routes._vision_inference_readiness("postgres://test") == (False, True, True, False)

    monkeypatch.setattr(vision_repository, "has_ready_worker", lambda *_args, **_kwargs: True)
    assert routes._vision_inference_readiness("postgres://test") == (True, True, True, True)


def test_vision_worker_heartbeat_uses_freshness_and_worker_identity(monkeypatch):
    from contextlib import contextmanager
    from storage import vision_repository

    statements = []

    class Result:
        def fetchone(self):
            return {"ready": True}

    class Connection:
        def execute(self, sql, params):
            statements.append((sql, params))
            return Result()

    @contextmanager
    def fake_connect(_url):
        yield Connection()

    monkeypatch.setattr(vision_repository.database, "connect", fake_connect)

    vision_repository.heartbeat_worker(
        "postgres://test", "worker-a", ready=True, status_detail="loaded",
    )
    assert vision_repository.has_ready_worker(
        "postgres://test", max_age_seconds=12, worker_id="worker-a",
    )

    assert "ON CONFLICT(worker_id)" in statements[0][0]
    assert statements[0][1] == ("worker-a", True, "loaded")
    assert "heartbeat_at > now()" in statements[1][0]
    assert statements[1][1] == (12, "worker-a")


def test_ready_worker_status_returns_optional_model_capabilities(monkeypatch):
    from contextlib import contextmanager
    from storage import vision_repository

    class Connection:
        def execute(self, statement, parameters):
            assert "status_detail" in statement
            assert parameters == (30,)
            return self

        def fetchone(self):
            return {"worker_id": "vision-a", "status_detail": '{"flood_segmentation_supported":true}'}

    @contextmanager
    def connect(_url):
        yield Connection()

    monkeypatch.setattr(vision_repository.database, "connect", connect)

    status = vision_repository.ready_worker_status("postgres://test")

    assert status["worker_id"] == "vision-a"
    assert status["status_detail"] == '{"flood_segmentation_supported":true}'
