"""A live run must produce projected evidence without leaking its raw payload."""

import pytest

from api.routes import _agent_response_to_legacy
from experiments.scenario_live_adapter import (
    build_poi_lookup,
    project_live_run,
    require_local_service,
    run_case,
)


def _lookup():
    return build_poi_lookup([
        {"id": "poi_001", "name": "玉兰2门", "aliases": ["玉兰二门"]},
        {"id": "poi_002", "name": "桂园操场", "aliases": ["桂操", "操场"]},
        {"id": "poi_003", "name": "梅园操场", "aliases": ["梅操", "操场"]},
        {"id": "poi_015", "name": "武汉大学图书馆(总馆)"},
    ])


def test_live_projection_keeps_exact_route_edges_and_stable_poi_ids():
    run = {"status": "completed", "result": {
        "response_kind": "route", "task_type": "path_planning",
        "route_state": {
            "start": {"name": "玉兰二门"}, "end": {"name": "桂操"},
            "via": None, "travel_mode": "walk", "strategy": {"name": "shortest"},
            "hard_constraints": {"avoid_steps": True},
        },
        "recommended_edge_ids": [[101, 102, 4]],
        "recommended": [[114.3, 30.5], [114.4, 30.6]],
        "requirement_results": {"route": "satisfied"},
    }}
    turn = project_live_run(run, _lookup())
    assert turn == {
        "status": "completed", "kind": "route", "start_poi_id": "poi_001",
        "end_poi_id": "poi_002", "via_poi_ids": [], "travel_mode": "walk",
        "strategy": "shortest", "hard_constraints": {"avoid_steps": True},
        "candidate_poi_ids": [], "clarification_slot": None,
        "outputs": ["route"], "route_edge_ids": [[101, 102, 4]], "unknowns": [],
    }
    assert "result" not in turn and "recommended" not in turn
    assert "independent_review" not in turn


def test_weather_output_requires_observation_and_user_facing_reply():
    route = {"start_name": "玉兰2门", "end_name": "桂园操场",
             "recommended_edge_ids": [[101, 102, 4]]}
    observed = {"weather": "晴", "temperature": 22}
    response = _agent_response_to_legacy({
        "response_kind": "route", "route": route,
        "message": "路线已规划。天气参考：晴、22°C。", "weather": observed,
    })
    projected = project_live_run({"status": "completed", "result": response}, _lookup())
    assert projected["outputs"] == ["route", "weather"]

    response["explanation"] = "路线已规划。"
    assert project_live_run({"status": "completed", "result": response}, _lookup())[
        "outputs"] == ["route"]


def test_weather_only_followup_projects_information_without_new_route():
    turn = project_live_run({"status": "completed", "result": {
        "response_kind": "chat", "task_type": "chat",
        "reply": "当前天气：晴，22°C。", "weather": {"weather": "晴", "temperature": 22},
    }}, _lookup())
    assert turn["kind"] == "information"
    assert turn["outputs"] == ["weather"]
    assert turn["route_edge_ids"] == []


def test_pending_origin_keeps_selected_destination_as_a_stable_poi():
    turn = project_live_run({"status": "needs_input", "result": {
        "response_kind": "clarify", "task_type": "unknown",
        "clarify": {"question": "你从哪里出发？"},
        "place_request": {"pending_slot": "start", "selected_poi_id": "poi_002",
                          "mode": "walk"},
    }}, _lookup())
    assert turn["kind"] == "clarify"
    assert turn["start_poi_id"] is None
    assert turn["end_poi_id"] == "poi_002"
    assert turn["clarification_slot"] == "start"


def test_route_projection_uses_canonical_result_when_state_name_is_not_an_alias():
    turn = project_live_run({"status": "completed", "result": {
        "response_kind": "route", "route_state": {
            "start": {"name": "玉兰二门"},
            "end": {"name": "武汉大学图书馆总馆"},
            "travel_mode": "walk", "strategy": {"name": "shortest"},
        },
        "start": {"name": "玉兰2门"},
        "end": {"name": "武汉大学图书馆(总馆)"},
        "recommended_edge_ids": [[1, 2, 0]],
    }}, _lookup())
    assert turn["end_poi_id"] == "poi_015"
    assert "poi_identity_unresolved" not in turn["unknowns"]


def test_route_projection_does_not_hide_conflicting_endpoint_identities():
    turn = project_live_run({"status": "completed", "result": {
        "response_kind": "route", "route_state": {
            "start": {"name": "玉兰二门"}, "end": {"name": "桂操"},
        },
        "end": {"name": "梅操"},
        "recommended_edge_ids": [[1, 2, 0]],
    }}, _lookup())
    assert turn["end_poi_id"] is None
    assert "route_end_identity_conflict" in turn["unknowns"]


def test_ambiguous_alias_is_not_guessed_and_missing_edges_are_not_evidence():
    run = {"status": "completed", "result": {
        "response_kind": "route", "task_type": "path_planning",
        "route_state": {"start": {"name": "操场"}, "end": {"name": "桂操"},
                        "travel_mode": "walk", "strategy": {"name": "shortest"},
                        "hard_constraints": {}},
    }}
    turn = project_live_run(run, _lookup())
    assert turn["start_poi_id"] is None
    assert turn["end_poi_id"] == "poi_002"
    assert turn["route_edge_ids"] == []
    assert "poi_identity_unresolved" in turn["unknowns"]
    assert "route_edge_sequence_missing" in turn["unknowns"]


@pytest.mark.parametrize("artifact_kind", ["candidates", "route_comparison"])
def test_live_projection_keeps_opt_in_tool_trace_but_drops_argument_values(artifact_kind):
    run = {"status": "completed", "events": [
        {"event_type": "agent_tool_trace_started", "payload": {}},
        {"event_type": "agent_tool_call", "payload": {
            "tool_name": "search_poi_candidates",
            "argument_keys": ["activities", "start"],
            "outcome": "success", "artifact_kind": artifact_kind,
            "raw_arguments": {"start": "武汉大学信息学部"},
            "raw_result": {"message": "private result"},
        }},
        {"event_type": "node_started", "payload": {"label": "处理中"}},
    ], "result": {"response_kind": "information", "message": "找到附近地点。"}}

    turn = project_live_run(run, _lookup())

    assert turn["tool_trace"] == [{
        "tool_name": "search_poi_candidates",
        "argument_keys": ["activities", "start"],
        "outcome": "success", "artifact_kind": artifact_kind,
    }]
    assert "raw_arguments" not in str(turn)
    assert "raw_result" not in str(turn)


def test_live_projection_rejects_unknown_tool_trace_artifact_kind():
    run = {"status": "completed", "events": [
        {"event_type": "agent_tool_trace_started", "payload": {}},
        {"event_type": "agent_tool_call", "payload": {
            "tool_name": "compare_routes", "argument_keys": [],
            "outcome": "success", "artifact_kind": "unknown_artifact",
        }},
    ], "result": {"response_kind": "information"}}

    assert "tool_trace" not in project_live_run(run, _lookup())


@pytest.mark.parametrize("outcome", [
    "skipped_after_route_verification_retry",
    "skipped_after_route_verification_failure",
])
def test_live_projection_keeps_planner_skipped_tool_trace(outcome):
    call = {"tool_name": "plan_route", "argument_keys": [],
            "outcome": outcome, "artifact_kind": None}
    run = {"status": "completed", "events": [
        {"event_type": "agent_tool_trace_started", "payload": {}},
        {"event_type": "agent_tool_call", "payload": call},
    ], "result": {"response_kind": "information"}}

    assert project_live_run(run, _lookup())["tool_trace"] == [call]


def test_live_projection_rejects_unknown_tool_trace_outcome():
    run = {"status": "completed", "events": [
        {"event_type": "agent_tool_trace_started", "payload": {}},
        {"event_type": "agent_tool_call", "payload": {
            "tool_name": "plan_route", "argument_keys": [],
            "outcome": "unknown_outcome", "artifact_kind": None,
        }},
    ], "result": {"response_kind": "information"}}

    assert "tool_trace" not in project_live_run(run, _lookup())


def test_live_projection_marks_empty_trace_when_trace_was_enabled():
    run = {"status": "completed", "events": [
        {"event_type": "agent_tool_trace_started", "payload": {}},
    ], "result": {"response_kind": "information", "message": "你好。"}}

    assert project_live_run(run, _lookup())["tool_trace"] == []


@pytest.mark.parametrize("artifact_kind", ["candidates", "route_comparison"])
def test_development_replay_counts_projected_agent_tool_calls(artifact_kind):
    import copy
    import json
    from experiments import scenario_live_adapter

    dataset = json.loads((scenario_live_adapter.ROOT
                          / "experiments/datasets/scenarios_dev_v1.json")
                         .read_text(encoding="utf-8"))
    case = copy.deepcopy(dataset["cases"][0])
    case["messages"] = case["messages"][:1]
    case["expected_turns"] = case["expected_turns"][:1]
    dataset["cases"] = [case]

    class ContextSession(_Session):
        def __init__(self):
            super().__init__()
            self.runs = iter([{
                "status": "completed", "task_id": "task-1", "current_task_revision": 1,
                "events": [
                    {"event_type": "agent_tool_trace_started", "payload": {}},
                    {"event_type": "agent_tool_call", "payload": {
                        "tool_name": "search_poi_candidates", "argument_keys": ["start"],
                        "outcome": "success", "artifact_kind": artifact_kind,
                    }},
                ],
                "result": {"response_kind": "information", "message": "候选已整理。"},
            }])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    rows = scenario_live_adapter.collect_development_replay(
        dataset, "http://127.0.0.1:5000", _lookup(), session_factory=ContextSession,
    )

    assert rows[0]["tool_calls"] == 1
    assert rows[0]["turns"][0]["tool_trace"][0]["tool_name"] == "search_poi_candidates"


def test_v2_development_replay_uses_same_isolated_service_gate():
    import json
    from experiments import scenario_live_adapter

    dataset = json.loads((scenario_live_adapter.ROOT
                          / "experiments/datasets/scenarios_dev_v2.json")
                         .read_text(encoding="utf-8"))
    dataset["cases"] = dataset["cases"][:1]

    class ContextSession(_Session):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    rows = scenario_live_adapter.collect_development_replay(
        dataset, "http://127.0.0.1:5000", _lookup(), session_factory=ContextSession,
    )
    assert len(rows) == dataset.get("repeats", 1)
    assert rows[0]["scenario_id"] == dataset["cases"][0]["id"]


def test_development_replay_refuses_service_without_isolated_database_proof():
    import copy
    import json
    from experiments import scenario_live_adapter
    from experiments.scenario_live_adapter import collect_development_replay

    dataset = json.loads((scenario_live_adapter.ROOT
                          / "experiments/datasets/scenarios_dev_v1.json")
                         .read_text(encoding="utf-8"))
    case = copy.deepcopy(dataset["cases"][0])
    case["messages"] = case["messages"][:1]
    case["expected_turns"] = case["expected_turns"][:1]
    dataset["cases"] = [case]

    class UnsafeSession(_Response):
        def __init__(self):
            super().__init__({"status": "blocked", "database_isolated": False})
            self.posts = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, *_args, **_kwargs):
            return self

        def post(self, *args, **kwargs):
            self.posts.append((args, kwargs))
            return _Response({"data": {"conversation_id": "unexpected"}})

    session = UnsafeSession()

    with pytest.raises(ValueError, match="isolated development database"):
        collect_development_replay(
            dataset, "http://127.0.0.1:5000", _lookup(),
            session_factory=lambda: session,
        )

    assert session.posts == []


def test_agent_api_preserves_edge_ids_needed_for_independent_replay():
    route = {"start_name": "玉兰2门", "end_name": "桂园操场",
             "recommended": [[114.3, 30.5]],
             "recommended_edge_ids": [[101, 102, 4]]}
    response = _agent_response_to_legacy({
        "response_kind": "route", "route": route, "message": "路线已生成",
    })
    assert response["recommended_edge_ids"] == [[101, 102, 4]]


def test_agent_api_exposes_only_delivered_weather_as_structured_output():
    route = {"start_name": "玉兰2门", "end_name": "桂园操场",
             "recommended_edge_ids": [[101, 102, 4]]}
    weather = {"weather": "晴", "temperature": 22}
    response = _agent_response_to_legacy({
        "response_kind": "route", "route": route,
        "message": "路线已规划。天气参考：晴、22°C。", "weather": weather,
    })
    assert response["weather"] == weather
    projection = project_live_run({"status": "completed", "result": response}, _lookup())
    assert projection["outputs"] == ["route", "weather"]


def test_agent_api_preserves_road_condition_count_used_in_route_computation():
    response = _agent_response_to_legacy({
        "response_kind": "route", "message": "路线已按路况计算",
        "route": {"start_name": "玉兰2门", "end_name": "桂园操场",
                  "recommended_edge_ids": [[101, 102, 4]],
                  "road_conditions_applied": 1},
    })
    assert response["road_conditions_applied"] == 1


def test_agent_clarification_preserves_route_verification_report():
    report = {"status": "fail", "issues": ["length_mismatch"]}
    response = _agent_response_to_legacy({
        "response_kind": "clarify",
        "message": "需要补充信息",
        "clarify": {"question": "需要补充信息", "options": []},
        "route_verification": report,
    })

    assert response["route_verification"] == report


class _Response:
    def __init__(self, data):
        self._data = {"data": data}

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Session:
    def __init__(self):
        self.posts = []
        self.runs = iter([
            {"status": "needs_input", "task_id": "task-1", "current_task_revision": 1,
             "result": {"response_kind": "clarify", "place_request": {"pending_slot": "activity"}}},
            {"status": "completed", "task_id": "task-1", "current_task_revision": 2,
             "result": {"response_kind": "candidates", "candidates": [
                 {"poi_id": "poi_002", "name": "桂园操场"}]}}
        ])

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        if url.endswith("/api/conversations"):
            return _Response({"conversation_id": "conversation-1"})
        return _Response({"run_id": f"run-{len(self.posts)}"})

    def get(self, url, params=None, timeout=10):
        if url.endswith("/api/research/replay-readiness"):
            return _Response({
                "status": "ready", "service": "local_development",
                "database_isolated": True,
            })
        return _Response(next(self.runs))


def test_live_case_resumes_pending_task_and_projects_only_known_fields():
    case = {"messages": [
        {"role": "user", "content": "我想打球"},
        {"role": "user", "content": "篮球"},
    ]}
    session = _Session()
    turns, latency = run_case(case, session, "http://127.0.0.1:5000", _lookup(),
                              poll_interval_s=0)
    assert turns[0]["clarification_slot"] == "activity"
    assert turns[1]["candidate_poi_ids"] == ["poi_002"]
    assert session.posts[-1][1]["continuation_task_id"] == "task-1"
    assert session.posts[-1][1]["base_revision"] == 1
    assert latency >= 0
    assert all("result" not in turn for turn in turns)


def test_direct_live_case_refuses_writes_without_isolated_database_proof():
    case = {"messages": [{"role": "user", "content": "我想打球"}]}

    class UnsafeSession:
        def __init__(self):
            self.posts = []

        def get(self, *_args, **_kwargs):
            return _Response({"status": "blocked", "database_isolated": False})

        def post(self, *args, **kwargs):
            self.posts.append((args, kwargs))
            return _Response({"data": {"conversation_id": "unexpected"}})

    session = UnsafeSession()

    with pytest.raises(ValueError, match="isolated development database"):
        run_case(case, session, "http://127.0.0.1:5000", _lookup(), poll_interval_s=0)

    assert session.posts == []


def test_live_replay_refuses_remote_or_non_http_service():
    for url in ("https://whuspati.online", "http://192.168.1.2:5000", "file:///tmp/run"):
        with pytest.raises(ValueError, match="local"):
            require_local_service(url)
    assert require_local_service("http://127.0.0.1:5000") == "http://127.0.0.1:5000"


def test_existing_output_is_rejected_before_any_live_request(tmp_path, monkeypatch):
    from experiments import scenario_live_adapter

    dataset = tmp_path / "dataset.json"
    dataset.write_text("{}", encoding="utf-8")
    output = tmp_path / "observations.jsonl"
    output.write_text("previous result\n", encoding="utf-8")

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("must not call the service before output preflight")

    monkeypatch.setattr(scenario_live_adapter, "collect_development_replay", unexpected_request)
    with pytest.raises(FileExistsError):
        scenario_live_adapter.main([
            "--dataset", str(dataset), "--service", "http://127.0.0.1:5000",
            "--output", str(output),
        ])
    assert output.read_text(encoding="utf-8") == "previous result\n"
