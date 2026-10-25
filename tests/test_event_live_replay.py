"""Synthetic event fixture preparation and guarded local collection."""

import hashlib
import json

import pytest

from experiments import event_live_replay as replay
from experiments.scenario_live_adapter import build_poi_lookup


def _case_and_snapshot():
    case = {
        "id": "event_dev_detour_001", "snapshot_id": "source-hash",
        "messages": [{"role": "user", "content": "查询路况并规划全程路线"}],
        "gold_derivation": {"parameters": {"event_snapshot_id": "simulated_detour_001"}},
    }
    snapshot = {
        "id": "simulated_detour_001", "scope": "synthetic_development_only",
        "campus_truth": False, "as_of_epoch": 150,
        "conditions": [
            {"id": "simulated_forward", "action": "closure",
             "affected_modes": ["walk"], "edge": {"u": 1, "v": 2, "key": 0},
             "start_time": 100, "end_time": 200},
            {"id": "simulated_reverse", "action": "closure",
             "affected_modes": ["walk"], "edge": {"u": 2, "v": 1, "key": 0},
             "start_time": 100, "end_time": 200},
        ],
    }
    return case, snapshot


def test_prepare_event_fixture_writes_only_synthetic_records_and_never_overwrites(tmp_path):
    case, snapshot = _case_and_snapshot()
    root = tmp_path / "event_runs"
    root.mkdir()

    prepared = replay.prepare_event_fixture(case, snapshot, "run_1", root=root)

    file = root / "run_1" / case["id"] / "road_conditions.json"
    rows = json.loads(file.read_text(encoding="utf-8"))
    assert prepared == {"path": str(file.resolve()), "as_of_epoch": 150,
                        "fixture_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    assert [row["id"] for row in rows] == ["simulated_forward", "simulated_reverse"]
    assert all(row["type"] == "closure"
               and row["source"] == {"kind": "research_simulated",
                                      "scope": "synthetic_development_only",
                                      "campus_truth": False}
               for row in rows)
    with pytest.raises(FileExistsError):
        replay.prepare_event_fixture(case, snapshot, "run_1", root=root)
    assert json.loads(file.read_text(encoding="utf-8")) == rows


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, digest):
        self.digest = digest
        self.posts = []

    def get(self, url, **_kwargs):
        if url.endswith("event-replay-readiness"):
            return _Response({
                "status": "ready", "service": "local_development",
                "database_isolated": True, "road_conditions_isolated": True,
                "as_of_epoch": 150, "fixture_sha256": self.digest,
            })
        return _Response({"data": {
            "status": "completed", "result": {
                "response_kind": "no_route", "message": "因封闭不可达。",
                "route_state": {"start": {"name": "起点"}, "end": {"name": "终点"},
                                "travel_mode": "walk", "strategy": {"name": "shortest"}},
            }, "events": [
                {"event_type": "agent_tool_trace_started", "payload": {}},
                {"event_type": "agent_tool_call", "payload": {
                    "tool_name": "list_road_conditions", "argument_keys": [],
                    "outcome": "success", "artifact_kind": None,
                    "road_event_ids": ["simulated_forward", "simulated_reverse"],
                }},
            ],
        }})

    def post(self, url, json, **_kwargs):
        self.posts.append((url, json))
        if url.endswith("/api/conversations"):
            return _Response({"data": {"conversation_id": "conv_1"}})
        return _Response({"data": {"run_id": "run_1"}})


def test_event_collection_checks_both_stores_before_writing_and_projects_real_run():
    case, _ = _case_and_snapshot()
    fixture = {"as_of_epoch": 150, "fixture_sha256": "a" * 64}
    lookup = build_poi_lookup([{"id": "poi_001", "name": "起点"},
                               {"id": "poi_015", "name": "终点"}])
    wrong = _Session("b" * 64)
    with pytest.raises(ValueError):
        replay.collect_event_case(case, wrong, "http://127.0.0.1:55440",
                                  lookup, fixture, poll_interval_s=0)
    assert wrong.posts == []

    session = _Session("a" * 64)
    row, metadata = replay.collect_event_case(
        case, session, "http://127.0.0.1:55440", lookup, fixture,
        poll_interval_s=0,
    )
    assert len(session.posts) == 2
    assert row["turns"][0]["status"] == "infeasible"
    assert row["turns"][0]["road_event_ids"] == [
        "simulated_forward", "simulated_reverse"
    ]
    assert metadata["run_id"] == "run_1"


def test_event_collection_records_timeout_as_failure_not_an_invented_answer():
    case, _ = _case_and_snapshot()
    fixture = {"as_of_epoch": 150, "fixture_sha256": "a" * 64}
    lookup = build_poi_lookup([])

    class PendingSession(_Session):
        def get(self, url, **kwargs):
            if url.endswith("event-replay-readiness"):
                return super().get(url, **kwargs)
            return _Response({"data": {"status": "running"}})

    row, metadata = replay.collect_event_case(
        case, PendingSession("a" * 64), "http://127.0.0.1:55440",
        lookup, fixture, poll_interval_s=0, timeout_s=0,
    )
    assert row["turns"] == []
    assert row["transport_error"] == "timeout"
    assert metadata["run_id"] == "run_1"


def test_event_collection_rejects_a_terminal_reply_after_total_deadline():
    case, _ = _case_and_snapshot()
    fixture = {"as_of_epoch": 150, "fixture_sha256": "a" * 64}
    row, _ = replay.collect_event_case(
        case, _Session("a" * 64), "http://127.0.0.1:55440",
        build_poi_lookup([]), fixture, poll_interval_s=0, timeout_s=0,
    )
    assert row["transport_error"] == "timeout"
    assert row["turns"] == []
