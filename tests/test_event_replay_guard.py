"""A synthetic event replay may read only a dedicated, immutable local fixture."""

import hashlib
import importlib
import json

import pytest

from spatial import road_conditions as rc


def _guard():
    try:
        return importlib.import_module("experiments.event_replay_guard")
    except ImportError:
        pytest.fail("event replay isolation guard is missing")


def _fixture(tmp_path, monkeypatch):
    guard = _guard()
    root = tmp_path / "event_runs"
    root.mkdir()
    target = root / "road_conditions.json"
    rows = [{
        "id": "simulated_detour_edge_001", "type": "closure",
        "action": "closure", "affected_modes": ["walk"],
        "edge": {"u": 1, "v": 2, "key": 0},
        "start_time": 100, "end_time": 200,
        "source": {"kind": "research_simulated",
                   "scope": "synthetic_development_only", "campus_truth": False},
    }]
    target.write_text(json.dumps(rows), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(guard, "REPLAY_ROOT", root)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("AGENT_REPLAY_ENABLE_LOCAL", "1")
    monkeypatch.setenv("AGENT_TOOL_TRACE_ENABLED", "1")
    monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "false")
    monkeypatch.setenv("AGENT_REPLAY_EVENT_EPOCH", "150")
    monkeypatch.setenv("AGENT_REPLAY_EVENT_SHA256", digest)
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(target))
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/campus_test_event")
    monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(target))
    monkeypatch.setattr(rc, "_cache", None)
    monkeypatch.setattr(rc, "_cache_mtime", 0.0)
    return target, digest


def test_event_replay_guard_certifies_fixture_and_freezes_condition_time(tmp_path, monkeypatch):
    target, digest = _fixture(tmp_path, monkeypatch)
    guard = _guard()
    assert guard.certify_local_event_fixture(target) == {
        "as_of_epoch": 150, "fixture_sha256": digest,
    }
    assert [row["id"] for row in rc.list_conditions(strict=True)] == [
        "simulated_detour_edge_001"
    ]


@pytest.mark.parametrize("change", ["production", "missing_hash", "missing_epoch",
                                     "missing_trace", "changed_file",
                                     "wrong_binding", "outside_root",
                                     "online_poi_lookup"])
def test_event_replay_guard_rejects_unisolated_or_mutated_fixture(
    tmp_path, monkeypatch, change,
):
    target, _ = _fixture(tmp_path, monkeypatch)
    guard = _guard()
    if change == "production":
        monkeypatch.setenv("FLASK_ENV", "production")
    elif change == "missing_hash":
        monkeypatch.delenv("AGENT_REPLAY_EVENT_SHA256")
    elif change == "missing_epoch":
        monkeypatch.delenv("AGENT_REPLAY_EVENT_EPOCH")
    elif change == "missing_trace":
        monkeypatch.delenv("AGENT_TOOL_TRACE_ENABLED")
    elif change == "online_poi_lookup":
        monkeypatch.setenv("AMAP_POI_LOOKUP_ENABLED", "true")
    elif change == "changed_file":
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif change == "wrong_binding":
        monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(tmp_path / "other.json"))
    else:
        outside = tmp_path / "business.json"
        outside.write_bytes(target.read_bytes())
        monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(outside))
        monkeypatch.setattr(rc, "_CONDITIONS_FILE", str(outside))
    with pytest.raises(ValueError):
        guard.certify_local_event_fixture(rc._CONDITIONS_FILE)


def test_replay_clock_without_opt_in_uses_wall_time(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("AGENT_REPLAY_EVENT_EPOCH")
    monkeypatch.delenv("AGENT_REPLAY_EVENT_SHA256")
    assert rc.list_conditions(strict=True) == []


def test_invalid_replay_configuration_rejects_before_reading_condition_file(
    tmp_path, monkeypatch,
):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENT_REPLAY_EVENT_SHA256", "0" * 64)
    monkeypatch.setattr(rc, "_load_conditions", lambda **_kw: pytest.fail(
        "unsafe file was read before replay proof"
    ))
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)


def test_missing_replay_root_fails_closed_as_unavailable(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    guard = _guard()
    monkeypatch.setattr(guard, "REPLAY_ROOT", tmp_path / "missing")
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)


def test_certified_route_reads_the_same_bytes_that_passed_digest(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(rc, "_load_conditions", lambda **_kw: pytest.fail(
        "replay route must not reread the condition file or its mtime cache"
    ))
    assert [row["id"] for row in rc.list_conditions(strict=True)] == [
        "simulated_detour_edge_001"
    ]


def test_replay_clock_rejects_business_database_at_service_and_route_boundary(
    tmp_path, monkeypatch,
):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/campus_business")
    with pytest.raises(rc.RoadConditionsUnavailableError):
        rc.list_conditions(strict=True)
    with pytest.raises(ValueError):
        from app import create_app
        create_app()


def test_replay_process_cannot_mutate_its_frozen_event_fixture(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    with pytest.raises(rc.RoadConditionsUnavailableError):
        with rc._condition_write_lock():
            pytest.fail("frozen fixture became writable")


def test_event_readiness_requires_both_isolated_db_and_condition_file(tmp_path, monkeypatch):
    from app import create_app
    from storage import database

    target, digest = _fixture(tmp_path, monkeypatch)

    class FakeResult:
        def fetchone(self):
            return {"name": "campus_test_event", "address": "127.0.0.1"}

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _sql):
            return FakeResult()

    monkeypatch.setattr(database, "connect", lambda _url: FakeConnection())
    monkeypatch.setattr(database, "initialize", lambda _url: None)
    app = create_app()
    app.config.update(TESTING=True,
                      DATABASE_URL="postgresql://localhost/campus_test_event")
    client = app.test_client()

    ready = client.get("/api/research/event-replay-readiness")
    assert ready.status_code == 200
    assert ready.get_json() == {
        "status": "ready", "service": "local_development",
        "database_isolated": True, "road_conditions_isolated": True,
        "as_of_epoch": 150, "fixture_sha256": digest,
    }

    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    blocked = client.get("/api/research/event-replay-readiness")
    assert blocked.status_code == 503
    assert blocked.get_json() == {
        "status": "blocked", "database_isolated": True,
        "road_conditions_isolated": False,
    }
