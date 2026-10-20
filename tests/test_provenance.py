"""Agent 执行溯源（PROV 风格）单测。"""

import json
from unittest.mock import patch

import pytest

from agents import provenance


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "PROVENANCE_PATH", tmp_path / "provenance.jsonl")
    yield


def _finished_record(**kwargs):
    recorder = provenance.ProvenanceRecorder(
        "从珞珈门到图书馆，避开陡坡",
        uid="u1",
        model="deepseek-chat",
        tool_names=["plan_route", "resolve_poi"],
        has_gps=True,
        has_waypoints=False,
    )
    recorder.add_activity("resolve_poi", ["name"], "success", None)
    recorder.add_activity("plan_route", ["start", "end", "constraints"],
                          "route_verification_failed", None)
    recorder.add_influence("constraint_vs_connectivity")
    return recorder.finish(response_kind="route", artifact_kinds=["route"],
                           verification_status="pass", turns=2)


class TestRecordShape:
    def test_prov_vocabulary_present(self):
        record = _finished_record()
        assert record["prov"]["namespace"] == provenance.PROV_NS
        assert record["activity"]["id"].startswith("activity:")
        assert record["agents"][0]["id"].startswith("agent:")
        assert record["agents"][0]["role"] == "model"

    def test_entities_cover_input_and_output(self):
        record = _finished_record()
        roles = {entity["role"] for entity in record["entities"]}
        assert roles == {"input", "output"}

    def test_relations_link_activity_to_entities_and_agent(self):
        record = _finished_record()
        activity = record["activity"]
        input_id = next(e["id"] for e in record["entities"] if e["role"] == "input")
        output_id = next(e["id"] for e in record["entities"] if e["role"] == "output")
        assert activity["used"] == [input_id]
        assert activity["wasGeneratedBy"] == [output_id]
        assert activity["wasAssociatedWith"] == [record["agents"][0]["id"]]

    def test_reflection_influence_recorded(self):
        record = _finished_record()
        assert record["activity"]["wasInfluencedBy"] == ["constraint_vs_connectivity"]

    def test_tool_activities_keep_order(self):
        record = _finished_record()
        assert [a["tool_name"] for a in record["tool_activities"]] == ["resolve_poi", "plan_route"]
        assert record["activity"]["tools_used"] == ["resolve_poi", "plan_route"]

    def test_finish_is_idempotent(self):
        recorder = provenance.ProvenanceRecorder("hi")
        first = recorder.finish(response_kind="chat")
        assert recorder.finish(response_kind="route") is first


class TestPrivacy:
    def test_no_raw_query_text(self):
        record = _finished_record()
        serialized = json.dumps(record, ensure_ascii=False)
        assert "珞珈门" not in serialized
        assert "避开陡坡" not in serialized

    def test_no_coordinates(self):
        record = _finished_record()
        serialized = json.dumps(record, ensure_ascii=False)
        assert "lng" not in serialized and "lat" not in serialized

    def test_query_fingerprint_is_stable_and_short(self):
        first = provenance.query_fingerprint("同一条查询")
        second = provenance.query_fingerprint("同一条查询")
        assert first == second
        assert len(first) == 16
        assert first != provenance.query_fingerprint("另一条查询")


class TestPersistence:
    def test_persist_and_load_roundtrip(self):
        record = _finished_record()
        assert provenance.persist(record) is True
        loaded = provenance.load(record["run_id"])
        assert loaded is not None
        assert loaded["run_id"] == record["run_id"]
        assert loaded["tool_activities"][0]["tool_name"] == "resolve_poi"

    def test_load_unknown_run_returns_none(self):
        assert provenance.load("does-not-exist") is None

    def test_persist_rejects_garbage(self):
        assert provenance.persist(None) is False
        assert provenance.persist({"no_run_id": True}) is False

    def test_persist_never_raises_on_failure(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            assert provenance.persist(_finished_record()) is False

    def test_prune_trims_oversized_file(self):
        record = _finished_record()
        for _ in range(6):
            provenance.persist(record)
        with patch.object(provenance, "MAX_FILE_BYTES", 10):
            provenance._prune(provenance.PROVENANCE_PATH)
        remaining = provenance.PROVENANCE_PATH.read_text(encoding="utf-8").splitlines()
        assert 0 < len(remaining) < 6

    def test_concurrent_persists_keep_every_complete_record(self):
        """多线程（对应多 Web 进程）并发追加时，每条记录都必须完整且不丢失。"""
        from concurrent.futures import ThreadPoolExecutor

        records = []
        for _ in range(24):
            records.append(_finished_record())

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(provenance.persist, records))

        assert all(results)
        lines = provenance.PROVENANCE_PATH.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(records)
        parsed = [json.loads(line) for line in lines]
        assert {item["run_id"] for item in parsed} == {r["run_id"] for r in records}

    def test_lock_file_lives_next_to_store(self):
        provenance.persist(_finished_record())
        lock_path = provenance.PROVENANCE_PATH.parent / (
            provenance.PROVENANCE_PATH.name + ".lock")
        assert lock_path.exists()


class TestAdminEndpoint:
    @pytest.fixture
    def client(self):
        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def test_requires_admin(self, client):
        record = _finished_record()
        provenance.persist(record)
        r = client.get(f"/api/agent/provenance/{record['run_id']}")
        assert r.status_code == 401

    def test_returns_record_for_admin_token(self, client, monkeypatch):
        from api import routes
        monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_TOKEN", "secret-token")
        record = _finished_record()
        provenance.persist(record)
        r = client.get(f"/api/agent/provenance/{record['run_id']}",
                       headers={"X-Admin-Token": "secret-token"})
        assert r.status_code == 200
        assert r.get_json()["data"]["run_id"] == record["run_id"]

    def test_unknown_run_is_404_for_admin(self, client, monkeypatch):
        from api import routes
        monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_TOKEN", "secret-token")
        r = client.get("/api/agent/provenance/nope",
                       headers={"X-Admin-Token": "secret-token"})
        assert r.status_code == 404
