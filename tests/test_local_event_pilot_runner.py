"""Safety checks for the local synthetic event replay launcher."""

import hashlib

import pytest

from experiments.run_local_event_pilot import (
    build_replay_environment, collection_source_versions,
    disable_child_dotenv, isolate_agent_sidecars, load_poi_lookup,
)


def _fixture(tmp_path):
    path = tmp_path / "road_conditions.json"
    path.write_bytes(b"[]\n")
    return {"path": str(path), "as_of_epoch": 150,
            "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_replay_child_receives_only_isolated_settings(tmp_path):
    fixture = _fixture(tmp_path)
    inherited = {"PATH": "bin", "SYSTEMROOT": "C:/Windows",
                 "DATABASE_URL": "postgresql://business.example/prod",
                 "AMAP_KEY": "business-map-key",
                 "ROAD_CONDITION_ADMIN_TOKEN": "admin-secret",
                 "DEEPSEEK_API_KEY": "business-model-key"}

    child = build_replay_environment(
        "postgresql://postgres@127.0.0.1:55439/whu_agent_event_test",
        fixture, model_key="dedicated-model-key", base_env=inherited,
    )

    assert child["DATABASE_URL"].endswith("/whu_agent_event_test")
    assert child["SYSTEMROOT"] == "C:/Windows"
    assert child["DEEPSEEK_API_KEY"] == "dedicated-model-key"
    assert child["ROAD_CONDITIONS_FILE"] == fixture["path"]
    assert child["AGENT_REPLAY_EVENT_SHA256"] == fixture["fixture_sha256"]
    assert child["AGENT_REPLAY_EVENT_EPOCH"] == "150"
    assert child["AGENT_REPLAY_ENABLE_LOCAL"] == "1"
    assert child["AGENT_TOOL_TRACE_ENABLED"] == "1"
    assert child["AMAP_POI_LOOKUP_ENABLED"] == "false"
    assert "AMAP_KEY" not in child
    assert "ROAD_CONDITION_ADMIN_TOKEN" not in child


@pytest.mark.parametrize("url", [
    "postgresql://postgres@127.0.0.1:5432/campus",
    "postgresql://postgres@business.example/campus_test",
])
def test_replay_child_rejects_nonisolated_database(tmp_path, url):
    with pytest.raises(ValueError, match="local test database"):
        build_replay_environment(url, _fixture(tmp_path), model_key="key")


def test_replay_child_rejects_missing_model_key_and_tampered_fixture(tmp_path):
    fixture = _fixture(tmp_path)
    url = "postgresql://postgres@127.0.0.1:55439/whu_agent_event_test"
    with pytest.raises(ValueError, match="model key"):
        build_replay_environment(url, fixture, model_key="")
    fixture["fixture_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixture"):
        build_replay_environment(url, fixture, model_key="key")


def test_lookup_reads_pois_inside_master_data_envelope(tmp_path):
    path = tmp_path / "pois.json"
    path.write_text('{"count": 1, "pois": [{"id": "poi_001", '
                    '"name": "梅园综合楼"}]}', encoding="utf-8")

    lookup = load_poi_lookup(path)

    assert lookup["names"]["梅园综合楼"] == "poi_001"
    assert lookup["ids"] == {"poi_001"}


def test_collection_versions_include_model_and_runtime_hashes_without_credentials():
    versions = collection_source_versions("https://api.deepseek.com/v1", "deepseek-chat")

    assert versions["model"] == "deepseek-chat"
    assert versions["provider_host"] == "api.deepseek.com"
    assert all(len(versions[name]) == 64 for name in (
        "runtime_source_tree_sha256", "graph_sha256", "poi_sha256",
        "prompt_sha256", "tool_code_sha256",
    ))
    assert "key" not in " ".join(versions).lower()


def test_replay_sidecars_are_bound_to_fixture_directory(tmp_path, monkeypatch):
    from agents import experience, provenance

    fixture = _fixture(tmp_path)
    before_experience = experience.EXPERIENCE_PATH
    before_provenance = provenance.PROVENANCE_PATH
    try:
        paths = isolate_agent_sidecars(fixture["path"])
        assert paths["experience"].parent == tmp_path / "agent_sidecars"
        assert paths["provenance"].parent == tmp_path / "agent_sidecars"
        assert experience.EXPERIENCE_PATH == paths["experience"]
        assert provenance.PROVENANCE_PATH == paths["provenance"]
    finally:
        monkeypatch.setattr(experience, "EXPERIENCE_PATH", before_experience)
        monkeypatch.setattr(provenance, "PROVENANCE_PATH", before_provenance)


def test_replay_child_never_loads_checkout_dotenv(tmp_path, monkeypatch):
    import dotenv

    config_file = tmp_path / ".env"
    config_file.write_text("AGENT_REPLAY_TEST_BUSINESS_SECRET=hidden\n",
                           encoding="utf-8")
    monkeypatch.delenv("AGENT_REPLAY_TEST_BUSINESS_SECRET", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", dotenv.load_dotenv)
    disable_child_dotenv()

    assert dotenv.load_dotenv(config_file) is False
    assert "AGENT_REPLAY_TEST_BUSINESS_SECRET" not in __import__("os").environ
