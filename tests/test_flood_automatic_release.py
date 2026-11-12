"""Automatic flood review must be separately backed by its own engineering evidence."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from vision import model_release, worker


def _make_release(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"flood-v7-test")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    manual = {}
    for name in ("parity", "resource", "workflow"):
        path = tmp_path / f"manual-{name}.json"
        path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
        manual[name] = {"status": "passed", "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    entry = {
        "mode": "assisted_review", "approved": True, "approved_by": "owner",
        "approved_at": now.isoformat(), "approval_expires_at": (now + timedelta(days=30)).isoformat(),
        "rights_status": "approved", "rights_evidence": "https://zenodo.org/records/7566797",
        "license": "CC BY 4.0", "model_revision": "doodleverse-floodnet-resunet-v7",
        "model_sha256": model_hash, "field_validated": False,
        "runtime_config": model_release.assisted_flood_runtime_config(), "acceptance": manual,
    }
    path = tmp_path / "release.json"

    def write():
        path.write_text(json.dumps({"schema_version": 1, "models": {"flood": entry}}), encoding="utf-8")

    write()
    return model, model_hash, entry, path, write


def _add_automatic_evidence(tmp_path, entry, model_hash):
    runtime = model_release.automatic_flood_runtime_config()
    evidence = {}
    for name in ("resource", "workflow"):
        path = tmp_path / f"automatic-{name}.json"
        path.write_text(json.dumps({"status": "passed", "scope": "automatic_flood",
                                    "model_sha256": model_hash, "runtime_config": runtime}),
                        encoding="utf-8")
        evidence[name] = {"status": "passed", "path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    entry["automatic_flood"] = {"enabled": True, "runtime_config": runtime, "acceptance": evidence}
    return evidence


def _check(model, path):
    return model_release.check_automatic_flood_release(model, "doodleverse-floodnet-resunet-v7", path)


def test_manual_release_alone_does_not_enable_automatic_flood(tmp_path):
    model, _, _, path, _ = _make_release(tmp_path)
    assert _check(model, path) == {"approved": False, "reason": "automatic_release_missing"}


def test_automatic_release_requires_bound_workflow_and_resource_evidence(tmp_path):
    model, model_hash, entry, path, write = _make_release(tmp_path)
    evidence = _add_automatic_evidence(tmp_path, entry, model_hash)
    write()
    assert _check(model, path) == {"approved": True, "reason": "approved"}

    # A copied or changed report cannot silently authorize the automatic path.
    (tmp_path / "automatic-resource.json").write_text('{"status":"failed"}', encoding="utf-8")
    assert _check(model, path)["reason"] == "automatic_resource_evidence_hash_mismatch"
    evidence["resource"]["sha256"] = hashlib.sha256((tmp_path / "automatic-resource.json").read_bytes()).hexdigest()
    write()
    assert _check(model, path)["reason"] == "automatic_resource_evidence_invalid"


def test_automatic_release_rejects_a_different_candidate_algorithm(tmp_path):
    model, model_hash, entry, path, write = _make_release(tmp_path)
    _add_automatic_evidence(tmp_path, entry, model_hash)
    entry["automatic_flood"]["runtime_config"]["max_candidates"] += 1
    write()
    assert _check(model, path)["reason"] == "automatic_runtime_contract_mismatch"


@pytest.mark.parametrize("auto_approved", [False, True])
def test_worker_exposes_automatic_capability_only_after_automatic_evidence(monkeypatch, tmp_path, auto_approved):
    model, _, _, manifest, _ = _make_release(tmp_path)
    detector = tmp_path / "detector.onnx"
    detector.write_bytes(b"detector")
    settings = {"VISION_TRIAL_ENABLED": False, "VISION_FLOOD_ENABLED": True,
                "VISION_FLOOD_RELEASE_MODE": "assisted_review", "VISION_FLOOD_MODEL_PATH": str(model),
                "VISION_MODEL_PATH": str(detector), "VISION_FLOOD_REVISION": "doodleverse-floodnet-resunet-v7",
                "VISION_FLOOD_PROBABILITY_THRESHOLD": 0.001, "VISION_FLOOD_MIN_AREA_RATIO": 0.08,
                "VISION_FLOOD_RUNTIME_PYTHON": "python", "VISION_FLOOD_TIMEOUT_SECONDS": 120,
                "VISION_OPTIONAL_MODELS_RELEASE_MANIFEST": str(manifest)}
    for key, value in settings.items():
        monkeypatch.setattr(worker.config, key, value, raising=False)
    class Model:
        model_id = "fixture"
        def __init__(self, *_args, model_version, **_kwargs):
            self.model_version = model_version
    monkeypatch.setattr(worker, "DoodleverseFloodSegmenter", Model)
    monkeypatch.setattr(worker, "check_automatic_flood_release", lambda *_args, **_kwargs: {
        "approved": auto_approved, "reason": "approved" if auto_approved else "automatic_release_missing"})
    _, capabilities = worker.load_trial_models(include_assisted_flood=True)
    assert capabilities["flood_assistance_supported"] is True
    assert capabilities["flood_assistance_automatic_supported"] is auto_approved
    assert capabilities["trial_flood_automatic_supported"] is auto_approved
