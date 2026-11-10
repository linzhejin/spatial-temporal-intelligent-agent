import hashlib
import json
from datetime import datetime, timedelta, timezone

from vision.model_release import assisted_flood_runtime_config, check_optional_model_release


def test_assisted_review_release_needs_exact_model_and_technical_evidence(tmp_path):
    model_path = tmp_path / "flood-v7.onnx"
    model_path.write_bytes(b"verified-fp32-onnx")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    acceptance = {}
    for name in ("parity", "resource", "workflow"):
        evidence_path = tmp_path / f"{name}.json"
        evidence_path.write_text(json.dumps({"status": "passed", "kind": name}), encoding="utf-8")
        acceptance[name] = {
            "status": "passed", "path": str(evidence_path),
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        }
    runtime = assisted_flood_runtime_config()
    now = datetime.now(timezone.utc)
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "models": {"flood": {
            "mode": "assisted_review",
            "approved": True,
            "approved_by": "project-owner",
            "approved_at": now.isoformat(),
            "approval_expires_at": (now + timedelta(days=90)).isoformat(),
            "rights_status": "approved",
            "license": "CC BY 4.0",
            "rights_evidence": "https://zenodo.org/records/7566797",
            "model_revision": "doodleverse-floodnet-resunet-v7",
            "model_sha256": model_hash,
            "runtime_config": runtime,
            "field_validated": False,
            "acceptance": acceptance,
        }},
    }), encoding="utf-8")

    result = check_optional_model_release(
        "flood", model_path, "doodleverse-floodnet-resunet-v7", manifest_path,
        readiness_report_path=None, runtime_config=runtime,
        release_mode="assisted_review",
    )

    assert result == {"approved": True, "reason": "approved"}


def test_assisted_review_release_does_not_claim_campus_field_validation(tmp_path):
    model_path = tmp_path / "flood-v7.onnx"
    model_path.write_bytes(b"verified-fp32-onnx")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    runtime = assisted_flood_runtime_config()
    acceptance = {}
    for name in ("parity", "resource", "workflow"):
        evidence_path = tmp_path / f"{name}.json"
        evidence_path.write_text(json.dumps({"status": "passed", "kind": name}), encoding="utf-8")
        acceptance[name] = {
            "status": "passed", "path": str(evidence_path),
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        }
    now = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "models": {"flood": {
            "mode": "assisted_review", "approved": True, "approved_by": "owner",
            "approved_at": now.isoformat(),
            "approval_expires_at": (now + timedelta(days=90)).isoformat(),
            "rights_status": "approved", "license": "CC BY 4.0",
            "rights_evidence": "https://zenodo.org/records/7566797",
            "model_revision": "doodleverse-floodnet-resunet-v7",
            "model_sha256": model_hash, "runtime_config": runtime,
            "field_validated": True,
            "acceptance": acceptance,
        }},
    }
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_optional_model_release(
        "flood", model_path, "doodleverse-floodnet-resunet-v7", manifest_path,
        readiness_report_path=None, runtime_config=runtime,
        release_mode="assisted_review",
    )

    assert result == {"approved": False, "reason": "field_validation_must_remain_false"}
