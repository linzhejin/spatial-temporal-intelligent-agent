"""Opt-in release-gated worker startup using real models and a test database."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from storage import database, vision_repository
from vision.model_release import assisted_flood_runtime_config

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


@pytest.mark.skipif(os.getenv("RUN_FLOOD_REAL_WORKER_STARTUP") != "1",
                    reason="requires isolated PostgreSQL, actual model and completed evidence")
def test_release_manifest_starts_real_worker_for_trial_and_operational_images(tmp_path):
    root = Path(r"D:\LuoJiaValidation")
    active = json.loads((root / "flood_resource_active_20261110.json").read_text(encoding="utf-8-sig"))
    evidence_paths = {
        "parity": root / "doodleverse_onnx_parity_low_memory_20261110.json",
        "resource": Path(active["stage"]) / "resource_acceptance.json",
        "workflow": root / "flood_workflow_evidence_20261110.json",
    }
    evidence = {}
    for kind, path in evidence_paths.items():
        assert path.is_file(), str(path)
        assert json.loads(path.read_text(encoding="utf-8"))["status"] == "passed"
        evidence[kind] = {"status": "passed", "path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    model_path = root / "doodleverse_floodnet_v7_fp32.onnx"
    traffic = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    runtime = Path(os.getenv("VISION_FLOOD_TEST_RUNTIME",
                             str(root / "flood_runtime_release_20261110/Scripts/python.exe")))
    image = root / "model_comparison_20261109/flood/road_inundation_delta_ian_2024/samples_v1_2/test/HURRICANE-IAN_000352_image.png"
    assert all(path.is_file() for path in (model_path, traffic, runtime, image))
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    now = datetime.now(timezone.utc)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": {"flood": {
        "mode": "assisted_review", "approved": True, "approved_by": "project-owner-authorized-session",
        "approved_at": now.isoformat(), "approval_expires_at": (now + timedelta(days=90)).isoformat(),
        "rights_status": "approved", "license": "CC BY 4.0",
        "rights_evidence": "https://zenodo.org/records/7566797", "field_validated": False,
        "model_revision": "doodleverse-floodnet-resunet-v7", "model_sha256": model_hash,
        "runtime_config": assisted_flood_runtime_config(), "acceptance": evidence,
    }}}), encoding="utf-8")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    shutil.copy2(image, uploads / image.name)
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    env = os.environ.copy()
    env.update({
        "DATABASE_URL": DATABASE_URL, "VISION_WORKER_ROLE": "trial", "VISION_TRIAL_ENABLED": "true",
        "VISION_FLOOD_ENABLED": "true", "VISION_FLOOD_RELEASE_MODE": "assisted_review",
        "VISION_MODEL_PATH": str(traffic), "VISION_UPLOAD_DIR": str(uploads),
        "VISION_FLOOD_MODEL_PATH": str(model_path), "VISION_FLOOD_REVISION": "doodleverse-floodnet-resunet-v7",
        "VISION_FLOOD_RUNTIME_PYTHON": str(runtime), "VISION_FLOOD_CPU_THREADS": "2",
        "VISION_OPTIONAL_MODELS_RELEASE_MANIFEST": str(manifest), "VISION_CPU_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    })

    def wait_until(predicate, process, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            if process.poll() is not None:
                raise AssertionError("worker exited before completing its test job")
            time.sleep(0.2)
        raise AssertionError("timed out waiting for worker readiness or result")

    for purpose in ("trial", "operational"):
        log_path = tmp_path / (purpose + "-worker.log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "vision.worker"],
                                       cwd=str(Path(__file__).resolve().parents[1]),
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                def approved_worker():
                    statuses = vision_repository.ready_worker_statuses(DATABASE_URL)
                    for status in statuses:
                        detail = json.loads(status.get("status_detail") or "{}")
                        if (status.get("worker_role") == "trial"
                                and detail.get("flood_assistance_release_status") == "approved"
                                and detail.get("flood_assistance_model_sha256") == model_hash):
                            return detail
                    return None
                detail = wait_until(approved_worker, process, timeout=45)
                assert detail.get("flood_assistance_automatic_supported") is False
                assert detail.get("trial_flood_automatic_supported") is False
                region = {"id": "road", "kind": "road_surface", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
                if purpose == "operational":
                    region["flood_assistance_mode"] = "assisted_review"
                job = vision_repository.create_job(
                    DATABASE_URL, created_by="isolated-release-worker", original_name=image.name,
                    media_kind="image", media_path=image.name, sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    anchor_gcj=None if purpose == "trial" else {"lng": 114.363, "lat": 30.5365},
                    captured_at=None if purpose == "trial" else datetime.now(timezone.utc),
                    observation_regions=[region], purpose=purpose,
                )
                def completed():
                    stored = vision_repository.get_job(DATABASE_URL, job["job_id"])
                    return stored if stored and stored["status"] in {"completed", "needs_review", "failed"} else None
                stored = wait_until(completed, process)
                assert stored["status"] == "needs_review", stored["error"]
                assert any(item.get("kind") == "possible_flooding" for item in stored["result"]["candidates"])
                assert stored["result"]["model"]["flood_model"]["sha256"] == model_hash
                if purpose == "trial":
                    assert stored["result"]["trial_coverage"]["flood"] == "checked"
                else:
                    assert stored["result"]["flood_assistance"]["field_validated"] is False
                process.wait(timeout=20)
                assert process.returncode == 0
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
