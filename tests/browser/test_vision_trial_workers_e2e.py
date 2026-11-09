"""Opt-in test of separate real trial and campus worker processes."""
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from storage import database, vision_repository

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")
RUN_E2E = os.getenv("RUN_VISION_TRIAL_REAL_E2E") == "1"


@pytest.mark.skipif(not RUN_E2E, reason="requires isolated PostgreSQL and actual local model weights")
def test_distinct_worker_processes_keep_trial_and_campus_jobs_separate(monkeypatch, tmp_path):
    import config
    from app import create_app
    from vision.worker import VISION_CAPACITY_LOCK_KEY

    root = Path(__file__).resolve().parents[2]
    detector = Path(r"D:\LuoJiaModels\visdrone-rtdetrv4-s.onnx")
    flood = Path(r"D:\LuoJiaValidation\model_trials_20261109\segformer_b0_fwisd_head\segformer-b0-fwisd.onnx")
    accident = Path(r"D:\LuoJiaValidation\model_trials_20261109\temporal-crash-trial.onnx")
    image = Path(r"D:\LuoJiaValidation\public_vision_20261109\fwisd\images\00270.png")
    assert all(item.is_file() for item in (detector, flood, accident, image))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    common_env = os.environ.copy()
    common_env.update({"DATABASE_URL": DATABASE_URL, "VISION_MODEL_PATH": str(detector),
                       "VISION_UPLOAD_DIR": str(uploads), "VISION_CPU_THREADS": "2",
                       "VISION_TRIAL_ENABLED": "true", "VISION_TRIAL_ACCIDENT_KIND": "temporal",
                       "VISION_TRIAL_ACCIDENT_MODEL_PATH": str(accident), "VISION_TRIAL_ACCIDENT_THRESHOLD": ".5",
                       "VISION_TRIAL_FLOOD_MODEL_PATH": str(flood), "VISION_TRIAL_FLOOD_MODEL_ID": "segformer-b0-trial",
                       "VISION_TRIAL_FLOOD_PROBABILITY": ".5", "VISION_ACCIDENT_ENABLED": "false",
                       "VISION_FLOOD_ENABLED": "false", "PYTHONPATH": str(root)})
    processes = []
    streams = []

    def launch(role):
        output = (tmp_path / f"{role}-worker.log").open("w", encoding="utf8")
        streams.append(output)
        env = dict(common_env, VISION_WORKER_ROLE=role)
        process = subprocess.Popen([sys.executable, "-m", "vision.worker"], cwd=root, env=env,
                                   stdout=output, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def wait_until(predicate, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert all(process.poll() is None for process in processes), "worker exited; inspect worker logs"
            value = predicate()
            if value:
                return value
            time.sleep(.2)
        pytest.fail("worker or result timed out; inspect worker logs")

    try:
        trial_process = launch("trial")
        wait_until(lambda: vision_repository.has_ready_worker(DATABASE_URL, role="trial"))
        assert not vision_repository.has_ready_worker(DATABASE_URL, role="operational")
        launch("operational")
        wait_until(lambda: vision_repository.has_ready_worker(DATABASE_URL, role="operational"))
        monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
        monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "trial-workers-secret")
        monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
        monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(uploads))
        monkeypatch.setattr(config, "VISION_MODEL_PATH", str(detector))
        monkeypatch.setattr(config, "VISION_TRIAL_ENABLED", True)
        app = create_app()
        app.config.update(TESTING=True, DATABASE_URL=DATABASE_URL, SECRET_KEY="trial-workers-secret")
        client = app.test_client()
        login = client.post("/api/admin/login", json={"password": "trial-workers-secret"})
        assert login.status_code == 200
        headers = {"X-CSRF-Token": login.get_json()["data"]["csrf_token"]}
        status = client.get("/api/manager/vision-status").get_json()["data"]
        assert status["trial"]["enabled"] and status["trial"]["accident_media"] == "video"
        assert status["trial"]["flood"] and status["inference_ready"]
        region = [{"id": "road", "kind": "road_surface", "polygon":
                   [[.01, .543], [.99, .543], [.99, .945], [.01, .945]]}]
        content = image.read_bytes()
        with database.connect(DATABASE_URL) as lock_connection:
            lock_connection.execute("SELECT pg_advisory_lock(%s)", (VISION_CAPACITY_LOCK_KEY,))
            lock_connection.commit()
            trial = client.post("/api/manager/vision-jobs", headers=headers, data={
                "purpose": "trial", "observation_regions": json.dumps(region),
                "media": (io.BytesIO(content), "public.png", "image/png"),
            })
            assert trial.status_code == 202, trial.get_json()
            live = client.post("/api/manager/vision-jobs", headers=headers, data={
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "media": (io.BytesIO(content), "campus.png", "image/png"),
            })
            assert live.status_code == 202, live.get_json()
            trial_id = trial.get_json()["data"]["job"]["job_id"]
            live_id = live.get_json()["data"]["job"]["job_id"]
            time.sleep(1.5)
            assert all(vision_repository.get_job(DATABASE_URL, item)["status"] == "queued"
                       for item in (trial_id, live_id))
            lock_connection.execute("SELECT pg_advisory_unlock(%s)", (VISION_CAPACITY_LOCK_KEY,))

        def both_finished():
            jobs = [vision_repository.get_job(DATABASE_URL, item) for item in (trial_id, live_id)]
            if all(job["status"] in {"completed", "needs_review", "failed"} for job in jobs):
                return jobs
            return None

        deadline = time.monotonic() + 120
        finished = None
        while time.monotonic() < deadline:
            assert all(process.poll() in (None, 0) for process in processes), "worker failed; inspect worker logs"
            finished = both_finished()
            if finished:
                break
            time.sleep(.2)
        assert finished, "worker results timed out; inspect worker logs"
        trial_job, live_job = finished
        assert trial_job["status"] in {"completed", "needs_review"}, trial_job.get("error")
        assert live_job["status"] in {"completed", "needs_review"}, live_job.get("error")
        assert trial_job["result"]["trial_specialized_only"] is True
        assert trial_job["result"]["trial_coverage"]["flood"] == "checked"
        assert trial_job["result"]["model"]["id"] == "trial-no-vehicle-detection"
        assert live_job["result"]["model"]["id"] != "trial-no-vehicle-detection"
        assert "research_replay" in trial_job["result"] and "research_replay" not in live_job["result"]
        deadline = time.monotonic() + 15
        while trial_process.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        assert trial_process.returncode == 0, "trial worker should recycle after completing a job"
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in streams:
            stream.close()
