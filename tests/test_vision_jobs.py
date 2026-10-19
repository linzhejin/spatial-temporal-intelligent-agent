import io
import time

import pytest

from storage import database, vision_repository

DATABASE_URL = None  # assigned only after isolated_test_database validates the target
pytestmark = pytest.mark.usefixtures("isolated_test_database")


def manager_client(monkeypatch, tmp_path):
    import api.routes as routes
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    from app import create_app

    monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_PASSWORD", "manager-secret")
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "_vision_inference_readiness", lambda *_args: (True, True, True, True))
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="test-secret", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    login = client.post("/api/admin/login", json={"password": "manager-secret"})
    assert login.status_code == 200
    return client, login.get_json()["data"]["csrf_token"]


def test_manager_upload_is_private_and_queues_reviewable_media(monkeypatch, tmp_path):
    from vision import worker

    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job")
    client, csrf = manager_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/manager/vision-jobs",
        data={
            "lng": "114.363", "lat": "30.5365",
            "captured_at": str(time.time()),
            "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nsynthetic-public-test"), "campus.png", "image/png"),
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 202
    payload = response.get_json()["data"]["job"]
    assert payload["status"] == "queued"
    assert "media_path" not in payload
    assert "sha256" not in payload

    monkeypatch.setattr(worker, "analyze_media", lambda *args, **kwargs: {
        "candidates": [{"kind": "vehicle_cluster_review", "review_required": True, "auto_publish": False}],
        "metrics": {"frames_analyzed": 1},
        "safety": {"automatically_changes_routing": False},
    })
    outcome = worker.process_next_job(DATABASE_URL, "vision-test-worker", lease_seconds=30)
    assert outcome["status"] == "needs_review"
    job = client.get("/api/manager/vision-jobs/" + payload["job_id"]).get_json()["data"]["job"]
    assert job["result"]["safety"]["automatically_changes_routing"] is False
    assert client.get(job["media_url"]).status_code == 200

    reviewed = client.post(
        "/api/manager/vision-jobs/" + payload["job_id"] + "/review",
        json={"status": "confirmed", "candidate_index": 0,
              "note": "现场复核后再手动选择路段"},
        headers={"X-CSRF-Token": csrf},
    )
    assert reviewed.status_code == 200
    assert reviewed.get_json()["data"]["job"]["review_status"] == "confirmed"
    assert reviewed.get_json()["data"]["job"]["review_candidate_index"] == 0
    assert client.post(
        "/api/manager/vision-jobs/" + payload["job_id"] + "/review",
        json={"status": "dismissed"}, headers={"X-CSRF-Token": csrf},
    ).status_code == 409


def test_upload_rejects_extension_content_mismatch_and_outside_campus(monkeypatch, tmp_path):
    client, csrf = manager_client(monkeypatch, tmp_path)
    bad_file = client.post(
        "/api/manager/vision-jobs",
        data={"lng": "114.363", "lat": "30.5365", "media": (io.BytesIO(b"not an image"), "fake.png")},
        headers={"X-CSRF-Token": csrf},
    )
    assert bad_file.status_code == 400
    outside = client.post(
        "/api/manager/vision-jobs",
        data={"lng": "114.7", "lat": "30.8", "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "campus.png")},
        headers={"X-CSRF-Token": csrf},
    )
    assert outside.status_code == 400


def test_unconfigured_detector_cannot_accept_jobs_that_would_silently_fail(monkeypatch, tmp_path):
    import api.routes as routes

    client, csrf = manager_client(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "_vision_inference_readiness", lambda *_args: (False, False, False, False))
    response = client.post(
        "/api/manager/vision-jobs",
        data={
            "lng": "114.363", "lat": "30.5365",
            "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "campus.png", "image/png"),
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 503
    assert response.get_json()["error"] == "vision_not_ready"
    assert list(tmp_path.iterdir()) == []


def test_vision_review_only_allows_one_audited_decision(monkeypatch):
    database.initialize(DATABASE_URL)
    item = vision_repository.create_job(
        DATABASE_URL, created_by="web", original_name="campus.png", media_kind="image",
        media_path="unused.png", sha256="a" * 64, anchor_gcj={"lng": 114.363, "lat": 30.5365},
    )
    claimed = vision_repository.claim_next_job(DATABASE_URL, "vision-review-test")
    assert claimed["job_id"] == item["job_id"]
    assert vision_repository.finish_job(
        DATABASE_URL, item["job_id"], "vision-review-test", status="needs_review", result={"candidates": []}
    )
    first = vision_repository.review_job(
        DATABASE_URL, item["job_id"], review_status="dismissed", review_note="误报", reviewed_by="web"
    )
    second = vision_repository.review_job(
        DATABASE_URL, item["job_id"], review_status="confirmed", review_note="改判",
        reviewed_by="web", candidate_index=0,
    )
    assert first["review_status"] == "dismissed"
    assert second is None


def test_video_stabilization_flag_is_persisted_and_manager_limit_is_applied(monkeypatch, tmp_path):
    database.initialize(DATABASE_URL)
    with database.connect(DATABASE_URL) as conn:
        conn.execute("TRUNCATE manager_vision_job")
    client, _csrf = manager_client(monkeypatch, tmp_path)
    for index in range(3):
        vision_repository.create_job(
            DATABASE_URL, created_by="web", original_name=f"campus-{index}.mp4",
            media_kind="video", media_path=f"unused-{index}.mp4", sha256=str(index) * 64,
            anchor_gcj={"lng": 114.363, "lat": 30.5365}, camera_stabilized=index == 1,
        )
    jobs = client.get("/api/manager/vision-jobs?limit=2").get_json()["data"]["jobs"]
    assert len(jobs) == 2
    assert any(job["camera_stabilized"] for job in vision_repository.list_jobs(DATABASE_URL))
    assert client.get("/api/manager/vision-jobs?limit=0").status_code == 400


def test_vision_worker_readiness_heartbeat_is_live_and_can_be_revoked():
    database.initialize(DATABASE_URL)
    worker_id = "vision-readiness-test"
    with database.connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM manager_vision_worker WHERE worker_id=%s", (worker_id,))

    assert not vision_repository.has_ready_worker(DATABASE_URL, worker_id=worker_id)
    vision_repository.heartbeat_worker(DATABASE_URL, worker_id, ready=True)
    assert vision_repository.has_ready_worker(DATABASE_URL, worker_id=worker_id)

    vision_repository.heartbeat_worker(
        DATABASE_URL, worker_id, ready=False, status_detail="权重读取失败",
    )
    assert not vision_repository.has_ready_worker(DATABASE_URL, worker_id=worker_id)
