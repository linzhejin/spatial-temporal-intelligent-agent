import os
import time
import uuid

def manager_client(monkeypatch, tmp_path):
    import api.routes as routes
    from app import create_app

    monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_PASSWORD", "manager-secret")
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(routes.config, "VISION_MAX_MEDIA_BYTES", 24 * 1024 * 1024)
    monkeypatch.setattr(routes.config, "VISION_MAX_VIDEO_BYTES", 1024, raising=False)
    monkeypatch.setattr(routes.config, "VISION_CHUNK_BYTES", 4, raising=False)
    monkeypatch.setattr(routes, "_vision_inference_readiness", lambda *_args: (True, True, True, True))
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="test-secret", DATABASE_URL=None)
    client = app.test_client()
    login = client.post("/api/admin/login", json={"password": "manager-secret"})
    assert login.status_code == 200
    return client, login.get_json()["data"]["csrf_token"]


def test_large_video_upload_resumes_by_offset_and_creates_private_job(monkeypatch, tmp_path):
    import api.routes as routes
    from storage import database, vision_repository

    client, csrf = manager_client(monkeypatch, tmp_path)
    monkeypatch.setattr(database, "initialize", lambda *_args: None)
    created = []

    def create_job(*args, **kwargs):
        created.append(kwargs)
        return {
            "job_id": str(uuid.uuid4()), "status": "queued",
            "original_name": kwargs["original_name"], "media_path": kwargs["media_path"],
            "media_kind": kwargs["media_kind"], "anchor_gcj": kwargs["anchor_gcj"],
            "camera_stabilized": kwargs["camera_stabilized"],
        }

    monkeypatch.setattr(vision_repository, "create_job", create_job)
    headers = {"X-CSRF-Token": csrf}
    started = client.post("/api/manager/vision-uploads", json={
        "filename": "campus.mp4", "size": 12,
        "camera_stabilized": True, "captured_at": time.time(),
        "observation_regions": [{"id": "lane", "kind": "vehicle_lane",
                                  "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
    }, headers=headers)
    assert started.status_code == 201
    upload_id = started.get_json()["data"]["upload_id"]
    assert started.get_json()["data"]["chunk_size"] == 4

    status = client.get(f"/api/manager/vision-uploads/{upload_id}")
    assert status.status_code == 200
    assert status.get_json()["data"]["offset"] == 0

    first = client.put(f"/api/manager/vision-uploads/{upload_id}/chunks",
                       data=b"\x00\x00\x00\x18", headers={**headers, "Upload-Offset": "0"})
    assert first.status_code == 200
    assert first.get_json()["data"]["offset"] == 4
    out_of_order = client.put(f"/api/manager/vision-uploads/{upload_id}/chunks",
                              data=b"isom", headers={**headers, "Upload-Offset": "8"})
    assert out_of_order.status_code == 409

    for offset, chunk in ((4, b"ftyp"), (8, b"isom")):
        response = client.put(f"/api/manager/vision-uploads/{upload_id}/chunks",
                              data=chunk, headers={**headers, "Upload-Offset": str(offset)})
        assert response.status_code == 200
    finished = client.post(f"/api/manager/vision-uploads/{upload_id}/complete",
                           json={}, headers=headers)
    assert finished.status_code == 202
    job = finished.get_json()["data"]["job"]
    assert job["status"] == "queued"
    assert "media_path" not in job and "sha256" not in job
    assert created[0]["camera_stabilized"] is True
    assert created[0]["anchor_gcj"] is None
    assert created[0]["observation_regions"][0]["id"] == "lane"
    assert created[0]["captured_at"].tzinfo is not None
    assert os.path.isfile(os.path.join(str(tmp_path), created[0]["media_path"]))
    resumed = client.get(f"/api/manager/vision-uploads/{upload_id}")
    assert resumed.status_code == 200
    assert resumed.get_json()["data"]["completed"] is True
    assert resumed.get_json()["data"]["job"]["job_id"] == job["job_id"]
    repeated = client.post(f"/api/manager/vision-uploads/{upload_id}/complete", json={}, headers=headers)
    assert repeated.status_code == 202
    assert repeated.get_json()["data"]["job"]["job_id"] == job["job_id"]
    assert len(created) == 1


def test_video_upload_requires_admin_csrf_and_enforces_declared_size(monkeypatch, tmp_path):
    from app import create_app

    client, csrf = manager_client(monkeypatch, tmp_path)
    started_without_csrf = client.post("/api/manager/vision-uploads", json={
        "filename": "campus.mp4", "size": 12, "lng": 114.363, "lat": 30.5365,
    })
    assert started_without_csrf.status_code == 403
    oversized = client.post("/api/manager/vision-uploads", json={
        "filename": "campus.mp4", "size": 1025, "lng": 114.363, "lat": 30.5365,
    }, headers={"X-CSRF-Token": csrf})
    assert oversized.status_code == 413

    anonymous = create_app().test_client()
    denied = anonymous.post("/api/manager/vision-uploads", json={
        "filename": "campus.mp4", "size": 12, "lng": 114.363, "lat": 30.5365,
    })
    assert denied.status_code == 401


def test_manager_vision_status_distinguishes_unconfigured_missing_and_failed_optional_models(
    monkeypatch, tmp_path,
):
    import api.routes as routes

    client, _csrf = manager_client(monkeypatch, tmp_path)
    missing_model = tmp_path / "missing-aider.onnx"
    failed_model = tmp_path / "bad-flood.onnx"
    failed_model.write_bytes(b"not-an-onnx-model")
    monkeypatch.setattr(routes.config, "VISION_ACCIDENT_MODEL_PATH", str(missing_model))
    monkeypatch.setattr(routes.config, "VISION_FLOOD_MODEL_PATH", str(failed_model))
    monkeypatch.setattr(routes.config, "VISION_ACCIDENT_ENABLED", True)
    monkeypatch.setattr(routes.config, "VISION_FLOOD_ENABLED", True)
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_args: {
        "accident_recognition_supported": False,
        "flood_segmentation_supported": False,
    })

    response = client.get("/api/manager/vision-status")

    assert response.status_code == 200
    status = response.get_json()["data"]["optional_model_status"]
    assert status == {"accident": "file_missing", "flood": "load_failed"}


def test_manager_status_keeps_present_optional_models_disabled_by_default(monkeypatch, tmp_path):
    import api.routes as routes

    client, _csrf = manager_client(monkeypatch, tmp_path)
    accident_model = tmp_path / "aider.onnx"
    flood_model = tmp_path / "flood.onnx"
    accident_model.write_bytes(b"present")
    flood_model.write_bytes(b"present")
    monkeypatch.setattr(routes.config, "VISION_ACCIDENT_MODEL_PATH", str(accident_model))
    monkeypatch.setattr(routes.config, "VISION_FLOOD_MODEL_PATH", str(flood_model))
    monkeypatch.setattr(routes.config, "VISION_ACCIDENT_ENABLED", False)
    monkeypatch.setattr(routes.config, "VISION_FLOOD_ENABLED", False)
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_args: {
        "accident_recognition_supported": True,
        "flood_segmentation_supported": True,
    })

    response = client.get("/api/manager/vision-status")

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["optional_model_status"] == {"accident": "disabled", "flood": "disabled"}
    assert data["capabilities"]["accident_recognition_supported"] is False
    assert data["capabilities"]["flood_segmentation_supported"] is False


def test_video_upload_requires_capture_time_and_normalized_roi(monkeypatch, tmp_path):
    import time

    client, csrf = manager_client(monkeypatch, tmp_path)
    headers = {"X-CSRF-Token": csrf}
    payload = {"filename": "campus.mp4", "size": 12, "lng": 114.363, "lat": 30.5365}
    missing_time = client.post("/api/manager/vision-uploads", json=payload, headers=headers)
    assert missing_time.status_code == 400
    payload["captured_at"] = time.time()
    payload["observation_regions"] = [{"kind": "vehicle_lane", "polygon": [[0, 0], [2, 0], [1, 1]]}]
    invalid_roi = client.post("/api/manager/vision-uploads", json=payload, headers=headers)
    assert invalid_roi.status_code == 400
    payload["observation_regions"] = [{"kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1]]}]
    valid = client.post("/api/manager/vision-uploads", json=payload, headers=headers)
    assert valid.status_code == 201

    payload["observation_regions"] = [
        {"id": "same", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1]]},
        {"id": "same", "kind": "road_surface", "polygon": [[0, 0], [1, 0], [1, 1]]},
    ]
    duplicate_id = client.post("/api/manager/vision-uploads", json=payload, headers=headers)
    assert duplicate_id.status_code == 400
    assert duplicate_id.get_json()["error"] == "invalid_observation_metadata"


def test_image_upload_requires_actual_capture_time(monkeypatch, tmp_path):
    from io import BytesIO
    from storage import database, vision_repository

    client, csrf = manager_client(monkeypatch, tmp_path)
    monkeypatch.setattr(database, "initialize", lambda *_args: None)
    monkeypatch.setattr(vision_repository, "create_job", lambda *_args, **kwargs: {
        "job_id": str(uuid.uuid4()), "status": "queued", "media_path": kwargs["media_path"],
        "media_kind": kwargs["media_kind"], "original_name": kwargs["original_name"],
    })
    image = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    headers = {"X-CSRF-Token": csrf}
    missing_time = client.post("/api/manager/vision-jobs", data={
        "lng": "114.363", "lat": "30.5365", "media": (BytesIO(image), "drone.png"),
    }, headers=headers, content_type="multipart/form-data")
    assert missing_time.status_code == 400
    assert missing_time.get_json()["error"] == "invalid_observation_metadata"

    supplied_time = client.post("/api/manager/vision-jobs", data={
        "lng": "114.363", "lat": "30.5365", "captured_at": time.time(),
        "media": (BytesIO(image), "drone.png"),
    }, headers=headers, content_type="multipart/form-data")
    assert supplied_time.status_code == 202


def test_manager_can_cancel_retry_and_review_candidates_independently(monkeypatch, tmp_path):
    from storage import database, vision_repository

    client, csrf = manager_client(monkeypatch, tmp_path)
    monkeypatch.setattr(database, "initialize", lambda *_args: None)
    job_id = str(uuid.uuid4())
    candidate = {"kind": "possible_congestion", "review_required": True, "auto_publish": False}
    job = {"job_id": job_id, "status": "needs_review", "review_status": None,
           "result": {"candidates": [candidate, candidate]}}
    monkeypatch.setattr(vision_repository, "get_job", lambda *_args: job)
    review_calls = []

    def review_candidate(*_args, **kwargs):
        review_calls.append(kwargs)
        return ({"job_id": job_id, "candidate_reviews": {str(kwargs["candidate_index"]): {
            "status": kwargs["review_status"], "note": kwargs["review_note"],
            "reviewed_by": kwargs["reviewed_by"],
        }}} if len(review_calls) == 1 else None)

    monkeypatch.setattr(vision_repository, "review_candidate", review_candidate)
    headers = {"X-CSRF-Token": csrf}
    body = {"status": "confirmed", "note": "逐帧核对后确认该候选需要人工继续处理。"}
    reviewed = client.post(f"/api/manager/vision-jobs/{job_id}/candidates/1/review",
                           json=body, headers=headers)
    assert reviewed.status_code == 200
    assert review_calls[0]["candidate_index"] == 1
    assert review_calls[0]["review_status"] == "confirmed"
    repeated = client.post(f"/api/manager/vision-jobs/{job_id}/candidates/1/review",
                           json=body, headers=headers)
    assert repeated.status_code == 409

    monkeypatch.setattr(vision_repository, "request_cancel_job", lambda *_args: {
        "job_id": job_id, "status": "running", "cancel_requested": True,
        "progress": {"phase": "analyzing", "percent": 30},
    })
    cancelled = client.post(f"/api/manager/vision-jobs/{job_id}/cancel", json={}, headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.get_json()["data"]["job"]["cancel_requested"] is True

    monkeypatch.setattr(vision_repository, "retry_job", lambda *_args: {
        "job_id": job_id, "status": "queued", "progress": {"phase": "queued", "percent": 0},
    })
    retried = client.post(f"/api/manager/vision-jobs/{job_id}/retry", json={}, headers=headers)
    assert retried.status_code == 200
    assert retried.get_json()["data"]["job"]["status"] == "queued"


def test_flood_candidate_is_reviewable_but_still_requires_road_event_confirmation(monkeypatch, tmp_path):
    from storage import database, vision_repository

    client, csrf = manager_client(monkeypatch, tmp_path)
    monkeypatch.setattr(database, "initialize", lambda *_args: None)
    job_id = str(uuid.uuid4())
    candidate = {"kind": "possible_flooding", "review_required": True, "auto_publish": False}
    job = {"job_id": job_id, "status": "needs_review", "captured_at": time.time(),
           "result": {"candidates": [candidate]}}
    monkeypatch.setattr(vision_repository, "get_job", lambda *_args: job)
    reviewed = []
    monkeypatch.setattr(vision_repository, "review_candidate", lambda *_args, **kwargs: (
        reviewed.append(kwargs) or {"job_id": job_id, "candidate_reviews": {"0": {
            "status": kwargs["review_status"], "note": kwargs["review_note"],
        }}}
    ))

    response = client.post(
        f"/api/manager/vision-jobs/{job_id}/candidates/0/review",
        json={"status": "confirmed", "note": "查看原片后确认存在积水候选，继续选择具体道路。"},
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 200
    assert reviewed[0]["review_status"] == "confirmed"


def test_retry_resets_attempt_budget_and_keeps_video_checkpoint(monkeypatch):
    statements = []

    class Cursor:
        def fetchone(self):
            return {"job_id": "retryable", "status": "queued", "progress": {"phase": "queued"}}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _parameters):
            statements.append(statement)
            return Cursor()

    from storage import database, vision_repository
    monkeypatch.setattr(database, "connect", lambda *_args: Connection())
    result = vision_repository.retry_job(None, "retryable")
    assert result["status"] == "queued"
    statement = " ".join(statements[0].split())
    assert "attempts=0" in statement
    assert "checkpoint" not in statement


def test_finishing_job_atomically_honors_a_late_cancel_request(monkeypatch):
    statements = []

    class Cursor:
        def fetchone(self):
            return {"status": "cancelled"}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _parameters):
            statements.append(statement)
            return Cursor()

    from storage import database, vision_repository
    monkeypatch.setattr(database, "connect", lambda *_args: Connection())

    status = vision_repository.finish_job(
        None, "job", "worker", status="needs_review",
        result={"candidates": [{"kind": "possible_congestion"}]},
    )

    statement = " ".join(statements[0].split())
    assert "CASE WHEN cancel_requested THEN 'cancelled' ELSE %s END" in statement
    assert "CASE WHEN cancel_requested THEN NULL ELSE %s END" in statement
    assert status == "cancelled"
