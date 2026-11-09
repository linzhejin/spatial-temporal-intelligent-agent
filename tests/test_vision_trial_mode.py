"""Trial media must stay separate from route-affecting vision jobs."""
import io
import json

import pytest

from storage import database, vision_repository

DATABASE_URL = None
pytestmark = pytest.mark.usefixtures("isolated_test_database")


@pytest.fixture(autouse=True)
def empty_trial_test_queue(isolated_test_database):
    database.initialize(isolated_test_database)
    with database.connect(isolated_test_database) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")


def manager_client(monkeypatch, tmp_path):
    import api.routes as routes
    from app import create_app

    monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_PASSWORD", "trial-test")
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(routes.config, "VISION_TRIAL_ENABLED", True, raising=False)
    monkeypatch.setattr(routes, "_vision_inference_readiness", lambda *_: (True, True, True, True))
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_: {
        "trial_accident_supported": True, "trial_flood_supported": True,
    })
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="trial-test", DATABASE_URL=DATABASE_URL)
    client = app.test_client()
    login = client.post("/api/admin/login", json={"password": "trial-test"})
    assert login.status_code == 200
    return client, {"X-CSRF-Token": login.get_json()["data"]["csrf_token"]}


def test_trial_purpose_is_immutable_and_result_retains_legacy_replay_guard():
    database.initialize(DATABASE_URL)
    job = vision_repository.create_job(
        DATABASE_URL, created_by="test", original_name="public.png", media_kind="image",
        media_path="unused-trial.png", sha256="1" * 64, anchor_gcj=None, purpose="trial",
    )
    assert job["purpose"] == "trial"
    claimed = vision_repository.claim_next_job(DATABASE_URL, "trial-test-worker", allow_trial=True)
    assert claimed["purpose"] == "trial"
    assert vision_repository.finish_job(DATABASE_URL, job["job_id"], "trial-test-worker",
                                        status="completed", result={"candidates": []}) == "completed"
    stored = vision_repository.get_job(DATABASE_URL, job["job_id"])
    assert "research_replay" in stored["result"]
    with pytest.raises(Exception):
        with database.connect(DATABASE_URL) as conn:
            conn.execute("UPDATE manager_vision_job SET purpose='operational' WHERE job_id=%s", (job["job_id"],))
    with database.connect(DATABASE_URL) as conn:
        conn.execute("UPDATE manager_vision_job SET result=%s WHERE job_id=%s", ('{"candidates":[]}', job["job_id"]))
    assert "research_replay" in vision_repository.get_job(DATABASE_URL, job["job_id"])["result"]


def test_trial_image_can_be_submitted_without_inventing_a_capture_time(monkeypatch, tmp_path):
    database.initialize(DATABASE_URL)
    client, headers = manager_client(monkeypatch, tmp_path)
    response = client.post("/api/manager/vision-jobs", headers=headers, data={
        "purpose": "trial",
        "observation_regions": json.dumps([{"id": "road", "kind": "road_surface",
                                            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]),
        "media": (io.BytesIO(b"\x89PNG\r\n\x1a\ntrial"), "public.png", "image/png"),
    })
    assert response.status_code == 202, response.get_json()
    job = vision_repository.get_job(DATABASE_URL, response.get_json()["data"]["job"]["job_id"])
    assert job["purpose"] == "trial"
    assert job["captured_at"] is None
    assert job["anchor_gcj"] is None


def test_trial_missed_case_can_be_recorded_after_no_candidate(monkeypatch, tmp_path):
    database.initialize(DATABASE_URL)
    client, headers = manager_client(monkeypatch, tmp_path)
    job = vision_repository.create_job(
        DATABASE_URL, created_by="test", original_name="public.png", media_kind="image",
        media_path="unused-feedback.png", sha256="2" * 64, anchor_gcj=None, purpose="trial",
    )
    claimed = vision_repository.claim_next_job(DATABASE_URL, "trial-feedback-worker", allow_trial=True)
    assert claimed["job_id"] == job["job_id"]
    vision_repository.finish_job(DATABASE_URL, job["job_id"], "trial-feedback-worker",
                                 status="completed", result={"candidates": []})
    response = client.post(f"/api/manager/vision-jobs/{job['job_id']}/trial-feedback", headers=headers,
                           json={"kind": "possible_flooding", "note": "画面中路面有明显积水"})
    assert response.status_code == 200, response.get_json()
    stored = vision_repository.get_job(DATABASE_URL, job["job_id"])
    assert stored["trial_feedback"][0]["kind"] == "possible_flooding"


def test_only_trial_models_process_trial_jobs(monkeypatch, tmp_path):
    from vision import worker

    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    observed = []
    monkeypatch.setattr(worker, "analyze_media", lambda *args, **kwargs: (
        observed.append((kwargs["accident_model"], kwargs["flood_segmenter"]))
        or {"candidates": [], "safety": {"automatically_changes_routing": False}}
    ))
    common = dict(created_by="test", original_name="one.png", media_kind="image",
                  media_path="unused.png", anchor_gcj=None)
    trial = vision_repository.create_job(DATABASE_URL, sha256="3" * 64, purpose="trial", **common)
    live = vision_repository.create_job(DATABASE_URL, sha256="4" * 64, **common)
    live_model, test_model = object(), object()
    first = worker.process_next_job(DATABASE_URL, "live-worker", accident_model=live_model)
    assert first["job_id"] == live["job_id"]
    second = worker.process_next_job(DATABASE_URL, "trial-worker", accident_model=live_model,
                                     trial_accident_model=test_model)
    assert second["job_id"] == trial["job_id"]
    assert observed == [(live_model, None), (test_model, None)]
    assert "research_replay" in vision_repository.get_job(DATABASE_URL, trial["job_id"])["result"]


def test_trial_cannot_publish_even_after_confirmation(monkeypatch, tmp_path):
    client, headers = manager_client(monkeypatch, tmp_path)
    job = vision_repository.create_job(
        DATABASE_URL, created_by="test", original_name="public.png", media_kind="image",
        media_path="unused.png", sha256="5" * 64, anchor_gcj=None, purpose="trial",
    )
    vision_repository.claim_next_job(DATABASE_URL, "trial-worker", allow_trial=True)
    candidate = {"kind": "possible_accident", "review_required": True, "auto_publish": False}
    vision_repository.finish_job(DATABASE_URL, job["job_id"], "trial-worker", status="needs_review",
                                 result={"candidates": [candidate]})
    vision_repository.review_candidate(DATABASE_URL, job["job_id"], candidate_index=0,
                                       review_status="confirmed", review_note="测试确认已查看原始视频",
                                       reviewed_by="test")
    response = client.post("/api/road-conditions", headers=headers, json={
        "type": "event", "name": "测试影像", "lng": 114.363, "lat": 30.5365,
        "source_vision_job_id": job["job_id"], "source_vision_candidate_index": 0,
        "field_confirmation": "测试确认已查看原始视频和道路",
    })
    assert response.status_code == 409, response.get_json()
    assert response.get_json()["error"] == "vision_trial_not_publishable"


def test_deleting_trial_duplicate_does_not_delete_operational_job():
    common = dict(created_by="test", original_name="same.png", media_kind="image",
                  sha256="6" * 64, anchor_gcj=None)
    live = vision_repository.create_job(DATABASE_URL, media_path="live.png", **common)
    trial = vision_repository.create_job(DATABASE_URL, media_path="trial.png", purpose="trial", **common)
    deleted = vision_repository.begin_delete_jobs(DATABASE_URL, trial["job_id"])
    assert [item["job_id"] for item in deleted] == [trial["job_id"]]
    assert vision_repository.get_job(DATABASE_URL, live["job_id"]) is not None


def test_trial_video_upload_keeps_its_purpose_across_resume(monkeypatch, tmp_path):
    import api.routes as routes

    monkeypatch.setattr(routes.config, "VISION_CHUNK_BYTES", 4)
    client, headers = manager_client(monkeypatch, tmp_path)
    region = {"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    started = client.post("/api/manager/vision-uploads", headers=headers, json={
        "purpose": "trial", "filename": "public.mp4", "size": 12, "observation_regions": [region],
    })
    assert started.status_code == 201, started.get_json()
    upload_id = started.get_json()["data"]["upload_id"]
    for offset, chunk in ((0, b"\x00\x00\x00\x18"), (4, b"ftyp"), (8, b"isom")):
        response = client.put(f"/api/manager/vision-uploads/{upload_id}/chunks", data=chunk,
                              headers={**headers, "Upload-Offset": str(offset)})
        assert response.status_code == 200
    response = client.post(f"/api/manager/vision-uploads/{upload_id}/complete", headers=headers, json={})
    assert response.status_code == 202, response.get_json()
    job_id = response.get_json()["data"]["job"]["job_id"]
    saved = vision_repository.get_job(DATABASE_URL, job_id)
    assert saved["purpose"] == "trial" and saved["captured_at"] is None
    repeated = client.post(f"/api/manager/vision-uploads/{upload_id}/complete", headers=headers, json={})
    assert repeated.get_json()["data"]["job"]["job_id"] == job_id


def test_trial_requires_a_usable_observation_region_and_cannot_claim_campus_location(monkeypatch, tmp_path):
    client, headers = manager_client(monkeypatch, tmp_path)
    def upload(**values):
        return client.post("/api/manager/vision-jobs", headers=headers, data={
            "purpose": "trial", "media": (io.BytesIO(b"\x89PNG\r\n\x1a\ntrial"), "public.png"), **values,
        })
    assert upload().get_json()["error"] == "trial_region_required"
    assert upload(lng="114.363", lat="30.5365").get_json()["error"] == "trial_location_not_allowed"


def test_trial_feedback_rejects_operational_jobs(monkeypatch, tmp_path):
    client, headers = manager_client(monkeypatch, tmp_path)
    job = vision_repository.create_job(DATABASE_URL, created_by="test", original_name="live.png",
        media_kind="image", media_path="live.png", sha256="7" * 64, anchor_gcj=None)
    response = client.post(f"/api/manager/vision-jobs/{job['job_id']}/trial-feedback", headers=headers,
                           json={"kind": "possible_accident", "note": "画面中出现疑似事故"})
    assert response.status_code == 409


def test_video_reports_actual_specialized_observations_separately():
    from vision.engine import _merge_segment_analyses

    segment = {"start_seconds": 0, "end_seconds": 2, "metrics": {"frames_analyzed": 3},
               "candidates": [], "specialized_observations": [
                   {"time_seconds": 0, "traffic_accident_probability": 0.3},
                   {"time_seconds": 1, "traffic_accident_probability": 0.4},
                   {"time_seconds": 1, "flooded_road_area_ratio": 0.1},
               ]}
    merged = _merge_segment_analyses([segment])
    assert merged["metrics"]["specialized_model_frames"] == {"accident": 2, "flood": 1}


def test_trial_coverage_does_not_claim_a_model_ran_without_observations(monkeypatch, tmp_path):
    from vision import worker
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "analyze_media", lambda *args, **kwargs: {
        "candidates": [], "safety": {"accident_recognition_supported": True,
                                    "flood_segmentation_supported": True},
        "metrics": {"specialized_model_frames": {"accident": 1, "flood": 0}},
    })
    job = vision_repository.create_job(DATABASE_URL, created_by="test", original_name="one.png",
        media_kind="image", media_path="unused.png", sha256="8" * 64, anchor_gcj=None, purpose="trial",
        observation_regions=[{"id": "road", "kind": "road_surface", "polygon": [[0, 0], [1, 0], [1, 1]]}])
    worker.process_next_job(DATABASE_URL, "trial-observations", trial_accident_model=object(),
                            trial_flood_segmenter=object())
    result = vision_repository.get_job(DATABASE_URL, job["job_id"])["result"]
    assert result["trial_coverage"] == {"accident": "checked", "flood": "no_observation"}


def test_temporal_accident_runs_only_on_trial_video_and_preserves_window_evidence(monkeypatch, tmp_path):
    from vision import worker

    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    observed = []
    monkeypatch.setattr(worker, "analyze_media", lambda *args, **kwargs: (
        observed.append(kwargs["accident_model"]) or {
            "candidates": [], "safety": {}, "metrics": {"specialized_model_frames": {}},
        }
    ))
    monkeypatch.setattr(worker, "attach_evidence_clips", lambda *args, **kwargs: {"clips_created": 0})

    class TemporalModel:
        trial_video_only = True
        model_version = "trial-sha256:test"

        def analyze_video(self, path, *, cancel_check, deadline):
            assert not cancel_check() and deadline > 0
            return {"windows_analyzed": 1, "windows": [{"start_seconds": 0, "end_seconds": 20, "score": .9}],
                    "candidates": [{"kind": "possible_accident", "review_required": True,
                                    "auto_publish": False, "evidence": {"summary": {"model_scope": "temporal_window"},
                                                                           "segments": [{"start_seconds": 0, "end_seconds": 20}]}}]}

    job = vision_repository.create_job(DATABASE_URL, created_by="test", original_name="one.mp4",
        media_kind="video", media_path="unused.mp4", sha256="9" * 64, anchor_gcj=None, purpose="trial")
    outcome = worker.process_next_job(DATABASE_URL, "trial-temporal", trial_accident_model=TemporalModel())
    assert outcome["status"] == "needs_review" and observed == []
    result = vision_repository.get_job(DATABASE_URL, job["job_id"])["result"]
    assert result["trial_coverage"]["accident"] == "checked"
    assert result["trial_model_details"]["accident"]["model_version"] == "trial-sha256:test"
    assert result["trial_temporal_only"] is True
    assert result["candidates"][0]["evidence"]["summary"]["model_scope"] == "temporal_window"
    assert "research_replay" in result


def test_temporal_only_trial_rejects_images_but_accepts_video_without_roi(monkeypatch, tmp_path):
    import api.routes as routes

    client, headers = manager_client(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_: {
        "trial_accident_supported": True, "trial_accident_media": "video", "trial_flood_supported": False,
    })
    image = client.post("/api/manager/vision-jobs", headers=headers, data={
        "purpose": "trial", "media": (io.BytesIO(b"\x89PNG\r\n\x1a\ntrial"), "public.png"),
    })
    assert image.status_code == 400 and image.get_json()["error"] == "trial_video_required"
    monkeypatch.setattr(routes.config, "VISION_CHUNK_BYTES", 4)
    video = client.post("/api/manager/vision-uploads", headers=headers, json={
        "purpose": "trial", "filename": "public.mp4", "size": 12,
    })
    assert video.status_code == 201, video.get_json()


def test_flood_trial_does_not_run_unneeded_vehicle_detector(monkeypatch, tmp_path):
    from vision import worker

    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(worker, "analyze_media", lambda *args, **kwargs: (
        calls.append(kwargs) or {"candidates": [], "metrics": {"specialized_model_frames": {"flood": 1}},
                                 "safety": {"flood_segmentation_supported": True}}
    ))
    region = {"id": "road", "kind": "road_surface", "polygon": [[0, 0], [1, 0], [1, 1]]}
    job = vision_repository.create_job(DATABASE_URL, created_by="test", original_name="flood.png",
        media_kind="image", media_path="unused.png", sha256="a" * 64, anchor_gcj=None, purpose="trial",
        observation_regions=[region])
    live_detector = object()
    outcome = worker.process_next_job(DATABASE_URL, "trial-light-flood", detector=live_detector,
                                      trial_flood_segmenter=object())
    assert outcome["status"] == "completed"
    assert calls[0]["detector"] is not live_detector
    assert calls[0]["detector"].detect(None) == []
    assert calls[0]["target_sample_fps"] == 1.0
    result = vision_repository.get_job(DATABASE_URL, job["job_id"])["result"]
    assert result["trial_specialized_only"] is True and result["trial_coverage"]["flood"] == "checked"


def test_dedicated_trial_worker_claims_no_operational_jobs():
    common = dict(created_by="test", original_name="one.png", media_kind="image", anchor_gcj=None)
    live = vision_repository.create_job(DATABASE_URL, media_path="live.png", sha256="b" * 64, **common)
    trial = vision_repository.create_job(DATABASE_URL, media_path="trial.png", sha256="c" * 64,
                                         purpose="trial", **common)
    claimed_trial = vision_repository.claim_next_job(DATABASE_URL, "dedicated-trial", allow_trial=True,
                                                      trial_only=True)
    assert claimed_trial["job_id"] == trial["job_id"]
    claimed_live = vision_repository.claim_next_job(DATABASE_URL, "normal-worker")
    assert claimed_live["job_id"] == live["job_id"]


def test_trial_and_operational_worker_heartbeats_are_independently_reported():
    import api.routes as routes

    vision_repository.heartbeat_worker(DATABASE_URL, "normal-worker", ready=True, role="operational",
        status_detail=json.dumps({"accident_recognition_supported": False}))
    vision_repository.heartbeat_worker(DATABASE_URL, "dedicated-trial", ready=True, role="trial",
        status_detail=json.dumps({"trial_accident_supported": True, "trial_accident_media": "video",
                                  "trial_flood_supported": True}))
    detail = routes._vision_worker_capabilities(DATABASE_URL)
    assert detail["accident_recognition_supported"] is False
    assert detail["trial_accident_supported"] and detail["trial_flood_supported"]
    assert vision_repository.has_ready_worker(DATABASE_URL, role="operational")
    vision_repository.heartbeat_worker(DATABASE_URL, "normal-worker", ready=False, role="operational")
    assert not vision_repository.has_ready_worker(DATABASE_URL, role="operational")
    assert vision_repository.has_ready_worker(DATABASE_URL, role="trial")


def test_systemd_templates_keep_trial_capacity_separate_from_campus_worker():
    from pathlib import Path
    systemd = Path(__file__).resolve().parents[1] / "ops" / "systemd"
    campus = (systemd / "whu-vision-worker.service").read_text(encoding="utf8")
    trial = (systemd / "whu-vision-trial-worker.service").read_text(encoding="utf8")
    assert "Environment=VISION_WORKER_ROLE=operational" in campus
    assert "Environment=VISION_WORKER_ROLE=trial" in trial
    assert "MemoryMax=640M" in trial and "MemorySwapMax=0" in trial
    assert "python -m vision.worker" in trial
