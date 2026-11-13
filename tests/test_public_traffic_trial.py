"""Public traffic trials use the normal detector without becoming campus jobs."""

import io
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_vision_upload_contract import _enable_doodleverse_assistance, _manager_client


def enable_public_traffic(monkeypatch):
    import api.routes as routes

    monkeypatch.setattr(routes.config, "VISION_TRIAL_ENABLED", True)
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_args: {
        "trial_traffic_supported": True,
    })


def test_public_video_decoder_threads_are_set_when_opening():
    from vision.engine import _open_video_capture

    calls = []

    class FakeCv2:
        CAP_FFMPEG = 1900
        CAP_PROP_N_THREADS = 70

        @staticmethod
        def VideoCapture(*args):
            calls.append(args)
            return "capture"

    assert _open_video_capture(FakeCv2, Path("sample.mp4"), 1) == "capture"
    assert calls == [("sample.mp4", 1900, [70, 1])]
    with pytest.raises(ValueError):
        _open_video_capture(FakeCv2, Path("sample.mp4"), 0)


@pytest.mark.parametrize("memory_arena", [True, False])
def test_detector_respects_cpu_memory_arena_setting(monkeypatch, tmp_path, memory_arena):
    from types import SimpleNamespace
    from tests.test_vision_onnx_detector import FakeSession
    from vision.engine import OnnxDetector

    class FakeSessionOptions:
        enable_cpu_mem_arena = True

    created = {}

    def make_session(_path, *, sess_options, providers):
        created["arena"] = sess_options.enable_cpu_mem_arena
        return FakeSession()

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=make_session,
    ))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    OnnxDetector(str(model), cpu_memory_arena=memory_arena)
    assert created["arena"] is memory_arena


def test_public_traffic_image_needs_no_region_or_location(monkeypatch, tmp_path):
    enable_public_traffic(monkeypatch)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    response = client.post(
        "/api/manager/vision-jobs",
        data={"purpose": "trial", "trial_mode": "traffic", "observation_regions": "[]",
              "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "public.png", "image/png")},
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 202, response.get_json()
    assert captured["purpose"] == "trial"
    assert captured["anchor_gcj"] is None
    assert captured["observation_regions"] == [{"kind": "public_traffic_analysis"}]


def test_public_traffic_video_keeps_mode_across_chunked_upload(monkeypatch, tmp_path):
    import api.routes as routes

    enable_public_traffic(monkeypatch)
    monkeypatch.setattr(routes.config, "VISION_CHUNK_BYTES", 4)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    headers = {"X-CSRF-Token": csrf}
    started = client.post("/api/manager/vision-uploads", headers=headers, json={
        "purpose": "trial", "trial_mode": "traffic", "filename": "public.mp4", "size": 12,
        "observation_regions": [],
    })
    assert started.status_code == 201, started.get_json()
    upload_id = started.get_json()["data"]["upload_id"]
    for offset, chunk in ((0, b"\x00\x00\x00\x18"), (4, b"ftyp"), (8, b"isom")):
        response = client.put(f"/api/manager/vision-uploads/{upload_id}/chunks", data=chunk,
                              headers={**headers, "Upload-Offset": str(offset)})
        assert response.status_code == 200, response.get_json()
    completed = client.post(f"/api/manager/vision-uploads/{upload_id}/complete", headers=headers, json={})
    assert completed.status_code == 202, completed.get_json()
    assert captured["purpose"] == "trial"
    assert captured["anchor_gcj"] is None
    assert captured["observation_regions"] == [{"kind": "public_traffic_analysis"}]


def test_traffic_trial_cannot_claim_campus_location(monkeypatch, tmp_path):
    enable_public_traffic(monkeypatch)
    client, csrf = _manager_client(monkeypatch, tmp_path, {})
    response = client.post(
        "/api/manager/vision-jobs", headers={"X-CSRF-Token": csrf},
        data={"purpose": "trial", "trial_mode": "traffic", "lng": "114.36", "lat": "30.53",
              "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "public.png", "image/png")},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "trial_location_not_allowed"


def test_public_traffic_requires_a_ready_traffic_worker(monkeypatch, tmp_path):
    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    response = client.post(
        "/api/manager/vision-jobs", headers={"X-CSRF-Token": csrf},
        data={"purpose": "trial", "trial_mode": "traffic",
              "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "public.png", "image/png")},
    )
    assert response.status_code == 503
    assert response.get_json()["error"] == "trial_traffic_not_ready"
    assert not captured


def test_public_flood_mode_accepts_automatic_image_and_rejects_video(monkeypatch, tmp_path):
    _enable_doodleverse_assistance(monkeypatch, tmp_path)
    captured = {}
    client, csrf = _manager_client(monkeypatch, tmp_path, captured)
    headers = {"X-CSRF-Token": csrf}
    image = client.post(
        "/api/manager/vision-jobs", headers=headers,
        data={"purpose": "trial", "trial_mode": "flood", "observation_regions": "[]",
              "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "public.png", "image/png")},
    )
    assert image.status_code == 202, image.get_json()
    assert captured["purpose"] == "trial"
    assert captured["observation_regions"] == [{"kind": "public_flood_analysis"}]
    video = client.post("/api/manager/vision-uploads", headers=headers, json={
        "purpose": "trial", "trial_mode": "flood", "filename": "public.mp4", "size": 12,
    })
    assert video.status_code == 400
    assert video.get_json()["error"] == "trial_image_required"


def test_operational_job_cannot_use_public_traffic_mode(monkeypatch, tmp_path):
    enable_public_traffic(monkeypatch)
    client, csrf = _manager_client(monkeypatch, tmp_path, {})
    response = client.post(
        "/api/manager/vision-jobs", headers={"X-CSRF-Token": csrf},
        data={"purpose": "operational", "trial_mode": "traffic",
              "media": (io.BytesIO(b"\x89PNG\r\n\x1a\nvalid"), "public.png", "image/png")},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_trial_mode"


def test_traffic_trial_uses_detector_without_flood_or_accident(monkeypatch, tmp_path):
    from vision import worker

    captured = {}
    detector = object()
    job = {"job_id": "public-traffic", "purpose": "trial", "media_kind": "video",
           "media_path": "public.mp4", "anchor_gcj": None, "camera_stabilized": False,
           "observation_regions": [{"kind": "public_traffic_analysis"}], "checkpoint": None}
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(worker.vision_repository, "claim_next_job", lambda *_a, **_kw: job)
    monkeypatch.setattr(worker.vision_repository, "update_job_progress",
                        lambda *_a, **_kw: {"cancel_requested": False})
    monkeypatch.setattr(worker.vision_repository, "get_job_cancel_state",
                        lambda *_a, **_kw: {"cancel_requested": False})

    def finish(_url, _job_id, _worker_id, *, status, result=None, error=None):
        captured.update(status=status, result=result, error=error)
        return status

    def analyze(*_args, **kwargs):
        captured["analysis"] = kwargs
        return {"candidates": [], "metrics": {"mean_vehicle_count": 2, "mean_pedestrian_count": 3}}

    monkeypatch.setattr(worker.vision_repository, "finish_job", finish)
    monkeypatch.setattr(worker, "analyze_media", analyze)
    outcome = worker._process_next_job_unlocked(
        "test", "worker", detector=detector, trial_flood_segmenter=object(),
        trial_traffic_supported=True, trial_only=True,
    )
    assert outcome["status"] == "completed", captured
    assert captured["analysis"]["detector"] is detector
    assert captured["analysis"]["accident_model"] is None
    assert captured["analysis"]["flood_segmenter"] is None
    assert captured["analysis"]["regions"] == []
    assert captured["analysis"]["video_decoder_threads"] == 1
    assert captured["result"]["trial_mode"] == "traffic"
    assert not captured["result"].get("trial_specialized_only")


def test_trial_worker_loads_a_memory_bounded_traffic_detector(monkeypatch, tmp_path):
    from vision import worker

    captured = {}
    job = {"job_id": "public-traffic", "purpose": "trial", "media_kind": "video",
           "media_path": "public.mp4", "anchor_gcj": None, "camera_stabilized": False,
           "observation_regions": [{"kind": "public_traffic_analysis"}], "checkpoint": None}
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(worker.vision_repository, "claim_next_job", lambda *_a, **_kw: job)
    monkeypatch.setattr(worker.vision_repository, "update_job_progress",
                        lambda *_a, **_kw: {"cancel_requested": False})
    monkeypatch.setattr(worker.vision_repository, "finish_job", lambda *_a, status, **_kw: status)

    def detector_factory(_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(worker, "OnnxDetector", detector_factory)
    monkeypatch.setattr(worker, "analyze_media", lambda *_a, **_kw: {"candidates": [], "metrics": {}})
    outcome = worker._process_next_job_unlocked("test", "worker", trial_traffic_supported=True, trial_only=True)
    assert outcome["status"] == "completed"
    assert captured["cpu_memory_arena"] is False


def test_explicit_flood_trial_does_not_run_the_accident_model(monkeypatch, tmp_path):
    from vision import worker

    captured = {}
    flood_model = object()
    job = {"job_id": "public-flood", "purpose": "trial", "media_kind": "image",
           "media_path": "public.png", "anchor_gcj": None, "camera_stabilized": False,
           "observation_regions": [{"kind": "public_flood_analysis"}], "checkpoint": None}
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(worker.vision_repository, "claim_next_job", lambda *_a, **_kw: job)
    monkeypatch.setattr(worker.vision_repository, "update_job_progress",
                        lambda *_a, **_kw: {"cancel_requested": False})

    def finish(_url, _job_id, _worker_id, *, status, result=None, error=None):
        captured.update(status=status, result=result, error=error)
        return status

    def analyze(*_args, **kwargs):
        captured["analysis"] = kwargs
        return {"candidates": [], "metrics": {"specialized_model_frames": {"flood": 1}}}

    monkeypatch.setattr(worker.vision_repository, "finish_job", finish)
    monkeypatch.setattr(worker, "analyze_media", analyze)
    outcome = worker._process_next_job_unlocked(
        "test", "worker", trial_accident_model=object(), trial_flood_segmenter=flood_model, trial_only=True,
    )
    assert outcome["status"] == "completed", captured
    assert captured["analysis"]["accident_model"] is None
    assert captured["analysis"]["flood_segmenter"] is flood_model
    assert captured["analysis"]["regions"] == []
    assert captured["result"]["trial_mode"] == "flood"
    assert captured["result"]["trial_specialized_only"] is True


def test_trial_queue_dispatches_traffic_only_to_capable_worker(isolated_test_database):
    from storage import database, vision_repository

    url = isolated_test_database
    database.initialize(url)
    with database.connect(url) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    common = {"created_by": "test", "original_name": "public.png", "media_kind": "image",
              "anchor_gcj": None, "purpose": "trial"}
    traffic = vision_repository.create_job(
        url, media_path="traffic.png", sha256="1" * 64,
        observation_regions=[{"kind": "public_traffic_analysis"}], **common,
    )
    flood = vision_repository.create_job(
        url, media_path="flood.png", sha256="2" * 64,
        observation_regions=[{"kind": "public_flood_analysis"}], **common,
    )
    flood_claim = vision_repository.claim_next_job(
        url, "flood-only", trial_only=True, allow_trial=True, allow_trial_traffic=False,
    )
    traffic_claim = vision_repository.claim_next_job(
        url, "traffic-only", trial_only=True, allow_trial=False, allow_trial_traffic=True,
    )
    assert flood_claim["job_id"] == flood["job_id"]
    assert traffic_claim["job_id"] == traffic["job_id"]
    assert flood_claim["purpose"] == traffic_claim["purpose"] == "trial"


@pytest.mark.skipif(
    not os.getenv("PUBLIC_TRAFFIC_SMOKE_VIDEO") or not os.getenv("PUBLIC_TRAFFIC_SMOKE_MODEL"),
    reason="requires an explicitly supplied public clip and ONNX model",
)
def test_public_video_upload_database_and_real_worker(monkeypatch, tmp_path, isolated_test_database):
    import api.routes as routes
    from app import create_app
    from storage import database, vision_repository
    from vision import worker

    video = Path(os.environ["PUBLIC_TRAFFIC_SMOKE_VIDEO"])
    model = Path(os.environ["PUBLIC_TRAFFIC_SMOKE_MODEL"])
    url = isolated_test_database
    database.initialize(url)
    with database.connect(url) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_PASSWORD", "local-vision-test")
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(routes.config, "VISION_MODEL_PATH", str(model))
    monkeypatch.setattr(routes.config, "VISION_CHUNK_BYTES", 1048576)
    monkeypatch.setattr(routes, "_vision_inference_readiness", lambda *_args: (True, True, True, True))
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_args: {"trial_traffic_supported": True})
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="local-vision-test", DATABASE_URL=url)
    client = app.test_client()
    csrf = client.post("/api/admin/login", json={"password": "local-vision-test"}).get_json()["data"]["csrf_token"]
    headers = {"X-CSRF-Token": csrf}
    content = video.read_bytes()
    started = client.post("/api/manager/vision-uploads", headers=headers, json={
        "purpose": "trial", "trial_mode": "traffic", "filename": "public.mp4", "size": len(content),
        "observation_regions": [],
    })
    assert started.status_code == 201, started.get_json()
    upload_id = started.get_json()["data"]["upload_id"]
    for offset in range(0, len(content), 1048576):
        uploaded = client.put(f"/api/manager/vision-uploads/{upload_id}/chunks",
                              data=content[offset:offset + 1048576],
                              headers={**headers, "Upload-Offset": str(offset)})
        assert uploaded.status_code == 200, uploaded.get_json()
    completed = client.post(f"/api/manager/vision-uploads/{upload_id}/complete", headers=headers, json={})
    assert completed.status_code == 202, completed.get_json()
    job_id = completed.get_json()["data"]["job"]["job_id"]
    outcome = worker._process_next_job_unlocked(url, "public-real-worker", trial_only=True,
                                                trial_traffic_supported=True)
    assert outcome["status"] == "completed", vision_repository.get_job(url, job_id).get("error")
    job = vision_repository.get_job(url, job_id)
    assert job["purpose"] == "trial" and job["anchor_gcj"] is None
    assert job["result"]["trial_mode"] == "traffic"
    assert job["result"]["metrics"]["frames_analyzed"] > 0
    preview = client.get(f"/api/manager/vision-jobs/{job_id}/preview-frames")
    assert preview.status_code == 200
    frames = preview.get_json()["data"]["preview_frames"]
    assert frames and any(frame.get("detections") for frame in frames)
    publish = client.post("/api/road-conditions", headers=headers, json={
        "type": "event", "name": "试用影像", "lng": 114.363, "lat": 30.5365,
        "source_vision_job_id": job_id, "source_vision_candidate_index": 0,
        "field_confirmation": "测试确认查看了原始视频和道路",
    })
    assert publish.status_code == 409
    assert publish.get_json()["error"] == "vision_trial_not_publishable"


@pytest.mark.skipif(
    not os.getenv("PUBLIC_TRAFFIC_SMOKE_VIDEO") or not os.getenv("PUBLIC_TRAFFIC_SMOKE_MODEL"),
    reason="requires an explicitly supplied public clip and ONNX model",
)
def test_dedicated_trial_worker_advertises_and_processes_public_traffic(
        monkeypatch, tmp_path, isolated_test_database):
    import config
    from app import create_app
    from storage import database, vision_repository

    video = Path(os.environ["PUBLIC_TRAFFIC_SMOKE_VIDEO"])
    model = Path(os.environ["PUBLIC_TRAFFIC_SMOKE_MODEL"])
    url = isolated_test_database
    database.initialize(url)
    with database.connect(url) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "local-vision-test")
    monkeypatch.setattr(config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(config, "VISION_MODEL_PATH", str(model))
    monkeypatch.setattr(config, "VISION_WORKER_ROLE", "trial")
    monkeypatch.setattr(config, "VISION_TRIAL_ENABLED", True)
    monkeypatch.setattr(config, "VISION_FLOOD_ENABLED", False)
    monkeypatch.setattr(config, "VISION_TRIAL_ACCIDENT_MODEL_PATH", "")
    monkeypatch.setattr(config, "VISION_TRIAL_FLOOD_MODEL_PATH", "")
    vision_repository.heartbeat_worker(url, "local-operational-ready", ready=True,
                                       status_detail="{}", role="operational")
    worker_env = os.environ.copy()
    worker_env.update({
        "DATABASE_URL": url,
        "VISION_UPLOAD_DIR": str(tmp_path),
        "VISION_MODEL_PATH": str(model),
        "VISION_WORKER_ROLE": "trial",
        "VISION_TRIAL_ENABLED": "true",
        "VISION_FLOOD_ENABLED": "false",
        "VISION_TRIAL_ACCIDENT_MODEL_PATH": "",
        "VISION_TRIAL_FLOOD_MODEL_PATH": "",
    })
    process = subprocess.Popen(
        [sys.executable, "-m", "vision.worker"], cwd=Path(__file__).resolve().parents[1],
        env=worker_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        deadline = time.monotonic() + 15
        while (process.poll() is None and not vision_repository.has_ready_worker(url, role="trial")
               and time.monotonic() < deadline):
            time.sleep(0.1)
        assert vision_repository.has_ready_worker(url, role="trial")
        app = create_app()
        app.config.update(TESTING=True, SECRET_KEY="local-vision-test", DATABASE_URL=url)
        client = app.test_client()
        csrf = client.post("/api/admin/login", json={"password": "local-vision-test"}).get_json()["data"]["csrf_token"]
        status = client.get("/api/manager/vision-status").get_json()["data"]
        assert status["trial"]["traffic"] is True
        response = client.post(
            "/api/manager/vision-jobs", headers={"X-CSRF-Token": csrf},
            data={"purpose": "trial", "trial_mode": "traffic", "observation_regions": "[]",
                  "media": (io.BytesIO(video.read_bytes()), "public.mp4", "video/mp4")},
        )
        assert response.status_code == 202, response.get_json()
        job_id = response.get_json()["data"]["job"]["job_id"]
        exit_code = process.wait(timeout=45)
        assert exit_code == 0, f"trial worker exited with {exit_code}"
        job = vision_repository.get_job(url, job_id)
        assert job["status"] == "completed", job.get("error")
        assert job["purpose"] == "trial" and job["result"]["trial_mode"] == "traffic"
        assert job["result"]["metrics"]["frames_analyzed"] > 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.skipif(
    not os.getenv("PUBLIC_FLOOD_SMOKE_IMAGE") or not os.getenv("PUBLIC_FLOOD_SMOKE_MODEL"),
    reason="requires an explicitly supplied public image and flood model",
)
def test_public_flood_image_uses_real_model_without_a_polygon(monkeypatch, tmp_path, isolated_test_database):
    import api.routes as routes
    from app import create_app
    from storage import database, vision_repository
    from vision import worker
    from vision.specialized import DoodleverseFloodSegmenter

    image = Path(os.environ["PUBLIC_FLOOD_SMOKE_IMAGE"])
    model = Path(os.environ["PUBLIC_FLOOD_SMOKE_MODEL"])
    url = isolated_test_database
    database.initialize(url)
    with database.connect(url) as conn:
        conn.execute("TRUNCATE manager_vision_job, manager_vision_worker")
    monkeypatch.setattr(routes.config, "ROAD_CONDITION_ADMIN_PASSWORD", "local-flood-test")
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(routes.config, "VISION_TRIAL_ENABLED", True)
    monkeypatch.setattr(routes, "_vision_worker_capabilities", lambda *_args: {
        "trial_flood_supported": True, "trial_flood_media": "image",
        "trial_flood_automatic_supported": True,
    })
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="local-flood-test", DATABASE_URL=url)
    client = app.test_client()
    csrf = client.post("/api/admin/login", json={"password": "local-flood-test"}).get_json()["data"]["csrf_token"]
    response = client.post(
        "/api/manager/vision-jobs", headers={"X-CSRF-Token": csrf},
        data={"purpose": "trial", "trial_mode": "flood", "observation_regions": "[]",
              "media": (io.BytesIO(image.read_bytes()), "public.png", "image/png")},
    )
    assert response.status_code == 202, response.get_json()
    job_id = response.get_json()["data"]["job"]["job_id"]
    segmenter = DoodleverseFloodSegmenter(
        str(model), runtime_python=os.getenv("PUBLIC_FLOOD_SMOKE_RUNTIME", sys.executable),
    )
    outcome = worker._process_next_job_unlocked(url, "public-flood-worker", trial_only=True,
                                                trial_flood_segmenter=segmenter,
                                                automatic_flood_supported=True)
    job = vision_repository.get_job(url, job_id)
    assert outcome["status"] in {"completed", "needs_review"}, job.get("error")
    assert job["purpose"] == "trial" and job["anchor_gcj"] is None
    assert job["result"]["trial_mode"] == "flood"
    assert job["result"]["automatic_flood"] is True
    assert job["result"]["trial_coverage"]["flood"] == "checked"
