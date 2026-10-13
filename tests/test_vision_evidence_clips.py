from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.evidence import attach_evidence_clips


class FakeCapture:
    def __init__(self, _path):
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {3: 1.0, 4: 120, 5: 80, 7: 20}.get(prop, 0)

    def set(self, prop, value):
        if prop == 1:
            self.position = int(value)
            return True
        return False

    def read(self):
        if self.position >= 20:
            return False, None
        frame = self.position
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, path, _fourcc, fps, size):
        self.path = path
        self.fps = fps
        self.size = size
        self.frames = []
        writers[path] = self

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        Path(self.path).write_bytes(b"clip" if self.frames else b"")


writers = {}


class FakeCV2:
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FPS = 3
    CAP_PROP_FRAME_WIDTH = 4
    CAP_PROP_FRAME_HEIGHT = 5
    CAP_PROP_FRAME_COUNT = 7
    VideoCapture = FakeCapture
    VideoWriter = FakeWriter

    @staticmethod
    def VideoWriter_fourcc(*_args):
        return 0


def test_candidate_evidence_gets_bounded_clips_with_safe_names(tmp_path):
    job_id = "12345678-1234-4234-8234-123456789abc"
    candidates = [{
        "kind": "possible_congestion",
        "evidence": {"segments": [
            {"start_seconds": 1.0, "end_seconds": 2.0, "confidence": 0.7},
            {"start_seconds": 4.0, "end_seconds": 6.0, "confidence": 0.9},
        ]},
    }]
    writers.clear()

    result = attach_evidence_clips(
        "source.mp4", candidates, tmp_path, job_id, cv2_module=FakeCV2,
        max_clips_per_candidate=1, padding_seconds=1.0,
        min_clip_seconds=5.0, max_clip_seconds=5.0,
    )

    clips = candidates[0]["evidence"]["clips"]
    assert result["clips_created"] == 1
    assert len(clips) == 1
    assert clips[0]["filename"] == f"{job_id}-evidence-c0-s0.webm"
    assert clips[0]["start_seconds"] == 2.0
    assert clips[0]["end_seconds"] == 7.0
    assert len(writers[str(tmp_path / clips[0]["filename"])].frames) == 5


def test_invalid_or_out_of_range_segments_do_not_create_artifacts(tmp_path):
    candidates = [{"evidence": {"segments": [
        {"start_seconds": -3, "end_seconds": 1},
        {"start_seconds": 50, "end_seconds": 60},
        {"start_seconds": "bad", "end_seconds": 2},
    ]}}]
    writers.clear()

    result = attach_evidence_clips(
        "source.mp4", candidates, tmp_path, "12345678-1234-4234-8234-123456789abc",
        cv2_module=FakeCV2,
    )

    assert result["clips_created"] == 0
    assert candidates[0]["evidence"]["clips"] == []
    assert writers == {}


def test_failed_seek_does_not_create_mislabeled_clip(tmp_path):
    class SeekFailureCapture(FakeCapture):
        def set(self, _prop, _value):
            return False

    class SeekFailureCV2(FakeCV2):
        VideoCapture = SeekFailureCapture

    candidates = [{"evidence": {"segments": [
        {"start_seconds": 1.0, "end_seconds": 2.0},
    ]}}]

    result = attach_evidence_clips(
        "source.mp4", candidates, tmp_path, "12345678-1234-4234-8234-123456789abc",
        cv2_module=SeekFailureCV2,
    )

    assert result["clips_created"] == 0
    assert result["clips_failed"] == 1
    assert candidates[0]["evidence"]["clip_status"] == "unavailable"
    assert not list(tmp_path.glob("*-evidence-*.webm"))


def test_partial_decode_is_deleted_instead_of_returned_as_evidence(tmp_path):
    class PartialCapture(FakeCapture):
        def read(self):
            if self.position >= 5:
                return False, None
            return super().read()

    class PartialCV2(FakeCV2):
        VideoCapture = PartialCapture

    candidates = [{"evidence": {"segments": [
        {"start_seconds": 2.0, "end_seconds": 8.0},
    ]}}]

    result = attach_evidence_clips(
        "source.mp4", candidates, tmp_path, "12345678-1234-4234-8234-123456789abc",
        cv2_module=PartialCV2, padding_seconds=0, min_clip_seconds=4,
        max_clip_seconds=4,
    )

    assert result["clips_created"] == 0
    assert result["clips_failed"] == 1
    assert candidates[0]["evidence"]["clips"] == []
    assert not list(tmp_path.glob("*-evidence-*.webm"))


def test_clip_is_not_registered_when_output_container_cannot_be_decoded(tmp_path):
    class UnreadableOutputCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    class UnreadableOutputCV2(FakeCV2):
        @staticmethod
        def VideoCapture(path):
            if "-evidence-" in str(path):
                return UnreadableOutputCapture()
            return FakeCapture(path)

    candidates = [{"evidence": {"segments": [
        {"start_seconds": 1.0, "end_seconds": 2.0},
    ]}}]

    result = attach_evidence_clips(
        "source.mp4", candidates, tmp_path, "12345678-1234-4234-8234-123456789abc",
        cv2_module=UnreadableOutputCV2,
    )

    assert result["clips_created"] == 0
    assert result["clips_failed"] == 1
    assert candidates[0]["evidence"]["clips"] == []
    assert candidates[0]["evidence"]["clip_status"] == "unavailable"
    assert not list(tmp_path.glob("*-evidence-*.webm"))


def test_total_clip_budget_bounds_disk_output_across_many_candidates(tmp_path):
    writers.clear()
    candidates = [{"evidence": {"segments": [
        {"start_seconds": 1.0, "end_seconds": 2.0, "confidence": 0.9},
        {"start_seconds": 4.0, "end_seconds": 5.0, "confidence": 0.8},
        {"start_seconds": 7.0, "end_seconds": 8.0, "confidence": 0.7},
    ]}} for _ in range(3)]

    result = attach_evidence_clips(
        "source.mp4", candidates, tmp_path, "12345678-1234-4234-8234-123456789abc",
        cv2_module=FakeCV2, max_total_clips=6,
    )

    assert result["clips_created"] == 6
    assert len(list(tmp_path.glob("*-evidence-*.webm"))) == 6
    assert candidates[0]["evidence"]["clip_status"] == "ready"
    assert candidates[1]["evidence"]["clip_status"] == "ready"
    assert candidates[2]["evidence"]["clip_status"] == "job_limit"
    assert result["clips_skipped"] == 3


def test_reprocessing_a_job_removes_unreferenced_old_clips(tmp_path):
    job_id = "12345678-1234-4234-8234-123456789abc"
    old_clip = tmp_path / f"{job_id}-evidence-c0-s0.webm"
    old_clip.write_bytes(b"old")
    candidates = [{"evidence": {"segments": []}}]

    attach_evidence_clips("source.mp4", candidates, tmp_path, job_id, cv2_module=FakeCV2)

    assert not old_clip.exists()


def test_generated_webm_clip_is_reopenable_in_the_installed_video_runtime(tmp_path):
    source = tmp_path / "source.webm"
    source_writer = None
    source_codec = None
    for codec in ("VP90", "VP80"):
        candidate = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*codec), 10, (160, 120),
        )
        if candidate.isOpened():
            source_writer = candidate
            source_codec = codec
            break
        candidate.release()
    if source_writer is None:
        pytest.skip("installed OpenCV build has no WebM test encoder")
    for index in range(30):
        source_writer.write(np.full((120, 160, 3), index * 5, dtype=np.uint8))
    source_writer.release()
    source_capture = cv2.VideoCapture(str(source))
    if not source_capture.isOpened():
        source_capture.release()
        pytest.skip(f"installed OpenCV build cannot decode {source_codec} WebM")
    source_capture.release()

    candidates = [{"evidence": {"segments": [
        {"start_seconds": 0.5, "end_seconds": 1.5, "confidence": 0.9},
    ]}}]
    result = attach_evidence_clips(
        source, candidates, tmp_path, "12345678-1234-4234-8234-123456789abc",
        min_clip_seconds=2, max_clip_seconds=3, output_fps=10,
    )

    assert result["clips_created"] == 1
    clip_path = tmp_path / candidates[0]["evidence"]["clips"][0]["filename"]
    assert clip_path.suffix == ".webm"
    capture = cv2.VideoCapture(str(clip_path))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape[:2] == (120, 160)


def test_public_manager_job_exposes_clip_urls_without_local_filenames():
    from api.routes import _public_vision_job

    job = {
        "job_id": "12345678-1234-4234-8234-123456789abc",
        "media_path": "source.mp4",
        "result": {"candidates": [{"evidence": {"clips": [{
            "filename": "12345678-1234-4234-8234-123456789abc-evidence-c0-s0.webm",
            "start_seconds": 2.0,
            "end_seconds": 7.0,
        }]}}]},
    }

    public = _public_vision_job(job)
    clip = public["result"]["candidates"][0]["evidence"]["clips"][0]

    assert clip["media_url"] == (
        "/api/manager/vision-jobs/12345678-1234-4234-8234-123456789abc/media?clip=0:0"
    )
    assert "filename" not in clip
    assert "filename" in job["result"]["candidates"][0]["evidence"]["clips"][0]


def test_public_manager_job_keeps_large_video_previews_out_of_five_second_poll_payload():
    from api.routes import _public_vision_job

    job = {
        "job_id": "12345678-1234-4234-8234-123456789abc",
        "media_kind": "video",
        "result": {
            "preview_frame_interval_seconds": 1,
            "preview_frames": [{"time_seconds": 0, "detections": []}],
            "candidates": [],
        },
    }

    public = _public_vision_job(job)

    assert "preview_frames" not in public["result"]
    assert "preview_frame_interval_seconds" not in public["result"]
    assert job["result"]["preview_frames"]


def test_vision_job_list_query_does_not_load_frame_level_previews(monkeypatch):
    from storage import database, vision_repository
    from unittest.mock import MagicMock

    connection = MagicMock()
    connection.__enter__.return_value.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr(database, "connect", lambda *_args, **_kwargs: connection)

    assert vision_repository.list_jobs("postgresql://unused") == []
    query = connection.__enter__.return_value.execute.call_args.args[0]
    assert "result - 'preview_frames' - 'preview_frame_interval_seconds' AS result" in query


def test_manager_preview_frames_endpoint_returns_only_review_overlay_data(monkeypatch):
    from app import create_app
    import api.routes as routes
    from storage import database, vision_repository

    job_id = "12345678-1234-4234-8234-123456789abc"
    previews = [{"time_seconds": 2, "detections": [{
        "box": [0.1, 0.2, 0.4, 0.5], "label": "car",
        "confidence": 0.9, "track_id": "7",
    }]}]
    job = {
        "job_id": job_id, "media_kind": "video",
        "result": {
            "media": {"width": 1000, "height": 800},
            "preview_frame_interval_seconds": 1,
            "preview_frames": previews,
            "candidates": [{"kind": "private-candidate"}],
        },
    }
    monkeypatch.setattr(routes, "_require_admin", lambda: None)
    monkeypatch.setattr(database, "initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vision_repository, "get_job", lambda *_args, **_kwargs: job)
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="vision-test")

    response = app.test_client().get(f"/api/manager/vision-jobs/{job_id}/preview-frames")

    assert response.status_code == 200
    assert "no-store" in response.headers.get("Cache-Control", "")
    assert response.get_json()["data"] == {
        "media": {"width": 1000, "height": 800},
        "preview_frame_interval_seconds": 1,
        "preview_frames": previews,
    }


def test_manager_preview_frames_endpoint_requires_admin_authentication():
    from app import create_app

    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="vision-test")
    response = app.test_client().get(
        "/api/manager/vision-jobs/12345678-1234-4234-8234-123456789abc/preview-frames"
    )

    assert response.status_code == 401


def test_private_manager_media_route_serves_only_registered_evidence_clip(monkeypatch, tmp_path):
    from app import create_app
    import api.routes as routes
    from storage import database, vision_repository

    job_id = "12345678-1234-4234-8234-123456789abc"
    original_name = "source.mp4"
    clip_name = f"{job_id}-evidence-c0-s0.webm"
    (tmp_path / original_name).write_bytes(b"original")
    (tmp_path / clip_name).write_bytes(b"clip-bytes")
    job = {
        "job_id": job_id,
        "media_path": original_name,
        "original_name": original_name,
        "result": {"candidates": [{"evidence": {"clips": [{"filename": clip_name}]}}]},
    }
    monkeypatch.setattr(routes.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "_require_admin", lambda: None)
    monkeypatch.setattr(database, "initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vision_repository, "get_job", lambda *_args, **_kwargs: job)
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="vision-test")
    client = app.test_client()

    valid = client.get(f"/api/manager/vision-jobs/{job_id}/media?clip=0:0")
    forged = client.get(f"/api/manager/vision-jobs/{job_id}/media?clip=0:1")

    assert valid.status_code == 200
    assert valid.data == b"clip-bytes"
    assert valid.mimetype == "video/webm"
    assert forged.status_code == 404


def test_video_worker_attaches_generated_clips_to_the_job_result(monkeypatch, tmp_path):
    from vision import worker
    from storage import vision_repository

    job_id = "12345678-1234-4234-8234-123456789abc"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = {
        "job_id": job_id, "media_kind": "video", "media_path": "source.mp4",
        "anchor_gcj": None, "camera_stabilized": True, "observation_regions": [],
        "checkpoint": None,
    }
    finished = {}
    progress_updates = []
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(vision_repository, "claim_next_job", lambda *_args, **_kwargs: job)
    monkeypatch.setattr(vision_repository, "update_job_progress", lambda _db, _job, _worker, *, progress, checkpoint=None: (
        progress_updates.append(progress) or {"cancel_requested": False}
    ))
    monkeypatch.setattr(vision_repository, "finish_job", lambda _db, _job, _worker, *, status, result=None, error=None:
                        finished.update(status=status, result=result, error=error) or status)
    monkeypatch.setattr(worker, "analyze_media", lambda *_args, **_kwargs: {
        "candidates": [{"kind": "possible_congestion", "evidence": {"segments": []}}],
        "video_coverage": {"sampled_frames": 1},
    })
    monkeypatch.setattr(worker, "attach_evidence_clips", lambda _source, candidates, _output, _job: {
        "clips_created": 1, "clips_failed": 0, "clips_skipped": 0,
    })

    outcome = worker.process_next_job("db", "worker", lease_seconds=90)

    assert outcome["status"] == "needs_review"
    assert finished["result"]["evidence_clip_summary"]["clips_created"] == 1
    assert progress_updates[-1]["total_sampled_frames"] == progress_updates[-1]["frames_analyzed"] == 1


def test_worker_honors_cancellation_requested_after_analysis_before_finalizing(monkeypatch, tmp_path):
    from vision import worker
    from storage import vision_repository

    job_id = "12345678-1234-4234-8234-123456789abc"
    (tmp_path / "source.png").write_bytes(b"source")
    job = {
        "job_id": job_id, "media_kind": "image", "media_path": "source.png",
        "anchor_gcj": None, "camera_stabilized": False, "observation_regions": [],
        "checkpoint": None,
    }
    finished = {}
    progress_updates = []
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(vision_repository, "claim_next_job", lambda *_args, **_kwargs: job)
    monkeypatch.setattr(vision_repository, "update_job_progress", lambda _db, _job, _worker, *, progress, checkpoint=None: (
        progress_updates.append(progress) or {"cancel_requested": True}
    ))
    monkeypatch.setattr(
        vision_repository, "finish_job",
        lambda _db, _job, _worker, *, status, result=None, error=None:
            finished.update(status=status, result=result, error=error) or status,
    )
    monkeypatch.setattr(worker, "analyze_media", lambda *_args, **_kwargs: {
        "candidates": [{"kind": "vehicle_cluster_review"}],
        "metrics": {"frames_analyzed": 1},
    })

    outcome = worker.process_next_job("db", "worker", lease_seconds=90)

    assert outcome["status"] == "cancelled"
    assert finished["status"] == "cancelled"
    assert finished["result"] is None
    assert progress_updates[-1] == {"phase": "finalizing", "percent": 99}


def test_worker_polls_mid_segment_cancellation_and_keeps_the_saved_checkpoint(monkeypatch, tmp_path):
    from vision import worker
    from storage import vision_repository

    job_id = "12345678-1234-4234-8234-123456789abc"
    (tmp_path / "source.mp4").write_bytes(b"source")
    job = {
        "job_id": job_id, "media_kind": "video", "media_path": "source.mp4",
        "anchor_gcj": None, "camera_stabilized": True, "observation_regions": [],
        "checkpoint": None,
    }
    saved = {}
    finished = {}
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(vision_repository, "claim_next_job", lambda *_args, **_kwargs: job)
    monkeypatch.setattr(vision_repository, "get_job_cancel_state", lambda *_args: {
        "cancel_requested": True,
    })

    def update_progress(_db, _job, _worker, *, progress, checkpoint=None):
        if checkpoint is not None:
            saved["checkpoint"] = checkpoint
        return {"cancel_requested": True}

    monkeypatch.setattr(vision_repository, "update_job_progress", update_progress)
    monkeypatch.setattr(
        vision_repository, "finish_job",
        lambda _db, _job, _worker, *, status, result=None, error=None:
            finished.update(status=status, result=result, error=error) or status,
    )

    def analyze_with_cancel(_path, _kind, _anchor, **kwargs):
        assert kwargs["cancel_check"]() is True
        kwargs["progress_callback"]({
            "progress": {"phase": "analyzing", "percent": 5},
            "checkpoint": {"next_frame_index": 10, "segments": [{
                "start_seconds": 0.0, "end_seconds": 0.9,
                "metrics": {"frames_analyzed": 5},
            }]},
        })
        pytest.fail("cancelled analysis must stop before returning a result")

    monkeypatch.setattr(worker, "analyze_media", analyze_with_cancel)

    outcome = worker.process_next_job("db", "worker", lease_seconds=90)

    assert outcome["status"] == "cancelled"
    assert saved["checkpoint"]["next_frame_index"] == 10
    assert finished["status"] == "cancelled"
    assert finished["result"] is None


def test_worker_does_not_abort_on_live_progress_before_cancel_checkpoint(monkeypatch, tmp_path):
    from vision import worker
    from storage import vision_repository

    job_id = "12345678-1234-4234-8234-123456789abc"
    (tmp_path / "source.mp4").write_bytes(b"source")
    job = {
        "job_id": job_id, "media_kind": "video", "media_path": "source.mp4",
        "anchor_gcj": None, "camera_stabilized": True, "observation_regions": [],
        "checkpoint": None,
    }
    saved = {"live": 0, "checkpoint": None}
    finished = {}
    monkeypatch.setattr(worker.config, "VISION_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(vision_repository, "claim_next_job", lambda *_args, **_kwargs: job)
    monkeypatch.setattr(vision_repository, "get_job_cancel_state", lambda *_args: {
        "cancel_requested": True,
    })

    def update_progress(_db, _job, _worker, *, progress, checkpoint=None):
        if checkpoint is None:
            saved["live"] += 1
        else:
            saved["checkpoint"] = checkpoint
        return {"cancel_requested": True}

    monkeypatch.setattr(vision_repository, "update_job_progress", update_progress)
    monkeypatch.setattr(
        vision_repository, "finish_job",
        lambda _db, _job, _worker, *, status, result=None, error=None:
            finished.update(status=status, result=result, error=error) or status,
    )

    def analyze_with_progress_then_cancel(_path, _kind, _anchor, **kwargs):
        kwargs["progress_callback"]({
            "progress": {"phase": "analyzing", "percent": 5, "frames_analyzed": 1},
        })
        assert saved["live"] == 1
        kwargs["progress_callback"]({
            "progress": {"phase": "analyzing", "percent": 5, "frames_analyzed": 1},
            "checkpoint": {"next_frame_index": 3, "segments": []},
        })
        pytest.fail("cancelled analysis must stop after preserving its checkpoint")

    monkeypatch.setattr(worker, "analyze_media", analyze_with_progress_then_cancel)

    outcome = worker.process_next_job("db", "worker", lease_seconds=90)

    assert outcome["status"] == "cancelled"
    assert saved["checkpoint"]["next_frame_index"] == 3
    assert finished["status"] == "cancelled"
