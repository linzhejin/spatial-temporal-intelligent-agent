"""Temporal accident trials must return review-only video evidence."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision.engine import VisionAnalysisCancelled
from vision.temporal_trial import OnnxTemporalAccident, _window_frame_ranges


class FakeSession:
    def __init__(self, score=0.9):
        self.score = score
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name="frames", shape=[1, 16, 3, 224, 224])]

    def get_outputs(self):
        return [SimpleNamespace(name="score", shape=[1, 1])]

    def run(self, _outputs, feed):
        self.feeds.append(feed["frames"])
        return [np.asarray([[self.score]], dtype=np.float32)]


@pytest.fixture
def short_video(tmp_path):
    path = tmp_path / "trial.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 64))
    assert writer.isOpened()
    for index in range(24):
        writer.write(np.full((64, 64, 3), index * 4, dtype=np.uint8))
    writer.release()
    return path


def test_temporal_model_returns_review_only_bounded_video_evidence(short_video):
    session = FakeSession()
    model = OnnxTemporalAccident("unused.onnx", session=session, threshold=0.5)
    result = model.analyze_video(short_video)
    assert result["windows_analyzed"] == 1
    assert session.feeds[0].shape == (1, 16, 3, 224, 224)
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["kind"] == "possible_accident"
    assert candidate["review_required"] is True and candidate["auto_publish"] is False
    assert candidate["evidence"]["segments"][0]["start_seconds"] == 0
    assert 0 < candidate["evidence"]["segments"][0]["end_seconds"] <= 6


def test_temporal_model_does_not_invent_a_candidate_for_low_score(short_video):
    model = OnnxTemporalAccident("unused.onnx", session=FakeSession(0.1), threshold=0.5)
    result = model.analyze_video(short_video)
    assert result["windows_analyzed"] == 1
    assert result["candidates"] == []


def test_temporal_video_can_be_cancelled_before_inference(short_video):
    session = FakeSession()
    model = OnnxTemporalAccident("unused.onnx", session=session)
    with pytest.raises(VisionAnalysisCancelled):
        model.analyze_video(short_video, cancel_check=lambda: True)
    assert session.feeds == []


def test_overlapping_video_windows_cover_the_full_recording():
    windows = _window_frame_ranges(total_frames=70, fps=1, window_seconds=30, stride_seconds=15)
    assert windows[0][0] == 0 and windows[-1][1] == 70
    assert all(end - start <= 30 for start, end in windows)
    assert all(left[1] >= right[0] for left, right in zip(windows, windows[1:]))


def test_trial_video_refuses_frames_above_its_memory_limit(short_video, monkeypatch):
    import config
    monkeypatch.setattr(config, "VISION_TRIAL_MAX_FRAME_PIXELS", 4000, raising=False)
    model = OnnxTemporalAccident("unused.onnx", session=FakeSession())
    with pytest.raises(ValueError, match="试用.*分辨率"):
        model.analyze_video(short_video)


def test_trial_video_refuses_duration_above_its_runtime_limit(short_video, monkeypatch):
    import config
    monkeypatch.setattr(config, "VISION_TRIAL_MAX_VIDEO_SECONDS", 5, raising=False)
    model = OnnxTemporalAccident("unused.onnx", session=FakeSession())
    with pytest.raises(ValueError, match="试用.*时长"):
        model.analyze_video(short_video)


def test_trial_flood_pipeline_enforces_the_same_video_limits(short_video):
    from vision.engine import analyze_media
    class EmptyDetector:
        def detect(self, _frame, *, tracking=False):
            return []
    with pytest.raises(ValueError, match="试用.*分辨率"):
        analyze_media(short_video, "video", None, detector=EmptyDetector(), max_frame_pixels=4000)
    with pytest.raises(ValueError, match="试用.*时长"):
        analyze_media(short_video, "video", None, detector=EmptyDetector(), max_video_seconds=5)


def test_temporal_window_is_not_misrepresented_by_a_short_center_clip(short_video, tmp_path):
    from vision.evidence import attach_evidence_clips
    model = OnnxTemporalAccident("unused.onnx", session=FakeSession())
    candidates = model.analyze_video(short_video)["candidates"]
    result = attach_evidence_clips(short_video, candidates, tmp_path,
                                   "12345678-1234-4234-8234-123456789abc")
    assert result["clips_created"] == 0
    assert candidates[0]["evidence"]["clips"] == []
    assert candidates[0]["evidence"]["clip_status"] == "window_only"
