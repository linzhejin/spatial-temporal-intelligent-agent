import sys

import cv2
import numpy as np
import pytest

from vision.engine import analyze_media, _compact_specialized_observations, _merge_segment_analyses
from vision.worker import load_optional_models


class FakeDetector:
    model_id = "test-detector"
    model_version = "fixture-v1"
    confidence_threshold = 0.4

    def __init__(self):
        self.calls = 0

    def detect(self, _frame, *, tracking=False):
        self.calls += 1
        return [
            {"label": "car", "box": [10, 10, 40, 40], "confidence": 0.91, "track_id": None},
            {"label": "pedestrian", "box": [50, 10, 60, 30], "confidence": 0.88, "track_id": None},
        ]


def test_optional_model_failure_does_not_disable_other_visual_capabilities():
    class AccidentLoader:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("bad accident model")

    class FloodLoader:
        def __init__(self, *_args, **_kwargs):
            self.model_id = "flood-fixture"

    accident, flood, capabilities = load_optional_models(
        accident_path="bad-aider.onnx", flood_path="valid-flood.onnx",
        accident_enabled=True, flood_enabled=True,
        accident_loader=AccidentLoader, flood_loader=FloodLoader,
        release_checker=lambda kind, *_args: {"approved": True, "reason": "approved"},
    )

    assert accident is None
    assert flood.model_id == "flood-fixture"
    assert capabilities == {
        "accident_recognition_supported": False,
        "flood_segmentation_supported": True,
        "accident_recognition_release_status": "model_load_failed",
        "flood_segmentation_release_status": "approved",
    }


def test_optional_models_stay_unloaded_until_release_evidence_is_present(tmp_path):
    class MustNotLoad:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("optional model was loaded without explicit release approval")

    accident_path = tmp_path / "available-aider.onnx"
    flood_path = tmp_path / "available-flood.onnx"
    accident_path.write_bytes(b"fixture")
    flood_path.write_bytes(b"fixture")
    accident, flood, capabilities = load_optional_models(
        accident_path=accident_path, flood_path=flood_path,
        accident_enabled=True, flood_enabled=True,
        accident_loader=MustNotLoad, flood_loader=MustNotLoad,
    )

    assert accident is None
    assert flood is None
    assert capabilities == {
        "accident_recognition_supported": False,
        "flood_segmentation_supported": False,
        "accident_recognition_release_status": "release_manifest_missing",
        "flood_segmentation_release_status": "release_manifest_missing",
    }


def test_optional_release_check_failure_does_not_disable_base_vision(tmp_path):
    model_path = tmp_path / "optional.onnx"
    model_path.write_bytes(b"fixture")

    def broken_release_check(*_args):
        raise OSError("release evidence file cannot be read")

    class MustNotLoad:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("model must stay unloaded when the release check fails")

    accident, flood, capabilities = load_optional_models(
        accident_path=model_path, flood_path=None,
        accident_enabled=True, flood_enabled=False,
        accident_loader=MustNotLoad,
        release_checker=broken_release_check,
    )

    assert accident is None
    assert flood is None
    assert capabilities["accident_recognition_supported"] is False
    assert capabilities["accident_recognition_release_status"] == "release_check_failed"


def test_single_image_pipeline_returns_vehicle_boxes_without_claiming_accident_or_geolocation(tmp_path):
    source = tmp_path / "image.png"
    assert cv2.imwrite(str(source), np.zeros((80, 100, 3), dtype=np.uint8))

    result = analyze_media(
        source, "image", {"lng": 114.363, "lat": 30.5365}, detector=FakeDetector(),
    )

    assert result["model"] == {
        "id": "test-detector", "version": "fixture-v1", "confidence_threshold": 0.4,
        "tracking": {
            "algorithm": "bytetrack",
            "implementation": "FoundationVision/ByteTrack@d1bf019",
            "camera_motion_compensation": "homography_projected_to_segment_anchor",
            "low_confidence_tracking": True,
            "high_confidence_threshold": 0.4,
            "low_confidence_threshold": 0.1,
        },
        "classes": result["model"]["classes"],
        "dataset_source": "VisDrone2019-DET (AISKYEYE team, Tianjin University)",
        "dataset_license": "CC BY-NC-SA 3.0; non-commercial academic research use",
        "weights_distributed_by_application": False,
        "accident_model": None,
        "flood_model": None,
    }
    assert [item["label"] for item in result["preview_detections"]] == ["car"]
    assert result["metrics"]["peak_vehicle_count"] == 1
    assert result["metrics"]["motion_assessment"] == "insufficient_single_frame"
    assert result["safety"]["accident_recognition_supported"] is False
    assert result["safety"]["automatically_changes_routing"] is False
    assert result["location_precision"] == "operator_selected_area_only"


def test_single_image_pipeline_explains_when_no_location_was_supplied(tmp_path):
    source = tmp_path / "image.png"
    assert cv2.imwrite(str(source), np.zeros((80, 100, 3), dtype=np.uint8))

    result = analyze_media(source, "image", None, detector=FakeDetector())

    assert result["anchor_gcj"] is None
    assert result["location_precision"] == "not_provided"
    assert "具体道路" in result["review_guidance"]
    assert "管理员标注" not in result["review_guidance"]


def test_optional_accident_inference_failure_does_not_fail_basic_image_analysis(tmp_path):
    class BrokenAccidentModel:
        model_id = "broken-aider"
        model_version = "fixture"

        def predict(self, _frame):
            raise ValueError("bad optional model output")

    source = tmp_path / "image.png"
    assert cv2.imwrite(str(source), np.zeros((80, 100, 3), dtype=np.uint8))
    region = {"id": "lane", "kind": "vehicle_lane", "polygon": [
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
    ]}

    result = analyze_media(
        source, "image", None, detector=FakeDetector(), accident_model=BrokenAccidentModel(),
        regions=[region],
    )

    assert result["metrics"]["peak_vehicle_count"] == 1
    assert result["safety"]["accident_recognition_supported"] is False
    assert result["model"]["accident_model"]["runtime_status"] == "inference_failed"
    assert result["optional_model_errors"] == [{"model": "accident", "error_type": "ValueError"}]


def test_video_checkpoint_keeps_only_the_strongest_flood_outline_per_road():
    observations = [
        {"time_seconds": 1.0, "region_id": "road-a", "flooded_road_area_ratio": 0.2,
         "outline_polygons": [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]]},
        {"time_seconds": 2.0, "region_id": "road-a", "flooded_road_area_ratio": 0.5,
         "outline_polygons": [[[0.3, 0.3], [0.4, 0.3], [0.4, 0.4]]]},
        {"time_seconds": 2.0, "region_id": "road-b", "flooded_road_area_ratio": 0.3,
         "outline_polygons": [[[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]]},
    ]

    compacted = _compact_specialized_observations(observations)

    assert compacted[0]["outline_polygons"] == []
    assert compacted[1]["outline_polygons"] == observations[1]["outline_polygons"]
    assert compacted[2]["outline_polygons"] == observations[2]["outline_polygons"]
    assert len(compacted) == len(observations)


def test_segment_merge_keeps_candidates_for_distinct_observation_regions_separate():
    segment = {
        "start_seconds": 0.0,
        "end_seconds": 5.0,
        "metrics": {"frames_analyzed": 5},
        "candidates": [
            {"kind": "possible_congestion", "confidence": 0.8, "reason": "queue",
             "evidence": {"summary": {"region_id": "north-lane"}}},
            {"kind": "possible_congestion", "confidence": 0.7, "reason": "queue",
             "evidence": {"summary": {"region_id": "south-lane"}}},
        ],
    }

    result = _merge_segment_analyses([segment])

    assert len(result["candidates"]) == 2
    assert {
        item["evidence"]["summary"]["region_id"] for item in result["candidates"]
    } == {"north-lane", "south-lane"}


def test_segment_merge_propagates_observation_region_reassociation_requirement():
    segment = {
        "start_seconds": 0.0,
        "end_seconds": 5.0,
        "metrics": {"frames_analyzed": 5},
        "candidates": [],
        "safety": {"observation_region_reassociation_required": True},
    }

    result = _merge_segment_analyses([segment])

    assert result["safety"]["observation_region_reassociation_required"] is True


def test_segment_merge_preserves_temporal_evidence_thresholds_for_manager_feedback():
    result = _merge_segment_analyses([{
        "start_seconds": 0.0,
        "end_seconds": 5.0,
        "metrics": {
            "frames_analyzed": 5,
            "minimum_congestion_duration_seconds": 15.0,
            "minimum_crowding_duration_seconds": 5.0,
        },
        "candidates": [],
        "safety": {},
    }])

    assert result["metrics"]["minimum_congestion_duration_seconds"] == 15.0
    assert result["metrics"]["minimum_crowding_duration_seconds"] == 5.0


def test_segment_merge_splits_same_region_events_when_an_intervening_segment_has_no_candidate():
    def segment(start, end, candidate=True):
        return {
            "start_seconds": start,
            "end_seconds": end,
            "metrics": {"frames_analyzed": 5},
            "candidates": ([{
                "kind": "possible_crowding", "confidence": 0.8, "reason": "crowd",
                "evidence": {"region_id": "plaza", "summary": {"region_id": "plaza"}},
            }] if candidate else []),
        }

    result = _merge_segment_analyses([
        segment(0.0, 4.0), segment(5.0, 9.0), segment(10.0, 14.0, candidate=False),
        segment(15.0, 19.0),
    ])

    candidates = result["candidates"]
    assert len(candidates) == 2
    assert [item["evidence"]["occurrences"] for item in candidates] == [2, 1]
    assert [item["evidence"]["segments"][0]["start_seconds"] for item in candidates] == [0.0, 15.0]

class FakeCapture:
    def __init__(self, total_frames, fps=10, *, seek_supported=True, decodable_frames=None):
        self.total_frames = total_frames
        self.fps = fps
        self.seek_supported = seek_supported
        self.decodable_frames = total_frames if decodable_frames is None else decodable_frames
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            FakeCV2.CAP_PROP_FRAME_WIDTH: 100,
            FakeCV2.CAP_PROP_FRAME_HEIGHT: 80,
            FakeCV2.CAP_PROP_FPS: self.fps,
            FakeCV2.CAP_PROP_FRAME_COUNT: self.total_frames,
            FakeCV2.CAP_PROP_POS_FRAMES: self.index,
        }.get(prop, 0)

    def set(self, prop, value):
        if prop == FakeCV2.CAP_PROP_POS_FRAMES:
            if not self.seek_supported:
                return False
            self.index = int(value)
            return True
        return False

    def read(self):
        if self.index >= self.decodable_frames:
            return False, None
        self.index += 1
        return True, np.zeros((80, 100, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    _seek_supported = True
    _decodable_frames = None
    IMREAD_COLOR = 1
    _total_frames = 1900

    @classmethod
    def VideoCapture(cls, _path):
        return FakeCapture(
            cls._total_frames, seek_supported=cls._seek_supported,
            decodable_frames=cls._decodable_frames,
        )


def test_video_pipeline_samples_entire_clip_in_segments_not_just_first_180_seconds(tmp_path, monkeypatch):
    import config

    source = tmp_path / "campus.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 1900  # 190 seconds at 10 fps
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 180)
    detector = FakeDetector()
    progress = []

    result = analyze_media(
        source, "video", {"lng": 114.363, "lat": 30.5365},
        camera_stabilized=True, detector=detector, progress_callback=progress.append,
    )

    assert result["video_coverage"]["complete"] is True
    assert result["video_coverage"]["duration_seconds"] == 190
    assert result["video_coverage"]["analyzed_through_seconds"] >= 189
    assert result["video_coverage"]["sampled_frames"] == 950
    assert result["video_coverage"]["segment_count"] == 6
    assert detector.calls == 950
    assert progress[-1]["checkpoint"]["next_frame_index"] == 1900
    preview_frames = result["preview_frames"]
    assert 1 <= len(preview_frames) <= 180
    assert preview_frames[0]["time_seconds"] == 0
    assert preview_frames[-1]["time_seconds"] >= 188
    assert preview_frames[0]["detections"]
    detection = preview_frames[0]["detections"][0]
    assert detection["label"] in {"car", "pedestrian"}
    assert detection["box"] == [0.1, 0.125, 0.4, 0.5]
    assert detection["track_id"] is not None


def test_congestion_evidence_continues_across_video_segment_boundary(tmp_path, monkeypatch):
    import config
    from vision.engine import VisionAnalysisCancelled

    class OneFpsCapture(FakeCapture):
        def __init__(self, total_frames):
            super().__init__(total_frames, fps=1)

        def read(self):
            if self.index >= self.decodable_frames:
                return False, None
            frame_index = self.index
            self.index += 1
            return True, np.full((80, 100, 3), frame_index, dtype=np.uint8)

    class OneFpsCV2(FakeCV2):
        _total_frames = 40

        @classmethod
        def VideoCapture(cls, _path):
            return OneFpsCapture(cls._total_frames)

    class TimedTrafficDetector:
        model_id = "segment-boundary-fixture"
        model_version = "1"
        confidence_threshold = 0.4

        def __init__(self):
            self.calls = 0

        def detect(self, frame, *, tracking=False):
            self.calls += 1
            frame_index = int(frame[0, 0, 0])
            if not 12 <= frame_index < 28:
                return []
            return [
                {"label": "car", "box": [2 + index * 6, 20, 7 + index * 6, 32],
                 "confidence": 0.95, "track_id": None}
                for index in range(14)
            ]

    source = tmp_path / "boundary.mp4"
    source.write_bytes(b"fake-video")
    monkeypatch.setitem(sys.modules, "cv2", OneFpsCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 1)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 20)
    lane = [{"id": "lane", "kind": "vehicle_lane",
             "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    checkpoint = {}
    first_detector = TimedTrafficDetector()

    def save_checkpoint(payload):
        checkpoint.update(payload["checkpoint"])

    with pytest.raises(VisionAnalysisCancelled):
        analyze_media(
            source, "video", None, camera_stabilized=True,
            detector=first_detector, regions=lane,
            progress_callback=save_checkpoint,
            cancel_check=lambda: first_detector.calls >= 20,
        )
    assert checkpoint["temporal_context"]["frame_indices"][-1] == 19
    result = analyze_media(
        source, "video", None, camera_stabilized=True,
        detector=TimedTrafficDetector(), regions=lane, resume_state=checkpoint,
    )

    candidates = [item for item in result["candidates"]
                  if item["kind"] == "possible_congestion"]
    assert len(candidates) == 1
    assert candidates[0]["evidence"]["summary"]["region_observed_duration_seconds"] >= 15
    assert candidates[0]["evidence"]["segments"][0]["start_seconds"] == 4.0
    assert candidates[0]["evidence"]["segments"][0]["end_seconds"] == 39.0
    assert result["video_coverage"]["sampled_frames"] == 40


def test_crowding_evidence_continues_across_video_segment_boundary(tmp_path, monkeypatch):
    import config

    class OneFpsCapture(FakeCapture):
        def __init__(self, total_frames):
            super().__init__(total_frames, fps=1)

    class OneFpsCV2(FakeCV2):
        _total_frames = 40

        @classmethod
        def VideoCapture(cls, _path):
            return OneFpsCapture(cls._total_frames)

    class TimedCrowdDetector:
        model_id = "crowd-boundary-fixture"
        model_version = "1"
        confidence_threshold = 0.4

        def __init__(self):
            self.frame_index = 0

        def detect(self, _frame, *, tracking=False):
            frame_index = self.frame_index
            self.frame_index += 1
            if not 17 <= frame_index < 24:
                return []
            return [
                {"label": "pedestrian", "box": [10 + index * 3, 20, 20 + index * 3, 50],
                 "confidence": 0.95, "track_id": None}
                for index in range(16)
            ]

    source = tmp_path / "crowd-boundary.mp4"
    source.write_bytes(b"fake-video")
    monkeypatch.setitem(sys.modules, "cv2", OneFpsCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 1)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 20)
    walkway = [{"id": "walkway", "kind": "pedestrian",
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]

    result = analyze_media(
        source, "video", None, camera_stabilized=True,
        detector=TimedCrowdDetector(), regions=walkway,
    )

    candidates = [item for item in result["candidates"]
                  if item["kind"] == "possible_crowding"]
    assert len(candidates) == 1
    assert candidates[0]["evidence"]["summary"]["sustained_cluster_duration_seconds"] >= 5


def test_moving_video_projects_the_manager_roi_before_detection_and_crowd_analysis(
    tmp_path, monkeypatch,
):
    import config
    from vision import engine

    class MovingCapture(FakeCapture):
        def read(self):
            if self.index >= self.decodable_frames:
                return False, None
            value = self.index * 0.1
            self.index += 1
            return True, np.full((80, 100, 3), value, dtype=np.float32)

    class MovingCV2(FakeCV2):
        _total_frames = 120

        @classmethod
        def VideoCapture(cls, _path):
            return MovingCapture(cls._total_frames)

    class RegionDetector:
        confidence_threshold = 0.369
        tracking_low_confidence_threshold = 0.1
        model_id = "moving-roi-fixture"
        model_version = "1"

        def __init__(self):
            self.rois = []

        def detect_regions(self, frame, regions, *, tracking=False):
            self.rois.append(regions[0]["polygon"])
            shift = float(frame[0, 0, 0])
            return [
                {"label": "pedestrian", "confidence": 0.9, "track_id": None,
                 "box": [12 + index * 2 + shift, 20, 14 + index * 2 + shift, 42]}
                for index in range(16)
            ]

        def detect(self, *_args, **_kwargs):
            pytest.fail("a valid projected ROI should use region-limited detection")

    source = tmp_path / "moving-drone.mp4"
    source.write_bytes(b"fake-video")
    monkeypatch.setitem(sys.modules, "cv2", MovingCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 100)
    monkeypatch.setattr(engine, "_gray_for_motion", lambda frame, _cv2: (frame[:, :, 0], 1.0))
    monkeypatch.setattr(
        engine, "_estimate_camera_transform",
        lambda previous, current, _cv2, _scale: [
            [1, 0, float(current.mean() - previous.mean())], [0, 1, 0], [0, 0, 1],
        ],
    )
    detector = RegionDetector()
    region = {"id": "walkway", "kind": "pedestrian", "polygon": [
        [0.1, 0.1], [0.7, 0.1], [0.7, 0.9], [0.1, 0.9],
    ]}

    result = analyze_media(
        source, "video", None, detector=detector, regions=[region],
    )

    crowd = next(item for item in result["candidates"] if item["kind"] == "possible_crowding")
    assert len(detector.rois) == 60
    assert detector.rois[0][0][0] == pytest.approx(0.1)
    assert detector.rois[-1][0][0] > detector.rois[0][0][0]
    assert crowd["evidence"]["summary"]["region_mapping_basis"] == "homography_projected_from_video_anchor"
    assert result["safety"]["observation_region_reassociation_required"] is False


def test_video_reports_live_progress_between_durable_segment_checkpoints(tmp_path, monkeypatch):
    import config
    from vision import engine

    source = tmp_path / "live-progress.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 20
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 50)
    monkeypatch.setattr(config, "VISION_PROGRESS_UPDATE_SECONDS", 0.5, raising=False)
    clock = {"value": 0.0}

    def monotonic():
        clock["value"] += 1.0
        return clock["value"]

    monkeypatch.setattr(engine.time, "monotonic", monotonic)
    progress = []

    analyze_media(
        source, "video", None, camera_stabilized=True, detector=FakeDetector(),
        progress_callback=progress.append,
    )

    live_updates = [item for item in progress if not item.get("checkpoint")]
    assert len(live_updates) >= 2
    assert all(item["progress"]["total_sampled_frames"] == 10 for item in live_updates)
    assert [item["progress"]["frames_analyzed"] for item in live_updates] == sorted(
        item["progress"]["frames_analyzed"] for item in live_updates
    )
    assert live_updates[0]["progress"]["frames_analyzed"] > 0
    assert progress[-1]["checkpoint"]["next_frame_index"] == 20


def test_video_cancel_stops_between_frames_and_persists_partial_segment_checkpoint(tmp_path, monkeypatch):
    import config
    from vision.engine import VisionAnalysisCancelled

    source = tmp_path / "cancel-mid-segment.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 100
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 50)
    detector = FakeDetector()
    progress = []
    checks = 0

    def cancellation_requested():
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises(VisionAnalysisCancelled):
        analyze_media(
            source, "video", None, camera_stabilized=True, detector=detector,
            progress_callback=progress.append, cancel_check=cancellation_requested,
        )

    assert detector.calls == 2
    assert progress[-1]["checkpoint"]["next_frame_index"] == 3
    assert progress[-1]["checkpoint"]["segments"][-1]["metrics"]["frames_analyzed"] == 2
    assert progress[-1]["progress"]["analyzed_through_seconds"] == 0.2


def test_specialized_video_evidence_is_merged_across_processing_segments(tmp_path, monkeypatch):
    import config

    class AccidentModel:
        model_id = "fake-aider"
        model_version = "fixture"
        threshold = 0.85

        def predict(self, _frame):
            return {"traffic_accident_probability": 0.96, "model_id": self.model_id}

    source = tmp_path / "short.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 80
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 5)

    result = analyze_media(
        source, "video", None, camera_stabilized=True, detector=FakeDetector(),
        accident_model=AccidentModel(),
        regions=[{"id": "lane", "kind": "vehicle_lane",
                  "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}],
    )

    candidate = next(item for item in result["candidates"] if item["kind"] == "possible_accident")
    assert candidate["evidence"]["summary"]["supporting_frames"] >= 2
    assert result["metrics"]["specialized_frames_analyzed"] >= 2
    assert result["safety"]["accident_recognition_supported"] is True
    assert candidate["review_required"] is True
    assert candidate["auto_publish"] is False


def test_accident_trigger_adds_dense_video_analysis_without_increasing_flood_sampling(tmp_path, monkeypatch):
    import config

    class AccidentModel:
        model_id = "fake-aider"
        model_version = "fixture"
        threshold = 0.85

        def __init__(self):
            self.calls = 0

        def predict(self, _frame):
            self.calls += 1
            score = 0.6 if self.calls == 1 else 0.96 if self.calls <= 11 else 0.1
            return {"traffic_accident_probability": score, "model_id": self.model_id}

    class FloodModel:
        model_id = "fake-flood"
        min_area_ratio = 0.08

        def __init__(self):
            self.calls = 0

        def predict(self, _frame, _region):
            self.calls += 1
            return None

    source = tmp_path / "short.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 80
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 50)
    monkeypatch.setattr(config, "VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD", 0.5, raising=False)
    monkeypatch.setattr(config, "VISION_ACCIDENT_DENSE_WINDOW_SECONDS", 2.0, raising=False)
    accident_model = AccidentModel()
    flood_model = FloodModel()
    region = {"id": "lane", "kind": "vehicle_lane", "polygon": [
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
    ]}
    road_surface = {"id": "road", "kind": "road_surface", "polygon": [
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
    ]}

    result = analyze_media(
        source, "video", None, camera_stabilized=True, detector=FakeDetector(),
        accident_model=accident_model, flood_segmenter=flood_model,
        regions=[region, road_surface],
    )

    # Regular specialized cadence is 1 fps (8 calls for this 8 second clip).
    # A trigger around t=0 adds accident-only calls at the detector's 5 fps cadence.
    assert 8 < accident_model.calls < 40
    assert flood_model.calls == 8
    assert result["metrics"]["specialized_frames_analyzed"] == accident_model.calls
    assert any(item["kind"] == "possible_accident" for item in result["candidates"])


def test_dense_sampling_trigger_is_not_the_accident_decision_threshold(tmp_path, monkeypatch):
    import config

    class LowScoreAccidentModel:
        model_id = "fake-aider"
        model_version = "fixture"
        threshold = 0.85

        def __init__(self):
            self.calls = 0

        def predict(self, _frame):
            self.calls += 1
            return {"traffic_accident_probability": 0.6, "model_id": self.model_id}

    source = tmp_path / "short.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 80
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 50)
    monkeypatch.setattr(config, "VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD", 0.5, raising=False)
    monkeypatch.setattr(config, "VISION_ACCIDENT_DENSE_WINDOW_SECONDS", 2.0, raising=False)
    accident_model = LowScoreAccidentModel()

    result = analyze_media(
        source, "video", None, camera_stabilized=True, detector=FakeDetector(),
        accident_model=accident_model,
        regions=[{"id": "lane", "kind": "vehicle_lane", "polygon": [
            [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
        ]}],
    )

    assert accident_model.calls > 8
    assert not any(item["kind"] == "possible_accident" for item in result["candidates"])


def test_accident_dense_window_is_saved_in_the_video_resume_checkpoint(tmp_path, monkeypatch):
    import config

    class LateTriggerAccidentModel:
        model_id = "fake-aider"
        model_version = "fixture"
        threshold = 0.85

        def __init__(self):
            self.calls = 0

        def predict(self, _frame):
            self.calls += 1
            score = 0.6 if self.calls == 10 else 0.1
            return {"traffic_accident_probability": score, "model_id": self.model_id}

    source = tmp_path / "checkpoint.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 1000
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 50)
    monkeypatch.setattr(config, "VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD", 0.5, raising=False)
    monkeypatch.setattr(config, "VISION_ACCIDENT_DENSE_WINDOW_SECONDS", 2.0, raising=False)
    checkpoint = {}

    def interrupt_after_segment(payload):
        checkpoint.update(payload["checkpoint"])
        raise RuntimeError("pause after durable segment")

    with pytest.raises(RuntimeError, match="pause after durable segment"):
        analyze_media(
            source, "video", None, camera_stabilized=True, detector=FakeDetector(),
            accident_model=LateTriggerAccidentModel(),
            regions=[{"id": "lane", "kind": "vehicle_lane", "polygon": [
                [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
            ]}],
            progress_callback=interrupt_after_segment,
        )

    assert checkpoint["next_frame_index"] == 99
    assert checkpoint["accident_dense_until_frame"] == 110


def test_optional_accident_failure_is_isolated_and_stopped_for_rest_of_video(tmp_path, monkeypatch):
    import config

    class BrokenAccidentModel:
        model_id = "broken-aider"
        model_version = "fixture"

        def __init__(self):
            self.calls = 0

        def predict(self, _frame):
            self.calls += 1
            raise ValueError("incompatible runtime output")

    source = tmp_path / "short.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 80
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 5)
    accident_model = BrokenAccidentModel()
    region = {"id": "lane", "kind": "vehicle_lane", "polygon": [
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
    ]}

    result = analyze_media(
        source, "video", None, camera_stabilized=True, detector=FakeDetector(),
        accident_model=accident_model, regions=[region],
    )

    assert result["video_coverage"]["complete"] is True
    assert result["metrics"]["frames_analyzed"] > 0
    assert accident_model.calls == 1
    assert result["safety"]["accident_recognition_supported"] is False
    assert result["optional_model_errors"] == [{"model": "accident", "error_type": "ValueError"}]


def test_interrupted_video_resumes_after_last_completed_segment(tmp_path, monkeypatch):
    import config

    source = tmp_path / "campus.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 1000  # 100 seconds at 10 fps
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 50)
    checkpoint = {}

    class StopAfterFirstSegment(RuntimeError):
        pass

    def interrupt(payload):
        checkpoint.update(payload["checkpoint"])
        raise StopAfterFirstSegment

    with pytest.raises(StopAfterFirstSegment):
        analyze_media(
            source, "video", {"lng": 114.363, "lat": 30.5365},
            camera_stabilized=True, detector=FakeDetector(), progress_callback=interrupt,
        )
    calls_before_resume = 0
    detector = FakeDetector()
    original_detect = detector.detect

    def count_detect(frame, *, tracking=False):
        nonlocal calls_before_resume
        calls_before_resume += 1
        return original_detect(frame, tracking=tracking)

    detector.detect = count_detect
    result = analyze_media(
        source, "video", {"lng": 114.363, "lat": 30.5365},
        camera_stabilized=True, detector=detector, resume_state=checkpoint,
    )

    assert checkpoint["next_frame_index"] == 99
    assert result["video_coverage"]["sampled_frames"] == 500
    assert calls_before_resume == 450


def test_video_checkpoint_is_discarded_when_detector_revision_changes(tmp_path, monkeypatch):
    import config

    source = tmp_path / "campus.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 1000
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 50)
    checkpoint = {}

    class StopAfterFirstSegment(RuntimeError):
        pass

    def interrupt(payload):
        checkpoint.update(payload["checkpoint"])
        raise StopAfterFirstSegment

    with pytest.raises(StopAfterFirstSegment):
        analyze_media(
            source, "video", None, camera_stabilized=True,
            detector=FakeDetector(), progress_callback=interrupt,
        )

    detector = FakeDetector()
    detector.model_version = "fixture-v2"
    result = analyze_media(
        source, "video", None, camera_stabilized=True,
        detector=detector, resume_state=checkpoint,
    )

    assert detector.calls == 500
    assert result["video_coverage"]["sampled_frames"] == 500


def test_resume_restarts_from_zero_and_replaces_checkpoint_when_decoder_cannot_seek(tmp_path, monkeypatch):
    import config

    source = tmp_path / "no-seek.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 100
    monkeypatch.setattr(FakeCV2, "_seek_supported", True)
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 20)
    checkpoint = {}

    def stop_after_first_segment(payload):
        checkpoint.update(payload["checkpoint"])
        raise RuntimeError("pause after segment")

    with pytest.raises(RuntimeError, match="pause after segment"):
        analyze_media(
            source, "video", None, camera_stabilized=True, detector=FakeDetector(),
            resume_state=None, progress_callback=stop_after_first_segment,
        )

    monkeypatch.setattr(FakeCV2, "_seek_supported", False)
    progress = []
    detector = FakeDetector()

    result = analyze_media(
        source, "video", None, camera_stabilized=True, detector=detector,
        resume_state=checkpoint, progress_callback=progress.append,
    )

    assert result["video_coverage"]["sampled_frames"] == 50
    assert result["video_coverage"]["segment_count"] == 3
    assert detector.calls == 50
    assert progress[0]["checkpoint"]["next_frame_index"] == 0
    assert progress[0]["checkpoint"]["segments"] == []
    assert progress[0]["checkpoint"]["analysis_fingerprint"]


def test_video_with_missing_declared_tail_frames_is_not_reported_as_complete(tmp_path, monkeypatch):
    import config

    source = tmp_path / "truncated.mp4"
    source.write_bytes(b"fake-video")
    FakeCV2._total_frames = 100
    monkeypatch.setattr(FakeCV2, "_decodable_frames", 98)
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    monkeypatch.setattr(config, "VISION_SAMPLE_FPS", 5)
    monkeypatch.setattr(config, "VISION_SEGMENT_FRAMES", 20)

    with pytest.raises(ValueError, match="声明帧数"):
        analyze_media(
            source, "video", None, camera_stabilized=True, detector=FakeDetector(),
        )
