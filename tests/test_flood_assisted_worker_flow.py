import weakref

from vision import assisted_image, worker


def test_assisted_worker_uses_doodleverse_without_loading_legacy_trial_models(monkeypatch, tmp_path):
    traffic_path = tmp_path / "traffic.onnx"
    flood_path = tmp_path / "flood.onnx"
    trial_accident_path = tmp_path / "trial-accident.onnx"
    trial_flood_path = tmp_path / "trial-flood.onnx"
    for path in (traffic_path, flood_path, trial_accident_path, trial_flood_path):
        path.write_bytes(b"fixture")

    config_values = {
        "VISION_TRIAL_ENABLED": True,
        "VISION_TRIAL_ACCIDENT_MODEL_PATH": str(trial_accident_path),
        "VISION_TRIAL_FLOOD_MODEL_PATH": str(trial_flood_path),
        "VISION_TRIAL_ACCIDENT_KIND": "temporal",
        "VISION_FLOOD_ENABLED": True,
        "VISION_FLOOD_RELEASE_MODE": "assisted_review",
        "VISION_FLOOD_MODEL_PATH": str(flood_path),
        "VISION_MODEL_PATH": str(traffic_path),
        "VISION_FLOOD_REVISION": "doodleverse-v7",
        "VISION_FLOOD_PROBABILITY_THRESHOLD": 0.001,
        "VISION_FLOOD_MIN_AREA_RATIO": 0.08,
        "VISION_FLOOD_RUNTIME_PYTHON": "python",
        "VISION_FLOOD_TIMEOUT_SECONDS": 120,
        "VISION_OPTIONAL_MODELS_RELEASE_MANIFEST": "release.json",
    }
    for name, value in config_values.items():
        monkeypatch.setattr(worker.config, name, value, raising=False)

    def unexpected_legacy_load(*_args, **_kwargs):
        raise AssertionError("legacy comparison model must not be loaded by the assisted worker")

    class DoodleverseFixture:
        model_id = "doodleverse-fixture"

        def __init__(self, _path, *, model_version, **_kwargs):
            self.model_version = model_version

    monkeypatch.setattr(worker, "OnnxTemporalAccident", unexpected_legacy_load)
    monkeypatch.setattr(worker, "OnnxFloodSegmenter", unexpected_legacy_load)
    monkeypatch.setattr(worker, "DoodleverseFloodSegmenter", DoodleverseFixture)
    monkeypatch.setattr(worker, "check_optional_model_release", lambda *_args, **_kwargs: {
        "approved": True, "reason": "approved",
    })

    models, capabilities = worker.load_trial_models(include_assisted_flood=True)

    assert models["accident"] is None
    assert models["flood"].model_id == "doodleverse-fixture"
    assert capabilities["flood_assistance_supported"] is True
    assert capabilities["flood_assistance_release_status"] == "approved"
    # The existing release evidence covers marked road regions only.
    assert capabilities["flood_assistance_automatic_supported"] is False
    assert capabilities["trial_flood_automatic_supported"] is False


def test_assisted_worker_is_not_ready_without_the_traffic_detector(monkeypatch, tmp_path):
    flood_path = tmp_path / "flood.onnx"
    flood_path.write_bytes(b"fixture")
    monkeypatch.setattr(worker.config, "VISION_TRIAL_ENABLED", False, raising=False)
    monkeypatch.setattr(worker.config, "VISION_FLOOD_ENABLED", True, raising=False)
    monkeypatch.setattr(worker.config, "VISION_FLOOD_RELEASE_MODE", "assisted_review", raising=False)
    monkeypatch.setattr(worker.config, "VISION_FLOOD_MODEL_PATH", str(flood_path), raising=False)
    monkeypatch.setattr(worker.config, "VISION_MODEL_PATH", str(tmp_path / "missing-traffic.onnx"), raising=False)
    def unexpected_release_check(*_args, **_kwargs):
        raise AssertionError("release approval is not checked without traffic detection")

    monkeypatch.setattr(worker, "check_optional_model_release", unexpected_release_check)

    models, capabilities = worker.load_trial_models(include_assisted_flood=True)

    assert models == {"accident": None, "flood": None}
    assert capabilities["flood_assistance_supported"] is False
    assert capabilities["flood_assistance_release_status"] == "traffic_detector_missing"


def test_assisted_flood_runs_traffic_stage_then_releases_detector_before_flood(monkeypatch):
    calls = []
    detector_ref = None
    regions = [
        {"id": "lane", "kind": "vehicle_lane", "polygon": [[0, 0], [1, 0], [1, 1]]},
        {"id": "road", "kind": "road_surface", "polygon": [[0, 0], [1, 0], [1, 1]]},
    ]

    class Detector:
        model_id = "traffic-fixture"
        model_version = "test-v1"

    def create_detector():
        nonlocal detector_ref
        detector = Detector()
        detector_ref = weakref.ref(detector)
        return detector

    def analyze(_path, _kind, _anchor, **kwargs):
        calls.append({key: value for key, value in kwargs.items() if key != "detector"})
        if kwargs["flood_segmenter"] is None:
            assert kwargs["detector"].model_id == "traffic-fixture"
            return {
                "candidates": [{"id": "traffic-candidate"}],
                "metrics": {"peak_vehicle_count": 2, "specialized_model_frames": {}},
                "safety": {"requires_human_review": True},
                "preview_detections": [{"label": "car"}],
                "model": {"id": "traffic-fixture", "version": "test-v1"},
            }
        assert detector_ref() is None
        assert kwargs["detector"].model_id == "trial-no-vehicle-detection"
        assert kwargs["accident_model"] is None
        assert kwargs["regions"] == [regions[1]]
        assert kwargs["require_flood_success"] is True
        return {
            "candidates": [{"id": "flood-candidate"}],
            "metrics": {"specialized_model_frames": {"flood": 1}},
            "safety": {"flood_segmentation_supported": True},
            "flood_review": [{"region_id": "road", "flooded_road_area_ratio": 0.04}],
            "model": {"flood_model": {"id": "doodleverse-fixture", "version": "v7"}},
        }

    monkeypatch.setattr(assisted_image, "analyze_media", analyze)
    result = worker._analyze_assisted_flood_image(
        "image.png", "image", {"lat": 1, "lng": 2},
        camera_stabilized=False, regions=regions,
        progress_callback=lambda _snapshot: None, resume_state=None,
        cancel_check=lambda: False, flood_segmenter=object(),
        detector_factory=create_detector,
    )

    assert len(calls) == 2
    assert calls[0]["accident_model"] is None and calls[0]["flood_segmenter"] is None
    assert result["candidates"] == [
        {"id": "traffic-candidate"}, {"id": "flood-candidate"},
    ]
    assert result["preview_detections"] == [{"label": "car"}]
    assert result["metrics"]["peak_vehicle_count"] == 2
    assert result["metrics"]["specialized_model_frames"] == {"flood": 1}
    assert result["safety"] == {
        "requires_human_review": True, "flood_segmentation_supported": True,
    }
    assert result["flood_review"][0]["region_id"] == "road"
    assert result["model"]["flood_model"]["id"] == "doodleverse-fixture"


def test_assisted_flood_stops_before_segmentation_when_cancelled_after_detection(monkeypatch):
    calls = []
    monkeypatch.setattr(assisted_image, "analyze_media", lambda *_args, **_kwargs: (
        calls.append("traffic") or {"candidates": [], "metrics": {}, "safety": {}}
    ))
    cancel_calls = iter([True])

    try:
        worker._analyze_assisted_flood_image(
            "image.png", "image", None,
            camera_stabilized=False,
            regions=[{"id": "road", "kind": "road_surface", "polygon": [[0, 0], [1, 0], [1, 1]]}],
            progress_callback=None, resume_state=None,
            cancel_check=lambda: next(cancel_calls), flood_segmenter=object(),
            detector_factory=lambda: object(),
        )
    except worker.VisionAnalysisCancelled:
        pass
    else:
        raise AssertionError("cancel request must prevent the flood subprocess from starting")

    assert calls == ["traffic"]
