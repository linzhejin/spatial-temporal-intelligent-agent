import copy

import pytest

from vision.event_evaluation import (
    extract_accident_event_candidates,
    extract_event_candidates,
    match_event_candidates,
    summarize_event_detection,
    validate_event_manifest,
)
from scripts.vision.evaluate_accident_events import run_event_evaluation
from scripts.vision.evaluate_campus_events import run_campus_event_evaluation


def _sample(sample_id, group_id, events=None):
    return {
        "sample_id": sample_id,
        "video": f"test/{sample_id}.mp4",
        "group_id": group_id,
        "camera_stabilized": True,
        "observation_regions": [{
            "id": "road-1",
            "kind": "vehicle_lane",
            "polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
        }],
        "events": events or [],
    }


def _event(event_id, start=10.0, end=14.0, region_id="road-1"):
    return {"event_id": event_id, "region_id": region_id,
            "start_seconds": start, "end_seconds": end}


def _manifest():
    return {
        "schema_version": 1,
        "task": "accident_event_detection",
        "dataset": "WHU UAV event review set",
        "split_unit": "flight_id",
        "split_groups": {
            "train": ["flight-train"],
            "validation": ["flight-validation"],
            "test": ["flight-a", "flight-b"],
        },
        "samples": [
            _sample("positive", "flight-a", [_event("event-1")]),
            _sample("negative", "flight-a"),
            _sample("missed", "flight-b", [_event("event-2", 30.0, 34.0)]),
        ],
    }


def _campus_event_manifest():
    return {
        "schema_version": 1,
        "task": "campus_event_detection",
        "dataset": "WHU UAV campus event set",
        "split_unit": "flight_id",
        "label_source": "manual",
        "event_kinds": ["congestion", "crowding", "accident", "flooding"],
        "split_groups": {
            "train": ["flight-train"],
            "validation": ["flight-validation"],
            "test": ["flight-a", "flight-b"],
        },
        "samples": [
            {
                "sample_id": "traffic",
                "video": "test/traffic.mp4",
                "group_id": "flight-a",
                "camera_stabilized": True,
                "event_kinds": ["congestion", "accident", "flooding"],
                "observation_regions": [
                    {"id": "lane-1", "kind": "vehicle_lane",
                     "polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]},
                    {"id": "surface-1", "kind": "road_surface",
                     "polygon": [[0.15, 0.25], [0.85, 0.25], [0.85, 0.75], [0.15, 0.75]]},
                ],
                "events": [
                    {**_event("queue-1", 10, 20, "lane-1"), "kind": "congestion"},
                    {**_event("water-1", 30, 36, "surface-1"), "kind": "flooding"},
                ],
            },
            {
                "sample_id": "people",
                "video": "test/people.mp4",
                "group_id": "flight-b",
                "camera_stabilized": True,
                "event_kinds": ["crowding"],
                "observation_regions": [
                    {"id": "walk-1", "kind": "pedestrian",
                     "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]},
                ],
                "events": [
                    {**_event("crowd-1", 5, 12, "walk-1"), "kind": "crowding"},
                ],
            },
        ],
    }


def test_event_manifest_requires_video_and_disjoint_flight_level_test_groups():
    manifest = _manifest()

    validated = validate_event_manifest(manifest)

    assert validated["split_groups"] == manifest["split_groups"]
    assert validated["samples"] == manifest["samples"]

    leaky = copy.deepcopy(manifest)
    leaky["split_groups"]["train"].append("flight-a")
    with pytest.raises(ValueError, match="must not overlap"):
        validate_event_manifest(leaky)

    frame_split = copy.deepcopy(manifest)
    frame_split["split_unit"] = "frame_id"
    with pytest.raises(ValueError, match="flight, scene, or incident"):
        validate_event_manifest(frame_split)

    unrepresented_group = copy.deepcopy(manifest)
    unrepresented_group["split_groups"]["test"].append("flight-c")
    with pytest.raises(ValueError, match="test groups must all have at least one video"):
        validate_event_manifest(unrepresented_group)


def test_event_manifest_rejects_invalid_intervals_and_regions():
    manifest = _manifest()
    manifest["samples"][0]["events"][0]["end_seconds"] = 10.0

    with pytest.raises(ValueError, match="end_seconds must be after start_seconds"):
        validate_event_manifest(manifest)

    manifest = _manifest()
    manifest["samples"][0]["events"][0]["region_id"] = "unknown-road"
    with pytest.raises(ValueError, match="must reference a marked vehicle_lane region"):
        validate_event_manifest(manifest)


def test_campus_event_manifest_validates_event_kinds_against_region_types():
    manifest = _campus_event_manifest()

    validated = validate_event_manifest(manifest)

    assert validated["event_kinds"] == ["congestion", "crowding", "accident", "flooding"]
    assert validated["samples"][0]["events"][0]["kind"] == "congestion"
    assert validated["samples"][1]["event_kinds"] == ["crowding"]

    wrong_region = copy.deepcopy(manifest)
    wrong_region["samples"][1]["events"][0]["region_id"] = "lane-1"
    with pytest.raises(ValueError, match="must reference a pedestrian region"):
        validate_event_manifest(wrong_region)

    missing_roi = copy.deepcopy(manifest)
    missing_roi["samples"][1]["observation_regions"][0]["kind"] = "vehicle_lane"
    with pytest.raises(ValueError, match="requires a pedestrian observation region"):
        validate_event_manifest(missing_roi)


def test_campus_event_manifest_requires_declared_supported_sample_kinds():
    manifest = _campus_event_manifest()
    manifest["samples"][0]["event_kinds"] = ["congestion", "unknown"]

    with pytest.raises(ValueError, match="unsupported event kind"):
        validate_event_manifest(manifest)

    manifest = _campus_event_manifest()
    manifest["samples"][0]["events"][0]["kind"] = "crowding"
    with pytest.raises(ValueError, match="not enabled for this sample"):
        validate_event_manifest(manifest)

    manifest = _campus_event_manifest()
    manifest["annotation_pending"] = True
    with pytest.raises(ValueError, match="annotation_pending"):
        validate_event_manifest(manifest)

def test_accident_candidates_use_evidence_time_segments_and_ignore_other_types():
    analysis = {"candidates": [
        {"kind": "possible_accident", "confidence": 0.92,
         "evidence": {"summary": {"region_id": "road-1"}, "segments": [
             {"start_seconds": 11.0, "end_seconds": 12.0},
             {"start_seconds": 12.0, "end_seconds": 13.0},
         ]}},
        {"kind": "possible_flooding", "confidence": 0.99,
         "evidence": {"summary": {"region_id": "road-1"}, "segments": [
             {"start_seconds": 11.0, "end_seconds": 13.0},
         ]}},
    ]}

    candidates = extract_accident_event_candidates(analysis)

    assert candidates == [{"candidate_id": "candidate-1", "region_id": "road-1",
                           "start_seconds": 11.0, "end_seconds": 13.0,
                           "confidence": 0.92}]


def test_event_candidates_map_all_production_event_types_and_evidence_shapes():
    analysis = {"candidates": [
        {"kind": "possible_congestion", "confidence": 0.81,
         "evidence": {"summary": {"region_id": "lane-1"}, "segments": [
             {"start_seconds": 10, "end_seconds": 20},
         ]}},
        {"kind": "possible_crowding", "confidence": 0.76,
         "evidence": {"region_id": "walk-1", "segments": [
             {"start_seconds": 4, "end_seconds": 9},
         ]}},
        {"kind": "possible_accident", "confidence": 0.91,
         "evidence": {"summary": {"region_id": "lane-1"}, "segments": [
             {"start_seconds": 31, "end_seconds": 33},
         ]}},
        {"kind": "possible_flooding", "confidence": 0.88,
         "evidence": {"summary": {"region_id": "surface-1"}, "segments": [
             {"start_seconds": 40, "end_seconds": 45},
         ]}},
        {"kind": "unrelated", "confidence": 1.0, "evidence": {}},
    ]}

    candidates = extract_event_candidates(analysis)

    assert [(item["kind"], item["region_id"]) for item in candidates] == [
        ("congestion", "lane-1"),
        ("crowding", "walk-1"),
        ("accident", "lane-1"),
        ("flooding", "surface-1"),
    ]
    assert candidates[0]["start_seconds"] == 10.0
    assert candidates[-1]["end_seconds"] == 45.0


def test_event_matching_is_one_to_one_and_requires_same_observation_region():
    actual = [_event("event-1"), _event("event-2", 20.0, 24.0)]
    predicted = [
        {"candidate_id": "candidate-1", "region_id": "road-1",
         "start_seconds": 10.5, "end_seconds": 12.0, "confidence": 0.91},
        {"candidate_id": "candidate-2", "region_id": "road-1",
         "start_seconds": 10.0, "end_seconds": 14.0, "confidence": 0.95},
        {"candidate_id": "candidate-3", "region_id": "road-2",
         "start_seconds": 20.0, "end_seconds": 24.0, "confidence": 0.99},
    ]

    result = match_event_candidates(actual, predicted, min_temporal_iou=0.2)

    assert [(pair["event_id"], pair["candidate_id"]) for pair in result["matches"]] == [
        ("event-1", "candidate-2"),
    ]
    assert result["unmatched_event_ids"] == ["event-2"]
    assert result["unmatched_candidate_ids"] == ["candidate-1", "candidate-3"]


def test_event_matching_requires_the_same_event_kind():
    actual = [{**_event("queue-1", 10, 20), "kind": "congestion"}]
    predicted = [{"candidate_id": "crowd-1", "kind": "crowding", "region_id": "road-1",
                  "start_seconds": 10, "end_seconds": 20, "confidence": 0.99}]

    result = match_event_candidates(actual, predicted)

    assert result["matches"] == []
    assert result["unmatched_event_ids"] == ["queue-1"]
    assert result["unmatched_candidate_ids"] == ["crowd-1"]


def test_event_report_computes_event_metrics_and_flight_cluster_intervals():
    manifest = validate_event_manifest(_manifest())
    predictions = {
        "positive": [{"candidate_id": "p1", "region_id": "road-1",
                      "start_seconds": 10.0, "end_seconds": 14.0, "confidence": 0.95}],
        "negative": [{"candidate_id": "p2", "region_id": "road-1",
                      "start_seconds": 5.0, "end_seconds": 7.0, "confidence": 0.9}],
        "missed": [],
    }

    report = summarize_event_detection(manifest, predictions, min_temporal_iou=0.2)

    assert report["counts"] == {"tp": 1, "fp": 1, "fn": 1}
    assert report["metrics"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert report["flight_cluster_bootstrap_95_ci"]["precision"]["resampling_group_count"] == 2
    assert report["samples"]["negative"]["counts"] == {"tp": 0, "fp": 1, "fn": 0}
    assert report["by_kind"]["accident"]["counts"] == {"tp": 1, "fp": 1, "fn": 1}


def test_campus_event_report_separates_metrics_by_kind():
    manifest = validate_event_manifest(_campus_event_manifest())
    predictions = {
        "traffic": [
            {"candidate_id": "queue-p", "kind": "congestion", "region_id": "lane-1",
             "start_seconds": 10, "end_seconds": 20, "confidence": 0.9},
            {"candidate_id": "accident-fp", "kind": "accident", "region_id": "lane-1",
             "start_seconds": 15, "end_seconds": 18, "confidence": 0.9},
        ],
        "people": [],
    }

    report = summarize_event_detection(manifest, predictions, bootstrap_repetitions=100)

    assert report["counts"] == {"tp": 1, "fp": 1, "fn": 2}
    assert report["by_kind"]["congestion"]["counts"] == {"tp": 1, "fp": 0, "fn": 0}
    assert report["by_kind"]["crowding"]["counts"] == {"tp": 0, "fp": 0, "fn": 1}
    assert report["by_kind"]["accident"]["counts"] == {"tp": 0, "fp": 1, "fn": 0}
    assert report["by_kind"]["flooding"]["counts"] == {"tp": 0, "fp": 0, "fn": 1}


def test_event_report_rejects_predictions_for_unknown_samples():
    manifest = validate_event_manifest(_manifest())

    with pytest.raises(ValueError, match="unknown sample_id"):
        summarize_event_detection(manifest, {"stale-id": []})


def test_event_evaluator_runs_the_production_video_pipeline_and_reports_model_provenance(tmp_path):
    manifest = validate_event_manifest(_manifest())
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["video"]).write_bytes(sample["sample_id"].encode("ascii"))
    detector_path = tmp_path / "detector.onnx"
    accident_path = tmp_path / "accident.onnx"
    detector_path.write_bytes(b"detector-artifact")
    accident_path.write_bytes(b"accident-artifact")

    calls = []

    class Detector:
        def __init__(self, path):
            assert path == str(detector_path)

    class Classifier:
        def __init__(self, path, *, threshold):
            assert path == str(accident_path)
            assert threshold == 0.87

    def analyze(path, media_kind, anchor, **kwargs):
        calls.append((path.name, media_kind, anchor, kwargs["camera_stabilized"], kwargs["regions"]))
        candidates = []
        if path.name == "positive.mp4":
            candidates = [{"kind": "possible_accident", "confidence": 0.95,
                           "evidence": {"summary": {"region_id": "road-1"},
                                        "segments": [{"start_seconds": 10, "end_seconds": 14}]}}]
        return {"video_coverage": {"complete": True, "duration_seconds": 60,
                                   "sample_interval_seconds": 1.0},
                "safety": {"accident_recognition_supported": True},
                "candidates": candidates}

    report = run_event_evaluation(
        manifest, dataset_root, detector_path, accident_path, threshold=0.87,
        analyzer=analyze, detector_factory=Detector, classifier_factory=Classifier,
    )

    assert report["event_metrics"]["counts"] == {"tp": 1, "fp": 0, "fn": 1}
    assert report["event_metrics"]["metrics"] == {"precision": 1.0, "recall": 0.5, "f1": 0.666667}
    assert report["model_provenance"]["detector_sha256"]
    assert report["model_provenance"]["accident_model_sha256"]
    assert len(calls) == 3
    assert all(call[1:4] == ("video", None, True) for call in calls)


def test_event_evaluator_rejects_labels_outside_decoded_video_duration(tmp_path):
    manifest = validate_event_manifest(_manifest())
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["video"]).write_bytes(b"video")
    detector_path = tmp_path / "detector.onnx"
    accident_path = tmp_path / "accident.onnx"
    detector_path.write_bytes(b"d")
    accident_path.write_bytes(b"a")
    manifest["samples"][0]["events"][0]["end_seconds"] = 70.0

    def analyze(*_args, **_kwargs):
        return {"video_coverage": {"complete": True, "duration_seconds": 60},
                "safety": {"accident_recognition_supported": True}, "candidates": []}

    with pytest.raises(ValueError, match="extends beyond decoded video duration"):
        run_event_evaluation(manifest, dataset_root, detector_path, accident_path,
                             threshold=0.87, analyzer=analyze,
                             detector_factory=lambda _path: object(),
                             classifier_factory=lambda *_args, **_kwargs: object())


def _people_and_traffic_manifest():
    manifest = _campus_event_manifest()
    manifest["event_kinds"] = ["congestion", "crowding"]
    traffic = manifest["samples"][0]
    traffic["event_kinds"] = ["congestion"]
    traffic["observation_regions"] = [traffic["observation_regions"][0]]
    traffic["events"] = [traffic["events"][0]]
    return manifest


def test_campus_event_evaluator_runs_people_and_traffic_without_optional_models(tmp_path):
    manifest = validate_event_manifest(_people_and_traffic_manifest())
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["video"]).write_bytes(sample["sample_id"].encode("ascii"))
    detector_path = tmp_path / "detector.onnx"
    detector_path.write_bytes(b"detector")
    calls = []

    def analyze(path, media_kind, anchor, **kwargs):
        calls.append((path.name, kwargs))
        if path.name == "traffic.mp4":
            candidates = [{
                "kind": "possible_congestion", "confidence": 0.9,
                "evidence": {"summary": {"region_id": "lane-1"},
                             "segments": [{"start_seconds": 10, "end_seconds": 20}]},
            }]
        else:
            candidates = [{
                "kind": "possible_crowding", "confidence": 0.8,
                "evidence": {"summary": {"region_id": "walk-1"},
                             "segments": [{"start_seconds": 5, "end_seconds": 12}]},
            }]
        return {
            "video_coverage": {"complete": True, "duration_seconds": 60,
                               "sample_interval_seconds": 0.2},
            "safety": {}, "candidates": candidates,
        }

    report = run_campus_event_evaluation(
        manifest, dataset_root, detector_path,
        analyzer=analyze, detector_factory=lambda _path: object(),
        bootstrap_repetitions=100,
    )

    assert report["event_metrics"]["counts"] == {"tp": 2, "fp": 0, "fn": 0}
    assert report["event_metrics"]["by_kind"]["congestion"]["metrics"]["recall"] == 1.0
    assert report["event_metrics"]["by_kind"]["crowding"]["metrics"]["recall"] == 1.0
    assert report["model_provenance"]["detector_sha256"]
    assert report["model_provenance"]["accident_model"] is None
    assert report["model_provenance"]["flood_model"] is None
    assert all(call[1]["accident_model"] is None for call in calls)
    assert all(call[1]["flood_segmenter"] is None for call in calls)


def test_campus_event_evaluator_requires_models_and_frozen_thresholds_for_declared_kinds(tmp_path):
    manifest = _campus_event_manifest()
    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    detector_path = tmp_path / "detector.onnx"
    detector_path.write_bytes(b"detector")

    with pytest.raises(ValueError, match="accident model and threshold are required"):
        run_campus_event_evaluation(
            manifest, dataset_root, detector_path,
            analyzer=lambda *_args, **_kwargs: {}, detector_factory=lambda _path: object(),
        )

    accident_path = tmp_path / "accident.onnx"
    accident_path.write_bytes(b"accident")
    with pytest.raises(ValueError, match="flood model and frozen thresholds are required"):
        run_campus_event_evaluation(
            manifest, dataset_root, detector_path,
            accident_model_path=accident_path, accident_threshold=0.87,
            analyzer=lambda *_args, **_kwargs: {}, detector_factory=lambda _path: object(),
            classifier_factory=lambda *_args, **_kwargs: object(),
        )


def test_campus_event_evaluator_rejects_duplicate_video_content_before_inference(tmp_path):
    manifest = validate_event_manifest(_people_and_traffic_manifest())
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["video"]).write_bytes(b"same video bytes")
    detector_path = tmp_path / "detector.onnx"
    detector_path.write_bytes(b"detector")

    with pytest.raises(ValueError, match="duplicate video content"):
        run_campus_event_evaluation(
            manifest, dataset_root, detector_path,
            analyzer=lambda *_args, **_kwargs: pytest.fail("inference must not start"),
            detector_factory=lambda _path: object(), bootstrap_repetitions=100,
        )


def test_campus_event_evaluator_rejects_a_source_video_changed_after_preparation(tmp_path):
    manifest = validate_event_manifest(_people_and_traffic_manifest())
    manifest["samples"][0]["source_sha256"] = "0" * 64
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["video"]).write_bytes(sample["sample_id"].encode("ascii"))
    detector_path = tmp_path / "detector.onnx"
    detector_path.write_bytes(b"detector")

    with pytest.raises(ValueError, match="source_sha256 does not match"):
        run_campus_event_evaluation(
            manifest, dataset_root, detector_path,
            analyzer=lambda *_args, **_kwargs: pytest.fail("inference must not start"),
            detector_factory=lambda _path: object(), bootstrap_repetitions=100,
        )
