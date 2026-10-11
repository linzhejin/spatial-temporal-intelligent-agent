import copy

import numpy as np
import pytest

from scripts.vision.evaluate_campus_detection import run_campus_detection_evaluation
from vision.campus_detection_evaluation import validate_campus_detection_manifest
from vision.evaluation import evaluate_grouped_frame


def _manifest():
    return {
        "schema_version": 1,
        "task": "campus_object_detection",
        "dataset": "WHU UAV object test",
        "split_unit": "flight_id",
        "label_source": "manual",
        "split_groups": {
            "train": ["flight-train"],
            "validation": ["flight-validation"],
            "test": ["flight-a", "flight-b"],
        },
        "samples": [
            {
                "sample_id": "a-1", "group_id": "flight-a", "image": "test/a.jpg",
                "width": 100, "height": 80,
                "objects": [
                    {"label": "pedestrian", "box": [0, 0, 10, 20]},
                    {"label": "vehicle", "box": [20, 20, 40, 40]},
                ],
            },
            {
                "sample_id": "b-1", "group_id": "flight-b", "image": "test/b.jpg",
                "width": 100, "height": 80,
                "objects": [
                    {"label": "ignore", "box": [50, 50, 70, 70]},
                ],
            },
        ],
    }


def test_campus_detection_manifest_requires_disjoint_flights_and_complete_test_groups():
    manifest = _manifest()

    validated = validate_campus_detection_manifest(manifest)

    assert validated["samples"][0]["objects"][1]["label"] == "vehicle"

    overlap = copy.deepcopy(manifest)
    overlap["split_groups"]["train"].append("flight-a")
    with pytest.raises(ValueError, match="must not overlap"):
        validate_campus_detection_manifest(overlap)

    missing = copy.deepcopy(manifest)
    missing["samples"] = missing["samples"][:1]
    with pytest.raises(ValueError, match="test groups must all have at least one image"):
        validate_campus_detection_manifest(missing)


def test_campus_detection_manifest_rejects_bad_boxes_and_unsafe_paths():
    manifest = _manifest()
    manifest["samples"][0]["objects"][0]["box"] = [0, 0, 101, 20]
    with pytest.raises(ValueError, match="inside the declared image dimensions"):
        validate_campus_detection_manifest(manifest)

    manifest = _manifest()
    manifest["samples"][0]["image"] = "../outside.jpg"
    with pytest.raises(ValueError, match="safe relative path"):
        validate_campus_detection_manifest(manifest)

    manifest = _manifest()
    manifest["annotation_pending"] = True
    with pytest.raises(ValueError, match="annotation_pending"):
        validate_campus_detection_manifest(manifest)


def test_grouped_detector_evaluation_matches_labels_by_route_relevant_group():
    annotations = [
        {"label": "pedestrian", "box": [0, 0, 10, 20]},
        {"label": "vehicle", "box": [20, 20, 40, 40]},
        {"label": "ignore", "box": [50, 50, 70, 70]},
    ]
    predictions = [
        {"label": "people", "confidence": 0.9, "box": [0, 0, 10, 20]},
        {"label": "bus", "confidence": 0.8, "box": [20, 20, 40, 40]},
        {"label": "car", "confidence": 0.7, "box": [50, 50, 70, 70]},
        {"label": "pedestrian", "confidence": 0.6, "box": [75, 50, 85, 70]},
    ]

    result = evaluate_grouped_frame(predictions, annotations, confidence_threshold=0.5)

    assert result["pedestrian"] == {"tp": 1, "fp": 1, "fn": 0, "ignored": 0}
    assert result["vehicle"] == {"tp": 1, "fp": 0, "fn": 0, "ignored": 1}


def test_campus_detection_runner_reports_flight_clustered_metrics_and_provenance(tmp_path):
    manifest = validate_campus_detection_manifest(_manifest())
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["image"]).write_bytes(sample["sample_id"].encode("ascii"))
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"detector")

    class Detector:
        model_id = "campus-detector"
        model_version = "test"

        def __init__(self, path, **kwargs):
            assert path == str(model_path)
            assert kwargs["confidence_threshold"] == 0.5

        def detect(self, frame):
            assert frame.shape == (80, 100, 3)
            return [
                {"label": "pedestrian", "confidence": 0.9, "box": [0, 0, 10, 20]},
                {"label": "car", "confidence": 0.8, "box": [20, 20, 40, 40]},
            ]

    report = run_campus_detection_evaluation(
        manifest, dataset_root, model_path,
        confidence_threshold=0.5, iou_threshold=0.5,
        detector_factory=Detector,
        image_loader=lambda _path: np.zeros((80, 100, 3), dtype=np.uint8),
        bootstrap_repetitions=100,
    )

    assert report["groups"]["pedestrian"]["tp"] == 1
    assert report["groups"]["pedestrian"]["fp"] == 1
    assert report["groups"]["vehicle"]["tp"] == 1
    assert report["groups"]["vehicle"]["fp"] == 1
    assert report["flight_group_count"] == 2
    assert report["model_sha256"]
    assert report["groups"]["vehicle"]["flight_cluster_bootstrap_95_ci"]["recall"][
        "resampling_group_count"
    ] == 2


def test_campus_detection_runner_rejects_decoded_dimension_mismatch(tmp_path):
    manifest = validate_campus_detection_manifest(_manifest())
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["image"]).write_bytes(sample["sample_id"].encode("ascii"))
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"detector")

    with pytest.raises(ValueError, match="decoded dimensions do not match"):
        run_campus_detection_evaluation(
            manifest, dataset_root, model_path,
            detector_factory=lambda *_args, **_kwargs: object(),
            image_loader=lambda _path: np.zeros((40, 50, 3), dtype=np.uint8),
            bootstrap_repetitions=100,
        )


def test_campus_detection_runner_rejects_duplicate_image_content(tmp_path):
    manifest = validate_campus_detection_manifest(_manifest())
    dataset_root = tmp_path / "data"
    (dataset_root / "test").mkdir(parents=True)
    for sample in manifest["samples"]:
        (dataset_root / sample["image"]).write_bytes(b"same image bytes")
    model_path = tmp_path / "detector.onnx"
    model_path.write_bytes(b"detector")

    with pytest.raises(ValueError, match="duplicate image content"):
        run_campus_detection_evaluation(
            manifest, dataset_root, model_path,
            detector_factory=lambda *_args, **_kwargs: object(),
            image_loader=lambda _path: np.zeros((80, 100, 3), dtype=np.uint8),
            bootstrap_repetitions=100,
        )
