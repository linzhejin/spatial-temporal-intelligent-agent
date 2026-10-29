import pytest
import numpy as np

from scripts.vision.evaluate_specialized import (
    _align_rgb_to_mask_grid,
    _evaluate_visible_water_road,
)

from vision.specialized_evaluation import (
    cluster_bootstrap_interval,
    summarize_binary_classification,
    summarize_binary_counts,
    summarize_classification,
    select_probability_threshold_from_histograms,
    validate_split_manifest,
)


def test_floodnet_evaluation_rejects_mismatched_mask_grid():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    labels = np.full((6, 8), 3, dtype=np.uint8)
    with pytest.raises(ValueError, match="dimensions differ"):
        _align_rgb_to_mask_grid(frame, labels)


def test_split_manifest_requires_scene_disjoint_groups_and_test_samples():
    manifest = {
        "schema_version": 1,
        "task": "accident_classification",
        "dataset": "AIDER local copy",
        "class_order": ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"],
        "split_unit": "incident_scene",
        "split_groups": {
            "train": ["incident-1"],
            "validation": ["incident-2"],
            "test": ["incident-3"],
        },
        "samples": [{"image": "test/image.jpg", "group_id": "incident-3",
                     "label": "traffic_incident"}],
    }

    assert validate_split_manifest(manifest, task="accident_classification")["split_groups"] == manifest["split_groups"]

    manifest["split_groups"]["train"].append("incident-3")
    with pytest.raises(ValueError, match="groups must not overlap"):
        validate_split_manifest(manifest, task="accident_classification")


def test_test_samples_cannot_belong_to_training_or_validation_group():
    manifest = {
        "schema_version": 1,
        "task": "accident_classification",
        "dataset": "AIDER",
        "class_order": ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"],
        "split_unit": "incident_scene",
        "split_groups": {"train": ["scene-a"], "validation": [], "test": ["scene-b"]},
        "samples": [{"image": "bad.jpg", "group_id": "scene-a", "label": "normal"}],
    }

    with pytest.raises(ValueError, match="test split"):
        validate_split_manifest(manifest, task="accident_classification")


def test_classification_report_contains_per_class_and_accident_binary_metrics():
    report = summarize_classification(
        ["normal", "traffic_incident", "traffic_incident", "fire"],
        ["normal", "traffic_incident", "normal", "fire"],
        classes=("fire", "normal", "traffic_incident"),
        accident_class="traffic_incident",
    )

    assert report["accuracy"] == pytest.approx(0.75)
    assert report["per_class"]["traffic_incident"]["recall"] == pytest.approx(0.5)
    assert report["traffic_incident_binary"] == {
        "tp": 1, "fp": 0, "fn": 1, "tn": 2,
        "precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3),
    }


def test_binary_segmentation_summary_reports_iou_precision_recall_and_empty_union():
    assert summarize_binary_counts(tp=3, fp=1, fn=2) == {
        "tp": 3, "fp": 1, "fn": 2,
        "iou": pytest.approx(0.5), "precision": pytest.approx(0.75),
        "recall": pytest.approx(0.6), "f1": pytest.approx(2 / 3),
    }
    assert summarize_binary_counts(tp=0, fp=0, fn=0)["iou"] is None


def test_bootstrap_report_names_resampling_groups_without_claiming_scene_independence():
    report = cluster_bootstrap_interval(
        {"image-1": 1, "image-2": 0}, lambda values: sum(values) / len(values),
    )

    assert report["resampling_group_count"] == 2
    assert "independent_groups" not in report


def test_incident_operating_point_uses_explicit_probability_threshold():
    report = summarize_binary_classification(
        [True, True, False, False], [True, False, True, False],
    )
    assert report == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 1,
        "specificity": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
    }


def test_probability_threshold_selection_uses_validation_histogram_and_iou():
    report = select_probability_threshold_from_histograms(
        positive_bins=[0, 4, 1], negative_bins=[5, 0, 0],
        bin_edges=[0.0, 0.33, 0.67, 1.0],
    )

    assert report["threshold"] == pytest.approx(0.33)
    assert report["tp"] == 5
    assert report["fp"] == 0
    assert report["fn"] == 0
    assert report["iou"] == 1.0


def test_probability_threshold_selection_rejects_empty_or_invalid_histograms():
    with pytest.raises(ValueError, match="positive validation pixels"):
        select_probability_threshold_from_histograms(
            positive_bins=[0, 0], negative_bins=[1, 1], bin_edges=[0.0, 0.5, 1.0],
        )
    with pytest.raises(ValueError, match="same length"):
        select_probability_threshold_from_histograms(
            positive_bins=[1], negative_bins=[1, 0], bin_edges=[0.0, 1.0],
        )


def test_manifest_validator_can_validate_only_validation_samples_for_calibration():
    manifest = {
        "schema_version": 1, "task": "flooded_road_segmentation", "dataset": "FloodNet",
        "split_unit": "labeled_image_id",
        "split_groups": {"train": ["train-1"], "validation": ["val-1"], "test": ["test-1"]},
        "samples": [{"image": "val/image.jpg", "mask": "val/mask.png", "group_id": "val-1",
                     "road_surface_polygon": [[0, 0], [1, 0], [1, 1]],
                     "flooded_road_class_id": 3}],
    }

    validated = validate_split_manifest(
        manifest, task="flooded_road_segmentation", split="validation",
    )

    assert validated["samples"] == manifest["samples"]


def test_campus_water_manifest_requires_disjoint_flights_and_binary_road_labels():
    manifest = {
        "schema_version": 1,
        "task": "visible_water_road_segmentation",
        "dataset": "WHU UAV pilot",
        "split_unit": "flight_id",
        "mask_encoding": "binary_0_255",
        "split_groups": {
            "train": ["flight-train"],
            "validation": ["flight-validation"],
            "test": ["flight-test"],
        },
        "samples": [{
            "image": "test/frame-001.jpg",
            "mask": "test/frame-001.png",
            "group_id": "flight-test",
            "road_surface_polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]],
        }],
    }

    validated = validate_split_manifest(
        manifest, task="visible_water_road_segmentation",
    )

    assert validated["mask_encoding"] == "binary_0_255"
    assert validated["split_unit"] == "flight_id"

    leaked = {**manifest, "split_groups": {
        **manifest["split_groups"], "train": ["flight-test"],
    }}
    with pytest.raises(ValueError, match="groups must not overlap"):
        validate_split_manifest(leaked, task="visible_water_road_segmentation")

    missing_semantics = {key: value for key, value in manifest.items() if key != "mask_encoding"}
    with pytest.raises(ValueError, match="mask_encoding"):
        validate_split_manifest(missing_semantics, task="visible_water_road_segmentation")

    missing_test_flight = {**manifest, "split_groups": {
        **manifest["split_groups"], "test": ["flight-test", "flight-without-frames"],
    }}
    with pytest.raises(ValueError, match="every declared test flight"):
        validate_split_manifest(missing_test_flight, task="visible_water_road_segmentation")


def test_campus_water_evaluator_scores_only_road_roi_and_reports_negative_flights(
        tmp_path, monkeypatch):
    import cv2
    from scripts.vision import evaluate_specialized

    class FakeVisibleWaterSegmenter:
        def __init__(self, model_path, *, flood_probability_threshold, **kwargs):
            assert flood_probability_threshold == pytest.approx(0.5)
            self.is_binary_visible_water = True

        def predict(self, frame, region, *, include_mask):
            assert region["kind"] == "road_surface"
            return {"flooded_road_mask": np.ones(frame.shape[:2], dtype=np.uint8)}

    monkeypatch.setattr(
        evaluate_specialized, "OnnxFloodSegmenter", FakeVisibleWaterSegmenter,
    )
    root = tmp_path / "data"
    (root / "test").mkdir(parents=True)
    cv2.imwrite(str(root / "test" / "positive.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / "test" / "negative.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / "test" / "positive.png"), np.full((4, 4), 255, dtype=np.uint8))
    cv2.imwrite(str(root / "test" / "negative.png"), np.zeros((4, 4), dtype=np.uint8))
    manifest = {
        "split_groups": {"test": ["flight-positive", "flight-negative"]},
        "samples": [
            {"image": "test/positive.jpg", "mask": "test/positive.png",
             "group_id": "flight-positive",
             "road_surface_polygon": [[0, 0], [1 / 3, 0], [1 / 3, 1], [0, 1]]},
            {"image": "test/negative.jpg", "mask": "test/negative.png",
             "group_id": "flight-negative",
             "road_surface_polygon": [[0, 0], [1 / 3, 0], [1 / 3, 1], [0, 1]]},
        ],
    }

    report = _evaluate_visible_water_road(
        manifest, root, tmp_path / "fake.onnx", probability_threshold=0.5,
    )

    assert report["tp"] == 8
    assert report["fp"] == 8
    assert report["fn"] == 0
    assert report["iou"] == pytest.approx(0.5)
    assert report["negative_sample_count"] == 1
    assert report["negative_sample_false_positive_rate"] == pytest.approx(1.0)
    assert report["mean_negative_road_false_positive_area_ratio"] == pytest.approx(1.0)
    assert report["group_bootstrap_95_ci_iou"]["resampling_group_count"] == 2
