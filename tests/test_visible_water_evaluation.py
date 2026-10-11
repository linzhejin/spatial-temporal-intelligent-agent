import numpy as np
import pytest
from PIL import Image
from pathlib import Path
import vision.visible_water_evaluation as water_evaluation

from vision.visible_water_evaluation import (
    binary_segmentation_counts,
    bootstrap_iou_interval,
    discover_manual_pairs,
    metrics_from_counts,
    visible_water_prediction_mask,
)


def test_manual_water_evaluation_cli_is_valid_python():
    source = Path("scripts/vision/evaluate_visible_water_manual.py").read_text(encoding="utf-8")

    compile(source, "evaluate_visible_water_manual.py", "exec")


def test_binary_segmentation_metrics_count_pixels_without_smoothing():
    counts = binary_segmentation_counts(
        np.asarray([[1, 1], [0, 0]], dtype=bool),
        np.asarray([[1, 0], [1, 0]], dtype=bool),
    )

    assert counts == {"tp": 1, "fp": 1, "fn": 1}
    assert metrics_from_counts(counts) == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "iou": pytest.approx(1 / 3),
        "precision": pytest.approx(1 / 2),
        "recall": pytest.approx(1 / 2),
    }


def test_image_bootstrap_interval_is_deterministic_and_reports_image_unit():
    counts = [
        {"tp": 8, "fp": 2, "fn": 2},
        {"tp": 1, "fp": 8, "fn": 9},
    ]

    first = bootstrap_iou_interval(counts, seed=21, replicates=100)
    second = bootstrap_iou_interval(counts, seed=21, replicates=100)

    assert first == second
    assert first[0] <= first[1]


def test_manual_pair_discovery_requires_matching_ids_and_binary_masks(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    Image.new("RGB", (2, 2)).save(image_dir / "sample_000001.jpg")
    Image.fromarray(np.asarray([[0, 255], [0, 255]], dtype=np.uint8)).save(
        mask_dir / "sample_000001.png",
    )

    pairs = discover_manual_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0]["sample_id"] == "sample_000001"
    assert pairs[0]["image"] == image_dir / "sample_000001.jpg"
    assert pairs[0]["mask"] == mask_dir / "sample_000001.png"

    (mask_dir / "sample_000001.png").unlink()
    with pytest.raises(ValueError, match="image/mask IDs do not match"):
        discover_manual_pairs(tmp_path)


def test_prediction_mask_uses_explicit_water_classes_for_each_model_contract():
    labels = np.asarray([[0, 1, 2, 3, 5, 9]])

    assert visible_water_prediction_mask(labels, class_count=2).tolist() == [[False, True, False, False, False, False]]
    assert visible_water_prediction_mask(labels, class_count=10).tolist() == [[False, False, False, True, True, False]]
    with pytest.raises(ValueError, match="class_count must be 2 or 10"):
        visible_water_prediction_mask(labels, class_count=5)


def test_manual_eval_report_describes_binary_visible_water_model_correctly():
    report_limitations = getattr(water_evaluation, "manual_evaluation_limitations", None)
    assert callable(report_limitations)
    limitations = report_limitations(class_count=2)

    assert any("two-class visible-water model" in item for item in limitations)
    assert not any("FloodNet checkpoint" in item for item in limitations)


def test_manual_eval_report_keeps_floodnet_semantic_warning_for_ten_class_model():
    report_limitations = getattr(water_evaluation, "manual_evaluation_limitations", None)
    assert callable(report_limitations)
    limitations = report_limitations(class_count=10)

    assert any("FloodNet" in item and "ten-class" in item for item in limitations)

    with pytest.raises(ValueError, match="class_count must be 2 or 10"):
        report_limitations(class_count=5)
