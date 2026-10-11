import pytest
import numpy as np

from scripts.vision.evaluate_floodwater_pseudolabel_holdout import (
    _visible_water_probability,
    _threshold_comparison_summaries,
)
from scripts.vision.calibrate_visible_water_pseudolabel_threshold import _threshold_counts
from vision.floodwater_holdout_evaluation import aggregate_source_video_counts


def test_aggregate_source_video_counts_reports_micro_and_video_macro_metrics():
    samples = [
        {"sample_id": "a-1", "source_video_id": "video-a", "split": "test"},
        {"sample_id": "a-2", "source_video_id": "video-a", "split": "test"},
        {"sample_id": "b-1", "source_video_id": "video-b", "split": "test"},
    ]
    counts = [
        {"tp": 2, "fp": 1, "fn": 1},
        {"tp": 1, "fp": 0, "fn": 1},
        {"tp": 0, "fp": 2, "fn": 0},
    ]

    result = aggregate_source_video_counts(samples, counts)

    assert result["sample_count"] == 3
    assert result["source_video_count"] == 2
    assert result["aggregate"]["tp"] == 3
    assert result["aggregate"]["fp"] == 3
    assert result["aggregate"]["fn"] == 2
    assert result["aggregate"]["iou"] == pytest.approx(3 / 8)
    assert result["macro_source_video_iou"] == pytest.approx((0.5 + 0.0) / 2)
    assert result["per_source_video"]["video-a"]["sample_count"] == 2


def test_aggregate_source_video_counts_rejects_non_test_or_misaligned_records():
    sample = {"sample_id": "a-1", "source_video_id": "video-a", "split": "train"}

    with pytest.raises(ValueError, match="test split"):
        aggregate_source_video_counts([sample], [{"tp": 1, "fp": 0, "fn": 0}])

    sample["split"] = "test"
    with pytest.raises(ValueError, match="same number"):
        aggregate_source_video_counts([sample], [])


def test_holdout_evaluator_uses_the_runtime_segmenter_on_the_full_frame():
    class RuntimeSegmenter:
        def predict(self, frame, region, *, include_probabilities):
            assert include_probabilities is True
            assert region["kind"] == "road_surface"
            assert region["polygon"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            return {"visible_water_probability": np.full(frame.shape[:2], 0.7)}

    frame = np.zeros((4, 5, 3), dtype=np.uint8)

    probabilities = _visible_water_probability(RuntimeSegmenter(), frame)

    assert probabilities.shape == (4, 5)
    assert np.all(probabilities == pytest.approx(0.7))


def test_holdout_evaluator_rejects_probability_map_not_aligned_to_frame():
    class RuntimeSegmenter:
        def predict(self, frame, region, *, include_probabilities):
            return {"visible_water_probability": np.zeros((3, 5), dtype=np.float32)}

    with pytest.raises(ValueError, match="match the source frame"):
        _visible_water_probability(RuntimeSegmenter(), np.zeros((4, 5, 3), dtype=np.uint8))


def test_threshold_comparison_names_the_fixed_point_five_baseline_honestly():
    samples = [{"sample_id": "a-1", "source_video_id": "video-a", "split": "test"}]
    selected = [{"tp": 7, "fp": 1, "fn": 2}]
    baseline = [{"tp": 5, "fp": 3, "fn": 4}]

    summaries = _threshold_comparison_summaries(samples, selected, baseline)

    assert set(summaries) == {"selected_probability_threshold", "fixed_probability_threshold_0_5"}
    assert summaries["fixed_probability_threshold_0_5"]["aggregate"]["tp"] == 5
    assert summaries["selected_probability_threshold"]["aggregate"]["tp"] == 7


def test_calibration_video_metrics_match_float32_runtime_threshold_comparisons():
    positive_bins = [0] * 512
    negative_bins = [0] * 512
    positive_bins[335:337] = [2, 3]
    negative_bins[335:337] = [4, 5]
    edges = np.linspace(0.0, 1.0, 513).tolist()

    metrics = _threshold_counts(positive_bins, negative_bins, edges, 0.65429688)

    assert metrics["tp"] == 5
    assert metrics["fp"] == 9
    assert metrics["fn"] == 0
