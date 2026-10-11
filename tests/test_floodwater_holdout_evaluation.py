import pytest

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
