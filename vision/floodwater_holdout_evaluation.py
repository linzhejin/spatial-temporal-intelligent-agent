"""Metrics for source-video-held-out visible-water pseudo-label evaluation."""
from __future__ import annotations

from collections import defaultdict

from vision.visible_water_evaluation import metrics_from_counts


def aggregate_source_video_counts(samples: list[dict], sample_counts: list[dict]) -> dict:
    """Aggregate pixel counts overall and by held-out source video.

    Frames from one source video are correlated. The report therefore exposes
    video-level metrics and does not treat frames as independent replicates.
    """
    if not isinstance(samples, list) or not samples:
        raise ValueError("at least one held-out sample is required")
    if not isinstance(sample_counts, list) or len(samples) != len(sample_counts):
        raise ValueError("samples and counts must contain the same number of records")

    totals = {key: 0 for key in ("tp", "fp", "fn")}
    groups: dict[str, dict] = defaultdict(
        lambda: {"sample_count": 0, "positive_label_frame_count": 0,
                 "tp": 0, "fp": 0, "fn": 0},
    )
    sample_ids = set()
    for index, (sample, counts) in enumerate(zip(samples, sample_counts)):
        if not isinstance(sample, dict) or sample.get("split") != "test":
            raise ValueError(f"sample {index} must belong to the test split")
        source_video_id = str(sample.get("source_video_id") or "").strip()
        sample_id = str(sample.get("sample_id") or "").strip()
        if not source_video_id or not sample_id or sample_id in sample_ids:
            raise ValueError(f"sample {index} requires a unique ID and source video")
        sample_ids.add(sample_id)
        if not isinstance(counts, dict):
            raise ValueError(f"counts for sample {sample_id} must be an object")
        normalized = {}
        for key in ("tp", "fp", "fn"):
            value = counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} for sample {sample_id} must be a non-negative integer")
            normalized[key] = value

        group = groups[source_video_id]
        group["sample_count"] += 1
        group["positive_label_frame_count"] += int(normalized["tp"] + normalized["fn"] > 0)
        for key, value in normalized.items():
            group[key] += value
            totals[key] += value

    per_source_video = {
        source_video_id: {
            **metrics_from_counts({key: int(value[key]) for key in ("tp", "fp", "fn")}),
            "sample_count": int(value["sample_count"]),
            "positive_label_frame_count": int(value["positive_label_frame_count"]),
        }
        for source_video_id, value in sorted(groups.items())
    }
    video_ious = [value["iou"] for value in per_source_video.values()
                  if value["iou"] is not None]
    return {
        "sample_count": len(samples),
        "source_video_count": len(per_source_video),
        "aggregate": metrics_from_counts(totals),
        "macro_source_video_iou": (
            sum(video_ious) / len(video_ious) if video_ious else None
        ),
        "per_source_video": per_source_video,
    }
