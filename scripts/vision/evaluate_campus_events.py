#!/usr/bin/env python3
"""Evaluate campus UAV congestion, crowding, accident, and flooding events.

The runner executes the same full-video analyzer as the visual worker. Labels
and predictions are matched by event kind, observation region, and temporal
intersection-over-union. Optional accident and flood models are mandatory only
when those event kinds are declared by the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.event_evaluation import (  # noqa: E402
    EVENT_KIND_TO_REGION_KIND,
    extract_event_candidates,
    summarize_event_detection,
    validate_event_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_path(root: Path, value: str) -> Path:
    normalized = str(value).replace("\\", "/")
    if (not normalized or PureWindowsPath(normalized).is_absolute()
            or Path(normalized).is_absolute()
            or any(part in {"", ".", ".."} for part in normalized.split("/"))):
        raise ValueError("video paths must be safe relative paths")
    base = root.resolve()
    path = (base / normalized).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError("video path resolves outside dataset root") from error
    if not path.is_file():
        raise ValueError(f"video does not exist: {normalized}")
    return path


def _probability(value, field: str, *, zero_allowed: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    lower_ok = result >= 0 if zero_allowed else result > 0
    if not lower_ok or result > 1:
        operator = "at least zero" if zero_allowed else "greater than zero"
        raise ValueError(f"{field} must be {operator} and at most one")
    return result


def _model_record(path: Path | None) -> dict | None:
    if path is None:
        return None
    return {"file": path.name, "sha256": _sha256(path)}


def run_campus_event_evaluation(
    manifest: dict,
    dataset_root: Path,
    detector_path: Path,
    *,
    accident_model_path: Path | None = None,
    accident_threshold: float | None = None,
    flood_model_path: Path | None = None,
    flood_min_area_ratio: float | None = None,
    flood_probability_threshold: float | None = None,
    min_temporal_iou: float = 0.1,
    bootstrap_repetitions: int = 1000,
    seed: int = 20261011,
    analyzer=None,
    detector_factory=None,
    classifier_factory=None,
    flood_factory=None,
) -> dict:
    """Run production analysis and report flight-clustered event metrics."""
    manifest = validate_event_manifest(manifest)
    if manifest["task"] != "campus_event_detection":
        raise ValueError("campus event evaluator requires task campus_event_detection")
    event_kinds = set(manifest["event_kinds"])

    if "accident" in event_kinds and (accident_model_path is None or accident_threshold is None):
        raise ValueError("accident model and threshold are required for declared accident events")
    if "flooding" in event_kinds and (
        flood_model_path is None
        or flood_min_area_ratio is None
        or flood_probability_threshold is None
    ):
        raise ValueError("flood model and frozen thresholds are required for declared flooding events")

    dataset_root = Path(dataset_root).resolve()
    detector_path = Path(detector_path).resolve()
    if not dataset_root.is_dir() or not detector_path.is_file():
        raise ValueError("dataset root and detector must exist")
    if accident_model_path is not None:
        accident_model_path = Path(accident_model_path).resolve()
        if not accident_model_path.is_file():
            raise ValueError("accident model must exist")
    if flood_model_path is not None:
        flood_model_path = Path(flood_model_path).resolve()
        if not flood_model_path.is_file():
            raise ValueError("flood model must exist")

    video_paths = {}
    video_hashes = {}
    source_sha256_by_sample = {}
    for sample in manifest["samples"]:
        video_path = _dataset_path(dataset_root, sample["video"])
        digest = _sha256(video_path)
        expected_digest = sample.get("source_sha256")
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(
                f"source_sha256 does not match video content for {sample['sample_id']}"
            )
        previous = video_hashes.get(digest)
        if previous is not None:
            raise ValueError(
                f"duplicate video content in samples {previous} and {sample['sample_id']}"
            )
        video_hashes[digest] = sample["sample_id"]
        video_paths[sample["sample_id"]] = video_path
        source_sha256_by_sample[sample["sample_id"]] = digest

    accident_threshold_value = (
        _probability(accident_threshold, "accident_threshold")
        if accident_threshold is not None else None
    )
    flood_area_value = (
        _probability(flood_min_area_ratio, "flood_min_area_ratio")
        if flood_min_area_ratio is not None else None
    )
    flood_probability_value = (
        _probability(flood_probability_threshold, "flood_probability_threshold", zero_allowed=True)
        if flood_probability_threshold is not None else None
    )
    _probability(min_temporal_iou, "min_temporal_iou", zero_allowed=True)
    if not isinstance(bootstrap_repetitions, int) or bootstrap_repetitions < 100:
        raise ValueError("bootstrap_repetitions must be an integer of at least 100")

    if analyzer is None:
        from vision.engine import analyze_media
        analyzer = analyze_media
    if detector_factory is None:
        from vision.engine import OnnxDetector
        detector_factory = OnnxDetector
    if classifier_factory is None:
        from vision.specialized import OnnxAiderClassifier
        classifier_factory = OnnxAiderClassifier
    if flood_factory is None:
        from vision.specialized import OnnxFloodSegmenter
        flood_factory = OnnxFloodSegmenter

    detector = detector_factory(str(detector_path))
    accident_model = (
        classifier_factory(str(accident_model_path), threshold=accident_threshold_value)
        if accident_model_path is not None else None
    )
    flood_segmenter = (
        flood_factory(
            str(flood_model_path), min_area_ratio=flood_area_value,
            flood_probability_threshold=flood_probability_value,
        )
        if flood_model_path is not None else None
    )

    predictions = {}
    duration_by_sample = {}
    analysis_seconds = {}
    for sample in manifest["samples"]:
        video_path = video_paths[sample["sample_id"]]
        started = time.monotonic()
        analysis = analyzer(
            video_path, "video", None,
            camera_stabilized=sample["camera_stabilized"],
            detector=detector,
            accident_model=accident_model,
            flood_segmenter=flood_segmenter,
            regions=sample["observation_regions"],
        )
        elapsed = time.monotonic() - started
        if not isinstance(analysis, dict):
            raise ValueError(f"video analyzer returned no result for {sample['sample_id']}")
        safety = analysis.get("safety")
        safety = safety if isinstance(safety, dict) else {}
        sample_kinds = set(sample["event_kinds"])
        if ("accident" in sample_kinds
                and safety.get("accident_recognition_supported") is not True):
            raise RuntimeError(f"accident model was not reliable for video {sample['sample_id']}")
        if ("flooding" in sample_kinds
                and safety.get("flood_segmentation_supported") is not True):
            raise RuntimeError(f"flood model was not reliable for video {sample['sample_id']}")
        coverage = analysis.get("video_coverage")
        if not isinstance(coverage, dict) or coverage.get("complete") is not True:
            raise RuntimeError(f"video analysis was incomplete for {sample['sample_id']}")
        duration = coverage.get("duration_seconds")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration)) or float(duration) <= 0):
            raise ValueError(f"video duration is unavailable for {sample['sample_id']}")
        duration = float(duration)
        sample_interval = coverage.get("sample_interval_seconds", 1.0)
        if (isinstance(sample_interval, bool) or not isinstance(sample_interval, (int, float))
                or not math.isfinite(float(sample_interval)) or float(sample_interval) <= 0):
            sample_interval = 1.0
        label_tolerance = max(0.5, float(sample_interval))
        if any(event["end_seconds"] > duration + label_tolerance for event in sample["events"]):
            raise ValueError(f"event label extends beyond decoded video duration: {sample['sample_id']}")

        region_kind_by_id = {
            str(region["id"]): str(region["kind"])
            for region in sample["observation_regions"]
        }
        candidates = [
            candidate
            for candidate in extract_event_candidates(analysis, event_kinds=sample["event_kinds"])
            if candidate["region_id"] in region_kind_by_id
            and region_kind_by_id[candidate["region_id"]]
            == EVENT_KIND_TO_REGION_KIND[candidate["kind"]]
            and candidate["start_seconds"] < duration
        ]
        for candidate in candidates:
            candidate["end_seconds"] = min(candidate["end_seconds"], duration)
        predictions[sample["sample_id"]] = candidates
        duration_by_sample[sample["sample_id"]] = round(duration, 3)
        analysis_seconds[sample["sample_id"]] = round(elapsed, 3)

    event_metrics = summarize_event_detection(
        manifest, predictions, min_temporal_iou=min_temporal_iou,
        bootstrap_repetitions=bootstrap_repetitions, seed=seed,
    )
    return {
        "task": manifest["task"],
        "dataset": manifest["dataset"],
        "split": "test",
        "split_unit": manifest["split_unit"],
        "event_kinds": manifest["event_kinds"],
        "test_group_count": len(manifest["split_groups"]["test"]),
        "sample_count": len(manifest["samples"]),
        "event_metrics": event_metrics,
        "video_duration_seconds": duration_by_sample,
        "analysis_seconds_per_video": analysis_seconds,
        "source_video_sha256_by_sample": source_sha256_by_sample,
        "thresholds": {
            "accident": accident_threshold_value,
            "flood_min_area_ratio": flood_area_value,
            "flood_probability": flood_probability_value,
            "min_temporal_iou": float(min_temporal_iou),
        },
        "model_provenance": {
            "detector_file": detector_path.name,
            "detector_sha256": _sha256(detector_path),
            "accident_model": _model_record(accident_model_path),
            "flood_model": _model_record(flood_model_path),
        },
        "limitations": [
            "事件匹配只使用人工标注的事件类型、观察区域和时间区间。",
            "拥堵与人群结果是画面区域事件，不代表未经标定的真实密度或通行速度。",
            "事故模型只提供场景线索；积水模型不推断水深，二者都不能自动封路。",
            "公开数据结果不能替代武汉大学独立航次评测；所有路线影响仍须管理员确认。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--accident-model", type=Path)
    parser.add_argument("--accident-threshold", type=float)
    parser.add_argument("--flood-model", type=Path)
    parser.add_argument("--flood-min-area-ratio", type=float)
    parser.add_argument("--flood-probability-threshold", type=float)
    parser.add_argument("--min-temporal-iou", type=float, default=0.1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("manifest must exist")
    try:
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        started = time.monotonic()
        report = run_campus_event_evaluation(
            manifest, args.dataset_root, args.detector,
            accident_model_path=args.accident_model,
            accident_threshold=args.accident_threshold,
            flood_model_path=args.flood_model,
            flood_min_area_ratio=args.flood_min_area_ratio,
            flood_probability_threshold=args.flood_probability_threshold,
            min_temporal_iou=args.min_temporal_iou,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
        report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
