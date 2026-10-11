#!/usr/bin/env python3
"""Evaluate pedestrian and vehicle detection on manual campus UAV test frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.vision.evaluate_visdrone import _load_model_manifest  # noqa: E402
from vision.campus_detection_evaluation import validate_campus_detection_manifest  # noqa: E402
from vision.evaluation import evaluate_grouped_frame, summarize_detection_counts  # noqa: E402
from vision.specialized_evaluation import cluster_bootstrap_interval  # noqa: E402


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
        raise ValueError("image paths must be safe relative paths")
    base = root.resolve()
    path = (base / normalized).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError("image path resolves outside dataset root") from error
    if not path.is_file():
        raise ValueError(f"image does not exist: {normalized}")
    return path


def _empty_counts() -> dict[str, Counter]:
    return {
        group: Counter({"tp": 0, "fp": 0, "fn": 0, "ignored": 0})
        for group in ("pedestrian", "vehicle")
    }


def _threshold(value, field: str, *, zero_allowed: bool) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if result > 1 or (result < 0 if zero_allowed else result <= 0):
        raise ValueError(f"{field} is outside the allowed range")
    return result


def _region_crop_boxes(sample: dict) -> list[tuple[int, int, int, int]]:
    """Mirror OnnxDetector.detect_regions crop bounds for ROI coverage checks."""
    width, height = sample["width"], sample["height"]
    boxes = []
    for region in sample["observation_regions"]:
        if region["kind"] not in {"vehicle_lane", "pedestrian", "road_surface", "parking"}:
            continue
        points = region["polygon"]
        x1 = max(0, min(width - 1, int(min(point[0] for point in points) * width)))
        y1 = max(0, min(height - 1, int(min(point[1] for point in points) * height)))
        x2 = max(x1 + 1, min(width, int(max(point[0] for point in points) * width + 0.999)))
        y2 = max(y1 + 1, min(height, int(max(point[1] for point in points) * height + 0.999)))
        box = (x1, y1, x2, y2)
        if box not in boxes:
            boxes.append(box)
    return boxes


def _validate_roi_coverage(sample: dict) -> int:
    if not any(region["kind"] in {"vehicle_lane", "pedestrian", "road_surface"}
               for region in sample["observation_regions"]):
        raise ValueError(
            f"sample {sample['sample_id']} requires a vehicle_lane, pedestrian, or road_surface ROI"
        )
    boxes = _region_crop_boxes(sample)
    if not boxes:
        raise ValueError(
            f"sample {sample['sample_id']} requires a vehicle_lane, pedestrian, or road_surface ROI"
        )
    for item in sample["objects"]:
        if item["label"] not in {"pedestrian", "vehicle"}:
            continue
        x1, y1, x2, y2 = item["box"]
        if not any(rx1 <= x1 and ry1 <= y1 and x2 <= rx2 and y2 <= ry2
                   for rx1, ry1, rx2, ry2 in boxes):
            raise ValueError(
                f"sample {sample['sample_id']} positive ground-truth box is not fully covered by an ROI"
            )
    return len(boxes)


def run_campus_detection_evaluation(
    manifest: dict,
    dataset_root: Path,
    model_path: Path,
    *,
    confidence_threshold: float = 0.369,
    iou_threshold: float = 0.5,
    bootstrap_repetitions: int = 1000,
    seed: int = 20261011,
    inference_path: str = "whole_frame",
    detector_factory=None,
    image_loader=None,
) -> dict:
    """Run the production detector and aggregate results by independent flight."""
    manifest = validate_campus_detection_manifest(manifest)
    if inference_path not in {"whole_frame", "road_roi_tiled"}:
        raise ValueError("inference_path must be whole_frame or road_roi_tiled")
    roi_region_count_by_sample = {}
    if inference_path == "road_roi_tiled":
        roi_region_count_by_sample = {
            sample["sample_id"]: _validate_roi_coverage(sample)
            for sample in manifest["samples"]
        }
    confidence_threshold = _threshold(
        confidence_threshold, "confidence_threshold", zero_allowed=True,
    )
    iou_threshold = _threshold(iou_threshold, "iou_threshold", zero_allowed=False)
    if not isinstance(bootstrap_repetitions, int) or bootstrap_repetitions < 100:
        raise ValueError("bootstrap_repetitions must be an integer of at least 100")
    dataset_root = Path(dataset_root).resolve()
    model_path = Path(model_path).resolve()
    if not dataset_root.is_dir() or not model_path.is_file():
        raise ValueError("dataset root and detector model must exist")

    image_paths = {}
    image_hashes = {}
    source_sha256_by_sample = {}
    for sample in manifest["samples"]:
        image_path = _dataset_path(dataset_root, sample["image"])
        digest = _sha256(image_path)
        previous = image_hashes.get(digest)
        if previous is not None:
            raise ValueError(
                f"duplicate image content in samples {previous} and {sample['sample_id']}"
            )
        image_hashes[digest] = sample["sample_id"]
        image_paths[sample["sample_id"]] = image_path
        source_sha256_by_sample[sample["sample_id"]] = digest

    model_manifest, model_manifest_sha256 = _load_model_manifest(model_path)
    if detector_factory is None:
        from vision.engine import OnnxDetector
        detector_factory = OnnxDetector
    if image_loader is None:
        import cv2

        def image_loader(path):
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
    detector = detector_factory(
        str(model_path), confidence_threshold=confidence_threshold,
        model_id=model_manifest.get("model_id"),
        model_version=model_manifest.get("model_revision"),
    )
    if inference_path == "road_roi_tiled" and not callable(
            getattr(detector, "detect_regions", None)):
        raise ValueError("road_roi_tiled evaluation requires a detector with detect_regions")

    totals = _empty_counts()
    flight_counts = {
        group: {flight_id: {"tp": 0, "fp": 0, "fn": 0, "ignored": 0}
                for flight_id in manifest["split_groups"]["test"]}
        for group in ("pedestrian", "vehicle")
    }
    samples = {}
    inference_seconds = []
    for sample in manifest["samples"]:
        image_path = image_paths[sample["sample_id"]]
        frame = image_loader(image_path)
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError(f"image cannot be decoded: {sample['image']}")
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if width != sample["width"] or height != sample["height"]:
            raise ValueError(f"decoded dimensions do not match manifest: {sample['sample_id']}")
        started = time.monotonic()
        predictions = (
            detector.detect_regions(frame, sample["observation_regions"])
            if inference_path == "road_roi_tiled" else detector.detect(frame)
        )
        inference_seconds.append(time.monotonic() - started)
        counts = evaluate_grouped_frame(
            predictions, sample["objects"],
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        for group, values in counts.items():
            totals[group].update(values)
            for key, value in values.items():
                flight_counts[group][sample["group_id"]][key] += value
        samples[sample["sample_id"]] = {
            "group_id": sample["group_id"],
            "counts": counts,
            "ground_truth_count": sum(
                item["label"] in {"pedestrian", "vehicle"} for item in sample["objects"]
            ),
            "prediction_count": len(predictions) if isinstance(predictions, list) else 0,
        }

    summaries = summarize_detection_counts({
        group: dict(values) for group, values in totals.items()
    })

    def metric_for_batches(batches, group: str, metric: str) -> float:
        combined = {
            key: sum(batch[key] for batch in batches)
            for key in ("tp", "fp", "fn", "ignored")
        }
        return summarize_detection_counts({group: combined})[group][metric]

    for group in ("pedestrian", "vehicle"):
        summaries[group]["flight_cluster_bootstrap_95_ci"] = {
            metric: cluster_bootstrap_interval(
                flight_counts[group],
                lambda batches, group=group, metric=metric: metric_for_batches(
                    batches, group, metric,
                ),
                repetitions=bootstrap_repetitions, seed=seed,
            )
            for metric in ("precision", "recall", "f1")
        }

    return {
        "task": manifest["task"],
        "dataset": manifest["dataset"],
        "split": "test",
        "split_unit": manifest["split_unit"],
        "flight_group_count": len(manifest["split_groups"]["test"]),
        "image_count": len(manifest["samples"]),
        "inference_path": inference_path,
        "roi_sample_count": len(roi_region_count_by_sample),
        "roi_region_count_by_sample": roi_region_count_by_sample,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "groups": summaries,
        "samples": samples,
        "source_image_sha256_by_sample": source_sha256_by_sample,
        "model": getattr(detector, "model_id", model_manifest.get("model_id")),
        "model_revision": getattr(detector, "model_version", model_manifest.get("model_revision")),
        "model_sha256": _sha256(model_path),
        "model_manifest_sha256": model_manifest_sha256,
        "mean_inference_seconds_per_image": round(
            sum(inference_seconds) / len(inference_seconds), 4,
        ) if inference_seconds else None,
        "limitations": [
            "车辆细分类合并为路线事件所需的 vehicle 组；本报告不评价车型分类准确率。",
            "框级指标不能替代拥堵、人流或事故的事件级评测。",
            "测试帧必须来自独立校园航次；相邻帧数量不等于独立样本数量。",
        ],
    }


def run_campus_detection_path_comparison(
    manifest: dict,
    dataset_root: Path,
    model_path: Path,
    *,
    confidence_threshold: float = 0.369,
    iou_threshold: float = 0.5,
    bootstrap_repetitions: int = 1000,
    seed: int = 20261011,
    detector_factory=None,
    image_loader=None,
) -> dict:
    """Compare whole-frame and production ROI inference on identical test flights."""
    manifest = validate_campus_detection_manifest(manifest)
    for sample in manifest["samples"]:
        _validate_roi_coverage(sample)
    common = {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "detector_factory": detector_factory,
        "image_loader": image_loader,
    }
    whole_frame = run_campus_detection_evaluation(
        manifest, dataset_root, model_path,
        inference_path="whole_frame", **common,
    )
    road_roi_tiled = run_campus_detection_evaluation(
        manifest, dataset_root, model_path,
        inference_path="road_roi_tiled", **common,
    )

    identical_fields = (
        "task", "dataset", "split", "split_unit", "flight_group_count", "image_count",
        "confidence_threshold", "iou_threshold", "model", "model_revision", "model_sha256",
        "model_manifest_sha256", "source_image_sha256_by_sample",
    )
    if any(whole_frame[field] != road_roi_tiled[field] for field in identical_fields):
        raise ValueError("paired inference reports do not use identical data, model, and thresholds")
    if whole_frame["samples"].keys() != road_roi_tiled["samples"].keys():
        raise ValueError("paired inference reports contain different test samples")
    for sample_id, whole_sample in whole_frame["samples"].items():
        if whole_sample["group_id"] != road_roi_tiled["samples"][sample_id]["group_id"]:
            raise ValueError("paired inference reports assign a sample to different flight groups")

    flight_ids = sorted({
        sample["group_id"] for sample in whole_frame["samples"].values()
    })
    paired_flight_counts = {
        flight_id: {
            path: _empty_counts()
            for path in ("whole_frame", "road_roi_tiled")
        }
        for flight_id in flight_ids
    }
    for path, report in (("whole_frame", whole_frame), ("road_roi_tiled", road_roi_tiled)):
        for sample in report["samples"].values():
            flight_counts_for_path = paired_flight_counts[sample["group_id"]][path]
            for group, counts in sample["counts"].items():
                flight_counts_for_path[group].update(counts)

    def paired_delta_for_batches(batches, group: str, metric: str) -> float | None:
        scores = {}
        for path in ("whole_frame", "road_roi_tiled"):
            combined = {
                key: sum(batch[path][group][key] for batch in batches)
                for key in ("tp", "fp", "fn", "ignored")
            }
            scores[path] = summarize_detection_counts({group: combined})[group][metric]
        if scores["whole_frame"] is None or scores["road_roi_tiled"] is None:
            return None
        return scores["road_roi_tiled"] - scores["whole_frame"]

    differences = {}
    for group in ("pedestrian", "vehicle"):
        differences[group] = {}
        for metric in ("precision", "recall", "f1"):
            whole_score = whole_frame["groups"][group][metric]
            roi_score = road_roi_tiled["groups"][group][metric]
            point_difference = (
                round(roi_score - whole_score, 4)
                if whole_score is not None and roi_score is not None else None
            )
            interval = cluster_bootstrap_interval(
                paired_flight_counts,
                lambda batches, group=group, metric=metric: paired_delta_for_batches(
                    batches, group, metric,
                ),
                repetitions=bootstrap_repetitions,
                seed=seed,
            )
            differences[group][metric] = {
                "difference": point_difference,
                "flight_cluster_bootstrap_95_ci": interval,
            }

    return {
        "task": whole_frame["task"],
        "dataset": whole_frame["dataset"],
        "split": whole_frame["split"],
        "split_unit": whole_frame["split_unit"],
        "flight_group_count": whole_frame["flight_group_count"],
        "image_count": whole_frame["image_count"],
        "inference_path": "paired_comparison",
        "confidence_threshold": whole_frame["confidence_threshold"],
        "iou_threshold": whole_frame["iou_threshold"],
        "model": whole_frame["model"],
        "model_revision": whole_frame["model_revision"],
        "model_sha256": whole_frame["model_sha256"],
        "model_manifest_sha256": whole_frame["model_manifest_sha256"],
        "paired_inputs": {
            "same_source_image_hashes": True,
            "same_model_hash": True,
            "same_thresholds": True,
            "bootstrap_unit": "flight_id",
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
        },
        "source_image_sha256_by_sample": whole_frame["source_image_sha256_by_sample"],
        "whole_frame": whole_frame,
        "road_roi_tiled": road_roi_tiled,
        "road_roi_minus_whole_frame": differences,
        "limitations": [
            "道路 ROI 与全图按同一人工标注目标比较；ROI 外目标须标注为 ignore 或确保样本不含该类目标。",
            "差值区间按独立航次成对重采样；航次数少时区间不稳定，不能替代更多独立航次。",
            "检测框指标不等同于拥堵、人流、事故或积水事件识别效果。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence", type=float, required=True,
                        help="threshold frozen on validation flights")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--inference-path", choices=("whole_frame", "road_roi_tiled", "compare"),
                        default="whole_frame",
                        help="whole-frame, production road-ROI tiled, or paired comparison")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("manifest must exist")
    try:
        manifest_bytes = args.manifest.read_bytes()
        started = time.monotonic()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        runner = (
            run_campus_detection_path_comparison
            if args.inference_path == "compare"
            else run_campus_detection_evaluation
        )
        report = runner(
            manifest, args.dataset_root, args.model,
            confidence_threshold=args.confidence, iou_threshold=args.iou,
            bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed,
            **({} if args.inference_path == "compare"
               else {"inference_path": args.inference_path}),
        )
        report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
