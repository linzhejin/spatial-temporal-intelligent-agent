#!/usr/bin/env python3
"""Evaluate local aerial vision ONNX weights on a held-out manifest.

The evaluator never creates a random image-level split. The manifest must list
disjoint train/validation/test groups and only test-split samples. Whether
those groups represent independent flights/scenes depends on the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from vision.floodnet_training import _require_unambiguous_geometry
from vision.specialized import AIDER_CLASSES, OnnxAiderClassifier, OnnxFloodSegmenter
from vision.specialized_evaluation import (
    cluster_bootstrap_interval,
    summarize_binary_classification,
    summarize_binary_counts,
    summarize_classification,
    validate_split_manifest,
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
        raise ValueError("dataset files must use safe relative paths")
    base = root.resolve()
    path = (base / normalized).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError("dataset file resolves outside dataset root") from error
    if not path.is_file():
        raise ValueError(f"dataset file does not exist: {normalized}")
    return path


def _classification_metric(records, classes):
    return summarize_classification(
        [record["actual"] for record in records],
        [record["predicted"] for record in records],
        classes=classes,
    )


def _evaluate_classification(manifest, root: Path, model_path: Path, threshold: float):
    classifier = OnnxAiderClassifier(str(model_path), threshold=threshold)
    records = []
    elapsed = []
    for sample in manifest["samples"]:
        image_path = _dataset_path(root, sample["image"])
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot decode test image: {sample['image']}")
        started = time.monotonic()
        prediction = classifier.predict(frame)
        elapsed.append(time.monotonic() - started)
        records.append({
            "group_id": sample["group_id"],
            "actual": sample["label"],
            "predicted": prediction["class"],
            "traffic_incident_probability": prediction["traffic_accident_probability"],
        })
    summary = _classification_metric(records, AIDER_CLASSES)
    summary["traffic_incident_binary"] = summarize_binary_classification(
        [record["actual"] == "traffic_incident" for record in records],
        [record["traffic_incident_probability"] >= threshold for record in records],
    )
    summary["traffic_incident_binary"]["threshold"] = threshold
    group_records = {}
    for record in records:
        group_records.setdefault(record["group_id"], []).append(record)
    summary["group_bootstrap_95_ci"] = {
        "traffic_incident_recall": cluster_bootstrap_interval(
            group_records,
            lambda batches: summarize_binary_classification(
                [record["actual"] == "traffic_incident"
                 for batch in batches for record in batch],
                [record["traffic_incident_probability"] >= threshold
                 for batch in batches for record in batch],
            )["recall"],
        ),
        "accuracy": cluster_bootstrap_interval(
            group_records,
            lambda batches: _classification_metric(
                [record for batch in batches for record in batch], AIDER_CLASSES,
            )["accuracy"],
        ),
    }
    summary["mean_inference_seconds_per_image"] = round(sum(elapsed) / len(elapsed), 4)
    return summary


def _polygon_mask(frame_shape, polygon):
    height, width = frame_shape[:2]
    points = np.asarray([
        [round(float(point[0]) * (width - 1)), round(float(point[1]) * (height - 1))]
        for point in polygon
    ], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _align_rgb_to_mask_grid(frame, truth_labels):
    """Reject unequal grids; equal dimensions alone do not establish registration."""
    if frame is None or truth_labels is None or truth_labels.ndim != 2:
        raise ValueError("image must be RGB/BGR and ground truth must be a 2D class mask")
    if frame.shape[:2] != truth_labels.shape:
        raise ValueError("image and mask dimensions differ; verify registration before evaluation")
    return frame, "matching_dimensions"


def _read_flood_pair(image_path: Path, mask_path: Path):
    """Check headers before decoding; no inferred rotation, resize, or crop."""
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        _require_unambiguous_geometry(image, mask)
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    truth_labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if frame is None or truth_labels is None or truth_labels.ndim != 2:
        raise ValueError("cannot decode image and grayscale class-index mask")
    frame, alignment = _align_rgb_to_mask_grid(frame, truth_labels)
    return frame, truth_labels, alignment


def _flood_valid_mask(labels, *, roi=None):
    """Use only documented FloodNet classes 0..9 in the requested ROI."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("label mask must be a 2D integer grid")
    region = np.ones(labels.shape, dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
    if region.shape != labels.shape:
        raise ValueError("ROI must share the image/mask grid")
    valid = region & (labels >= 0) & (labels <= 9)
    if not valid.any():
        raise ValueError("road ROI contains no valid annotated pixels")
    return valid


def _flood_pixel_counts(prediction, labels, *, roi=None, positive_class_id=3):
    """Score annotated FloodNet classes only, matching calibration's 0..9 mask."""
    prediction, labels = np.asarray(prediction), np.asarray(labels)
    valid = _flood_valid_mask(labels, roi=roi)
    if prediction.shape != labels.shape:
        raise ValueError("prediction and label mask must share a 2D grid")
    valid_count = int(valid.sum())
    predicted = prediction.astype(bool) & valid
    actual = (labels == positive_class_id) & valid
    return (int(np.logical_and(predicted, actual).sum()),
            int(np.logical_and(predicted, ~actual & valid).sum()),
            int(np.logical_and(~predicted, actual).sum()), valid_count)


def _evaluate_flood(manifest, root: Path, model_path: Path, min_area_ratio: float,
                    flood_probability_threshold: float | None = None):
    segmenter = OnnxFloodSegmenter(
        str(model_path), min_area_ratio=min_area_ratio,
        flood_probability_threshold=flood_probability_threshold,
    )
    tp = fp = fn = 0
    valid_pixel_count = ignored_pixel_count = 0
    group_counts = {}
    sample_iou = []
    alignment_counts = {"matching_dimensions": 0, "image_resized_to_mask_grid": 0}
    for sample in manifest["samples"]:
        image_path = _dataset_path(root, sample["image"])
        mask_path = _dataset_path(root, sample["mask"])
        frame, truth_labels, alignment = _read_flood_pair(image_path, mask_path)
        alignment_counts[alignment] += 1
        region = {
            "id": sample["group_id"], "kind": "road_surface",
            "polygon": sample["road_surface_polygon"],
        }
        prediction = segmenter.predict(frame, region, include_mask=True)
        if prediction is None:
            raise ValueError(f"road ROI is empty: {sample['image']}")
        roi = _polygon_mask(frame.shape, sample["road_surface_polygon"])
        sample_tp, sample_fp, sample_fn, sample_valid = _flood_pixel_counts(
            prediction["flooded_road_mask"], truth_labels, roi=roi,
            positive_class_id=int(sample.get("flooded_road_class_id", 3)),
        )
        valid_pixel_count += sample_valid
        ignored_pixel_count += int(roi.sum()) - sample_valid
        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
        group = group_counts.setdefault(sample["group_id"], [0, 0, 0])
        group[0] += sample_tp
        group[1] += sample_fp
        group[2] += sample_fn
        per_sample = summarize_binary_counts(tp=sample_tp, fp=sample_fp, fn=sample_fn)
        sample_iou.append(per_sample["iou"])
    overall = summarize_binary_counts(tp=tp, fp=fp, fn=fn)

    def bootstrap_iou(batches):
        counts = [sum(batch[index] for batch in batches) for index in range(3)]
        return summarize_binary_counts(tp=counts[0], fp=counts[1], fn=counts[2])["iou"]

    return {
        **overall,
        "flood_probability_threshold": segmenter.flood_probability_threshold,
        "sample_count": len(manifest["samples"]),
        "image_mask_grid_alignment": alignment_counts,
        "pixel_alignment_verified": False,
        "valid_pixel_count": valid_pixel_count,
        "ignored_pixel_count": ignored_pixel_count,
        "mean_per_image_iou": round(
            sum(value for value in sample_iou if value is not None)
            / max(1, sum(value is not None for value in sample_iou)), 6,
        ) if any(value is not None for value in sample_iou) else None,
        "group_bootstrap_95_ci_iou": cluster_bootstrap_interval(group_counts, bootstrap_iou),
        "interpretation": "像素指标仅在标注的路面观察区域计算；不代表水深，也不等同于道路封闭准确率。",
    }


def _evaluate_visible_water_road(manifest, root: Path, model_path: Path, *,
                                 probability_threshold: float):
    """Evaluate a binary visible-water model on manually labeled road pixels.

    Unlike the FloodNet evaluator, this contract uses masks encoded as 0/255,
    requires exact image/mask alignment, and reports false alarms on negative
    frames and whole flights separately from pixel overlap.
    """
    if (isinstance(probability_threshold, bool)
            or not isinstance(probability_threshold, (int, float))
            or not np.isfinite(float(probability_threshold))
            or not 0 <= probability_threshold <= 1):
        raise ValueError("visible-water probability threshold must be between 0 and 1")
    segmenter = OnnxFloodSegmenter(
        str(model_path), flood_probability_threshold=float(probability_threshold),
    )
    if not segmenter.is_binary_visible_water:
        raise ValueError("campus visible-water evaluation requires a two-class model")

    total = [0, 0, 0]
    group_counts: dict[str, list[int]] = {}
    group_sample_states: dict[str, list[dict]] = {}
    per_sample = []
    negative_sample_ratios = []
    negative_sample_false_alarms = 0
    for sample in manifest["samples"]:
        image_path = _dataset_path(root, sample["image"])
        mask_path = _dataset_path(root, sample["mask"])
        frame, truth_mask, _ = _read_flood_pair(image_path, mask_path)
        if not np.isin(np.unique(truth_mask), (0, 255)).all():
            raise ValueError(f"ground truth mask must contain only 0 and 255: {sample['mask']}")
        roi = _polygon_mask(frame.shape, sample["road_surface_polygon"])
        roi_pixels = int(roi.sum())
        if roi_pixels == 0:
            raise ValueError(f"road ROI is empty: {sample['image']}")
        prediction = segmenter.predict(
            frame,
            {"id": sample["group_id"], "kind": "road_surface",
             "polygon": sample["road_surface_polygon"]},
            include_mask=True,
        )
        if prediction is None or "flooded_road_mask" not in prediction:
            raise ValueError(f"visible-water model did not return a mask: {sample['image']}")
        predicted_mask = np.asarray(prediction["flooded_road_mask"])
        if predicted_mask.shape != roi.shape:
            raise ValueError(f"predicted mask dimensions do not match the image: {sample['image']}")
        predicted = predicted_mask.astype(bool) & roi
        actual = (truth_mask == 255) & roi
        counts = [
            int(np.logical_and(predicted, actual).sum()),
            int(np.logical_and(predicted, ~actual & roi).sum()),
            int(np.logical_and(~predicted, actual).sum()),
        ]
        for index, value in enumerate(counts):
            total[index] += value
        group_id = sample["group_id"]
        group = group_counts.setdefault(group_id, [0, 0, 0])
        for index, value in enumerate(counts):
            group[index] += value

        predicted_ratio = float(predicted.sum()) / roi_pixels
        actual_ratio = float(actual.sum()) / roi_pixels
        is_negative = not bool(actual.any())
        false_alarm = is_negative and bool(predicted.any())
        if is_negative:
            negative_sample_ratios.append(predicted_ratio)
            negative_sample_false_alarms += int(false_alarm)
        state = {"has_actual_water": bool(actual.any()), "has_predicted_water": bool(predicted.any())}
        group_sample_states.setdefault(group_id, []).append(state)
        per_sample.append({
            "sample_id": sample.get("sample_id", Path(sample["image"]).stem),
            "group_id": group_id,
            **summarize_binary_counts(tp=counts[0], fp=counts[1], fn=counts[2]),
            "road_roi_pixel_count": roi_pixels,
            "actual_road_water_area_ratio": round(actual_ratio, 6),
            "predicted_road_water_area_ratio": round(predicted_ratio, 6),
            "negative_sample": is_negative,
            "false_alarm": bool(false_alarm),
            "image_sha256": _sha256(image_path),
            "mask_sha256": _sha256(mask_path),
        })

    def _iou_for_groups(groups):
        counts = [sum(batch[index] for batch in groups) for index in range(3)]
        return summarize_binary_counts(tp=counts[0], fp=counts[1], fn=counts[2])["iou"]

    per_group = {}
    negative_flights = []
    for group_id, counts in group_counts.items():
        states = group_sample_states[group_id]
        group_is_negative = not any(item["has_actual_water"] for item in states)
        predicted_water = any(item["has_predicted_water"] for item in states)
        per_group[group_id] = {
            **summarize_binary_counts(tp=counts[0], fp=counts[1], fn=counts[2]),
            "sample_count": len(states),
            "negative_flight": group_is_negative,
            "false_alarm": bool(group_is_negative and predicted_water),
        }
        if group_is_negative:
            negative_flights.append(bool(predicted_water))

    negative_count = len(negative_sample_ratios)
    return {
        **summarize_binary_counts(tp=total[0], fp=total[1], fn=total[2]),
        "pixel_alignment_verified": False,
        "model_class_count": 2,
        "visible_water_probability_threshold": float(probability_threshold),
        "sample_count": len(per_sample),
        "flight_count": len(group_counts),
        "negative_sample_count": negative_count,
        "negative_sample_false_positive_rate": (
            negative_sample_false_alarms / negative_count if negative_count else None
        ),
        "mean_negative_road_false_positive_area_ratio": (
            round(sum(negative_sample_ratios) / negative_count, 6) if negative_count else None
        ),
        "negative_flight_count": len(negative_flights),
        "negative_flight_false_alarm_rate": (
            sum(negative_flights) / len(negative_flights) if negative_flights else None
        ),
        "mean_per_sample_iou": round(
            sum(item["iou"] for item in per_sample if item["iou"] is not None)
            / max(1, sum(item["iou"] is not None for item in per_sample)), 6,
        ) if any(item["iou"] is not None for item in per_sample) else None,
        "group_bootstrap_95_ci_iou": cluster_bootstrap_interval(group_counts, _iou_for_groups),
        "interpretation": (
            "像素指标仅在人工标注的道路观察区域内计算；负样本误报率按图像与航次分别报告；"
            "不代表水深、道路通行状态或自动封路准确率。"
        ),
        "per_group": per_group,
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(
        "accident_classification", "flooded_road_segmentation",
        "visible_water_road_segmentation",
    ), required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="UTF-8 JSON held-out manifest with disjoint train/validation/test groups")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="local ONNX weights; never copied into Git")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="AIDER traffic_incident probability threshold")
    parser.add_argument("--flood-min-area", type=float, default=0.08,
                        help="Flooded-road minimum image-region fraction")
    parser.add_argument("--flood-class-threshold", type=float,
                        help="optional flooded-road class probability threshold")
    parser.add_argument("--visible-water-class-threshold", type=float, default=0.5,
                        help="fixed binary visible-water probability threshold; choose on validation flights")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if not args.manifest.is_file() or not args.dataset_root.is_dir() or not args.model.is_file():
        parser.error("manifest, dataset root, and model must exist")
    if (not 0 < args.threshold <= 1 or not 0 < args.flood_min_area <= 1
            or (args.flood_class_threshold is not None
                and not 0 <= args.flood_class_threshold <= 1)
            or not 0 <= args.visible_water_class_threshold <= 1):
        parser.error("thresholds are outside their allowed ranges")
    try:
        manifest_bytes = args.manifest.read_bytes()
        manifest = validate_split_manifest(
            json.loads(manifest_bytes.decode("utf-8-sig")), task=args.task,
        )
        model_path = args.model.resolve()
        started = time.monotonic()
        if args.task == "accident_classification":
            metrics = _evaluate_classification(
                manifest, args.dataset_root, model_path, args.threshold,
            )
        elif args.task == "flooded_road_segmentation":
            metrics = _evaluate_flood(
                manifest, args.dataset_root, model_path, args.flood_min_area,
                args.flood_class_threshold,
            )
        else:
            metrics = _evaluate_visible_water_road(
                manifest, args.dataset_root, model_path,
                probability_threshold=args.visible_water_class_threshold,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    report = {
        "task": args.task,
        "dataset": manifest["dataset"],
        "split": "test",
        "split_unit": manifest["split_unit"],
        "test_group_count": len(manifest["split_groups"]["test"]),
        "sample_count": len(manifest["samples"]),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_sha256": _sha256(model_path),
        "model_file": model_path.name,
        "metrics": metrics,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "此脚本评测本地提供的权重，不训练模型。",
            "bootstrap 按清单声明的分组重采样；若分组单位是图像 ID，结果不代表独立航次或事故事件。",
            "正式验收应按航次、事故或独立场景划分，不能将相邻视频帧随机分入不同集合。",
            "公开数据结果不能替代武汉大学航拍样本的外域验证。",
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
