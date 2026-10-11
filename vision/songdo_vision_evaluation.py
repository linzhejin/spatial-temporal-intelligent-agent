"""Evaluate the route detector on the annotated Songdo Vision COCO test ZIP."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable


SONGDO_CLASS_NAMES = {"car", "bus", "truck", "motorcycle"}
PREDICTION_TO_SONGDO_CLASS = {
    "car": "car",
    "van": "car",  # Songdo Vision groups vans and light-duty vehicles with cars.
    "bus": "bus",
    "truck": "truck",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
}
ANNOTATION_MEMBER = "test/coco_annotations.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _safe_filename(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("COCO image has an empty filename")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe image path")
    return path.as_posix()


def _validate_coco(coco: object, archive_names: set[str]) -> tuple[dict, str]:
    if not isinstance(coco, dict):
        raise ValueError("COCO annotation root must be an object")
    categories_raw = coco.get("categories")
    images_raw = coco.get("images")
    annotations_raw = coco.get("annotations")
    if not all(isinstance(value, list) for value in
               (categories_raw, images_raw, annotations_raw)):
        raise ValueError("COCO categories, images, and annotations must be arrays")

    category_by_id: dict[int, str] = {}
    for category in categories_raw:
        if not isinstance(category, dict):
            raise ValueError("invalid COCO category record")
        category_id, name = category.get("id"), category.get("name")
        if type(category_id) is not int or not isinstance(name, str):
            raise ValueError("invalid COCO category id or name")
        normalized_name = name.casefold()
        if category_id in category_by_id or normalized_name in category_by_id.values():
            raise ValueError("duplicate COCO category id or name")
        category_by_id[category_id] = normalized_name
    if set(category_by_id.values()) != SONGDO_CLASS_NAMES:
        raise ValueError("COCO categories must be car, bus, truck, and motorcycle")

    images: dict[int, dict] = {}
    member_by_image_id: dict[int, str] = {}
    filenames: set[str] = set()
    for image in images_raw:
        if not isinstance(image, dict):
            raise ValueError("invalid COCO image record")
        image_id = image.get("id")
        filename = _safe_filename(image.get("file_name"))
        width, height = image.get("width"), image.get("height")
        if type(image_id) is not int or image_id in images:
            raise ValueError("duplicate image id or filename")
        if (type(width) is not int or type(height) is not int
                or width <= 0 or height <= 0):
            raise ValueError("invalid COCO image dimensions")
        if filename in filenames:
            raise ValueError("duplicate image id or filename")
        filenames.add(filename)
        member_name = f"test/images/{filename}"
        images[image_id] = {"width": width, "height": height, "file_name": filename}
        member_by_image_id[image_id] = member_name
    if not images:
        raise ValueError("COCO test split contains no images")
    missing = [name for name in member_by_image_id.values() if name not in archive_names]
    if missing:
        raise ValueError(f"archive is missing {len(missing)} image members")

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for annotation in annotations_raw:
        if not isinstance(annotation, dict):
            raise ValueError("invalid COCO annotation record")
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if (type(annotation_id) is not int or annotation_id in annotation_ids
                or type(image_id) is not int or image_id not in images
                or type(category_id) is not int or category_id not in category_by_id):
            raise ValueError("invalid or duplicate COCO annotation id/reference")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(
                _finite_number(value) for value in bbox):
            raise ValueError("invalid bbox")
        x, y, width, height = map(float, bbox)
        image_info = images[image_id]
        if (x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > image_info["width"] + 1e-3
                or y + height > image_info["height"] + 1e-3):
            raise ValueError("invalid bbox")
        if annotation.get("iscrowd", 0) not in (0, False):
            raise ValueError("crowd annotations are not supported by this evaluator")
        annotation_ids.add(annotation_id)
        annotations_by_image[image_id].append({
            "class": category_by_id[category_id],
            "box": [x, y, x + width, y + height],
        })
    return {
        "info": coco.get("info") if isinstance(coco.get("info"), dict) else {},
        "licenses": coco.get("licenses") if isinstance(coco.get("licenses"), list) else [],
        "images": images,
        "member_by_image_id": member_by_image_id,
        "annotations_by_image": annotations_by_image,
        "annotation_count": len(annotation_ids),
    }, hashlib.sha256(json.dumps(coco, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _score_image(predictions: object, annotations: list[dict], iou_threshold: float) -> tuple[dict, int]:
    if not isinstance(predictions, list):
        raise ValueError("detector predictions must be a list")
    ground_truth = {name: [] for name in SONGDO_CLASS_NAMES}
    for annotation in annotations:
        ground_truth[annotation["class"]].append(annotation["box"])
    candidates = {name: [] for name in SONGDO_CLASS_NAMES}
    excluded = 0
    for prediction in predictions:
        if not isinstance(prediction, dict):
            excluded += 1
            continue
        class_name = PREDICTION_TO_SONGDO_CLASS.get(prediction.get("label"))
        confidence = prediction.get("confidence")
        box = prediction.get("box")
        if class_name is None:
            excluded += 1
            continue
        if (not _finite_number(confidence) or not isinstance(box, (list, tuple))
                or len(box) != 4 or not all(_finite_number(value) for value in box)):
            excluded += 1
            continue
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            excluded += 1
            continue
        candidates[class_name].append({"confidence": float(confidence), "box": [x1, y1, x2, y2]})

    counts = {}
    for class_name in sorted(SONGDO_CLASS_NAMES):
        unmatched = set(range(len(ground_truth[class_name])))
        tp = fp = 0
        for prediction in sorted(candidates[class_name], key=lambda row: row["confidence"], reverse=True):
            best_index = None
            best_overlap = 0.0
            for index in unmatched:
                overlap = _iou(prediction["box"], ground_truth[class_name][index])
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index is not None and best_overlap >= iou_threshold:
                unmatched.remove(best_index)
                tp += 1
            else:
                fp += 1
        counts[class_name] = {"tp": tp, "fp": fp, "fn": len(unmatched)}
    return counts, excluded


def _summarize(counts: dict[str, dict[str, int]]) -> dict:
    total = Counter({"tp": 0, "fp": 0, "fn": 0})
    by_class = {}
    for name in sorted(SONGDO_CLASS_NAMES):
        row = counts[name]
        total.update(row)
        precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class[name] = {
            **row,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    micro = {
        **dict(total),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    return {"by_class": by_class, "micro": micro}


def _decode_image(encoded: bytes):
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)


def run_songdo_vision_evaluation(
    archive_path: Path,
    model_path: Path,
    *,
    confidence_threshold: float = 0.369,
    iou_threshold: float = 0.5,
    limit: int = 0,
    sample_seed: int | None = None,
    inference_path: str = "whole_frame",
    detector_factory=None,
    image_decoder: Callable[[bytes], object] | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict:
    """Score a full or explicitly limited Songdo Vision COCO test ZIP in place."""
    if (not _finite_number(confidence_threshold) or not 0 <= confidence_threshold <= 1
            or not _finite_number(iou_threshold) or not 0 < iou_threshold <= 1):
        raise ValueError("confidence and IoU thresholds must be within their valid ranges")
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if sample_seed is not None and type(sample_seed) is not int:
        raise ValueError("sample_seed must be an integer or None")
    if inference_path not in {"whole_frame", "full_frame_tiled"}:
        raise ValueError("inference_path must be whole_frame or full_frame_tiled")
    archive_path = Path(archive_path).resolve()
    model_path = Path(model_path).resolve()
    if not archive_path.is_file() or not model_path.is_file():
        raise ValueError("dataset ZIP and model file must exist")
    if detector_factory is None:
        from vision.engine import OnnxDetector
        detector_factory = OnnxDetector
    if image_decoder is None:
        image_decoder = _decode_image

    started = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
        if ANNOTATION_MEMBER not in archive_names:
            raise ValueError(f"archive does not contain {ANNOTATION_MEMBER}")
        raw_annotations = archive.read(ANNOTATION_MEMBER)
        try:
            coco = json.loads(raw_annotations.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("COCO annotation JSON is invalid") from error
        dataset, annotations_sha256 = _validate_coco(coco, archive_names)
        image_ids = sorted(dataset["images"])
        if limit and sample_seed is not None:
            selected_ids = sorted(random.Random(sample_seed).sample(
                image_ids, min(limit, len(image_ids)),
            ))
        else:
            selected_ids = image_ids[:limit] if limit else image_ids
        detector = detector_factory(
            str(model_path), confidence_threshold=float(confidence_threshold),
        )
        if inference_path == "full_frame_tiled" and not callable(
                getattr(detector, "detect_regions", None)):
            raise ValueError("full_frame_tiled evaluation requires detector.detect_regions")
        totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in SONGDO_CLASS_NAMES}
        excluded_prediction_count = 0
        inference_seconds = []
        for index, image_id in enumerate(selected_ids, start=1):
            image_info = dataset["images"][image_id]
            member_name = dataset["member_by_image_id"][image_id]
            frame = image_decoder(archive.read(member_name))
            if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
                raise ValueError(f"image cannot be decoded: {member_name}")
            if (int(frame.shape[1]) != image_info["width"]
                    or int(frame.shape[0]) != image_info["height"]):
                raise ValueError(f"decoded dimensions do not match COCO record: {member_name}")
            inference_started = time.monotonic()
            if inference_path == "full_frame_tiled":
                predictions = detector.detect_regions(frame, [{
                    "id": "songdo-full-frame",
                    # This rectangle measures tiling only; it is not a road ROI.
                    "kind": "vehicle_lane",
                    "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                }])
            else:
                predictions = detector.detect(frame)
            inference_seconds.append(time.monotonic() - inference_started)
            frame_counts, excluded = _score_image(
                predictions,
                dataset["annotations_by_image"].get(image_id, []),
                float(iou_threshold),
            )
            excluded_prediction_count += excluded
            for class_name in SONGDO_CLASS_NAMES:
                for metric in ("tp", "fp", "fn"):
                    totals[class_name][metric] += frame_counts[class_name][metric]
            if progress_callback and (index % 50 == 0 or index == len(selected_ids)):
                progress_callback(index, len(selected_ids), time.monotonic() - started)

        metrics = _summarize(totals)
        full_test_split = len(selected_ids) == len(image_ids)
        report = {
            "dataset": "Songdo Vision v2 test",
            "dataset_version": dataset["info"].get("version"),
            "inference_path": inference_path,
            "region_scope": (
                "entire_image_for_tiling_comparison"
                if inference_path == "full_frame_tiled" else None
            ),
            "dataset_archive": archive_path.name,
            "dataset_archive_sha256": _sha256(archive_path),
            "annotations_sha256": annotations_sha256,
            "image_count_in_archive": len(image_ids),
            "image_count_evaluated": len(selected_ids),
            "evaluated_image_ids": selected_ids,
            "sample_seed": sample_seed,
            "annotation_count_in_archive": dataset["annotation_count"],
            "official_test_split_complete": full_test_split,
            "evaluation_scope": "full_official_test_split" if full_test_split else "limited_smoke_sample",
            "model": getattr(detector, "model_id", "unspecified"),
            "model_revision": getattr(detector, "model_version", "unspecified"),
            "model_sha256": _sha256(model_path),
            "execution_provider": "CPUExecutionProvider",
            "confidence_threshold": float(confidence_threshold),
            "iou_threshold": float(iou_threshold),
            "excluded_prediction_count": excluded_prediction_count,
            **metrics,
            "mean_inference_seconds_per_image": round(
                sum(inference_seconds) / len(inference_seconds), 4,
            ) if inference_seconds else None,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "dataset_license": "CC BY 4.0; cite Fonod et al. (2025), DOI 10.1016/j.trc.2025.105205",
            "limitations": [
                "Songdo, South Korea is an external domain and does not measure Wuhan University campus accuracy.",
                "The official test split is random by image, not an independent flight or scene split.",
                "Songdo Vision v2 used the geo-trax detector family to surface some test annotations for human review; related detectors may receive a small favorable bias.",
                "Metrics are class-aware precision/recall/F1 at one confidence and IoU operating point, not COCO mAP or an event-level congestion score.",
                (
                    "Full-image tiling is a resolution-preservation proxy only; it does not measure road-only ROI selection."
                    if inference_path == "full_frame_tiled"
                    else "Inference uses the whole-frame detector path and does not measure the production tiled path."
                ),
            ],
        }
    return report


def run_songdo_vision_paired_comparison(
    archive_path: Path,
    model_path: Path,
    *,
    confidence_threshold: float = 0.369,
    iou_threshold: float = 0.5,
    limit: int = 0,
    sample_seed: int | None = None,
    detector_factory=None,
    image_decoder: Callable[[bytes], object] | None = None,
    progress_callback: Callable[[str, int, int, float], None] | None = None,
) -> dict:
    """Compare whole-image resizing and tiled inference on identical COCO frames."""
    started = time.monotonic()
    common = {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "limit": limit,
        "sample_seed": sample_seed,
        "detector_factory": detector_factory,
        "image_decoder": image_decoder,
    }

    def path_progress(path_name: str):
        if progress_callback is None:
            return None

        def callback(current: int, total: int, elapsed: float) -> None:
            progress_callback(path_name, current, total, elapsed)

        return callback

    whole_frame = run_songdo_vision_evaluation(
        archive_path, model_path, inference_path="whole_frame",
        progress_callback=path_progress("whole_frame"), **common,
    )
    full_frame_tiled = run_songdo_vision_evaluation(
        archive_path, model_path, inference_path="full_frame_tiled",
        progress_callback=path_progress("full_frame_tiled"), **common,
    )
    invariant_fields = (
        "dataset_archive_sha256", "annotations_sha256", "model_sha256",
        "evaluated_image_ids", "confidence_threshold", "iou_threshold",
    )
    if any(whole_frame[field] != full_frame_tiled[field] for field in invariant_fields):
        raise ValueError("paired comparison inputs changed or selected different images")

    deltas = {}

    def metric_delta(tiled_value, whole_value, metric: str):
        difference = tiled_value - whole_value
        return difference if metric in {"tp", "fp", "fn"} else round(difference, 4)

    for group in ("by_class", "micro"):
        whole_metrics, tiled_metrics = whole_frame[group], full_frame_tiled[group]
        metrics = ("tp", "fp", "fn", "precision", "recall", "f1")
        if group == "by_class":
            deltas[group] = {
                class_name: {
                    metric: metric_delta(
                        tiled_metrics[class_name][metric],
                        whole_metrics[class_name][metric],
                        metric,
                    )
                    for metric in metrics
                }
                for class_name in sorted(SONGDO_CLASS_NAMES)
            }
        else:
            deltas[group] = {
                metric: metric_delta(tiled_metrics[metric], whole_metrics[metric], metric)
                for metric in metrics
            }

    return {
        "dataset": whole_frame["dataset"],
        "dataset_archive_sha256": whole_frame["dataset_archive_sha256"],
        "annotations_sha256": whole_frame["annotations_sha256"],
        "model_sha256": whole_frame["model_sha256"],
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "image_count_evaluated": whole_frame["image_count_evaluated"],
        "evaluated_image_ids": whole_frame["evaluated_image_ids"],
        "sample_seed": sample_seed,
        "comparison": "full_frame_tiled_minus_whole_frame",
        "paths": {"whole_frame": whole_frame, "full_frame_tiled": full_frame_tiled},
        "deltas": deltas,
        "paired_elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "Full-image tiling tests resolution preservation, not hand-labeled road ROI selection.",
            "Songdo Vision test images are not independent flight or scene groups; no flight-level confidence interval is reported.",
            "A limited random image sample is exploratory and cannot replace the complete external test evaluation.",
            "Sequential path timings are affected by cache and system load and are not a stable throughput benchmark.",
            "Songdo results do not measure Wuhan University campus performance.",
        ],
    }
