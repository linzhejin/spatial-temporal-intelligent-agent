"""CPU-only aerial image/video inference and bounded, review-only analysis."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import config
from vision.analysis import (
    VEHICLE_LABELS, analyze_observations, build_specialized_candidates,
    project_regions_for_frame,
)


VISDRONE_CLASSES = (
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor", "others",
)
logger = logging.getLogger(__name__)
MAX_PREVIEW_DETECTIONS = 250
# Bump this when detector/tracker/segment aggregation semantics change.
VIDEO_CHECKPOINT_ANALYSIS_REVISION = 3
VIDEO_TEMPORAL_CONTEXT_SECONDS = 15.0


def _checkpoint_model_identity(model) -> dict | None:
    if model is None:
        return None
    identity = {
        name: getattr(model, name, None)
        for name in (
            "model_id", "model_version", "confidence_threshold",
            "tracking_low_confidence_threshold", "threshold", "min_area_ratio",
            "flood_probability_threshold", "class_count",
        )
    }
    model_path = getattr(model, "model_path", None)
    if model_path:
        identity["model_path"] = str(model_path)
        try:
            stat = Path(model_path).stat()
        except (OSError, TypeError, ValueError):
            pass
        else:
            identity["model_file"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return identity


def _video_checkpoint_fingerprint(*, path: Path, width: int, height: int,
                                  fps: float, total_frames: int | None,
                                  target_fps: float, stride: int, segment_limit: int,
                                  dense_trigger_threshold: float, dense_window_seconds: float,
                                  camera_stabilized: bool, regions: list[dict] | None,
                                  anchor_gcj: dict | None, detector, accident_model,
                                  flood_segmenter) -> str:
    source_stat = path.stat()
    contract = {
        "analysis_revision": VIDEO_CHECKPOINT_ANALYSIS_REVISION,
        "source": {"size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
        "video": {"width": width, "height": height, "fps": round(fps, 6),
                  "total_frames": total_frames},
        "sampling": {"target_fps": target_fps, "stride": stride,
                     "segment_limit": segment_limit},
        "accident_dense_sampling": {"trigger_threshold": dense_trigger_threshold,
                                    "window_seconds": dense_window_seconds},
        "camera_stabilized": bool(camera_stabilized),
        "regions": regions or [],
        "anchor_gcj": anchor_gcj,
        "models": {
            "detector": _checkpoint_model_identity(detector),
            "accident": _checkpoint_model_identity(accident_model),
            "flood": _checkpoint_model_identity(flood_segmenter),
        },
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class VisionConfigurationError(RuntimeError):
    """The optional vision runtime or its model is not usable."""


class VisionAnalysisCancelled(RuntimeError):
    """A video analysis was cancelled between sampled frames."""


def _validate_frame_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("影像分辨率无效")
    if width * height > config.VISION_MAX_FRAME_PIXELS:
        raise ValueError(
            f"影像分辨率超过 {config.VISION_MAX_FRAME_PIXELS:,} 像素上限，请先裁剪或降低分辨率。"
        )


class OnnxDetector:
    """Loads a local, fixed-shape RT-DETRv4 ONNX model using CPU execution only."""

    def __init__(self, model_path: str, *, session=None,
                 confidence_threshold: float | None = None,
                 tracking_low_confidence_threshold: float | None = None,
                 model_id: str | None = None, model_version: str | None = None):
        if session is None and (not model_path or not Path(model_path).is_file()):
            raise VisionConfigurationError("VISION_MODEL_PATH 未配置或 ONNX 权重文件不存在")
        cpu_threads = config.resolve_vision_cpu_threads() if session is None else None
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise VisionConfigurationError(
                    "缺少 onnxruntime，请在独立视觉工作进程中安装 requirements-vision.txt"
                ) from error
            options = ort.SessionOptions()
            options.intra_op_num_threads = cpu_threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            try:
                session = ort.InferenceSession(
                    str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
                )
            except Exception as error:
                raise VisionConfigurationError(f"ONNX 模型加载失败：{error}") from error

        self.session = session
        self.cpu_threads = cpu_threads
        self.model_path = str(model_path or "injected-session")
        self.model_id = model_id or os.getenv("VISION_MODEL_ID", "visdrone-rtdetrv4-s")
        self.model_version = model_version or os.getenv("VISION_MODEL_REVISION", "unspecified")
        self.confidence_threshold = (
            float(confidence_threshold) if confidence_threshold is not None
            else float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.369"))
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise VisionConfigurationError("VISION_CONFIDENCE_THRESHOLD 必须介于 0 和 1 之间")
        self.tracking_low_confidence_threshold = (
            float(tracking_low_confidence_threshold)
            if tracking_low_confidence_threshold is not None
            else float(getattr(config, "VISION_TRACKING_MIN_CONFIDENCE", os.getenv(
                "VISION_TRACKING_MIN_CONFIDENCE", "0.1",
            )))
        )
        if not 0.0 <= self.tracking_low_confidence_threshold <= self.confidence_threshold:
            raise VisionConfigurationError(
                "VISION_TRACKING_MIN_CONFIDENCE 必须介于 0 和 VISION_CONFIDENCE_THRESHOLD 之间"
            )

        try:
            inputs = {item.name: item for item in session.get_inputs()}
            outputs = {item.name for item in session.get_outputs()}
            image_shape = inputs["images"].shape
            inputs["orig_target_sizes"]
        except (AttributeError, KeyError, TypeError) as error:
            raise VisionConfigurationError("ONNX 模型必须提供 images 与 orig_target_sizes 输入") from error
        if (
            len(image_shape) != 4 or image_shape[1] != 3
            or not isinstance(image_shape[2], int) or not isinstance(image_shape[3], int)
            or image_shape[2] <= 0 or image_shape[2] != image_shape[3]
        ):
            raise VisionConfigurationError("ONNX 模型必须使用静态方形输入，例如 [N, 3, 960, 960]")
        if not {"labels", "boxes", "scores"}.issubset(outputs):
            raise VisionConfigurationError("ONNX 模型输出必须包含 labels、boxes 和 scores")
        self.input_size = image_shape[2]
        batch_dimension = image_shape[0]
        if isinstance(batch_dimension, int):
            if batch_dimension != 1:
                raise VisionConfigurationError("ONNX 模型批次必须为 1 或动态维度")
            self.tile_batch_size = 1
        else:
            try:
                requested_batch_size = int(os.getenv("VISION_TILE_BATCH_SIZE", "1"))
            except ValueError as error:
                raise VisionConfigurationError("VISION_TILE_BATCH_SIZE 必须是 1 到 4 之间的整数") from error
            if not 1 <= requested_batch_size <= 4:
                raise VisionConfigurationError("VISION_TILE_BATCH_SIZE 必须是 1 到 4 之间的整数")
            self.tile_batch_size = requested_batch_size

    @staticmethod
    def _normalize_batch_output(value, expected_rank: int, batch_size: int):
        import numpy as np

        array = np.asarray(value)
        if array.ndim == expected_rank and batch_size == 1:
            array = array[None, ...]
        if array.ndim != expected_rank + 1 or array.shape[0] != batch_size:
            raise VisionConfigurationError("ONNX 模型输出维度无效")
        return array

    def detect(self, frame, *, tracking: bool = False) -> list[dict[str, Any]]:
        """Return original-frame pixel boxes and normalized VisDrone labels."""
        return self._detect_batch([frame], tracking=tracking)[0]

    def _detect_batch(self, frames, *, tracking: bool = False) -> list[list[dict[str, Any]]]:
        """Infer a small image batch and restore each result to its source frame."""
        import cv2
        import numpy as np

        if not frames:
            return []
        if len(frames) > self.tile_batch_size:
            raise ValueError("单次 ONNX 推理批次超过当前模型支持的大小")

        tensors = []
        transforms = []
        for frame in frames:
            if (frame is None or not hasattr(frame, "shape")
                    or len(frame.shape) != 3 or frame.shape[2] != 3):
                raise ValueError("待识别画面必须是三通道彩色图像")
            height, width = frame.shape[:2]
            _validate_frame_dimensions(width, height)
            ratio = min(self.input_size / width, self.input_size / height)
            resized_width = max(1, round(width * ratio))
            resized_height = max(1, round(height * ratio))
            resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pad_x = (self.input_size - resized_width) // 2
            pad_y = (self.input_size - resized_height) // 2
            padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            padded[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = rgb
            tensors.append(np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1)))
            transforms.append((width, height, ratio, pad_x, pad_y))
        tensor = np.stack(tensors, axis=0)
        target_size = np.full((len(frames), 2), self.input_size, dtype=np.int64)
        try:
            raw = self.session.run(
                ["labels", "boxes", "scores"],
                {"images": tensor, "orig_target_sizes": target_size},
            )
        except Exception as error:
            raise VisionConfigurationError(f"ONNX 推理失败：{error}") from error
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            raise VisionConfigurationError("ONNX 模型返回了不支持的结果")
        labels = self._normalize_batch_output(raw[0], 1, len(frames))
        boxes = self._normalize_batch_output(raw[1], 2, len(frames))
        scores = self._normalize_batch_output(raw[2], 1, len(frames))
        if (boxes.shape[-1] != 4 or labels.shape[1] != boxes.shape[1]
                or scores.shape[1] != boxes.shape[1]):
            raise VisionConfigurationError("ONNX 模型输出维度不一致")

        results = []
        minimum_confidence = (
            self.tracking_low_confidence_threshold if tracking else self.confidence_threshold
        )
        for batch_index, (width, height, ratio, pad_x, pad_y) in enumerate(transforms):
            detections = []
            for class_id, box, confidence in zip(labels[batch_index], boxes[batch_index], scores[batch_index]):
                if not np.isfinite(class_id) or not np.isfinite(confidence):
                    continue
                class_index = int(class_id)
                confidence = float(confidence)
                if class_index < 0 or class_index >= len(VISDRONE_CLASSES) or confidence < minimum_confidence:
                    continue
                if not np.isfinite(box).all():
                    continue
                x1, y1, x2, y2 = map(float, box)
                x1 = min(float(width), max(0.0, (x1 - pad_x) / ratio))
                y1 = min(float(height), max(0.0, (y1 - pad_y) / ratio))
                x2 = min(float(width), max(0.0, (x2 - pad_x) / ratio))
                y2 = min(float(height), max(0.0, (y2 - pad_y) / ratio))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({
                    "label": VISDRONE_CLASSES[class_index],
                    "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "confidence": round(confidence, 4),
                    "track_id": None,
                })
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            results.append(detections)
        return results

    def detect_regions(self, frame, regions: list[dict], *, tracking: bool = False) -> list[dict[str, Any]]:
        """Run inference only on usable annotated areas, tiling large crops as needed."""
        height, width = frame.shape[:2]
        crop_boxes = []
        for region in regions or []:
            if not isinstance(region, dict) or region.get("kind") not in {
                "vehicle_lane", "pedestrian", "road_surface", "parking",
            }:
                continue
            polygon = region.get("polygon")
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            try:
                points = [(float(point[0]), float(point[1])) for point in polygon]
            except (TypeError, ValueError, IndexError):
                continue
            if any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y in points):
                continue
            x1 = max(0, min(width - 1, int(min(x for x, _ in points) * width)))
            y1 = max(0, min(height - 1, int(min(y for _, y in points) * height)))
            x2 = max(x1 + 1, min(width, int(max(x for x, _ in points) * width + 0.999)))
            y2 = max(y1 + 1, min(height, int(max(y for _, y in points) * height + 0.999)))
            box = (x1, y1, x2, y2)
            if box not in crop_boxes:
                crop_boxes.append(box)
        if not crop_boxes:
            raise ValueError("排除区域不能单独用于识别，请重新圈选道路、人行或停车区域。")

        detections = []
        pending_tiles = []

        def infer_pending_tiles():
            if not pending_tiles:
                return
            results = self._detect_batch([tile for tile, _ in pending_tiles], tracking=tracking)
            for (_, (origin_x, origin_y)), tile_detections in zip(pending_tiles, results):
                for item in tile_detections:
                    shifted = dict(item)
                    bx1, by1, bx2, by2 = map(float, item["box"])
                    shifted["box"] = [bx1 + origin_x, by1 + origin_y, bx2 + origin_x, by2 + origin_y]
                    detections.append(shifted)
            pending_tiles.clear()

        for x1, y1, x2, y2 in crop_boxes:
            crop = frame[y1:y2, x1:x2]
            crop_height, crop_width = crop.shape[:2]
            # A 1.25x source tile is reduced only 20% at inference, versus a
            # much larger reduction when resizing a full 1080p/4K frame.
            tile_size = max(self.input_size, round(self.input_size * 1.25))
            overlap = max(1, round(tile_size * 0.2))
            x_offsets = self._tile_offsets(crop_width, tile_size, overlap)
            y_offsets = self._tile_offsets(crop_height, tile_size, overlap)
            for tile_y in y_offsets:
                for tile_x in x_offsets:
                    tile = crop[tile_y:min(tile_y + tile_size, crop_height),
                                tile_x:min(tile_x + tile_size, crop_width)]
                    pending_tiles.append((tile, (x1 + tile_x, y1 + tile_y)))
                    if len(pending_tiles) >= self.tile_batch_size:
                        infer_pending_tiles()
        infer_pending_tiles()
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        deduplicated = []
        for item in detections:
            x1, y1, x2, y2 = item["box"]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            duplicate = False
            for kept in deduplicated:
                if item["label"] != kept["label"]:
                    continue
                kx1, ky1, kx2, ky2 = kept["box"]
                intersection = max(0.0, min(x2, kx2) - max(x1, kx1)) * max(
                    0.0, min(y2, ky2) - max(y1, ky1)
                )
                union = area + max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1) - intersection
                kept_area = max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1)
                overlap_over_smaller = intersection / min(area, kept_area) if min(area, kept_area) > 0 else 0
                if union > 0 and (intersection / union >= 0.6 or overlap_over_smaller >= 0.8):
                    # A box clipped at one tile edge may be completed by its overlapping neighbor.
                    kept["box"] = [min(x1, kx1), min(y1, ky1), max(x2, kx2), max(y2, ky2)]
                    duplicate = True
                    break
            if not duplicate:
                deduplicated.append(item)
        return deduplicated

    @staticmethod
    def _tile_offsets(length: int, tile_size: int, overlap: int) -> list[int]:
        """Cover an axis with overlapping tiles and anchor the final tile at its end."""
        if length <= tile_size:
            return [0]
        stride = max(1, tile_size - overlap)
        final_offset = length - tile_size
        tile_count = math.ceil(final_offset / stride) + 1
        return [round(index * final_offset / (tile_count - 1)) for index in range(tile_count)]


def _gray_for_motion(frame, cv2):
    height, width = frame.shape[:2]
    scale = min(640.0 / max(width, height), 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale


def _estimate_camera_transform(previous_gray, current_gray, cv2, scale: float):
    """Estimate a conservative previous-frame -> current-frame background homography."""
    import numpy as np

    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=6)
    previous_keypoints, previous_descriptors = orb.detectAndCompute(previous_gray, None)
    current_keypoints, current_descriptors = orb.detectAndCompute(current_gray, None)
    if previous_descriptors is None or current_descriptors is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(previous_descriptors, current_descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 20:
        return None
    source = np.float32([previous_keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    target = np.float32([current_keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    matrix, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if matrix is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.sum())
    if inliers < 12 or inliers / len(good) < 0.35:
        return None
    # Convert the transform from downscaled coordinates back to original pixels.
    scale_matrix = np.diag([scale, scale, 1.0])
    inverse_scale = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    full_resolution_matrix = inverse_scale @ matrix @ scale_matrix
    if not np.isfinite(full_resolution_matrix).all() or abs(np.linalg.det(full_resolution_matrix)) < 1e-8:
        return None
    return full_resolution_matrix.tolist()


def _relative_camera_transform(previous_anchor_transform, current_anchor_transform):
    """Derive previous-frame -> current-frame motion from two anchor-frame homographies."""
    if previous_anchor_transform is None or current_anchor_transform is None:
        return None
    import numpy as np

    try:
        previous = np.asarray(previous_anchor_transform, dtype=np.float64)
        current = np.asarray(current_anchor_transform, dtype=np.float64)
        if (previous.shape != (3, 3) or current.shape != (3, 3)
                or not np.isfinite(previous).all() or not np.isfinite(current).all()
                or abs(np.linalg.det(previous)) < 1e-8):
            return None
        relative = current @ np.linalg.inv(previous)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(relative).all() or abs(np.linalg.det(relative)) < 1e-8:
        return None
    return relative.tolist()


def _merge_segment_analyses(segments: list[dict], *, accident_threshold: float = 0.85,
                            flood_area_threshold: float = 0.08,
                            accident_model_enabled: bool = False,
                            flood_model_enabled: bool = False) -> dict:
    """Combine bounded per-segment results without retaining all frame detections."""
    total_frames = sum(item["metrics"].get("frames_analyzed", 0) for item in segments)
    if total_frames <= 0:
        raise ValueError("视频中没有可读取的画面")
    weighted_keys = ("mean_vehicle_count", "mean_pedestrian_count", "mean_vehicle_occupied_area_ratio")
    metrics = {key: 0.0 for key in weighted_keys}
    peaks = {"peak_vehicle_count": 0, "peak_pedestrian_count": 0}
    classes_mean: dict[str, float] = {}
    classes_peak: dict[str, int] = {}
    tracked = stationary = valid_transitions = transitions = 0
    candidate_events: dict[tuple[str, str | None], list[list[dict]]] = {}
    specialized_observations: list[dict] = []
    specialized_frame_times: set[float] = set()
    for segment_index, segment in enumerate(segments):
        segment_metrics = segment["metrics"]
        count = segment_metrics.get("frames_analyzed", 0)
        for key in weighted_keys:
            metrics[key] += segment_metrics.get(key, 0.0) * count
        for key in peaks:
            peaks[key] = max(peaks[key], int(segment_metrics.get(key, 0)))
        for label, count_value in segment_metrics.get("mean_class_counts", {}).items():
            classes_mean[label] = classes_mean.get(label, 0.0) + count_value * count
        for label, count_value in segment_metrics.get("peak_class_counts", {}).items():
            classes_peak[label] = max(classes_peak.get(label, 0), int(count_value))
        tracked += int(segment_metrics.get("tracked_vehicle_count", 0))
        stationary += int(segment_metrics.get("stationary_track_count", 0))
        valid_transitions += int(segment_metrics.get("motion_compensation_valid_transitions", 0))
        transitions += int(segment_metrics.get("motion_compensation_transition_count", 0))
        for candidate in segment.get("candidates", []):
            evidence = candidate.get("evidence", {})
            summary = evidence.get("summary") if isinstance(evidence, dict) else None
            if not isinstance(summary, dict) and isinstance(evidence, dict):
                summary = evidence
            region_id = summary.get("region_id") if isinstance(summary, dict) else None
            if region_id is None and isinstance(evidence, dict):
                region_id = evidence.get("region_id")
            region_id = str(region_id) if region_id is not None else None
            key = (candidate["kind"], region_id)
            groups = candidate_events.setdefault(key, [])
            occurrence = {
                "confidence": candidate.get("confidence", 0.0),
                "reason": candidate.get("reason", ""),
                "evidence": evidence,
                "start_seconds": candidate.get("_candidate_window_start_seconds", segment.get(
                    "candidate_window_start_seconds", segment["start_seconds"],
                )),
                "end_seconds": candidate.get("_candidate_window_end_seconds", segment.get(
                    "candidate_window_end_seconds", segment["end_seconds"],
                )),
                "segment_index": segment_index,
            }
            if groups and groups[-1][-1]["segment_index"] == segment_index - 1:
                groups[-1].append(occurrence)
            else:
                groups.append([occurrence])
        observations = segment.get("specialized_observations", [])
        if isinstance(observations, list):
            specialized_observations.extend(
                item for item in observations if isinstance(item, dict)
            )
            specialized_frame_times.update(
                float(item["time_seconds"]) for item in observations
                if isinstance(item, dict)
                and isinstance(item.get("time_seconds"), (int, float))
                and not isinstance(item.get("time_seconds"), bool)
            )
    for key in weighted_keys:
        metrics[key] = round(metrics[key] / total_frames, 4 if "ratio" in key else 2)
    for label in classes_mean:
        classes_mean[label] = round(classes_mean[label] / total_frames, 2)
    ratio = valid_transitions / transitions if transitions else 0.0
    candidates = []
    for (kind, region_id), event_groups in candidate_events.items():
        for evidence_segments in event_groups:
            strongest = max(evidence_segments, key=lambda item: item["confidence"])
            strongest_evidence = strongest.get("evidence", {})
            summary = (
                strongest_evidence.get("summary")
                if isinstance(strongest_evidence, dict) else None
            )
            if not isinstance(summary, dict) and isinstance(strongest_evidence, dict):
                summary = strongest_evidence
            summary = dict(summary) if isinstance(summary, dict) else {}
            if region_id is not None:
                summary.setdefault("region_id", region_id)
            candidates.append({
                "kind": kind,
                "confidence": strongest["confidence"],
                "reason": strongest["reason"],
                "evidence": {
                    "occurrences": len(evidence_segments),
                    "segments": [
                        {key: value for key, value in item.items() if key != "segment_index"}
                        for item in evidence_segments[:200]
                    ],
                    "segments_truncated": len(evidence_segments) > 200,
                    "summary": summary,
                },
                "review_required": True,
                "auto_publish": False,
                "status": "pending_review",
            })
    candidates.extend(build_specialized_candidates(
        specialized_observations, sample_interval_s=1.0,
        accident_threshold=accident_threshold,
        flood_area_threshold=flood_area_threshold,
    ))
    stabilized = all(item.get("safety", {}).get("camera_stabilized_assumed") for item in segments)
    motion_ready = stabilized or (transitions > 0 and ratio >= 0.8)
    return {
        "metrics": {
            "frames_analyzed": total_frames,
            **metrics,
            **peaks,
            "mean_class_counts": classes_mean,
            "peak_class_counts": classes_peak,
            "tracked_vehicle_count": tracked,
            "stationary_track_count": stationary,
            "stationary_track_ratio": round(stationary / tracked, 3) if tracked else 0.0,
            "motion_compensation_valid_transitions": valid_transitions,
            "motion_compensation_transition_count": transitions,
            "motion_compensation_valid_ratio": round(ratio, 3),
            "motion_assessment": (
                "operator_declared_stabilized" if stabilized
                else "camera_motion_compensated" if motion_ready
                else "camera_motion_uncompensated"
            ),
            "analysis_segments": len(segments),
            "specialized_frames_analyzed": len(specialized_frame_times),
        },
        "candidates": candidates,
        "safety": {
            "requires_human_review": True,
            "camera_motion_compensated": bool(motion_ready and not stabilized),
            "camera_stabilized_assumed": stabilized,
            "observation_region_reassociation_required": any(
                item.get("safety", {}).get("observation_region_reassociation_required") is True
                for item in segments
            ),
            "accident_recognition_supported": accident_model_enabled,
            "flood_segmentation_supported": flood_model_enabled,
            "automatically_changes_routing": False,
        },
    }


def _region_crop(frame, region: dict):
    height, width = frame.shape[:2]
    points = region.get("polygon") or []
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x1 = max(0, min(width - 1, int(min(xs) * width)))
    y1 = max(0, min(height - 1, int(min(ys) * height)))
    x2 = max(x1 + 1, min(width, int(max(xs) * width + 0.999)))
    y2 = max(y1 + 1, min(height, int(max(ys) * height + 0.999)))
    return frame[y1:y2, x1:x2]


def _specialized_predictions(frame, time_seconds, regions, accident_model, flood_segmenter):
    if accident_model is None and flood_segmenter is None:
        return [], []
    observations = []
    failures = []
    road_regions = [region for region in regions or []
                    if isinstance(region, dict) and region.get("kind") in {"vehicle_lane", "road_surface"}]
    if accident_model is not None and road_regions:
        region = next((item for item in road_regions if item.get("kind") == "vehicle_lane"), road_regions[0])
        try:
            prediction = accident_model.predict(_region_crop(frame, region))
            observations.append({
                "time_seconds": float(time_seconds),
                "traffic_accident_probability": prediction["traffic_accident_probability"],
                "accident_model_id": prediction["model_id"],
                "region_id": str(region.get("id") or "road"),
            })
        except Exception as error:
            logger.exception("Optional accident model inference failed")
            failures.append({"model": "accident", "error_type": type(error).__name__})
    if flood_segmenter is not None:
        for region in regions or []:
            if not isinstance(region, dict) or region.get("kind") != "road_surface":
                continue
            try:
                prediction = flood_segmenter.predict(frame, region)
                if prediction is not None:
                    observations.append({"time_seconds": float(time_seconds), **prediction})
            except Exception as error:
                logger.exception("Optional flood segmentation inference failed")
                failures.append({"model": "flood", "error_type": type(error).__name__})
    return observations, failures


def _compact_specialized_observations(observations: list[dict]) -> list[dict]:
    """Keep temporal scores but only one largest flood outline per road/segment."""
    strongest_outline = {}
    for index, item in enumerate(observations):
        ratio = item.get("flooded_road_area_ratio")
        outlines = item.get("outline_polygons")
        if (not isinstance(ratio, (int, float)) or isinstance(ratio, bool)
                or not math.isfinite(float(ratio)) or not outlines):
            continue
        key = str(item.get("region_id") or "road-surface")
        if key not in strongest_outline or ratio > strongest_outline[key][0]:
            strongest_outline[key] = (float(ratio), index)
    keep_indices = {value[1] for value in strongest_outline.values()}
    compacted = []
    for index, item in enumerate(observations):
        copied = dict(item)
        if "flooded_road_area_ratio" in copied and index not in keep_indices:
            copied["outline_polygons"] = []
        compacted.append(copied)
    return compacted


def analyze_media(media_path: str | Path, media_kind: str, anchor_gcj: dict | None,
                  camera_stabilized: bool = False,
                  *, detector=None, accident_model=None, flood_segmenter=None,
                  regions: list[dict] | None = None,
                  progress_callback=None, resume_state: dict | None = None,
                  cancel_check=None) -> dict:
    """Analyze all sampled frames in bounded chunks and retain resumable review evidence."""
    path = Path(media_path)
    if not path.is_file():
        raise FileNotFoundError("巡检影像文件已丢失")
    detector = detector or OnnxDetector(
        config.VISION_MODEL_PATH,
        model_id=getattr(config, "VISION_MODEL_ID", "visdrone-rtdetrv4-s"),
        model_version=getattr(config, "VISION_MODEL_REVISION", "unspecified"),
    )
    requested_accident_model = accident_model
    requested_flood_segmenter = flood_segmenter
    optional_model_errors: dict[str, str] = {}
    started = time.monotonic()
    try:
        import cv2
    except ImportError as error:
        raise VisionConfigurationError("视觉依赖未安装，请安装 requirements-vision.txt") from error

    frames: list[list[dict]] = []
    transforms: list[list[list[float]] | None] = [None]
    region_transforms: list[list[list[float]] | None] = []
    frame_indices: list[int] = []
    completed_segments = []
    temporal_context = {
        "frames": [], "transforms": [], "region_transforms": [], "frame_indices": [],
    }
    width = height = 0
    sample_interval = 1.0
    duration_seconds = None
    fps = None
    total_frames = None
    image_detections = None
    specialized_observations: list[dict] = []
    if media_kind == "image":
        try:
            from PIL import Image
        except ImportError as error:
            raise VisionConfigurationError("缺少图片尺寸校验依赖，请安装 requirements-vision.txt") from error
        try:
            with Image.open(path) as image:
                _validate_frame_dimensions(*image.size)
        except OSError as error:
            raise ValueError("图片无法读取或分辨率无效") from error
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("图片无法解码，请上传有效的 JPG、PNG 或 WebP 文件")
        height, width = frame.shape[:2]
        _validate_frame_dimensions(width, height)
        image_detections = (
            detector.detect_regions(frame, regions) if regions and hasattr(detector, "detect_regions")
            else detector.detect(frame, tracking=False)
        )
        analysis = analyze_observations(
            [image_detections], frame_width=width,
            frame_height=height, camera_stabilized=True, regions=regions,
        )
        specialized_observations, failures = _specialized_predictions(
            frame, 0.0, regions, accident_model, flood_segmenter,
        )
        optional_model_errors.update({item["model"]: item["error_type"] for item in failures})
        analysis["candidates"].extend(build_specialized_candidates(
            specialized_observations, sample_interval_s=1.0, single_image=True,
            accident_threshold=getattr(accident_model, "threshold", 0.85),
            flood_area_threshold=getattr(flood_segmenter, "min_area_ratio", 0.08),
        ))
        analysis["safety"]["accident_recognition_supported"] = (
            accident_model is not None and "accident" not in optional_model_errors
        )
        analysis["safety"]["flood_segmentation_supported"] = (
            flood_segmenter is not None and "flood" not in optional_model_errors
        )
        completed_segments = [{"start_seconds": 0.0, "end_seconds": 0.0, **analysis}]
    elif media_kind == "video":
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("视频无法解码，请上传有效的视频文件")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _validate_frame_dimensions(width, height)
            anchor_gray = None
            anchor_scale = 1.0
            if regions and not camera_stabilized:
                anchor_capture = cv2.VideoCapture(str(path))
                try:
                    anchor_ok, anchor_frame = anchor_capture.read()
                    if not anchor_ok or anchor_frame is None:
                        raise ValueError("无法读取视频首帧，不能安全关联观察区域")
                    anchor_gray, anchor_scale = _gray_for_motion(anchor_frame, cv2)
                finally:
                    anchor_capture.release()
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = float(fps) if fps and fps > 0 else 1.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
            duration_seconds = total_frames / fps if total_frames else None
            target_fps = max(0.1, float(getattr(config, "VISION_SAMPLE_FPS", 5)))
            stride = max(1, round(fps / target_fps))
            total_sampled_frames = math.ceil(total_frames / stride) if total_frames else None
            sample_interval = stride / fps
            specialized_stride = max(stride, round(fps))
            dense_trigger_threshold = float(getattr(
                config, "VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD", 0.5,
            ))
            if (not math.isfinite(dense_trigger_threshold)
                    or not 0.0 <= dense_trigger_threshold <= 1.0):
                dense_trigger_threshold = 0.5
            dense_window_seconds = float(getattr(
                config, "VISION_ACCIDENT_DENSE_WINDOW_SECONDS", 2.0,
            ))
            if not math.isfinite(dense_window_seconds) or dense_window_seconds < 0:
                dense_window_seconds = 2.0
            accident_dense_until_frame = -1
            try:
                progress_interval_seconds = float(getattr(
                    config, "VISION_PROGRESS_UPDATE_SECONDS", 2.0,
                ))
            except (TypeError, ValueError):
                progress_interval_seconds = 2.0
            if not math.isfinite(progress_interval_seconds) or progress_interval_seconds <= 0:
                progress_interval_seconds = 2.0
            last_live_progress_at = started
            segment_limit = max(5, int(getattr(config, "VISION_SEGMENT_FRAMES", config.VISION_MAX_VIDEO_FRAMES)))
            checkpoint_fingerprint = _video_checkpoint_fingerprint(
                path=path, width=width, height=height, fps=fps, total_frames=total_frames,
                target_fps=target_fps, stride=stride, segment_limit=segment_limit,
                dense_trigger_threshold=dense_trigger_threshold,
                dense_window_seconds=dense_window_seconds,
                camera_stabilized=camera_stabilized, regions=regions,
                anchor_gcj=anchor_gcj, detector=detector,
                accident_model=accident_model, flood_segmenter=flood_segmenter,
            )

            def report_checkpoint_reset():
                if progress_callback:
                    progress_callback({
                        "progress": {
                            "phase": "analyzing", "percent": 0,
                            "frames_analyzed": 0, "total_frames": total_frames,
                            "total_sampled_frames": total_sampled_frames,
                            "analyzed_through_seconds": 0.0,
                            "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
                        },
                        "checkpoint": {
                            "next_frame_index": 0, "segments": [],
                            "accident_dense_until_frame": -1,
                            "analysis_fingerprint": checkpoint_fingerprint,
                        },
                    })

            saved_checkpoint = resume_state or {}
            checkpoint_has_work = bool(
                saved_checkpoint.get("next_frame_index")
                or saved_checkpoint.get("segments")
            )
            checkpoint_matches = (
                saved_checkpoint.get("analysis_fingerprint") == checkpoint_fingerprint
            )
            if checkpoint_has_work and not checkpoint_matches:
                # Never combine completed segments produced under different
                # weights, thresholds, ROIs, or sampling rules.
                report_checkpoint_reset()
            else:
                completed_segments = list(saved_checkpoint.get("segments", []))
                accident_dense_until_frame = int(
                    saved_checkpoint.get("accident_dense_until_frame", -1)
                )
                saved_context = saved_checkpoint.get("temporal_context")
                if isinstance(saved_context, dict):
                    context_frames = saved_context.get("frames")
                    context_transforms = saved_context.get("transforms")
                    context_indices = saved_context.get("frame_indices")
                    context_region_transforms = saved_context.get("region_transforms")
                    if (isinstance(context_frames, list)
                            and isinstance(context_transforms, list)
                            and isinstance(context_indices, list)
                            and isinstance(context_region_transforms, list)
                            and len(context_frames) == len(context_transforms)
                            and len(context_frames) == len(context_indices)
                            and (not regions or len(context_frames) == len(context_region_transforms))
                            and len(context_indices) <= math.ceil(
                                VIDEO_TEMPORAL_CONTEXT_SECONDS / sample_interval,
                            ) + 1):
                        temporal_context = {
                            "frames": context_frames,
                            "transforms": context_transforms,
                            "region_transforms": context_region_transforms,
                            "frame_indices": context_indices,
                        }
            frame_index = (
                int(saved_checkpoint.get("next_frame_index", 0))
                if checkpoint_matches else 0
            )
            if frame_index:
                seeked = (
                    hasattr(capture, "set")
                    and capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                )
                reported_position = (
                    capture.get(cv2.CAP_PROP_POS_FRAMES)
                    if hasattr(capture, "get") else None
                )
                seek_position_valid = (
                    isinstance(reported_position, (int, float))
                    and math.isfinite(float(reported_position))
                    and abs(float(reported_position) - frame_index) <= 0.5
                )
                if not seeked or not seek_position_valid:
                    # Some codecs report success but do not honor random access.
                    # Discard stale checkpoints and reprocess from the beginning
                    # rather than assigning incorrect timestamps to earlier frames.
                    frame_index = 0
                    accident_dense_until_frame = -1
                    completed_segments = []
                    temporal_context = {
                        "frames": [], "transforms": [],
                        "region_transforms": [], "frame_indices": [],
                    }
                    report_checkpoint_reset()
            previous_gray = None
            previous_scale = 1.0
            previous_region_transform = None

            def flush_segment(next_frame_index: int):
                nonlocal frames, transforms, region_transforms, frame_indices, specialized_observations
                nonlocal last_live_progress_at, temporal_context
                if not frames:
                    return
                segment_start = frame_indices[0] / fps
                segment_end = frame_indices[-1] / fps
                result = analyze_observations(
                    frames, frame_width=width, frame_height=height,
                    sample_interval_s=sample_interval,
                    camera_stabilized=camera_stabilized,
                    frame_transforms=transforms,
                    region_transforms=region_transforms if regions else None,
                    regions=regions,
                    tracking_high_threshold=getattr(detector, "confidence_threshold", 0.369),
                    tracking_low_threshold=getattr(detector, "tracking_low_confidence_threshold", 0.1),
                )
                context_frames = temporal_context["frames"]
                temporal_frames = context_frames + frames
                temporal_frame_indices = temporal_context["frame_indices"] + frame_indices
                if context_frames:
                    temporal_transforms = (
                        temporal_context["transforms"] + [None] + transforms[1:]
                    )
                    temporal_region_transforms = (
                        temporal_context["region_transforms"] + region_transforms
                    ) if regions else None
                else:
                    temporal_transforms = transforms
                    temporal_region_transforms = region_transforms if regions else None
                if context_frames:
                    context_window_start_seconds = temporal_frame_indices[0] / fps
                    for candidate in result["candidates"]:
                        if candidate.get("kind") in {"possible_congestion", "possible_crowding"}:
                            candidate["_candidate_window_start_seconds"] = round(segment_start, 3)
                            candidate["_candidate_window_end_seconds"] = round(segment_end, 3)
                    temporal_result = analyze_observations(
                        temporal_frames, frame_width=width, frame_height=height,
                        sample_interval_s=sample_interval,
                        camera_stabilized=camera_stabilized,
                        frame_transforms=temporal_transforms,
                        region_transforms=temporal_region_transforms,
                        regions=regions,
                        tracking_high_threshold=getattr(detector, "confidence_threshold", 0.369),
                        tracking_low_threshold=getattr(detector, "tracking_low_confidence_threshold", 0.1),
                    )
                    temporal_kinds = {"possible_congestion", "possible_crowding"}
                    for candidate in temporal_result["candidates"]:
                        if candidate.get("kind") in temporal_kinds:
                            candidate["_candidate_window_start_seconds"] = round(
                                context_window_start_seconds, 3,
                            )
                            candidate["_candidate_window_end_seconds"] = round(segment_end, 3)
                    selected_temporal_candidates = {}
                    for candidate in result["candidates"] + temporal_result["candidates"]:
                        if candidate.get("kind") not in temporal_kinds:
                            continue
                        evidence = candidate.get("evidence", {})
                        summary = evidence.get("summary", {}) if isinstance(evidence, dict) else {}
                        region_id = summary.get("region_id") if isinstance(summary, dict) else None
                        key = (candidate.get("kind"), str(region_id) if region_id is not None else None)
                        current = selected_temporal_candidates.get(key)
                        current_summary = (
                            current.get("evidence", {}).get("summary", {})
                            if current and isinstance(current.get("evidence"), dict) else {}
                        )
                        current_duration = current_summary.get("region_observed_duration_seconds", 0)
                        candidate_duration = summary.get("region_observed_duration_seconds", 0)
                        if (current is None or candidate_duration > current_duration
                                or (candidate_duration == current_duration
                                    and candidate.get("confidence", 0) > current.get("confidence", 0))):
                            selected_temporal_candidates[key] = candidate
                    result["candidates"] = [
                        candidate for candidate in result["candidates"]
                        if candidate.get("kind") not in temporal_kinds
                    ] + list(selected_temporal_candidates.values())
                candidate_window_start_seconds = (
                    temporal_frame_indices[0] / fps if temporal_frame_indices else segment_start
                )
                context_limit = max(
                    2, math.ceil(VIDEO_TEMPORAL_CONTEXT_SECONDS / sample_interval) + 1,
                )
                all_temporal_transforms = temporal_transforms
                keep_from = max(0, len(temporal_frames) - context_limit)
                temporal_context = {
                    "frames": temporal_frames[keep_from:],
                    "transforms": ([None] + all_temporal_transforms[keep_from + 1:]
                                   if keep_from < len(temporal_frames) else []),
                    "region_transforms": (
                        temporal_region_transforms[keep_from:]
                        if regions and temporal_region_transforms is not None else []
                    ),
                    "frame_indices": temporal_frame_indices[keep_from:],
                }
                result["safety"]["accident_recognition_supported"] = accident_model is not None
                result["safety"]["flood_segmentation_supported"] = flood_segmenter is not None
                result["metrics"]["specialized_frames_analyzed"] = len({
                    item["time_seconds"] for item in specialized_observations
                })
                segment_observations = _compact_specialized_observations(specialized_observations)
                completed_segments.append({
                    "start_seconds": round(segment_start, 3),
                    "end_seconds": round(segment_end, 3),
                    "candidate_window_start_seconds": round(candidate_window_start_seconds, 3),
                    "candidate_window_end_seconds": round(segment_end, 3),
                    "specialized_observations": segment_observations,
                    **result,
                })
                sampled_count = sum(
                    item["metrics"].get("frames_analyzed", 0) for item in completed_segments
                )
                percent = min(99, round(next_frame_index / total_frames * 100)) if total_frames else None
                checkpoint = {
                    "next_frame_index": next_frame_index,
                    "segments": completed_segments,
                    "accident_dense_until_frame": accident_dense_until_frame,
                    "temporal_context": temporal_context,
                    "analysis_fingerprint": checkpoint_fingerprint,
                }
                if progress_callback:
                    progress_callback({
                        "progress": {
                            "phase": "analyzing",
                            "percent": percent,
                            "frames_analyzed": sampled_count,
                            "total_frames": total_frames,
                            "total_sampled_frames": total_sampled_frames,
                            "analyzed_through_seconds": round(segment_end, 3),
                            "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
                        },
                        "checkpoint": checkpoint,
                    })
                    last_live_progress_at = time.monotonic()
                frames = []
                transforms = [None]
                region_transforms = []
                frame_indices = []
                specialized_observations = []

            def persist_cancel_checkpoint(next_frame_index: int):
                if frames:
                    flush_segment(next_frame_index)
                    return
                if progress_callback:
                    sampled_count = sum(
                        item["metrics"].get("frames_analyzed", 0)
                        for item in completed_segments
                    )
                    analyzed_through = max(
                        (item["end_seconds"] for item in completed_segments), default=0.0,
                    )
                    percent = (
                        min(99, round(next_frame_index / total_frames * 100))
                        if total_frames else None
                    )
                    progress_callback({
                        "progress": {
                            "phase": "analyzing", "percent": percent,
                            "frames_analyzed": sampled_count, "total_frames": total_frames,
                            "total_sampled_frames": total_sampled_frames,
                            "analyzed_through_seconds": analyzed_through,
                            "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
                        },
                        "checkpoint": {
                            "next_frame_index": next_frame_index,
                            "segments": completed_segments,
                            "accident_dense_until_frame": accident_dense_until_frame,
                            "temporal_context": temporal_context,
                            "analysis_fingerprint": checkpoint_fingerprint,
                        },
                    })

            try:
                while True:
                    if cancel_check and cancel_check():
                        persist_cancel_checkpoint(frame_index)
                        raise VisionAnalysisCancelled("管理员已取消该影像任务")
                    ok, frame = capture.read()
                    if not ok:
                        if total_frames and frame_index < total_frames:
                            raise ValueError(
                                f"视频实际可解码帧数（{frame_index}）少于文件声明帧数（{total_frames}）；"
                                "未分析的尾部不会作为完整结果发布，请检查视频后重新上传。"
                            )
                        break
                    if frame_index % stride == 0:
                        height, width = frame.shape[:2]
                        _validate_frame_dimensions(width, height)
                        gray = None
                        scale = 1.0
                        if regions and not camera_stabilized:
                            gray, scale = _gray_for_motion(frame, cv2)
                            region_transform = (
                                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                                if frame_index == 0 else
                                _estimate_camera_transform(anchor_gray, gray, cv2, scale)
                                if anchor_gray is not None and abs(scale - anchor_scale) < 1e-6
                                else None
                            )
                            frame_regions = project_regions_for_frame(
                                regions, region_transform, width, height,
                            )
                        else:
                            region_transform = (
                                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                                if camera_stabilized and regions else None
                            )
                            frame_regions = regions or []
                        if regions:
                            region_transforms.append(region_transform)
                        frame_detections = (
                            detector.detect_regions(frame, frame_regions, tracking=True)
                            if frame_regions and hasattr(detector, "detect_regions")
                            else detector.detect(frame, tracking=True)
                        )
                        frames.append(frame_detections)
                        frame_indices.append(frame_index)
                        regular_specialized_frame = frame_index % specialized_stride == 0
                        accident_model_available = (
                            accident_model is not None and "accident" not in optional_model_errors
                        )
                        flood_model_available = (
                            flood_segmenter is not None and "flood" not in optional_model_errors
                        )
                        dense_accident_frame = (
                            accident_model_available
                            and frame_index <= accident_dense_until_frame
                        )
                        if (
                            regular_specialized_frame
                            and (accident_model_available or flood_model_available)
                        ) or dense_accident_frame:
                            was_in_dense_window = dense_accident_frame
                            observations, failures = _specialized_predictions(
                                frame, frame_index / fps, frame_regions,
                                accident_model if (
                                    regular_specialized_frame or dense_accident_frame
                                ) and accident_model_available else None,
                                flood_segmenter if (
                                    regular_specialized_frame and flood_model_available
                                ) else None,
                            )
                            specialized_observations.extend(observations)
                            optional_model_errors.update({
                                item["model"]: item["error_type"] for item in failures
                            })
                            if regular_specialized_frame and not was_in_dense_window:
                                accident_score = max((
                                    float(item["traffic_accident_probability"])
                                    for item in observations
                                    if isinstance(item.get("traffic_accident_probability"), (int, float))
                                    and not isinstance(item.get("traffic_accident_probability"), bool)
                                    and 0.0 <= float(item["traffic_accident_probability"]) <= 1.0
                                    and math.isfinite(float(item["traffic_accident_probability"]))
                                ), default=0.0)
                                if (accident_model_available
                                        and accident_score >= dense_trigger_threshold):
                                    accident_dense_until_frame = max(
                                        accident_dense_until_frame,
                                        frame_index + math.ceil(dense_window_seconds * fps),
                                    )
                        if len(frames) > 1 and camera_stabilized:
                            transforms.append([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                        elif len(frames) > 1:
                            if regions:
                                transform = _relative_camera_transform(
                                    previous_region_transform, region_transform,
                                )
                            else:
                                if gray is None:
                                    gray, scale = _gray_for_motion(frame, cv2)
                                transform = (
                                    _estimate_camera_transform(previous_gray, gray, cv2, scale)
                                    if previous_gray is not None and abs(scale - previous_scale) < 1e-6
                                    else None
                                )
                            transforms.append(transform)
                            if not regions:
                                previous_gray, previous_scale = gray, scale
                        elif not camera_stabilized:
                            if not regions and gray is None:
                                gray, scale = _gray_for_motion(frame, cv2)
                            if not regions:
                                previous_gray, previous_scale = gray, scale
                        if regions and not camera_stabilized:
                            previous_region_transform = region_transform
                        now = time.monotonic()
                        if (progress_callback
                                and now - last_live_progress_at >= progress_interval_seconds):
                            sampled_count = sum(
                                item["metrics"].get("frames_analyzed", 0)
                                for item in completed_segments
                            ) + len(frames)
                            next_frame_index = min(
                                frame_index + 1, total_frames or frame_index + 1,
                            )
                            percent = (
                                min(99, round(next_frame_index / total_frames * 100))
                                if total_frames else None
                            )
                            progress_callback({
                                "progress": {
                                    "phase": "analyzing", "percent": percent,
                                    "frames_analyzed": sampled_count, "total_frames": total_frames,
                                    "total_sampled_frames": total_sampled_frames,
                                    "analyzed_through_seconds": round(
                                        min(duration_seconds, frame_index / fps)
                                        if duration_seconds else frame_index / fps, 3,
                                    ),
                                    "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
                                },
                            })
                            last_live_progress_at = now
                        if len(frames) >= segment_limit:
                            flush_segment(frame_index + 1)
                        if time.monotonic() - started > config.VISION_MAX_ANALYSIS_SECONDS:
                            raise TimeoutError("影像分析超过配置时限；任务会从最近完成的视频分段继续")
                    frame_index += 1
                flush_segment(frame_index)
            finally:
                capture.release()
            if not completed_segments:
                raise ValueError("视频中没有可读取的画面")
            analysis = _merge_segment_analyses(
                completed_segments,
                accident_threshold=getattr(accident_model, "threshold", 0.85),
                flood_area_threshold=getattr(flood_segmenter, "min_area_ratio", 0.08),
                accident_model_enabled=accident_model is not None,
                flood_model_enabled=flood_segmenter is not None,
            )
            analysis["safety"]["accident_recognition_supported"] = (
                accident_model is not None and "accident" not in optional_model_errors
            )
            analysis["safety"]["flood_segmentation_supported"] = (
                flood_segmenter is not None and "flood" not in optional_model_errors
            )
            analyzed_frames = analysis["metrics"]["frames_analyzed"]
            last_sampled = max((segment["end_seconds"] for segment in completed_segments), default=0.0)
            observed_span = min(duration_seconds or last_sampled + sample_interval,
                                last_sampled + sample_interval)
            coverage_ratio = min(1.0, observed_span / duration_seconds) if duration_seconds else None
            analysis["video_coverage"] = {
                "complete": True,
                "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
                "analyzed_from_seconds": completed_segments[0]["start_seconds"],
                "analyzed_through_seconds": round(last_sampled, 3),
                "sampled_frames": analyzed_frames,
                "coverage_ratio": round(coverage_ratio, 4) if coverage_ratio is not None else None,
                "segment_count": len(completed_segments),
                "sample_interval_seconds": round(sample_interval, 4),
            }
        except Exception:
            capture.release()
            raise
    else:
        raise ValueError("不支持的影像类型")

    if time.monotonic() - started > config.VISION_MAX_ANALYSIS_SECONDS:
        raise TimeoutError("影像分析超过配置时限")
    model_info = {
        "id": getattr(detector, "model_id", "injected-detector"),
        "version": getattr(detector, "model_version", "unspecified"),
        "confidence_threshold": getattr(detector, "confidence_threshold", None),
        "tracking": {
            "algorithm": "bytetrack",
            "implementation": "FoundationVision/ByteTrack@d1bf019",
            "camera_motion_compensation": "homography_projected_to_segment_anchor",
            "low_confidence_tracking": True,
            "high_confidence_threshold": getattr(detector, "confidence_threshold", 0.369),
            "low_confidence_threshold": getattr(detector, "tracking_low_confidence_threshold", 0.1),
        },
        "classes": list(VISDRONE_CLASSES),
        "dataset_source": "VisDrone2019-DET (AISKYEYE team, Tianjin University)",
        "dataset_license": "CC BY-NC-SA 3.0; non-commercial academic research use",
        "weights_distributed_by_application": False,
        "accident_model": ({
            "id": getattr(requested_accident_model, "model_id", "unknown"),
            "version": getattr(requested_accident_model, "model_version", "unverified"),
            "classes": ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"],
            "spatial_precision": "scene_classification_only",
            "runtime_status": "inference_failed" if "accident" in optional_model_errors else "loaded",
        } if requested_accident_model is not None else None),
        "flood_model": ({
            "id": getattr(requested_flood_segmenter, "model_id", "unknown"),
            "version": getattr(requested_flood_segmenter, "model_version", "unverified"),
            "classes": getattr(requested_flood_segmenter, "classes", ["flooded_road"]),
            "output_scope": getattr(requested_flood_segmenter, "output_scope", "flooded_road_class"),
            "roi_kind": "road_surface",
            "physical_area_or_depth": False,
            "runtime_status": "inference_failed" if "flood" in optional_model_errors else "loaded",
        } if requested_flood_segmenter is not None else None),
    }
    preview_detections = []
    if media_kind == "image":
        preview_detections = [
            item for item in image_detections
            if item.get("label", "").strip().lower() in VEHICLE_LABELS | {"bicycle"}
        ][:MAX_PREVIEW_DETECTIONS]
    has_anchor = anchor_gcj is not None
    return {
        **analysis,
        "media": {
            "kind": media_kind, "width": width, "height": height,
            "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
            "source_fps": round(fps, 3) if fps else None,
        },
        "model": model_info,
        "optional_model_errors": [
            {"model": model, "error_type": error_type}
            for model, error_type in sorted(optional_model_errors.items())
        ],
        "preview_detections": preview_detections,
        "anchor_gcj": anchor_gcj,
        "location_precision": "operator_selected_area_only" if has_anchor else "not_provided",
        "review_guidance": (
            "图像位置仅代表管理员标注的观察区域，不是检测框的地面坐标；确认管制前须人工选取具体道路段。"
            if has_anchor else
            "影像未提供地面坐标；确认候选并转入路况流程后，须由管理员人工选取具体道路段。"
        ),
    }
