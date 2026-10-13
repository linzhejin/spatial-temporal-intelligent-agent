"""Optional ONNX scene and flood-road models for human-reviewed evidence."""
from __future__ import annotations

import math
import os
from pathlib import Path

import config
from vision.tiling import tile_windows


AIDER_CLASSES = (
    # TakuNet's AIDER loader follows ImageFolder's sorted directory order.
    "collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident",
)
FLOODNET_CLASSES = (
    "background", "flooded_building", "non_flooded_building", "flooded_road",
    "non_flooded_road", "water", "tree", "vehicle", "pool", "grass",
)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _onnx_contract(model_path, session, *, expected_classes: int | tuple[int, ...] | None = None,
                   expected_output_rank: int | None = None, class_axis: int = -1):
    if session is None:
        if not model_path or not Path(model_path).is_file():
            raise ValueError("模型权重文件不存在")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("缺少 onnxruntime，无法加载专用视觉模型") from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.resolve_vision_cpu_threads()
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
        )
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = session.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise ValueError("专用模型必须有一个图像输入和至少一个输出")
    input_tensor = next(iter(inputs.values()))
    shape = input_tensor.shape
    if (len(shape) != 4 or shape[1] not in (3, "3")
            or not all(isinstance(value, int) and value > 0 for value in shape[2:])):
        raise ValueError("专用模型输入必须使用 [N,3,H,W]，且通道和空间尺寸固定")
    output_shape = outputs[0].shape
    if isinstance(output_shape, (list, tuple)):
        if expected_output_rank is not None and len(output_shape) != expected_output_rank:
            raise ValueError(f"模型输出维度必须为 {expected_output_rank}")
        if expected_classes is not None:
            allowed_classes = (expected_classes if isinstance(expected_classes, tuple)
                               else (expected_classes,))
            normalized_axis = class_axis if class_axis >= 0 else len(output_shape) + class_axis
            if normalized_axis < 0 or normalized_axis >= len(output_shape):
                raise ValueError("模型类别维度配置错误")
            size = output_shape[normalized_axis]
            if size not in allowed_classes and size not in tuple(str(value) for value in allowed_classes):
                expected_text = " 或 ".join(str(value) for value in allowed_classes)
                raise ValueError(f"模型输出类别数必须为 {expected_text}")
    return session, input_tensor.name, outputs[0].name, (int(shape[3]), int(shape[2]))


def _image_tensor(frame, input_name, size):
    import cv2
    import numpy as np

    if (frame is None or not hasattr(frame, "shape") or len(frame.shape) != 3
            or frame.shape[2] != 3):
        raise ValueError("模型输入必须是三通道彩色画面")
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - np.asarray(_IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        _IMAGENET_STD, dtype=np.float32,
    )
    return {input_name: np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)}


class OnnxAiderClassifier:
    """A five-class AIDER classifier; its output is a scene cue, not a location."""

    model_id = "aider-five-class-scene-classifier"

    def __init__(self, model_path: str, *, session=None, threshold: float | None = None,
                 model_version: str | None = None):
        self.session, self.input_name, self.output_name, self.input_size = _onnx_contract(
            model_path, session, expected_classes=len(AIDER_CLASSES),
        )
        self.model_path = str(model_path or "injected-session")
        self.threshold = float(threshold if threshold is not None else os.getenv(
            "VISION_ACCIDENT_THRESHOLD", "0.85",
        ))
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("VISION_ACCIDENT_THRESHOLD 必须大于 0 且不超过 1")
        self.model_version = model_version or os.getenv("VISION_ACCIDENT_REVISION", "unverified")

    def predict(self, frame) -> dict:
        import numpy as np

        try:
            raw = self.session.run([self.output_name], _image_tensor(frame, self.input_name, self.input_size))
        except Exception as error:
            raise RuntimeError(f"事故场景模型推理失败：{error}") from error
        values = np.asarray(raw[0], dtype=np.float64)
        if values.shape not in {(1, len(AIDER_CLASSES)), (len(AIDER_CLASSES),)}:
            raise ValueError("事故场景模型必须返回 5 个类别分数")
        values = values.reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("事故场景模型返回非有限分数")
        total = float(values.sum())
        if (np.all(values >= 0) and np.all(values <= 1)
                and math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3)):
            probabilities = values
        else:
            shifted = values - values.max()
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
        scores = {name: float(score) for name, score in zip(AIDER_CLASSES, probabilities)}
        label = max(scores, key=scores.get)
        return {
            "class": label,
            "traffic_accident_probability": round(scores["traffic_incident"], 6),
            "class_probabilities": scores,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "decision_threshold": self.threshold,
            "spatial_precision": "scene_classification_only",
        }


class OnnxFloodSegmenter:
    """Water segmentation clipped to an admin-marked road ROI for human review."""

    FLOODED_ROAD_CLASS = 3
    VISIBLE_WATER_CLASS = 1

    def __init__(self, model_path: str, *, session=None, model_version: str | None = None,
                 min_area_ratio: float | None = None,
                 flood_probability_threshold: float | None = None):
        self.session, self.input_name, self.output_name, self.input_size = _onnx_contract(
            model_path, session, expected_classes=(2, len(FLOODNET_CLASSES)),
            expected_output_rank=4, class_axis=1,
        )
        class_count = self.session.get_outputs()[0].shape[1]
        self.class_count = int(class_count)
        self.is_binary_visible_water = self.class_count == 2
        self.model_id = ("visible-water-mobilenetv3-unet" if self.is_binary_visible_water
                         else "floodnet-mobilenetv3-unet")
        self.classes = (["background_or_non_water", "visible_water"]
                        if self.is_binary_visible_water else list(FLOODNET_CLASSES))
        self.output_scope = ("visible_water_intersection_with_marked_road_roi"
                             if self.is_binary_visible_water else "flooded_road_class")
        self.water_class_id = (self.VISIBLE_WATER_CLASS if self.is_binary_visible_water
                               else self.FLOODED_ROAD_CLASS)
        self.model_path = str(model_path or "injected-session")
        self.model_version = model_version or os.getenv("VISION_FLOOD_REVISION", "unverified")
        self.min_area_ratio = float(min_area_ratio if min_area_ratio is not None else os.getenv(
            "VISION_FLOOD_MIN_AREA_RATIO", "0.08",
        ))
        if not math.isfinite(self.min_area_ratio) or not 0 < self.min_area_ratio <= 1:
            raise ValueError("VISION_FLOOD_MIN_AREA_RATIO 必须大于 0 且不超过 1")
        threshold_env = ("VISION_VISIBLE_WATER_CLASS_PROBABILITY_THRESHOLD"
                         if self.is_binary_visible_water
                         else "VISION_FLOOD_CLASS_PROBABILITY_THRESHOLD")
        raw_threshold = (os.getenv(threshold_env)
                         if flood_probability_threshold is None else flood_probability_threshold)
        if raw_threshold in (None, ""):
            self.flood_probability_threshold = None
        else:
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError) as error:
                raise ValueError("积水类别概率阈值必须是 0 到 1 之间的数") from error
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("积水类别概率阈值必须是 0 到 1 之间的数")
            self.flood_probability_threshold = threshold

    def predict(self, frame, region: dict, *, include_mask: bool = False,
                include_probabilities: bool = False) -> dict | None:
        import cv2
        import numpy as np

        if not isinstance(region, dict) or region.get("kind") != "road_surface":
            return None
        polygon = region.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            return None
        height, width = frame.shape[:2]
        points = np.asarray([
            [round(float(point[0]) * (width - 1)), round(float(point[1]) * (height - 1))]
            for point in polygon
        ], dtype=np.int32)
        x1 = max(0, int(points[:, 0].min()))
        y1 = max(0, int(points[:, 1].min()))
        x2 = min(width, int(points[:, 0].max()) + 1)
        y2 = min(height, int(points[:, 1].max()) + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        crop_points = points - np.asarray([x1, y1], dtype=np.int32)
        region_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(region_mask, [crop_points], 1)
        roi_pixels = int(region_mask.sum())
        if not roi_pixels:
            return None
        tile_width, tile_height = self.input_size
        tile_overlap = max(1, min(128, min(tile_width, tile_height) // 8))
        windows = tile_windows(
            width=x2 - x1, height=y2 - y1,
            tile_size=(tile_width, tile_height), overlap=tile_overlap,
        )
        best_confidence = np.full((y2 - y1, x2 - x1), -1.0, dtype=np.float32)
        best_flood_probability = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        labels = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        for tile_x, tile_y in windows:
            tile_x2 = min(x2 - x1, tile_x + tile_width)
            tile_y2 = min(y2 - y1, tile_y + tile_height)
            tile = crop[tile_y:tile_y2, tile_x:tile_x2]
            try:
                raw = self.session.run(
                    [self.output_name], _image_tensor(tile, self.input_name, self.input_size),
                )
            except Exception as error:
                raise RuntimeError(f"积水分割模型推理失败：{error}") from error
            logits = np.asarray(raw[0], dtype=np.float32)
            if (logits.ndim != 4 or logits.shape[0] != 1
                    or logits.shape[1] != self.class_count):
                raise ValueError(f"积水/水体分割模型必须返回 {self.class_count} 类语义分割结果")
            if not np.isfinite(logits).all():
                raise ValueError("积水分割模型返回非有限结果")
            tile_logits = logits[0]
            if tile_logits.shape[-2:] != (tile_y2 - tile_y, tile_x2 - tile_x):
                tile_logits = np.stack([
                    cv2.resize(channel, (tile_x2 - tile_x, tile_y2 - tile_y),
                               interpolation=cv2.INTER_LINEAR)
                    for channel in tile_logits
                ])
            maxima = tile_logits.max(axis=0)
            exponentials = np.exp(tile_logits - maxima[None, :, :])
            denominator = exponentials.sum(axis=0)
            confidence = 1.0 / denominator
            flood_probability = exponentials[self.water_class_id] / denominator
            predicted = tile_logits.argmax(axis=0).astype(np.uint8)
            target_confidence = best_confidence[tile_y:tile_y2, tile_x:tile_x2]
            better = confidence > target_confidence
            target_confidence[better] = confidence[better]
            target_labels = labels[tile_y:tile_y2, tile_x:tile_x2]
            target_labels[better] = predicted[better]
            target_flood_probability = best_flood_probability[tile_y:tile_y2, tile_x:tile_x2]
            target_flood_probability[better] = flood_probability[better]
        classified_flood = (
            labels == self.water_class_id
            if self.flood_probability_threshold is None
            else best_flood_probability >= self.flood_probability_threshold
        )
        flooded_mask = (classified_flood & (region_mask != 0)).astype(np.uint8)
        ratio = float(flooded_mask.sum()) / roi_pixels
        contours, _ = cv2.findContours(flooded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            if cv2.contourArea(contour) < 2:
                continue
            epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(simplified) < 3:
                continue
            outlines.append([
                [round(float(x + x1) / max(1, width - 1), 5), round(float(y + y1) / max(1, height - 1), 5)]
                for x, y in simplified[:128]
            ])
        prediction = {
            "region_id": str(region.get("id") or "road-surface"),
            "flooded_road_area_ratio": round(ratio, 6),
            "outline_polygons": outlines,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "class_label": "visible_water" if self.is_binary_visible_water else "flooded_road",
            "water_evidence_type": "visible_water" if self.is_binary_visible_water else "flooded_road",
            "model_output_scope": self.output_scope,
            "area_measurement": "fraction_of_marked_image_region",
            "water_depth_estimated": False,
            "analysis_tile_count": len(windows),
            "analysis_overlap_pixels": tile_overlap,
            "flood_probability_threshold": self.flood_probability_threshold,
        }
        if include_mask:
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = flooded_mask
            prediction["flooded_road_mask"] = full_mask
        if include_probabilities:
            full_probability = np.zeros((height, width), dtype=np.float32)
            full_probability[y1:y2, x1:x2] = best_flood_probability
            prediction["flooded_road_probability"] = full_probability
            if self.is_binary_visible_water:
                prediction["visible_water_probability"] = full_probability
        return prediction
