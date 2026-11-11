"""Optional ONNX scene and flood-road models for human-reviewed evidence."""
from __future__ import annotations

import math
import os
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from vision.tiling import tile_windows


logger = logging.getLogger(__name__)


AIDER_CLASSES = (
    # TakuNet's AIDER loader follows ImageFolder's sorted directory order.
    "collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident",
)
FLOODNET_CLASSES = (
    "background", "flooded_building", "non_flooded_building", "flooded_road",
    "non_flooded_road", "water", "tree", "vehicle", "pool", "grass",
)
# Bound the evidence stored for one automatic candidate.  If a component has
# more holes than this representation can carry, clients must draw its border
# only rather than falsely filling in the missing holes.
MAX_AUTOMATIC_FLOOD_HOLES = 128
MAX_AUTOMATIC_FLOOD_HOLE_POINTS = 4096
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _onnx_contract(model_path, session, *, expected_classes: int | tuple[int, ...] | None = None,
                   expected_output_rank: int | None = None, class_axis: int = -1):
    if session is None:
        if not model_path or not Path(model_path).is_file():
            raise ValueError("模型权重文件不存在")
        import config
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


class DoodleverseFloodSegmenter:
    """Official FloodNet v7 probabilities clipped to marked road regions.

    The default path runs one image in a short-lived child interpreter. Passing a
    session is intended for parity tests and controlled offline evaluation.
    Frames passed to this adapter are RGB, matching the official model input.
    """

    INPUT_HEIGHT = 768
    INPUT_WIDTH = 1024
    CLASS_COUNT = 10
    FLOODED_ROAD_CLASS = 3
    PREPROCESSING = "doodleverse_rgb_resize_uint8_global_standardize_v1"
    model_id = "doodleverse-floodnet-resunet-v7"
    classes = list(FLOODNET_CLASSES)
    output_scope = "flooded_road_class"

    def __init__(self, model_path: str, *, session=None,
                 model_version: str | None = None,
                 probability_threshold: float = 0.001,
                 min_area_ratio: float = 0.08,
                 single_image_trigger_threshold: float = 0.08,
                 runtime_python: str | None = None,
                 timeout_seconds: float = 120.0):
        self.model_path = str(model_path or "")
        self.model_version = model_version or os.getenv(
            "VISION_FLOOD_REVISION", "unverified",
        )
        self.probability_threshold = self._validate_threshold(
            probability_threshold, "积水类别概率阈值", allow_zero=True,
        )
        self.min_area_ratio = self._validate_threshold(
            min_area_ratio, "积水面积阈值", allow_zero=False,
        )
        self.single_image_trigger_threshold = self._validate_threshold(
            single_image_trigger_threshold, "单张图片候选阈值", allow_zero=False,
        )
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 120:
            raise ValueError("单张图片推理时限必须大于 0 且不超过 120 秒")
        self.runtime_python = runtime_python or os.getenv("VISION_FLOOD_RUNTIME_PYTHON", "")
        self.session = session
        self.input_name = None
        self.output_name = None
        if session is not None:
            self._validate_session(session)
        elif not self.model_path or not Path(self.model_path).is_file():
            raise ValueError("Doodleverse v7 权重文件不存在")

    @staticmethod
    def _validate_threshold(value, label, *, allow_zero):
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}必须是 0 到 1 之间的数") from error
        lower_ok = number >= 0 if allow_zero else number > 0
        if not math.isfinite(number) or not lower_ok or number > 1:
            raise ValueError(f"{label}必须是 0 到 1 之间的数")
        return number

    def _validate_session(self, session):
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Doodleverse v7 必须有单个图像输入和概率输出")
        input_shape = list(inputs[0].shape)
        output_shape = list(outputs[0].shape)
        if (len(input_shape) != 4 or input_shape[1:] != [
                self.INPUT_HEIGHT, self.INPUT_WIDTH, 3,
        ]):
            raise ValueError("Doodleverse v7 输入必须为 NHWC [N,768,1024,3]")
        if (len(output_shape) != 4 or output_shape[1:] != [
                self.INPUT_HEIGHT, self.INPUT_WIDTH, self.CLASS_COUNT,
        ]):
            raise ValueError("Doodleverse v7 输出必须为 NHWC [N,768,1024,10]")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def _load_session(self):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.getenv("VISION_FLOOD_CPU_THREADS", "2")))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            self.model_path, sess_options=options, providers=["CPUExecutionProvider"],
        )
        self._validate_session(session)
        return session

    @staticmethod
    def _prepare_rgb(frame_rgb):
        from vision.doodleverse_preprocessing import prepare_doodleverse_rgb

        return prepare_doodleverse_rgb(frame_rgb)

    def _predict_output(self, frame_rgb):
        model_input = self._prepare_rgb(frame_rgb)
        if self.session is None:
            # Large source-image resize buffers must be released before the
            # ONNX session reserves its own inference workspace.
            import gc
            gc.collect()
            if sys.platform.startswith("linux"):
                try:
                    import ctypes
                    ctypes.CDLL(None).malloc_trim(0)
                except (AttributeError, OSError):
                    pass
            self.session = self._load_session()
        try:
            raw = self.session.run([self.output_name], {self.input_name: model_input})
        except Exception as error:
            raise RuntimeError(f"Doodleverse v7 推理失败：{error}") from error
        return raw[0]

    def _predict_probabilities(self, frame_rgb):
        from vision.doodleverse_preprocessing import extract_flooded_road_probability

        return extract_flooded_road_probability(self._predict_output(frame_rgb))

    def _predict_automatic_maps(self, frame_rgb):
        """Return the flooded-road class winner and its raw probability plane."""
        import numpy as np

        from vision.doodleverse_preprocessing import extract_flooded_road_probability

        output = self._predict_output(frame_rgb)
        probability = extract_flooded_road_probability(output)
        scores = output[0]
        winner = (probability >= np.max(scores, axis=-1))
        # Match argmax's first-class tie behaviour without allocating an int64
        # class map the size of the image.
        for class_id in range(self.FLOODED_ROAD_CLASS):
            winner &= probability > scores[:, :, class_id]
        winner &= probability >= self.probability_threshold
        return probability.copy(), winner

    def _automatic_predictions_from_maps(self, probability, winner):
        """Create bounded image-space proposals, not geographic road matches."""
        import cv2
        import numpy as np

        if probability.shape != winner.shape or probability.ndim != 2:
            raise ValueError("积水候选概率图尺寸无效")
        height, width = winner.shape
        winner_mask = winner.astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            winner_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        if hierarchy is None:
            return {}
        components = [(index, contour) for index, contour in enumerate(contours)
                      if hierarchy[0][index][3] == -1]
        components.sort(key=lambda item: (-cv2.contourArea(item[1]),
                                           cv2.boundingRect(item[1])[1], cv2.boundingRect(item[1])[0]))
        truncated = sum(cv2.contourArea(item) >= 2 for _, item in components) > 24
        _, component_labels = cv2.connectedComponents(winner_mask, connectivity=8)
        predictions = {}
        for contour_index, contour in components:
            if cv2.contourArea(contour) < 2:
                continue
            epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            while len(simplified) > 128:
                epsilon *= 1.5
                simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(simplified) < 3:
                continue
            polygon = [[round(float(x) / max(1, width - 1), 5),
                        round(float(y) / max(1, height - 1), 5)]
                       for x, y in simplified]
            holes = []
            hole_points = 0
            holes_complete = True
            child_index = int(hierarchy[0][contour_index][2])
            while child_index != -1:
                points = contours[child_index].reshape(-1, 2)
                hole_points += len(points)
                if (len(holes) >= MAX_AUTOMATIC_FLOOD_HOLES or len(points) < 3
                        or hole_points > MAX_AUTOMATIC_FLOOD_HOLE_POINTS):
                    holes = []
                    holes_complete = False
                    break
                holes.append([[round(float(px) / max(1, width - 1), 5),
                               round(float(py) / max(1, height - 1), 5)]
                              for px, py in points])
                child_index = int(hierarchy[0][child_index][0])
            x, y, w, h = cv2.boundingRect(contour)
            anchor_x, anchor_y = contour[0, 0]
            component_id = component_labels[anchor_y, anchor_x]
            local_mask = component_labels[y:y + h, x:x + w] == component_id
            score = float(probability[y:y + h, x:x + w][local_mask].mean())
            region_id = f"auto-flood-{len(predictions) + 1}"
            predictions[region_id] = {
                "region_id": region_id,
                "region_origin": "automatic_model",
                "region_polygon": polygon,
                "outline_polygons": [polygon],
                "outline_holes": holes,
                "outline_holes_complete": holes_complete,
                "automatic_flood": True,
                "automatic_candidates_truncated": truncated,
                "model_score": round(score, 6),
                "model_id": self.model_id,
                "model_version": self.model_version,
                "water_evidence_type": "flooded_road",
                "water_depth_estimated": False,
            }
            if len(predictions) >= 24:
                break
        return predictions

    def predict_automatic(self, frame_rgb, *, cancel_check=None):
        if cancel_check and cancel_check():
            raise RuntimeError("积水检查已取消")
        if self.session is not None:
            probability, winner = self._predict_automatic_maps(frame_rgb)
            return self._automatic_predictions_from_maps(probability, winner)
        return self._run_isolated(frame_rgb, [], automatic=True, cancel_check=cancel_check)

    def _predictions_from_probability(self, frame_rgb, regions, class_probability):
        import cv2
        import numpy as np
        from skimage.transform import resize

        height, width = frame_rgb.shape[:2]
        probability = resize(
            class_probability, (height, width), preserve_range=False, clip=True,
        ).astype(np.float32)
        if (not np.isfinite(probability).all() or float(probability.min()) < -1e-6
                or float(probability.max()) > 1.000001):
            raise ValueError("Doodleverse v7 原图尺寸概率图无效")
        predictions = {}
        for region in regions or []:
            if not isinstance(region, dict) or region.get("kind") != "road_surface":
                continue
            polygon = region.get("polygon")
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            try:
                points = np.asarray([
                    [round(float(point[0]) * (width - 1)),
                     round(float(point[1]) * (height - 1))]
                    for point in polygon
                ], dtype=np.int32)
            except (IndexError, TypeError, ValueError):
                continue
            roi = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi, [points], 1)
            roi_pixels = int(roi.sum())
            if not roi_pixels:
                continue
            flood_mask = ((probability >= self.probability_threshold) & (roi != 0)).astype(np.uint8)
            ratio = float(flood_mask.sum()) / roi_pixels
            contours, _ = cv2.findContours(
                flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            outlines = []
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
                if cv2.contourArea(contour) < 2:
                    continue
                epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
                simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                if len(simplified) < 3:
                    continue
                outlines.append([
                    [round(float(x) / max(1, width - 1), 5),
                     round(float(y) / max(1, height - 1), 5)]
                    for x, y in simplified[:128]
                ])
            region_id = str(region.get("id") or "road-surface")
            predictions[region_id] = {
                "region_id": region_id,
                "flooded_road_area_ratio": round(ratio, 6),
                "outline_polygons": outlines,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "class_label": "flooded_road",
                "water_evidence_type": "flooded_road",
                "model_output_scope": self.output_scope,
                "area_measurement": "fraction_of_marked_image_region",
                "water_depth_estimated": False,
                "flood_probability_threshold": self.probability_threshold,
                "candidate_triggered": ratio >= self.single_image_trigger_threshold,
            }
        return predictions

    def predict_regions(self, frame_rgb, regions, *, cancel_check=None):
        import numpy as np

        if cancel_check and cancel_check():
            raise RuntimeError("积水检查已取消")
        road_regions = [region for region in regions or []
                        if isinstance(region, dict) and region.get("kind") == "road_surface"]
        if not road_regions:
            return {}
        if self.session is not None:
            probabilities = self._predict_probabilities(frame_rgb)
            return self._predictions_from_probability(frame_rgb, road_regions, probabilities)
        return self._run_isolated(frame_rgb, road_regions, automatic=False, cancel_check=cancel_check)

    def _run_isolated(self, frame_rgb, road_regions, *, automatic, cancel_check):
        import numpy as np

        request_root = Path(tempfile.mkdtemp(prefix="whu-flood-"))
        process = None
        try:
            image_path = request_root / "image.npy"
            np.save(image_path, np.asarray(frame_rgb), allow_pickle=False)
            request_path = request_root / "request.json"
            response_path = request_root / "response.json"
            stderr_path = request_root / "stderr.log"
            request_path.write_text(json.dumps({
                "image_path": str(image_path),
                "response_path": str(response_path),
                "model_path": self.model_path,
                "model_version": self.model_version,
                "probability_threshold": self.probability_threshold,
                "min_area_ratio": self.min_area_ratio,
                "single_image_trigger_threshold": self.single_image_trigger_threshold,
                "regions": road_regions,
                "automatic": automatic,
            }, separators=(",", ":")), encoding="utf-8")
            project_root = Path(__file__).resolve().parents[1]
            python = self.runtime_python or sys.executable
            environment = {key: value for key, value in os.environ.items()
                           if key in {"PATH", "SYSTEMROOT", "WINDIR", "HOME", "TMPDIR",
                                      "LD_LIBRARY_PATH", "OMP_NUM_THREADS"}}
            environment["PYTHONPATH"] = str(project_root)
            with stderr_path.open("wb") as stderr_file:
                process = subprocess.Popen(
                    [python, "-m", "vision.flood_inference_child", str(request_path)],
                    cwd=str(project_root), env=environment,
                    stdout=subprocess.DEVNULL, stderr=stderr_file,
                )
            deadline = time.monotonic() + self.timeout_seconds
            while process.poll() is None:
                if cancel_check and cancel_check():
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3.5)
                    raise RuntimeError("积水检查已取消")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3.5)
                    raise TimeoutError("积水图片分析超过 120 秒")
                try:
                    process.wait(timeout=min(0.25, remaining))
                except subprocess.TimeoutExpired:
                    pass
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            if process.returncode != 0:
                raise RuntimeError(f"积水分割子进程失败：{stderr[-1000:]}")
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if response.get("status") != "ok" or not isinstance(response.get("predictions"), dict):
                raise RuntimeError("积水分割子进程未返回有效结果")
            return response["predictions"]
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.5)
            import shutil
            for attempt in range(4):
                try:
                    shutil.rmtree(request_root)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    if attempt == 3:
                        logger.exception("Could not remove temporary flood inference files")
                    else:
                        time.sleep(0.05 * (attempt + 1))
