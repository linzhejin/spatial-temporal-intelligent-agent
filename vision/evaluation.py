"""Small, dependency-free evaluator for VisDrone-style detection annotations."""
from __future__ import annotations

from math import isfinite


VISDRONE_CATEGORY_BY_LABEL = {
    "pedestrian": 1,
    "people": 2,
    "bicycle": 3,
    "car": 4,
    "van": 5,
    "truck": 6,
    "tricycle": 7,
    "awning-tricycle": 8,
    "bus": 9,
    "motor": 10,
    "others": 11,
}
PEDESTRIAN_CATEGORIES = {1, 2}
VEHICLE_CATEGORIES = set(range(3, 11))
SUPPORTED_CATEGORIES = PEDESTRIAN_CATEGORIES | VEHICLE_CATEGORIES
_GROUPS = {
    "pedestrian": PEDESTRIAN_CATEGORIES,
    "vehicle": VEHICLE_CATEGORIES,
}


def _prediction_group(label: object) -> str | None:
    category = VISDRONE_CATEGORY_BY_LABEL.get(label)
    if category in PEDESTRIAN_CATEGORIES:
        return "pedestrian"
    if category in VEHICLE_CATEGORIES:
        return "vehicle"
    return None


def _valid_box(box: object) -> list[float] | None:
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not isfinite(value) for value in box)):
        return None
    normalized = [float(value) for value in box]
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    return normalized


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _parse_annotations(text: str) -> tuple[dict[int, list[list[float]]], list[list[float]]]:
    valid: dict[int, list[list[float]]] = {}
    ignored: list[list[float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        values = line.split(",")
        if len(values) != 8:
            raise ValueError(f"annotation row {line_number} must contain 8 comma-separated fields")
        try:
            x, y, width, height = (float(value.strip()) for value in values[:4])
            score = float(values[4].strip())
            category = int(values[5].strip())
        except (TypeError, ValueError) as error:
            raise ValueError(f"annotation row {line_number} contains invalid values") from error
        if not all(isfinite(value) for value in (x, y, width, height, score)) or width <= 0 or height <= 0:
            raise ValueError(f"annotation row {line_number} contains invalid geometry")
        box = [x, y, x + width, y + height]
        if score <= 0 or category not in SUPPORTED_CATEGORIES:
            ignored.append(box)
        else:
            valid.setdefault(category, []).append(box)
    return valid, ignored


def evaluate_visdrone_frame(
    predictions: list[dict], annotation_text: str, *,
    confidence_threshold: float = 0.369, iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Count TP/FP/FN at one confidence and IoU operating point.

    Categories 1-2 are pedestrian/person; categories 3-10 are road users.
    Category 11 (``others``), category 0, and score-0 boxes are ignored regions.
    This deliberately reports precision/recall counts, not mAP or campus accuracy.
    """
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("evaluation thresholds must be within valid ranges")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")

    ground_truth, ignored_boxes = _parse_annotations(annotation_text)
    matched = {category: set() for category in ground_truth}
    counts = {
        group: {"tp": 0, "fp": 0, "fn": 0, "ignored": 0}
        for group in _GROUPS
    }
    valid_predictions = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        label = prediction.get("label")
        category = VISDRONE_CATEGORY_BY_LABEL.get(label)
        if category not in SUPPORTED_CATEGORIES:
            continue
        confidence = prediction.get("confidence")
        box = prediction.get("box")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not isfinite(confidence) or confidence < confidence_threshold
                or not isinstance(box, (tuple, list)) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not isfinite(value) for value in box)):
            continue
        normalized_box = [float(value) for value in box]
        if normalized_box[2] <= normalized_box[0] or normalized_box[3] <= normalized_box[1]:
            continue
        valid_predictions.append((float(confidence), category, normalized_box))

    for _confidence, category, box in sorted(valid_predictions, key=lambda row: row[0], reverse=True):
        group = "pedestrian" if category in PEDESTRIAN_CATEGORIES else "vehicle"
        category_boxes = ground_truth.get(category, [])
        best_index, best_overlap = None, 0.0
        for index, target in enumerate(category_boxes):
            if index in matched.get(category, set()):
                continue
            overlap = _iou(box, target)
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is not None and best_overlap >= iou_threshold:
            matched[category].add(best_index)
            counts[group]["tp"] += 1
            continue
        if any(_iou(box, ignored) >= iou_threshold for ignored in ignored_boxes):
            counts[group]["ignored"] += 1
        else:
            counts[group]["fp"] += 1

    for category, boxes in ground_truth.items():
        group = "pedestrian" if category in PEDESTRIAN_CATEGORIES else "vehicle"
        counts[group]["fn"] += len(boxes) - len(matched.get(category, set()))
    return counts


def evaluate_grouped_frame(
    predictions: list[dict], annotations: list[dict], *,
    confidence_threshold: float = 0.369, iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Evaluate route-relevant pedestrian/vehicle groups on one campus frame.

    Ground truth labels are ``pedestrian``, ``vehicle``, or ``ignore``. A
    detector's detailed road-user class is deliberately collapsed to vehicle,
    because downstream congestion logic uses road-user presence and tracks.
    """
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("evaluation thresholds must be within valid ranges")
    if not isinstance(predictions, list) or not isinstance(annotations, list):
        raise ValueError("predictions and annotations must be lists")

    ground_truth = {group: [] for group in _GROUPS}
    ignored_boxes = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError(f"annotation {index} must be an object")
        label = annotation.get("label")
        box = _valid_box(annotation.get("box"))
        if label not in {"pedestrian", "vehicle", "ignore"} or box is None:
            raise ValueError(f"annotation {index} has an invalid label or box")
        if label == "ignore":
            ignored_boxes.append(box)
        else:
            ground_truth[label].append(box)

    matched = {group: set() for group in _GROUPS}
    counts = {
        group: {"tp": 0, "fp": 0, "fn": 0, "ignored": 0}
        for group in _GROUPS
    }
    usable_predictions = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        group = _prediction_group(prediction.get("label"))
        confidence = prediction.get("confidence")
        box = _valid_box(prediction.get("box"))
        if (group is None or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float)) or not isfinite(confidence)
                or confidence < confidence_threshold or box is None):
            continue
        usable_predictions.append((float(confidence), group, box))

    for _confidence, group, box in sorted(usable_predictions, key=lambda row: row[0], reverse=True):
        best_index, best_overlap = None, 0.0
        for index, target in enumerate(ground_truth[group]):
            if index in matched[group]:
                continue
            overlap = _iou(box, target)
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is not None and best_overlap >= iou_threshold:
            matched[group].add(best_index)
            counts[group]["tp"] += 1
        elif any(_iou(box, ignored) >= iou_threshold for ignored in ignored_boxes):
            counts[group]["ignored"] += 1
        else:
            counts[group]["fp"] += 1

    for group, boxes in ground_truth.items():
        counts[group]["fn"] = len(boxes) - len(matched[group])
    return counts


def summarize_detection_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    """Add precision, recall, and F1 without hiding raw integer counts."""
    summary = {}
    for group, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary[group] = {
            **values,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return summary
