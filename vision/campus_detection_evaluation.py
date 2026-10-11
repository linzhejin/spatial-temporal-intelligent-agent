"""Validation for flight-disjoint campus pedestrian and vehicle test labels."""
from __future__ import annotations

import math
from pathlib import PurePosixPath, PureWindowsPath


_SPLITS = ("train", "validation", "test")
_LABELS = {"pedestrian", "vehicle", "ignore"}


def _relative_path(value: object, *, sample_id: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    if (not normalized or posix.is_absolute() or PureWindowsPath(normalized).is_absolute()
            or any(part in {"", ".", ".."} for part in posix.parts)):
        raise ValueError(f"sample {sample_id} image must use a safe relative path")
    return posix.as_posix()


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _box(value: object, *, sample_id: str, object_index: int,
         width: int, height: int) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(float(item)) for item in value)):
        raise ValueError(f"sample {sample_id} object {object_index} has an invalid box")
    x1, y1, x2, y2 = (float(item) for item in value)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"sample {sample_id} object {object_index} box must stay inside "
            "the declared image dimensions"
        )
    return [x1, y1, x2, y2]


def validate_campus_detection_manifest(manifest: object) -> dict:
    """Validate manual frame labels and flight-disjoint train/validation/test groups."""
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("task") != "campus_object_detection"):
        raise ValueError("detection manifest must use schema v1 and task campus_object_detection")
    if manifest.get("split_unit") != "flight_id":
        raise ValueError("detection manifest split_unit must be flight_id")
    if manifest.get("label_source") != "manual":
        raise ValueError("detection manifest label_source must be manual")
    if manifest.get("annotation_pending") is True:
        raise ValueError("detection manifest annotation_pending must be removed after manual review")
    if not str(manifest.get("dataset") or "").strip():
        raise ValueError("detection manifest dataset name is required")

    groups = manifest.get("split_groups")
    if not isinstance(groups, dict):
        raise ValueError("split_groups is required")
    normalized_groups = {}
    for split in _SPLITS:
        values = groups.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split_groups.{split} must be a non-empty list")
        normalized = [str(value).strip() for value in values]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError(f"split_groups.{split} contains empty or duplicate flight IDs")
        normalized_groups[split] = normalized
    all_groups = [item for values in normalized_groups.values() for item in values]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("train, validation, and test flight groups must not overlap")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("detection manifest samples must be a non-empty list")
    sample_ids, image_paths, observed_groups = set(), set(), set()
    normalized_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        sample_id = str(sample.get("sample_id") or "").strip()
        group_id = str(sample.get("group_id") or "").strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"sample {index} requires a unique sample_id")
        if group_id not in normalized_groups["test"]:
            raise ValueError(f"sample {sample_id} must belong to a declared test flight")
        image = _relative_path(sample.get("image"), sample_id=sample_id)
        if image in image_paths:
            raise ValueError("detection manifest contains a duplicate image path")
        width = _positive_integer(sample.get("width"), f"sample {sample_id} width")
        height = _positive_integer(sample.get("height"), f"sample {sample_id} height")
        objects = sample.get("objects")
        if not isinstance(objects, list):
            raise ValueError(f"sample {sample_id} objects must be a list")
        normalized_objects = []
        for object_index, item in enumerate(objects):
            if not isinstance(item, dict) or item.get("label") not in _LABELS:
                raise ValueError(
                    f"sample {sample_id} object {object_index} label must be pedestrian, vehicle, or ignore"
                )
            normalized_objects.append({
                **item,
                "label": item["label"],
                "box": _box(
                    item.get("box"), sample_id=sample_id, object_index=object_index,
                    width=width, height=height,
                ),
            })
        sample_ids.add(sample_id)
        image_paths.add(image)
        observed_groups.add(group_id)
        normalized_samples.append({
            **sample, "sample_id": sample_id, "group_id": group_id,
            "image": image, "width": width, "height": height,
            "objects": normalized_objects,
        })
    if observed_groups != set(normalized_groups["test"]):
        raise ValueError("test groups must all have at least one image")

    result = dict(manifest)
    result["split_groups"] = normalized_groups
    result["samples"] = normalized_samples
    return result
