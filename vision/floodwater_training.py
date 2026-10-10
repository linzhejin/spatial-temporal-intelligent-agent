"""Manifest preparation for source-video-safe visible-water experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath


def _metadata_path(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
        raise ValueError(f"{field} must stay inside the dataset")
    return posix.as_posix()


def _canonical_split(value: str | None) -> str | None:
    return "validation" if value == "val" else value


def build_floodwater_manifest(chunks_csv, samples_csv, *, sample_period_frames: int = 25) -> dict:
    """Select sparse train/validation masks while preserving source-video splits.

    Test-source frames are intentionally omitted. The separate manually
    annotated 700-image set remains external evaluation only.
    """
    if (isinstance(sample_period_frames, bool) or not isinstance(sample_period_frames, int)
            or sample_period_frames < 1):
        raise ValueError("sample_period_frames must be a positive integer")

    chunks_path = Path(chunks_csv)
    samples_path = Path(samples_csv)
    with chunks_path.open("r", encoding="utf-8-sig", newline="") as source:
        chunks = list(csv.DictReader(source))
    with samples_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not chunks or not rows:
        raise ValueError("Floodwater chunk and sample metadata must not be empty")

    by_chunk = {}
    source_splits = defaultdict(set)
    groups = {split: set() for split in ("train", "validation", "test")}
    for index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        source_video_id = chunk.get("source_video_id")
        split = _canonical_split(chunk.get("split"))
        if not chunk_id or not source_video_id or split not in groups:
            raise ValueError(f"invalid chunk metadata row {index}")
        if chunk_id in by_chunk:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        video_path = _metadata_path(chunk.get("video_path"), f"chunk {chunk_id} video_path")
        mask_directory = _metadata_path(chunk.get("mask_directory"), f"chunk {chunk_id} mask_directory")
        by_chunk[chunk_id] = {
            "source_video_id": source_video_id,
            "split": split,
            "video_path": video_path,
            "mask_directory": mask_directory,
        }
        source_splits[source_video_id].add(split)
        groups[split].add(source_video_id)
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("source video appears in multiple splits")

    selected = []
    seen_sample_ids = set()
    for index, row in enumerate(rows):
        chunk_id = row.get("chunk_id")
        chunk = by_chunk.get(chunk_id)
        if chunk is None:
            raise ValueError(f"sample row {index} references unknown chunk_id")
        if (row.get("source_video_id") != chunk["source_video_id"]
                or _canonical_split(row.get("split")) != chunk["split"]):
            raise ValueError(f"sample row {index} conflicts with its chunk metadata")
        try:
            frame_index = int(row.get("chunk_frame_index", ""))
            time_seconds = float(row.get("chunk_time_seconds", ""))
        except (TypeError, ValueError) as error:
            raise ValueError(f"sample row {index} has invalid frame/time") from error
        if frame_index < 0 or not time_seconds >= 0:
            raise ValueError(f"sample row {index} has invalid frame/time")
        sample_id = row.get("sample_id")
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError(f"missing or duplicate sample_id in row {index}")
        seen_sample_ids.add(sample_id)
        if frame_index % sample_period_frames or chunk["split"] == "test":
            continue
        mask_path = _metadata_path(row.get("mask_path"), f"sample {sample_id} mask_path")
        video_path = _metadata_path(row.get("video_path"), f"sample {sample_id} video_path")
        if mask_path != f"{chunk['mask_directory']}/{frame_index:05d}.png":
            raise ValueError(f"sample {sample_id} mask_path does not match chunk metadata")
        if video_path != chunk["video_path"]:
            raise ValueError(f"sample {sample_id} video_path does not match chunk metadata")
        selected.append({
            "sample_id": sample_id,
            "source_video_id": chunk["source_video_id"],
            "group_id": chunk["source_video_id"],
            "chunk_id": chunk_id,
            "split": chunk["split"],
            "chunk_frame_index": frame_index,
            "chunk_time_seconds": time_seconds,
            "image": f"derived/frames/{chunk_id}/{frame_index:05d}.jpg",
            "mask": mask_path,
            "video": video_path,
            "label_type": row.get("label_type") or "unknown",
        })
    if not any(sample["split"] == "train" for sample in selected):
        raise ValueError("no training samples selected")
    if not any(sample["split"] == "validation" for sample in selected):
        raise ValueError("no validation samples selected")

    ordered_groups = {split: sorted(values) for split, values in groups.items()}
    return {
        "schema_version": 1,
        "task": "visible_water_segmentation",
        "dataset": "Floodwater Dataset v1.0.0",
        "source": "https://github.com/decide-ugent/floodwater-dataset",
        "license": "GPL-3.0-only",
        "split_unit": "source_video_id",
        "sampling": {"period_frames": sample_period_frames, "source_fps": 25},
        "label_values": {"0": "background_or_non_water", "255": "visible_water"},
        "split_groups": ordered_groups,
        "samples": selected,
        "limitations": [
            "The source masks are SAM2-assisted pseudo-labels, not all manual ground truth.",
            "The manually annotated 700-image set is not included in these training samples.",
            "The archive does not document whether the manual evaluation images are source-video-disjoint.",
            "Flood-affected Belgian UAV imagery does not establish shallow-water performance on Wuhan campus roads.",
        ],
    }


def validate_visible_water_manifest(manifest: object) -> dict:
    """Validate video-disjoint train/validation groups and exclude test frames."""
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("task") != "visible_water_segmentation"
            or manifest.get("split_unit") != "source_video_id"):
        raise ValueError("visible-water manifest must use schema v1 and source_video_id splits")
    groups = manifest.get("split_groups")
    if not isinstance(groups, dict):
        raise ValueError("visible-water split_groups are required")
    normalized_groups = {}
    for split in ("train", "validation", "test"):
        values = groups.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split_groups.{split} must be a non-empty list")
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError(f"split_groups.{split} contains an empty or duplicate source video")
        normalized_groups[split] = normalized
    all_groups = [group for values in normalized_groups.values() for group in values]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("source video appears in multiple splits")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("visible-water samples must be a non-empty list")
    normalized_samples = []
    sample_ids, image_paths, mask_paths = set(), set(), set()
    present_splits = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        split = str(sample.get("split") or "").strip()
        if split == "test":
            raise ValueError("training manifest must not contain test split samples")
        group_id = str(sample.get("group_id") or sample.get("source_video_id") or "").strip()
        source_video_id = str(sample.get("source_video_id") or group_id).strip()
        sample_id = str(sample.get("sample_id") or "").strip()
        if split not in ("train", "validation") or not group_id or not sample_id:
            raise ValueError(f"sample {index} requires a sample ID, train/validation split, and group")
        if group_id != source_video_id or group_id not in normalized_groups[split]:
            raise ValueError(f"sample {sample_id} source-video group conflicts with its split")
        image = _metadata_path(sample.get("image"), f"sample {sample_id} image")
        mask = _metadata_path(sample.get("mask"), f"sample {sample_id} mask")
        if sample_id in sample_ids or image in image_paths or mask in mask_paths:
            raise ValueError("visible-water manifest contains duplicate sample, image, or mask IDs")
        sample_ids.add(sample_id)
        image_paths.add(image)
        mask_paths.add(mask)
        present_splits.add(split)
        normalized_samples.append({**sample, "sample_id": sample_id, "group_id": group_id,
                                  "source_video_id": source_video_id, "split": split,
                                  "image": image, "mask": mask})
    if present_splits != {"train", "validation"}:
        raise ValueError("visible-water samples must include train and validation only")
    return {**manifest, "split_groups": normalized_groups, "samples": normalized_samples}


def validate_visible_water_training_manifest(manifest: object) -> dict:
    """Validate public-video or flight-disjoint binary-water training data.

    The test split is declared for provenance and leakage checks but test frames
    are forbidden in this training manifest. Campus samples must be manually
    labeled, road-scoped binary masks from a separate flight-level split.
    """
    if isinstance(manifest, dict) and manifest.get("task") == "visible_water_segmentation":
        return validate_visible_water_manifest(manifest)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("task") != "visible_water_road_segmentation"
            or manifest.get("split_unit") != "flight_id"):
        raise ValueError(
            "campus water training manifest must use schema v1 and flight_id splits",
        )
    if not str(manifest.get("dataset") or "").strip():
        raise ValueError("campus water dataset name is required")
    if manifest.get("mask_encoding") != "binary_0_255":
        raise ValueError("campus training masks must use binary_0_255 encoding")
    if manifest.get("label_source") != "manual":
        raise ValueError("campus visible-water training requires manually labeled masks")

    groups = manifest.get("split_groups")
    if not isinstance(groups, dict):
        raise ValueError("campus water split_groups are required")
    normalized_groups = {}
    for split in ("train", "validation", "test"):
        values = groups.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split_groups.{split} must contain at least one flight")
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError(f"split_groups.{split} contains an empty or duplicate flight")
        normalized_groups[split] = normalized
    all_groups = [group for values in normalized_groups.values() for group in values]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("flight appears in multiple splits")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("campus water training samples must not be empty")
    sample_ids, image_paths, mask_paths = set(), set(), set()
    present_groups = {"train": set(), "validation": set()}
    normalized_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        sample_id = str(sample.get("sample_id") or "").strip()
        split = str(sample.get("split") or "").strip()
        group_id = str(sample.get("group_id") or "").strip()
        if split == "test":
            raise ValueError("test samples must not enter model training")
        if split not in ("train", "validation") or not sample_id or not group_id:
            raise ValueError(f"sample {index} requires a sample ID, split, and flight group")
        if group_id not in normalized_groups[split]:
            raise ValueError(f"sample {sample_id} conflicts with its flight split")
        image = _metadata_path(sample.get("image"), f"sample {sample_id} image")
        mask = _metadata_path(sample.get("mask"), f"sample {sample_id} mask")
        polygon = sample.get("road_surface_polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"sample {sample_id} requires a road_surface_polygon")
        for point in polygon:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) or not 0 <= value <= 1
                           for value in point)):
                raise ValueError(f"sample {sample_id} has an invalid normalized road polygon")
        if sample_id in sample_ids or image in image_paths or mask in mask_paths:
            raise ValueError("campus training manifest contains duplicate sample or file paths")
        sample_ids.add(sample_id)
        image_paths.add(image)
        mask_paths.add(mask)
        present_groups[split].add(group_id)
        normalized_samples.append({**sample, "sample_id": sample_id, "split": split,
                                   "group_id": group_id, "image": image, "mask": mask})
    for split in ("train", "validation"):
        if present_groups[split] != set(normalized_groups[split]):
            raise ValueError(f"every declared {split} flight must have at least one sample")
    return {**manifest, "split_groups": normalized_groups, "samples": normalized_samples}


def validate_visible_water_training_files(dataset_root, manifest: dict) -> dict:
    """Preflight every train/validation image and mask before model startup."""
    from PIL import Image
    import numpy as np

    root = Path(dataset_root).resolve(strict=True)
    records = []
    seen_image_content = {}
    total_bytes = 0

    def _resolve_file(relative_path: str, sample_id: str, kind: str) -> Path:
        try:
            resolved = (root / relative_path).resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"sample {sample_id} {kind} is missing or resolves outside the dataset",
            ) from error
        if not resolved.is_file():
            raise ValueError(f"sample {sample_id} {kind} is not a file")
        return resolved

    def _sha256(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        return digest.hexdigest(), size

    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        image_path = _resolve_file(sample["image"], sample_id, "image")
        mask_path = _resolve_file(sample["mask"], sample_id, "mask")
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image_size = image.size
            with Image.open(mask_path) as mask_image:
                mask_format = mask_image.format
                mask = np.asarray(mask_image.convert("L"))
        except (OSError, ValueError) as error:
            raise ValueError(f"sample {sample_id} image or mask cannot be decoded") from error

        if mask_format != "PNG":
            raise ValueError(f"sample {sample_id} mask must be a PNG")
        if mask.ndim != 2 or mask.shape[::-1] != image_size:
            raise ValueError(f"sample {sample_id} image and mask dimensions differ")
        if not np.isin(np.unique(mask), (0, 255)).all():
            raise ValueError(f"sample {sample_id} mask must contain only 0 and 255")

        polygon = sample.get("road_surface_polygon")
        if polygon is not None:
            twice_area = sum(
                float(polygon[index][0]) * float(polygon[(index + 1) % len(polygon)][1])
                - float(polygon[(index + 1) % len(polygon)][0]) * float(polygon[index][1])
                for index in range(len(polygon))
            )
            if abs(twice_area) <= 1e-12:
                raise ValueError(f"sample {sample_id} road polygon has zero area")

        image_hash, image_bytes = _sha256(image_path)
        mask_hash, mask_bytes = _sha256(mask_path)
        previous = seen_image_content.get(image_hash)
        current_group = (sample["split"], sample["group_id"])
        if previous is not None and previous[0] != current_group[0]:
            raise ValueError("identical image content across train/validation splits")
        seen_image_content[image_hash] = current_group
        total_bytes += image_bytes + mask_bytes
        records.append({
            "sample_id": sample_id,
            "split": sample["split"],
            "group_id": sample["group_id"],
            "image_sha256": image_hash,
            "mask_sha256": mask_hash,
        })

    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return {
        "sample_count": len(records),
        "validated_bytes": total_bytes,
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
    }


class VisibleWaterDataset:
    """Read RGB frames and 0/255 masks, returning 0=background and 1=water tiles."""

    def __init__(self, root, samples: list[dict], *, tile_size: int = 256,
                 augment: bool = False, random_crop: bool = True,
                 water_crop_probability: float = 0.6):
        from PIL import Image

        if not isinstance(samples, list) or not samples:
            raise ValueError("visible-water samples must be a non-empty list")
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 32:
            raise ValueError("tile_size must be an integer of at least 32")
        if (isinstance(water_crop_probability, bool)
                or not isinstance(water_crop_probability, (int, float))
                or not 0 <= water_crop_probability <= 1):
            raise ValueError("water_crop_probability must be between 0 and 1")
        self.root = Path(root).resolve()
        self.tile_size = tile_size
        self.augment = bool(augment)
        self.random_crop = bool(random_crop)
        self.water_crop_probability = float(water_crop_probability)
        self.samples = []
        seen_ids = set()
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"sample {index} must be an object")
            sample_id = str(sample.get("sample_id") or f"sample-{index}")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate visible-water sample_id: {sample_id}")
            seen_ids.add(sample_id)
            image_relative = _metadata_path(sample.get("image"), f"sample {sample_id} image")
            mask_relative = _metadata_path(sample.get("mask"), f"sample {sample_id} mask")
            image_path = self._resolve(image_relative)
            mask_path = self._resolve(mask_relative)
            if not image_path.is_file() or not mask_path.is_file():
                raise ValueError(f"missing image or mask for sample {sample_id}")
            with Image.open(image_path) as image:
                image_size = image.size
            with Image.open(mask_path) as mask:
                mask_size = mask.size
            if image_size != mask_size:
                raise ValueError(f"image/mask dimensions differ for sample {sample_id}")
            self.samples.append({
                **sample,
                "sample_id": sample_id,
                "image": image_relative,
                "mask": mask_relative,
            })

    def _resolve(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("sample path resolves outside the dataset root") from error
        return path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        sample = self.samples[index]
        image_bgr = cv2.imread(str(self._resolve(sample["image"])), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"could not decode image {sample['image']}")
        with Image.open(self._resolve(sample["mask"])) as source:
            mask = np.asarray(source.convert("L"))
        values = np.unique(mask)
        if not np.isin(values, (0, 255)).all():
            raise ValueError(f"visible-water mask must contain only 0 and 255: {sample['mask']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        road_roi = None
        polygon = sample.get("road_surface_polygon")
        if polygon is not None:
            height, width = mask.shape
            points = np.asarray([
                [round(float(point[0]) * (width - 1)),
                 round(float(point[1]) * (height - 1))]
                for point in polygon
            ], dtype=np.int32)
            road_roi = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(road_roi, [points], 1)
            if not road_roi.any():
                raise ValueError(f"road_surface_polygon is empty for sample {sample['sample_id']}")
            mask = np.where(road_roi != 0, mask, 0).astype(np.uint8, copy=False)
        tile = self.tile_size
        height, width = mask.shape
        if height < tile or width < tile:
            pad_height, pad_width = max(0, tile - height), max(0, tile - width)
            image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0)
            if road_roi is not None:
                road_roi = np.pad(
                    road_roi, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0,
                )
            height, width = mask.shape
        if self.random_crop:
            positive = np.argwhere(mask == 255)
            if len(positive) and random.random() < self.water_crop_probability:
                center_y, center_x = positive[random.randrange(len(positive))]
                x = min(max(0, int(center_x) - tile // 2), width - tile)
                y = min(max(0, int(center_y) - tile // 2), height - tile)
            elif road_roi is not None:
                road_pixels = np.argwhere(road_roi != 0)
                center_y, center_x = road_pixels[random.randrange(len(road_pixels))]
                x = min(max(0, int(center_x) - tile // 2), width - tile)
                y = min(max(0, int(center_y) - tile // 2), height - tile)
            else:
                x = random.randint(0, width - tile)
                y = random.randint(0, height - tile)
        else:
            if road_roi is not None:
                road_y, road_x = np.where(road_roi != 0)
                center_x = int((road_x.min() + road_x.max()) / 2)
                center_y = int((road_y.min() + road_y.max()) / 2)
                x = min(max(0, center_x - tile // 2), width - tile)
                y = min(max(0, center_y - tile // 2), height - tile)
            else:
                x = max(0, (width - tile) // 2)
                y = max(0, (height - tile) // 2)
        image = image[y:y + tile, x:x + tile]
        mask = mask[y:y + tile, x:x + tile]
        if self.augment and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).float() / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        image_tensor = (image_tensor - mean) / std
        labels = torch.from_numpy((mask == 255).astype(np.int64, copy=False))
        return image_tensor, labels


def evaluate_visible_water_model(model, data_loader, *, device) -> dict:
    """Compute binary water pixel metrics from a two-logit segmentation model."""
    import torch

    matrix = torch.zeros((2, 2), dtype=torch.int64, device="cpu")
    model.eval()
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)
            logits = model(images)
            if logits.ndim != 4 or logits.shape[1] != 2 or logits.shape[0] != labels.shape[0]:
                raise ValueError("visible-water model must emit [N,2,H,W] logits")
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False,
                )
            if torch.any((labels < 0) | (labels > 1)):
                raise ValueError("visible-water labels must contain only class IDs 0 and 1")
            predicted = logits.argmax(dim=1)
            encoded = labels * 2 + predicted
            matrix += torch.bincount(encoded.cpu().reshape(-1), minlength=4).reshape(2, 2)
    result = {}
    for class_id, name in enumerate(("background", "visible_water")):
        tp = int(matrix[class_id, class_id].item())
        fp = int(matrix[:, class_id].sum().item()) - tp
        fn = int(matrix[class_id, :].sum().item()) - tp
        union = tp + fp + fn
        result[name] = {
            "class_id": class_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "iou": tp / union if union else None,
        }
    present = [row["iou"] for row in result.values() if row["iou"] is not None]
    result["mean_iou"] = sum(present) / len(present) if present else None
    return result


def train_visible_water_model(model, train_loader, validation_loader, *, device,
                              checkpoint_path, epochs: int = 2,
                              learning_rate: float = 3e-4,
                              positive_class_weight: float = 2.0,
                              progress_callback=None) -> dict:
    """Train a two-class model with validation-only checkpoint selection."""
    import torch
    from tqdm.auto import tqdm

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(positive_class_weight) or positive_class_weight < 1:
        raise ValueError("positive_class_weight must be at least 1 and finite")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(
        (1.0, positive_class_weight), dtype=torch.float32, device=device,
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_iou = -1.0
    best_epoch = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, labels in tqdm(train_loader, desc=f"water epoch {epoch}/{epochs}", leave=False):
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("visible-water training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if not batches:
            raise ValueError("visible-water training loader produced no batches")
        validation = evaluate_visible_water_model(model, validation_loader, device=device)
        water_iou = validation["visible_water"]["iou"]
        score = float(water_iou) if water_iou is not None else -1.0
        record = {
            "epoch": epoch,
            "mean_train_loss": total_loss / batches,
            "validation_visible_water_iou": water_iou,
            "validation_mean_iou": validation["mean_iou"],
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(dict(record))
        if score > best_iou:
            with tempfile.NamedTemporaryFile(prefix=target.stem + "-", suffix=".tmp",
                                             dir=target.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_visible_water_iou": water_iou,
                }, temporary_path)
                os.replace(temporary_path, target)
            finally:
                temporary_path.unlink(missing_ok=True)
            best_iou = score
            best_epoch = epoch
    return {
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_validation_visible_water_iou": None if best_iou < 0 else best_iou,
        "checkpoint_selection": "validation_visible_water_iou",
        "history": history,
        "checkpoint": str(target),
    }
