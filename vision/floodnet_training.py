"""Training-data contracts for the optional FloodNet road-water model.

The production inference worker does not import this module's optional ML
dependencies. Training dependencies are loaded only by the explicit CLI.
"""
from __future__ import annotations

import math
import random
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from vision.tiling import tile_windows


_SPLITS = ("train", "validation", "test")


def _require_unambiguous_geometry(image, mask) -> None:
    """Reject unsupported inputs; matching headers alone do not prove alignment."""
    if image.getexif().get(274, 1) != 1 or mask.getexif().get(274, 1) != 1:
        raise ValueError("unresolved image or mask EXIF orientation; verify registration before use")
    if image.size != mask.size:
        raise ValueError("image and mask dimensions differ; verify registration before use")
    if mask.mode not in {"L", "P", "I", "I;16"}:
        raise ValueError("mask must contain class-index pixels")


class FloodNetSegmentationDataset:
    """Read FloodNet's grayscale class-index masks and produce square image tiles."""

    def __init__(self, root, samples: list[dict], *, tile_size: int = 512,
                 overlap: int = 128, augment: bool = False,
                 one_tile_per_sample: bool = False,
                 random_crop_per_sample: bool = True,
                 focus_class_id: int | None = None,
                 focus_probability: float = 0.0,
                 focus_class_center: bool = False):
        from PIL import Image

        if not isinstance(samples, list) or not samples:
            raise ValueError("FloodNet samples must be a non-empty list")
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < tile_size:
            raise ValueError("overlap must be non-negative and smaller than tile_size")
        if focus_class_id is not None and (
                isinstance(focus_class_id, bool) or not isinstance(focus_class_id, int)
                or not 0 <= focus_class_id <= 9):
            raise ValueError("focus_class_id must be a FloodNet class ID from 0 through 9")
        if isinstance(focus_probability, bool) or not isinstance(focus_probability, (int, float)) or not 0 <= focus_probability <= 1:
            raise ValueError("focus_probability must be between zero and one")
        if not isinstance(focus_class_center, bool) or (focus_class_center and focus_class_id is None):
            raise ValueError("focus_class_center requires a FloodNet focus_class_id")
        self.root = Path(root).resolve()
        self.augment = bool(augment)
        self.random_crop_per_sample = bool(random_crop_per_sample)
        self.focus_class_id = focus_class_id
        self.focus_probability = float(focus_probability)
        self.focus_class_center = focus_class_center
        self.tile_size = tile_size
        self._cached_sample_key = None
        self._cached_image = None
        self._cached_mask = None
        self.items = []
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"sample {index} must be an object")
            image_rel = _relative_dataset_path(sample.get("image"), f"sample {index} image")
            mask_rel = _relative_dataset_path(sample.get("mask"), f"sample {index} mask")
            image_path = self._resolve(image_rel)
            mask_path = self._resolve(mask_rel)
            if not image_path.is_file() or not mask_path.is_file():
                raise ValueError(f"sample {index} image or mask does not exist")
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                _require_unambiguous_geometry(image, mask)
                width, height = mask.size
            windows = ([(None, None)] if one_tile_per_sample else
                       tile_windows(width=width, height=height, tile_size=tile_size, overlap=overlap))
            for x, y in windows:
                self.items.append((image_rel, mask_rel, width, height, x, y))
        self.alignment_report = {
            "identity": len(samples),
            "image_resized_to_mask_grid": 0,
            "mappings": [],
            "rule": "matching dimensions and identity orientation only; pixel alignment not verified",
        }

    def _resolve(self, relative: str):
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("dataset sample resolves outside the dataset root") from error
        return path

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image

        image_rel, mask_rel, width, height, x, y = self.items[index]
        cache_key = (image_rel, mask_rel, width, height)
        if cache_key != self._cached_sample_key:
            with Image.open(self._resolve(image_rel)) as source, Image.open(self._resolve(mask_rel)) as label:
                _require_unambiguous_geometry(source, label)
                if source.size != (width, height):
                    raise ValueError("image and mask dimensions changed after dataset inspection")
                image = np.asarray(source.convert("RGB"), dtype=np.uint8)
                mask = np.array(label, copy=True)
            self._cached_sample_key = cache_key
            self._cached_image = image
            self._cached_mask = mask
        image = self._cached_image
        mask = self._cached_mask
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError(f"mask shape does not match image: {mask_rel}")
        values = np.unique(mask)
        if np.any((values > 9) & (values != 255)):
            raise ValueError(f"mask contains invalid FloodNet label values: {values[:12].tolist()}")
        if x is None or y is None:
            locations = None
            if self.focus_class_id is not None:
                if self.random_crop_per_sample:
                    focused = random.random() < self.focus_probability
                else:
                    focused = self.focus_class_center
                if focused:
                    locations = np.argwhere(mask == self.focus_class_id)
            if locations is not None and len(locations):
                if self.random_crop_per_sample:
                    focus_y, focus_x = locations[random.randrange(len(locations))]
                else:
                    focus_y, focus_x = np.rint(locations.mean(axis=0)).astype(int)
                x = min(max(0, int(focus_x) - self.tile_size // 2), max(0, width - self.tile_size))
                y = min(max(0, int(focus_y) - self.tile_size // 2), max(0, height - self.tile_size))
            elif self.random_crop_per_sample:
                x = random.randint(0, max(0, width - self.tile_size))
                y = random.randint(0, max(0, height - self.tile_size))
            else:
                x = max(0, (width - self.tile_size) // 2)
                y = max(0, (height - self.tile_size) // 2)
        x = min(int(x), max(0, width - self.tile_size))
        y = min(int(y), max(0, height - self.tile_size))
        image_tile = image[y:y + self.tile_size, x:x + self.tile_size]
        mask_tile = mask[y:y + self.tile_size, x:x + self.tile_size]
        if image_tile.shape[:2] != (self.tile_size, self.tile_size):
            pad_y = self.tile_size - image_tile.shape[0]
            pad_x = self.tile_size - image_tile.shape[1]
            image_tile = np.pad(image_tile, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
            mask_tile = np.pad(mask_tile, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=255)
        if self.augment and random.random() < 0.5:
            image_tile = image_tile[:, ::-1]
            mask_tile = mask_tile[:, ::-1]
        if self.augment and random.random() < 0.5:
            image_tile = image_tile[::-1]
            mask_tile = mask_tile[::-1]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_tile.transpose(2, 0, 1))).float().div_(255.0)
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=image_tensor.dtype).view(3, 1, 1)
        return (image_tensor - mean) / std, torch.from_numpy(np.ascontiguousarray(mask_tile)).long()


def floodnet_class_weights(root, samples: list[dict], *, num_classes: int = 10) -> Any:
    """Compute bounded inverse-square-root pixel weights from training masks only."""
    import numpy as np
    import torch
    from PIL import Image
    from pathlib import Path

    if num_classes != 10:
        raise ValueError("FloodNet class weights require the ten official class IDs")
    base = Path(root).resolve()
    counts = np.zeros(num_classes, dtype=np.int64)
    for sample in samples:
        relative = _relative_dataset_path(sample.get("mask"), "training mask")
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ValueError("training mask resolves outside the dataset root") from error
        with Image.open(path) as source:
            labels = np.asarray(source)
        if labels.ndim != 2 or np.any((labels > 9) & (labels != 255)):
            raise ValueError(f"invalid FloodNet mask values: {relative}")
        counts += np.bincount(labels[(labels >= 0) & (labels < num_classes)].ravel(), minlength=num_classes)
    present = counts > 0
    if not present.any():
        raise ValueError("training masks contain no valid class pixels")
    weights = np.ones(num_classes, dtype=np.float32)
    median = float(np.median(counts[present]))
    weights[present] = np.sqrt(median / counts[present])
    weights = np.clip(weights, 0.25, 6.0)
    weights /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def floodnet_focus_sample_weights(root, samples: list[dict], *, positive_class_id: int = 3,
                                  positive_weight: float = 8.0) -> tuple[list[float], dict]:
    """Oversample training images that contain the target road-water class.

    FloodNet's official image-level split can leave very few flooded-road images
    in the train split. This changes sampling probability only; it never reads
    validation/test masks and does not duplicate samples across splits.
    """
    import numpy as np
    from PIL import Image

    if not isinstance(samples, list) or not samples:
        raise ValueError("training samples must be a non-empty list")
    if (isinstance(positive_class_id, bool) or not isinstance(positive_class_id, int)
            or not 0 <= positive_class_id <= 9):
        raise ValueError("positive class ID must be a FloodNet class from 0 through 9")
    if (isinstance(positive_weight, bool) or not isinstance(positive_weight, (int, float))
            or not math.isfinite(float(positive_weight)) or float(positive_weight) < 1):
        raise ValueError("positive sample weight must be finite and at least one")
    base = Path(root).resolve()
    weights = []
    positive_images = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("split") != "train":
            raise ValueError(f"sample {index} must belong to the training split")
        relative = _relative_dataset_path(sample.get("mask"), f"sample {index} mask")
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as error:
            raise ValueError("training mask resolves outside the dataset root") from error
        if not path.is_file():
            raise ValueError(f"training mask does not exist: {relative}")
        with Image.open(path) as source:
            labels = np.asarray(source)
        if labels.ndim != 2 or np.any((labels > 9) & (labels != 255)):
            raise ValueError(f"invalid FloodNet mask values: {relative}")
        positive = bool(np.any(labels == positive_class_id))
        positive_images += int(positive)
        weights.append(float(positive_weight) if positive else 1.0)
    positive_draw_mass = positive_images * float(positive_weight)
    total_draw_mass = positive_draw_mass + (len(samples) - positive_images)
    return weights, {
        "positive_images": positive_images,
        "negative_images": len(samples) - positive_images,
        "positive_sampling_weight": float(positive_weight),
        "expected_positive_draw_fraction": positive_draw_mass / total_draw_mass,
    }


def evaluate_floodnet(model, data_loader, *, device, num_classes: int = 10) -> dict:
    """Compute per-class pixel counts, precision, recall, and IoU."""
    import torch

    if num_classes != 10:
        raise ValueError("FloodNet evaluation requires ten classes")
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")
    model.eval()
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)
            logits = model(images)
            if logits.ndim != 4 or logits.shape[1] != num_classes or logits.shape[0] != labels.shape[0]:
                raise ValueError("model must emit [N,10,H,W] logits")
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = logits.argmax(dim=1)
            valid = (labels != 255) & (labels >= 0) & (labels < num_classes)
            encoded = labels[valid] * num_classes + predicted[valid]
            matrix += torch.bincount(encoded.cpu(), minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    result = {}
    for class_id, name in enumerate(("background", "flooded_building", "non_flooded_building",
                                     "flooded_road", "non_flooded_road", "water", "tree",
                                     "vehicle", "pool", "grass")):
        tp = int(matrix[class_id, class_id].item())
        fp = int(matrix[:, class_id].sum().item()) - tp
        fn = int(matrix[class_id, :].sum().item()) - tp
        union = tp + fp + fn
        predicted_count = tp + fp
        actual_count = tp + fn
        result[name] = {
            "class_id": class_id, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / predicted_count if predicted_count else None,
            "recall": tp / actual_count if actual_count else None,
            "iou": tp / union if union else None,
        }
    present_ious = [record["iou"] for record in result.values() if record["iou"] is not None]
    result["mean_iou"] = sum(present_ious) / len(present_ious) if present_ious else None
    return result


def train_floodnet(model, train_loader, validation_loader, *, device,
                   checkpoint_path, epochs: int = 10, learning_rate: float = 1e-3,
                   class_weights=None, gradient_clip: float = 1.0,
                   progress_callback=None) -> dict:
    """Train one deterministic CPU/GPU run and atomically save its best state dict."""
    import os
    import tempfile
    import torch
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not (learning_rate > 0) or not (gradient_clip > 0):
        raise ValueError("learning_rate and gradient_clip must be positive")
    device = torch.device(device)
    model.to(device)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    if weights is not None and (weights.shape != (10,) or not torch.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("class_weights must contain ten finite positive values")
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    target_path = Path(checkpoint_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = (-1.0, -1.0)
    best_iou = -1.0
    best_epoch = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.ndim != 4 or logits.shape[1] != 10:
                raise ValueError("model must emit [N,10,H,W] logits")
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("training loss is not finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if not batches:
            raise ValueError("training loader produced no batches")
        metrics = evaluate_floodnet(model, validation_loader, device=device)
        road_iou = metrics["flooded_road"]["iou"]
        score = (
            float(road_iou) if road_iou is not None else -1.0,
            float(metrics["mean_iou"]) if metrics["mean_iou"] is not None else -1.0,
        )
        history.append({"epoch": epoch, "mean_train_loss": total_loss / batches,
                        "validation_mean_iou": metrics["mean_iou"],
                        "validation_flooded_road_iou": road_iou})
        if progress_callback is not None:
            progress_callback(dict(history[-1]))
        if score > best_score:
            with tempfile.NamedTemporaryFile(prefix=target_path.stem + "-", suffix=".tmp",
                                             dir=target_path.parent, delete=False) as temporary:
                temp_path = Path(temporary.name)
            try:
                torch.save(model.state_dict(), temp_path)
                os.replace(temp_path, target_path)
            finally:
                temp_path.unlink(missing_ok=True)
            best_score = score
            best_iou = score[0]
            best_epoch = epoch
    return {"epochs_completed": epochs, "best_epoch": best_epoch,
            "best_validation_flooded_road_iou": None if best_iou < 0 else best_iou,
            "checkpoint_selection": "flooded_road_iou_then_validation_mean_iou",
            "history": history, "checkpoint": str(target_path)}


def export_floodnet_onnx(model, output_path, *, input_size: int = 512, opset_version: int = 17) -> dict:
    """Export fixed RGB tensors and verify the ONNX graph with CPU inference."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size < 32:
        raise ValueError("input_size must be an integer of at least 32")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(torch.device("cpu")).eval()
    sample = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(model, sample, str(path), input_names=["image"], output_names=["logits"],
                          opset_version=opset_version, do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"image": sample.numpy()})[0]
    with torch.inference_mode():
        expected = model(sample).cpu().numpy()
    expected_shape = tuple(expected.shape)
    max_error = float(np.max(np.abs(actual - expected)))
    if (expected.ndim != 4 or expected_shape[0] != 1
            or expected_shape[1] not in (2, 10)
            or expected_shape[-2:] != (input_size, input_size)
            or actual.shape != expected_shape
            or not np.allclose(actual, expected, rtol=1e-3, atol=1e-4)):
        raise ValueError("exported ONNX model failed CPU parity check")
    return {"path": str(path), "input_shape": [1, 3, input_size, input_size],
            "output_shape": list(actual.shape), "class_count": int(actual.shape[1]),
            "max_absolute_error": max_error}


def _relative_dataset_path(value: object, field: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if (not normalized or path.is_absolute() or windows_path.is_absolute()
            or any(part in {"", ".", ".."} for part in normalized.split("/"))):
        raise ValueError(f"{field} must be a safe relative path inside the dataset root")
    return normalized


def validate_floodnet_manifest(manifest: object) -> dict:
    """Validate a train/validation/test manifest grouped by independent scenes.

    Each sample must carry a ``split`` and ``group_id``. The same scene group
    cannot occur in multiple splits, preventing adjacent views of one flight or
    scene from leaking into evaluation.
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if manifest.get("task") != "flooded_road_segmentation":
        raise ValueError("manifest task must be flooded_road_segmentation")
    if not str(manifest.get("dataset") or "").strip():
        raise ValueError("manifest dataset name is required")
    if not str(manifest.get("split_unit") or "").strip():
        raise ValueError("manifest split_unit must identify the independent scene or flight")

    groups = manifest.get("split_groups")
    if not isinstance(groups, dict):
        raise ValueError("manifest split_groups is required")
    normalized_groups: dict[str, list[str]] = {}
    for split in _SPLITS:
        values = groups.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split_groups.{split} must be a non-empty list")
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError(f"split_groups.{split} contains an empty or duplicate group")
        normalized_groups[split] = normalized
    all_groups = [group for split in _SPLITS for group in normalized_groups[split]]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("train, validation, and test groups must not overlap")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a non-empty list")
    seen_images: set[str] = set()
    seen_masks: set[str] = set()
    normalized_samples = []
    present_splits: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        split = str(sample.get("split") or "").strip()
        group_id = str(sample.get("group_id") or "").strip()
        if split not in _SPLITS or not group_id:
            raise ValueError(f"sample {index} requires a valid split and group_id")
        if group_id not in normalized_groups[split]:
            raise ValueError(f"sample {index} group_id does not match its declared split")
        image = _relative_dataset_path(sample.get("image"), f"sample {index} image")
        mask = _relative_dataset_path(sample.get("mask"), f"sample {index} mask")
        if image in seen_images or mask in seen_masks:
            raise ValueError("manifest contains duplicate image or mask paths")
        seen_images.add(image)
        seen_masks.add(mask)
        present_splits.add(split)
        normalized_samples.append({**sample, "image": image, "mask": mask,
                                  "group_id": group_id, "split": split})
    if present_splits != set(_SPLITS):
        raise ValueError("manifest must contain samples for train, validation, and test")

    result = dict(manifest)
    result["split_groups"] = normalized_groups
    result["samples"] = normalized_samples
    return result


def get_training_device() -> Any:
    """Return one device for a run; Windows CPU training is the safe default."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_training_seed(seed: int) -> None:
    """Seed Python and PyTorch random generators for reproducible runs."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_floodnet_model(*, num_classes: int = 10, pretrained: bool = False) -> Any:
    """Create a MobileNetV3-small encoder with a lightweight U-Net decoder.

    The model consumes ImageNet-normalized RGB tensors and emits full-resolution
    class logits. Ten outputs retain FloodNet's class order; two outputs are
    background and visible water for the Floodwater experiment.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes not in (2, 10):
        raise ValueError("segmentation model must output either two visible-water or ten FloodNet classes")
    import torch
    from torch import nn
    from torch.nn import functional as functional
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    encoder = mobilenet_v3_small(weights=weights).features

    def collect_features(value, layers):
        outputs = []
        previous_size = None
        for layer in layers:
            value = layer(value)
            size = tuple(value.shape[-2:])
            if size != previous_size:
                outputs.append(value)
                previous_size = size
            else:
                outputs[-1] = value
        return outputs

    with torch.inference_mode():
        probe = collect_features(torch.zeros((1, 3, 128, 128)), encoder)
    channels = [int(value.shape[1]) for value in probe]
    if len(channels) < 2:
        raise RuntimeError("MobileNetV3 encoder did not produce a feature pyramid")

    class DecoderBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1,
                          groups=output_channels, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class FloodNetMobileNetUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            decoder = []
            input_channels = channels[-1]
            for skip_channels in reversed(channels[:-1]):
                output_channels = max(24, min(64, skip_channels * 2))
                decoder.append(DecoderBlock(input_channels + skip_channels, output_channels))
                input_channels = output_channels
            self.decoder = nn.ModuleList(decoder)
            self.classifier = nn.Conv2d(input_channels, num_classes, kernel_size=1)

        def forward(self, value):
            input_size = value.shape[-2:]
            features = collect_features(value, self.encoder)
            value = features[-1]
            for block, skip in zip(self.decoder, reversed(features[:-1])):
                value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                value = block(torch.cat((value, skip), dim=1))
            value = self.classifier(value)
            return functional.interpolate(value, size=input_size, mode="bilinear", align_corners=False)

    return FloodNetMobileNetUNet()
