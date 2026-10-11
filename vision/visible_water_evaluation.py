"""Small, dependency-light helpers for water-mask evaluation reports."""
from __future__ import annotations

from pathlib import Path


def visible_water_prediction_mask(class_labels, *, class_count: int):
    """Map supported model labels to the shared visible-water binary target."""
    import numpy as np

    labels = np.asarray(class_labels)
    if class_count == 2:
        return labels == 1
    if class_count == 10:
        return np.isin(labels, (3, 5))
    raise ValueError("class_count must be 2 or 10")


def manual_evaluation_limitations(*, class_count: int) -> list[str]:
    """Describe what a global visible-water score does and does not validate."""
    if class_count == 2:
        model_note = (
            "The two-class visible-water model predicts generic visible water; it does not "
            "classify road flooding, water depth, or road access safety."
        )
    elif class_count == 10:
        model_note = (
            "The ten-class FloodNet checkpoint predicts post-disaster scene classes; its "
            "flooded-road and water labels are not a binary road-flooding decision."
        )
    else:
        raise ValueError("class_count must be 2 or 10")
    return [
        "The archive does not establish that manual images are source-video-disjoint from model training data.",
        model_note,
        "This evaluates whole-image visible water, not road-only flooding or shallow campus water.",
        "This benchmark does not authorize production deployment of the model or dataset-derived weights.",
    ]


def discover_manual_pairs(dataset_root, *, expected_count: int | None = None) -> list[dict]:
    """Match manually annotated images and masks by sample ID."""
    root = Path(dataset_root).resolve()
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise ValueError("manual dataset must contain images/ and masks/ directories")

    images = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.stem in images:
                raise ValueError(f"duplicate image ID: {path.stem}")
            images[path.stem] = path
    masks = {
        path.stem: path for path in sorted(mask_dir.glob("*.png")) if path.is_file()
    }
    if not images or set(images) != set(masks):
        raise ValueError("image/mask IDs do not match or the manual dataset is empty")
    if expected_count is not None and len(images) != expected_count:
        raise ValueError(f"expected {expected_count} manual pairs, found {len(images)}")
    return [
        {"sample_id": sample_id, "image": images[sample_id], "mask": masks[sample_id]}
        for sample_id in sorted(images)
    ]


def validate_binary_water_mask(mask) -> None:
    """Require an 8-bit grayscale mask with only non-water/water values."""
    import numpy as np

    values = np.unique(np.asarray(mask))
    if values.size == 0 or not np.isin(values, (0, 255)).all():
        raise ValueError("manual water mask must contain only 0 and 255")


def binary_segmentation_counts(prediction, truth) -> dict[str, int]:
    """Return exact binary pixel counts without smoothing or ignored pixels."""
    import numpy as np

    predicted = np.asarray(prediction, dtype=bool)
    actual = np.asarray(truth, dtype=bool)
    if predicted.shape != actual.shape:
        raise ValueError("prediction and truth masks must have identical shapes")
    return {
        "tp": int(np.count_nonzero(predicted & actual)),
        "fp": int(np.count_nonzero(predicted & ~actual)),
        "fn": int(np.count_nonzero(~predicted & actual)),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict:
    tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
    if min(tp, fp, fn) < 0:
        raise ValueError("pixel counts cannot be negative")
    union = tp + fp + fn
    predicted = tp + fp
    actual = tp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / union if union else None,
        "precision": tp / predicted if predicted else None,
        "recall": tp / actual if actual else None,
    }


def bootstrap_iou_interval(image_counts: list[dict[str, int]], *, seed: int,
                           replicates: int = 1000) -> list[float]:
    """Bootstrap aggregate IoU by image ID, not by correlated pixels."""
    import numpy as np

    if not image_counts:
        raise ValueError("at least one image is required for bootstrap")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    values = np.asarray([
        [int(row[key]) for key in ("tp", "fp", "fn")] for row in image_counts
    ], dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("pixel counts cannot be negative")
    rng = np.random.default_rng(seed)
    scores = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        denominator = int(sample.sum())
        scores[index] = float(sample[0] / denominator) if denominator else 0.0
    return [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]
