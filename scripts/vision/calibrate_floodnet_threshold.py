#!/usr/bin/env python3
"""Select a flooded-road pixel threshold using only the FloodNet validation split."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.vision.evaluate_specialized import _flood_valid_mask, _polygon_mask, _read_flood_pair
from vision.specialized import OnnxFloodSegmenter
from vision.specialized_evaluation import (
    select_probability_threshold_from_histograms,
    validate_split_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(str(relative).replace("\\", "/"))
    windows = PureWindowsPath(str(relative))
    if (posix.is_absolute() or windows.is_absolute() or windows.drive
            or ".." in posix.parts or (posix.parts and ":" in posix.parts[0])):
        raise ValueError("dataset sample path must be relative and stay inside dataset-root")
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("dataset sample path resolves outside dataset-root") from error
    if not path.is_file():
        raise ValueError(f"dataset sample file is missing: {relative}")
    return path


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="full train/validation/test FloodNet manifest")
    parser.add_argument("--model", type=Path, required=True, help="local ONNX model")
    parser.add_argument("--output", type=Path, required=True,
                        help="write the calibration report outside the source repository")
    parser.add_argument("--bins", type=int, default=512)
    args = parser.parse_args()
    if not args.dataset_root.is_dir() or not args.manifest.is_file() or not args.model.is_file():
        parser.error("dataset-root, manifest, and model must exist")
    if args.bins < 16 or args.bins > 8192:
        parser.error("bins must be between 16 and 8192")
    try:
        args.output.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        parser.error("output must be outside the source repository")
    try:
        manifest_bytes = args.manifest.read_bytes()
        source_manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        validation_samples = [sample for sample in source_manifest.get("samples", [])
                              if isinstance(sample, dict) and sample.get("split") == "validation"]
        validation_manifest = dict(source_manifest)
        validation_manifest["samples"] = validation_samples
        manifest = validate_split_manifest(
            validation_manifest, task="flooded_road_segmentation", split="validation",
        )
        root = args.dataset_root.resolve()
        for sample in manifest["samples"]:
            _dataset_path(root, sample["image"])
            _dataset_path(root, sample["mask"])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    segmenter = OnnxFloodSegmenter(str(args.model), flood_probability_threshold=0.0)
    positive_histogram = np.zeros(args.bins, dtype=np.int64)
    negative_histogram = np.zeros(args.bins, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, args.bins + 1).tolist()
    positive_image_count = 0
    aligned_count = 0
    valid_pixel_count = ignored_pixel_count = 0
    started = time.monotonic()
    for index, sample in enumerate(manifest["samples"], start=1):
        image_path = _dataset_path(root, sample["image"])
        mask_path = _dataset_path(root, sample["mask"])
        frame, truth_labels, alignment = _read_flood_pair(image_path, mask_path)
        aligned_count += int(alignment == "image_resized_to_mask_grid")
        region = {"id": sample["group_id"], "kind": "road_surface",
                  "polygon": sample["road_surface_polygon"]}
        prediction = segmenter.predict(frame, region, include_probabilities=True)
        if prediction is None:
            raise ValueError(f"validation road ROI is empty: {sample['image']}")
        roi = _polygon_mask(frame.shape, sample["road_surface_polygon"])
        valid = _flood_valid_mask(truth_labels, roi=roi)
        valid_pixel_count += int(valid.sum())
        ignored_pixel_count += int(roi.sum()) - int(valid.sum())
        actual_positive = valid & (truth_labels == int(sample.get("flooded_road_class_id", 3)))
        scores = prediction["flooded_road_probability"]
        positive_image_count += int(actual_positive.any())
        positive_histogram += np.histogram(scores[actual_positive], bins=bin_edges)[0]
        negative_histogram += np.histogram(scores[valid & ~actual_positive], bins=bin_edges)[0]
        print(json.dumps({"validation_progress": {"completed": index,
                                                     "total": len(manifest["samples"])}},
                         ensure_ascii=False), flush=True)

    selected = select_probability_threshold_from_histograms(
        positive_bins=positive_histogram.tolist(),
        negative_bins=negative_histogram.tolist(), bin_edges=bin_edges,
    )
    report = {
        "task": "flooded_road_probability_calibration",
        "dataset": manifest["dataset"], "split": "validation",
        "split_unit": manifest["split_unit"],
        "sample_count": len(manifest["samples"]),
        "validation_positive_images": positive_image_count,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_sha256": _sha256(args.model.resolve()),
        "histogram_bins": args.bins,
        "image_mask_grid_resized_count": aligned_count,
        "pixel_alignment_verified": False,
        "valid_pixel_count": valid_pixel_count,
        "ignored_pixel_count": ignored_pixel_count,
        "operating_point_selected_on_validation_pixels": selected,
        "selection_metric": "micro pixel IoU within annotated road region",
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "Threshold was selected on validation only; evaluate the frozen threshold once on the separate test split.",
            "FloodNet splits are labeled image IDs, not flight/scene IDs; results do not establish independent-scene generalization.",
            "Public post-disaster imagery does not establish Wuhan University shallow-water performance or road-closure accuracy.",
            "Matching dimensions and default EXIF orientation do not prove image/mask pixel registration.",
            "A probability threshold cannot repair missing visual features or domain shift; keep the model disabled unless held-out evaluation and campus validation pass.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output),
                      "selected_validation_threshold": selected["threshold"],
                      "validation_metrics": selected,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "limitations": report["limitations"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
