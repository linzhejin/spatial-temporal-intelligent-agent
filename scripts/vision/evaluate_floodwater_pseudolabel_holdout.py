"""Evaluate binary visible-water ONNX on source-video-held-out pseudo-labels.

This is a data-disjoint consistency check against SAM2-assisted masks. It is
not manual ground truth, road-only flooding, or campus performance validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.floodwater_holdout_evaluation import aggregate_source_video_counts  # noqa: E402
from vision.floodwater_training import build_floodwater_manifest  # noqa: E402
from vision.specialized import OnnxFloodSegmenter  # noqa: E402
from vision.visible_water_evaluation import (  # noqa: E402
    binary_segmentation_counts,
    metrics_from_counts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_file(root: Path, relative_path: str) -> Path:
    base = root.resolve()
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError(f"dataset path escapes root: {relative_path}") from error
    if not path.is_file():
        raise ValueError(f"dataset file does not exist: {relative_path}")
    return path


def _visible_water_probability(segmenter: OnnxFloodSegmenter, frame: np.ndarray) -> np.ndarray:
    """Read full-frame probabilities through the same tiler used by the app."""
    region = {
        "id": "full-frame-evaluation",
        "kind": "road_surface",
        "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    }
    prediction = segmenter.predict(frame, region, include_probabilities=True)
    if not isinstance(prediction, dict) or "visible_water_probability" not in prediction:
        raise ValueError("runtime segmenter did not return visible-water probabilities")
    probabilities = np.asarray(prediction["visible_water_probability"], dtype=np.float32)
    if probabilities.shape != frame.shape[:2]:
        raise ValueError("runtime probabilities must match the source frame")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("runtime segmenter returned invalid visible-water probabilities")
    return probabilities


def _threshold_comparison_summaries(samples, selected_counts, baseline_counts) -> dict:
    """Name the selected operating point and fixed 0.5 baseline without implying argmax."""
    return {
        "selected_probability_threshold": aggregate_source_video_counts(samples, selected_counts),
        "fixed_probability_threshold_0_5": aggregate_source_video_counts(samples, baseline_counts),
    }


def evaluate(model_path: Path, dataset_root: Path, *, sample_period_frames: int = 250,
             probability_threshold: float = 0.5, threads: int = 2) -> dict:
    """Score evenly sampled frames from only the official test-source videos."""
    if (isinstance(probability_threshold, bool)
            or not isinstance(probability_threshold, (int, float))
            or not math.isfinite(float(probability_threshold))
            or not 0 <= probability_threshold <= 1):
        raise ValueError("probability threshold must be between 0 and 1")
    root = dataset_root.resolve()
    if not root.is_dir():
        raise ValueError("dataset root does not exist")
    chunks_csv = _dataset_file(root, "metadata/chunks.csv")
    samples_csv = _dataset_file(root, "metadata/samples.csv")
    manifest = build_floodwater_manifest(
        chunks_csv, samples_csv, sample_period_frames=sample_period_frames,
        sample_splits=("test",),
    )
    samples = manifest["samples"]
    if any(sample["split"] != "test" for sample in samples):
        raise ValueError("holdout evaluation manifest contains a non-test sample")
    label_types = sorted({sample["label_type"] for sample in samples})

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    segmenter = OnnxFloodSegmenter(str(model_path), session=session)
    if not segmenter.is_binary_visible_water:
        raise ValueError("held-out pseudo-label evaluation requires a two-class visible-water model")
    tile_width, tile_height = segmenter.input_size

    samples = sorted(samples, key=lambda sample: (
        sample["video"], sample["chunk_frame_index"], sample["sample_id"],
    ))
    sample_counts = []
    baseline_counts = []
    per_sample = []
    dimension_counts = {}
    fingerprint_rows = []
    started = time.monotonic()
    active_video_path = None
    capture = None
    try:
        for index, sample in enumerate(samples, 1):
            video_path = _dataset_file(root, sample["video"])
            if str(video_path) != active_video_path:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise ValueError(f"could not open source video: {sample['video']}")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or abs(fps - 25.0) > 0.5:
                    raise ValueError(f"expected a 25 fps source video, got {fps}: {sample['video']}")
                active_video_path = str(video_path)

            frame_index = int(sample["chunk_frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"could not decode frame {frame_index}: {sample['video']}")
            next_frame = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if math.isfinite(next_frame) and abs(next_frame - (frame_index + 1)) > 1:
                raise ValueError(f"video seek did not reach frame {frame_index}: {sample['video']}")

            mask_path = _dataset_file(root, sample["mask"])
            truth_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if truth_mask is None or truth_mask.ndim != 2:
                raise ValueError(f"mask must be a decodable grayscale PNG: {sample['mask']}")
            if not np.isin(np.unique(truth_mask), (0, 255)).all():
                raise ValueError(f"mask must use only 0 and 255: {sample['mask']}")
            if frame.shape[:2] != truth_mask.shape:
                raise ValueError(f"frame/mask dimensions differ for {sample['sample_id']}")

            height, width = frame.shape[:2]
            dimension = f"{width}x{height}"
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
            probabilities = _visible_water_probability(segmenter, frame)
            actual = truth_mask == 255
            baseline = binary_segmentation_counts(probabilities >= 0.5, actual)
            counts = binary_segmentation_counts(probabilities >= probability_threshold, actual)
            baseline_counts.append(baseline)
            sample_counts.append(counts)
            frame_digest = hashlib.sha256(frame.tobytes()).hexdigest()
            mask_digest = _sha256(mask_path)
            fingerprint_rows.append(
                f"{sample['sample_id']}:{frame_digest}:{mask_digest}",
            )
            per_sample.append({
                "sample_id": sample["sample_id"],
                "source_video_id": sample["source_video_id"],
                "chunk_frame_index": frame_index,
                "chunk_time_seconds": sample["chunk_time_seconds"],
                "label_type": sample["label_type"],
                "fixed_probability_threshold_0_5": metrics_from_counts(baseline),
                **metrics_from_counts(counts),
            })
            if index % 20 == 0 or index == len(samples):
                print(f"evaluated {index}/{len(samples)}", flush=True)
    finally:
        if capture is not None:
            capture.release()

    comparison_summaries = _threshold_comparison_summaries(
        samples, sample_counts, baseline_counts,
    )
    summary = comparison_summaries["selected_probability_threshold"]
    report = {
        "task": "visible_water_source_video_holdout_pseudolabel_consistency",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Floodwater Dataset v1.0.0",
        "dataset_license": "GPL-3.0-only",
        "dataset_source": "https://github.com/decide-ugent/floodwater-dataset",
        "split_unit": "source_video_id",
        "evaluated_split": "test",
        "dataset_split_source_video_counts": {
            split: len(source_video_ids)
            for split, source_video_ids in manifest["split_groups"].items()
        },
        "test_source_video_ids": sorted({sample["source_video_id"] for sample in samples}),
        "sampling": {
            "period_frames": sample_period_frames,
            "source_fps": 25,
            "approximate_interval_seconds": round(sample_period_frames / 25, 3),
            "sample_count": len(samples),
        },
        "label_semantics": "SAM2-assisted pseudo-labels of visible water; not human ground truth",
        "label_types": label_types,
        "model_sha256": _sha256(model_path),
        "model_class_count": 2,
        "execution_provider": "CPUExecutionProvider",
        "prediction_rule": {
            "tile_size": [tile_width, tile_height],
            "overlap_pixels": max(1, min(128, min(tile_width, tile_height) // 8)),
            "overlap_merge": "per-pixel highest-confidence tile, matching OnnxFloodSegmenter",
            "probability_threshold": float(probability_threshold),
            "fixed_baseline_probability_threshold": 0.5,
            "threshold_selection": "chosen on validation source videos only; test split was not used",
        },
        "image_dimensions": dimension_counts,
        "dataset_metadata_sha256": {
            "chunks_csv": _sha256(chunks_csv),
            "samples_csv": _sha256(samples_csv),
        },
        "data_fingerprint_sha256": hashlib.sha256(
            "\n".join(fingerprint_rows).encode("utf-8"),
        ).hexdigest(),
        "summary": summary,
        "fixed_probability_threshold_0_5_summary": (
            comparison_summaries["fixed_probability_threshold_0_5"]
        ),
        "uncertainty_interval": None,
        "uncertainty_note": (
            "No frame-level confidence interval is reported because frames within a source video "
            "are correlated and only five held-out source videos are available."
        ),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "The test source-video IDs are disjoint from the train and validation source-video groups recorded in dataset metadata; masks are SAM2-assisted pseudo-labels.",
            "This measures consistency with the dataset masks, not accuracy against independently checked human ground truth.",
            "The model segments generic visible water over the full image; this is not road-only flooding, water-depth, or passability recognition.",
            "Flood imagery does not establish performance on Wuhan University campus roads or shallow puddles.",
            "This offline evaluation does not enable or authorize model deployment or automatic route control.",
        ],
        "per_sample": per_sample,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="binary visible-water ONNX")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="extracted Floodwater Dataset root with metadata/ and data/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-period-frames", type=int, default=250,
                        help="default 250 frames, about 10 seconds at 25 fps")
    parser.add_argument("--probability-threshold", type=float, default=0.5,
                        help="fixed visible-water probability threshold; pass the validation-selected value explicitly")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error("model file does not exist")
    if (args.sample_period_frames < 1 or args.threads < 1
            or not math.isfinite(args.probability_threshold)
            or not 0 <= args.probability_threshold <= 1):
        parser.error("sample period/threads must be positive and threshold must be between 0 and 1")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error("output exists; pass --overwrite to replace it")
    report = evaluate(
        args.model.resolve(), args.dataset_root,
        sample_period_frames=args.sample_period_frames,
        probability_threshold=args.probability_threshold, threads=args.threads,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "summary": report["summary"],
        "fixed_probability_threshold_0_5_summary": report[
            "fixed_probability_threshold_0_5_summary"
        ],
        "elapsed_seconds": report["elapsed_seconds"],
        "labels": report["label_semantics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
