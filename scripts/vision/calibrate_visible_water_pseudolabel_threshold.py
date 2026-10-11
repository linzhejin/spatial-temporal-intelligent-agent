"""Choose a visible-water probability threshold using validation videos only."""
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

from scripts.vision.evaluate_floodwater_pseudolabel_holdout import (  # noqa: E402
    _dataset_file,
    _sha256,
    _visible_water_probability,
)
from vision.floodwater_training import build_floodwater_manifest  # noqa: E402
from vision.specialized import OnnxFloodSegmenter  # noqa: E402
from vision.specialized_evaluation import select_probability_threshold_from_histograms  # noqa: E402
from vision.visible_water_evaluation import metrics_from_counts  # noqa: E402


def _threshold_counts(positive_bins: list[int], negative_bins: list[int],
                      bin_edges: list[float], threshold: float) -> dict:
    # Runtime probabilities are float32, so the runtime comparison casts the
    # scalar threshold to float32 as well. Match that boundary when querying bins.
    runtime_threshold = float(np.float32(threshold))
    index = int(np.searchsorted(np.asarray(bin_edges), runtime_threshold, side="left"))
    tp = sum(positive_bins[index:])
    fp = sum(negative_bins[index:])
    fn = sum(positive_bins[:index])
    return metrics_from_counts({"tp": tp, "fp": fp, "fn": fn})


def calibrate(model_path: Path, dataset_root: Path, *, sample_period_frames: int = 250,
              bins: int = 512, threads: int = 2) -> dict:
    """Select a micro-pixel-IoU operating point from held-out validation videos."""
    if isinstance(bins, bool) or not isinstance(bins, int) or not 16 <= bins <= 8192:
        raise ValueError("histogram bins must be between 16 and 8192")
    root = dataset_root.resolve()
    if not root.is_dir():
        raise ValueError("dataset root does not exist")
    chunks_csv = _dataset_file(root, "metadata/chunks.csv")
    samples_csv = _dataset_file(root, "metadata/samples.csv")
    manifest = build_floodwater_manifest(
        chunks_csv, samples_csv, sample_period_frames=sample_period_frames,
        sample_splits=("validation",),
    )
    samples = manifest["samples"]
    if any(sample["split"] != "validation" for sample in samples):
        raise ValueError("threshold calibration manifest contains a non-validation sample")
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
        raise ValueError("threshold calibration requires a two-class visible-water model")

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    positive_histogram = np.zeros(bins, dtype=np.int64)
    negative_histogram = np.zeros(bins, dtype=np.int64)
    per_video_histograms: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    sample_fingerprints = []
    dimension_counts = {}
    active_video_path = None
    capture = None
    started = time.monotonic()
    try:
        for index, sample in enumerate(sorted(samples, key=lambda item: (
                item["video"], item["chunk_frame_index"], item["sample_id"])), 1):
            video_path = _dataset_file(root, sample["video"])
            if str(video_path) != active_video_path:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise ValueError(f"could not open validation video: {sample['video']}")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or abs(fps - 25.0) > 0.5:
                    raise ValueError(f"expected 25 fps validation video, got {fps}: {sample['video']}")
                active_video_path = str(video_path)

            frame_index = int(sample["chunk_frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"could not decode validation frame {frame_index}: {sample['video']}")
            next_frame = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if math.isfinite(next_frame) and abs(next_frame - (frame_index + 1)) > 1:
                raise ValueError(f"video seek missed validation frame {frame_index}: {sample['video']}")

            mask_path = _dataset_file(root, sample["mask"])
            truth_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if truth_mask is None or truth_mask.ndim != 2:
                raise ValueError(f"validation mask must be grayscale: {sample['mask']}")
            if not np.isin(np.unique(truth_mask), (0, 255)).all():
                raise ValueError(f"validation mask must use only 0 and 255: {sample['mask']}")
            if frame.shape[:2] != truth_mask.shape:
                raise ValueError(f"validation frame/mask dimensions differ: {sample['sample_id']}")

            probabilities = _visible_water_probability(segmenter, frame)
            actual = truth_mask == 255
            positive = np.histogram(probabilities[actual], bins=bin_edges)[0]
            negative = np.histogram(probabilities[~actual], bins=bin_edges)[0]
            positive_histogram += positive
            negative_histogram += negative
            video_id = sample["source_video_id"]
            video_positive, video_negative, frame_count = per_video_histograms.setdefault(
                video_id, (np.zeros(bins, dtype=np.int64), np.zeros(bins, dtype=np.int64), 0),
            )
            video_positive += positive
            video_negative += negative
            per_video_histograms[video_id] = (video_positive, video_negative, frame_count + 1)
            dimension = f"{frame.shape[1]}x{frame.shape[0]}"
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
            sample_fingerprints.append(
                f"{sample['sample_id']}:{hashlib.sha256(frame.tobytes()).hexdigest()}:{_sha256(mask_path)}",
            )
            if index % 20 == 0 or index == len(samples):
                print(f"calibrated {index}/{len(samples)}", flush=True)
    finally:
        if capture is not None:
            capture.release()

    selection = select_probability_threshold_from_histograms(
        positive_bins=positive_histogram.tolist(),
        negative_bins=negative_histogram.tolist(),
        bin_edges=bin_edges.tolist(),
    )
    threshold = float(selection["threshold"])
    per_video = {
        video_id: {
            "sample_count": count,
            **_threshold_counts(pos.tolist(), neg.tolist(), bin_edges.tolist(), threshold),
        }
        for video_id, (pos, neg, count) in sorted(per_video_histograms.items())
    }
    report = {
        "task": "visible_water_probability_threshold_calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Floodwater Dataset v1.0.0",
        "dataset_license": "GPL-3.0-only",
        "dataset_source": "https://github.com/decide-ugent/floodwater-dataset",
        "split_unit": "source_video_id",
        "calibration_split": "validation",
        "dataset_split_source_video_counts": {
            split: len(source_video_ids)
            for split, source_video_ids in manifest["split_groups"].items()
        },
        "validation_source_video_ids": sorted(per_video_histograms),
        "sampling": {
            "period_frames": sample_period_frames,
            "source_fps": 25,
            "approximate_interval_seconds": round(sample_period_frames / 25, 3),
            "sample_count": len(samples),
        },
        "label_semantics": "SAM2-assisted pseudo-labels of visible water; not human ground truth",
        "label_types": label_types,
        "model_sha256": _sha256(model_path),
        "execution_provider": "CPUExecutionProvider",
        "prediction_rule": {
            "tile_size": list(segmenter.input_size),
            "overlap_pixels": max(1, min(128, min(segmenter.input_size) // 8)),
            "overlap_merge": "per-pixel highest-confidence tile, matching OnnxFloodSegmenter",
        },
        "image_dimensions": dimension_counts,
        "dataset_metadata_sha256": {
            "chunks_csv": _sha256(chunks_csv),
            "samples_csv": _sha256(samples_csv),
        },
        "data_fingerprint_sha256": hashlib.sha256(
            "\n".join(sample_fingerprints).encode("utf-8"),
        ).hexdigest(),
        "histogram_bins": bins,
        "selection_metric": "micro pixel IoU over validation source videos",
        "selected_operating_point": selection,
        "validation_video_metrics_at_selected_threshold": per_video,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "The threshold is selected from validation videos only and must be frozen before test evaluation.",
            "Validation masks are SAM2-assisted pseudo-labels, not independently checked human ground truth.",
            "The model segments full-frame visible water, not road-only flooding, depth, or passability.",
            "Public flood imagery does not establish performance on Wuhan University campus roads or shallow puddles.",
            "This calibration does not authorize production deployment or automatic route control.",
        ],
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="write the calibration report outside the source repository")
    parser.add_argument("--sample-period-frames", type=int, default=250)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if (not args.model.is_file() or not args.dataset_root.is_dir()
            or args.sample_period_frames < 1 or args.threads < 1
            or isinstance(args.bins, bool) or not 16 <= args.bins <= 8192):
        parser.error("model/dataset must exist; frame period, threads and bins must be valid")
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        parser.error("calibration output must be outside the source repository")
    if output.exists():
        parser.error("output exists; choose a new path to preserve the previous calibration")
    report = calibrate(
        args.model.resolve(), args.dataset_root,
        sample_period_frames=args.sample_period_frames,
        bins=args.bins,
        threads=args.threads,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "selected_threshold": report["selected_operating_point"]["threshold"],
        "validation_metrics": report["selected_operating_point"],
        "sample_count": report["sampling"]["sample_count"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
