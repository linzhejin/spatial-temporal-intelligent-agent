#!/usr/bin/env python3
"""Evaluate pedestrian detections against a held-out MovingDroneCrowd++ sample."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from scripts.vision.evaluate_visdrone import _load_model_manifest, _sha256  # noqa: E402
from vision.engine import OnnxDetector  # noqa: E402
from vision.evaluation import evaluate_head_point_frame, parse_moving_drone_rows  # noqa: E402


def load_sequence(dataset_dir: Path) -> tuple[list[Path], str, dict]:
    """Validate the sample's source record, frame sequence, and annotation index."""
    source_path = dataset_dir / "source.json"
    annotations_path = dataset_dir / "annotations.csv"
    if not source_path.is_file() or not annotations_path.is_file():
        raise ValueError("dataset directory must contain source.json and annotations.csv")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("source.json must contain a JSON object")
    split = source.get("split")
    if (source.get("dataset") != "MovingDroneCrowd++"
            or source.get("license") != "CC BY-NC 4.0"
            or not isinstance(split, str)
            or split not in {"official val.txt", "official test.txt"}):
        raise ValueError(
            "sample metadata must identify the official MovingDroneCrowd++ val.txt or test.txt split and license"
        )
    images = sorted(
        (path for path in dataset_dir.glob("*.jpg") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    expected_count = source.get("frames")
    if (isinstance(expected_count, bool) or not isinstance(expected_count, int)
            or expected_count <= 0 or len(images) != expected_count):
        raise ValueError("JPEG count does not match the official sample metadata")
    if [int(path.stem) for path in images] != list(range(1, len(images) + 1)):
        raise ValueError("JPEG names must be consecutive and start at 1.jpg")

    annotation_text = annotations_path.read_text(encoding="utf-8-sig")
    rows = [line.strip() for line in annotation_text.splitlines() if line.strip()]
    expected_rows = source.get("annotation_rows")
    if (isinstance(expected_rows, bool) or not isinstance(expected_rows, int)
            or len(rows) != expected_rows):
        raise ValueError("annotation row count does not match source.json")
    for line_number, row in enumerate(rows, start=1):
        values = row.split(",")
        if len(values) != 10:
            raise ValueError(f"annotation row {line_number} must contain 10 comma-separated fields")
        try:
            frame_index = int(values[0].strip())
        except ValueError as error:
            raise ValueError(f"annotation row {line_number} has an invalid frame index") from error
        if not 0 <= frame_index < len(images):
            raise ValueError(f"annotation row {line_number} references an unknown frame")
    validate_sequence_map(images, source)
    return images, annotation_text, source


def build_evaluation_limitations(source: dict, *, images_evaluated: int) -> list[str]:
    """Describe sample size and frame sampling without assuming one sequence."""
    sequences = source.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        sequence = source.get("sequence")
        sequences = [sequence] if isinstance(sequence, str) and sequence else []
    sequence_count = len(sequences)
    sampling = source.get("sampling")
    split = source.get("split", "official split")
    limitations = [
        f"This subset covers {sequence_count} sequences and {images_evaluated} sampled frames; "
        f"frames from a sequence are correlated and the subset is not the complete {split} sequence set.",
        "Dataset labels head boxes while the model predicts full-person boxes; head-point coverage is a cross-task proxy.",
        "This framewise metric does not evaluate tracking or temporal crowding-event recognition.",
        "Head centers outside the image were excluded from visible-target counts.",
    ]
    if isinstance(sampling, str) and sampling.strip():
        limitations.append(f"Sampling: {sampling.strip()}.")
    limitations.append("Results must not be generalized to campus flights or used to set a deployment threshold.")
    return limitations


def _metric_scores(counts: Counter | dict[str, int]) -> tuple[float, float, float]:
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize_sequence_metrics(frame_metrics: list[dict]) -> dict:
    """Report per-sequence scores and equal-weight means across sampled sequences."""
    if not isinstance(frame_metrics, list) or not frame_metrics:
        raise ValueError("at least one frame metric is required")

    grouped: dict[str, dict] = {}
    count_keys = ("tp", "fp", "fn", "predicted", "ground_truth")
    for index, frame in enumerate(frame_metrics):
        if not isinstance(frame, dict):
            raise ValueError(f"frame metric {index} must be an object")
        sequence = frame.get("sequence")
        counts = frame.get("counts")
        count_error = frame.get("count_error")
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"frame metric {index} must identify a sequence")
        if not isinstance(counts, dict) or any(
            isinstance(counts.get(key), bool)
            or not isinstance(counts.get(key), int)
            or counts[key] < 0
            for key in count_keys
        ):
            raise ValueError(f"frame metric {index} has invalid detection counts")
        if (isinstance(count_error, bool) or not isinstance(count_error, (int, float))
                or count_error < 0):
            raise ValueError(f"frame metric {index} has an invalid count error")
        if counts["tp"] + counts["fp"] != counts["predicted"]:
            raise ValueError(f"frame metric {index} has inconsistent predicted counts")
        if counts["tp"] + counts["fn"] != counts["ground_truth"]:
            raise ValueError(f"frame metric {index} has inconsistent ground-truth counts")

        group = grouped.setdefault(sequence.strip(), {
            "counts": Counter({key: 0 for key in count_keys}),
            "count_errors": [],
        })
        group["counts"].update(counts)
        group["count_errors"].append(float(count_error))

    per_sequence = []
    for sequence in sorted(grouped):
        group = grouped[sequence]
        precision, recall, f1 = _metric_scores(group["counts"])
        per_sequence.append({
            "sequence": sequence,
            "images_evaluated": len(group["count_errors"]),
            "counts": {key: group["counts"][key] for key in count_keys},
            "head_point_precision": round(precision, 4),
            "head_point_recall": round(recall, 4),
            "head_point_f1": round(f1, 4),
            "per_frame_count_mae": round(
                sum(group["count_errors"]) / len(group["count_errors"]), 4,
            ),
        })

    sequence_count = len(per_sequence)
    return {
        "sequence_count": sequence_count,
        "macro_head_point_precision": round(
            sum(item["head_point_precision"] for item in per_sequence) / sequence_count, 4,
        ),
        "macro_head_point_recall": round(
            sum(item["head_point_recall"] for item in per_sequence) / sequence_count, 4,
        ),
        "macro_head_point_f1": round(
            sum(item["head_point_f1"] for item in per_sequence) / sequence_count, 4,
        ),
        "macro_per_frame_count_mae": round(
            sum(item["per_frame_count_mae"] for item in per_sequence) / sequence_count, 4,
        ),
        "macro_definition": "Unweighted mean of the per-sequence metric; each sampled sequence has equal weight.",
        "per_sequence": per_sequence,
    }


def sequence_for_image(image_path: Path, source: dict) -> str:
    """Resolve a sampled image to its official source sequence when available."""
    frame_map = source.get("frame_map")
    if frame_map is None:
        sequence = source.get("sequence")
        return sequence if isinstance(sequence, str) and sequence.strip() else "unspecified"
    if not isinstance(frame_map, list):
        raise ValueError("source frame_map must be a list")
    matches = [
        item for item in frame_map
        if isinstance(item, dict) and item.get("sample_file") == image_path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"source frame_map must identify {image_path.name} exactly once")
    sequence = matches[0].get("source_sequence")
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError(f"source frame_map has no sequence for {image_path.name}")
    return sequence.strip()


def validate_sequence_map(images: list[Path], source: dict) -> None:
    """Reject ambiguous or incomplete provenance before model inference starts."""
    frame_map = source.get("frame_map")
    sequences = source.get("sequences")
    if frame_map is None:
        if isinstance(sequences, list) and len(sequences) > 1:
            raise ValueError("multi-sequence samples require source frame_map metadata")
        return
    if not isinstance(frame_map, list) or len(frame_map) != len(images):
        raise ValueError("source frame_map must identify every sample image exactly once")
    mapped_files = [
        item.get("sample_file") if isinstance(item, dict) else None
        for item in frame_map
    ]
    expected_files = [image.name for image in images]
    if (any(not isinstance(name, str) for name in mapped_files)
            or len(set(mapped_files)) != len(mapped_files)
            or set(mapped_files) != set(expected_files)):
        raise ValueError("source frame_map must identify every sample image exactly once")
    mapped_sequences = set()
    for image in images:
        mapped_sequences.add(sequence_for_image(image, source))
    if isinstance(sequences, list) and (
        any(not isinstance(sequence, str) or not sequence.strip() for sequence in sequences)
        or set(sequences) != mapped_sequences
    ):
        raise ValueError("source sequences must match the sequences in frame_map")


def evaluate_sequence(dataset_dir: Path, model_path: Path, confidence_threshold: float) -> dict:
    if not dataset_dir.is_dir() or not model_path.is_file():
        raise ValueError("dataset directory and ONNX model must exist")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be between zero and one")
    images, annotation_text, source = load_sequence(dataset_dir)
    model_manifest, manifest_sha256 = _load_model_manifest(model_path)
    detector = OnnxDetector(
        str(model_path), confidence_threshold=confidence_threshold,
        model_id=model_manifest.get("model_id", "visdrone-rtdetrv4-s"),
        model_version=model_manifest.get("model_revision", "unspecified"),
    )

    totals = Counter({"tp": 0, "fp": 0, "fn": 0, "predicted": 0, "ground_truth": 0})
    absolute_count_errors = []
    frame_metrics = []
    inference_seconds = []
    image_dimensions = set()
    started = time.perf_counter()
    for image_path in images:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not decode {image_path.name}")
        height, width = frame.shape[:2]
        image_dimensions.add((width, height))
        frame_index = int(image_path.stem) - 1
        heads = parse_moving_drone_rows(
            annotation_text, frame_index=frame_index,
            image_width=width, image_height=height,
        )
        inference_started = time.perf_counter()
        predictions = detector.detect(frame)
        inference_seconds.append(time.perf_counter() - inference_started)
        counts = evaluate_head_point_frame(
            predictions, heads, confidence_threshold=confidence_threshold,
        )
        totals.update(counts)
        count_error = abs(counts["predicted"] - counts["ground_truth"])
        absolute_count_errors.append(count_error)
        frame_metrics.append({
            "sequence": sequence_for_image(image_path, source),
            "counts": counts,
            "count_error": count_error,
        })

    precision, recall, f1 = _metric_scores(totals)
    annotation_rows = int(source["annotation_rows"])
    return {
        "dataset": source,
        "model": {
            "id": detector.model_id,
            "revision": detector.model_version,
            "onnx_sha256": _sha256(model_path),
            "manifest_sha256": manifest_sha256,
            "execution_provider": "CPUExecutionProvider",
            "cpu_intra_op_threads": detector.cpu_threads,
        },
        "evaluation": {
            "images_evaluated": len(images),
            "image_dimensions": [list(size) for size in sorted(image_dimensions)],
            "confidence_threshold": confidence_threshold,
            "annotation_rows_total": annotation_rows,
            "off_frame_head_centers_excluded": annotation_rows - totals["ground_truth"],
            "counts": {key: totals[key] for key in ("tp", "fp", "fn", "predicted", "ground_truth")},
            "head_point_precision": round(precision, 4),
            "head_point_recall": round(recall, 4),
            "head_point_f1": round(f1, 4),
            "per_frame_count_mae": round(sum(absolute_count_errors) / len(absolute_count_errors), 4),
            "sequence_summary": summarize_sequence_metrics(frame_metrics),
            "mean_inference_seconds_per_image": round(sum(inference_seconds) / len(inference_seconds), 4),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "method": (
                "Maximum-cardinality one-to-one matching between predicted pedestrian boxes and annotated head centers; "
                "also reports per-frame count MAE. This is not full-body box IoU, identity tracking accuracy, "
                "an event-level crowding metric, or Wuhan University campus accuracy."
            ),
            "limitations": build_evaluation_limitations(source, images_evaluated=len(images)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="directory with 1.jpg..N.jpg, annotations.csv, and source.json")
    parser.add_argument("--model", type=Path, required=True, help="local pedestrian detector ONNX")
    parser.add_argument("--confidence", type=float, default=0.369,
                        help="fixed detector confidence threshold; default matches the current baseline")
    parser.add_argument("--output", type=Path, help="optional UTF-8 JSON report path")
    args = parser.parse_args()
    try:
        report = evaluate_sequence(args.dataset_dir, args.model, args.confidence)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
