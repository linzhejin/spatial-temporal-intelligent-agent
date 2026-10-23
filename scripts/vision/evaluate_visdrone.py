#!/usr/bin/env python3
"""Evaluate the local ONNX detector on an annotated VisDrone validation ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from vision.engine import OnnxDetector
from vision.evaluation import evaluate_visdrone_frame, summarize_detection_counts


def _empty_counts():
    return {group: Counter({"tp": 0, "fp": 0, "fn": 0, "ignored": 0})
            for group in ("pedestrian", "vehicle")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model_manifest(model_path: Path) -> tuple[dict, str | None]:
    """Load and verify adjacent model metadata when the artifact has a manifest."""
    manifest_path = model_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return {}, None
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("model manifest must contain a JSON object")
    expected_sha256 = metadata.get("onnx_sha256")
    if expected_sha256:
        actual_sha256 = _sha256(model_path)
        if str(expected_sha256).lower() != actual_sha256:
            raise ValueError("model ONNX SHA256 does not match its adjacent manifest")
    return metadata, _sha256(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="VisDrone2019-DET-val.zip; files are read in place without extraction")
    parser.add_argument("--model", type=Path, default=ROOT / "models/vision/visdrone-rtdetrv4-s.onnx")
    parser.add_argument("--confidence", type=float, default=0.369)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0,
                        help="optional first-N image smoke sample; 0 evaluates every image")
    parser.add_argument("--selection", type=Path,
                        help="fixed JSON manifest of image paths; preferred for reproducible V2 audit")
    parser.add_argument("--output", type=Path, help="optional UTF-8 JSON output path")
    args = parser.parse_args()
    if not args.archive.is_file() or not args.model.is_file():
        parser.error("archive and ONNX model files must exist")
    if args.limit < 0:
        parser.error("--limit must be zero or a positive image count")
    if args.selection and args.limit:
        parser.error("--selection and --limit are mutually exclusive")

    try:
        model_manifest, model_manifest_sha256 = _load_model_manifest(args.model)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"model manifest validation failed: {error}")

    started = time.monotonic()
    detector = OnnxDetector(
        str(args.model), confidence_threshold=args.confidence,
        model_id=model_manifest.get("model_id"),
        model_version=model_manifest.get("model_revision"),
    )
    totals = _empty_counts()
    class_counts = Counter()
    decode_failures = []
    inference_seconds = []
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        images = sorted(name for name in names if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg"}
                        and "/images/" in f"/{name}")
        annotation_names = {
            PurePosixPath(name).stem: name for name in names
            if PurePosixPath(name).suffix.lower() == ".txt" and "/annotations/" in f"/{name}"
        }
        if not images:
            parser.error("archive contains no VisDrone image files")
        missing_annotations = [name for name in images if PurePosixPath(name).stem not in annotation_names]
        if missing_annotations:
            parser.error(f"archive is missing {len(missing_annotations)} matching annotation files")
        if args.selection:
            selection = json.loads(args.selection.read_text(encoding="utf-8"))
            if (selection.get("format") != "visdrone-v2-selection"
                    or selection.get("archive_sha256") != _sha256(args.archive)
                    or not isinstance(selection.get("images"), list)
                    or len(selection["images"]) != len(set(selection["images"]))
                    or not selection["images"]
                    or any(name not in images for name in selection["images"])):
                parser.error("selection is invalid or does not match this archive")
            selected = selection["images"]
        else:
            selected = images[:args.limit] if args.limit else images
        for index, image_name in enumerate(selected, start=1):
            relative_name = PurePosixPath(image_name)
            encoded = np.frombuffer(archive.read(image_name), dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                decode_failures.append(relative_name.name)
                continue
            inference_start = time.monotonic()
            predictions = detector.detect(frame)
            inference_seconds.append(time.monotonic() - inference_start)
            for prediction in predictions:
                class_counts[prediction["label"]] += 1
            annotation = archive.read(annotation_names[relative_name.stem]).decode("utf-8-sig")
            frame_counts = evaluate_visdrone_frame(
                predictions, annotation,
                confidence_threshold=args.confidence, iou_threshold=args.iou,
            )
            for group, values in frame_counts.items():
                totals[group].update(values)
            if index % 50 == 0 or index == len(selected):
                elapsed = time.monotonic() - started
                print(f"evaluated {index}/{len(selected)} images ({elapsed:.1f}s)", file=sys.stderr)

    summary = summarize_detection_counts({
        group: dict(values) for group, values in totals.items()
    })
    report = {
        "dataset": "VisDrone2019-DET-val",
        "dataset_archive_sha256": _sha256(args.archive),
        "model": detector.model_id,
        "model_revision": detector.model_version,
        "model_sha256": _sha256(args.model),
        "model_manifest_sha256": model_manifest_sha256,
        "execution_provider": "CPUExecutionProvider",
        "image_count_in_archive": len(images),
        "image_count_evaluated": len(selected) - len(decode_failures),
        "selection_manifest_sha256": _sha256(args.selection) if args.selection else None,
        "decode_failures": decode_failures,
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou,
        "groups": summary,
        "detections_by_class": dict(sorted(class_counts.items())),
        "mean_inference_seconds_per_image": round(
            sum(inference_seconds) / len(inference_seconds), 4,
        ) if inference_seconds else None,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "interpretation": (
            "Precision/recall at one confidence and IoU operating point on public VisDrone aerial images. "
            "This is not mAP, not an event-level congestion metric, and not Wuhan University campus accuracy."
        ),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["image_count_evaluated"] == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
