"""Evaluate a FloodNet or binary visible-water ONNX model on manual water labels.

This is a global visible-water benchmark, not road-only flooding or campus
puddle validation. It never calibrates thresholds from the evaluation set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.specialized import _image_tensor  # noqa: E402
from vision.tiling import tile_windows  # noqa: E402
from vision.visible_water_evaluation import (  # noqa: E402
    binary_segmentation_counts,
    bootstrap_iou_interval,
    discover_manual_pairs,
    manual_evaluation_limitations,
    metrics_from_counts,
    validate_binary_water_mask,
    visible_water_prediction_mask,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(model_path: Path, dataset_root: Path, *, threads: int = 2,
             bootstrap_seed: int = 20261011, replicates: int = 1000) -> dict:
    pairs = discover_manual_pairs(dataset_root, expected_count=700)
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("model must have one image input and one segmentation output")
    input_shape = inputs[0].shape
    output_shape = outputs[0].shape
    if (len(input_shape) != 4 or input_shape[1] not in (3, "3")
            or not all(isinstance(value, int) and value > 0 for value in input_shape[2:])):
        raise ValueError("model input must be fixed [N,3,H,W]")
    if (len(output_shape) != 4 or output_shape[1] not in (2, 10, "2", "10")):
        raise ValueError("model output must use either two visible-water or ten FloodNet classes")
    class_count = int(output_shape[1])
    tile_height, tile_width = int(input_shape[2]), int(input_shape[3])
    overlap = max(1, min(128, min(tile_width, tile_height) // 4))
    image_counts = []
    dimensions = {}
    pair_fingerprints = []
    started = time.monotonic()
    for index, pair in enumerate(pairs, 1):
        frame = cv2.imread(str(pair["image"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not decode {pair['image'].name}")
        with Image.open(pair["mask"]) as source:
            mask = np.asarray(source.convert("L"))
        validate_binary_water_mask(mask)
        height, width = mask.shape
        if frame.shape[:2] != (height, width):
            raise ValueError(f"image/mask size mismatch for {pair['sample_id']}")
        dimensions[f"{width}x{height}"] = dimensions.get(f"{width}x{height}", 0) + 1
        logits_sum = np.zeros((class_count, height, width), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.float32)
        for x, y in tile_windows(
                width=width, height=height, tile_size=(tile_width, tile_height), overlap=overlap):
            x2, y2 = min(width, x + tile_width), min(height, y + tile_height)
            tile = frame[y:y2, x:x2]
            logits = session.run(
                [outputs[0].name], _image_tensor(tile, inputs[0].name, (tile_width, tile_height)),
            )[0][0]
            if logits.shape[-2:] != (y2 - y, x2 - x):
                logits = np.stack([
                    cv2.resize(channel, (x2 - x, y2 - y), interpolation=cv2.INTER_LINEAR)
                    for channel in logits
                ])
            logits_sum[:, y:y2, x:x2] += logits
            coverage[y:y2, x:x2] += 1
        labels = (logits_sum / coverage[None, :, :]).argmax(axis=0)
        predicted_water = visible_water_prediction_mask(labels, class_count=class_count)
        truth_water = mask == 255
        image_counts.append(binary_segmentation_counts(predicted_water, truth_water))
        pair_fingerprints.append(
            f"{pair['sample_id']}:{_sha256(pair['image'])}:{_sha256(pair['mask'])}"
        )
        if index % 50 == 0 or index == len(pairs):
            print(f"evaluated {index}/{len(pairs)}", flush=True)

    totals = {key: sum(row[key] for row in image_counts) for key in ("tp", "fp", "fn")}
    per_image_iou = [metrics_from_counts(row)["iou"] for row in image_counts]
    report = {
        "task": "visible_water_segmentation_external_manual_eval",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Floodwater Dataset manual evaluation set v1.0.0",
        "dataset_license": "GPL-3.0-only",
        "sample_count": len(pairs),
        "dataset_manifest_sha256": hashlib.sha256(
            "\n".join(pair_fingerprints).encode("utf-8"),
        ).hexdigest(),
        "image_dimensions": dimensions,
        "label_semantics": "all visibly annotated water pixels; not road-only flooding or campus puddles",
        "model_sha256": _sha256(model_path),
        "model_class_count": class_count,
        "prediction_rule": {
            "tiling": [tile_width, tile_height],
            "overlap_pixels": overlap,
            "overlap_merge": "average logits",
            "water_classes": ({"1": "visible_water"} if class_count == 2
                              else {"3": "flooded_road", "5": "water"}),
            "decision": "argmax; no threshold tuned on this set",
        },
        "aggregate": {
            **metrics_from_counts(totals),
            "macro_image_iou": float(np.mean([value for value in per_image_iou if value is not None])),
        },
        "image_id_bootstrap_95pct_iou_ci": bootstrap_iou_interval(
            image_counts, seed=bootstrap_seed, replicates=replicates,
        ),
        "bootstrap": {"unit": "image_id", "seed": bootstrap_seed, "replicates": replicates},
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": manual_evaluation_limitations(class_count=class_count),
        "per_image": [
            {"sample_id": pair["sample_id"], **row, "iou": metrics_from_counts(row)["iou"]}
            for pair, row in zip(pairs, image_counts)
        ],
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="2-class visible-water or 10-class FloodNet ONNX model")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="manual dataset folder containing images/ and masks/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error("model file does not exist")
    if args.threads < 1 or args.bootstrap_replicates < 1:
        parser.error("threads and bootstrap replicates must be positive")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        parser.error("output exists; pass --overwrite to replace it")
    report = evaluate(args.model, args.dataset_root, threads=args.threads,
                      bootstrap_seed=args.bootstrap_seed,
                      replicates=args.bootstrap_replicates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": report["aggregate"],
                      "ci": report["image_id_bootstrap_95pct_iou_ci"],
                      "elapsed_seconds": report["elapsed_seconds"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
