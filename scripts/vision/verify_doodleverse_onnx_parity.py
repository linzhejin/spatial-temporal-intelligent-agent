"""Compare FP32 ONNX v7 with frozen original-model probability maps."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from skimage import io as skio
from skimage.transform import resize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args) -> dict:
    from vision.doodleverse_preprocessing import (
        extract_flooded_road_probability,
        prepare_doodleverse_rgb,
    )
    import onnxruntime as ort

    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    if protocol.get("protocol_version") != "road-inundation-cross-event-v1.2":
        raise ValueError("Unexpected or superseded frozen model protocol")
    if protocol.get("frozen_before_any_model_inference") is not True:
        raise ValueError("The comparison protocol was not frozen before inference")
    model_hash = sha256_file(args.model)
    meta_path = args.model.with_suffix(args.model.suffix + ".metadata.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("output_sha256") != model_hash:
        raise ValueError("ONNX file hash differs from its export metadata")
    if metadata.get("source_sha256") != protocol["models"]["candidate"]["sha256"]:
        raise ValueError("ONNX conversion source differs from frozen v7 weights")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    reference_root = args.reference_run
    protocol_hash = sha256_file(args.protocol)
    lock_path = reference_root / "flood_thresholds_frozen_v1_2.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("protocol_sha256") != protocol_hash
            or lock.get("model_hashes", {}).get("candidate") != metadata["source_sha256"]):
        raise ValueError("Frozen threshold lock does not match this protocol/model")
    extraction_manifest = reference_root / "extraction_manifest_v1_2.json"
    extraction_hash = sha256_file(extraction_manifest)
    test_metrics_path = reference_root / "flood_test_metrics_v1_2.json"
    test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    if (test_metrics.get("protocol_sha256") != protocol_hash
            or test_metrics.get("threshold_lock_sha256") != sha256_file(lock_path)
            or test_metrics.get("extraction_manifest_sha256") != extraction_hash):
        raise ValueError("Frozen test report does not match threshold lock/extraction manifest")
    rows = []
    aggregate_intersection = 0
    aggregate_union = 0
    split_results = {}
    started_all = time.perf_counter()
    for split in ("validation", "test"):
        records = [item for item in protocol["records"] if item.get("split") == split]
        expected_count = protocol["frozen_evaluation_counts"][split]["scenes"]
        if len(records) != expected_count:
            raise ValueError(f"Frozen {split} count mismatch: {len(records)} != {expected_count}")
        reference_path = reference_root / f"probabilities_doodleverse_{split}_v1_2.npz"
        reference_predictions_path = reference_root / f"predictions_doodleverse_{split}_v1_2.json"
        reference_hash = sha256_file(reference_path)
        reference_predictions = json.loads(reference_predictions_path.read_text(encoding="utf-8"))
        if (reference_predictions.get("protocol_sha256") != protocol_hash
                or reference_predictions.get("extraction_manifest_sha256") != extraction_hash
                or reference_predictions.get("model_sha256") != metadata["source_sha256"]
                or reference_predictions.get("sample_count") != expected_count
                or reference_predictions.get("success_count") != expected_count
                or reference_predictions.get("failure_count") != 0
                or reference_predictions.get("probability_maps_sha256") != reference_hash):
            raise ValueError(f"Frozen {split} reference prediction integrity check failed")
        if split == "validation":
            expected_json_hash = lock.get("validation_artifacts", {}).get(
                "predictions_doodleverse_validation_v1_2.json",
            )
            expected_map_hash = lock.get("validation_artifacts", {}).get(
                "probabilities_doodleverse_validation_v1_2.npz",
            )
            if (sha256_file(reference_predictions_path) != expected_json_hash
                    or reference_hash != expected_map_hash):
                raise ValueError("Frozen validation reference artifacts differ from the threshold lock")
        with np.load(reference_path, allow_pickle=False) as reference_maps:
            split_deltas = []
            split_union = split_intersection = 0
            split_decisions_equal = True
            for index, record in enumerate(records, start=1):
                scene_id = record["scene_id"]
                image_path = args.samples / split / f"{scene_id}_image.png"
                mask_path = args.samples / split / f"{scene_id}_mask.png"
                if sha256_file(image_path) != record["image_sha256"]:
                    raise ValueError(f"Frozen image hash mismatch: {scene_id}")
                if sha256_file(mask_path) != record["mask_sha256"]:
                    raise ValueError(f"Frozen scoring mask hash mismatch: {scene_id}")
                image = skio.imread(str(image_path))
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"Expected uint8 RGB source image: {scene_id}")
                # Ground-truth mask is read only after inference and used only for scoring.
                started = time.perf_counter()
                model_input = prepare_doodleverse_rgb(image)
                class_probability = extract_flooded_road_probability(
                    session.run([output_name], {input_name: model_input})[0],
                )
                inference_seconds = time.perf_counter() - started
                onnx_probability = resize(
                    class_probability, image.shape[:2], preserve_range=False, clip=True,
                ).astype(np.float32)
                truth = skio.imread(str(mask_path))
                if truth.ndim != 2 or truth.shape != image.shape[:2]:
                    raise ValueError(f"Unexpected frozen road mask dimensions: {scene_id}")
                road_region = (truth == 1) | (truth == 2)
                if not np.any(road_region):
                    raise ValueError(f"Frozen scoring mask has no road pixels: {scene_id}")
                reference_probability = np.asarray(reference_maps[scene_id], dtype=np.float32)
                if reference_probability.shape != onnx_probability.shape:
                    raise ValueError(f"Reference and ONNX output shape mismatch: {scene_id}")
                onnx_mask = (onnx_probability >= 0.001) & road_region
                reference_mask = (reference_probability >= 0.001) & road_region
                union = int(np.logical_or(onnx_mask, reference_mask).sum())
                intersection = int(np.logical_and(onnx_mask, reference_mask).sum())
                aggregate_union += union
                aggregate_intersection += intersection
                split_union += union
                split_intersection += intersection
                onnx_ratio = float(onnx_mask.sum()) / int(road_region.sum())
                reference_ratio = float(reference_mask.sum()) / int(road_region.sum())
                delta_pp = abs(onnx_ratio - reference_ratio) * 100
                onnx_trigger = onnx_ratio >= 0.08
                reference_trigger = reference_ratio >= 0.08
                decisions_equal = onnx_trigger == reference_trigger
                split_decisions_equal &= decisions_equal
                split_deltas.append(delta_pp)
                rows.append({
                    "scene_id": scene_id,
                    "split": split,
                    "image_sha256": record["image_sha256"],
                    "scoring_mask_sha256": record["mask_sha256"],
                    "road_pixels": int(road_region.sum()),
                    "onnx_area_ratio": onnx_ratio,
                    "reference_area_ratio": reference_ratio,
                    "area_ratio_delta_percentage_points": delta_pp,
                    "onnx_candidate": onnx_trigger,
                    "reference_candidate": reference_trigger,
                    "candidate_decision_equal": decisions_equal,
                    "mask_intersection": intersection,
                    "mask_union": union,
                    "mask_iou": intersection / union if union else 1.0,
                    "probability_mean_absolute_error_on_road": float(
                        np.abs(onnx_probability[road_region] -
                               reference_probability[road_region]).mean(),
                    ),
                    "onnx_inference_seconds": inference_seconds,
                })
                if index % 10 == 0 or index == len(records):
                    print(f"{split}: {index}/{len(records)}", flush=True)
        split_results[split] = {
            "sample_count": len(records),
            "max_area_ratio_delta_percentage_points": max(split_deltas, default=0.0),
            "mask_iou": split_intersection / split_union if split_union else 1.0,
            "candidate_decisions_equal": split_decisions_equal,
            "reference_predictions_sha256": sha256_file(
                reference_root / f"predictions_doodleverse_{split}_v1_2.json",
            ),
            "reference_probability_maps_sha256": reference_hash,
        }

    overall_iou = aggregate_intersection / aggregate_union if aggregate_union else 1.0
    max_delta = max(
        (row["area_ratio_delta_percentage_points"] for row in rows), default=0.0,
    )
    decisions_equal = all(row["candidate_decision_equal"] for row in rows)
    passed = max_delta <= 0.5 and overall_iou >= 0.99 and decisions_equal
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "acceptance_criteria": {
            "max_area_ratio_delta_percentage_points": 0.5,
            "aggregate_binary_mask_iou": 0.99,
            "candidate_decisions_must_match": True,
        },
        "observed": {
            "max_area_ratio_delta_percentage_points": max_delta,
            "aggregate_binary_mask_iou": overall_iou,
            "candidate_decisions_equal": decisions_equal,
        },
        "frozen_protocol_sha256": protocol_hash,
        "frozen_threshold_lock_sha256": sha256_file(lock_path),
        "frozen_extraction_manifest_sha256": extraction_hash,
        "frozen_test_report_sha256": sha256_file(test_metrics_path),
        "onnx_sha256": model_hash,
        "source_weights_sha256": metadata["source_sha256"],
        "export_metadata_sha256": sha256_file(meta_path),
        "preprocessor_source_sha256": sha256_file(
            Path(__file__).resolve().parents[2] / "vision" / "doodleverse_preprocessing.py",
        ),
        "preprocessing": metadata["input_type"],
        "probability_threshold": 0.001,
        "candidate_area_threshold": 0.08,
        "labels_used_only_for_scoring": True,
        "device": "CPU",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_image": __import__("skimage").__version__,
            "onnxruntime": ort.__version__,
        },
        "splits": split_results,
        "sample_results": rows,
        "elapsed_seconds": time.perf_counter() - started_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite parity report: {args.output}")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "max_area_ratio_delta_percentage_points": report["observed"][
            "max_area_ratio_delta_percentage_points"],
        "aggregate_binary_mask_iou": report["observed"]["aggregate_binary_mask_iou"],
        "candidate_decisions_equal": report["observed"]["candidate_decisions_equal"],
        "samples": len(report["sample_results"]),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
