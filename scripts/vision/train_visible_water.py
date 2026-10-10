"""Train a research-only binary visible-water model from video or flight splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.floodnet_training import (  # noqa: E402
    build_floodnet_model,
    export_floodnet_onnx,
    get_training_device,
    set_training_seed,
)
from vision.floodwater_training import (  # noqa: E402
    VisibleWaterDataset,
    train_visible_water_model,
    validate_visible_water_training_files,
    validate_visible_water_training_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="dataset root containing the relative image/mask paths in the manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="research artifact directory outside the application repository")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--positive-class-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--pretrained", action="store_true",
                        help="initialize the MobileNetV3-small encoder with ImageNet weights")
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.threads < 1
            or args.tile_size < 32):
        parser.error("epochs, batch size, threads, and tile size must be positive")
    try:
        args.output_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        parser.error("output-dir must stay outside the source repository")
    if args.output_dir.exists():
        parser.error("output-dir already exists; refusing to overwrite research artifacts")
    if not args.dataset_root.is_dir() or not args.manifest.is_file():
        parser.error("dataset root and manifest must exist")
    try:
        manifest_bytes = args.manifest.read_bytes()
        manifest = validate_visible_water_training_manifest(
            json.loads(manifest_bytes.decode("utf-8-sig")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    train_samples = [sample for sample in manifest["samples"] if sample["split"] == "train"]
    validation_samples = [sample for sample in manifest["samples"] if sample["split"] == "validation"]
    try:
        data_preflight = validate_visible_water_training_files(args.dataset_root, manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    set_training_seed(args.seed)
    device = get_training_device()
    train_data = VisibleWaterDataset(
        args.dataset_root, train_samples, tile_size=args.tile_size,
        augment=True, random_crop=True, water_crop_probability=0.6,
    )
    validation_data = VisibleWaterDataset(
        args.dataset_root, validation_samples, tile_size=args.tile_size,
        augment=False, random_crop=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir / "visible-water-best-state.pt"
    model = build_floodnet_model(num_classes=2, pretrained=args.pretrained)
    started = time.monotonic()
    training = train_visible_water_model(
        model, train_loader, validation_loader, device=device,
        checkpoint_path=checkpoint, epochs=args.epochs,
        learning_rate=args.learning_rate,
        positive_class_weight=args.positive_class_weight,
        progress_callback=lambda row: print(
            json.dumps({"epoch_validation": row}, ensure_ascii=False), flush=True,
        ),
    )
    best = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    model.to(device).eval()
    onnx_path = args.output_dir / "visible-water-mobilenetv3-unet.onnx"
    export = export_floodnet_onnx(model, onnx_path, input_size=args.tile_size)
    report = {
        "task": "binary_visible_water_segmentation",
        "dataset": manifest["dataset"],
        "dataset_license": manifest.get("license", "not_declared"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "training_data_preflight": data_preflight,
        "split_unit": manifest["split_unit"],
        "dataset_task": manifest["task"],
        "split_group_counts": {
            split: len(manifest["split_groups"][split])
            for split in ("train", "validation", "test")
        },
        "label_source": manifest.get(
            "label_source",
            "sam2_assisted_pseudo_labels" if manifest["task"] == "visible_water_segmentation"
            else "not_declared",
        ),
        "sample_counts": {"train": len(train_data), "validation": len(validation_data),
                           "held_out_test_used": 0},
        "class_order": ["background_or_non_water", "visible_water"],
        "training_config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "tile_size": args.tile_size,
            "threads": args.threads,
            "device": str(device),
            "pretrained_encoder": args.pretrained,
            "learning_rate": args.learning_rate,
            "positive_class_weight": args.positive_class_weight,
            "water_crop_probability": 0.6,
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
            "validation_crop": "fixed center tile; centered on road ROI when provided",
        },
        "training": training,
        "onnx_export": export,
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx_sha256": _sha256(onnx_path),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "artifacts_status": "local research only; not enabled or deployed",
        "limitations": list(manifest.get("limitations", [])) + [
            "The declared test split is not used for training, checkpoint selection, or threshold fitting.",
            "Validation uses one deterministic center tile per sample and is only a model-selection signal.",
            "Dataset permissions and derived-weight deployment rights must be checked before distribution.",
            "A strong public-set result still requires campus UAV evaluation for shallow road water.",
        ],
    }
    report_path = args.output_dir / "visible-water-training-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "onnx": str(onnx_path),
        "device": str(device),
        "best_epoch": training["best_epoch"],
        "validation_visible_water_iou": training["best_validation_visible_water_iou"],
        "elapsed_seconds": report["elapsed_seconds"],
        "production_enabled": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
