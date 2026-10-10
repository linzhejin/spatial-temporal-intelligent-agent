import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.vision.prepare_floodwater_training_data import main as prepare_main
from scripts.vision.train_visible_water import main as train_main
from vision.floodwater_training import (
    VisibleWaterDataset,
    build_floodwater_manifest,
    evaluate_visible_water_model,
    train_visible_water_model,
    validate_visible_water_manifest,
    validate_visible_water_training_manifest,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tiny_metadata(tmp_path):
    chunks = tmp_path / "chunks.csv"
    samples = tmp_path / "samples.csv"
    _write_csv(chunks, ["chunk_id", "source_video_id", "split", "video_path", "mask_directory"], [
        {"chunk_id": "chunk-a", "source_video_id": "video-a", "split": "train",
         "video_path": "data/videos/chunk-a.mp4", "mask_directory": "data/masks/chunk-a"},
        {"chunk_id": "chunk-b", "source_video_id": "video-b", "split": "val",
         "video_path": "data/videos/chunk-b.mp4", "mask_directory": "data/masks/chunk-b"},
        {"chunk_id": "chunk-c", "source_video_id": "video-c", "split": "test",
         "video_path": "data/videos/chunk-c.mp4", "mask_directory": "data/masks/chunk-c"},
    ])
    sample_rows = []
    for chunk_id, video_id, split in (
        ("chunk-a", "video-a", "train"),
        ("chunk-b", "video-b", "val"),
        ("chunk-c", "video-c", "test"),
    ):
        for frame_index in (0, 1, 24, 25, 50):
            sample_rows.append({
                "sample_id": f"{chunk_id}/{frame_index:05d}",
                "source_video_id": video_id,
                "chunk_id": chunk_id,
                "split": split,
                "chunk_frame_index": str(frame_index),
                "chunk_time_seconds": str(frame_index / 25),
                "mask_path": f"data/masks/{chunk_id}/{frame_index:05d}.png",
                "video_path": f"data/videos/{chunk_id}.mp4",
            })
    _write_csv(samples, list(sample_rows[0]), sample_rows)
    return chunks, samples


def test_manifest_samples_at_one_second_and_excludes_pseudo_test_video(tmp_path):
    chunks, samples = _tiny_metadata(tmp_path)

    manifest = build_floodwater_manifest(chunks, samples, sample_period_frames=25)

    assert manifest["split_unit"] == "source_video_id"
    assert manifest["split_groups"] == {
        "train": ["video-a"], "validation": ["video-b"], "test": ["video-c"],
    }
    assert [sample["chunk_frame_index"] for sample in manifest["samples"]] == [0, 25, 50, 0, 25, 50]
    assert {sample["split"] for sample in manifest["samples"]} == {"train", "validation"}
    assert all(sample["source_video_id"] != "video-c" for sample in manifest["samples"])


def test_manifest_rejects_source_video_leakage_across_splits(tmp_path):
    chunks, samples = _tiny_metadata(tmp_path)
    content = chunks.read_text(encoding="utf-8").replace("video-b,val", "video-a,val")
    chunks.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="source video appears in multiple splits"):
        build_floodwater_manifest(chunks, samples, sample_period_frames=25)


def test_training_manifest_rejects_test_samples_and_cross_split_group_leakage(tmp_path):
    chunks, samples = _tiny_metadata(tmp_path)
    manifest = build_floodwater_manifest(chunks, samples, sample_period_frames=25)

    validated = validate_visible_water_manifest(manifest)

    assert len(validated["samples"]) == 6
    leaked = dict(manifest)
    leaked["samples"] = [*manifest["samples"], {
        "sample_id": "test-sample", "source_video_id": "video-c", "group_id": "video-c",
        "split": "test", "image": "derived/test.jpg", "mask": "data/test.png",
    }]
    with pytest.raises(ValueError, match="must not contain test split samples"):
        validate_visible_water_manifest(leaked)


def test_campus_training_manifest_uses_manual_flight_groups_and_excludes_test_frames():
    manifest = {
        "schema_version": 1,
        "task": "visible_water_road_segmentation",
        "dataset": "WHU UAV pilot",
        "split_unit": "flight_id",
        "mask_encoding": "binary_0_255",
        "label_source": "manual",
        "split_groups": {
            "train": ["flight-1"],
            "validation": ["flight-2"],
            "test": ["flight-3"],
        },
        "samples": [
            {"sample_id": "flight-1-frame-1", "split": "train", "group_id": "flight-1",
             "image": "train/frame-1.jpg", "mask": "train/frame-1.png",
             "road_surface_polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8]]},
            {"sample_id": "flight-2-frame-1", "split": "validation", "group_id": "flight-2",
             "image": "validation/frame-1.jpg", "mask": "validation/frame-1.png",
             "road_surface_polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8]]},
        ],
    }

    validated = validate_visible_water_training_manifest(manifest)

    assert [item["split"] for item in validated["samples"]] == ["train", "validation"]
    assert validated["split_unit"] == "flight_id"

    test_leak = {**manifest, "samples": [*manifest["samples"], {
        "sample_id": "flight-3-frame-1", "split": "test", "group_id": "flight-3",
        "image": "test/frame-1.jpg", "mask": "test/frame-1.png",
        "road_surface_polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8]],
    }]}
    with pytest.raises(ValueError, match="test samples must not enter model training"):
        validate_visible_water_training_manifest(test_leak)


def test_campus_training_manifest_rejects_reused_flight_and_unsafe_paths():
    manifest = {
        "schema_version": 1,
        "task": "visible_water_road_segmentation",
        "dataset": "WHU UAV pilot",
        "split_unit": "flight_id",
        "mask_encoding": "binary_0_255",
        "label_source": "manual",
        "split_groups": {
            "train": ["flight-1"], "validation": ["flight-2"], "test": ["flight-3"],
        },
        "samples": [
            {"sample_id": "a", "split": "train", "group_id": "flight-1",
             "image": "../outside.jpg", "mask": "train/a.png",
             "road_surface_polygon": [[0, 0], [1, 0], [1, 1]]},
            {"sample_id": "b", "split": "validation", "group_id": "flight-2",
             "image": "validation/b.jpg", "mask": "validation/b.png",
             "road_surface_polygon": [[0, 0], [1, 0], [1, 1]]},
        ],
    }

    with pytest.raises(ValueError, match="image must stay inside the dataset"):
        validate_visible_water_training_manifest(manifest)

    reused = {**manifest, "samples": [dict(manifest["samples"][0], image="train/a.jpg"),
                                        dict(manifest["samples"][1], group_id="flight-1")]}
    with pytest.raises(ValueError, match="conflicts with its flight split"):
        validate_visible_water_training_manifest(reused)


def test_prepare_cli_dry_run_reports_split_counts_without_writing_frames(tmp_path, capsys):
    chunks, samples = _tiny_metadata(tmp_path)
    corpus = tmp_path / "corpus"
    metadata = corpus / "metadata"
    metadata.mkdir(parents=True)
    shutil.copyfile(chunks, metadata / "chunks.csv")
    shutil.copyfile(samples, metadata / "samples.csv")

    assert prepare_main(["--dataset-root", str(corpus), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert '"train": 3' in output
    assert '"validation": 3' in output
    assert '"test": 0' in output
    assert not (corpus / "derived").exists()


def test_visible_water_training_cli_help_is_available():
    with pytest.raises(SystemExit) as exit_info:
        train_main(["--help"])

    assert exit_info.value.code == 0


def test_visible_water_training_cli_help_does_not_load_torch_runtime():
    script = Path("scripts/vision/train_visible_water.py").resolve()

    completed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert b"--dataset-root" in completed.stdout


def test_visible_water_dataset_maps_255_to_foreground_class(tmp_path):
    import numpy as np
    import torch
    from PIL import Image

    Image.new("RGB", (32, 32), (20, 40, 60)).save(tmp_path / "frame.jpg")
    Image.fromarray(np.asarray([[0] * 16 + [255] * 16] * 32, dtype=np.uint8)).save(
        tmp_path / "mask.png",
    )
    dataset = VisibleWaterDataset(
        tmp_path,
        [{"image": "frame.jpg", "mask": "mask.png", "split": "train", "group_id": "flight-a"}],
        tile_size=32,
        augment=False,
        random_crop=False,
    )

    image, labels = dataset[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert labels.dtype == torch.long
    assert set(torch.unique(labels).tolist()) == {0, 1}
    assert labels[0].tolist() == [0] * 16 + [1] * 16


def test_visible_water_dataset_limits_campus_training_labels_to_road_roi(tmp_path):
    import numpy as np
    import torch
    from PIL import Image

    Image.new("RGB", (32, 32), (20, 40, 60)).save(tmp_path / "frame.jpg")
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(tmp_path / "mask.png")
    dataset = VisibleWaterDataset(
        tmp_path,
        [{"image": "frame.jpg", "mask": "mask.png", "split": "train",
          "group_id": "flight-a",
          "road_surface_polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}],
        tile_size=32,
        augment=False,
        random_crop=False,
    )

    _, labels = dataset[0]

    assert torch.all(labels[:, :8] == 1)
    assert torch.all(labels[:, 24:] == 0)


def test_binary_water_evaluator_reports_foreground_iou():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    images = torch.zeros((1, 3, 2, 2))
    labels = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.long)
    model = nn.Conv2d(3, 2, kernel_size=1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 1.0])

    report = evaluate_visible_water_model(
        model, DataLoader(TensorDataset(images, labels), batch_size=1),
        device=torch.device("cpu"),
    )

    assert report["visible_water"]["tp"] == 2
    assert report["visible_water"]["fp"] == 2
    assert report["visible_water"]["fn"] == 0
    assert report["visible_water"]["iou"] == pytest.approx(0.5)


def test_binary_water_training_saves_state_dict_and_validation_iou(tmp_path):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(7)
    images = torch.rand((2, 3, 8, 8))
    labels = torch.zeros((2, 8, 8), dtype=torch.long)
    labels[:, 2:6, 2:6] = 1
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Conv2d(3, 2, kernel_size=1)
    checkpoint = tmp_path / "visible-water.pt"

    report = train_visible_water_model(
        model,
        loader,
        loader,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        epochs=1,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert set(saved) >= {"model_state_dict", "epoch", "validation_visible_water_iou"}
    assert report["epochs_completed"] == 1
    assert report["best_validation_visible_water_iou"] is not None
