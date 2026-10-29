import pytest
from PIL import Image

try:
    import torch as _torch_probe  # noqa: F401 - verify the optional training runtime can load
except Exception as error:  # pragma: no cover - depends on the developer's isolated training runtime
    pytest.skip(
        f"FloodNet training tests require the isolated PyTorch environment: {type(error).__name__}",
        allow_module_level=True,
    )

from vision.floodnet_training import (
    FloodNetSegmentationDataset,
    evaluate_floodnet,
    floodnet_focus_sample_weights,
    tile_windows,
    train_floodnet,
    validate_floodnet_manifest,
)


def _manifest():
    return {
        "schema_version": 1,
        "task": "flooded_road_segmentation",
        "dataset": "FloodNet",
        "split_unit": "scene",
        "split_groups": {
            "train": ["flight-a"],
            "validation": ["flight-b"],
            "test": ["flight-c"],
        },
        "samples": [
            {"image": "train/a.jpg", "mask": "train/a.png", "group_id": "flight-a", "split": "train"},
            {"image": "val/b.jpg", "mask": "val/b.png", "group_id": "flight-b", "split": "validation"},
            {"image": "test/c.jpg", "mask": "test/c.png", "group_id": "flight-c", "split": "test"},
        ],
    }


def test_floodnet_manifest_accepts_scene_disjoint_train_validation_and_test_splits():
    validated = validate_floodnet_manifest(_manifest())

    assert validated["split_groups"]["train"] == ["flight-a"]
    assert len(validated["samples"]) == 3


def test_floodnet_manifest_rejects_scene_leakage_between_splits():
    manifest = _manifest()
    manifest["split_groups"]["test"].append("flight-a")

    with pytest.raises(ValueError, match="must not overlap"):
        validate_floodnet_manifest(manifest)


@pytest.mark.parametrize("unsafe_path", ["../outside.png", "C:/outside.png", "/outside.png"])
def test_floodnet_manifest_rejects_paths_outside_dataset_root(unsafe_path):
    manifest = _manifest()
    manifest["samples"][0]["image"] = unsafe_path

    with pytest.raises(ValueError, match="relative|outside"):
        validate_floodnet_manifest(manifest)


def test_floodnet_manifest_rejects_sample_assigned_to_the_wrong_scene_split():
    manifest = _manifest()
    manifest["samples"][0]["split"] = "test"

    with pytest.raises(ValueError, match="does not match"):
        validate_floodnet_manifest(manifest)


def test_floodnet_tiles_cover_full_raster_with_fixed_size_windows():
    windows = tile_windows(width=900, height=600, tile_size=512, overlap=128)

    assert windows == [(0, 0), (388, 0), (0, 88), (388, 88)]


def test_floodnet_raster_smaller_than_tile_uses_one_origin_window():
    assert tile_windows(width=320, height=240, tile_size=512, overlap=64) == [(0, 0)]


@pytest.mark.parametrize("tile_size,overlap", [(0, 0), (512, -1), (512, 512), (512, 700)])
def test_floodnet_tile_configuration_rejects_invalid_size_or_overlap(tile_size, overlap):
    with pytest.raises(ValueError):
        tile_windows(width=900, height=600, tile_size=tile_size, overlap=overlap)


def test_floodnet_dataset_reads_index_mask_and_returns_normalized_tensors(tmp_path):
    import numpy as np

    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(tmp_path / "image.png")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 4:] = 3
    Image.fromarray(mask).save(tmp_path / "mask.png")
    sample = {"image": "image.png", "mask": "mask.png", "group_id": "id1", "split": "train"}

    dataset = FloodNetSegmentationDataset(tmp_path, [sample], tile_size=4, overlap=0)
    image, labels = dataset[1]

    assert tuple(image.shape) == (3, 4, 4)
    assert image.dtype == __import__("torch").float32
    assert labels.dtype == __import__("torch").int64
    assert set(labels.unique().tolist()) == {3}
    assert float(image.max()) <= 3.0


def test_floodnet_dataset_rejects_different_sized_image_and_mask(tmp_path):
    # A mask-grid evaluator convention alone does not prove image registration.
    Image.new("RGB", (12, 8)).save(tmp_path / "image.png")
    Image.new("L", (8, 8)).save(tmp_path / "mask.png")
    sample = {"image": "image.png", "mask": "mask.png", "group_id": "id1", "split": "train"}

    with pytest.raises(ValueError, match="dimensions differ"):
        FloodNetSegmentationDataset(tmp_path, [sample], tile_size=8, overlap=0)


def test_floodnet_tiled_evaluation_reuses_decoded_source_image_and_mask(tmp_path):
    import numpy as np
    from unittest.mock import patch

    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(tmp_path / "mask.png")
    sample = {"image": "image.png", "mask": "mask.png", "group_id": "id1", "split": "test"}
    dataset = FloodNetSegmentationDataset(tmp_path, [sample], tile_size=8, overlap=0)

    with patch("PIL.Image.open", wraps=Image.open) as image_open:
        dataset[0]
        dataset[1]

    assert image_open.call_count == 2


def test_floodnet_dataset_rejects_unknown_label_values(tmp_path):
    import numpy as np

    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    mask = np.full((8, 8), 254, dtype=np.uint8)
    Image.fromarray(mask).save(tmp_path / "mask.png")
    sample = {"image": "image.png", "mask": "mask.png", "group_id": "id1", "split": "train"}
    dataset = FloodNetSegmentationDataset(tmp_path, [sample], tile_size=8, overlap=0)

    with pytest.raises(ValueError, match="label values"):
        dataset[0]


def test_floodnet_training_crop_can_focus_a_rare_class_without_leaving_image_bounds(tmp_path):
    import numpy as np

    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[15, 15] = 3
    Image.fromarray(mask).save(tmp_path / "mask.png")
    sample = {"image": "image.png", "mask": "mask.png", "group_id": "id1", "split": "train"}
    dataset = FloodNetSegmentationDataset(
        tmp_path, [sample], tile_size=8, overlap=0, one_tile_per_sample=True,
        focus_class_id=3, focus_probability=1.0,
    )

    _, labels = dataset[0]

    assert tuple(labels.shape) == (8, 8)
    assert int((labels == 3).sum()) == 1


def test_floodnet_fixed_validation_crop_centers_on_rare_class_when_present(tmp_path):
    import numpy as np

    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[14:16, 14:16] = 3
    Image.fromarray(mask).save(tmp_path / "mask.png")
    sample = {"image": "image.png", "mask": "mask.png", "group_id": "id1", "split": "validation"}
    dataset = FloodNetSegmentationDataset(
        tmp_path, [sample], tile_size=8, overlap=0, one_tile_per_sample=True,
        random_crop_per_sample=False, focus_class_id=3, focus_class_center=True,
    )

    _, labels = dataset[0]

    assert int((labels == 3).sum()) == 4


def test_floodnet_positive_scene_sampling_weights_only_training_masks(tmp_path):
    import numpy as np

    flooded = np.zeros((8, 8), dtype=np.uint8)
    flooded[2:4, 2:4] = 3
    ordinary = np.full((8, 8), 4, dtype=np.uint8)
    Image.fromarray(flooded).save(tmp_path / "flooded.png")
    Image.fromarray(ordinary).save(tmp_path / "ordinary.png")
    samples = [
        {"image": "f1.jpg", "mask": "flooded.png", "group_id": "a", "split": "train"},
        {"image": "f2.jpg", "mask": "flooded.png", "group_id": "a", "split": "train"},
        {"image": "n.jpg", "mask": "ordinary.png", "group_id": "a", "split": "train"},
    ]

    weights, summary = floodnet_focus_sample_weights(
        tmp_path, samples, positive_class_id=3, positive_weight=8,
    )

    assert weights == [8.0, 8.0, 1.0]
    assert summary == {"positive_images": 2, "negative_images": 1,
                       "positive_sampling_weight": 8.0,
                       "expected_positive_draw_fraction": pytest.approx(16 / 17)}


def test_floodnet_training_saves_best_state_and_returns_pixel_metrics(tmp_path):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(4)
    images = torch.rand(2, 3, 8, 8)
    labels = torch.zeros((2, 8, 8), dtype=torch.long)
    labels[:, 2:6, 2:6] = 3
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Conv2d(3, 10, kernel_size=1)
    checkpoint = tmp_path / "best.pt"

    report = train_floodnet(model, loader, loader, device=torch.device("cpu"),
                            epochs=1, checkpoint_path=checkpoint)
    metrics = evaluate_floodnet(model, loader, device=torch.device("cpu"))

    assert checkpoint.is_file()
    assert set(torch.load(checkpoint, map_location="cpu", weights_only=True)) == set(model.state_dict())
    assert report["epochs_completed"] == 1
    assert report["best_validation_flooded_road_iou"] is not None
    assert metrics["flooded_road"]["tp"] + metrics["flooded_road"]["fn"] == 32
