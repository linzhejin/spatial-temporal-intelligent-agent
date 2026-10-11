import json
import zipfile

import pytest

from vision.songdo_vision_evaluation import run_songdo_vision_evaluation


class _Frame:
    def __init__(self, name, width, height):
        self.name = name
        self.shape = (height, width, 3)


class _Detector:
    model_id = "fixture-detector"
    model_version = "fixture-v1"

    def __init__(self, *_args, **_kwargs):
        pass

    def detect(self, frame):
        if frame.name == "image-one":
            return [
                {"label": "car", "box": [10, 10, 30, 30], "confidence": 0.9},
                {"label": "van", "box": [10, 10, 30, 30], "confidence": 0.8},
                {"label": "bus", "box": [40, 40, 45, 45], "confidence": 0.7},
                {"label": "pedestrian", "box": [70, 70, 80, 80], "confidence": 0.99},
            ]
        return [
            {"label": "motor", "box": [20, 20, 30, 30], "confidence": 0.95},
        ]


def _archive(path, *, omit_image=None):
    coco = {
        "info": {"version": "2.0"},
        "licenses": [{"id": 1, "name": "CC BY 4.0"}],
        "images": [
            {"id": 1, "file_name": "00001.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "00002.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "bus"},
            {"id": 3, "name": "truck"},
            {"id": 4, "name": "motorcycle"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [10, 10, 20, 20], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 3,
             "bbox": [50, 50, 10, 10], "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 4,
             "bbox": [20, 20, 10, 10], "iscrowd": 0},
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("test/coco_annotations.json", json.dumps(coco))
        if omit_image != "00001.jpg":
            archive.writestr("test/images/00001.jpg", b"image-one")
        if omit_image != "00002.jpg":
            archive.writestr("test/images/00002.jpg", b"image-two")


def _run(archive_path, model_path, **kwargs):
    return run_songdo_vision_evaluation(
        archive_path,
        model_path,
        detector_factory=_Detector,
        image_decoder=lambda payload: _Frame(
            payload.decode(), width=100, height=100,
        ),
        **kwargs,
    )


def test_full_coco_vehicle_scoring_maps_van_to_car_and_excludes_pedestrians(tmp_path):
    archive = tmp_path / "songdo-test.zip"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    _archive(archive)

    report = _run(archive, model)

    assert report["dataset"] == "Songdo Vision v2 test"
    assert report["image_count_evaluated"] == 2
    assert report["official_test_split_complete"] is True
    assert report["by_class"]["car"] == {
        "tp": 1, "fp": 1, "fn": 0,
        "precision": 0.5, "recall": 1.0, "f1": 0.6667,
    }
    assert report["by_class"]["truck"]["fn"] == 1
    assert report["by_class"]["motorcycle"]["tp"] == 1
    assert report["micro"]["tp"] == 2
    assert report["micro"]["fp"] == 2
    assert report["micro"]["fn"] == 1
    assert report["excluded_prediction_count"] == 1
    assert report["dataset_archive_sha256"]
    assert report["annotations_sha256"]


def test_missing_coco_image_fails_instead_of_reporting_partial_benchmark(tmp_path):
    archive = tmp_path / "songdo-test.zip"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    _archive(archive, omit_image="00002.jpg")

    with pytest.raises(ValueError, match="missing 1 image members"):
        _run(archive, model)


def test_partial_limit_is_explicit_and_cannot_claim_full_test_split(tmp_path):
    archive = tmp_path / "songdo-test.zip"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    _archive(archive)

    report = _run(archive, model, limit=1)

    assert report["image_count_in_archive"] == 2
    assert report["image_count_evaluated"] == 1
    assert report["official_test_split_complete"] is False
    assert report["evaluation_scope"] == "limited_smoke_sample"


@pytest.mark.parametrize("mutate, message", [
    (lambda coco: coco["images"][0].update(file_name="../outside.jpg"), "unsafe image path"),
    (lambda coco: coco["annotations"][0].update(bbox=[10, 10, -2, 20]), "invalid bbox"),
    (lambda coco: coco["images"].append(dict(coco["images"][0])), "duplicate image id or filename"),
])
def test_invalid_coco_records_are_rejected(tmp_path, mutate, message):
    archive = tmp_path / "songdo-test.zip"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    _archive(archive)

    with zipfile.ZipFile(archive) as source:
        coco = json.loads(source.read("test/coco_annotations.json"))
    mutate(coco)
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("test/coco_annotations.json", json.dumps(coco))
        output.writestr("test/images/00001.jpg", b"image-one")
        output.writestr("test/images/00002.jpg", b"image-two")

    with pytest.raises(ValueError, match=message):
        _run(archive, model)
