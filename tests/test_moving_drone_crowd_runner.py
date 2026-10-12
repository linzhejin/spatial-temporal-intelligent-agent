from importlib import import_module, util
import json

import pytest


def _runner():
    module_name = "scripts.vision.evaluate_moving_drone_sample"
    spec = util.find_spec(module_name)
    assert spec is not None, "MovingDroneCrowd sample evaluator must be implemented"
    return import_module(module_name)


def _write_sequence(root, *, split="official test.txt", frame_indices=(0, 1)):
    root.mkdir(parents=True)
    for index in (1, 2):
        (root / f"{index}.jpg").write_bytes(b"fixture")
    annotation_rows = [f"{index},7,10,10,6,6,-1,-1,-1,-1" for index in frame_indices]
    (root / "annotations.csv").write_text("\n".join(annotation_rows), encoding="utf-8")
    (root / "source.json").write_text(json.dumps({
        "dataset": "MovingDroneCrowd++",
        "license": "CC BY-NC 4.0",
        "split": split,
        "sequence": "scene_23/1",
        "frames": 2,
        "annotation_rows": len(annotation_rows),
    }), encoding="utf-8")


def test_loader_accepts_official_sample_and_sorts_frame_files_numerically(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root)

    images, annotation_text, source = _runner().load_sequence(root)

    assert [path.name for path in images] == ["1.jpg", "2.jpg"]
    assert len(annotation_text.splitlines()) == 2
    assert source["split"] == "official test.txt"


def test_loader_accepts_official_validation_split_for_threshold_free_comparison(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root, split="official val.txt")

    images, _annotation_text, source = _runner().load_sequence(root)

    assert len(images) == 2
    assert source["split"] == "official val.txt"
    limitations = _runner().build_evaluation_limitations(source, images_evaluated=len(images))
    assert "not the complete official val.txt sequence set" in " ".join(limitations)


def test_loader_rejects_training_split(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root, split="official train.txt")

    with pytest.raises(ValueError, match="official.*test"):
        _runner().load_sequence(root)


def test_loader_rejects_malformed_split_metadata_as_a_validation_error(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root)
    source_path = root / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["split"] = ["official val.txt"]
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="official.*val.*test"):
        _runner().load_sequence(root)


def test_loader_rejects_annotations_with_unknown_frame_index(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root, frame_indices=(0, 2))

    with pytest.raises(ValueError, match="frame"):
        _runner().load_sequence(root)


def test_loader_rejects_annotation_row_count_mismatch(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root)
    source_path = root / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["annotation_rows"] = 99
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="annotation row count"):
        _runner().load_sequence(root)


def test_limitations_describe_the_actual_multiscene_framewise_sample():
    describe = getattr(_runner(), "build_evaluation_limitations", None)
    assert callable(describe), "evaluation limitations must be derived from sample metadata"

    limitations = describe({
        "sequences": [f"scene_{index}/1" for index in range(9)],
        "sampling": "4 uniformly spaced frames per sequence; framewise evaluation only",
    }, images_evaluated=36)

    combined = " ".join(limitations)
    assert "9 sequences" in combined
    assert "36 sampled frames" in combined
    assert "tracking" in combined
    assert "Single 30-frame sequence" not in combined
