from importlib import import_module, util
import json
from pathlib import Path

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


def test_loader_rejects_incomplete_multiscene_frame_map(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root)
    source_path = root / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["sequences"] = ["scene_23/1", "scene_31/1"]
    source["frame_map"] = [{"sample_file": "1.jpg", "source_sequence": "scene_23/1"}]
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="frame_map.*every sample image"):
        _runner().load_sequence(root)


def test_loader_requires_frame_map_when_source_lists_multiple_sequences(tmp_path):
    root = tmp_path / "sample"
    _write_sequence(root)
    source_path = root / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["sequences"] = ["scene_23/1", "scene_31/1"]
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="multi-sequence.*frame_map"):
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


def test_sequence_summary_reports_scene_metrics_and_unweighted_macro_means():
    summarize = getattr(_runner(), "summarize_sequence_metrics", None)
    assert callable(summarize), "multi-scene evaluation must report per-sequence variation"

    result = summarize([
        {
            "sequence": "scene_a/1",
            "counts": {"tp": 1, "fp": 1, "fn": 0, "predicted": 2, "ground_truth": 1},
            "count_error": 1,
        },
        {
            "sequence": "scene_a/1",
            "counts": {"tp": 1, "fp": 0, "fn": 1, "predicted": 1, "ground_truth": 2},
            "count_error": 1,
        },
        {
            "sequence": "scene_b/1",
            "counts": {"tp": 2, "fp": 0, "fn": 2, "predicted": 2, "ground_truth": 4},
            "count_error": 2,
        },
    ])

    assert result["sequence_count"] == 2
    assert result["macro_head_point_f1"] == 0.6667
    assert result["macro_per_frame_count_mae"] == 1.5
    assert [item["sequence"] for item in result["per_sequence"]] == ["scene_a/1", "scene_b/1"]
    assert result["per_sequence"][0]["images_evaluated"] == 2
    assert result["per_sequence"][0]["per_frame_count_mae"] == 1.0


def test_sequence_summary_rejects_invalid_or_empty_frame_metrics():
    summarize = getattr(_runner(), "summarize_sequence_metrics", None)
    assert callable(summarize), "multi-scene evaluation must validate its frame metrics"

    with pytest.raises(ValueError, match="at least one"):
        summarize([])
    with pytest.raises(ValueError, match="sequence"):
        summarize([{"sequence": "", "counts": {}, "count_error": 0}])


def test_sampled_frame_is_mapped_back_to_official_source_sequence():
    resolve_sequence = getattr(_runner(), "sequence_for_image", None)
    assert callable(resolve_sequence), "evaluation must preserve official sequence provenance"

    assert resolve_sequence(Path("2.jpg"), {
        "frame_map": [
            {"sample_file": "1.jpg", "source_sequence": "scene_1/1"},
            {"sample_file": "2.jpg", "source_sequence": "scene_3/1"},
        ],
    }) == "scene_3/1"
    with pytest.raises(ValueError, match="exactly once"):
        resolve_sequence(Path("3.jpg"), {
            "frame_map": [{"sample_file": "1.jpg", "source_sequence": "scene_1/1"}],
        })
