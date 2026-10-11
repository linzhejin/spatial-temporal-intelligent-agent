import importlib
import importlib.util
import json
import shutil

import pytest
from PIL import Image, ImageDraw


def _api():
    spec = importlib.util.find_spec("vision.perceptual_overlap")
    assert spec is not None, "perceptual overlap audit module should exist"
    return importlib.import_module("vision.perceptual_overlap")


def _scene(path):
    image = Image.new("RGB", (128, 96), (48, 93, 122))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 58, 127, 95), fill=(34, 72, 45))
    draw.polygon([(12, 70), (53, 42), (96, 73)], fill=(66, 118, 65))
    draw.ellipse((75, 10, 117, 51), fill=(80, 140, 177))
    draw.line((0, 54, 127, 53), fill=(225, 229, 217), width=3)
    image.save(path)
    return image


def test_perceptual_hash_is_stable_after_jpeg_reencoding(tmp_path):
    audit = _api()
    source = tmp_path / "source.png"
    jpeg = tmp_path / "reencoded.jpg"
    _scene(source).save(jpeg, quality=95)

    source_hash = audit.perceptual_hash(source)
    reencoded_hash = audit.perceptual_hash(jpeg)

    assert isinstance(source_hash, int)
    assert audit.hamming_distance(source_hash, reencoded_hash) <= 2


def test_audit_uses_train_validation_only_and_reports_near_duplicates(tmp_path):
    audit = _api()
    reference_root = tmp_path / "references"
    candidate_dir = tmp_path / "manual"
    reference_root.mkdir()
    candidate_dir.mkdir()
    scene = _scene(reference_root / "train.png")
    candidate = candidate_dir / "sample_000001.jpg"
    scene.save(candidate, quality=95)
    shutil.copyfile(candidate, reference_root / "test.jpg")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "samples": [
            {"sample_id": "train-1", "source_video_id": "video-train",
             "split": "train", "image": "train.png"},
            {"sample_id": "test-1", "source_video_id": "video-test",
             "split": "test", "image": "test.jpg"},
        ],
    }), encoding="utf-8")

    report = audit.audit_overlap(
        manifest_path, reference_root, candidate_dir, near_threshold=2,
    )

    assert report["summary"]["reference_count"] == 1
    assert report["summary"]["candidate_count"] == 1
    assert report["summary"]["exact_sha256_matches"] == 0
    assert report["summary"]["near_threshold_count"] == 1
    assert len(report["reference_manifest_sha256"]) == 64
    assert len(report["reference_image_set_sha256"]) == 64
    assert len(report["candidate_image_set_sha256"]) == 64
    match = report["matches"][0]
    assert match["nearest_source_video_id"] == "video-train"
    assert match["nearest_split"] == "train"
    assert match["nearest_distance"] <= 2
    assert match["exact_sha256_match"] is False
    assert match["nearest_reference_sha256"] == audit.sha256_file(reference_root / "train.png")


def test_audit_detects_exact_byte_matches_in_reference_splits(tmp_path):
    audit = _api()
    reference_root = tmp_path / "references"
    candidate_dir = tmp_path / "manual"
    reference_root.mkdir()
    candidate_dir.mkdir()
    _scene(reference_root / "train.png")
    shutil.copyfile(reference_root / "train.png", candidate_dir / "same.png")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "samples": [{
            "sample_id": "train-1", "source_video_id": "video-train",
            "split": "train", "image": "train.png",
        }],
    }), encoding="utf-8")

    report = audit.audit_overlap(manifest_path, reference_root, candidate_dir)

    assert report["summary"]["exact_sha256_matches"] == 1
    assert report["matches"][0]["exact_sha256_match"] is True


def test_audit_rejects_source_video_declared_in_multiple_splits(tmp_path):
    audit = _api()
    reference_root = tmp_path / "references"
    candidate_dir = tmp_path / "manual"
    reference_root.mkdir()
    candidate_dir.mkdir()
    _scene(reference_root / "train.png")
    _scene(candidate_dir / "sample.png")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "split_groups": {
            "train": ["video-shared"],
            "validation": [],
            "test": ["video-shared"],
        },
        "samples": [{
            "sample_id": "train-1", "source_video_id": "video-shared",
            "split": "train", "image": "train.png",
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="assigned to multiple splits"):
        audit.audit_overlap(manifest_path, reference_root, candidate_dir)
