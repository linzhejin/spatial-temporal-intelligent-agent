import hashlib
import json
import sys

import cv2
import numpy as np

from scripts.vision import smoke_test


def test_video_smoke_report_records_coverage_and_model_provenance(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test model bytes")
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "model_id": "test-aerial-detector",
        "model_revision": "revision-123",
        "onnx_sha256": model_sha256,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video fixture")
    output_path = tmp_path / "report.json"

    class StubDetector:
        def __init__(self, _path, confidence_threshold=0.369, **kwargs):
            self.model_id = kwargs.get("model_id", "stub")
            self.model_version = kwargs.get("model_version", "stub-revision")
            self.confidence_threshold = confidence_threshold
            self.cpu_threads = 4

        def detect(self, _frame):
            return []

    coverage = {
        "complete": True,
        "duration_seconds": 51.0,
        "analyzed_from_seconds": 0.0,
        "analyzed_through_seconds": 50.0,
        "sampled_frames": 26,
        "coverage_ratio": 1.0,
        "segment_count": 3,
        "sample_interval_seconds": 2.0,
    }
    monkeypatch.setattr(smoke_test, "OnnxDetector", StubDetector)
    monkeypatch.setattr(smoke_test, "analyze_media", lambda *_args, **_kwargs: {
        "metrics": {"frames_analyzed": 26},
        "candidates": [],
        "safety": {"automatically_changes_routing": False},
        "video_coverage": coverage,
    })
    monkeypatch.setattr(sys, "argv", [
        "smoke_test.py", "--model", str(model_path), "--images", str(image_path),
        "--video", str(video_path), "--output", str(output_path),
    ])

    assert smoke_test.main() == 0

    report = json.loads(capsys.readouterr().out)
    assert report["model"] == "test-aerial-detector"
    assert report["model_revision"] == "revision-123"
    assert report["cpu_threads"] == 4
    assert report["model_sha256"] == model_sha256
    assert report["model_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert report["video_sequence_smoke"]["video_sha256"] == hashlib.sha256(
        video_path.read_bytes()
    ).hexdigest()
    assert report["video_sequence_smoke"]["video_bytes"] == video_path.stat().st_size
    assert report["video_sequence_smoke"]["video_coverage"] == coverage
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
