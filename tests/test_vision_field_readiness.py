import json
import hashlib

from scripts.vision.audit_campus_field_readiness import audit_campus_field_readiness


def test_readiness_audit_names_missing_campus_evidence_without_claiming_ready(tmp_path):
    result = audit_campus_field_readiness(dataset_root=tmp_path)

    assert result["campus_field_data_present"] is False
    assert result["checks"]["flight_inventory"]["status"] == "missing"
    assert result["checks"]["object_detection"]["status"] == "missing"
    assert result["checks"]["event_evaluation"]["status"] == "missing"
    assert any("真实校园航次" in item for item in result["next_actions"])
    assert result["accuracy_claim_allowed"] is False


def test_readiness_audit_rejects_a_report_from_another_manifest(tmp_path):
    manifest_path = tmp_path / "detection.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "task": "campus_object_detection",
        "dataset": "WHU test",
        "split_unit": "flight_id",
        "label_source": "manual",
        "split_groups": {
            "train": ["train-1"],
            "validation": ["validation-1"],
            "test": ["test-1"],
        },
        "samples": [{
            "sample_id": "sample-1", "group_id": "test-1", "image": "frame.jpg",
            "width": 100, "height": 100,
            "objects": [{"label": "pedestrian", "box": [1, 1, 10, 10]}],
        }],
    }), encoding="utf-8")
    detector_path = tmp_path / "detector.onnx"
    detector_path.write_bytes(b"model")
    report_path = tmp_path / "detection-report.json"
    report_path.write_text(json.dumps({
        "task": "campus_object_detection",
        "manifest_sha256": "0" * 64,
        "model_sha256": "not-the-detector-hash",
    }), encoding="utf-8")

    result = audit_campus_field_readiness(
        dataset_root=tmp_path,
        detection_manifest=manifest_path,
        detector_model=detector_path,
        detection_report=report_path,
    )

    assert result["checks"]["object_detection"]["status"] == "invalid"
    assert "manifest_sha256" in result["checks"]["object_detection"]["issues"]
    assert "model_sha256" in result["checks"]["object_detection"]["issues"]
    assert result["accuracy_claim_allowed"] is False


def test_event_report_below_precision_and_recall_targets_cannot_support_claims(tmp_path):
    detector_path = tmp_path / "detector.onnx"
    detector_path.write_bytes(b"model")
    unrelated_accident_model = tmp_path / "accident.onnx"
    unrelated_accident_model.write_bytes(b"accident-model")
    event_manifest = {
        "schema_version": 1,
        "task": "campus_event_detection",
        "dataset": "WHU UAV event test",
        "split_unit": "flight_id",
        "label_source": "manual",
        "event_kinds": ["congestion"],
        "split_groups": {
            "train": ["train-1"],
            "validation": ["validation-1"],
            "test": ["test-1", "test-2", "test-3"],
        },
        "samples": [],
    }
    for index in range(1, 4):
        event_manifest["samples"].append({
            "sample_id": f"sample-{index}",
            "video": f"test/flight-{index}.mp4",
            "group_id": f"test-{index}",
            "camera_stabilized": True,
            "event_kinds": ["congestion"],
            "observation_regions": [{
                "id": "lane-1", "kind": "vehicle_lane",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            }],
            "events": [{
                "event_id": f"queue-{index}", "kind": "congestion",
                "region_id": "lane-1", "start_seconds": 1, "end_seconds": 2,
            }],
        })
    manifest_path = tmp_path / "events.json"
    manifest_bytes = json.dumps(event_manifest).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    event_report_path = tmp_path / "event-report.json"
    event_report_path.write_text(json.dumps({
        "task": "campus_event_detection",
        "split": "test",
        "test_group_count": 3,
        "sample_count": 3,
        "event_kinds": ["congestion"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_provenance": {"detector_sha256": hashlib.sha256(b"model").hexdigest()},
        "event_metrics": {
            "by_kind": {"congestion": {
                "counts": {"tp": 0, "fp": 1, "fn": 3},
                "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }},
        },
    }), encoding="utf-8")

    result = audit_campus_field_readiness(
        event_manifest=manifest_path,
        event_report=event_report_path,
        detector_model=detector_path,
        accident_model=unrelated_accident_model,
    )

    assert result["checks"]["event_evaluation"]["status"] == "ready"
    assert result["checks"]["event_evaluation"]["event_metric_targets"]["congestion"]["met"] is False
    assert result["checks"]["event_evaluation"]["accuracy_claim_ready"] is False

    broken_report = json.loads(event_report_path.read_text(encoding="utf-8"))
    broken_report["event_metrics"]["by_kind"]["congestion"]["counts"]["tp"] = "bad"
    event_report_path.write_text(json.dumps(broken_report), encoding="utf-8")
    malformed_result = audit_campus_field_readiness(
        event_manifest=manifest_path,
        event_report=event_report_path,
        detector_model=detector_path,
        accident_model=unrelated_accident_model,
    )
    assert malformed_result["checks"]["event_evaluation"]["status"] == "invalid"


def test_water_report_below_road_iou_target_is_not_ready(tmp_path):
    flood_model = tmp_path / "visible-water.onnx"
    flood_model.write_bytes(b"flood-model")
    manifest = {
        "schema_version": 1,
        "task": "visible_water_road_segmentation",
        "dataset": "WHU campus road water",
        "split_unit": "flight_id",
        "label_source": "manual",
        "mask_encoding": "binary_0_255",
        "split_groups": {
            "train": ["train-1"],
            "validation": ["validation-1"],
            "test": ["test-1", "test-2", "test-3"],
        },
        "samples": [],
    }
    for index in range(1, 4):
        manifest["samples"].append({
            "image": f"test/{index}.jpg",
            "mask": f"test/{index}.png",
            "group_id": f"test-{index}",
            "road_surface_polygon": [[0, 0], [1, 0], [1, 1]],
        })
    manifest_path = tmp_path / "water-manifest.json"
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    report_path = tmp_path / "water-report.json"
    report_path.write_text(json.dumps({
        "task": "visible_water_road_segmentation",
        "dataset": "WHU campus road water",
        "split": "test",
        "test_group_count": 3,
        "sample_count": 3,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(b"flood-model").hexdigest(),
        "metrics": {
            "iou": 0.0716,
            "per_group": {
                "test-1": {"negative_flight": False},
                "test-2": {"negative_flight": True},
                "test-3": {"negative_flight": True},
            },
        },
    }), encoding="utf-8")

    result = audit_campus_field_readiness(
        visible_water_manifest=manifest_path,
        visible_water_report=report_path,
        flood_model=flood_model,
    )

    assert result["checks"]["visible_water_segmentation"]["status"] == "invalid"
    assert "road-region IoU target 0.60" in result["checks"]["visible_water_segmentation"]["issues"]
    assert result["accuracy_claim_allowed"] is False
