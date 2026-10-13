import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from vision.model_release import check_optional_model_release


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _release_fixture(tmp_path, model_kind):
    model_path = tmp_path / f"{model_kind}.onnx"
    model_path.write_bytes(f"fixture-{model_kind}".encode())
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    runtime_config = ({
        "sample_fps": 5.0,
        "specialized_sampling_policy": "detector_sample_grid_about_1hz_v2",
        "decision_threshold": 0.93,
        "dense_trigger_threshold": 0.5,
        "dense_window_seconds": 2.0,
        "roi_policy": "all_vehicle_lanes_or_first_road_region_v1",
    } if model_kind == "accident" else {
        "sample_fps": 5.0,
        "specialized_sampling_policy": "detector_sample_grid_about_1hz_v2",
        "min_area_ratio": 0.08,
        "probability_threshold": 0.5,
        "roi_policy": "all_marked_road_surface_regions_v1",
    })
    event_model = {
        "status": "ready",
        "accuracy_claim_ready": True,
        "event_kinds": ["accident"],
        "campus_negative_safety_ready": True,
        "thresholds": {"accident": 0.93},
        "event_metric_targets": {
            "accident": {
                "precision": 0.93,
                "recall": 0.84,
                "test_flight_count": 3,
                "positive_test_flight_count": 2,
                "false_positive_count": 0,
                "met": True,
            },
        },
    }
    water = {"status": "missing", "iou": None, "test_flight_count": None}
    if model_kind == "flood":
        event_model["event_kinds"] = ["flooding"]
        event_model["thresholds"] = {
            "flood_min_area_ratio": 0.08, "flood_probability": 0.5,
        }
        event_model["event_metric_targets"] = {
            "flooding": {
                "precision": 0.93,
                "recall": 0.84,
                "test_flight_count": 3,
                "positive_test_flight_count": 2,
                "met": True,
            },
        }
        water = {"status": "ready", "iou": 0.66, "test_flight_count": 3}
    readiness = {
        "task": "campus_uav_field_readiness",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campus_field_data_present": True,
        "accuracy_claim_allowed": True,
        "checks": {
            "models": {
                "detector": {
                    "status": "present",
                    "sha256": hashlib.sha256(b"detector").hexdigest(),
                },
                "accident": {"status": "present", "sha256": model_hash},
                "flood": {"status": "present", "sha256": model_hash},
            },
            "event_evaluation": event_model,
            "visible_water_segmentation": water,
            "external_accident_evaluation": {
                "status": "ready",
                "scope": "public_benchmark",
                "source_provenance": {
                    "source": "public accident event test fixture",
                    "url": "https://example.org/accident-set",
                    "license": "CC BY 4.0",
                    "license_evidence": "fixture-dataset-card",
                },
                "manifest_sha256": "a" * 64,
                "report_sha256": "b" * 64,
                "model_sha256": model_hash,
                "detector_sha256": hashlib.sha256(b"detector").hexdigest(),
                "threshold": 0.93,
                "test_group_count": 3,
                "positive_test_group_count": 1,
                "precision": 0.93,
                "recall": 0.84,
            },
        },
    }
    readiness_path = _write_json(tmp_path / "readiness.json", readiness)
    readiness_hash = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
    release = {
        "schema_version": 1,
        "models": {
            model_kind: {
                "approved": True,
                "rights_status": "approved",
                "rights_evidence": "license-review-2026-10-13",
                "approved_by": "reviewer",
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "approval_expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
                "model_sha256": model_hash,
                "model_revision": f"{model_kind}-v1",
                "readiness_report_sha256": readiness_hash,
                "runtime_config": runtime_config,
                **({
                    "benchmark_approval": {
                        "approved": True,
                        "approved_by": "dataset-reviewer",
                        "evidence": "dataset-rights-review-2026-10-14",
                        "manifest_sha256": "a" * 64,
                        "approved_at": datetime.now(timezone.utc).isoformat(),
                        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
                    },
                } if model_kind == "accident" else {}),
            },
        },
    }
    release_path = _write_json(tmp_path / "release.json", release)
    return model_path, readiness_path, release_path, release, readiness, runtime_config


@pytest.mark.parametrize("model_kind", ["accident", "flood"])
def test_optional_model_requires_matching_rights_and_field_evidence(tmp_path, model_kind):
    model_path, readiness_path, release_path, _release, _readiness, runtime_config = _release_fixture(
        tmp_path, model_kind,
    )

    result = check_optional_model_release(
        model_kind, model_path, f"{model_kind}-v1", release_path, readiness_path,
        runtime_config,
    )

    assert result == {"approved": True, "reason": "approved"}


def test_optional_model_release_fails_closed_when_files_are_missing(tmp_path):
    result = check_optional_model_release(
        "accident", tmp_path / "missing.onnx", "accident-v1",
        tmp_path / "missing-release.json", tmp_path / "missing-readiness.json",
    )

    assert result["approved"] is False
    assert result["reason"] in {"model_missing", "release_manifest_missing"}


def test_optional_model_release_rejects_runtime_config_drift(tmp_path):
    model_path, readiness_path, release_path, _release, _readiness, runtime_config = (
        _release_fixture(tmp_path, "accident")
    )
    changed = {**runtime_config, "dense_window_seconds": 3.0}

    result = check_optional_model_release(
        "accident", model_path, "accident-v1", release_path, readiness_path, changed,
    )

    assert result == {"approved": False, "reason": "runtime_config_mismatch"}


def test_accident_release_uses_public_positive_benchmark_and_campus_negative_flights(tmp_path):
    model_path, readiness_path, release_path, release, readiness, runtime_config = (
        _release_fixture(tmp_path, "accident")
    )
    readiness["campus_field_data_present"] = True
    readiness["accuracy_claim_allowed"] = False
    readiness["checks"]["event_evaluation"] = {
        "status": "ready",
        "campus_negative_safety_ready": True,
        "event_kinds": ["accident"],
        "thresholds": {"accident": 0.93},
        "event_metric_targets": {
            "accident": {
                "false_positive_count": 0,
                "test_flight_count": 3,
                "positive_test_flight_count": 0,
            },
        },
    }
    readiness["checks"]["external_accident_evaluation"] = {
        "status": "ready",
        "scope": "public_benchmark",
        "source_provenance": {
            "source": "public video benchmark",
            "url": "https://example.org/benchmark",
            "license": "CC BY 4.0",
            "license_evidence": "dataset-card",
        },
        "manifest_sha256": "a" * 64,
        "report_sha256": "b" * 64,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "detector_sha256": readiness["checks"]["models"]["detector"]["sha256"],
        "threshold": 0.93,
        "positive_test_group_count": 3,
        "test_group_count": 4,
        "precision": 0.93,
        "recall": 0.84,
    }
    _write_json(readiness_path, readiness)
    release["models"]["accident"]["readiness_report_sha256"] = hashlib.sha256(
        readiness_path.read_bytes()
    ).hexdigest()
    _write_json(release_path, release)

    result = check_optional_model_release(
        "accident", model_path, "accident-v1", release_path, readiness_path,
        runtime_config,
    )

    assert result == {"approved": True, "reason": "approved"}


@pytest.mark.parametrize("tamper", ["missing", "expired", "too_long"])
def test_accident_release_requires_current_benchmark_source_approval(tmp_path, tamper):
    model_path, readiness_path, release_path, release, _readiness, runtime_config = (
        _release_fixture(tmp_path, "accident")
    )
    approval = release["models"]["accident"]["benchmark_approval"]
    now = datetime.now(timezone.utc)
    if tamper == "missing":
        release["models"]["accident"].pop("benchmark_approval")
    elif tamper == "expired":
        approval["approved_at"] = (now - timedelta(days=2)).isoformat()
        approval["expires_at"] = (now - timedelta(seconds=1)).isoformat()
    else:
        approval["approved_at"] = now.isoformat()
        approval["expires_at"] = (now + timedelta(days=366)).isoformat()
    _write_json(release_path, release)

    result = check_optional_model_release(
        "accident", model_path, "accident-v1", release_path, readiness_path,
        runtime_config,
    )

    assert result == {"approved": False, "reason": "accident_field_evidence_not_met"}


def test_optional_model_release_rejects_evaluation_threshold_drift(tmp_path):
    model_path, readiness_path, release_path, release, readiness, runtime_config = (
        _release_fixture(tmp_path, "accident")
    )
    readiness["checks"]["event_evaluation"]["thresholds"]["accident"] = 0.92
    _write_json(readiness_path, readiness)
    release["models"]["accident"]["readiness_report_sha256"] = hashlib.sha256(
        readiness_path.read_bytes()
    ).hexdigest()
    _write_json(release_path, release)

    result = check_optional_model_release(
        "accident", model_path, "accident-v1", release_path, readiness_path,
        runtime_config,
    )

    assert result == {
        "approved": False, "reason": "accident_runtime_threshold_mismatch",
    }


def test_flood_release_accepts_zero_probability_threshold(tmp_path):
    model_path, readiness_path, release_path, release, readiness, runtime_config = (
        _release_fixture(tmp_path, "flood")
    )
    runtime_config = {**runtime_config, "probability_threshold": 0.0}
    release["models"]["flood"]["runtime_config"] = runtime_config
    readiness["checks"]["event_evaluation"]["thresholds"]["flood_probability"] = 0.0
    _write_json(readiness_path, readiness)
    release["models"]["flood"]["readiness_report_sha256"] = hashlib.sha256(
        readiness_path.read_bytes()
    ).hexdigest()
    _write_json(release_path, release)

    result = check_optional_model_release(
        "flood", model_path, "flood-v1", release_path, readiness_path,
        runtime_config,
    )

    assert result == {"approved": True, "reason": "approved"}


@pytest.mark.parametrize(
    "tamper", [
        "model_hash", "readiness_hash", "rights", "metrics", "campus_false_alarm",
        "campus_source_link", "benchmark_source_approval",
    ]
)
def test_optional_model_release_rejects_unverified_or_ineligible_evidence(tmp_path, tamper):
    model_path, readiness_path, release_path, release, readiness, runtime_config = _release_fixture(
        tmp_path, "accident",
    )
    entry = release["models"]["accident"]
    if tamper == "model_hash":
        entry["model_sha256"] = "0" * 64
    elif tamper == "readiness_hash":
        entry["readiness_report_sha256"] = "0" * 64
    elif tamper == "rights":
        entry["rights_status"] = "unknown"
    elif tamper == "campus_false_alarm":
        readiness["checks"]["event_evaluation"]["event_metric_targets"]["accident"][
            "false_positive_count"
        ] = 1
        _write_json(readiness_path, readiness)
        entry["readiness_report_sha256"] = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
    elif tamper == "campus_source_link":
        readiness["checks"]["event_evaluation"]["campus_negative_safety_ready"] = False
        _write_json(readiness_path, readiness)
        entry["readiness_report_sha256"] = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
    elif tamper == "benchmark_source_approval":
        entry["benchmark_approval"]["manifest_sha256"] = "0" * 64
    else:
        readiness["checks"]["external_accident_evaluation"]["recall"] = 0.2
        _write_json(readiness_path, readiness)
        entry["readiness_report_sha256"] = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
    _write_json(release_path, release)

    result = check_optional_model_release(
        "accident", model_path, "accident-v1", release_path, readiness_path,
        runtime_config,
    )

    assert result["approved"] is False


@pytest.mark.parametrize(
    ("tamper", "expected_reason"),
    [
        ("missing_expiry", "approval_period_invalid"),
        ("expired_approval", "approval_expired"),
        ("future_approval", "approval_time_invalid"),
        ("stale_report", "readiness_report_stale"),
        ("future_report", "readiness_report_time_invalid"),
    ],
)
def test_optional_model_release_requires_current_approval_and_report(
    tmp_path, tamper, expected_reason,
):
    model_path, readiness_path, release_path, release, readiness, runtime_config = _release_fixture(
        tmp_path, "accident",
    )
    entry = release["models"]["accident"]
    now = datetime.now(timezone.utc)
    if tamper == "missing_expiry":
        entry.pop("approval_expires_at")
    elif tamper == "expired_approval":
        entry["approved_at"] = (now - timedelta(days=10)).isoformat()
        entry["approval_expires_at"] = (now - timedelta(seconds=1)).isoformat()
    elif tamper == "future_approval":
        entry["approved_at"] = (now + timedelta(seconds=60)).isoformat()
        entry["approval_expires_at"] = (now + timedelta(days=30)).isoformat()
    elif tamper == "stale_report":
        readiness["generated_at"] = (now - timedelta(days=91)).isoformat()
        _write_json(readiness_path, readiness)
        entry["readiness_report_sha256"] = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
    else:
        readiness["generated_at"] = (now + timedelta(seconds=60)).isoformat()
        _write_json(readiness_path, readiness)
        entry["readiness_report_sha256"] = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
    _write_json(release_path, release)

    result = check_optional_model_release(
        "accident", model_path, "accident-v1", release_path, readiness_path,
        runtime_config,
    )

    assert result == {"approved": False, "reason": expected_reason}
