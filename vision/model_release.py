"""Fail-closed release checks for optional UAV models."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vision.specialized import (MAX_AUTOMATIC_FLOOD_CANDIDATES, MAX_AUTOMATIC_FLOOD_HOLES,
                                MAX_AUTOMATIC_FLOOD_HOLE_POINTS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _same_number(left: object, right: object) -> bool:
    left_value = _number(left)
    right_value = _number(right)
    return (
        left_value is not None and right_value is not None
        and math.isclose(left_value, right_value, rel_tol=0, abs_tol=1e-9)
    )


MAX_APPROVAL_VALIDITY = timedelta(days=365)
MAX_READINESS_AGE = timedelta(days=90)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _aware_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def assisted_flood_runtime_config(*, probability_threshold: float = 0.001,
                                  min_area_ratio: float = 0.08) -> dict[str, object]:
    return {
        "input_width": 1024,
        "input_height": 768,
        "input_color": "RGB",
        "class_count": 10,
        "preprocessing": "doodleverse_rgb_resize_uint8_global_standardize_v1",
        "flooded_road_class": 3,
        "probability_threshold": float(probability_threshold),
        "min_area_ratio": float(min_area_ratio),
        "single_image_trigger_threshold": float(min_area_ratio),
        "media": ["image"],
        "human_confirmation_required": True,
        "subprocess_timeout_seconds": 120,
    }


def automatic_flood_runtime_config() -> dict[str, object]:
    """Identify the automatic candidate path independently of marked-road analysis."""
    return {
        "candidate_method": "flooded_road_argmax_v1",
        "flooded_road_class": 3,
        "probability_threshold": 0.001,
        "max_candidates": MAX_AUTOMATIC_FLOOD_CANDIDATES,
        "max_holes": MAX_AUTOMATIC_FLOOD_HOLES,
        "max_hole_points": MAX_AUTOMATIC_FLOOD_HOLE_POINTS,
        "human_confirmation_required": True,
    }


def check_automatic_flood_release(model_path: str | Path | None,
                                  model_revision: str | None,
                                  release_manifest_path: str | Path | None) -> dict[str, object]:
    """Require automatic-path engineering evidence in addition to the manual release."""
    base = check_optional_model_release(
        "flood", model_path, model_revision, release_manifest_path, None,
        assisted_flood_runtime_config(), release_mode="assisted_review",
    )
    if base.get("approved") is not True:
        return base
    release = _read_json(Path(release_manifest_path))
    if release is None:
        return {"approved": False, "reason": "release_manifest_invalid"}
    entry = release["models"]["flood"]
    automatic = entry.get("automatic_flood")
    if not isinstance(automatic, dict) or automatic.get("enabled") is not True:
        return {"approved": False, "reason": "automatic_release_missing"}
    runtime = automatic_flood_runtime_config()
    if automatic.get("runtime_config") != runtime:
        return {"approved": False, "reason": "automatic_runtime_contract_mismatch"}
    acceptance = automatic.get("acceptance")
    if not isinstance(acceptance, dict):
        return {"approved": False, "reason": "automatic_acceptance_evidence_missing"}
    for name in ("resource", "workflow"):
        evidence = acceptance.get(name)
        prefix = f"automatic_{name}_evidence"
        if (not isinstance(evidence, dict) or evidence.get("status") != "passed"
                or not isinstance(evidence.get("path"), str) or not evidence["path"].strip()
                or not isinstance(evidence.get("sha256"), str)
                or not _SHA256_RE.fullmatch(evidence["sha256"])):
            return {"approved": False, "reason": prefix + "_invalid"}
        evidence_path = Path(evidence["path"])
        if not evidence_path.is_file():
            return {"approved": False, "reason": prefix + "_missing"}
        if _sha256(evidence_path) != evidence["sha256"]:
            return {"approved": False, "reason": prefix + "_hash_mismatch"}
        record = _read_json(evidence_path)
        if (record is None or record.get("status") != "passed"
                or record.get("scope") != "automatic_flood"
                or record.get("model_sha256") != entry.get("model_sha256")
                or record.get("runtime_config") != runtime):
            return {"approved": False, "reason": prefix + "_invalid"}
    return {"approved": True, "reason": "approved"}


def check_optional_model_release(
    model_kind: str,
    model_path: str | Path | None,
    model_revision: str | None,
    release_manifest_path: str | Path | None,
    readiness_report_path: str | Path | None,
    runtime_config: dict | None = None,
    release_mode: str | None = None,
) -> dict[str, object]:
    """Validate rights, exact weight identity, and campus field evidence before loading.

    The readiness report is produced by ``audit_campus_field_readiness``. The
    separate release manifest records the operator's license review and binds
    that report to the exact deployed model file.
    """
    def rejected(reason: str) -> dict[str, object]:
        return {"approved": False, "reason": reason}

    if model_kind not in {"accident", "flood"}:
        return rejected("unsupported_model")
    if not model_path:
        return rejected("model_missing")
    model_file = Path(model_path)
    if not model_file.is_file():
        return rejected("model_missing")
    if not release_manifest_path:
        return rejected("release_manifest_missing")
    release_path = Path(release_manifest_path)
    if not release_path.is_file():
        return rejected("release_manifest_missing")
    release = _read_json(release_path)
    if release is None or release.get("schema_version") != 1:
        return rejected("release_manifest_invalid")

    entries = release.get("models")
    entry = entries.get(model_kind) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        return rejected("model_release_missing")

    if release_mode == "assisted_review":
        if model_kind != "flood":
            return rejected("assisted_review_is_flood_only")
        now = datetime.now(timezone.utc)
        approved_at = _aware_time(entry.get("approved_at"))
        expires_at = _aware_time(entry.get("approval_expires_at"))
        if approved_at is None or approved_at > now:
            return rejected("approval_time_invalid")
        if (expires_at is None or expires_at <= approved_at
                or expires_at - approved_at > MAX_APPROVAL_VALIDITY):
            return rejected("approval_period_invalid")
        if expires_at <= now:
            return rejected("approval_expired")
        if (entry.get("mode") != "assisted_review"
                or entry.get("approved") is not True
                or entry.get("rights_status") != "approved"
                or entry.get("license") != "CC BY 4.0"
                or not isinstance(entry.get("rights_evidence"), str)
                or not entry["rights_evidence"].lower().startswith(("https://", "http://"))
                or not isinstance(entry.get("approved_by"), str)
                or not entry["approved_by"].strip()):
            return rejected("rights_not_approved")
        if entry.get("field_validated") is not False:
            return rejected("field_validation_must_remain_false")
        actual_model_hash = _sha256(model_file)
        if entry.get("model_sha256") != actual_model_hash:
            return rejected("model_hash_mismatch")
        if not model_revision or entry.get("model_revision") != model_revision:
            return rejected("model_revision_mismatch")
        if not isinstance(runtime_config, dict) or not runtime_config:
            return rejected("runtime_config_missing")
        if entry.get("runtime_config") != runtime_config:
            return rejected("runtime_config_mismatch")
        required_runtime = assisted_flood_runtime_config()
        if runtime_config != required_runtime:
            return rejected("assisted_runtime_contract_mismatch")
        acceptance = entry.get("acceptance")
        if not isinstance(acceptance, dict):
            return rejected("acceptance_evidence_missing")
        for name in ("parity", "resource", "workflow"):
            evidence = acceptance.get(name)
            if (not isinstance(evidence, dict) or evidence.get("status") != "passed"
                    or not isinstance(evidence.get("path"), str)
                    or not evidence["path"].strip()
                    or not isinstance(evidence.get("sha256"), str)
                    or not _SHA256_RE.fullmatch(evidence["sha256"])):
                return rejected(f"{name}_evidence_invalid")
            evidence_path = Path(evidence["path"])
            if not evidence_path.is_file():
                return rejected(f"{name}_evidence_missing")
            if _sha256(evidence_path) != evidence["sha256"]:
                return rejected(f"{name}_evidence_hash_mismatch")
        return {"approved": True, "reason": "approved"}

    if release_mode not in (None, "field_validated"):
        return rejected("unsupported_release_mode")
    if not readiness_report_path:
        return rejected("readiness_report_missing")
    readiness_path = Path(readiness_report_path)
    if not readiness_path.is_file():
        return rejected("readiness_report_missing")
    readiness = _read_json(readiness_path)
    if readiness is None:
        return rejected("readiness_report_invalid")
    now = datetime.now(timezone.utc)
    approved_at = _aware_time(entry.get("approved_at"))
    expires_at = _aware_time(entry.get("approval_expires_at"))
    if approved_at is None or approved_at > now:
        return rejected("approval_time_invalid")
    if (expires_at is None or expires_at <= approved_at
            or expires_at - approved_at > MAX_APPROVAL_VALIDITY):
        return rejected("approval_period_invalid")
    if expires_at <= now:
        return rejected("approval_expired")
    if (entry.get("approved") is not True or entry.get("rights_status") != "approved"
            or not isinstance(entry.get("rights_evidence"), str)
            or not entry["rights_evidence"].strip()
            or not isinstance(entry.get("approved_by"), str)
            or not entry["approved_by"].strip()):
        return rejected("rights_not_approved")

    actual_model_hash = _sha256(model_file)
    if entry.get("model_sha256") != actual_model_hash:
        return rejected("model_hash_mismatch")
    if not model_revision or entry.get("model_revision") != model_revision:
        return rejected("model_revision_mismatch")
    if entry.get("readiness_report_sha256") != _sha256(readiness_path):
        return rejected("readiness_hash_mismatch")
    if not isinstance(runtime_config, dict) or not runtime_config:
        return rejected("runtime_config_missing")
    if entry.get("runtime_config") != runtime_config:
        return rejected("runtime_config_mismatch")
    readiness_generated_at = _aware_time(readiness.get("generated_at"))
    if readiness_generated_at is None or readiness_generated_at > now:
        return rejected("readiness_report_time_invalid")
    if now - readiness_generated_at > MAX_READINESS_AGE:
        return rejected("readiness_report_stale")
    if readiness.get("task") != "campus_uav_field_readiness":
        return rejected("campus_readiness_not_met")
    if model_kind == "accident":
        if readiness.get("campus_field_data_present") is not True:
            return rejected("campus_readiness_not_met")
    elif readiness.get("accuracy_claim_allowed") is not True:
        return rejected("campus_readiness_not_met")

    checks = readiness.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    models = checks.get("models")
    models = models if isinstance(models, dict) else {}
    model_check = models.get(model_kind)
    if (not isinstance(model_check, dict) or model_check.get("status") != "present"
            or model_check.get("sha256") != actual_model_hash):
        return rejected("readiness_model_mismatch")

    if model_kind == "accident":
        evaluation = checks.get("event_evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        thresholds = evaluation.get("thresholds")
        validated_threshold = _number(thresholds.get("accident")) if isinstance(thresholds, dict) else None
        if not _same_number(
                validated_threshold, runtime_config.get("decision_threshold")):
            return rejected("accident_runtime_threshold_mismatch")
        campus_targets = evaluation.get("event_metric_targets")
        campus_target = campus_targets.get("accident") if isinstance(campus_targets, dict) else None
        campus_fp = (campus_target.get("false_positive_count")
                     if isinstance(campus_target, dict) else None)
        campus_flights = (campus_target.get("test_flight_count")
                          if isinstance(campus_target, dict) else None)

        external = checks.get("external_accident_evaluation")
        external = external if isinstance(external, dict) else {}
        benchmark_approval = entry.get("benchmark_approval")
        benchmark_approval = benchmark_approval if isinstance(benchmark_approval, dict) else {}
        benchmark_approved_at = _aware_time(benchmark_approval.get("approved_at"))
        benchmark_expires_at = _aware_time(benchmark_approval.get("expires_at"))
        benchmark_approval_valid = bool(
            benchmark_approval.get("approved") is True
            and isinstance(benchmark_approval.get("approved_by"), str)
            and benchmark_approval["approved_by"].strip()
            and isinstance(benchmark_approval.get("evidence"), str)
            and benchmark_approval["evidence"].strip()
            and benchmark_approval.get("manifest_sha256") == external.get("manifest_sha256")
            and benchmark_approved_at is not None and benchmark_approved_at <= now
            and benchmark_expires_at is not None
            and benchmark_expires_at > now
            and benchmark_expires_at > benchmark_approved_at
            and benchmark_expires_at - benchmark_approved_at <= MAX_APPROVAL_VALIDITY
        )
        external_source = external.get("source_provenance")
        external_source = external_source if isinstance(external_source, dict) else {}
        external_precision = _number(external.get("precision"))
        external_recall = _number(external.get("recall"))
        external_groups = external.get("test_group_count")
        external_positive_groups = external.get("positive_test_group_count")
        external_hashes_valid = all(
            isinstance(external.get(field), str) and _SHA256_RE.fullmatch(external[field])
            for field in ("manifest_sha256", "report_sha256")
        )
        detector_model_check = models.get("detector")
        detector_model_hash = (detector_model_check.get("sha256")
                               if isinstance(detector_model_check, dict) else None)
        if (evaluation.get("status") != "ready"
                or evaluation.get("campus_negative_safety_ready") is not True
                or "accident" not in (evaluation.get("event_kinds") or [])
                or type(campus_flights) is not int or campus_flights < 3
                or type(campus_fp) is not int or campus_fp != 0
                or external.get("status") != "ready"
                or external.get("scope") != "public_benchmark"
                or not benchmark_approval_valid
                or not isinstance(external_source.get("source"), str)
                or not external_source["source"].strip()
                or not isinstance(external_source.get("license"), str)
                or not external_source["license"].strip()
                or not isinstance(external_source.get("url"), str)
                or not external_source["url"].lower().startswith(("https://", "http://"))
                or not isinstance(external_source.get("license_evidence"), str)
                or not external_source["license_evidence"].strip()
                or external.get("model_sha256") != actual_model_hash
                or external.get("detector_sha256") != detector_model_hash
                or not _same_number(
                    external.get("threshold"), runtime_config.get("decision_threshold"))
                or not external_hashes_valid
                or external_precision is None or external_precision < 0.90
                or external_recall is None or external_recall < 0.80
                or type(external_groups) is not int or external_groups < 3
                or type(external_positive_groups) is not int or external_positive_groups < 1):
            return rejected("accident_field_evidence_not_met")
    else:
        evaluation = checks.get("event_evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        thresholds = evaluation.get("thresholds")
        validated_area = _number(thresholds.get("flood_min_area_ratio")) if isinstance(thresholds, dict) else None
        validated_probability = (
            thresholds.get("flood_probability")
            if isinstance(thresholds, dict) and "flood_probability" in thresholds else "missing"
        )
        runtime_probability = runtime_config.get("probability_threshold")
        if (not _same_number(validated_area, runtime_config.get("min_area_ratio"))
                or validated_probability == "missing"
                or (validated_probability is None) != (runtime_probability is None)
                or (validated_probability is not None and (
                    not _same_number(validated_probability, runtime_probability)
                ))):
            return rejected("flood_runtime_threshold_mismatch")
        water = checks.get("visible_water_segmentation")
        water = water if isinstance(water, dict) else {}
        iou = _number(water.get("iou"))
        flights = water.get("test_flight_count")
        if (water.get("status") != "ready" or iou is None or iou < 0.60
                or type(flights) is not int or flights < 3):
            return rejected("flood_field_evidence_not_met")

    return {"approved": True, "reason": "approved"}
