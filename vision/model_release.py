"""Fail-closed release checks for optional UAV models."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


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


def check_optional_model_release(
    model_kind: str,
    model_path: str | Path | None,
    model_revision: str | None,
    release_manifest_path: str | Path | None,
    readiness_report_path: str | Path | None,
    runtime_config: dict | None = None,
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
    if not readiness_report_path:
        return rejected("readiness_report_missing")
    readiness_path = Path(readiness_report_path)
    if not readiness_path.is_file():
        return rejected("readiness_report_missing")

    release = _read_json(release_path)
    readiness = _read_json(readiness_path)
    if release is None or release.get("schema_version") != 1:
        return rejected("release_manifest_invalid")
    if readiness is None:
        return rejected("readiness_report_invalid")

    entries = release.get("models")
    entry = entries.get(model_kind) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        return rejected("model_release_missing")
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
    if (readiness.get("task") != "campus_uav_field_readiness"
            or readiness.get("accuracy_claim_allowed") is not True):
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
        targets = evaluation.get("event_metric_targets")
        target = targets.get("accident") if isinstance(targets, dict) else None
        precision = _number(target.get("precision")) if isinstance(target, dict) else None
        recall = _number(target.get("recall")) if isinstance(target, dict) else None
        flights = target.get("test_flight_count") if isinstance(target, dict) else None
        positive_flights = target.get("positive_test_flight_count") if isinstance(target, dict) else None
        if (evaluation.get("status") != "ready"
                or evaluation.get("accuracy_claim_ready") is not True
                or "accident" not in (evaluation.get("event_kinds") or [])
                or precision is None or precision < 0.90
                or recall is None or recall < 0.80
                or type(flights) is not int or flights < 3
                or type(positive_flights) is not int or positive_flights < 1):
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
