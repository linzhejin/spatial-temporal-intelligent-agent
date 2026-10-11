"""Read-only readiness audit for a campus UAV field evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from vision.campus_detection_evaluation import validate_campus_detection_manifest
from vision.campus_water_data import build_campus_flight_plan
from vision.event_evaluation import validate_event_manifest
from vision.specialized_evaluation import validate_split_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_file(path: Path | None, field: str, issues: list[str]) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(f"{field}: {error}")
        return None, None
    if not isinstance(value, dict):
        issues.append(f"{field}: JSON root must be an object")
        return None, None
    return value, hashlib.sha256(raw).hexdigest()


def _asset(path: Path | None, field: str, issues: list[str]) -> dict:
    if path is None:
        return {"status": "missing", "file": None, "sha256": None}
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError:
        issues.append(f"{field}: file does not exist")
        return {"status": "invalid", "file": str(path), "sha256": None}
    if not resolved.is_file():
        issues.append(f"{field}: path is not a file")
        return {"status": "invalid", "file": str(resolved), "sha256": None}
    return {"status": "present", "file": str(resolved), "sha256": _sha256(resolved)}


def _count(row: object, field: str) -> int | None:
    if not isinstance(row, dict):
        return None
    value = row.get(field)
    return value if type(value) is int and value >= 0 else None


def _metric(row: object, field: str):
    if not isinstance(row, dict):
        return None
    value = row.get(field)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0 <= value <= 1):
        return None
    return float(value)


def _report_check(
    report_path: Path | None,
    *,
    expected_task: str,
    manifest_sha256: str | None,
    model_hashes: dict[str, str | None],
) -> tuple[dict, dict | None]:
    issues: list[str] = []
    report, _ = _json_file(report_path, "report", issues)
    if report_path is None:
        return {"status": "missing", "file": None, "issues": []}, None
    if report is None:
        return {"status": "invalid", "file": str(report_path), "issues": issues}, None
    if report.get("task") != expected_task:
        issues.append("task")
    if manifest_sha256 is None or report.get("manifest_sha256") != manifest_sha256:
        issues.append("manifest_sha256")
    if expected_task == "campus_object_detection":
        if model_hashes.get("detector") is None or report.get("model_sha256") != model_hashes["detector"]:
            issues.append("model_sha256")
        if report.get("split") != "test" or not isinstance(report.get("groups"), dict):
            issues.append("test metrics")
    else:
        provenance = report.get("model_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        if (model_hashes.get("detector") is None
                or provenance.get("detector_sha256") != model_hashes["detector"]):
            issues.append("detector_sha256")
        for key, report_key in (("accident", "accident_model"), ("flood", "flood_model")):
            expected = model_hashes.get(key)
            if expected is not None:
                actual_record = provenance.get(report_key)
                actual = actual_record.get("sha256") if isinstance(actual_record, dict) else None
                if actual != expected:
                    issues.append(f"{report_key}.sha256")
        if report.get("split") != "test" or not isinstance(report.get("event_metrics"), dict):
            issues.append("test metrics")
    return {
        "status": "ready" if not issues else "invalid",
        "file": str(Path(report_path).resolve()),
        "sha256": _sha256(Path(report_path)),
        "issues": issues,
    }, report


def audit_campus_field_readiness(
    *,
    dataset_root: Path | None = None,
    flight_manifest: Path | None = None,
    detection_manifest: Path | None = None,
    detection_report: Path | None = None,
    event_manifest: Path | None = None,
    event_report: Path | None = None,
    visible_water_manifest: Path | None = None,
    visible_water_report: Path | None = None,
    detector_model: Path | None = None,
    accident_model: Path | None = None,
    flood_model: Path | None = None,
) -> dict:
    """Check source flights, reviewed labels, model files, and report provenance."""
    root = Path(dataset_root).resolve() if dataset_root is not None else None
    next_actions: list[str] = []
    checks: dict[str, dict] = {}

    flight_issues: list[str] = []
    flight_manifest_value, flight_manifest_sha256 = _json_file(
        Path(flight_manifest) if flight_manifest is not None else None,
        "flight_manifest", flight_issues,
    )
    flight_plan = None
    if flight_manifest is not None and flight_manifest_value is not None:
        if root is None or not root.is_dir():
            flight_issues.append("dataset_root: source dataset directory is missing")
        else:
            try:
                flight_plan = build_campus_flight_plan(flight_manifest_value, root)
            except (OSError, ValueError) as error:
                flight_issues.append(f"flight_manifest: {error}")
    if flight_manifest is None:
        flight_status = "missing"
        next_actions.append("整理真实校园航次清单，记录航次、地点、时间和原片路径。")
    else:
        flight_status = "ready" if flight_plan and not flight_issues else "invalid"
        if flight_status == "invalid":
            next_actions.append("修正航次清单或原片路径，直到每个训练、验证、测试航次都能通过校验。")
    flight_counts = ({split: len(values) for split, values in flight_plan["split_groups"].items()}
                     if flight_plan else None)
    flight_locations = [item["location"] for item in flight_plan["flights"]] if flight_plan else []
    dataset_name = str(flight_manifest_value.get("dataset") or "") if flight_manifest_value else ""
    campus_markers = ("whu", "wuhan", "武汉大学", "武大")
    explicitly_campus = any(marker in dataset_name.casefold() for marker in campus_markers) or any(
        any(marker in location.casefold() for marker in campus_markers)
        for location in flight_locations
    )
    checks["flight_inventory"] = {
        "status": flight_status,
        "dataset": dataset_name or None,
        "split_flight_counts": flight_counts,
        "campus_named_in_metadata": explicitly_campus,
        "issues": flight_issues,
    }
    if flight_status == "ready" and not explicitly_campus:
        next_actions.append("在航次清单的数据集名称或地点字段注明武汉大学校园，以区分公开样例。")

    detector_issues: list[str] = []
    detector = _asset(Path(detector_model) if detector_model is not None else None,
                      "detector_model", detector_issues)
    accident = _asset(Path(accident_model) if accident_model is not None else None,
                      "accident_model", [])
    flood = _asset(Path(flood_model) if flood_model is not None else None,
                   "flood_model", [])

    detection_issues: list[str] = []
    detection_value, detection_manifest_sha256 = _json_file(
        Path(detection_manifest) if detection_manifest is not None else None,
        "detection_manifest", detection_issues,
    )
    detection_validated = None
    if detection_manifest is not None and detection_value is not None:
        try:
            detection_validated = validate_campus_detection_manifest(detection_value)
        except ValueError as error:
            detection_issues.append(f"detection_manifest: {error}")
    detection_report_check, detection_report_value = _report_check(
        Path(detection_report) if detection_report is not None else None,
        expected_task="campus_object_detection",
        manifest_sha256=detection_manifest_sha256,
        model_hashes={"detector": detector["sha256"]},
    )
    if detection_report_value is not None and detection_validated is not None:
        if detection_report_value.get("flight_group_count") != len(
                detection_validated["split_groups"]["test"]):
            detection_issues.append("report flight_group_count does not match manifest")
        if detection_report_value.get("image_count") != len(detection_validated["samples"]):
            detection_issues.append("report image_count does not match manifest")
    if detection_manifest is None:
        detection_status = "missing"
        next_actions.append("人工标注独立测试航次中的行人和车辆，生成检测测试清单。")
    elif detection_issues:
        detection_status = "invalid"
    elif detector["status"] != "present" or detection_report_check["status"] != "ready":
        detection_status = "missing"
        if detector["status"] != "present":
            next_actions.append("准备可加载的校园行人车辆检测模型。")
        if detection_report_check["status"] == "missing":
            next_actions.append("在人工标注完成后运行校园检测评测，保存测试报告。")
    else:
        detection_status = "ready"
    if detection_report_check["status"] == "invalid":
        detection_issues.extend(detection_report_check["issues"])
        detection_status = "invalid"
    detection_split_groups = (detection_validated["split_groups"]["test"]
                              if detection_validated else [])
    detection_support = None
    if detection_report_value and isinstance(detection_report_value.get("groups"), dict):
        detection_support = {}
        for name in ("pedestrian", "vehicle"):
            tp = _count(detection_report_value["groups"].get(name), "tp")
            fn = _count(detection_report_value["groups"].get(name), "fn")
            if tp is None or fn is None:
                detection_issues.append(f"report has invalid {name} TP/FN counts")
            else:
                detection_support[name] = tp + fn
    if detection_issues:
        detection_status = "invalid"
    detection_claim_ready = (
        detection_status == "ready" and len(detection_split_groups) >= 3
        and all(detection_support and detection_support.get(name, 0) > 0
                for name in ("pedestrian", "vehicle"))
    )
    if detection_status == "ready" and not detection_claim_ready:
        next_actions.append("扩大检测测试集到至少 3 个独立校园航次，并确保行人、车辆均有人工标注正例。")
    checks["object_detection"] = {
        "status": detection_status,
        "test_flight_count": len(detection_split_groups) if detection_validated else None,
        "test_ground_truth_support": detection_support,
        "report": detection_report_check,
        "issues": detection_issues,
        "accuracy_claim_ready": bool(detection_claim_ready),
    }

    event_issues: list[str] = []
    event_value, event_manifest_sha256 = _json_file(
        Path(event_manifest) if event_manifest is not None else None,
        "event_manifest", event_issues,
    )
    event_validated = None
    if event_manifest is not None and event_value is not None:
        try:
            event_validated = validate_event_manifest(event_value)
        except ValueError as error:
            event_issues.append(f"event_manifest: {error}")
    event_kinds = event_validated["event_kinds"] if event_validated else []
    event_model_hashes = {
        "detector": detector["sha256"],
        "accident": accident["sha256"] if "accident" in event_kinds else None,
        "flood": flood["sha256"] if "flooding" in event_kinds else None,
    }
    event_report_check, event_report_value = _report_check(
        Path(event_report) if event_report is not None else None,
        expected_task="campus_event_detection",
        manifest_sha256=event_manifest_sha256,
        model_hashes=event_model_hashes,
    )
    if event_report_value is not None and event_validated is not None:
        if event_report_value.get("test_group_count") != len(
                event_validated["split_groups"]["test"]):
            event_issues.append("report test_group_count does not match manifest")
        if event_report_value.get("sample_count") != len(event_validated["samples"]):
            event_issues.append("report sample_count does not match manifest")
        if event_report_value.get("event_kinds") != event_validated["event_kinds"]:
            event_issues.append("report event_kinds do not match manifest")
    missing_models = []
    if "accident" in event_kinds and accident["status"] != "present":
        missing_models.append("accident")
    if "flooding" in event_kinds and flood["status"] != "present":
        missing_models.append("flooding")
    if event_manifest is None:
        event_status = "missing"
        next_actions.append("按独立航次逐段复核视频，填写拥堵、人流、事故、积水类型和观察区域。")
    elif event_issues:
        event_status = "invalid"
    elif (detector["status"] != "present" or missing_models
          or event_report_check["status"] != "ready"):
        event_status = "missing"
        if missing_models:
            next_actions.append("准备并完成许可核查：" + "、".join(missing_models) + "候选模型。")
        if event_report_check["status"] == "missing":
            next_actions.append("使用已复核的视频清单运行事件级评测，按类型保存结果。")
    else:
        event_status = "ready"
    if event_report_check["status"] == "invalid":
        event_issues.extend(event_report_check["issues"])
        event_status = "invalid"
    event_test_groups = event_validated["split_groups"]["test"] if event_validated else []
    event_counts = None
    event_targets = None
    if event_report_value:
        metrics = event_report_value.get("event_metrics")
        by_kind = metrics.get("by_kind") if isinstance(metrics, dict) else None
        if isinstance(by_kind, dict):
            event_counts = {
                kind: {
                    "tp": _count(row.get("counts"), "tp"),
                    "fp": _count(row.get("counts"), "fp"),
                    "fn": _count(row.get("counts"), "fn"),
                }
                for kind, row in by_kind.items() if isinstance(row, dict)
            }
            for kind, counts in event_counts.items():
                if any(value is None for value in counts.values()):
                    event_issues.append(f"report has invalid {kind} event counts")
            event_targets = {
                kind: {
                    "precision": _metric(row.get("metrics"), "precision")
                    if isinstance(row, dict) else None,
                    "recall": _metric(row.get("metrics"), "recall")
                    if isinstance(row, dict) else None,
                    "precision_target": 0.90,
                    "recall_target": 0.80,
                    "met": bool(
                        isinstance(row, dict)
                        and _metric(row.get("metrics"), "precision") is not None
                        and _metric(row.get("metrics"), "recall") is not None
                        and _metric(row.get("metrics"), "precision") >= 0.90
                        and _metric(row.get("metrics"), "recall") >= 0.80
                    ),
                }
                for kind, row in by_kind.items()
            }
    event_claim_ready = (
        event_status == "ready" and len(event_test_groups) >= 3
        and all(event_counts and event_counts.get(kind, {}).get("tp") is not None
                and event_counts.get(kind, {}).get("fn") is not None
                and event_counts[kind]["tp"] + event_counts[kind]["fn"] > 0
                for kind in event_kinds)
        and all(event_targets and event_targets.get(kind, {}).get("met") is True
                for kind in event_kinds)
    )
    if event_issues:
        event_status = "invalid"
        event_claim_ready = False
    if event_status == "ready" and not event_claim_ready:
        next_actions.append("每类要报告效果的事件至少覆盖 3 个独立校园航次，并包含人工确认的正例。")

    water_issues: list[str] = []
    water_manifest_value, water_manifest_sha256 = _json_file(
        Path(visible_water_manifest) if visible_water_manifest is not None else None,
        "visible_water_manifest", water_issues,
    )
    water_report, _ = _json_file(
        Path(visible_water_report) if visible_water_report is not None else None,
        "visible_water_report", water_issues,
    )
    water_check = {"status": "missing", "file": None, "issues": []}
    water_test_groups = []
    if visible_water_manifest is not None and water_manifest_value is not None:
        try:
            validated_water_manifest = validate_split_manifest(
                water_manifest_value, task="visible_water_road_segmentation",
            )
            water_test_groups = validated_water_manifest["split_groups"]["test"]
            if water_manifest_value.get("label_source") != "manual":
                water_issues.append("visible-water manifest label_source must be manual")
            if water_manifest_value.get("annotation_pending") is True:
                water_issues.append("visible-water manifest still has annotation_pending")
            if water_manifest_value.get("split_unit") != "flight_id":
                water_issues.append("visible-water split_unit must be flight_id")
        except ValueError as error:
            water_issues.append(f"visible_water_manifest: {error}")
    elif visible_water_report is not None:
        water_issues.append("visible_water_manifest is required to verify report provenance")
    if visible_water_report is not None:
        water_report_path = Path(visible_water_report)
        if water_report is None:
            water_check = {
                "status": "invalid", "file": str(water_report_path), "issues": water_issues,
            }
        else:
            water_report_issues = []
            water_report_issues.extend(water_issues)
            if water_report.get("task") != "visible_water_road_segmentation":
                water_report_issues.append("task")
            if (water_manifest_sha256 is None
                    or water_report.get("manifest_sha256") != water_manifest_sha256):
                water_report_issues.append("manifest_sha256")
            if (flood["sha256"] is None
                    or water_report.get("model_sha256") != flood["sha256"]):
                water_report_issues.append("model_sha256")
            if water_report.get("split") != "test":
                water_report_issues.append("split")
            if (water_manifest_value is not None
                    and water_report.get("sample_count") != len(
                        water_manifest_value.get("samples", [])
                    )):
                water_report_issues.append("sample_count does not match manifest")
            if (len(water_test_groups) < 3
                    or water_report.get("test_group_count") != len(water_test_groups)):
                water_report_issues.append("at least 3 flight-disjoint test groups")
            if water_report.get("dataset") is None or not any(
                marker in str(water_report.get("dataset")).casefold()
                for marker in campus_markers
            ):
                water_report_issues.append("campus dataset identity")
            water_metrics = water_report.get("metrics")
            water_iou = water_metrics.get("iou") if isinstance(water_metrics, dict) else None
            if _metric(water_metrics, "iou") is None or _metric(water_metrics, "iou") < 0.60:
                water_report_issues.append("road-region IoU target 0.60")
            water_group_metrics = (
                water_metrics.get("per_group") if isinstance(water_metrics, dict) else None
            )
            if not isinstance(water_group_metrics, dict):
                water_report_issues.append("per-flight water metrics")
            elif set(water_group_metrics) != set(water_test_groups):
                water_report_issues.append("per-flight metrics do not match manifest")
            else:
                has_positive_flight = any(
                    isinstance(row, dict) and row.get("negative_flight") is False
                    for row in water_group_metrics.values()
                )
                has_negative_flight = any(
                    isinstance(row, dict) and row.get("negative_flight") is True
                    for row in water_group_metrics.values()
                )
                if not has_positive_flight or not has_negative_flight:
                    water_report_issues.append("test set requires positive and negative water flights")
            water_check = {
                "status": "ready" if not water_report_issues else "invalid",
                "file": str(water_report_path.resolve()),
                "sha256": _sha256(water_report_path),
                "iou": water_iou,
                "iou_target": 0.60,
                "test_flight_count": len(water_test_groups),
                "issues": water_report_issues,
            }
    if visible_water_report is None:
        if water_issues:
            water_check = {
                "status": "invalid",
                "file": str(Path(visible_water_manifest).resolve())
                if visible_water_manifest is not None else None,
                "issues": water_issues,
            }
        next_actions.append("用人工标注的独立校园航次评测道路可见水体分割，并报告路面 IoU。")
    elif water_check["status"] != "ready":
        water_issues.extend(water_check["issues"])
        next_actions.append("积水评测须使用校园测试航次、匹配权重和道路区域标注，并达到 IoU 0.60。")
    checks["event_evaluation"] = {
        "status": event_status,
        "event_kinds": event_kinds,
        "test_flight_count": len(event_test_groups) if event_validated else None,
        "test_ground_truth_counts": event_counts,
        "event_metric_targets": event_targets,
        "missing_specialized_models": missing_models,
        "report": event_report_check,
        "issues": event_issues,
        "accuracy_claim_ready": bool(event_claim_ready),
    }
    checks["models"] = {"detector": detector, "accident": accident, "flood": flood}

    campus_field_data_present = flight_status == "ready" and explicitly_campus
    accuracy_claim_allowed = (
        campus_field_data_present and detection_claim_ready and event_claim_ready
        and ("flooding" not in event_kinds or water_check["status"] == "ready")
    )
    checks["visible_water_segmentation"] = water_check
    return {
        "task": "campus_uav_field_readiness",
        "campus_field_data_present": campus_field_data_present,
        "accuracy_claim_allowed": bool(accuracy_claim_allowed),
        "checks": checks,
        "next_actions": next_actions,
        "limitations": [
            "就绪审计只检查清单、人工标签、模型文件和报告来源一致性，不运行模型推理。",
            "公开数据和示例视频不能作为武汉大学校园效果证据。",
            "只有独立校园航次上的人工测试标签与对应评测报告齐备，才允许报告校园准确率。",
        ],
    }
