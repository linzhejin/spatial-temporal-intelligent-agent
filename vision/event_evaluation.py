"""Leakage-aware event-level evaluation for reviewed aerial-video candidates."""
from __future__ import annotations

import math
import re

from vision.specialized_evaluation import cluster_bootstrap_interval, summarize_binary_counts


EVENT_KIND_TO_CANDIDATE_KIND = {
    "congestion": "possible_congestion",
    "crowding": "possible_crowding",
    "accident": "possible_accident",
    "flooding": "possible_flooding",
}
CANDIDATE_KIND_TO_EVENT_KIND = {
    value: key for key, value in EVENT_KIND_TO_CANDIDATE_KIND.items()
}
EVENT_KIND_TO_REGION_KIND = {
    "congestion": "vehicle_lane",
    "crowding": "pedestrian",
    "accident": "vehicle_lane",
    "flooding": "road_surface",
}
OBSERVATION_REGION_KINDS = {"vehicle_lane", "pedestrian", "road_surface", "parking", "exclude"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _event_kind(item: dict) -> str:
    """Treat old accident-only rows without a kind as accident labels."""
    return str(item.get("kind") or "accident").strip()


def _event_kinds(value, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    kinds = [str(item).strip() for item in value]
    if (any(kind not in EVENT_KIND_TO_CANDIDATE_KIND for kind in kinds)
            or len(kinds) != len(set(kinds))):
        raise ValueError(f"{field} contains an unsupported event kind or duplicate")
    return kinds


def _safe_time(value, field: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _validate_polygon(polygon, *, sample_id: str, region_id: str) -> None:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError(f"sample {sample_id} region {region_id} requires a polygon with at least three points")
    points = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"sample {sample_id} region {region_id} has an invalid polygon point")
        x, y = (_safe_time(value, "polygon coordinate") for value in point)
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"sample {sample_id} region {region_id} polygon coordinates must be between zero and one")
        points.append((x, y))
    area = abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2
    if area <= 1e-9:
        raise ValueError(f"sample {sample_id} region {region_id} polygon must have non-zero area")


def validate_event_manifest(manifest: dict) -> dict:
    """Validate one held-out video per event window and flight/scene-disjoint groups."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("event manifest schema_version must be 1")
    task = manifest.get("task")
    if task not in {"accident_event_detection", "campus_event_detection"}:
        raise ValueError("event manifest task must be accident_event_detection or campus_event_detection")
    legacy_accident_task = task == "accident_event_detection"
    manifest_event_kinds = (
        ["accident"] if legacy_accident_task
        else _event_kinds(manifest.get("event_kinds"), "event_kinds")
    )
    if not legacy_accident_task and manifest.get("label_source") != "manual":
        raise ValueError("campus event manifest label_source must be manual")
    if not legacy_accident_task and manifest.get("annotation_pending") is True:
        raise ValueError("campus event manifest annotation_pending must be removed after manual review")
    if not str(manifest.get("dataset") or "").strip():
        raise ValueError("event manifest dataset name is required")
    split_unit = str(manifest.get("split_unit") or "").strip()
    split_unit_key = split_unit.lower().replace("-", "_")
    if not split_unit or any(token in split_unit_key for token in
                             ("frame", "image", "pixel", "sample", "clip", "video")):
        raise ValueError("split_unit must identify an independent flight, scene, or incident")

    groups = manifest.get("split_groups")
    if not isinstance(groups, dict):
        raise ValueError("event manifest split_groups is required")
    normalized_groups = {}
    for split in ("train", "validation", "test"):
        values = groups.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"split_groups.{split} must be a non-empty list")
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError(f"split_groups.{split} contains empty or duplicate IDs")
        normalized_groups[split] = normalized
    all_groups = [group for values in normalized_groups.values() for group in values]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("train, validation, and test groups must not overlap")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("event manifest samples must be a non-empty list")
    sample_ids, video_paths, observed_test_groups, observed_event_kinds = set(), set(), set(), set()
    normalized_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        sample_id = str(sample.get("sample_id") or "").strip()
        group_id = str(sample.get("group_id") or "").strip()
        video = str(sample.get("video") or "").strip().replace("\\", "/")
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"sample {index} requires a unique sample_id")
        if group_id not in normalized_groups["test"]:
            raise ValueError(f"sample {sample_id} must belong to a declared test group")
        if (not video or video.startswith("/") or ":" in video.split("/")[0]
                or any(part in {"", ".", ".."} for part in video.split("/"))):
            raise ValueError(f"sample {sample_id} video must use a safe relative path")
        if video in video_paths:
            raise ValueError("event manifest contains a duplicate video path")
        sample_ids.add(sample_id)
        video_paths.add(video)
        observed_test_groups.add(group_id)
        if not isinstance(sample.get("camera_stabilized"), bool):
            raise ValueError(f"sample {sample_id} camera_stabilized must be boolean")
        source_sha256 = sample.get("source_sha256")
        if source_sha256 is not None:
            source_sha256 = str(source_sha256).strip().lower()
            if not _SHA256.fullmatch(source_sha256):
                raise ValueError(f"sample {sample_id} source_sha256 must be a lowercase SHA-256")

        sample_event_kinds = (
            ["accident"] if legacy_accident_task
            else _event_kinds(sample.get("event_kinds"), f"sample {sample_id} event_kinds")
        )
        if not set(sample_event_kinds).issubset(manifest_event_kinds):
            raise ValueError(f"sample {sample_id} event kind is not declared by the manifest")
        observed_event_kinds.update(sample_event_kinds)

        regions = sample.get("observation_regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"sample {sample_id} requires marked observation regions")
        region_ids = set()
        region_kinds = {}
        for region in regions:
            if not isinstance(region, dict):
                raise ValueError(f"sample {sample_id} observation regions must be objects")
            region_id = str(region.get("id") or "").strip()
            region_kind = str(region.get("kind") or "").strip()
            if not region_id or region_id in region_ids or region_kind not in OBSERVATION_REGION_KINDS:
                raise ValueError(f"sample {sample_id} regions must have unique IDs and supported kinds")
            _validate_polygon(region.get("polygon"), sample_id=sample_id, region_id=region_id)
            region_ids.add(region_id)
            region_kinds[region_id] = region_kind
        for event_kind in sample_event_kinds:
            required_region_kind = EVENT_KIND_TO_REGION_KIND[event_kind]
            if required_region_kind not in region_kinds.values():
                raise ValueError(
                    f"sample {sample_id} event kind {event_kind} requires a "
                    f"{required_region_kind} observation region"
                )

        events = sample.get("events")
        if not isinstance(events, list):
            raise ValueError(f"sample {sample_id} events must be a list; use an empty list for a negative clip")
        event_ids = set()
        normalized_events = []
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"sample {sample_id} event {event_index} must be an object")
            event_id = str(event.get("event_id") or "").strip()
            region_id = str(event.get("region_id") or "").strip()
            event_kind = "accident" if legacy_accident_task else _event_kind(event)
            start = _safe_time(event.get("start_seconds"), "start_seconds")
            end = _safe_time(event.get("end_seconds"), "end_seconds")
            if not event_id or event_id in event_ids:
                raise ValueError(f"sample {sample_id} requires unique event_id values")
            if event_kind not in EVENT_KIND_TO_CANDIDATE_KIND:
                raise ValueError(f"sample {sample_id} event {event_id} has an unsupported event kind")
            if event_kind not in sample_event_kinds:
                raise ValueError(f"sample {sample_id} event {event_id} kind is not enabled for this sample")
            expected_region_kind = EVENT_KIND_TO_REGION_KIND[event_kind]
            if region_id not in region_ids or region_kinds[region_id] != expected_region_kind:
                marked = "marked " if legacy_accident_task else ""
                raise ValueError(
                    f"sample {sample_id} event {event_id} must reference a {marked}"
                    f"{expected_region_kind} region"
                )
            if start < 0 or end <= start:
                raise ValueError("end_seconds must be after start_seconds and start_seconds cannot be negative")
            event_ids.add(event_id)
            normalized_event = {**event, "event_id": event_id, "region_id": region_id,
                                "start_seconds": start, "end_seconds": end}
            if not legacy_accident_task:
                normalized_event["kind"] = event_kind
            normalized_events.append(normalized_event)
        normalized_sample = {**sample, "sample_id": sample_id, "group_id": group_id,
                             "video": video, "events": normalized_events}
        if source_sha256 is not None:
            normalized_sample["source_sha256"] = source_sha256
        if not legacy_accident_task:
            normalized_sample["event_kinds"] = sample_event_kinds
        normalized_samples.append(normalized_sample)

    if observed_test_groups != set(normalized_groups["test"]):
        raise ValueError("test groups must all have at least one video")
    if observed_event_kinds != set(manifest_event_kinds):
        raise ValueError("every declared event kind must be enabled by at least one sample")

    result = dict(manifest)
    result["event_kinds"] = manifest_event_kinds
    result["split_groups"] = normalized_groups
    result["samples"] = normalized_samples
    return result


def extract_event_candidates(analysis: dict, *, event_kinds=None) -> list[dict]:
    """Convert supported production candidates to typed temporal event intervals."""
    selected_kinds = (
        set(EVENT_KIND_TO_CANDIDATE_KIND)
        if event_kinds is None else set(_event_kinds(list(event_kinds), "event_kinds"))
    )
    candidates = []
    rows = analysis.get("candidates", []) if isinstance(analysis, dict) else []
    if not isinstance(rows, list):
        return candidates
    for index, candidate in enumerate(rows):
        if not isinstance(candidate, dict):
            continue
        event_kind = CANDIDATE_KIND_TO_EVENT_KIND.get(candidate.get("kind"))
        if event_kind not in selected_kinds:
            continue
        evidence = candidate.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        summary = evidence.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        region_id = str(summary.get("region_id") or evidence.get("region_id") or "").strip()
        segments = evidence.get("segments")
        if not region_id or not isinstance(segments, list):
            continue
        intervals = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            try:
                start = _safe_time(segment.get("start_seconds"), "candidate start_seconds")
                end = _safe_time(segment.get("end_seconds"), "candidate end_seconds")
            except ValueError:
                continue
            if start >= 0 and end > start:
                intervals.append((start, end))
        if not intervals:
            continue
        try:
            confidence = _safe_time(candidate.get("confidence"), "candidate confidence")
        except ValueError:
            confidence = 0.0
        candidates.append({
            "candidate_id": str(candidate.get("candidate_id") or f"candidate-{index + 1}"),
            "kind": event_kind,
            "region_id": region_id,
            "start_seconds": min(item[0] for item in intervals),
            "end_seconds": max(item[1] for item in intervals),
            "confidence": confidence,
        })
    return candidates


def extract_accident_event_candidates(analysis: dict) -> list[dict]:
    """Backward-compatible accident-only candidate extraction."""
    return [
        {key: value for key, value in candidate.items() if key != "kind"}
        for candidate in extract_event_candidates(analysis, event_kinds=["accident"])
    ]


def _temporal_iou(left: dict, right: dict) -> float:
    start = max(float(left["start_seconds"]), float(right["start_seconds"]))
    end = min(float(left["end_seconds"]), float(right["end_seconds"]))
    overlap = max(0.0, end - start)
    union_start = min(float(left["start_seconds"]), float(right["start_seconds"]))
    union_end = max(float(left["end_seconds"]), float(right["end_seconds"]))
    union = union_end - union_start
    return overlap / union if union > 0 else 0.0


def match_event_candidates(actual_events: list[dict], predicted_events: list[dict], *,
                           min_temporal_iou: float = 0.1) -> dict:
    """One-to-one matching by region and maximum temporal intersection-over-union."""
    if (isinstance(min_temporal_iou, bool) or not isinstance(min_temporal_iou, (int, float))
            or not math.isfinite(float(min_temporal_iou)) or not 0 <= min_temporal_iou <= 1):
        raise ValueError("min_temporal_iou must be between zero and one")
    actual = list(actual_events)
    predicted = list(predicted_events)
    if not actual or not predicted:
        return {"matches": [],
                "unmatched_event_ids": [str(item.get("event_id") or "") for item in actual],
                "unmatched_candidate_ids": [str(item.get("candidate_id") or "") for item in predicted]}

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    scores = np.zeros((len(actual), len(predicted)), dtype=np.float64)
    for row, event in enumerate(actual):
        for col, candidate in enumerate(predicted):
            if (_event_kind(event) != _event_kind(candidate)
                    or str(event.get("region_id")) != str(candidate.get("region_id"))):
                continue
            iou = _temporal_iou(event, candidate)
            if iou >= float(min_temporal_iou) and iou > 0:
                scores[row, col] = iou
    rows, columns = linear_sum_assignment(-scores)
    matches = []
    matched_actual, matched_predicted = set(), set()
    for row, column in zip(rows, columns):
        score = float(scores[row, column])
        if score <= 0:
            continue
        event, candidate = actual[row], predicted[column]
        matches.append({"event_id": str(event.get("event_id") or ""),
                        "candidate_id": str(candidate.get("candidate_id") or ""),
                        "kind": _event_kind(event),
                        "region_id": str(event.get("region_id") or ""),
                        "temporal_iou": round(score, 6)})
        matched_actual.add(row)
        matched_predicted.add(column)
    matches.sort(key=lambda item: (item["event_id"], item["candidate_id"]))
    return {
        "matches": matches,
        "unmatched_event_ids": [str(item.get("event_id") or "") for index, item in enumerate(actual)
                                if index not in matched_actual],
        "unmatched_candidate_ids": [str(item.get("candidate_id") or "")
                                    for index, item in enumerate(predicted)
                                    if index not in matched_predicted],
    }


def summarize_event_detection(manifest: dict, predictions_by_sample: dict[str, list[dict]], *,
                              min_temporal_iou: float = 0.1,
                              bootstrap_repetitions: int = 1000,
                              seed: int = 20261010) -> dict:
    """Report event precision/recall with confidence intervals clustered by flight."""
    if not isinstance(predictions_by_sample, dict):
        raise ValueError("predictions_by_sample must be an object keyed by sample_id")
    known_ids = {sample["sample_id"] for sample in manifest["samples"]}
    unknown_ids = set(predictions_by_sample) - known_ids
    if unknown_ids:
        raise ValueError(f"unknown sample_id: {sorted(unknown_ids)[0]}")
    event_kinds = list(manifest.get("event_kinds") or ["accident"])
    group_counts = {}
    group_counts_by_kind = {kind: {} for kind in event_kinds}
    totals_by_kind = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in event_kinds}
    per_sample = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for sample in manifest["samples"]:
        candidates = predictions_by_sample.get(sample["sample_id"], [])
        if not isinstance(candidates, list):
            raise ValueError(f"predictions for {sample['sample_id']} must be a list")
        sample_event_kinds = list(sample.get("event_kinds") or ["accident"])
        unsupported_predictions = [
            candidate for candidate in candidates
            if not isinstance(candidate, dict) or _event_kind(candidate) not in sample_event_kinds
        ]
        if unsupported_predictions:
            raise ValueError(
                f"predictions for {sample['sample_id']} contain a kind not enabled for this sample"
            )
        result = match_event_candidates(sample["events"], candidates,
                                        min_temporal_iou=min_temporal_iou)
        counts = {"tp": len(result["matches"]), "fp": len(result["unmatched_candidate_ids"]),
                  "fn": len(result["unmatched_event_ids"])}
        for key, value in counts.items():
            total[key] += value
        group = group_counts.setdefault(sample["group_id"], {"tp": 0, "fp": 0, "fn": 0})
        for key, value in counts.items():
            group[key] += value
        sample_counts_by_kind = {}
        for event_kind in sample_event_kinds:
            kind_result = match_event_candidates(
                [event for event in sample["events"] if _event_kind(event) == event_kind],
                [candidate for candidate in candidates if _event_kind(candidate) == event_kind],
                min_temporal_iou=min_temporal_iou,
            )
            kind_counts = {
                "tp": len(kind_result["matches"]),
                "fp": len(kind_result["unmatched_candidate_ids"]),
                "fn": len(kind_result["unmatched_event_ids"]),
            }
            sample_counts_by_kind[event_kind] = kind_counts
            kind_group = group_counts_by_kind[event_kind].setdefault(
                sample["group_id"], {"tp": 0, "fp": 0, "fn": 0}
            )
            for key, value in kind_counts.items():
                totals_by_kind[event_kind][key] += value
                kind_group[key] += value
        per_sample[sample["sample_id"]] = {
            "group_id": sample["group_id"], "counts": counts,
            "counts_by_kind": sample_counts_by_kind,
            "ground_truth_event_count": len(sample["events"]),
            "candidate_count": len(candidates), "matches": result["matches"],
            "unmatched_event_ids": result["unmatched_event_ids"],
            "unmatched_candidate_ids": result["unmatched_candidate_ids"],
        }

    aggregate = summarize_binary_counts(**total)
    metrics = {key: aggregate[key] for key in ("precision", "recall", "f1")}

    def bootstrap_metric(batches, metric):
        counts = {key: sum(batch[key] for batch in batches) for key in ("tp", "fp", "fn")}
        return summarize_binary_counts(**counts)[metric]

    intervals = {
        metric: cluster_bootstrap_interval(
            group_counts, lambda batches, metric=metric: bootstrap_metric(batches, metric),
            repetitions=bootstrap_repetitions, seed=seed,
        )
        for metric in ("precision", "recall", "f1")
    }
    by_kind = {}
    for event_kind in event_kinds:
        kind_summary = summarize_binary_counts(**totals_by_kind[event_kind])
        kind_metrics = {key: kind_summary[key] for key in ("precision", "recall", "f1")}
        kind_intervals = {
            metric: cluster_bootstrap_interval(
                group_counts_by_kind[event_kind],
                lambda batches, metric=metric: bootstrap_metric(batches, metric),
                repetitions=bootstrap_repetitions, seed=seed,
            )
            for metric in ("precision", "recall", "f1")
        }
        by_kind[event_kind] = {
            "counts": totals_by_kind[event_kind],
            "metrics": kind_metrics,
            "flight_cluster_bootstrap_95_ci": kind_intervals,
            "flight_group_count": len(group_counts_by_kind[event_kind]),
        }
    return {
        "counts": total, "metrics": metrics,
        "by_kind": by_kind,
        "flight_cluster_bootstrap_95_ci": intervals,
        "flight_group_count": len(group_counts),
        "test_sample_count": len(per_sample),
        "ground_truth_event_count": sum(item["ground_truth_event_count"] for item in per_sample.values()),
        "candidate_count": sum(item["candidate_count"] for item in per_sample.values()),
        "min_temporal_iou": float(min_temporal_iou), "samples": per_sample,
    }
