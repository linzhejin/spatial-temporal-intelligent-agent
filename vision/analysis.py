"""Conservative evidence summarization for aerial traffic detections.

Outputs are review candidates, never road closures or routing updates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from math import hypot
from statistics import mean
from typing import Any

from vision.tracking import assign_track_ids


VEHICLE_LABELS = {
    "car", "bus", "truck", "motorcycle", "motorbike", "motor", "van",
    "vehicle", "tricycle", "awning-tricycle",
}
INCIDENT_LABELS = {"accident", "crash", "collision", "vehicle_accident", "overturned_vehicle"}
PEDESTRIAN_LABELS = {"pedestrian", "person", "people"}
PREVIEW_LABELS = VEHICLE_LABELS | PEDESTRIAN_LABELS | {"bicycle"}
ROI_KINDS = {"vehicle_lane", "pedestrian", "road_surface", "parking", "exclude"}
MAX_PREVIEW_DETECTIONS_PER_FRAME = 80


def _validated_regions(regions, frame_width: int, frame_height: int) -> list[dict]:
    if regions is None:
        return []
    if not isinstance(regions, list) or len(regions) > 24:
        raise ValueError("观察区域必须是最多 24 个多边形组成的列表")
    normalized = []
    seen_ids = set()
    for region in regions:
        if not isinstance(region, dict) or region.get("kind") not in ROI_KINDS:
            raise ValueError("观察区域类型无效")
        polygon = region.get("polygon")
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 128:
            raise ValueError("观察区域至少需要 3 个、最多 128 个顶点")
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("观察区域坐标无效")
            x, y = point
            if (isinstance(x, bool) or isinstance(y, bool)
                    or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                    or not 0 <= x <= 1 or not 0 <= y <= 1):
                raise ValueError("观察区域坐标必须是 0 到 1 之间的画面比例")
            points.append([float(x) * frame_width, float(y) * frame_height])
        region_id = str(region.get("id") or f"region-{len(normalized) + 1}")[:64]
        if not region_id or region_id in seen_ids:
            raise ValueError("观察区域标识必须非空且唯一")
        seen_ids.add(region_id)
        normalized.append({
            "id": region_id,
            "kind": region["kind"],
            "polygon": points,
        })
    return normalized


def _inside_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _in_regions(item: dict, regions: list[dict]) -> bool:
    x, y = _center(item["box"])
    return any(_inside_polygon(x, y, region["polygon"]) for region in regions)


def _polygons_intersect(first: list[list[float]], second: list[list[float]]) -> bool:
    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def segments_intersect(a, b, c, d):
        ab_c, ab_d = orientation(a, b, c), orientation(a, b, d)
        cd_a, cd_b = orientation(c, d, a), orientation(c, d, b)

        def on_segment(start, end, point, cross):
            return (abs(cross) < 1e-9
                    and min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
                    and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9)

        if ab_c * ab_d < 0 and cd_a * cd_b < 0:
            return True
        return (
            on_segment(a, b, c, ab_c) or on_segment(a, b, d, ab_d)
            or on_segment(c, d, a, cd_a) or on_segment(c, d, b, cd_b)
        )

    if any(_inside_polygon(x, y, second) for x, y in first):
        return True
    if any(_inside_polygon(x, y, first) for x, y in second):
        return True
    return any(
        segments_intersect(first[index - 1], first[index], second[other - 1], second[other])
        for index in range(len(first))
        for other in range(len(second))
    )


def _valid_detection(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("label"), str)
        and isinstance(item.get("box"), (tuple, list))
        and len(item["box"]) == 4
        and all(isinstance(value, (int, float)) for value in item["box"])
        and isinstance(item.get("confidence"), (int, float))
    )


def _center(box):
    return ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)


def _count_people_with_nearby_peers(people: list[dict]) -> int:
    """Count detections with a neighbor within two mean person-box heights."""
    centers = []
    for person in people:
        x1, y1, x2, y2 = map(float, person["box"])
        centers.append((_center(person["box"]), max(0.0, y2 - y1)))
    grouped_count = 0
    for index, (center, height) in enumerate(centers):
        if any(
            hypot(center[0] - other_center[0], center[1] - other_center[1])
            <= height + other_height
            for other_index, (other_center, other_height) in enumerate(centers)
            if other_index != index
        ):
            grouped_count += 1
    return grouped_count


def _largest_image_space_group(people: list[dict]) -> int:
    """Return the largest connected group under the image-space neighbor rule."""
    centers = []
    for person in people:
        x1, y1, x2, y2 = map(float, person["box"])
        centers.append((_center(person["box"]), max(0.0, y2 - y1)))
    neighbors = [set() for _ in centers]
    for left_index, (left_center, left_height) in enumerate(centers):
        for right_index in range(left_index + 1, len(centers)):
            right_center, right_height = centers[right_index]
            if hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]) <= (
                    left_height + right_height):
                neighbors[left_index].add(right_index)
                neighbors[right_index].add(left_index)

    largest_group = 0
    visited = set()
    for start_index in range(len(centers)):
        if start_index in visited:
            continue
        pending = [start_index]
        visited.add(start_index)
        group_size = 0
        while pending:
            current = pending.pop()
            group_size += 1
            for neighbor in neighbors[current] - visited:
                visited.add(neighbor)
                pending.append(neighbor)
        largest_group = max(largest_group, group_size)
    return largest_group


def _transform_point(matrix, point):
    if matrix is None:
        return None
    x, y = point
    try:
        denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        if abs(float(denominator)) < 1e-9:
            return None
        return (
            (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
            (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
        )
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        return None


def _project_pixel_regions(regions, matrix, frame_width: int, frame_height: int) -> list[dict]:
    """Project anchor-frame ROIs into a frame, dropping any polygon not fully visible."""
    if matrix is None:
        return []
    projected = []
    for region in regions:
        source_points = region["polygon"]
        try:
            denominators = [
                matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
                for x, y in source_points
            ]
        except (IndexError, TypeError, ValueError):
            continue
        if (any(not math.isfinite(float(value)) or abs(float(value)) < 1e-9
                for value in denominators)
                or min(denominators) < 0 < max(denominators)):
            continue
        points = [_transform_point(matrix, point) for point in source_points]
        if any(point is None or not all(math.isfinite(float(value)) for value in point)
               for point in points):
            continue
        if any(not 0 <= x <= frame_width or not 0 <= y <= frame_height for x, y in points):
            continue
        projected.append({**region, "polygon": [[float(x), float(y)] for x, y in points]})
    return projected


def project_regions_for_frame(regions, matrix, frame_width: int, frame_height: int) -> list[dict]:
    """Return normalized manager ROIs projected from the anchor frame, if fully visible."""
    validated = _validated_regions(regions, frame_width, frame_height)
    return [
        {
            **region,
            "polygon": [[x / frame_width, y / frame_height] for x, y in region["polygon"]],
        }
        for region in _project_pixel_regions(validated, matrix, frame_width, frame_height)
    ]


def _candidate(kind: str, confidence: float, reason: str, evidence: dict) -> dict:
    return {
        "kind": kind,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "reason": reason,
        "evidence": evidence,
        "review_required": True,
        "auto_publish": False,
        "status": "pending_review",
    }


def _cluster_temporal_records(records: list[dict], sample_interval_s: float) -> list[list[dict]]:
    """Group nearby observations so distinct incidents in one ROI stay separate."""
    ordered = sorted(records, key=lambda item: item["time_seconds"])
    max_gap = max(1.5, sample_interval_s * 2.1)
    clusters: list[list[dict]] = []
    for record in ordered:
        if (not clusters
                or record["time_seconds"] - clusters[-1][-1]["time_seconds"] > max_gap):
            clusters.append([record])
        else:
            clusters[-1].append(record)
    return clusters


def build_specialized_candidates(
    observations: list[dict], *, sample_interval_s: float,
    single_image: bool = False,
    accident_threshold: float = 0.85,
    flood_area_threshold: float = 0.08,
) -> list[dict]:
    """Convert optional scene/segmentation outputs into review-only candidates.

    Accident predictions are scene-level AIDER scores and never imply a crash
    coordinate. Flood predictions are restricted to a marked road-surface ROI;
    ratios are image-region fractions, not physical area or water depth.
    """
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")
    candidates = []
    accident_records = []
    flood_records = []
    for observation in observations or []:
        if not isinstance(observation, dict):
            continue
        when = observation.get("time_seconds")
        if isinstance(when, bool) or not isinstance(when, (int, float)) or not math.isfinite(float(when)):
            continue
        probability = observation.get("traffic_accident_probability")
        if (isinstance(probability, (int, float)) and not isinstance(probability, bool)
                and math.isfinite(float(probability)) and accident_threshold <= probability <= 1):
            accident_records.append({
                "time_seconds": float(when), "confidence": float(probability),
                "region_id": str(observation.get("region_id") or "road"),
            })
        ratio = observation.get("flooded_road_area_ratio")
        if (isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
                and math.isfinite(float(ratio)) and flood_area_threshold <= ratio <= 1):
            flood_records.append({
                "time_seconds": float(when), "area_ratio": float(ratio),
                "region_id": str(observation.get("region_id") or "road-surface"),
                "water_evidence_type": ("visible_water"
                                         if observation.get("water_evidence_type") == "visible_water"
                                         else "flooded_road"),
                "outline_polygons": observation.get("outline_polygons") or [],
            })

    accident_regions: dict[str, list[dict]] = defaultdict(list)
    for item in accident_records:
        accident_regions[item["region_id"]].append(item)
    for region_id, region_records in accident_regions.items():
        selected_by_time = {}
        for item in region_records:
            current = selected_by_time.get(item["time_seconds"])
            if current is None or item["confidence"] > current["confidence"]:
                selected_by_time[item["time_seconds"]] = item
        selected = sorted(selected_by_time.values(), key=lambda item: item["time_seconds"])
        for event in _cluster_temporal_records(selected, sample_interval_s):
            accident_supported = (
                len(event) >= 2
                or single_image and max((item["confidence"] for item in event), default=0) >= 0.95
            )
            if not accident_supported:
                continue
            candidates.append(_candidate(
                "possible_accident",
                mean(item["confidence"] for item in event),
                "事故场景分类模型在连续画面中给出事故类别线索；模型只判断画面类别，无法定位事故车辆，须查看原片并人工确认。",
                {
                    "segments": [{
                        "start_seconds": item["time_seconds"],
                        "end_seconds": item["time_seconds"] + sample_interval_s,
                        "confidence": round(item["confidence"], 4),
                    } for item in event[:200]],
                    "summary": {
                        "peak_traffic_accident_probability": round(max(item["confidence"] for item in event), 4),
                        "supporting_frames": len(event),
                        "region_id": region_id,
                        "spatial_precision": "scene_classification_only",
                    },
                },
            ))

    flood_regions: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in flood_records:
        flood_regions[(item["region_id"], item["water_evidence_type"])].append(item)
    for (region_id, water_evidence_type), region_records in flood_regions.items():
        selected_by_time = {}
        for item in region_records:
            current = selected_by_time.get(item["time_seconds"])
            if current is None or item["area_ratio"] > current["area_ratio"]:
                selected_by_time[item["time_seconds"]] = item
        selected = sorted(selected_by_time.values(), key=lambda item: item["time_seconds"])
        for event in _cluster_temporal_records(selected, sample_interval_s):
            flood_supported = (
                len(event) >= 2
                or single_image and max((item["area_ratio"] for item in event), default=0) >= 0.25
            )
            if not flood_supported:
                continue
            strongest = max(event, key=lambda item: item["area_ratio"])
            visible_water_only = water_evidence_type == "visible_water"
            reason = (
                "可见水体分割模型在管理员圈定的路面画面区域内检出可见水体线索；"
                "模型无法区分路面积水与其他可见水体，须人工核对。"
                if visible_water_only else
                "积水分割模型在已圈定路面区域内检出疑似淹水像素；"
                "面积比例按画面区域计算，不代表实际水深或地面范围，须人工核对。"
            )
            summary = {
                "region_id": region_id,
                "water_evidence_type": water_evidence_type,
                "supporting_frames": len(event),
                "outline_time_seconds": round(strongest["time_seconds"], 3),
                "outline_polygons": strongest["outline_polygons"][:8],
                "water_depth_estimated": False,
            }
            summary["max_visible_water_area_ratio" if visible_water_only
                    else "max_flooded_road_area_ratio"] = round(strongest["area_ratio"], 4)
            candidates.append(_candidate(
                "possible_flooding",
                min(0.99, 0.6 + strongest["area_ratio"] / 2),
                reason,
                {
                    "segments": [{
                        "start_seconds": item["time_seconds"],
                        "end_seconds": item["time_seconds"] + sample_interval_s,
                        "confidence": round(item["area_ratio"], 4),
                    } for item in event[:200]],
                    "summary": summary,
                },
            ))
    return candidates


def analyze_observations(
    frames: list[list[dict]], *, frame_width: int, frame_height: int,
    sample_interval_s: float = 1.0,
    frame_times_s: list[float] | None = None,
    camera_stabilized: bool = False,
    frame_transforms: list[list[list[float]] | None] | None = None,
    region_transforms: list[list[list[float]] | None] | None = None,
    accident_model_enabled: bool = False,
    regions: list[dict] | None = None,
    min_vehicle_count: int = 6,
    stationary_ratio_threshold: float = 0.5,
    min_congestion_duration_s: float = 15.0,
    min_crowding_duration_s: float = 5.0,
    min_crowding_cluster_size: int = 8,
    tracking_high_threshold: float = 0.369,
    tracking_low_threshold: float = 0.1,
    include_preview_frames: bool = False,
    preview_frame_interval_s: float = 1.0,
) -> dict:
    """Summarize detections and compensate vehicle motion for verified camera motion.

    ``frame_transforms[i]`` maps coordinates in frame ``i-1`` to frame ``i``.
    A missing transform is treated as failed camera-motion estimation. A video
    track-motion estimate requires at least 80% valid frame-to-frame transforms,
    unless the manager explicitly declares a fixed or already stabilized camera.
    ``region_transforms[i]`` maps the manager's anchor-frame ROI into frame i.
    Candidate events require reliable frame-to-frame motion compensation and a
    fully visible ROI mapping for at least 80% of analyzed frames.
    """
    if frame_width <= 0 or frame_height <= 0 or sample_interval_s <= 0:
        raise ValueError("valid frame dimensions and sample interval are required")
    if frame_times_s is None:
        frame_times_s = [index * sample_interval_s for index in range(len(frames))]
    if (len(frame_times_s) != len(frames)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in frame_times_s)
            or any(float(right) <= float(left)
                   for left, right in zip(frame_times_s, frame_times_s[1:]))):
        raise ValueError("frame timestamps must be finite, increasing, and match the frames")
    frame_times_s = [float(value) for value in frame_times_s]
    if (isinstance(min_congestion_duration_s, bool)
            or not isinstance(min_congestion_duration_s, (int, float))
            or not math.isfinite(float(min_congestion_duration_s))
            or min_congestion_duration_s <= 0):
        raise ValueError("minimum congestion observation duration must be finite and positive")
    if (isinstance(min_crowding_duration_s, bool)
            or not isinstance(min_crowding_duration_s, (int, float))
            or not math.isfinite(float(min_crowding_duration_s))
            or min_crowding_duration_s <= 0):
        raise ValueError("minimum crowd observation duration must be finite and positive")
    if (isinstance(min_crowding_cluster_size, bool)
            or not isinstance(min_crowding_cluster_size, int)
            or min_crowding_cluster_size < 2):
        raise ValueError("minimum image-space crowd cluster size must be an integer of at least two")
    regions = _validated_regions(regions, frame_width, frame_height)
    if region_transforms is not None and len(region_transforms) != len(frames):
        raise ValueError("observation region transforms must match the analyzed frame count")
    regions_by_frame = []
    mapping_valid_by_region = {region["id"]: [] for region in regions}
    for frame_index in range(len(frames)):
        if region_transforms is not None:
            frame_regions = _project_pixel_regions(
                regions, region_transforms[frame_index], frame_width, frame_height,
            )
        elif camera_stabilized or len(frames) <= 1:
            frame_regions = regions
        elif frame_index == 0:
            frame_regions = regions
        else:
            frame_regions = []
        regions_by_frame.append(frame_regions)
        visible_ids = {region["id"] for region in frame_regions}
        for region_id in mapping_valid_by_region:
            mapping_valid_by_region[region_id].append(region_id in visible_ids)
    region_mapping_ratios = {
        region_id: (sum(validity) / len(validity) if validity else 0.0)
        for region_id, validity in mapping_valid_by_region.items()
    }

    def candidate_mapping_ratio(region):
        required_kinds = {"exclude"}
        if region["kind"] == "vehicle_lane":
            required_kinds.add("parking")
        required_ids = [region["id"]] + [item["id"] for item in regions
            if item["kind"] in required_kinds
            and _polygons_intersect(region["polygon"], item["polygon"])]
        return min((region_mapping_ratios.get(region_id, 0.0) for region_id in required_ids),
                   default=0.0)

    vehicle_regions = [region for region in regions if region["kind"] == "vehicle_lane"]
    pedestrian_regions = [region for region in regions if region["kind"] == "pedestrian"]

    valid_frames = []
    for frame_index, frame in enumerate(frames):
        exclusions = [region for region in regions_by_frame[frame_index]
                      if region["kind"] == "exclude"]
        valid_frames.append([
            item for item in frame
            if _valid_detection(item)
            and not any(_in_regions(item, [region]) for region in exclusions)
        ])
    transition_count = max(0, len(valid_frames) - 1)
    valid_transition_count = 0 if camera_stabilized else sum(
        1 for index in range(1, len(valid_frames))
        if frame_transforms and index < len(frame_transforms) and frame_transforms[index] is not None
    )
    valid_transition_ratio = (
        1.0 if camera_stabilized and transition_count
        else valid_transition_count / transition_count if transition_count else 0.0
    )
    motion_compensation_ready = (
        camera_stabilized
        or (transition_count > 0 and valid_transition_ratio >= 0.8)
    )
    track_frames = assign_track_ids(
        valid_frames, frame_transforms=frame_transforms,
        camera_stabilized=camera_stabilized,
        high_threshold=tracking_high_threshold,
        low_threshold=tracking_low_threshold,
    )
    preview_frames = []
    if include_preview_frames:
        if (isinstance(preview_frame_interval_s, bool)
                or not isinstance(preview_frame_interval_s, (int, float))
                or not math.isfinite(float(preview_frame_interval_s))
                or preview_frame_interval_s <= 0):
            raise ValueError("preview frame interval must be finite and positive")
        last_preview_time = -float("inf")
        for frame_index in range(len(track_frames)):
            frame_time = frame_times_s[frame_index]
            if frame_time + 1e-9 < last_preview_time + preview_frame_interval_s:
                continue
            detections = []
            for item in track_frames[frame_index]:
                label = str(item.get("label", "")).strip().lower()
                if label not in PREVIEW_LABELS or not _valid_detection(item):
                    continue
                x1, y1, x2, y2 = map(float, item["box"])
                box = [
                    round(max(0.0, min(1.0, x1 / frame_width)), 5),
                    round(max(0.0, min(1.0, y1 / frame_height)), 5),
                    round(max(0.0, min(1.0, x2 / frame_width)), 5),
                    round(max(0.0, min(1.0, y2 / frame_height)), 5),
                ]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.0))
                except (TypeError, ValueError, OverflowError):
                    continue
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    continue
                detection = {
                    "label": label,
                    "box": box,
                    "confidence": round(confidence, 4),
                }
                track_id = item.get("track_id")
                if (isinstance(track_id, (int, str)) and not isinstance(track_id, bool)
                        and str(track_id).strip()):
                    detection["track_id"] = str(track_id)[:24]
                detections.append(detection)
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            preview_frames.append({
                "frame_index": frame_index,
                "time_seconds": round(frame_time, 3),
                "detections": detections[:MAX_PREVIEW_DETECTIONS_PER_FRAME],
            })
            last_preview_time = frame_time
    vehicle_frames: list[list[dict]] = []
    pedestrian_counts: list[int] = []
    track_positions: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    vehicle_counts_by_region = {region["id"]: [] for region in vehicle_regions}
    vehicle_area_by_region = {region["id"]: [] for region in vehicle_regions}
    track_positions_by_region = {region["id"]: defaultdict(list) for region in vehicle_regions}
    pedestrian_counts_by_region = {region["id"]: [] for region in pedestrian_regions}
    pedestrian_group_sizes_by_region = {region["id"]: [] for region in pedestrian_regions}
    vehicle_observed_frames = {region["id"]: [] for region in vehicle_regions}
    pedestrian_observed_frames = {region["id"]: [] for region in pedestrian_regions}
    pedestrian_track_positions_by_region = {
        region["id"]: defaultdict(list) for region in pedestrian_regions
    }
    nearby_peer_counts_by_region = {region["id"]: [] for region in pedestrian_regions}
    accident_detections: list[tuple[int, float, str, str]] = []
    coverage_values: list[float] = []
    class_counts: list[Counter] = []

    for frame_index, detections in enumerate(track_frames):
        frame_regions = regions_by_frame[frame_index]
        frame_vehicle_regions = [region for region in frame_regions
                                 if region["kind"] == "vehicle_lane"]
        frame_pedestrian_regions = [region for region in frame_regions
                                    if region["kind"] == "pedestrian"]
        frame_parking_regions = [region for region in frame_regions
                                 if region["kind"] == "parking"]
        counted_detections = [
            item for item in detections
            if float(item.get("confidence", 0)) >= tracking_high_threshold
            or item.get("track_id") is not None
        ]
        all_vehicles = [item for item in counted_detections
                        if item["label"].strip().lower() in VEHICLE_LABELS]
        vehicles = (
            [item for item in all_vehicles if _in_regions(item, frame_vehicle_regions)]
            if vehicle_regions and frame_vehicle_regions else [] if vehicle_regions else all_vehicles
        )
        if frame_parking_regions:
            vehicles = [item for item in vehicles if not _in_regions(item, frame_parking_regions)]
        vehicle_frames.append(vehicles)
        for region in vehicle_regions:
            mapped_region = next((item for item in frame_vehicle_regions
                                  if item["id"] == region["id"]), None)
            if mapped_region is None:
                continue
            region_vehicles = [item for item in all_vehicles if _in_regions(item, [mapped_region])]
            if frame_parking_regions:
                region_vehicles = [item for item in region_vehicles
                                   if not _in_regions(item, frame_parking_regions)]
            region_id = region["id"]
            vehicle_observed_frames[region_id].append(frame_index)
            vehicle_counts_by_region[region_id].append(len(region_vehicles))
            region_covered = 0.0
            for item in region_vehicles:
                x1, y1, x2, y2 = map(float, item["box"])
                region_covered += max(0.0, x2 - x1) * max(0.0, y2 - y1)
                track_id = item.get("track_id")
                if track_id is not None:
                    cx, cy = _center(item["box"])
                    track_positions_by_region[region_id][str(track_id)].append(
                        (cx, cy, frame_index)
                    )
            vehicle_area_by_region[region_id].append(
                min(1.0, region_covered / (frame_width * frame_height))
            )
        class_counts.append(Counter(item["label"].strip().lower() for item in counted_detections))
        people = [item for item in counted_detections if item["label"].strip().lower() in PEDESTRIAN_LABELS]
        if pedestrian_regions:
            people = [item for item in people if _in_regions(item, frame_pedestrian_regions)]
        pedestrian_counts.append(len(people))
        for region in pedestrian_regions:
            mapped_region = next((item for item in frame_pedestrian_regions
                                  if item["id"] == region["id"]), None)
            if mapped_region is None:
                continue
            region_people = [
                item for item in counted_detections
                if item["label"].strip().lower() in PEDESTRIAN_LABELS
                and _in_regions(item, [mapped_region])
            ]
            pedestrian_observed_frames[region["id"]].append(frame_index)
            pedestrian_counts_by_region[region["id"]].append(len(region_people))
            nearby_peer_counts_by_region[region["id"]].append(
                _count_people_with_nearby_peers(region_people)
            )
            pedestrian_group_sizes_by_region[region["id"]].append(
                _largest_image_space_group(region_people)
            )
            for item in region_people:
                track_id = item.get("track_id")
                if track_id is not None:
                    cx, cy = _center(item["box"])
                    pedestrian_track_positions_by_region[region["id"]][str(track_id)].append(
                        (cx, cy, frame_index)
                    )
        covered = 0.0
        for item in vehicles:
            x1, y1, x2, y2 = map(float, item["box"])
            covered += max(0.0, x2 - x1) * max(0.0, y2 - y1)
            track_id = item.get("track_id")
            if track_id is not None:
                cx, cy = _center(item["box"])
                track_positions[str(track_id)].append((cx, cy, frame_index))
        coverage_values.append(min(1.0, covered / (frame_width * frame_height)))
        if accident_model_enabled:
            for item in detections:
                label = item["label"].strip().lower().replace(" ", "_")
                if label in INCIDENT_LABELS:
                    for region in frame_vehicle_regions:
                        if _in_regions(item, [region]):
                            accident_detections.append((
                                frame_index, float(item["confidence"]), label, region["id"],
                            ))

    vehicle_counts = [len(items) for items in vehicle_frames]
    diagonal = hypot(frame_width, frame_height)
    def speeds_for_tracks(track_groups):
        speeds = []
        for positions in track_groups.values():
            positions.sort(key=lambda value: value[2])
            residuals = []
            for left, right in zip(positions, positions[1:]):
                if right[2] != left[2] + 1:
                    continue
                elapsed_s = frame_times_s[right[2]] - frame_times_s[left[2]]
                if elapsed_s <= 0 or elapsed_s > max(0.5, sample_interval_s * 1.5):
                    continue
                matrix = (
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]] if camera_stabilized
                    else frame_transforms[right[2]]
                    if frame_transforms and right[2] < len(frame_transforms)
                    else None
                )
                predicted = _transform_point(matrix, (left[0], left[1]))
                if predicted is not None:
                    residuals.append(
                        hypot(right[0] - predicted[0], right[1] - predicted[1]) / elapsed_s
                    )
            if len(residuals) >= 2:
                speeds.append(mean(residuals) / diagonal)
        return speeds

    track_speeds = speeds_for_tracks(track_positions)
    speeds_by_region = {
        region_id: speeds_for_tracks(positions)
        for region_id, positions in track_positions_by_region.items()
    }
    pedestrian_speeds_by_region = {
        region_id: speeds_for_tracks(positions)
        for region_id, positions in pedestrian_track_positions_by_region.items()
    }

    stationary_count = sum(speed <= 0.0025 for speed in track_speeds)
    stationary_ratio = stationary_count / len(track_speeds) if track_speeds else 0.0
    avg_vehicle_count = mean(vehicle_counts) if vehicle_counts else 0.0
    candidates = []
    observed_duration_s = (
        max(0.0, frame_times_s[-1] - frame_times_s[0]) if frame_times_s else 0.0
    )

    def mapped_duration(frame_indices):
        return max(0.0, frame_times_s[frame_indices[-1]] - frame_times_s[frame_indices[0]]) \
            if len(frame_indices) >= 2 else 0.0

    def is_contiguous(previous_index, current_index):
        return (
            current_index == previous_index + 1
            and frame_times_s[current_index] - frame_times_s[previous_index]
            <= max(0.5, sample_interval_s * 1.5)
        )

    def longest_contiguous_duration(frame_indices):
        longest = []
        current = []
        for index in frame_indices:
            if current and not is_contiguous(current[-1], index):
                current = []
            current.append(index)
            if len(current) > len(longest):
                longest = current
        return mapped_duration(longest)

    region_mapping_basis = (
        "operator_declared_stabilized" if camera_stabilized
        else "homography_projected_from_video_anchor" if region_transforms is not None
        else "single_frame_anchor" if len(frames) <= 1
        else "unavailable"
    )

    # Keep every candidate attached to the exact image ROI that produced it.
    # A single frame can show a vehicle cluster, but cannot establish congestion.
    for region in vehicle_regions:
        region_id = region["id"]
        counts = vehicle_counts_by_region[region_id]
        mapping_ratio = region_mapping_ratios.get(region_id, 0.0)
        supporting_mapping_ratio = candidate_mapping_ratio(region)
        region_observed_span_s = mapped_duration(vehicle_observed_frames[region_id])
        region_duration_s = longest_contiguous_duration(vehicle_observed_frames[region_id])
        if len(valid_frames) == 1 and counts and counts[0] >= min_vehicle_count:
            candidates.append(_candidate(
                "vehicle_cluster_review", min(0.9, counts[0] / (min_vehicle_count * 2)),
                "单帧检测到多辆车辆；静态影像不能判断速度或拥堵，需查看原片并现场复核。",
                {"summary": {
                    "region_id": region_id,
                    "vehicle_count": counts[0],
                    "occupied_area_ratio": round(vehicle_area_by_region[region_id][0], 4),
                }},
            ))

        region_speeds = speeds_by_region[region_id]
        region_stationary_count = sum(speed <= 0.0025 for speed in region_speeds)
        region_stationary_ratio = (
            region_stationary_count / len(region_speeds) if region_speeds else 0.0
        )
        if (
            motion_compensation_ready and supporting_mapping_ratio >= 0.8
            and region_duration_s >= min_congestion_duration_s
            and mean(counts) >= min_vehicle_count and len(region_speeds) >= 3
            and region_stationary_ratio >= stationary_ratio_threshold
        ):
            candidates.append(_candidate(
                "possible_congestion", min(0.95, 0.5 + region_stationary_ratio / 2),
                "多帧车辆在相机运动校正后仍呈低位移；请核实观察区域、道路位置和现场通行情况。",
                {"summary": {
                    "region_id": region_id,
                    "mean_vehicle_count": round(mean(counts), 2),
                    "stationary_track_count": region_stationary_count,
                    "tracked_vehicle_count": len(region_speeds),
                    "stationary_track_ratio": round(region_stationary_ratio, 3),
                    "observed_duration_seconds": round(observed_duration_s, 3),
                    "region_observed_duration_seconds": round(region_duration_s, 3),
                    "region_observed_span_seconds": round(region_observed_span_s, 3),
                    "minimum_duration_seconds": float(min_congestion_duration_s),
                    "camera_motion_compensated": bool(camera_stabilized or motion_compensation_ready),
                    "camera_stabilization_basis": "operator_declared" if camera_stabilized else "estimated_transforms",
                    "valid_motion_transition_ratio": round(valid_transition_ratio, 3),
                    "region_mapping_basis": region_mapping_basis,
                    "region_mapping_valid_ratio": round(mapping_ratio, 3),
                    "supporting_regions_mapping_valid_ratio": round(supporting_mapping_ratio, 3),
                }},
            ))

    for region in pedestrian_regions:
        counts = pedestrian_counts_by_region[region["id"]]
        mapping_ratio = region_mapping_ratios.get(region["id"], 0.0)
        supporting_mapping_ratio = candidate_mapping_ratio(region)
        region_duration_s = mapped_duration(pedestrian_observed_frames[region["id"]])
        peer_counts = nearby_peer_counts_by_region[region["id"]]
        group_sizes = pedestrian_group_sizes_by_region[region["id"]]
        pedestrian_speeds = pedestrian_speeds_by_region[region["id"]]
        observed_frames = pedestrian_observed_frames[region["id"]]
        longest_cluster_run = []
        current_cluster_run = []
        for frame_index, group_size in zip(observed_frames, group_sizes):
            if group_size < min_crowding_cluster_size:
                current_cluster_run = []
                continue
            if current_cluster_run and not is_contiguous(current_cluster_run[-1], frame_index):
                current_cluster_run = []
            current_cluster_run.append(frame_index)
            if len(current_cluster_run) > len(longest_cluster_run):
                longest_cluster_run = list(current_cluster_run)
        sustained_cluster_duration_s = mapped_duration(longest_cluster_run)
        longest_run_frames = set(longest_cluster_run)
        sustained_cluster_peak_size = max(
            (group_size for frame_index, group_size in zip(observed_frames, group_sizes)
             if frame_index in longest_run_frames),
            default=0,
        )
        moving_people = sum(speed > 0.003 for speed in pedestrian_speeds)
        low_motion_people = len(pedestrian_speeds) - moving_people
        if not motion_compensation_ready:
            movement_assessment = "camera_motion_uncompensated"
        elif not pedestrian_speeds:
            movement_assessment = "insufficient_track_data"
        elif moving_people > low_motion_people:
            movement_assessment = "mostly_moving_in_image"
        else:
            movement_assessment = "mostly_stationary_in_image"
        peak_peer_count = max(peer_counts, default=0)
        if (motion_compensation_ready and supporting_mapping_ratio >= 0.8
                and region_duration_s >= min_crowding_duration_s
                and len(counts) >= 3 and max(counts, default=0) >= 15
                and sustained_cluster_duration_s >= min_crowding_duration_s):
            candidates.append(_candidate(
                "possible_crowding", min(0.9, 0.55 + sustained_cluster_peak_size / 100),
                f"行人观察区域内连续检测到至少 {min_crowding_cluster_size} 名画面相邻行人；"
                "这不是地面密度估计，请结合原片和现场核实。",
                {"region_id": region["id"],
                 "peak_person_count": max(counts),
                 "mean_person_count": round(mean(counts), 2),
                 "frames_with_people": sum(count > 0 for count in counts),
                 "sustained_cluster_frames": len(longest_cluster_run),
                 "sustained_cluster_duration_seconds": round(sustained_cluster_duration_s, 3),
                 "sustained_cluster_peak_size": sustained_cluster_peak_size,
                 "peak_image_space_group_size": max(group_sizes, default=0),
                 "minimum_cluster_size": min_crowding_cluster_size,
                 "observed_duration_seconds": round(observed_duration_s, 3),
                 "region_observed_duration_seconds": round(region_duration_s, 3),
                 "region_mapping_basis": region_mapping_basis,
                 "region_mapping_valid_ratio": round(mapping_ratio, 3),
                 "supporting_regions_mapping_valid_ratio": round(supporting_mapping_ratio, 3),
                 "minimum_duration_seconds": float(min_crowding_duration_s),
                 "peak_people_with_nearby_peer_count": peak_peer_count,
                 "aggregation_assessment": (
                     "image_space_neighbors_detected" if peak_peer_count else
                     "no_image_space_cluster_detected"
                 ),
                 "aggregation_basis": (
                     "画面内相邻行人中心距离不超过两倍平均检测框高度；"
                     "受透视影响，不代表地面距离或人数密度。"
                 ),
                 "movement_assessment": movement_assessment,
                 "moving_person_track_count": moving_people,
                 "low_motion_person_track_count": low_motion_people,
                 "movement_basis": (
                    "相机变换校正后的行人轨迹像素位移；不是地面速度。"
                 )},
            ))

    if accident_detections:
        by_region: dict[str, list[tuple[int, float, str, str]]] = defaultdict(list)
        for detection in accident_detections:
            by_region[detection[3]].append(detection)
        for region_id, region_detections in by_region.items():
            by_label: dict[str, list[tuple[int, float, str, str]]] = defaultdict(list)
            for detection in region_detections:
                by_label[detection[2]].append(detection)
            strongest = max(
                (max(items, key=lambda value: value[1]) for items in by_label.values()),
                key=lambda value: value[1],
            )
            repeated = len({item[0] for item in by_label[strongest[2]]}) >= 2
            threshold = 0.6 if len(valid_frames) > 1 and repeated else 0.85
            if strongest[1] >= threshold:
                candidates.append(_candidate(
                    "possible_accident", strongest[1],
                    "专用事故模型给出事故类别候选；需管理员查看原始影像并核实位置后确认。",
                    {"summary": {
                        "region_id": region_id,
                        "detector_label": strongest[2],
                        "frames_with_detection": len({item[0] for item in by_label[strongest[2]]}),
                        "spatial_precision": "marked_vehicle_lane_only",
                    }},
                ))

    all_classes = set().union(*(counts.keys() for counts in class_counts)) if class_counts else set()
    mean_class_counts = {
        label: round(mean(counts.get(label, 0) for counts in class_counts), 2)
        for label in sorted(all_classes)
    }
    peak_class_counts = {
        label: max((counts.get(label, 0) for counts in class_counts), default=0)
        for label in sorted(all_classes)
    }
    result = {
        "metrics": {
            "frames_analyzed": len(valid_frames),
            "observed_duration_seconds": round(observed_duration_s, 3),
            "minimum_congestion_duration_seconds": float(min_congestion_duration_s),
            "minimum_crowding_duration_seconds": float(min_crowding_duration_s),
            "minimum_crowding_cluster_size": min_crowding_cluster_size,
            "mean_vehicle_count": round(avg_vehicle_count, 2),
            "peak_vehicle_count": max(vehicle_counts, default=0),
            "mean_pedestrian_count": round(mean(pedestrian_counts), 2) if pedestrian_counts else 0.0,
            "peak_pedestrian_count": max(pedestrian_counts, default=0),
            "mean_vehicle_occupied_area_ratio": round(mean(coverage_values), 4) if coverage_values else 0.0,
            "mean_class_counts": mean_class_counts,
            "peak_class_counts": peak_class_counts,
            "tracked_vehicle_count": len(track_speeds),
            "stationary_track_count": stationary_count,
            "stationary_track_ratio": round(stationary_ratio, 3),
            "motion_compensation_valid_transitions": valid_transition_count,
            "motion_compensation_transition_count": transition_count,
            "motion_compensation_valid_ratio": round(valid_transition_ratio, 3),
            "observation_region_mapping_valid_ratios": {
                region_id: round(ratio, 3) for region_id, ratio in region_mapping_ratios.items()
            },
            "motion_assessment": (
                "insufficient_single_frame" if len(valid_frames) < 2
                else "operator_declared_stabilized" if camera_stabilized
                else "camera_motion_compensated" if motion_compensation_ready
                else "camera_motion_uncompensated"
            ),
        },
        "candidates": candidates,
        "safety": {
            "requires_human_review": True,
            "camera_motion_compensated": bool(motion_compensation_ready and not camera_stabilized),
            "camera_stabilized_assumed": bool(camera_stabilized),
            "observation_region_reassociation_required": bool(
                regions and transition_count > 0 and not camera_stabilized
                and any(ratio < 0.8 for ratio in region_mapping_ratios.values())
            ),
            "accident_recognition_supported": False,
            "pedestrian_region_required_for_crowding": True,
            "automatically_changes_routing": False,
        },
    }
    if include_preview_frames:
        result["preview_frames"] = preview_frames
    return result
